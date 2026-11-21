"""Агент с проверяемыми вызовами инструментов (FR-7 — FR-10)."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено достаточно информации для ответа."
SYSTEM = """Ты диспетчер сервисной службы «ПолярТех».
Для ответов по документам используй search_knowledge, для общих обзоров summarize_topic.
Для остатков и цен используй check_stock и get_price; не придумывай артикулы, остатки и цены.
Если номер детали не известен, сначала вызови search_knowledge.
Никогда не создавай заказ самовольно. При необходимости укажи количество, склад и модель.
Отличай отсутствие детали в ERP от нулевого остатка и от недоступности ERP.
Обязательно называй источники документов, когда используешь документы.
"""
_PART = re.compile(r"P-\d{4}[A-Z]?\Z")
_ORDER = re.compile(r"\bSO-\d{6}\b")
_PRICE = re.compile(r"(?<![\w-])(\d[\d\s]*?)\s*(?:руб(?:лей|ля|ль|\.)?\b|₽)", re.IGNORECASE)
_STOCK = re.compile(r"(?<![\w-])(\d+)\s*(?:шт\.?\b|штук\b|штуки\b|единиц\b)", re.IGNORECASE)
_QUANTITY = re.compile(
    r"\b(\d+)\s*(?:шт\.?\b|штук\b|штуки\b|единиц\b|единицы\b)|"
    r"\b(?:количество|закажи|заказать|оформи\s+заказ\s+на)\s*[:=]?\s*(\d+)\b",
    re.IGNORECASE,
)
_QUANTITY_WORDS = {
    "одну": 1, "один": 1, "одна": 1, "две": 2, "два": 2, "три": 3,
    "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9, "десять": 10,
}


def _all_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _codes(value: object) -> set[str]:
    return set(find_codes(_all_text(value)))


def _user_quantities(question: str) -> set[int]:
    quantities = {int(x or y) for x, y in _QUANTITY.findall(question)}
    for word, number in _QUANTITY_WORDS.items():
        if re.search(rf"\b{word}\s+(?:штук|штуки|шт|детал[ьи]|единиц)\b", question, re.IGNORECASE):
            quantities.add(number)
    return quantities


def _answer_is_grounded(answer: str, question: str, successful: list[dict]) -> bool:
    """Проверка кодов, цен, остатков и номеров заказов по входным данным."""
    evidence = [question, *successful]
    text = _all_text(evidence)
    if not set(find_codes(answer)).issubset(_codes(evidence)):
        return False
    allowed_orders = set(_ORDER.findall(text))
    if not set(_ORDER.findall(answer)).issubset(allowed_orders):
        return False
    prices = {str(data["price_rub"]) for data in successful if isinstance(data.get("price_rub"), (int, float))}
    prices.update(m.group(1).replace(" ", "") for m in _PRICE.finditer(question))
    for match in _PRICE.finditer(answer):
        if match.group(1).replace(" ", "") not in prices:
            return False
    quantities = {str(data["qty"]) for data in successful if isinstance(data.get("qty"), int)}
    quantities.update(str(qty) for qty in _user_quantities(question))
    for match in _STOCK.finditer(answer):
        if match.group(1) not in quantities:
            return False
    return True


def _refused(answer: str) -> bool:
    """Распознать семантический отказ, даже если модель изменила формулировку."""
    lower = answer.casefold()
    phrases = (
        "в базе знаний не найдено",
        "не найдено в базе знаний",
        "не найдена в базе знаний",
        "не найден в базе знаний",
        "не найдены в базе знаний",
        "данных в базе знаний нет",
        "данных нет в базе знаний",
        "не найдена в системе erp",
        "не найдено в системе erp",
        "не найден в системе erp",
        "данных нет в erp",
    )
    return any(phrase in lower for phrase in phrases)


def _tools_confirm_no_data(steps: list[ToolStep]) -> bool:
    """Считать отказ подтверждённым, если ВСЕ запросы данных вернули их отсутствие.

    Не путать нулевой остаток с отсутствием данных и не считать
    временную недоступность ERP ответом «деталь не существует».
    """
    relevant = [
        step for step in steps
        if step.tool in {"search_knowledge", "summarize_topic", "get_price", "check_stock"}
    ]
    if not relevant:
        return False
    for step in relevant:
        result = step.result if isinstance(step.result, dict) else {}
        if step.tool in {"search_knowledge", "summarize_topic"}:
            if result.get("refused") is not True:
                return False
        elif not str(result.get("error", "")).casefold().startswith("данных нет в erp"):
            return False
    return True


def _tool_result(
    name: str,
    args: dict,
    tools: dict[str, Callable[..., dict]],
    cache: dict[tuple[str, str], dict],
) -> tuple[dict, str | None]:
    key = (name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
    if key in cache:
        result = cache[key]
        return result, result.get("error")

    # Временная недоступность ERP: исходный вызов и ещё две попытки.
    for attempt in range(3):
        try:
            result = tools[name](**args)
            if not isinstance(result, dict):
                result = {"error": "Инструмент вернул ответ неправильного формата"}
            break
        except ErpUnavailable as exc:
            if attempt == 2:
                result = {"error": f"Данные ERP получить не удалось: {exc}"}
        except (TypeError, ValueError) as exc:
            result = {"error": f"Некорректные аргументы инструмента: {exc}"}
            break
    cache[key] = result
    return result, result.get("error")


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Вызвать модель и инструменты с защитой от выдуманных данных и заказов."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: set[str] = set()
    successful: list[dict] = []
    cache: dict[tuple[str, str], dict] = {}
    user_codes = set(find_codes(question))
    user_qty = _user_quantities(question)
    last_knowledge_refused = False
    last_erp_error = False

    def finish(content: str) -> AgentResult:
        nonlocal last_knowledge_refused
        answer = content.strip()
        if not answer:
            if last_erp_error:
                answer = "Данные ERP получить не удалось."
            elif last_knowledge_refused:
                answer = "В базе знаний не найдено."
            else:
                answer = "Не удалось сформировать ответ."
        if not _answer_is_grounded(answer, question, successful):
            messages.append({"role": "assistant", "content": answer})
            messages.append({
                "role": "user",
                "content": (
                    "Исправь ответ: в нём есть код, цена, остаток или номер заказа, которых нет "
                    "в вопросе или результатах инструментов. Используй ТОЛЬКО проверенные факты. "
                    "Если информации недостаточно, напиши «В базе знаний не найдено»."
                ),
            })
            corrected = llm.chat(messages, max_tokens=1024).content.strip()
            if not _answer_is_grounded(corrected, question, successful):
                return AgentResult(REFUSAL, steps=steps, sources=sorted(sources), refused=True)
            answer = corrected
        no_data_confirmed = _tools_confirm_no_data(steps)
        if no_data_confirmed and not _refused(answer):
            # Инструменты не нашли данных: не пропускаем уверенную выдумку LLM.
            answer = REFUSAL
        return AgentResult(
            answer,
            steps=steps,
            sources=sorted(sources),
            refused=_refused(answer) or no_data_confirmed,
        )

    while True:
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            return finish(response.content)

        for call in response.tool_calls:
            # Лимит учитывает все вызовы, даже неверные/кешированные.
            if len(steps) >= max_steps:
                snippets = [str(item.get("answer", "")) for item in successful if item.get("answer")]
                detail = " ".join(snippets).strip()
                text = "Работа прервана: достигнут лимит вызовов инструментов."
                if detail:
                    text = f"{text} Известно: {detail}"
                return AgentResult(text, steps=steps, sources=sorted(sources))

            args = call.arguments
            error: str | None = None
            result: dict | None = None
            if call.name not in tools:
                error = f"Неизвестный инструмент {call.name}"
            elif not isinstance(args, dict) or "_raw" in args:
                error = "Не удалось прочитать аргументы инструмента (требуется JSON)"
            else:
                part_id = args.get("part_id")
                if part_id is not None and (
                    not isinstance(part_id, str)
                    or not _PART.fullmatch(part_id)
                    or part_id not in user_codes | _codes(successful)
                ):
                    error = (
                        "Артикул не подтверждён вопросом или результатами инструментов; "
                        "сначала вызови search_knowledge"
                    )
                elif call.name == "create_service_order":
                    qty = args.get("qty")
                    if isinstance(qty, bool) or not isinstance(qty, int) or qty not in user_qty:
                        return AgentResult(
                            "Сколько штук заказать? Уточните количество деталей.",
                            steps=steps + [ToolStep(call.name, args, error="Количество не указано в вопросе")],
                            sources=sorted(sources),
                        )
                    if not confirm:
                        return AgentResult(
                            "Для оформления заказа требуется подтверждение человека.",
                            steps=steps + [ToolStep(call.name, args, error="Ожидается подтверждение заказа")],
                            sources=sorted(sources),
                            needs_confirmation=True,
                            pending_action={"tool": "create_service_order", "args": args},
                        )
                if error is None:
                    result, error = _tool_result(call.name, args, tools, cache)

            if result is None:
                result = {"error": error or "Неизвестная ошибка"}
            step = ToolStep(tool=call.name, args=args if isinstance(args, dict) else {}, result=result, error=error)
            steps.append(step)
            if error and "Данные ERP получить не удалось" in error:
                last_erp_error = True
            if error:
                if "Артикул не подтверждён" in error:
                    result["hint"] = "Вызови search_knowledge, чтобы определить артикул по документам"
            else:
                successful.append(result)
                sources.update(str(src) for src in result.get("sources", []) if isinstance(src, str))
                if call.name in ("search_knowledge", "summarize_topic"):
                    last_knowledge_refused = bool(result.get("refused", False))
            messages.append(tool_message(call, result))

        if len(steps) >= max_steps:
            snippets = [str(item.get("answer", "")) for item in successful if item.get("answer")]
            detail = " ".join(snippets).strip()
            text = "Работа прервана: достигнут лимит вызовов инструментов."
            if detail:
                text += f" Известно: {detail}"
            return AgentResult(text, steps=steps, sources=sorted(sources))
