"""Инструменты агента: поиск по графу и операции ERP (FR-6)."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
_PART_PATTERN = re.compile(r"P-\d{4}[A-Z]?\Z")
_MODEL_PATTERN = re.compile(r"(?:AR|BZ|GS)-\d{3}\Z")


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_PART = {"type": "string", "description": "Артикул в формате P-1042 или P-1042B"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES, "description": "Название склада"}

TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Поиск точных фактов по документам: коды ошибок, модель, компонент, деталь и замены.",
        {"question": {"type": "string", "description": "Вопрос для локального графового поиска"}},
        ["question"],
    ),
    _schema(
        "summarize_topic",
        "Обзорная сводка по документам, например частые отказы по линейке.",
        {"question": {"type": "string", "description": "Тема для глобального поиска"}},
        ["question"],
    ),
    _schema(
        "check_stock",
        "Узнать количество детали на конкретном складе и снята ли она с производства.",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
        ["part_id", "warehouse"],
    ),
    _schema(
        "get_price",
        "Узнать актуальную цену в рублях и признак снятия детали с производства.",
        {"part_id": _PART},
        ["part_id"],
    ),
    _schema(
        "create_service_order",
        "Создать сервисный заказ. Использовать только после явного подтверждения человека.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1, "description": "Количество деталей"},
            "model": {"type": "string", "description": "Модель оборудования, например AR-210"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


def _check_part(part_id: object) -> str | None:
    if not isinstance(part_id, str) or not _PART_PATTERN.fullmatch(part_id):
        return "Неверный артикул детали: ожидается P-1042 или P-1042B"
    return None


def _check_warehouse(warehouse: object) -> str | None:
    if not isinstance(warehouse, str) or warehouse not in WAREHOUSES:
        return "Неизвестный склад: укажите Москва, Санкт-Петербург, Казань или Новосибирск"
    return None


def _erp_result(operation: Callable[[], dict]) -> dict:
    """Ошибки запросов/отказов превращаем в словарь; недоступность ERP передаём агенту."""
    try:
        return operation()
    except ErpNotFound as exc:
        return {"error": f"Данных нет в ERP: {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Создать пять функций, возвращающих сериализуемые словари (FR-6)."""

    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Нужен непустой вопрос"}
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": list(answer.facts),
            "sources": list(answer.sources),
            "refused": answer.refused,
        }

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Нужна непустая тема"}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": list(answer.sources), "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        error = _check_part(part_id) or _check_warehouse(warehouse)
        return {"error": error} if error else _erp_result(lambda: erp.get_stock(part_id, warehouse))

    def get_price(part_id: str) -> dict:
        error = _check_part(part_id)
        return {"error": error} if error else _erp_result(lambda: erp.get_price(part_id))

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        error = _check_part(part_id) or _check_warehouse(warehouse)
        if error:
            return {"error": error}
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": "Количество должно быть положительным целым числом"}
        if not isinstance(model, str) or not _MODEL_PATTERN.fullmatch(model):
            return {"error": "Неверный код модели оборудования"}
        return _erp_result(lambda: erp.create_order(part_id, warehouse, qty, model))

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
