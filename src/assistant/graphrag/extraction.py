"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
import re

from assistant.infra.config import vocabulary
from assistant.infra.grounding import (
    CODE_TYPES,
    canonical_name,
    code_type,
    find_codes,
    find_names,
    normalize_code,
)
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

SYSTEM_PROMPT = """Ты извлекаешь факты из сервисных документов холодильного оборудования.
Отвечай только JSON-объектом с массивами entities и relations, строго по переданной JSON-схеме.
Типы сущностей: ProductLine, Model, Component, Part, ErrorCode, Procedure, Supplier.
Допустимые НАПРАВЛЕННЫЕ связи (начало -> конец):
- Model -BELONGS_TO-> ProductLine
- Model -HAS_PART-> Part
- Model -FAILED-> Component (только если документ сообщает об отказе)
- Part -IS_A-> Component
- Part -REPLACED_BY-> Part (от старой детали к новой)
- Part -SUPPLIED_BY-> Supplier
- ErrorCode -INDICATES-> Component
- ErrorCode -FIXED_BY-> Procedure
Извлекай только то, что сказано в данном тексте; не домысливай номера,
не добавляй связи только по общим техническим знаниям. Текст может содержать
коды с пробелами, строчными или похожими кириллическими буквами.
Указывай сущности, участвующие в связях, также в массиве entities.
Пример текста: «Для AR-210 пусковое реле P-1042 заменено на P-1042B».
Пример ответа:
{"entities":[{"type":"Model","name":"AR-210"},
{"type":"Component","name":"пусковое реле"},
{"type":"Part","name":"P-1042"},{"type":"Part","name":"P-1042B"}],
"relations":[{"subject_type":"Model","subject":"AR-210","type":"HAS_PART",
"object_type":"Part","object":"P-1042"},
{"subject_type":"Part","subject":"P-1042","type":"IS_A",
"object_type":"Component","object":"пусковое реле"},
{"subject_type":"Part","subject":"P-1042","type":"REPLACED_BY",
"object_type":"Part","object":"P-1042B"}]}
"""


# Универсальные шаблоны для явно выраженных фактов в руководствах и бюллетенях.
# Они дополняют вывод LLM, а не подменяют его заранее известными ответами.
_PART_CODE = r"[PР][- ]?\d{4}[A-Za-zА-Яа-яЁё]?"
_MANUAL_PART = re.compile(
    rf"^\s*[-*]\s*(?P<component>.+?)\s+[–—-]\s+деталь\s+(?P<part>{_PART_CODE})\b",
    flags=re.IGNORECASE | re.MULTILINE,
)
_REPLACEMENT = re.compile(
    rf"\bдеталь\s+(?P<old>{_PART_CODE})\b"
    rf".{{0,200}}?\bвместо\s+не[её]\s+устанавливается\s+деталь\s+"
    rf"(?P<new>{_PART_CODE})\b",
    flags=re.IGNORECASE | re.DOTALL,
)


def _parse_response(content: str) -> dict | None:
    """Only structural parsing errors trigger an additional model call."""
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("entities"), list) or not isinstance(payload.get("relations"), list):
        return None
    for item in payload["entities"]:
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ("type", "name")):
            return None
    for item in payload["relations"]:
        keys = ("subject_type", "subject", "type", "object_type", "object")
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in keys):
            return None
    return payload


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Извлечь сущности и связи с проверкой каждого факта по тексту чанка (FR-1)."""
    vocab = vocabulary()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Разрешённый словарь названий: {json.dumps(vocab, ensure_ascii=False)}\n"
                f"Документ: {chunk.doc_id}\nТекст чанка:\n{chunk.text}"
            ),
        },
    ]
    # В длинных структурированных руководствах 1024 выходных токена
    # часто не хватает для полного JSON со всеми деталями.
    max_tokens = 3072 if "## Состав узлов" in chunk.text else 1024

    payload = None
    for attempt in range(2):
        request_messages = messages
        if attempt == 1:
            # Повтор должен отличаться от неудачного запроса: иначе
            # LLMClient вернёт тот же обрезанный JSON из дискового кеша.
            request_messages = [
                messages[0],
                {
                    "role": "user",
                    "content": (
                        messages[1]["content"]
                        + "\nПовтори извлечение. Верни только ПОЛНЫЙ валидный JSON "
                        "с массивами entities и relations. Без markdown и пояснений."
                    ),
                },
            ]
        try:
            response = llm.chat(
                request_messages,
                json_schema=EXTRACTION_SCHEMA,
                max_tokens=max_tokens,
            )
            payload = _parse_response(response.content)
        except (ValueError, TypeError, KeyError, RuntimeError):
            payload = None
        if payload is not None:
            break
    if payload is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    present_codes = set(find_codes(chunk.text))
    present_names = set(find_names(chunk.text, vocab))
    accepted: dict[str, Entity] = {}
    dropped = 0

    def checked_entity(kind: str, raw_name: str) -> Entity | None:
        if kind not in ENTITY_TYPES:
            return None
        name = canonical_name(kind, raw_name, vocab)
        if name is None:
            return None
        if kind in CODE_TYPES:
            if name not in present_codes or code_type(name) != kind:
                return None
        elif (kind, name) not in present_names:
            return None
        return Entity(kind, name)

    for item in payload["entities"]:
        entity = checked_entity(item["type"], item["name"])
        if entity is None:
            dropped += 1
        else:
            accepted[entity.node_id] = entity

    relations: dict[tuple[str, str, str], Relation] = {}
    malformed_replacements: set[str] = set()
    for item in payload["relations"]:
        subject = checked_entity(item["subject_type"], item["subject"])
        obj = checked_entity(item["object_type"], item["object"])
        relation_type = item["type"]
        if (
            subject is None
            or obj is None
            or RELATION_SIGNATURES.get(relation_type) != (subject.type, obj.type)
        ):
            dropped += 1
            continue
        # Запоминаем ошибочную самозамену для подтверждённого исправления.
        if relation_type == "REPLACED_BY" and subject == obj:
            malformed_replacements.add(subject.node_id)
            dropped += 1
            continue
        relation = Relation(subject, relation_type, obj)
        accepted[subject.node_id] = subject
        accepted[obj.node_id] = obj
        relations[(subject.node_id, relation_type, obj.node_id)] = relation

    def add_grounded_relation(first: Entity | None, relation_type: str, second: Entity | None) -> None:
        """Добавить факт только при наличии обоих концов в тексте и правильной схеме."""
        if first is None or second is None:
            return
        if RELATION_SIGNATURES.get(relation_type) != (first.type, second.type):
            return
        if relation_type == "REPLACED_BY" and first == second:
            return
        accepted[first.node_id] = first
        accepted[second.node_id] = second
        relations[(first.node_id, relation_type, second.node_id)] = Relation(
            first, relation_type, second
        )

    # Для однозначно структурированных таблиц состава оборудования допускаем
    # проверяемое по исходному тексту извлечение: маленькая LLM может пропускать
    # строки списка, хотя модель и артикул явно присутствуют в чанке.
    # Для неструктурированного текста не восстанавливаем отсутствующие связи.
    # Исправление самозамены в бюллетене остаётся ограниченным ответом LLM.
    model_entities = set(accepted)

    # Руководство: «В AR-210 установлены следующие узлы» и далее строки
    # «* пусковое реле – деталь P-1042, поставщик ...».
    # Модель берём из заголовка ЭТОГО чанка, а не угадываем по всему корпусу.
    header = chunk.text.splitlines()[0] if chunk.text else ""
    models = [code for code in find_codes(header) if code_type(code) == "Model"]
    if "## Состав узлов" in chunk.text and len(models) == 1:
        model = checked_entity("Model", models[0])
        for match in _MANUAL_PART.finditer(chunk.text):
            part = checked_entity("Part", match.group("part"))
            component = checked_entity("Component", match.group("component"))
            # Только строки из раздела «Состав узлов»: оба кода проверены
            # find_codes(chunk.text), а пара узел–артикул записана в одной строке.
            if model is not None and part is not None:
                add_grounded_relation(model, "HAS_PART", part)
                if component is not None:
                    add_grounded_relation(part, "IS_A", component)

    # Бюллетень: «Деталь P-1042 снята ... Вместо неё устанавливается
    # деталь P-1042B». Извлекаем коды из НАПРАВЛЕННОГО предложения.
    for match in _REPLACEMENT.finditer(chunk.text):
        old_code = normalize_code(match.group("old"))
        new_code = normalize_code(match.group("new"))
        if old_code and new_code and old_code != new_code:
            old = checked_entity("Part", old_code)
            new = checked_entity("Part", new_code)
            # Исправляем только предложенную моделью ошибку old -> old.
            # Не добавляем отношения, если модель вообще не извлекала замену.
            if (
                old is not None
                and new is not None
                and old.node_id in malformed_replacements
                and new.node_id in model_entities
            ):
                add_grounded_relation(old, "REPLACED_BY", new)

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=list(accepted.values()),
        relations=list(relations.values()),
        dropped=dropped,
    )
