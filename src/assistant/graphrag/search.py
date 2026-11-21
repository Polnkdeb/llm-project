"""Локальный и глобальный поиск по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL_TEXT = "В базе знаний не найдено достаточно информации для ответа."
FACTS_CHAR_LIMIT = 8000


def _document_sources(chunks: list[str] | set[str]) -> list[str]:
    """Преобразует идентификаторы чанков в идентификаторы документов."""
    return sorted({str(chunk).split("#", 1)[0] for chunk in chunks if chunk})


def _anchors(question: str, kb: KnowledgeBase) -> tuple[list[str], bool]:
    """Возвращает опорные узлы и признак неизвестного кода."""
    graph = kb.graph
    codes = find_codes(question)
    code_nodes = [f"{code_type(code)}:{code}" for code in codes]
    if any(node not in graph for node in code_nodes):
        return [], True

    name_nodes = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    nodes = list(dict.fromkeys(node for node in code_nodes + name_nodes if node in graph))
    return nodes, False


def _edges_on_paths(graph: nx.DiGraph, anchors: list[str]) -> tuple[set[tuple[str, str]], set[str]]:
    """Возвращает только рёбра на путях до трёх шагов между опорами."""
    undirected = graph.to_undirected()
    edges: set[tuple[str, str]] = set()
    visited_nodes: set[str] = set()
    for start, end in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, start, end, cutoff=3):
            visited_nodes.update(path)
            for left, right in zip(path, path[1:], strict=False):
                if graph.has_edge(left, right):
                    edges.add((left, right))
                if graph.has_edge(right, left):
                    edges.add((right, left))
    return edges, visited_nodes


def _neighbourhood_edges(graph: nx.DiGraph, anchor: str) -> tuple[set[tuple[str, str]], set[str]]:
    """Двухшаговая окрестность; направление при обходе не имеет значения."""
    undirected = graph.to_undirected()
    nearby = set(nx.single_source_shortest_path_length(undirected, anchor, cutoff=2))
    edges = {(u, v) for u, v in graph.edges if u in nearby and v in nearby}
    return edges, nearby


def _replacement_edges(graph: nx.DiGraph, selected_nodes: set[str]) -> set[tuple[str, str]]:
    """Добавляет ориентированные цепочки замены от деталей на найденных путях."""
    todo = [node for node in selected_nodes if graph.nodes[node].get("type") == "Part"]
    visited = set(todo)
    result: set[tuple[str, str]] = set()
    while todo:
        current = todo.pop()
        for next_node in graph.successors(current):
            if graph[current][next_node].get("type") != "REPLACED_BY":
                continue
            result.add((current, next_node))
            if next_node not in visited:
                visited.add(next_node)
                todo.append(next_node)
    return result


def _format_fact(graph: nx.DiGraph, source: str, target: str) -> str:
    source_name = graph.nodes[source].get("name", source.split(":", 1)[-1])
    target_name = graph.nodes[target].get("name", target.split(":", 1)[-1])
    return f"{source_name} -{graph[source][target]['type']}-> {target_name}"


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Ответ по подтверждённым фактам локального подграфа."""
    graph = kb.graph
    anchors, unknown_code = _anchors(question, kb)
    if unknown_code or not anchors:
        return Answer(text=REFUSAL_TEXT, refused=True)

    if len(anchors) == 1:
        selected, path_nodes = _neighbourhood_edges(graph, anchors[0])
    else:
        selected, path_nodes = _edges_on_paths(graph, anchors)

    # Дополняем именно найденные маршруты, а не весь граф, актуальными заменами.
    replacements = _replacement_edges(graph, path_nodes)
    ordered_edges = sorted(replacements, key=lambda edge: (edge[0], edge[1]))
    ordered_edges += sorted(selected - replacements)

    facts: list[str] = []
    used_edges: list[tuple[str, str]] = []
    chars = 0
    for source, target in ordered_edges:
        fact = _format_fact(graph, source, target)
        numbered = f"{len(facts) + 1}. {fact}\n"
        if chars + len(numbered) > FACTS_CHAR_LIMIT:
            break
        facts.append(fact)
        used_edges.append((source, target))
        chars += len(numbered)

    if not facts:
        return Answer(text=REFUSAL_TEXT, refused=True)

    sources = _document_sources(
        {chunk for source, target in used_edges for chunk in graph[source][target].get("sources", [])}
    )
    evidence = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Ты ассистент сервисной службы. Отвечай только по пронумерованным фактам. "
                    "Не придумывай детали, коды и цены. Если цепочка REPLACED_BY ведёт к новой "
                    "детали, называй конечную актуальную деталь. Если фактов недостаточно, "
                    "скажи: в базе знаний не найдено."
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\nПроверенные факты:\n{evidence}"},
        ]
    )
    text = response.content.strip()
    if not text:
        return Answer(text=REFUSAL_TEXT, sources=sources, facts=facts, refused=True)
    return Answer(text=text, sources=sources, facts=facts)


def _community_sources(kb: KnowledgeBase, nodes: list[str]) -> list[str]:
    """Источники узлов и рёбер сообщества, сведённые к документам."""
    graph = kb.graph
    selected = set(nodes) & set(graph.nodes)
    chunks = {item for node in selected for item in graph.nodes[node].get("sources", [])}
    chunks.update(
        item
        for source, target, attrs in graph.edges(data=True)
        if source in selected and target in selected
        for item in attrs.get("sources", [])
    )
    return _document_sources(chunks)


def _parse_map_answer(raw: str) -> tuple[str, int] | None:
    """Пропускает битые или бесполезные частичные ответы."""
    try:
        parsed = json.loads(raw.strip())
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    answer, score = parsed.get("answer"), parsed.get("score")
    if not isinstance(answer, str) or not answer.strip():
        return None
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
        return None
    return answer.strip(), score


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Map/Reduce по сводкам сообществ; частичный JSON валидируется кодом."""
    useful: list[tuple[int, str, list[str]]] = []
    for community in kb.communities:
        if not community.summary.strip():
            continue
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Ответь на вопрос только на основе сводки. Верни JSON с ключами "
                        "answer (текст) и score (целое от 0 до 10). "
                        "Если сводка бесполезна, выставь score=0. Не выдумывай факты."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {question}\nСообщество №{community.community_id}. "
                        f"Сводка: {community.summary}"
                    ),
                },
            ],
            json_schema=MAP_SCHEMA,
        )
        parsed = _parse_map_answer(response.content)
        if parsed is None:
            continue
        answer, score = parsed
        if score > 0:
            useful.append((score, answer, _community_sources(kb, community.nodes)))

    if not useful:
        return Answer(text=REFUSAL_TEXT, refused=True)

    useful.sort(key=lambda result: -result[0])
    pieces = "\n".join(
        f"{index}. (полезность {score}/10) {answer}"
        for index, (score, answer, _) in enumerate(useful, start=1)
    )
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы по сводкам сообществ в один точный ответ. "
                    "Не добавляй факты, отсутствующие в частичных ответах. "
                    "Если есть противоречия, укажи на них."
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\nПолезные частичные ответы:\n{pieces}"},
        ]
    )
    sources = sorted({source for _, _, items in useful for source in items})
    text = response.content.strip()
    if not text:
        return Answer(text=REFUSAL_TEXT, sources=sources, refused=True)
    return Answer(text=text, sources=sources)
