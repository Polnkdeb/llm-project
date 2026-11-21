
"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ENTITY_TYPES, Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Собрать единый направленный граф с подтверждениями из разных чанков (FR-2)."""
    graph = nx.DiGraph()
    graph.graph["conflicts"] = 0

    node_sources: dict[str, set[str]] = {}
    edge_sources: dict[tuple[str, str], set[str]] = {}

    def add_node(entity: Entity, chunk_id: str) -> None:
        if entity.type not in ENTITY_TYPES:
            return

        node_id = entity.node_id

        if node_id not in graph:
            graph.add_node(
                node_id,
                type=entity.type,
                name=entity.name,
                sources=[],
            )

        node_sources.setdefault(node_id, set()).add(chunk_id)

    for result in results:
        for entity in result.entities:
            add_node(entity, result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            add_node(relation.subject, result.chunk_id)
            add_node(relation.object, result.chunk_id)

            start = relation.subject.node_id
            end = relation.object.node_id

            # Исправление 1: исключаем замену детали самой собой
            if relation.type == "REPLACED_BY" and start == end:
                continue

            edge_sources.setdefault((start, end), set()).add(
                result.chunk_id
            )

            graph.add_edge(
                start,
                end,
                type=relation.type,
                weight=0,
                sources=[],
            )

    # Записываем источники для каждой вершины
    for node_id, sources in node_sources.items():
        graph.nodes[node_id]["sources"] = sorted(sources)

    # Записываем веса и источники для каждой связи
    for (start, end), sources in edge_sources.items():
        graph.edges[start, end]["weight"] = len(sources)
        graph.edges[start, end]["sources"] = sorted(sources)

    # Обработка конфликтов REPLACED_BY
    seen: set[tuple[str, str]] = set()

    for start, end, attrs in list(graph.edges(data=True)):

        # Исправление 2: защита от петель
        if start == end:
            continue

        if attrs["type"] != "REPLACED_BY":
            continue

        if not graph.has_edge(end, start):
            continue

        if graph.edges[end, start]["type"] != "REPLACED_BY":
            continue

        pair = tuple(sorted((start, end)))

        if pair in seen:
            continue

        seen.add(pair)
        graph.graph["conflicts"] += 1

        forward = graph.edges[start, end]["weight"]
        backward = graph.edges[end, start]["weight"]

        # Оставляем направление с большим числом подтверждений
        if forward <= backward:
            graph.remove_edge(start, end)

        if backward <= forward:
            graph.remove_edge(end, start)

    return graph
