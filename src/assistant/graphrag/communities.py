"""Поиск сообществ и построение сводок. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

LOUVAIN_SEED = 7
SUMMARY_WORD_LIMIT = 150


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Выделяет непересекающиеся сообщества методом Лувена.

    Алгоритм работает с неориентированным представлением графа и учитывает
    ``weight``. Результат сортируется, чтобы ID не зависели от порядка обхода.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))

    for source, target, data in graph.edges(data=True):
        weight = max(0, int(data.get("weight", 1)))
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)

    if not undirected:
        return []

    groups = nx.community.louvain_communities(
        undirected, weight="weight", seed=LOUVAIN_SEED
    )
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: nodes[0])
    return [Community(community_id=index, nodes=nodes) for index, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Создаёт сводку <= 150 слов за один вызов LLM.

    Частоты отказов и другие численные показатели вычисляются в Python;
    модель получает только задачу связно изложить вычисленные факты.
    """
    members = set(community.nodes) & set(graph.nodes)
    names = sorted(str(graph.nodes[node].get("name", node.split(":", 1)[-1])) for node in members)
    relation_counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    facts: list[str] = []

    for source, target, attrs in graph.edges(data=True):
        if source not in members or target not in members:
            continue
        kind = str(attrs.get("type", ""))
        weight = max(0, int(attrs.get("weight", 1)))
        relation_counts[kind] += weight
        source_name = str(graph.nodes[source].get("name", source.split(":", 1)[-1]))
        target_name = str(graph.nodes[target].get("name", target.split(":", 1)[-1]))
        facts.append(f"{source_name} -{kind}-> {target_name} (вес {weight})")
        if kind == "FAILED":
            failures[target_name] += weight

    frequency_lines = [
        f"{name}: {value}"
        for name, value in sorted(failures.items(), key=lambda item: (-item[1], item[0]))
    ]
    relation_lines = [
        f"{kind}: {count}"
        for kind, count in sorted(relation_counts.items())
    ]
    prompt = (
        f"Сообщество №{community.community_id}. Число сущностей: {len(members)}.\n"
        f"Сущности: {', '.join(names)}\n"
        "Сумма весов связей по типам (вычислено кодом):\n"
        + ("\n".join(relation_lines) or "нет")
        + "\nЧастота отказов узлов по весам рёбер FAILED (вычислено кодом):\n"
        + ("\n".join(frequency_lines) or "нет данных об отказах")
        + "\nПроверенные связи внутри сообщества:\n"
        + ("\n".join(sorted(facts))[:12000] or "нет")
        + "\nНапиши связную сводку по этим данным (не более 150 слов). "
        "Нельзя придумывать дополнительные числа, сущности и связи. "
        "Если данных об отказах нет, не называй самый частый отказ."
    )
    result = llm.chat(
        [
            {
                "role": "system",
                "content": "Ты составитель точных сводок графа знаний. Используй только предоставленные факты.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=320,
    )
    return " ".join(result.content.split()[:SUMMARY_WORD_LIMIT])
