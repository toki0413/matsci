"""Query utilities for the project knowledge graph."""

from __future__ import annotations

from typing import Any

import networkx as nx


class GraphQuery:
    """Lightweight query engine over a ProjectKnowledgeGraph."""

    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

    def find_nodes(self, seed: str, top_k: int = 5) -> list[str]:
        """Find nodes whose label contains the seed (case-insensitive)."""
        seed_lower = seed.lower()
        matches: list[tuple[int, str]] = []
        for node, data in self._graph.nodes(data=True):
            label = data.get("label", "")
            if seed_lower in label.lower():
                # Prefer exact matches, then higher confidence, then mentions.
                exact = label.lower() == seed_lower
                score = (
                    int(exact),
                    data.get("confidence", 0.0),
                    data.get("mentions", 0),
                )
                matches.append((score, node))
        matches.sort(reverse=True)
        return [node for _, node in matches[:top_k]]

    def neighborhood(
        self,
        seed_nodes: list[str],
        depth: int = 1,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Return the subgraph induced by seed nodes and their neighbors up to depth."""
        visited: set[str] = set(seed_nodes)
        frontier = set(seed_nodes)
        for _ in range(depth):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self._graph.successors(node))
                next_frontier.update(self._graph.predecessors(node))
            frontier = next_frontier - visited
            visited.update(frontier)

        # If too many nodes, keep only the highest-confidence ones.
        if len(visited) > top_k:
            scored = [(self._graph.nodes[n].get("confidence", 0.0), n) for n in visited]
            scored.sort(reverse=True)
            visited = {n for _, n in scored[:top_k]}

        sub = self._graph.subgraph(visited)
        return {
            "nodes": [{"id": n, **self._graph.nodes[n]} for n in sub.nodes()],
            "edges": [
                {"source": u, "target": v, **d} for u, v, d in sub.edges(data=True)
            ],
        }

    def find_nodes_multi(self, seeds: list[str], top_k: int = 5) -> list[str]:
        """Find nodes matching any of the provided seed labels."""
        found: set[str] = set()
        for seed in seeds:
            found.update(self.find_nodes(seed, top_k=top_k))
        return list(found)

    def query(self, seed: str, depth: int = 1, top_k: int = 10) -> dict[str, Any]:
        """High-level query: extract entities from seed and expand neighborhood."""
        from huginn.kg.extractor import extract_entities

        entities = extract_entities(seed)
        seed_terms = (
            list(entities["tools"])
            + list(entities["methods"])
            + list(entities["materials"])
        )
        # Fallback to the raw seed if no entities were extracted.
        if not seed_terms:
            seed_terms = [seed]
        seeds = self.find_nodes_multi(seed_terms, top_k=max(1, top_k // 2))
        if not seeds:
            return {"nodes": [], "edges": []}
        return self.neighborhood(seeds, depth=depth, top_k=top_k)
