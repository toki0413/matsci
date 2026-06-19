"""Project-level knowledge graph backed by NetworkX."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from huginn.kg.entities import node_id, normalize_props


class ProjectKnowledgeGraph:
    """A local, persistent knowledge graph for a workspace."""

    FILENAME = "project_kg.json"

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self._graph = nx.DiGraph()
        if self.path.exists():
            self.load()

    def load(self) -> None:
        """Load graph from JSON node-link data."""
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._graph = nx.node_link_graph(data, directed=True, multigraph=False)

    def save(self) -> None:
        """Persist graph as JSON node-link data."""
        data = nx.node_link_data(self._graph)
        self.path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def add_entity(
        self,
        label: str,
        entity_type: str,
        *,
        source: str = "auto",
        confidence: float = 0.5,
        **attrs: Any,
    ) -> str:
        """Add or update a node and return its stable id."""
        eid = node_id(label, entity_type)
        now = datetime.now().isoformat()
        if eid in self._graph:
            self._graph.nodes[eid]["mentions"] = (
                self._graph.nodes[eid].get("mentions", 0) + 1
            )
            self._graph.nodes[eid]["last_seen"] = now
            # Update confidence upward slightly on re-encounter.
            old_conf = self._graph.nodes[eid].get("confidence", confidence)
            self._graph.nodes[eid]["confidence"] = min(0.99, old_conf + 0.05)
        else:
            self._graph.add_node(
                eid,
                label=label,
                type=entity_type,
                source=source,
                confidence=confidence,
                created_at=now,
                last_seen=now,
                mentions=1,
                **normalize_props(attrs),
            )
        return eid

    def add_relation(
        self,
        src_id: str,
        relation: str,
        dst_id: str,
        *,
        source: str = "auto",
        confidence: float = 0.5,
        **attrs: Any,
    ) -> None:
        """Add or update a directed edge between two existing nodes."""
        if src_id not in self._graph or dst_id not in self._graph:
            return
        now = datetime.now().isoformat()
        if self._graph.has_edge(src_id, dst_id):
            data = self._graph.edges[src_id, dst_id]
            data["mentions"] = data.get("mentions", 0) + 1
            data["last_seen"] = now
            old_conf = data.get("confidence", confidence)
            data["confidence"] = min(0.99, old_conf + 0.05)
        else:
            self._graph.add_edge(
                src_id,
                dst_id,
                relation=relation,
                source=source,
                confidence=confidence,
                created_at=now,
                last_seen=now,
                mentions=1,
                **normalize_props(attrs),
            )

    def has_entity(self, label: str, entity_type: str) -> bool:
        return node_id(label, entity_type) in self._graph

    def get_entity(self, label: str, entity_type: str) -> dict[str, Any] | None:
        eid = node_id(label, entity_type)
        if eid in self._graph:
            return dict(self._graph.nodes[eid])
        return None

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "node_types": self._count_node_types(),
        }

    def _count_node_types(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, data in self._graph.nodes(data=True):
            t = data.get("type", "Unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def export(self, fmt: str = "json") -> str | dict[str, Any]:
        """Export graph as JSON node-link data or GML string."""
        if fmt == "gml":
            return "\n".join(nx.generate_gml(self._graph))
        return nx.node_link_data(self._graph)

    def query(self, seed: str, depth: int = 1, top_k: int = 10) -> dict[str, Any]:
        """Query the graph for seed entities and return a subgraph."""
        from huginn.kg.query import GraphQuery

        q = GraphQuery(self._graph)
        return q.query(seed, depth=depth, top_k=top_k)

    def to_text(self, nodes: set[str]) -> str:
        """Convert a set of node ids into a prompt-friendly text summary."""
        lines: list[str] = []
        for eid in sorted(nodes):
            data = self._graph.nodes[eid]
            label = data.get("label", eid)
            etype = data.get("type", "Unknown")
            lines.append(f"- {etype}:{label}")
            # Add outgoing edges
            for _, dst, edge_data in self._graph.out_edges(eid, data=True):
                dst_label = self._graph.nodes[dst].get("label", dst)
                rel = edge_data.get("relation", "related_to")
                lines.append(f"  → {rel} {dst_label}")
        return "\n".join(lines)
