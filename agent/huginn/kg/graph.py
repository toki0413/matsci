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
        # 显式指定 edges key, 兼容旧数据并消除 NetworkX 3.6 FutureWarning
        edges_key = "links" if "links" in data else "edges"
        self._graph = nx.node_link_graph(data, directed=True, multigraph=False, edges=edges_key)

    def save(self) -> None:
        """Persist graph as JSON node-link data."""
        data = nx.node_link_data(self._graph, edges="links")
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

    def add_hyperedge(
        self,
        node_ids: list[str],
        relation: str,
        *,
        source: str = "auto",
        confidence: float = 0.5,
        **attrs: Any,
    ) -> str | None:
        """Add an n-ary relationship (hyperedge) as a clique + metadata node.

        ponytail: SimplicialComplex (TopoNetX) is the proper structure for this,
        but requires rewriting the KG layer. This clique-based approach captures
        n-ary semantics with zero new dependencies. Upgrade to SimplicialComplex
        when >3-ary relations become common.

        Returns the hyperedge node ID, or None if <2 nodes.
        """
        if len(node_ids) < 2:
            return None
        now = datetime.now().isoformat()
        he_id = f"he_{relation}_{hash(tuple(sorted(node_ids))) & 0xFFFFFFFF:x}"

        # Create metadata node for the hyperedge
        self._graph.add_node(he_id, type="hyperedge", relation=relation,
                             created_at=now, last_seen=now, mentions=1,
                             members=node_ids, **normalize_props(attrs))

        # Connect all members to the hyperedge node (star topology)
        for nid in node_ids:
            if nid in self._graph:
                self._graph.add_edge(nid, he_id, relation="member_of",
                                     source=source, confidence=confidence,
                                     created_at=now, last_seen=now, mentions=1)
        return he_id

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
        return nx.node_link_data(self._graph, edges="links")

    def query(self, seed: str, depth: int = 1, top_k: int = 10) -> dict[str, Any]:
        """Query the graph for seed entities and return a subgraph."""
        from huginn.kg.query import GraphQuery

        q = GraphQuery(self._graph)
        return q.community_aware_query(seed, depth=depth, top_k=top_k)

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

    # 按实体类型分配颜色, 和 provenance DAG 风格一致
    _TYPE_COLORS: dict[str, str] = {
        "Topic": "#bbdefb",
        "Material": "#c8e6c9",
        "Tool": "#fff9c4",
        "Method": "#e1bee7",
        "ErrorPattern": "#ffcdd2",
        "Fact": "#b2dfdb",
        "Session": "#f5f5f5",
        "Resource": "#ffe0b2",
        "Literature": "#d1c4e9",
        "experiment": "#cfd8dc",
    }

    def to_mermaid(self, max_nodes: int = 40) -> str:
        """导出 Mermaid 流程图, 用于 /graph 命令或上下文注入.

        复用 provenance/dag_visualizer 的 Mermaid 模式: 节点按类型着色,
        边标注关系类型. 节点超 max_nodes 只取连接度最高的.
        """
        g = self._graph
        if g.number_of_nodes() == 0:
            return 'graph TD\n  empty["(empty knowledge graph)"]'

        # 超限: 按度排序只取 top-N, 避免图太大没法看
        if g.number_of_nodes() > max_nodes:
            nodes_by_degree = sorted(
                g.degree(), key=lambda x: x[1], reverse=True
            )[:max_nodes]
            keep = {n for n, _ in nodes_by_degree}
            g = g.subgraph(keep).copy()

        lines: list[str] = ["graph TD"]
        type_classes: dict[str, list[str]] = {}

        # 节点
        for i, (nid, data) in enumerate(g.nodes(data=True)):
            mid = f"K{i}"
            label = data.get("label", nid)[:30]
            etype = data.get("type", "Unknown")
            conf = data.get("confidence", 0)
            # Mermaid 节点 label 里引号要转义
            safe_label = label.replace('"', "'").replace("\n", " ")
            lines.append(f'  {mid}["{safe_label}<br/><small>{etype} · conf={conf:.2f}</small>"]')
            type_classes.setdefault(etype, []).append(mid)

        # 边
        for src, dst, edge_data in g.edges(data=True):
            # 找回 Mermaid 节点 id — 用 enumerate 顺序做映射
            # ponytail: O(n) lookup per edge, ok for <100 nodes;
            # build index if graph grows past 500.
            src_idx = list(g.nodes()).index(src)
            dst_idx = list(g.nodes()).index(dst)
            rel = edge_data.get("relation", "→")
            lines.append(f"  K{src_idx} -->|{rel}| K{dst_idx}")

        # 按类型上色
        for etype, ids in type_classes.items():
            color = self._TYPE_COLORS.get(etype, "#f5f5f5")
            # Mermaid classDef 名不能有空格
            cls = etype.replace(" ", "_").lower()
            lines.append(f"  classDef {cls} fill:{color},stroke:#666")
            lines.append(f"  class {','.join(ids)} {cls}")

        return "\n".join(lines)
