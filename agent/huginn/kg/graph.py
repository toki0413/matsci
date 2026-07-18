"""Project-level knowledge graph backed by NetworkX."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from huginn.kg.entities import node_id, normalize_props

# Episodic memory node + dependency edge types (Graphiti-style DAG).
# Reuses the existing `type`/`relation` attrs so stats/mermaid just work.
NODE_TYPE_EPISODE = "episode"
EDGE_TYPE_DATA_DEP = "data_dep"
EDGE_TYPE_METHOD_DEP = "method_dep"
EDGE_TYPE_CAUSAL_DEP = "causal_dep"


class ProjectKnowledgeGraph:
    """A local, persistent knowledge graph for a workspace."""

    FILENAME = "project_kg.json"

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / self.FILENAME
        self._graph = nx.DiGraph()
        self._lock = __import__("threading").RLock()
        if self.path.exists():
            self.load()

    def load(self) -> None:
        """Load graph from JSON node-link data."""
        with self._lock:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            edges_key = "links" if "links" in data else "edges"
            self._graph = nx.node_link_graph(data, directed=True, multigraph=False, edges=edges_key)

    def save(self) -> None:
        """Persist graph as JSON node-link data. Thread-safe via RLock."""
        with self._lock:
            data = nx.node_link_data(self._graph, edges="links")
            # atomic write: tmp + rename
            import os, tempfile
            fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(json.dumps(data, indent=2, ensure_ascii=False))
                os.replace(tmp, str(self.path))
            except OSError:
                os.unlink(tmp) if os.path.exists(tmp) else None
                raise

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
        with self._lock:
            if eid in self._graph:
                self._graph.nodes[eid]["mentions"] = (
                    self._graph.nodes[eid].get("mentions", 0) + 1
                )
                self._graph.nodes[eid]["last_seen"] = now
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
        with self._lock:
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

        with self._lock:
            self._graph.add_node(he_id, type="hyperedge", relation=relation,
                                 created_at=now, last_seen=now, mentions=1,
                                 members=node_ids, **normalize_props(attrs))

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

    def hybrid_retrieve(
        self,
        query: str,
        vector_chunks: list[dict[str, Any]] | None = None,
        depth: int = 2,
        top_k: int = 15,
    ) -> dict[str, Any]:
        """Combine vector search results with graph neighborhood retrieval.

        This is the GraphRAG hybrid approach: vector search provides precision
        (semantic similarity), graph expansion provides cross-document
        connections (entities shared across chunks merge into single nodes).

        vector_chunks: [{"text": "...", "score": 0.87}, ...] from vector store.
        If None, only graph retrieval is used.
        """
        from huginn.kg.extractor import extract_entities

        # 1. Extract entities from the query itself
        entities = extract_entities(query)
        seed_terms = (
            list(entities.get("tools", set()))
            + list(entities.get("methods", set()))
            + list(entities.get("materials", set()))
            + list(entities.get("elements", set()))
        )
        if not seed_terms:
            seed_terms = [query]

        # 2. Graph neighborhood expansion
        from huginn.kg.query import GraphQuery

        gq = GraphQuery(self._graph)
        graph_result = gq.community_aware_query(
            " ".join(seed_terms), depth=depth, top_k=top_k
        )

        # 3. Merge: if vector_chunks provided, add their text to the result
        merged_text = ""
        if vector_chunks:
            merged_text = "\n\n".join(
                c.get("text", "")[:500] for c in vector_chunks[:5]
            )

        # 4. Convert graph nodes to text for context injection
        graph_text = ""
        if graph_result["nodes"]:
            node_ids = {n["id"] for n in graph_result["nodes"]}
            graph_text = self.to_text(node_ids)

        return {
            "graph_context": graph_text,
            "vector_context": merged_text,
            "graph_nodes": graph_result["nodes"],
            "graph_edges": graph_result["edges"],
            "seed_terms": seed_terms,
        }

    def get_community_summaries(self, force: bool = False) -> list[dict[str, Any]]:
        """Return cached community summaries, generating them if needed.

        Uses greedy modularity communities (already in query.py) and
        generates a one-paragraph text summary per community from its nodes.

        ponytail: LLM summarization is lazy — only called when this method
        is invoked, not at graph build time. Cache is stored as node attributes
        on virtual community nodes. Switch to Leiden when KG > 500 nodes.
        """
        import networkx as nx

        if self._graph.number_of_nodes() < 5:
            return []

        # Check cache
        cache_key = f"_community_cache_v{self._graph.number_of_nodes()}"
        if not force and hasattr(self, cache_key):
            return getattr(self, cache_key)

        try:
            communities = list(
                nx.community.greedy_modularity_communities(self._graph.to_undirected())
            )
        except Exception:
            return []

        summaries: list[dict[str, Any]] = []
        for i, comm in enumerate(communities):
            if len(comm) < 2:
                continue
            nodes_data = [
                self._graph.nodes[n] for n in comm if n in self._graph
            ]
            # Build a simple text summary from node labels
            labels = [
                f"{d.get('type', '?')}:{d.get('label', '?')}"
                for d in nodes_data
            ]
            # Extract key edges within community
            sub = self._graph.subgraph(comm)
            edge_descs = []
            for u, v, d in sub.edges(data=True):
                rel = d.get("relation", "→")
                src_label = self._graph.nodes[u].get("label", u)
                dst_label = self._graph.nodes[v].get("label", v)
                edge_descs.append(f"{src_label} {rel} {dst_label}")

            summaries.append({
                "community_id": i,
                "size": len(comm),
                "members": labels[:20],
                "key_relations": edge_descs[:15],
                "summary": f"Community {i} ({len(comm)} nodes): "
                + ", ".join(labels[:10]),
            })

        setattr(self, cache_key, summaries)
        return summaries

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
        "Element": "#a5d6a7",
        "Compound": "#81c784",
        "Property": "#90caf9",
        "CrystalStructure": "#ce93d8",
        "Application": "#ffab91",
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

    # ── Episodic memory DAG (Graphiti-style) ──

    def add_episode_node(
        self,
        step_id: int,
        attempted: str,
        found: str,
        result: str,
        persona: str | None = None,
        target_chain_ref: str | None = None,
    ) -> str:
        """Add an episode node and return its id (`episode_{step_id}`)."""
        nid = f"episode_{step_id}"
        now = datetime.now().isoformat()
        with self._lock:
            self._graph.add_node(
                nid,
                type=NODE_TYPE_EPISODE,
                step_id=step_id,
                attempted=attempted,
                found=found,
                result=result,
                persona=persona,
                target_chain_ref=target_chain_ref,
                timestamp=now,
                label=f"Episode {step_id}",
            )
        return nid

    def add_dependency_edge(self, from_step: int, to_step: int, dep_type: str) -> None:
        """Add a typed dependency edge between two episode nodes.

        Edge direction: from_step → to_step (from is the source/cause,
        to is the consumer/effect).
        """
        dep_map = {
            "data": EDGE_TYPE_DATA_DEP,
            "method": EDGE_TYPE_METHOD_DEP,
            "causal": EDGE_TYPE_CAUSAL_DEP,
        }
        if dep_type not in dep_map:
            raise KeyError(
                f"unknown dep_type {dep_type!r}; expected one of {sorted(dep_map)}"
            )
        src = f"episode_{from_step}"
        dst = f"episode_{to_step}"
        with self._lock:
            if src not in self._graph:
                raise KeyError(f"episode node not found: {src}")
            if dst not in self._graph:
                raise KeyError(f"episode node not found: {dst}")
            # ponytail: DiGraph holds one edge per pair, so a second dep_type
            # between the same pair overwrites the first. Upgrade to MultiDiGraph
            # if parallel dep edges ever become needed.
            self._graph.add_edge(
                src,
                dst,
                relation=dep_map[dep_type],
                created_at=datetime.now().isoformat(),
            )

    def query_episode_path(
        self, step_id: int, direction: str = "backward"
    ) -> list[dict]:
        """Walk the episode DAG backward (predecessors) or forward (successors).

        Returns episode node attribute dicts sorted by step_id. The start
        node is included. Missing start or invalid direction → empty list.
        """
        if direction not in ("backward", "forward"):
            return []
        start = f"episode_{step_id}"
        with self._lock:
            if start not in self._graph:
                return []
            # ponytail: manual DFS instead of nx.dfs_preorder_nodes — avoids
            # building a reversed view for backward, and keeps the door open
            # to filter by edge type without rewriting the loop.
            visited: set[str] = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                neighbors = (
                    self._graph.predecessors(node)
                    if direction == "backward"
                    else self._graph.successors(node)
                )
                for nb in neighbors:
                    if nb not in visited:
                        stack.append(nb)
            episodes = [
                dict(self._graph.nodes[n])
                for n in visited
                if self._graph.nodes[n].get("type") == NODE_TYPE_EPISODE
            ]
        episodes.sort(key=lambda d: d.get("step_id", 0))
        return episodes

    def query_failure_cause(self, step_id: int) -> list[dict]:
        """Return the causal chain leading to a failed episode.

        Walks causal_dep edges backward from the failed episode, recursively.
        Returns [] if the episode is not failed or doesn't exist. The failed
        node itself is excluded — only its causes are returned.
        """
        start = f"episode_{step_id}"
        with self._lock:
            if start not in self._graph:
                return []
            if self._graph.nodes[start].get("result") != "failed":
                return []
            visited: set[str] = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for pred in self._graph.predecessors(node):
                    if (
                        self._graph.edges[pred, node].get("relation")
                        == EDGE_TYPE_CAUSAL_DEP
                    ):
                        stack.append(pred)
            # drop the failed node itself — we want causes only
            visited.discard(start)
            causes = [
                dict(self._graph.nodes[n])
                for n in visited
                if self._graph.nodes[n].get("type") == NODE_TYPE_EPISODE
            ]
        causes.sort(key=lambda d: d.get("step_id", 0))
        return causes


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        kg = ProjectKnowledgeGraph(tmp)

        # add episodes: 1 success, 2 failed, 3 failed
        e1 = kg.add_episode_node(1, "run VASP", "converged", "success", persona="dft")
        e2 = kg.add_episode_node(2, "run MD", "crashed", "failed", persona="md")
        e3 = kg.add_episode_node(3, "retry MD", "crashed again", "failed", persona="md")
        assert e1 == "episode_1" and e2 == "episode_2" and e3 == "episode_3"
        assert "episode_1" in kg._graph and "episode_2" in kg._graph
        assert "episode_3" in kg._graph

        # missing start → empty
        assert kg.query_episode_path(99) == []
        assert kg.query_failure_cause(99) == []
        # invalid direction → empty
        assert kg.query_episode_path(1, direction="sideways") == []

        # bad dep_type → KeyError
        try:
            kg.add_dependency_edge(1, 2, "bogus")
            raise AssertionError("expected KeyError for bad dep_type")
        except KeyError:
            pass
        # missing episode → KeyError
        try:
            kg.add_dependency_edge(1, 99, "data")
            raise AssertionError("expected KeyError for missing episode")
        except KeyError:
            pass

        # edges: 1→2 causal, 2→3 causal, 1→3 data
        kg.add_dependency_edge(1, 2, "causal")
        kg.add_dependency_edge(2, 3, "causal")
        kg.add_dependency_edge(1, 3, "data")

        # backward path from 3 → [1, 2, 3]
        path = kg.query_episode_path(3, direction="backward")
        assert [p["step_id"] for p in path] == [1, 2, 3], path
        # forward path from 1 → [1, 2, 3]
        path = kg.query_episode_path(1, direction="forward")
        assert [p["step_id"] for p in path] == [1, 2, 3], path
        # backward path from 2 → [1, 2]
        path = kg.query_episode_path(2, direction="backward")
        assert [p["step_id"] for p in path] == [1, 2], path

        # failure cause for episode 3 (failed): transitive causal walk
        # 3 ← 2 (causal), 2 ← 1 (causal) → [1, 2]
        causes = kg.query_failure_cause(3)
        assert [c["step_id"] for c in causes] == [1, 2], causes
        # failure cause for episode 2 (failed): 2 ← 1 (causal) → [1]
        causes = kg.query_failure_cause(2)
        assert [c["step_id"] for c in causes] == [1], causes
        # episode 1 succeeded → []
        assert kg.query_failure_cause(1) == []

        # persistence round-trip: save + reload, episodes survive
        kg.save()
        kg2 = ProjectKnowledgeGraph(tmp)
        assert "episode_2" in kg2._graph
        causes = kg2.query_failure_cause(3)
        assert [c["step_id"] for c in causes] == [1, 2], causes

        print("all episode-graph checks passed")
