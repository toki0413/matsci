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

    # ── Math concept layer join (双层 KG) ──

    def attach_math_concept_graph(self, concept: str, depth: int = 1) -> dict[str, Any]:
        """把 MathConceptGraph 里 concept 的邻域 merge 进当前 ProjectKnowledgeGraph.

        双层 KG 设计: ProjectKnowledgeGraph 存项目作用域实体 (材料/工具/方法),
        MathConceptGraph 存全局数学概念依赖. 用户在项目里提到某个数学概念
        (e.g. "用 Sobolev 空间理论分析 PDE 弱解") 时, 调此方法把该概念的祖先
        链拉进项目 KG, 让项目内的实体能跨连到数学概念层.

        返回 merge 的节点/边数: {"added_nodes": N, "added_edges": M}.
        Ponytail: 不持久化 MathConcept 子图 (它是全局只读的), 只在项目 KG 里
        加引向 MathConcept 的 "anchor" 节点. 真要持久化时手动 save().
        """
        try:
            mcg = get_math_concept_graph()
        except Exception:
            return {"added_nodes": 0, "added_edges": 0, "error": "mcg_unavailable"}

        nb = mcg.query_concept_neighborhood(concept, depth=depth)
        if not nb.get("found"):
            return {"added_nodes": 0, "added_edges": 0, "error": "concept_not_found"}

        added_nodes = 0
        added_edges = 0
        with self._lock:
            # 把 concept + ancestors + descendants + duals 全部作为 MathConcept
            # 节点加进项目 KG
            all_concepts = {nb["concept"]} | set(nb["ancestors"]) | \
                           set(nb["descendants"]) | set(nb["duals"])
            for c in all_concepts:
                eid = node_id(c, "MathConcept")
                if eid not in self._graph:
                    self._graph.add_node(
                        eid, label=c, type="MathConcept",
                        source="math_concept_graph", confidence=1.0,
                        created_at=datetime.now().isoformat(), mentions=1,
                    )
                    added_nodes += 1

            # 从 MathConceptGraph 复制依赖边 (depends_on / generalizes / dual_to)
            for path in nb.get("paths_to_ancestors", []) + nb.get("paths_to_descendants", []):
                for i in range(len(path) - 1):
                    src_eid = node_id(path[i], "MathConcept")
                    dst_eid = node_id(path[i + 1], "MathConcept")
                    # 查 MathConceptGraph 里这条边的 relation
                    edge_data = mcg._graph.get_edge_data(path[i], path[i + 1])
                    rel = edge_data.get("relation", "depends_on") if edge_data else "depends_on"
                    if not self._graph.has_edge(src_eid, dst_eid):
                        self._graph.add_edge(
                            src_eid, dst_eid, relation=rel,
                            source="math_concept_graph", confidence=1.0,
                            created_at=datetime.now().isoformat(),
                        )
                        added_edges += 1

            # duals 加双向 dual_to 边
            for dual in nb["duals"]:
                src_eid = node_id(nb["concept"], "MathConcept")
                dst_eid = node_id(dual, "MathConcept")
                if not self._graph.has_edge(src_eid, dst_eid):
                    self._graph.add_edge(
                        src_eid, dst_eid, relation="dual_to",
                        source="math_concept_graph", confidence=1.0,
                        created_at=datetime.now().isoformat(),
                    )
                    added_edges += 1

        return {"added_nodes": added_nodes, "added_edges": added_edges}

    # ── P13 CrossDomain transfer history ──

    def add_transfer_edge(self, transfer: Any) -> str:
        """把 CrossDomain transfer 写入 KG.

        加 CROSS_DOMAIN_TRANSFER 节点 + 3 条边:
        - original_problem node → transfer node: cross_domain_analogy
        - transfer node → target_domain node: transfers_to
        - transfer node ↔ math_concept node: structurally_isomorphic

        返回 transfer_node_id. 字段缺失时 getattr 兜底, 不 raise.
        ponytail: 节点 id 用 uuid 防冲突 — 同一 source→target 可能多次 transfer.
        """
        from huginn.kg.entities import EntityType, Relation

        original = getattr(transfer, "original_problem", "") or ""
        target = getattr(transfer, "target_domain", "") or ""
        math_concept = getattr(transfer, "math_concept", "") or ""
        lca = getattr(transfer, "lca_concept", "") or ""
        shared_math = getattr(transfer, "shared_math", []) or []
        reframed = getattr(transfer, "reframed_problem", "") or ""
        confidence = float(getattr(transfer, "confidence", 0.0) or 0.0)

        transfer_id = f"transfer:{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{id(transfer) & 0xFFFFFF:x}"
        now = datetime.now().isoformat()

        with self._lock:
            # transfer 节点本体
            self._graph.add_node(
                transfer_id,
                type=EntityType.CROSS_DOMAIN_TRANSFER,
                label=f"{original[:40]} → {target}",
                original_problem=original,
                target_domain=target,
                math_concept=math_concept,
                lca_concept=lca,
                shared_math=list(shared_math),
                reframed_problem=reframed,
                confidence=confidence,
                status="proposed",  # 后续 hypothesis 验证时更新
                created_at=now,
                last_seen=now,
                mentions=1,
            )

            # source 节点 (Topic) + cross_domain_analogy 边
            if original:
                src_eid = node_id(original[:80], EntityType.TOPIC)
                if src_eid not in self._graph:
                    self._graph.add_node(
                        src_eid, label=original[:80], type=EntityType.TOPIC,
                        source="cross_domain", confidence=0.5,
                        created_at=now, last_seen=now, mentions=1,
                    )
                self._graph.add_edge(
                    src_eid, transfer_id, relation=Relation.CROSS_DOMAIN_ANALOGY,
                    source="cross_domain", confidence=confidence,
                    created_at=now, last_seen=now, mentions=1,
                )

            # target 节点 (Topic) + transfers_to 边
            if target:
                dst_eid = node_id(target, EntityType.TOPIC)
                if dst_eid not in self._graph:
                    self._graph.add_node(
                        dst_eid, label=target, type=EntityType.TOPIC,
                        source="cross_domain", confidence=0.5,
                        created_at=now, last_seen=now, mentions=1,
                    )
                self._graph.add_edge(
                    transfer_id, dst_eid, relation=Relation.TRANSFERS_TO,
                    source="cross_domain", confidence=confidence,
                    created_at=now, last_seen=now, mentions=1,
                )

            # math_concept 节点 (MathConcept) + 双向 structurally_isomorphic 边
            if math_concept:
                mc_eid = node_id(math_concept, EntityType.MATH_CONCEPT)
                if mc_eid not in self._graph:
                    self._graph.add_node(
                        mc_eid, label=math_concept, type=EntityType.MATH_CONCEPT,
                        source="cross_domain", confidence=1.0,
                        created_at=now, last_seen=now, mentions=1,
                    )
                self._graph.add_edge(
                    transfer_id, mc_eid, relation=Relation.STRUCTURALLY_ISOMORPHIC,
                    source="cross_domain", confidence=confidence,
                    created_at=now, last_seen=now, mentions=1,
                )
                self._graph.add_edge(
                    mc_eid, transfer_id, relation=Relation.STRUCTURALLY_ISOMORPHIC,
                    source="cross_domain", confidence=confidence,
                    created_at=now, last_seen=now, mentions=1,
                )

        return transfer_id

    def query_transfer_history(
        self,
        source_domain: str | None = None,
        target_domain: str | None = None,
        limit: int = 3,
    ) -> list[dict]:
        """查 transfer 历史. 返回 list[dict], 每条含 transfer 节点全部属性.

        source_domain: 子串匹配 original_problem (None 跳过).
        target_domain: 严格匹配 target_domain 字段 (None 跳过).
        limit: 最多返回条数, 按创建时间倒序.
        """
        from huginn.kg.entities import EntityType

        results: list[dict] = []
        with self._lock:
            for nid, data in self._graph.nodes(data=True):
                if data.get("type") != EntityType.CROSS_DOMAIN_TRANSFER:
                    continue
                if target_domain and data.get("target_domain", "") != target_domain:
                    continue
                if source_domain and source_domain not in data.get("original_problem", ""):
                    continue
                row = dict(data)
                row["node_id"] = nid
                results.append(row)
        # 按 created_at 倒序, 取前 limit 条
        results.sort(key=lambda d: d.get("created_at", ""), reverse=True)
        return results[:limit]


# ── Math concept dependency graph ──────────────────────────────────────
# Ponytail 路径分析:
# 1. 真做 `lake graph` 解析 Mathlib import 树 — 需要本地 mathlib 仓库 (几 GB),
#    用户多数没装, 强依赖外部状态. 不可行.
# 2. 真做"自动从 Lean 文件 import 解析" — 需要 Lean 工具链 + 全量 mathlib 源码.
#    不可行.
# 3. 懒方案: 内置核心数学概念依赖表 (~40 条), 覆盖拓扑/代数/分析/概率/动力系统
#    核心概念. `lake graph` 升级路径明确标注, 用户装了 mathlib 后可 seed 扩展.

# (parent_concept, relation, child_concept) — child depends_on / generalizes / dual_to parent
# depends_on: child 需要 parent 才能定义 (Hausdorff 需要 Topological Space)
# generalizes: child 是 parent 的特例 (Metric → Topological, Metric generalizes Topological)
#              箭头方向: child --generalizes--> parent
# dual_to: 双向, 标记对偶关系 (Max ↔ Min)
_MATH_CONCEPT_DEPS: list[tuple[str, str, str]] = [
    # ── 拓扑 ──
    ("topological_space", "depends_on", "set_theory"),
    ("hausdorff_space", "generalizes", "topological_space"),
    ("compact_space", "generalizes", "hausdorff_space"),
    ("metric_space", "generalizes", "topological_space"),
    ("banach_space", "generalizes", "metric_space"),
    ("hilbert_space", "generalizes", "banach_space"),
    ("manifold", "depends_on", "hausdorff_space"),
    ("riemannian_manifold", "generalizes", "manifold"),
    ("symplectic_manifold", "generalizes", "manifold"),
    ("lie_group", "depends_on", "manifold"),
    ("covering_space", "depends_on", "topological_space"),
    ("fundamental_group", "depends_on", "topological_space"),
    ("homology_group", "depends_on", "topological_space"),
    ("cohomology_group", "depends_on", "homology_group"),
    # ── 代数 ──
    ("group", "depends_on", "set_theory"),
    ("ring", "generalizes", "group"),
    ("field", "generalizes", "ring"),
    ("vector_space", "depends_on", "field"),
    ("module", "generalizes", "vector_space"),
    ("algebra", "generalizes", "vector_space"),
    ("lie_algebra", "depends_on", "vector_space"),
    ("tensor", "depends_on", "vector_space"),
    ("representation", "depends_on", "lie_algebra"),
    ("category", "depends_on", "set_theory"),
    ("functor", "depends_on", "category"),
    ("natural_transformation", "depends_on", "functor"),
    # ── 分析 ──
    ("measure", "depends_on", "set_theory"),
    ("lebesgue_measure", "generalizes", "measure"),
    ("probability_measure", "generalizes", "measure"),
    ("banach_space", "depends_on", "vector_space"),
    ("hilbert_space", "depends_on", "banach_space"),
    ("operator_algebra", "depends_on", "hilbert_space"),
    ("c_star_algebra", "generalizes", "banach_algebra"),
    ("banach_algebra", "generalizes", "algebra"),
    ("sobolev_space", "generalizes", "banach_space"),
    ("distribution", "depends_on", "sobolev_space"),
    ("fourier_transform", "depends_on", "hilbert_space"),
    # ── 动力系统 ──
    ("dynamical_system", "depends_on", "metric_space"),
    ("flow", "generalizes", "dynamical_system"),
    ("ergodic_theory", "depends_on", "measure"),
    ("markov_chain", "depends_on", "probability_measure"),
    ("stochastic_process", "depends_on", "probability_measure"),
    # ── 微分方程 ──
    ("ode", "depends_on", "banach_space"),
    ("pde", "depends_on", "sobolev_space"),
    ("variational_formulation", "depends_on", "hilbert_space"),
    ("weak_formulation", "generalizes", "variational_formulation"),
    ("fem", "depends_on", "weak_formulation"),
    # ── 几何 ──
    ("riemannian_geometry", "depends_on", "riemannian_manifold"),
    ("differential_geometry", "depends_on", "manifold"),
    ("algebraic_geometry", "depends_on", "ring"),
    # ── 数论 ──
    ("number_theory", "depends_on", "ring"),
    ("galois_theory", "depends_on", "field"),
    # ── 对偶关系 (关键) ──
    ("maximize_stability", "dual_to", "minimize_instability_path"),
    ("primal_problem", "dual_to", "dual_problem"),
    ("contravariant", "dual_to", "covariant"),
    ("position_space", "dual_to", "momentum_space"),
]


class MathConceptGraph:
    """Global math concept dependency graph (process-wide singleton).

    Ponytail: 与 ProjectKnowledgeGraph 分离 — 项目 KG 是 workspace-scoped,
    数学概念图是全局知识, 不应每个项目重建. networkx DiGraph 内存中
    一次构建, 进程内共享.

    升级路径:
    - 用户装了 mathlib + lake 后, 调 `seed_from_mathlib(lake_graph_output)`
      把 Mathlib import 树 merge 进来, 覆盖更细粒度的依赖.
    - 当前 _MATH_CONCEPT_DEPS 是核心种子 (~50 条), 覆盖主流数学分支.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._lock = __import__("threading").RLock()
        self._build_from_seed()

    def _build_from_seed(self) -> None:
        # 种子表三元组 (child, rel, parent): child 是 parent 的特例 / child 依赖 parent
        # 箭头方向: child --rel--> parent (出边指向 ancestor)
        for child, rel, parent in _MATH_CONCEPT_DEPS:
            self._graph.add_node(parent, type="MathConcept", label=parent)
            self._graph.add_node(child, type="MathConcept", label=child)
            self._graph.add_edge(child, parent, relation=rel)

    def query_concept_neighborhood(
        self, concept: str, depth: int = 1
    ) -> dict[str, Any]:
        """查 concept 的邻域 (前驱+后继, depth 跳).

        返回 {concept, ancestors, descendants, duals, paths}:
        - ancestors: concept 依赖/特例化的更一般概念 (depth 跳内)
        - descendants: 依赖 concept 的更具体概念
        - duals: 对偶概念 (dual_to 关系)
        - paths: concept 到各 ancestor 的路径列表
        """
        concept = concept.strip().lower()
        with self._lock:
            if concept not in self._graph:
                return {"concept": concept, "found": False}

            # ancestors: 沿出边走 (child → parent)
            ancestors = set()
            paths: list[list[str]] = []
            self._bfs_collect(concept, direction="successors", depth=depth,
                              collected=ancestors, paths=paths)

            # descendants: 沿入边走 (parent ← child)
            descendants = set()
            desc_paths: list[list[str]] = []
            self._bfs_collect(concept, direction="predecessors", depth=depth,
                              collected=descendants, paths=desc_paths)

            # duals: dual_to 关系 (双向)
            duals = set()
            for _, nbr, ed in self._graph.edges(concept, data=True):
                if ed.get("relation") == "dual_to":
                    duals.add(nbr)
            for nbr, _, ed in self._graph.in_edges(concept, data=True):
                if ed.get("relation") == "dual_to":
                    duals.add(nbr)

            return {
                "concept": concept,
                "found": True,
                "ancestors": sorted(ancestors),
                "descendants": sorted(descendants),
                "duals": sorted(duals),
                "paths_to_ancestors": paths,
                "paths_to_descendants": desc_paths,
            }

    def _bfs_collect(
        self,
        start: str,
        direction: str,
        depth: int,
        collected: set[str],
        paths: list[list[str]],
    ) -> None:
        """BFS 走 depth 跳, 收集节点 + 记录路径."""
        from collections import deque
        queue: deque[tuple[str, int, list[str]]] = deque([(start, 0, [start])])
        visited: set[str] = {start}
        while queue:
            node, d, path = queue.popleft()
            if d >= depth:
                continue
            neighbors = (
                self._graph.successors(node)
                if direction == "successors"
                else self._graph.predecessors(node)
            )
            for nb in neighbors:
                if nb in visited:
                    continue
                visited.add(nb)
                collected.add(nb)
                new_path = path + [nb]
                paths.append(new_path)
                queue.append((nb, d + 1, new_path))

    def find_common_ancestor(self, c1: str, c2: str, max_depth: int = 4) -> str | None:
        """找两个概念的最近公共祖先 (LCA). 用于跨域类比的结构同构验证.

        Ponytail: 双 BFS 到 max_depth, 取祖先交集的第一个. 不上 Tarjan LCA
        (需要预处理), 当前图小 (~50 节点), O(n²) 可接受.
        """
        c1 = c1.strip().lower()
        c2 = c2.strip().lower()
        with self._lock:
            if c1 not in self._graph or c2 not in self._graph:
                return None
            anc1: set[str] = set()
            self._bfs_collect(c1, "successors", max_depth, anc1, [])
            anc2: set[str] = set()
            self._bfs_collect(c2, "successors", max_depth, anc2, [])
            common = anc1 & anc2
            if not common:
                return None
            # 取交集里 depth 最浅的 — 简化: 任取一个, 不严格 LCA
            # ponytail: 严格 LCA 需要记 depth, 当前近似够用
            return next(iter(common))

    def stats(self) -> dict[str, int]:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }


# 模块级单例: 进程内共享, 双检锁懒加载
_math_concept_graph_singleton: MathConceptGraph | None = None
_math_concept_graph_lock = __import__("threading").Lock()


def get_math_concept_graph() -> MathConceptGraph:
    """拿进程级单例 MathConceptGraph. 线程安全懒加载."""
    global _math_concept_graph_singleton
    if _math_concept_graph_singleton is None:
        with _math_concept_graph_lock:
            if _math_concept_graph_singleton is None:
                _math_concept_graph_singleton = MathConceptGraph()
    return _math_concept_graph_singleton


def seed_from_mathlib(lake_graph_output: str) -> int:
    """升级路径: 从 `lake graph` 输出 merge import 依赖进 MathConceptGraph.

    lake_graph_output 是 `lake graph` 命令的文本输出 (DOT 或类似格式).
    解析失败返回 0, 成功返回新增边数. 当前留作 hook, 不实现解析
    (避免依赖 lake 二进制; 用户真要用时再补 parser).
    """
    # ponytail: 留 stub, 真要接 lake graph 时在这里加 parser.
    # 当前 _MATH_CONCEPT_DEPS 种子表已覆盖核心场景.
    return 0


if __name__ == "__main__":
    import tempfile

    # MathConceptGraph self-check
    mcg = MathConceptGraph()
    assert mcg.stats()["nodes"] >= 40, f"种子表应 ≥40 节点, got {mcg.stats()}"
    assert mcg.stats()["edges"] >= 40, f"种子表应 ≥40 边, got {mcg.stats()}"

    # query_concept_neighborhood
    nb = mcg.query_concept_neighborhood("hilbert_space", depth=2)
    assert nb["found"], "hilbert_space 应在图中"
    assert "banach_space" in nb["ancestors"], "hilbert → banach 应在 ancestors"
    assert "metric_space" in nb["ancestors"], "depth=2 应能到 metric_space"
    assert nb["duals"] == [], "hilbert_space 无 dual"

    # duals
    nb_primal = mcg.query_concept_neighborhood("primal_problem", depth=1)
    assert "dual_problem" in nb_primal["duals"], "primal ↔ dual 应互为对偶"

    # find_common_ancestor
    lca = mcg.find_common_ancestor("hilbert_space", "banach_space")
    # hilbert → banach → metric → topological; banach → metric → topological
    # LCA 应是 banach_space 或 metric_space (近似 LCA)
    assert lca in {"banach_space", "metric_space", "topological_space"}, \
        f"LCA 应在祖先链上, got {lca}"

    # 不存在的概念
    assert not mcg.query_concept_neighborhood("nonexistent_concept")["found"]

    # 单例
    assert get_math_concept_graph() is get_math_concept_graph(), "单例应一致"

    # 原有 ProjectKnowledgeGraph self-check
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

        # ── P13 CrossDomain transfer selfcheck ──
        from huginn.metacog.cross_domain_pipeline import TransferHypothesis

        kg_t = ProjectKnowledgeGraph(tempfile.mkdtemp(prefix="kg_transfer_"))

        # 场景 1: mock transfer 写入 → 查询能拿到
        t1 = TransferHypothesis(
            original_problem="predict Fe magnetic transition temperature",
            target_domain="ferromagnet",
            shared_math=["landau_phi4"],
            math_concept="lie_group",
            lca_concept="group",
            reframed_problem="reframed via Landau theory",
            confidence=0.7,
            trace=["s1", "s2", "s3", "s4", "s5"],
        )
        tid1 = kg_t.add_transfer_edge(t1)
        assert tid1.startswith("transfer:"), tid1

        hist = kg_t.query_transfer_history(limit=5)
        assert len(hist) == 1, f"expected 1, got {len(hist)}"
        assert hist[0]["original_problem"] == t1.original_problem
        assert hist[0]["target_domain"] == "ferromagnet"
        assert hist[0]["math_concept"] == "lie_group"
        assert hist[0]["status"] == "proposed"
        assert hist[0]["confidence"] == 0.7
        # 3 条边都建了
        edges_out = [
            d for _, _, d in kg_t._graph.out_edges(tid1, data=True)
        ]
        edge_rels = {d.get("relation") for d in edges_out}
        assert "transfers_to" in edge_rels, edge_rels
        assert "structurally_isomorphic" in edge_rels, edge_rels
        in_rels = {
            d.get("relation") for _, _, d in kg_t._graph.in_edges(tid1, data=True)
        }
        assert "cross_domain_analogy" in in_rels, in_rels
        print("1. add_transfer_edge + query round-trip OK")

        # 场景 2: 3 条不同 transfer 写入 → limit=3 拿到 3 条
        import time as _t
        for i in range(3):
            _ti = TransferHypothesis(
                original_problem=f"problem_{i} Fe based",
                target_domain=f"domain_{i}",
                shared_math=[],
                math_concept=f"concept_{i}",
                lca_concept="",
                reframed_problem="",
                confidence=0.3 + i * 0.1,
                trace=[],
            )
            kg_t.add_transfer_edge(_ti)
            _t.sleep(0.01)  # 错开 created_at 让排序稳定
        hist3 = kg_t.query_transfer_history(limit=3)
        assert len(hist3) == 3, f"limit=3 应返 3, got {len(hist3)}"
        # 总数应是 4 (1 + 3)
        hist_all = kg_t.query_transfer_history(limit=100)
        assert len(hist_all) == 4, f"total 应 4, got {len(hist_all)}"
        print("2. multi-transfer + limit OK")

        # 场景 3: source/target 过滤
        fe_hist = kg_t.query_transfer_history(
            source_domain="Fe", target_domain="ferromagnet", limit=5
        )
        assert len(fe_hist) == 1, fe_hist
        assert fe_hist[0]["original_problem"] == t1.original_problem
        domain_2_hist = kg_t.query_transfer_history(target_domain="domain_2", limit=5)
        assert len(domain_2_hist) == 1
        assert domain_2_hist[0]["target_domain"] == "domain_2"
        none_hist = kg_t.query_transfer_history(target_domain="nonexistent", limit=5)
        assert none_hist == []
        print("3. source/target filter OK")

        print("all transfer-graph checks passed")
