"""Unified query view over KG + Meta-Trace simplicial complex.

v14 Task 16: 把 KG vertex 跟 Meta-Trace vertex 拉到同一个查询接口下. ID 前缀
(kg:entity:xxx / trace:xxx) 隔离两套来源, KG triple (s, p, o) 作为 2-simplex
进入 complex.

依赖注入: cross_task_store / kg 都通过构造器传入, 任一为 None 只查另一边.
import 时 try/except, 不强依赖 CrossTaskStore (Task 14) 或 KG 模块.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 软依赖: import 失败不阻塞模块加载, 运行时 store/kg 通过依赖注入.
# ponytail: 升级路径是真正注入实例后, type hint 自动从 Optional[Any] 收紧.
try:  # Task 14 的 CrossTaskStore, 可能尚未实现
    from huginn.metacog.cross_task_store import CrossTaskStore  # noqa: F401
    _HAS_CROSS_TASK_STORE = True
except Exception:
    _HAS_CROSS_TASK_STORE = False

try:  # 项目已有 KG
    from huginn.kg.graph import ProjectKnowledgeGraph  # noqa: F401
    _HAS_KG = True
except Exception:
    _HAS_KG = False


@dataclass
class Vertex:
    """Unified complex 上的 vertex, 可来自 KG 或 Meta-Trace.

    vertex_id 前缀区分来源: 'kg:entity:xxx' / 'trace:xxx'.
    """
    vertex_id: str
    source: str  # 'kg' 或 'trace'
    content: str
    domain: str = "unknown"
    task_id: str = "unknown"
    darwin_score: float = 0.0


# KG vertex 默认 darwin — KG 没有 darwin 概念, 给中性 prior.
# ponytail: 真正升级路径是让 KG node 带 confidence → darwin 映射, 当前 0.5 中性够用.
_KG_DEFAULT_DARWIN = 0.5


class UnifiedComplexView:
    """统一查询 KG vertex + Meta-Trace vertex.

    ID 空间前缀隔离: KG 用 'kg:entity:xxx', Meta-Trace 用 'trace:{task_id}:iter_{N}:{role}'
    (Task 1 _make_simplex_id 已定义). KG triple (s, p, o) 作为 2-simplex 进入 complex.

    ponytail: 简单 union 查询, 不真正合并 complex. 升级路径: 真正的 pushout / colimit
    (TopoNetX SimplicialComplex + coproduct glue 共享 vertex). 当前 O(n) scan + sort
    够用, 上限是 store + kg 双边各 top_k, 不上同构/边界矩阵.
    """

    def __init__(self, cross_task_store=None, kg=None):
        self.store = cross_task_store
        self.kg = kg

    def query(
        self,
        domain: str,
        task_id: Optional[str] = None,
        keyword: Optional[str] = None,
        top_k: int = 10,
    ) -> list[Vertex]:
        """统一查 KG + Meta-Trace, 返回混合 vertex list, 按 darwin 降序截 top_k.

        - store 为 None 跳过 trace 侧; kg 为 None 或 keyword 为空跳过 KG 侧.
        - 双边都 None 返回空 list.
        - KG vertex 默认 darwin=0.5, trace vertex 用 entry 自带 darwin_score.

        ponytail: 双边各自取 top_k 再合并截断 — 不是全局 top_k 的精确解, 但避免
        双边 fetch 全集再排的开销. 升级路径: store / kg 都支持 cursor + score
        返回后做真全局 top-k.
        """
        if top_k <= 0:
            return []
        vertices: list[Vertex] = []

        # ── Meta-Trace 侧 ──
        if self.store is not None:
            try:
                entries = self.store.query(
                    domain=domain, task_id=task_id,
                    keyword=keyword, top_k=top_k,
                )
            except TypeError:
                # 老签名兼容: 只接 domain 位置参数
                entries = self.store.query(domain)
            for e in entries or []:
                if not isinstance(e, dict):
                    continue
                e_domain = e.get("domain", "unknown")
                # 跨 domain 隔离: store 应已过滤, 这里再保险 (general 永远放行)
                if e_domain != domain and e_domain != "general":
                    continue
                sid = e.get("simplex_id") or ""
                if not sid.startswith("trace:"):
                    # 兜底: 没有 simplex_id 的旧 entry 拼一个
                    sid = f"trace:{e.get('task_id', 'unknown')}:iter_{e.get('iteration', 0)}:{e.get('role', 'unknown')}"
                content = f"{e.get('attempted', '')} -> {e.get('found', '')}"
                vertices.append(Vertex(
                    vertex_id=sid,
                    source="trace",
                    content=content,
                    domain=e_domain,
                    task_id=e.get("task_id", "unknown"),
                    darwin_score=float(e.get("darwin_score", 0.0) or 0.0),
                ))

        # ── KG 侧 ──
        # keyword 为空时 KG 没法查 (KG query 接口都需 seed), 跳过
        if self.kg is not None and keyword:
            for ent in self._kg_query_entities(keyword, top_k):
                name = ent.get("name") or ent.get("label") or ""
                if not name:
                    continue
                vertices.append(Vertex(
                    vertex_id=f"kg:entity:{name}",
                    source="kg",
                    content=name,
                    domain=domain,
                    task_id=task_id or "unknown",
                    darwin_score=_KG_DEFAULT_DARWIN,
                ))

        # 按 darwin 降序截断
        vertices.sort(key=lambda v: v.darwin_score, reverse=True)
        return vertices[:top_k]

    def query_triples(
        self,
        domain: str,
        keyword: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """KG triple (s, p, o) 作为 2-simplex (triangle) 进入 complex.

        每个 triple 包装成:
        {
            "simplex_id": "kg:triple:{hash}",
            "simplex_type": "triangle",
            "vertices": ["kg:entity:{s}", "kg:entity:{o}", "kg:predicate:{p}"],
            "domain": domain,
        }

        ponytail: 用 Python 内置 hash() — 进程内稳定, 跨进程会因 PYTHONHASHSEED
        变化. 当前 simplex_id 只在单次 RCBench run 内用, 不持久化跨 run, 够用.
        升级路径: 换 hashlib.md5 (跨进程稳定) 当 ID 需要写盘时.
        """
        if self.kg is None:
            return []
        triples = self._kg_query_triples(keyword=keyword, top_k=top_k)
        result: list[dict] = []
        for t in triples:
            s = t.get("subject") or t.get("source") or ""
            p = t.get("predicate") or t.get("relation") or ""
            o = t.get("object") or t.get("target") or ""
            if not (s and p and o):
                continue
            sid = f"kg:triple:{hash((s, p, o)) & 0xFFFFFFFF:x}"
            result.append({
                "simplex_id": sid,
                "simplex_type": "triangle",
                "vertices": [
                    f"kg:entity:{s}",
                    f"kg:entity:{o}",
                    f"kg:predicate:{p}",
                ],
                "domain": domain,
            })
        return result[:top_k]

    # ── KG duck-typed adapters ──

    def _kg_query_entities(self, keyword: str, top_k: int) -> list[dict]:
        """调 KG 的 query_entities (新接口) 或 fallback 到 query() 返回的 nodes.

        ponytail: 用 getattr 检测, 不强绑 KG 接口形状. 老 ProjectKnowledgeGraph
        的 query() 返回 {'nodes': [...], 'edges': [...]}, nodes 是
        [{'id': ..., 'label': ...}, ...]; 新接口 (mock / 后续适配器) 返回
        [{'name': ...}, ...]. 这里两个都兼容.
        """
        kg = self.kg
        if kg is None:
            return []
        # 新接口优先 (mock + 未来 adapter)
        if hasattr(kg, "query_entities"):
            try:
                result = kg.query_entities(keyword, top_k=top_k)
                if isinstance(result, list):
                    return result
            except Exception:
                pass
        # 老 ProjectKnowledgeGraph 接口 fallback
        if hasattr(kg, "query"):
            try:
                sub = kg.query(keyword, top_k=top_k)
                if isinstance(sub, dict):
                    nodes = sub.get("nodes", [])
                    return [
                        {
                            "name": n.get("label", n.get("id", "")),
                            "type": n.get("type", "entity"),
                        }
                        for n in nodes
                    ]
            except Exception:
                pass
        return []

    def _kg_query_triples(
        self,
        keyword: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """调 KG 的 query_triples 接口拿 (s, p, o) 三元组.

        ponytail: 没有 query_triples 方法时直接返 [], 不做 edges → triple 转换.
        原 ProjectKnowledgeGraph.query() 返回的 edges dict 里 'source' 字段被
        edge provenance 属性覆盖 (add_relation 默认 source='auto'), 拿不到真实
        src node id. 升级路径: 给 ProjectKnowledgeGraph 加 query_triples adapter,
        或修 GraphQuery edges 的字段冲突 (source/target → src_id/dst_id).
        """
        kg = self.kg
        if kg is None or not hasattr(kg, "query_triples"):
            return []
        try:
            result = kg.query_triples(
                subject=subject, predicate=predicate,
                object=object, top_k=top_k,
            )
            return result if isinstance(result, list) else []
        except Exception:
            return []


# ── self-check ──

class _MockCrossTaskStore:
    """Mock store for self-check — 只返 astronomy domain 的 1 条 trace entry."""

    def query(self, domain, task_id=None, keyword=None, top_k=10):
        if domain != "astronomy":
            return []
        return [
            {
                "simplex_id": "trace:Astronomy_000:iter_0:rcb_exec",
                "attempted": "superradiance",
                "found": "...",
                "domain": "astronomy",
                "task_id": "Astronomy_000",
                "darwin_score": 0.8,
            },
        ]


class _MockKG:
    """Mock KG for self-check — keyword 含 'superradiance' 时返 1 个 entity, 1 个 triple."""

    def query_entities(self, keyword, top_k=10):
        if not keyword or "superradiance" not in keyword:
            return []
        return [{"name": "Superradiance", "type": "phenomenon"}]

    def query_triples(self, subject=None, predicate=None, object=None, top_k=10):
        return [{"subject": "BlackHole", "predicate": "exhibits", "object": "Superradiance"}]


def _run_self_check() -> None:
    """v14 Task 16 self-check.

    构造 mock CrossTaskStore + mock KG, 断言:
      1. query(domain=astronomy, keyword=superradiance) 返回 ≥2 vertex (1 trace + 1 kg)
      2. query(domain=astronomy) 无 keyword 也至少返回 1 trace vertex
      3. query(domain=material) 返回 0 (mock 只返 astronomy)
      4. query_triples(domain=astronomy, keyword=superradiance) 返回 1 个 triangle dict
      5. 跨 domain 隔离: query(domain=material) 不含 astronomy 内容
      6. source 字段: trace vertex source='trace', kg vertex source='kg'
      7. vertex_id 前缀: trace 含 'trace:', kg 含 'kg:entity:'

    ponytail: 全用 assert, 不引框架. mock store / kg 最小够覆盖以上 case.
    """
    view = UnifiedComplexView(
        cross_task_store=_MockCrossTaskStore(),
        kg=_MockKG(),
    )

    # case 1: 双边都查, 至少 1 trace + 1 kg
    v1 = view.query(domain="astronomy", keyword="superradiance", top_k=10)
    assert len(v1) >= 2, f"case 1 fail: expected ≥2 vertices, got {len(v1)}"
    sources_1 = {v.source for v in v1}
    assert sources_1 == {"trace", "kg"}, f"case 1 fail: sources={sources_1}"

    # case 2: 无 keyword, 只查 trace 侧, 至少 1 vertex
    v2 = view.query(domain="astronomy", top_k=10)
    assert len(v2) >= 1, f"case 2 fail: expected ≥1 vertex, got {len(v2)}"
    assert all(v.source == "trace" for v in v2), "case 2 fail: should be all trace"

    # case 3: 不存在的 domain, mock store 返 [], kg 无 keyword → 0
    v3 = view.query(domain="material", top_k=10)
    assert len(v3) == 0, f"case 3 fail: expected 0 vertices, got {len(v3)}"

    # case 4: query_triples 返 1 个 triangle
    t4 = view.query_triples(domain="astronomy", keyword="superradiance", top_k=10)
    assert len(t4) == 1, f"case 4 fail: expected 1 triple, got {len(t4)}"
    tri = t4[0]
    assert tri["simplex_type"] == "triangle", "case 4 fail: simplex_type"
    assert tri["domain"] == "astronomy", "case 4 fail: domain"
    assert tri["simplex_id"].startswith("kg:triple:"), "case 4 fail: simplex_id prefix"
    assert len(tri["vertices"]) == 3, "case 4 fail: triangle has 3 vertices"
    # triangle vertices: [kg:entity:s, kg:entity:o, kg:predicate:p]
    assert "kg:entity:BlackHole" in tri["vertices"], "case 4 fail: subject vertex"
    assert "kg:entity:Superradiance" in tri["vertices"], "case 4 fail: object vertex"
    assert "kg:predicate:exhibits" in tri["vertices"], "case 4 fail: predicate vertex"

    # case 5: 跨 domain 隔离 — query(material) 不含 astronomy 内容
    v5 = view.query(domain="material", keyword="superradiance", top_k=10)
    # mock store 不返 material, kg 仍会返 Superradiance (kg 不分 domain)
    # 但即便如此, vertex.domain 字段应是 material (调用方传入的 domain), 不是 astronomy
    for v in v5:
        assert v.domain == "material", f"case 5 fail: vertex.domain={v.domain}"
        # content 不能含 astronomy 的 trace 内容
        assert "Astronomy_000" not in v.content, "case 5 fail: leak astronomy trace"

    # case 6: source 字段对得上
    v6 = view.query(domain="astronomy", keyword="superradiance", top_k=10)
    trace_vs = [v for v in v6 if v.source == "trace"]
    kg_vs = [v for v in v6 if v.source == "kg"]
    assert len(trace_vs) >= 1, "case 6 fail: no trace vertex"
    assert len(kg_vs) >= 1, "case 6 fail: no kg vertex"

    # case 7: vertex_id 前缀
    for v in trace_vs:
        assert v.vertex_id.startswith("trace:"), f"case 7 fail: trace id={v.vertex_id}"
    for v in kg_vs:
        assert v.vertex_id.startswith("kg:entity:"), f"case 7 fail: kg id={v.vertex_id}"

    # 额外: 两边都 None → 空 list (不崩)
    empty_view = UnifiedComplexView(cross_task_store=None, kg=None)
    assert empty_view.query(domain="astronomy", keyword="x") == [], "empty view should return []"
    assert empty_view.query_triples(domain="astronomy", keyword="x") == [], "empty view triples []"

    # 额外: darwin 降序 — trace 0.8 应在 kg 0.5 之前
    v8 = view.query(domain="astronomy", keyword="superradiance", top_k=10)
    if len(v8) >= 2:
        # trace (0.8) > kg (0.5)
        assert v8[0].darwin_score >= v8[1].darwin_score, "case 8 fail: not sorted desc"

    print("[CHECK v14 Task 16] UnifiedComplexView OK "
          f"(case1={len(v1)}, case2={len(v2)}, case3={len(v3)}, "
          f"case4={len(t4)}, case5={len(v5)}, case6={len(trace_vs)}/{len(kg_vs)})")
    print("v14 Task 16 self-check PASSED")


if __name__ == "__main__":
    _run_self_check()
