"""知识图谱 (KG) HTTP 端点.

暴露 ProjectKnowledgeGraph 的核心能力: stats / graph / query / mermaid / search.
让前端能消费图数据, 渲染力导向图或树状可视化.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from huginn.security.auth import require_api_key
from huginn.server_core import get_context

router = APIRouter(tags=["knowledge_graph"], dependencies=[Depends(require_api_key)])

logger = logging.getLogger(__name__)


class GraphQueryRequest(BaseModel):
    seed: str
    depth: int = 1
    top_k: int = 10


def _get_kg():
    """从 context 拿 KG 实例, 没有就返回 None."""
    kg = getattr(get_context(), "kg", None)
    if kg is None:
        # autoloop 没跑时, context.kg 可能没初始化, 尝试从 workspace 加载
        try:
            from huginn.kg.graph import ProjectKnowledgeGraph
            cfg = get_context().config
            kg = ProjectKnowledgeGraph(root=cfg.workspace)
        except Exception:
            logger.debug("KG 不可用", exc_info=True)
            return None
    return kg


@router.get("/kg/stats")
async def kg_stats() -> dict[str, Any]:
    """KG 统计: 节点数, 边数, 节点类型分布."""
    kg = _get_kg()
    if kg is None:
        return {"available": False}
    try:
        return {"available": True, **kg.stats()}
    except Exception as e:
        return {"available": False, "error": str(e)}


@router.get("/kg/graph")
async def kg_graph(
    max_nodes: int = Query(200, ge=1, le=2000, description="最多返回的节点数, 防止图太大"),
) -> dict[str, Any]:
    """返回完整图数据 (nodes + edges), 供前端渲染力导向图.

    节点超 max_nodes 时按连接度截断, 只保留核心节点.
    """
    kg = _get_kg()
    if kg is None:
        return {"available": False, "nodes": [], "edges": []}
    try:
        import networkx as nx
        g = kg._graph
        # 截断: 按度排序只取 top-N
        if g.number_of_nodes() > max_nodes:
            nodes_by_degree = sorted(g.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]
            keep = {n for n, _ in nodes_by_degree}
            g = g.subgraph(keep).copy()

        nodes = []
        for nid, data in g.nodes(data=True):
            nodes.append({
                "id": nid,
                "label": data.get("label", nid),
                "type": data.get("type", "Unknown"),
                "confidence": data.get("confidence", 0.5),
                "mentions": data.get("mentions", 1),
                "source": data.get("source", ""),
                "created_at": data.get("created_at", ""),
            })

        edges = []
        for src, dst, data in g.edges(data=True):
            edges.append({
                "source": src,
                "target": dst,
                "relation": data.get("relation", "RELATED_TO"),
                "confidence": data.get("confidence", 0.5),
                "mentions": data.get("mentions", 1),
            })

        return {
            "available": True,
            "nodes": nodes,
            "edges": edges,
            "total_nodes": kg._graph.number_of_nodes(),
            "total_edges": kg._graph.number_of_edges(),
            "truncated": kg._graph.number_of_nodes() > max_nodes,
        }
    except Exception as e:
        logger.error("kg_graph failed", exc_info=True)
        return {"available": False, "error": str(e), "nodes": [], "edges": []}


@router.post("/kg/query")
async def kg_query(req: GraphQueryRequest) -> dict[str, Any]:
    """社区感知图查询: 输入 seed 文本, 返回相关子图 (nodes + edges).

    走 GraphQuery.community_aware_query: 先提取实体, 再 BFS 邻域扩展,
    最后用社区检测裁剪到同社区.
    """
    kg = _get_kg()
    if kg is None:
        return {"available": False, "nodes": [], "edges": []}
    try:
        result = kg.query(req.seed, depth=req.depth, top_k=req.top_k)
        return {"available": True, "seed": req.seed, **result}
    except Exception as e:
        logger.error("kg_query failed", exc_info=True)
        return {"available": False, "error": str(e), "nodes": [], "edges": []}


@router.get("/kg/search")
async def kg_search(
    q: str = Query(..., min_length=1, description="搜索关键词"),
    top_k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """节点搜索: 按 label 子串匹配, 返回相关节点列表."""
    kg = _get_kg()
    if kg is None:
        return {"available": False, "nodes": []}
    try:
        from huginn.kg.query import GraphQuery
        gq = GraphQuery(kg._graph)
        nodes = gq.find_nodes(q, top_k=top_k)
        return {"available": True, "query": q, "nodes": nodes}
    except Exception as e:
        return {"available": False, "error": str(e), "nodes": []}


@router.get("/kg/mermaid")
async def kg_mermaid(
    max_nodes: int = Query(40, ge=1, le=200, description="Mermaid 图最多节点数"),
) -> dict[str, Any]:
    """导出 Mermaid 流程图文本, 前端可直接渲染."""
    kg = _get_kg()
    if kg is None:
        return {"available": False, "mermaid": ""}
    try:
        mermaid_text = kg.to_mermaid(max_nodes=max_nodes)
        return {"available": True, "mermaid": mermaid_text}
    except Exception as e:
        return {"available": False, "error": str(e), "mermaid": ""}
