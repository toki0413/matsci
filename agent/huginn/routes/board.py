"""任务看板 / 专家团画布 endpoints.

把会话里已发生的子任务派发 (subagent_tool) 重建成一张前端可直接渲染的图:
节点 (子任务) + 边 (依赖) + 分层 (可并行层) + 专家泳道.

读 side 投影, 不写任何东西 —— 数据源是 ``SessionEventLog`` 的原始事件,
重建逻辑在 ``huginn.agents.board.build_board``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from huginn.agents.board import build_board
from huginn.routes.threads import _check_thread_owner

logger = logging.getLogger(__name__)

router = APIRouter(tags=["board"])


def _expert_roster() -> list[dict[str, Any]]:
    """可用专家清单 (subagent spec). 取不到就返回空 —— 画布照常渲染已派发的."""
    try:
        from huginn.agents.subagent import SubagentDispatch

        return SubagentDispatch().list_specs()
    except Exception:  # noqa: BLE001 — 专家注册表不可用不该拖垮看板
        logger.debug("board: 读取专家清单失败", exc_info=True)
        return []


@router.get("/threads/{thread_id}/board")
async def get_thread_board(thread_id: str, request: Request) -> dict[str, Any]:
    """任务看板 / 专家团画布.

    响应: ``{thread_id, exists, nodes, edges, layers, critical_path, width,
    experts, stats, next_seq}``.
      - ``nodes`` 每个子任务一个节点, ``status ∈ {running, done, failed}``
        (派发后还没结果就是 running), ``layer`` 是拓扑层号.
      - ``layers`` 同层节点可并行; ``width`` 是最大并行度 (最大反链).
      - ``critical_path`` 是 wall-clock 下限路径.
      - ``experts`` 是专家泳道, 含还没被派发的专家 (来自可用 spec 清单).
    """
    err = _check_thread_owner(thread_id, request)
    if err:
        return err
    return build_board(thread_id, expert_roster=_expert_roster())


@router.get("/board/experts")
async def list_experts() -> dict[str, Any]:
    """列出可用的专家 (subagent) 类型及各自的工具/预算配置."""
    roster = _expert_roster()
    return {"success": True, "experts": roster, "count": len(roster)}
