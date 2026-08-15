"""H5-c: 首轮锚定晋升 — 从持久事件源推导 promotion 状态.

对齐 dsh-super-injector SPEC §6.1: 晋升状态从持久事件源 (agent.session.events)
推导, 而非内存标志, 保证 resume/reload 不丢状态. 工具执行失败也算已持久化,
下一步照常晋升; 首次响应未调用工具则不晋升.

实现: 复用 huginn.evolution.logger.ExecutionLogger 持久化的 tool_calls.jsonl
(每条 ToolCallRecord 含 session_id 字段). ExecutionLogger 构造时 _load_existing
从磁盘装载历史, 因此其 in-memory _tool_calls 本身就是持久事件源 — 跨进程 resume
也能读到上次会话的工具调用记录.

注意: 本模块刻意不缓存 ExecutionLogger 实例, 而是每次重新构建 — 这样能拾取
其他进程/先前会话写入的新记录, 晋升判定始终反映磁盘最新状态.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 首轮锚定: 未晋升时只暴露的平台无关核心工具 (对齐 injector SPEC §6.3 —
# 首轮保留集 = 平台无关核心工具 1-2 个, 首个 tool/call 后恢复完整目录).
CORE_BOOTSTRAP_TOOLS = frozenset({"file_read_tool", "bash_tool"})


def _logger() -> Any | None:
    """懒加载 ExecutionLogger (等同 engine._get_evolution 模式). 失败返回 None."""
    try:
        from huginn.evolution.logger import ExecutionLogger

        return ExecutionLogger()
    except Exception:
        logger.debug("promotion: ExecutionLogger build failed", exc_info=True)
        return None


def has_durable_tool_call(session_id: str | None) -> bool:
    """是否已发生持久化工具调用 (晋升)? 从持久事件源推导.

    - session_id 为空 → 无会话上下文, 恒 True (不启用首轮锚定).
    - 日志读到任一该 session 的 tool 记录 → True (失败也算, 对齐 SPEC).
    - 日志不可用 / 无记录 → False (未晋升, 走首轮锚定).
    """
    if not session_id:
        return True
    lg = _logger()
    if lg is None:
        return True
    try:
        calls = getattr(lg, "_tool_calls", None)
        if calls is None:
            return True
        return any(getattr(r, "session_id", None) == session_id for r in calls)
    except Exception:
        logger.debug("promotion: has_durable_tool_call failed", exc_info=True)
        return True


def effective_tool_whitelist(
    phase: str | None,
    session_id: str | None = None,
    whitelist: list[str] | None = None,
) -> list[str] | None:
    """晋升前把白名单折叠到核心工具; 晋升后返回完整白名单.

    - whitelist 为空/None → 原样返回 (保持"无白名单不限制"语义).
    - 已晋升 (或 session_id 为空) → 返回完整白名单.
    - 未晋升 → 返回 whitelist ∩ 核心工具 (可能为空 → bootstrap 期该 phase
      无额外放开的工具, 保持现有空列表语义).
    """
    if not whitelist:
        return whitelist
    if session_id is None or has_durable_tool_call(session_id):
        return whitelist
    return [t for t in whitelist if t in CORE_BOOTSTRAP_TOOLS]