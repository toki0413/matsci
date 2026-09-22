"""JEV 驱动的工具子集扩展 — 只补"规则路由器覆盖不到的未知域".

``task_tool_router.compute_effective_subset`` 在 keyword 无命中时只给 CORE 基础子集.
本模块兜底用 JEV 的并行 Noul 问"候选工具对当前任务是否相关", 按校准概率+置信度
宽松纳入额外工具. 关键性质:

  - 仅在 ``jev_enabled("jev_tool_router")`` 且隐私允许外发时被调用 (见 _enabled).
  - 纯 advisory: 结果只**增加**候选可见面, 不改执行/栅栏; 失败/无 key/超时 → 空.
  - 候选范围 = available 中不在常驻 CORE 且未被规则命中的工具 (控制一次请求规模).
  - 命中断言: ``p >= threshold 且 (confidence is None 或 >= min_confidence)``.
"""

from __future__ import annotations

import logging
from typing import Any

from huginn.runtime.jev._enabled import jev_enabled
from huginn.runtime.jev.client import JevClient

logger = logging.getLogger(__name__)

# 工具名 → 一句可判"相关性"的描述. 命中才问; 缺描述的候选跳过 (保持语义锚点).
_DESC = {
    "analysis_tool": "对数据做统计分析 / 可视化 / 拟合",
    "literature_tool": "检索 / 归纳文献与论文",
    "web_search_tool": "联网搜索公开信息",
    "diagnose_tool": "对任务或步骤失败做诊断",
    "knowledge_tool": "检索领域知识库 / 常识",
}


def _tool_description(name: str) -> str | None:
    """尽力取候选工具的一句话描述; 取不到返回 None (该候选跳过)."""
    if name in _DESC:
        return _DESC[name]
    try:
        from huginn.tools.registry import ToolRegistry

        tool = ToolRegistry.get(name)
        desc = getattr(tool, "description", "") if tool is not None else ""
        return str(desc).strip() or None
    except Exception:  # noqa: BLE001 — 注册表不可用按无描述处理
        return None


def jev_expand_subset(
    task_message: str,
    available: list[str],
    *,
    client: JevClient | None = None,
    threshold: float = 0.6,
    min_confidence: float = 0.5,
) -> tuple[list[str], dict[str, Any]]:
    """对未知任务, 返回 JEV 建议额外纳入的工具子集.

    Returns:
        (候选工具名列表, meta). JEV 不可用/失败/全不命中 → (empty, {"source": ...}).
    """
    empty = ([], {"source": "jev:advisory", "eval": False})
    if not task_message or not available:
        return empty
    if not jev_enabled("jev_tool_router"):
        return empty

    c = client or JevClient()
    if not c.available:
        return empty

    # 候选 = available 中不在常驻 CORE、且规则未覆盖的 (见并集里其它分支).
    from huginn.runtime.task_tool_router import CORE_TOOL_NAMES

    candidates = [
        name for name in available if name not in set(CORE_TOOL_NAMES)
    ]
    instructions: dict[str, str] = {}
    for name in candidates:
        desc = _tool_description(name)
        if desc:
            instructions[name] = f"工具 '{name}' 对当前任务是否相关? 相关则约为 1. 工具用途: {desc}"

    if not instructions:
        return empty

    state = {"task": str(task_message)[:800]}
    results = c.noul_batch(state, instructions)
    if not results:
        return empty

    picked: list[str] = []
    for name in candidates:
        entry = results.get(name)
        if not entry:
            continue
        p = entry.get("p", 0.0)
        conf = entry.get("confidence")
        if p >= threshold and (conf is None or conf >= min_confidence):
            picked.append(name)

    logger.debug("jev: expanded %, chose %d extra tools", len(candidates), len(picked))
    return (picked, {"source": "jev:advisory", "eval": True, "picked": picked})
