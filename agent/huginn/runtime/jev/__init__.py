"""JEV (System One decision model) 接入层.

把 TypeSafe JEV 作为 Huginn 的**外部语义判断**来源, 补"规则门禁管不到的
中间地带". 本包默认全关 (advisory 不拦截), 且严格受隐私档约束:

  - 对外调用只在 ``jev_enabled`` 且隐私允许外发时发生 (见 ``_enabled.py``).
  - 判断结果显式标 ``source=jev:advisory``, 只影响工具可见面/软分数,
    永不进 claim_grounding 证据门或 harness 统计门 (可证伪性红线).
  - 任一调用失败都 fail-open 返回 None/空, 绝不因 JEV 不可用而拖垮 agent.

对外主要入口:
  - :class:`JevClient` — Noul/Choice/Score 的容错客户端 (httpx, 零硬依赖可选).
  - :func:`jev_tool_router.jev_expand_subset` — 未知域工具子集的宽松补充.
  - ``security.gate.jev_adapter`` — 把 JEV 判断归一化进统一门禁链.
"""

from __future__ import annotations

from huginn.runtime.jev._enabled import (
    jev_egress_allowed,
    jev_enabled,
)
from huginn.runtime.jev.client import JevClient
from huginn.runtime.jev.tool_router import jev_expand_subset

__all__ = [
    "JevClient",
    "jev_enabled",
    "jev_egress_allowed",
    "jev_expand_subset",
]