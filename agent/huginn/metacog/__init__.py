"""元认知层 — 多 agent 搜索的显式控制回路.

五个模块对应 prompt 设计的五个启发点:

- context_isolation: 信息隔离层, 控制哪些上下文下发给哪些探索 agent
- method_registry:   方法族注册表, 按思想本质聚类, 监控收敛度
- equivalence_auditor: 等价性审计 agent, 检测"换名归约"伪进展
- failure_modes:     材料科学领域失败模式清单, 给对抗 agent 用
- block_registry:    阻塞-新机制重启协议, 防止换名重启死路线

接入点见:
- autoloop/red_team.py        → 读取 failure_modes
- autoloop/hypothesis_loop.py → refine_failed 接入 block_registry
- autoloop/engine.py          → PhaseGateHook / _hypothesize 接入
"""

from __future__ import annotations

from huginn.metacog.failure_modes import FailureMode, FailureModeRegistry, DEFAULT_REGISTRY
from huginn.metacog.context_isolation import ContextBundle, IsolationPolicy, isolate
from huginn.metacog.method_registry import MethodFamily, MethodRegistry
from huginn.metacog.equivalence_auditor import EquivalenceAuditor, EquivalenceVerdict
from huginn.metacog.block_registry import BlockedRoute, BlockRegistry, ReopenAttempt

__all__ = [
    "FailureMode",
    "FailureModeRegistry",
    "DEFAULT_REGISTRY",
    "ContextBundle",
    "IsolationPolicy",
    "isolate",
    "MethodFamily",
    "MethodRegistry",
    "EquivalenceAuditor",
    "EquivalenceVerdict",
    "BlockedRoute",
    "BlockRegistry",
    "ReopenAttempt",
]
