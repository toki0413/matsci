"""元认知层 — 多 agent 搜索的显式控制回路.

六个模块对应 prompt 设计的启发点:

- context_isolation:   信息隔离层, 控制哪些上下文下发给哪些探索 agent
- method_registry:     方法族注册表, 按思想本质聚类, 监控收敛度
- equivalence_auditor: 等价性审计 agent, 检测"换名归约"伪进展
- failure_modes:       材料科学领域失败模式清单, 给对抗 agent 用
- block_registry:      阻塞-新机制重启协议, 防止换名重启死路线
- depth_search:        深度搜索机制, 对抗 LLM 快速收敛偏差
- topology_lens:       高阶网络视角的结构判据 (四族/闭包/Hodge签名/拓扑许可/层粘合)

接入点见:
- autoloop/red_team.py        → 读取 failure_modes
- autoloop/hypothesis_loop.py → refine_failed 接入 block_registry + 连通分量监控
- autoloop/engine.py          → PhaseGateHook / _hypothesize 接入 + 最小努力下限
- metacog/topology_lens.py    → 给 red_team / equivalence_auditor 调用的结构透镜
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def recall_context(category: str, query: str = "", top_k: int = 3) -> list[dict]:
    """通用 context recall — 从 long-term memory 按类别捞外置化上下文 (G19).

    支持的 category: autoloop_summary / benchmark_run_summary / skill_invocation /
    knowledge_seed / stable_principles / hypothesis / failure / subgoal 等.

    Args:
        category: memory.longterm 的 category 字段
        query: 可选, 用于 FTS 过滤
        top_k: 最多返回多少条

    返回 list[dict], 每条至少含 {"content": str, "category": str, ...}
    失败返回空 list — 调用方不应因 memory 不可用而崩溃.
    """
    try:
        from huginn.memory.longterm import LongTermMemory
        # ponytail: 每次新建实例, 不缓存. 当前 metacog 没有持有 memory_manager 的入口,
        # 临时实例化开销可接受 (SQLite 即开即关). 升级路径: 让 engine 注入 memory_manager, 复用连接池.
        mem = LongTermMemory()
        # semantic=False: 确定性 + 不依赖 vector_store 配置, audit/recall 一致
        results = mem.retrieve(
            query=query, category=category, top_k=top_k, semantic=False
        )
        return results if results else []
    except Exception as e:
        logger.debug(f"recall_context({category}) failed: {e}")
        return []


def recall_audit_context(category: str, query: str = "", limit: int = 20) -> list[dict]:
    """向后兼容 wrapper — 老的 audit 入口, 委托给通用 recall_context.

    签名不变 (limit 参数保留), 内部把 limit 映射到 top_k.
    """
    return recall_context(category=category, query=query, top_k=limit)


from huginn.metacog.failure_modes import FailureMode, FailureModeRegistry, DEFAULT_REGISTRY
from huginn.metacog.context_isolation import ContextBundle, IsolationPolicy, isolate
from huginn.metacog.method_registry import MethodFamily, MethodRegistry
from huginn.metacog.equivalence_auditor import EquivalenceAuditor, EquivalenceVerdict
from huginn.metacog.block_registry import BlockedRoute, BlockRegistry, ReopenAttempt
from huginn.metacog.depth_search import (
    MinEffortFloor,
    DynamicComponentFloor,
    EffortStatus,
    PrematureConvergenceDetector,
)
from huginn.metacog.completion_auditor import (
    CompletionChecklist,
    CompletionAuditor,
    parse_unexplored_declaration,
)
from huginn.metacog.topology_lens import (
    Family,
    FamilyVerdict,
    ClosureCheck,
    HodgeSignature,
    TopologyPermit,
    GluingObstruction,
    TopologyDiagnosis,
    classify_system,
    needs_downward_closure,
    hodge_signature,
    topology_permits,
    gluing_obstruction,
    diagnose,
)
from huginn.metacog.signal_hub import SignalHub

__all__ = [
    "recall_context",
    "recall_audit_context",
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
    "MinEffortFloor",
    "DynamicComponentFloor",
    "EffortStatus",
    "PrematureConvergenceDetector",
    "CompletionChecklist",
    "CompletionAuditor",
    "parse_unexplored_declaration",
    # 拓扑透镜 (高阶网络视角)
    "Family",
    "FamilyVerdict",
    "ClosureCheck",
    "HodgeSignature",
    "TopologyPermit",
    "GluingObstruction",
    "TopologyDiagnosis",
    "classify_system",
    "needs_downward_closure",
    "hodge_signature",
    "topology_permits",
    "gluing_obstruction",
    "diagnose",
    # SignalHub (G17)
    "SignalHub",
]
