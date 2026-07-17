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


# === G43: recall 策略学习 — v4 预留接口第 2 项 ===
# 策略表: (phase, metacog_state) → (category, top_k, query_template)
# 由 S7 self-modification 学习更新, 下次 recall 用学到的策略.
# ponytail: 策略学习是 S7 闭环的自然延伸, 不需要新组件. 升级路径是
# 让策略表走 RAG + embedding 相似度匹配, 当前精确字符串匹配够用.

import json as _json
import os as _os
from pathlib import Path as _Path

# 默认策略 — 覆盖 7 phase × 关键 metacog_state. S7 可在此基础上扩展.
# v6 G52: 加 structure_relation_type 字段 (可选), 让策略表能按结构关系过滤 recall.
_DEFAULT_RECALL_STRATEGIES: list[dict] = [
    {"phase": "perceive", "metacog_state": "S1_DISCOVER", "category": "autoloop_summary", "top_k": 2, "query_template": "", "reason": "perceive 阶段需要历史任务上下文", "structure_relation_type": ""},
    {"phase": "hypothesize", "metacog_state": "S2_HYPOTHESIZE", "category": "hypothesis", "top_k": 3, "query_template": "", "reason": "hypothesize 阶段需要历史假设做对照", "structure_relation_type": ""},
    {"phase": "plan", "metacog_state": "S3_PLAN", "category": "stable_principles", "top_k": 5, "query_template": "", "reason": "plan 阶段必须遵守 stable_principles", "structure_relation_type": ""},
    {"phase": "execute", "metacog_state": "S4_ACT", "category": "skill_invocation", "top_k": 3, "query_template": "", "reason": "execute 阶段需要工具调用历史", "structure_relation_type": ""},
    {"phase": "validate", "metacog_state": "S5_VERIFY", "category": "failure", "top_k": 3, "query_template": "", "reason": "validate 阶段需要历史失败模式", "structure_relation_type": ""},
    {"phase": "learn", "metacog_state": "S6_RESOLVE", "category": "knowledge_seed", "top_k": 5, "query_template": "", "reason": "learn 阶段需要知识种子做归纳", "structure_relation_type": ""},
    {"phase": "report", "metacog_state": "S6_RESOLVE", "category": "benchmark_run_summary", "top_k": 2, "query_template": "", "reason": "report 阶段需要过往 benchmark 结果对照", "structure_relation_type": ""},
    # v6 G52: 结构关系专用策略 — validate 阶段遇到结构违反时, recall 同结构的历史失败
    {"phase": "validate", "metacog_state": "S5_VERIFY", "category": "failure", "top_k": 5, "query_template": "structure:{structure_relation_type}", "reason": "结构违反时 recall 同类结构历史失败", "structure_relation_type": "catalytic_geometry"},
    {"phase": "validate", "metacog_state": "S5_VERIFY", "category": "failure", "top_k": 5, "query_template": "structure:{structure_relation_type}", "reason": "结构违反时 recall 同类结构历史失败", "structure_relation_type": "interface_binding"},
    {"phase": "validate", "metacog_state": "S5_VERIFY", "category": "failure", "top_k": 5, "query_template": "structure:{structure_relation_type}", "reason": "结构违反时 recall 同类结构历史失败", "structure_relation_type": "percolation_topology"},
    {"phase": "validate", "metacog_state": "S5_VERIFY", "category": "failure", "top_k": 5, "query_template": "structure:{structure_relation_type}", "reason": "结构违反时 recall 同类结构历史失败", "structure_relation_type": "band_symmetry"},
    {"phase": "validate", "metacog_state": "S5_VERIFY", "category": "failure", "top_k": 5, "query_template": "structure:{structure_relation_type}", "reason": "结构违反时 recall 同类结构历史失败", "structure_relation_type": "defect_chemistry"},
]


def _strategy_file(workspace: str | _Path | None = None) -> _Path:
    """策略表路径: workspace/.huginn/recall_strategy.jsonl
    workspace=None 时用 HUGINN_CACHE_DIR 或 ~/.huginn.
    ponytail: 跟 stable_principles 同位置, 跨任务复用.
    """
    if workspace is not None:
        base = _Path(workspace)
    else:
        cache = _os.environ.get("HUGINN_CACHE_DIR")
        base = _Path(cache) if cache else _Path.home() / ".huginn"
    base.mkdir(parents=True, exist_ok=True)
    return base / "recall_strategy.jsonl"


def load_recall_strategy(workspace: str | _Path | None = None) -> list[dict]:
    """加载策略表. 文件不存在时写默认策略, 然后返回.
    ponytail: 文件不存在不抛异常, 写默认 — 第一次调用自动初始化.
    """
    path = _strategy_file(workspace)
    if not path.exists():
        # 首次调用 — 写默认策略
        path.write_text(
            "\n".join(_json.dumps(s, ensure_ascii=False) for s in _DEFAULT_RECALL_STRATEGIES) + "\n",
            encoding="utf-8",
        )
        return list(_DEFAULT_RECALL_STRATEGIES)
    try:
        strategies: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                strategies.append(_json.loads(line))
        return strategies if strategies else list(_DEFAULT_RECALL_STRATEGIES)
    except Exception:
        logger.debug("recall_strategy.jsonl corrupted, using defaults")
        return list(_DEFAULT_RECALL_STRATEGIES)


def match_recall_strategy(
    phase: str,
    metacog_state: str = "",
    strategies: list[dict] | None = None,
    workspace: str | _Path | None = None,
    structure_relation_type: str = "",
) -> dict | None:
    """匹配策略: 结构关系专用 > 精确 (phase, metacog_state) > 只 phase.

    v6 G52: structure_relation_type 非空时, 优先匹配同结构关系的策略
    (用于 validate 阶段结构违反时 recall 同类结构历史失败).

    metacog_state 可空 (未传时只按 phase 匹配).
    """
    if strategies is None:
        strategies = load_recall_strategy(workspace)
    # v6 G52: 结构关系专用策略优先
    if structure_relation_type:
        for s in strategies:
            if (
                s.get("phase") == phase
                and s.get("structure_relation_type") == structure_relation_type
            ):
                return s
    # 精确匹配 (phase, metacog_state)
    if metacog_state:
        for s in strategies:
            if s.get("phase") == phase and s.get("metacog_state") == metacog_state:
                # 跳过结构关系专用策略 (没传 structure_relation_type 时不该命中)
                if not s.get("structure_relation_type"):
                    return s
    # 退到只匹配 phase
    for s in strategies:
        if s.get("phase") == phase and not s.get("metacog_state"):
            return s
    # 最后退到 phase 任意 metacog_state
    for s in strategies:
        if s.get("phase") == phase and not s.get("structure_relation_type"):
            return s
    return None


def recall_context_with_strategy(
    phase: str,
    metacog_state: str = "",
    query: str = "",
    workspace: str | _Path | None = None,
    structure_relation_type: str = "",
) -> list[dict]:
    """按学到的策略 recall — 匹配 (phase, metacog_state, structure_relation_type).

    无策略命中时返回空 list (调用方自行决定降级路径).
    ponytail: 这是 recall_context 的策略版本, 不替代原 recall_context.
    v6 G52: structure_relation_type 非空时优先匹配结构关系专用策略.
    """
    strategy = match_recall_strategy(
        phase, metacog_state, workspace=workspace,
        structure_relation_type=structure_relation_type,
    )
    if strategy is None:
        return []
    # query_template 可扩展为模板替换, 当前直接用 query
    return recall_context(
        category=strategy.get("category", ""),
        query=query or strategy.get("query_template", ""),
        top_k=int(strategy.get("top_k", 3)),
    )


def update_recall_strategy(
    entry: dict,
    workspace: str | _Path | None = None,
) -> bool:
    """S7 self-modification 调用: 更新/新增策略条目.

    entry 必须含 phase + category + top_k. metacog_state 可选.
    已有 (phase, metacog_state) 匹配的条目会被替换, 否则追加.
    返回 True=更新成功, False=entry 不合法.
    ponytail: 这是 S7 闭环的写入端, 不引入新组件.
    """
    if not entry.get("phase") or not entry.get("category"):
        return False
    if not isinstance(entry.get("top_k"), int) or entry["top_k"] <= 0:
        return False

    path = _strategy_file(workspace)
    strategies = load_recall_strategy(workspace)
    # 替换匹配的旧条目, 否则追加
    replaced = False
    key_phase = entry["phase"]
    key_state = entry.get("metacog_state", "")
    for i, s in enumerate(strategies):
        if s.get("phase") == key_phase and s.get("metacog_state", "") == key_state:
            strategies[i] = entry
            replaced = True
            break
    if not replaced:
        strategies.append(entry)
    # 原子写 — 先写 tmp 再 rename, 防止写一半被读到
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "\n".join(_json.dumps(s, ensure_ascii=False) for s in strategies) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return True


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
    # G43: recall 策略学习
    "load_recall_strategy",
    "match_recall_strategy",
    "recall_context_with_strategy",
    "update_recall_strategy",
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
