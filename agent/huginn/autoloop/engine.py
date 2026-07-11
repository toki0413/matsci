"""Autoloop Engine — the main autonomous loop for Huginn.

Ties together exploration, coder, workflow, benchmark, and report
into a single closed-loop ecosystem:

    Perceive → Hypothesize → Plan → Execute → Validate → Learn → Report

Usage:
    engine = AutoloopEngine(workspace=Path("."))
    asyncio.run(engine.run(objective="Optimize C-S-H defect kinetics"))
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from huginn.autoloop.budget import IterationBudget, ProgressiveBudget
from huginn.autoloop.goal_scheduler import Goal, GoalScheduler
from huginn.autoloop.phase_gate import (
    PhaseGate,
    PhaseGateHook,
    get_shared_phase_gate_state,
)
from huginn.bench.runner import BenchmarkRunner
from huginn.coder.loop import CoderRunner
from huginn.config import get_settings
from huginn.exploration.orchestrator import ExplorationOrchestrator
from huginn.exploration.strategies import ParetoPruningStrategy
from huginn.interaction.progress import ProgressTracker, get_progress_tracker
from huginn.kg.builder import ProjectKnowledgeGraph
from huginn.llm import get_model
from huginn.memory.manager import MemoryManager
from huginn.api.event import EventType, WorkflowStageEvent
from huginn.tools.report_tool import ReportTool
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.templates import standard_dft_workflow


# Autoloop 7-phase pipeline — single source of truth for phase names.
# ponytail: constants, not an enum — engine phases are imperative control
# flow labels, not a declarative state machine like ResearchPhase.
# If phases diverge enough to need transitions/validation, promote to Enum.
AUTOLOOP_PHASES = (
    "perceive",
    "hypothesize",
    "plan",
    "execute",
    "validate",
    "learn",
    "report",
)

# 7 阶段 → persona 分派表. None 表示该阶段不走 LLM persona 注入
# (比如 Execute 直接调 workflow, 不需要 persona 影响输出).
# Hypothesize 用 default, 真正的 persona 在 _hypothesize 里按研究类型动态选.
_PHASE_PERSONAS: dict[str, str | None] = {
    "perceive": "default",
    "hypothesize": None,  # 动态选 dft_expert / md_expert, 见 _hypothesize
    "plan": "default",
    "execute": None,  # 直接调 workflow / coder, 不走 LLM persona
    "validate": "reviewer",  # 关键: 校验阶段用 reviewer persona 做批判性审视
    "learn": "default",
    "report": "tutor",  # 教学风格输出
}
assert set(_PHASE_PERSONAS.keys()) == set(AUTOLOOP_PHASES), (
    "Phase persona keys must match AUTOLOOP_PHASES"
)

# result_data 里的 key -> 文献检索时用的性质名. _literature_comparison 遍历这个表.
_LIT_PROPERTY_MAP: dict[str, str] = {
    "energy": "total energy",
    "band_gap": "band gap",
    "volume": "volume",
    "bulk_modulus": "bulk modulus",
    "magnetization": "magnetization",
    "lattice_a": "lattice constant a",
    "lattice_b": "lattice constant b",
    "lattice_c": "lattice constant c",
}


def _extract_tests_passed(validation: Any) -> bool:
    """从 validation 结果里抽 tests_passed 布尔, 给 validate→learn 门用.

    validation 形状不固定 (dict / str / None), 抽不出明确失败就默认 True,
    避免门控把现有 happy path 误阻断. 只有明确说 fail / passed=False 才拦.
    """
    if isinstance(validation, dict):
        for key in ("tests_passed", "passed", "success", "ok"):
            if key in validation:
                return bool(validation[key])
        return True
    if isinstance(validation, str):
        low = validation.lower()
        if "fail" in low:
            return False
        return True
    # None 或其它: 没有明确失败信号, 放行
    return True


@dataclass
class LoopPhase:
    """A single phase in the autonomous loop."""

    name: str
    status: str = "pending"  # pending | running | completed | failed
    start_time: float | None = None
    end_time: float | None = None
    result: Any = None
    error: str | None = None


@dataclass
class AutoloopResult:
    """Result of a full autonomous loop iteration."""

    run_id: str
    objective: str
    phases: list[LoopPhase]
    success: bool
    report_path: str | None = None
    total_time_seconds: float = 0.0
    trajectory_path: str | None = None
    goal_achieved: bool | None = None
    goal_judgment: dict[str, Any] | None = None
    # 落盘的 provenance JSONL, run 结束后可回放整条 tool chain
    provenance_path: str | None = None


class AutoloopEngine:
    """Main autonomous loop engine.

    Orchestrates perception, hypothesis generation, planning, execution,
    validation, learning, and reporting into a single cohesive loop.
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        goal_scheduler: GoalScheduler | None = None,
        verification_model: Any = None,
        memory_manager: MemoryManager | None = None,
    ):
        self.workspace = Path(workspace or ".").resolve()
        self.settings = get_settings()
        self.model = get_model(self.settings)
        # Moonshine 三槽: verification 用独立 LLM 验证假设, 避免确认偏差.
        # 默认 None 时退回 self.model, 保持向后兼容.
        self.verification_model = verification_model or self.model
        # 共享 MemoryManager: 由 agent/CLI 传入, 避免引擎私有实例和 agent 的
        # memory 隔离. 默认 None 时 new 一个, 保持向后兼容.
        self.memory = memory_manager or MemoryManager()
        self.kg = ProjectKnowledgeGraph(root=self.workspace)
        # 假设图: 跟踪 hypothesis 的 support/refute/derive 关系,
        # refute 时触发 RedTeam 审查 → 修正假设入队, 形成闭环
        from huginn.autoloop.hypothesis_loop import HypothesisGraph
        self.hypothesis_graph = HypothesisGraph()
        self.report_tool = ReportTool()

        # Sub-engines
        self.explorer = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(),
            max_parallel=3,
        )
        self.workflow_engine = WorkflowEngine(
            tool_registry=None,  # Will use default tool registry
        )
        self.coder = CoderRunner()

        self._should_stop = False
        self._iteration = 0
        # 连续验证失败计数: 给 _maybe_clarify 判断是否该问用户
        self._consecutive_failures = 0
        # refine 循环计数: 防止 refute→refine 无限循环
        self._refine_count = 0
        self._max_refines = 8
        # ClarificationManager 懒加载 — autoloop 期间在关键决策点提问用户
        self._clarification_mgr = None
        # Evolution engine 懒加载——只在 _learn 真正用到时初始化
        self._evolution = None
        # PersonaManager 懒加载 — 避免实例化时就扫描 .huginn/personas 目录
        self._persona_manager = None
        # 领域知识库 (first-principles seed docs) 懒加载 — 避免实例化时拉 ChromaDB
        self._kb = None
        # PerceptionLayer 懒加载 — 长生命周期, start() 后后台线程持续积累事件
        self._perception = None
        # Plan store: 持久化 plan 到 plans.json, 跨会话可恢复. 懒加载,
        # 跟 goal_scheduler 一套, 避免实例化时碰磁盘
        self._plan_store = None
        # 进度跟踪: 默认走进程级单例, 跟 WorkflowEngine 共享, 让 /tasks
        # 路由能汇总所有引擎的进度. 测试时可注入独立 tracker 隔离.
        self.progress_tracker: ProgressTracker | None = None
        # 投机执行 hint: on_turn_start 写入, _build_*_prompt 读出注入 LLM
        self._speculator_hint: str = ""
        # 视觉基元: _validate 从 tool 输出提取, _build_*_prompt 注入 LLM.
        # 跨迭代传递 — 上轮 tool 的数值指针下轮假设/计划能用到.
        self._last_visual_context: str = ""
        # JEPA 式预测: plan 阶段 LLM 预测预期结果, validate 阶段对比实际,
        # 预测误差 = surprise = intrinsic motivation 信号.
        # ponytail: 文本空间预测, 不是真正的嵌入空间 JEPA. 但原理一致 —
        # 执行前预测, 执行后对比, 误差驱动探索. 升级路径: 训练真正的编码器+预测器.
        self._current_prediction: str = ""
        self._last_surprise: float = 0.0
        # 上一轮执行结果, 给 _build_plan_prompt 的 pipeline suggest_next 用
        self._last_execution_result: dict | None = None
        # 阶段门 hook: 在 plan→execute / execute→validate / validate→learn
        # 三个转移点评估证据, 不足时阻断并把 feedback 拼进 _speculator_hint
        # 让下轮 prompt 带上"缺什么证据". R3 接入 red-team reviewer_fn:
        # 在 validate→learn 做 adversarial 审查, 有 high 发现则阻断.
        from huginn.autoloop.red_team import RedTeamReviewer
        from huginn.autoloop.phase_gate import MathEvidenceChecker

        self.phase_gate_hook = PhaseGateHook(
            reviewer_fn=RedTeamReviewer(model=self.model),
            math_checker=MathEvidenceChecker(),
        )
        # Goal scheduler: 持久化目标到 $HUGINN_CACHE_DIR/goals.json.
        # engine.run(goal=...) 时每轮 learn 后查 completion, 满足则提前停.
        # None → 懒加载, 避免实例化时就碰磁盘 (测试隔离用).
        self._goal_scheduler = goal_scheduler
        # 侧边对话 channel: 轮空时 drain 待答问题. 默认走进程级单例,
        # 跟 HTTP /side 路由共享. None 时用 get_shared_side_channel() 懒拿.
        self._side_channel = None
        # 侧边对话开关: 测试或不需要侧边对话时关掉, 避免 idle 时碰 LLM.
        self._side_channel_enabled = True
        # 事件总线: 让外部插件能在阶段开始/结束/失败时挂钩.
        # 懒加载, 避免 import 时拉起 StarHandlerRegistry.
        self._event_bus = None
        # 阶段索引: 给 WorkflowStageEvent 用, 从 phase name 推算.
        self._phase_order = list(AUTOLOOP_PHASES)

    def _get_evolution(self):
        """懒加载 EvolutionEngine, 避免实例化时就拉起日志和规则文件。"""
        if self._evolution is None:
            from huginn.evolution.engine import EvolutionEngine
            from huginn.evolution.logger import ExecutionLogger

            self._evolution = EvolutionEngine(logger=ExecutionLogger())
        return self._evolution

    def _get_perception(self):
        """懒加载 PerceptionLayer — 长生命周期, start 后持续监听文件/日志事件.

        一旦创建就在后台持续运行, _perceive() 只取 snapshot 不再 start/stop.
        析构时由 GC 或显式 stop() 回收线程.
        """
        if self._perception is None:
            try:
                from huginn.perception import PerceptionLayer
                self._perception = PerceptionLayer(self.workspace)
                self._perception.start()
            except Exception:
                return None
        return self._perception

    def _get_persona_manager(self):
        """懒加载 PersonaManager, 实例化时才扫描 persona 文件."""
        if self._persona_manager is None:
            from huginn.personas import PersonaManager

            self._persona_manager = PersonaManager(workspace=self.workspace)
        return self._persona_manager

    def _get_kb(self):
        """懒加载领域知识库. ChromaDB 或 seed 文件不可用时返回 None,
        调用方需自行判空."""
        if self._kb is None:
            try:
                from huginn.knowledge.store import get_knowledge_base

                self._kb = get_knowledge_base(str(self.workspace))
            except Exception:
                return None
        return self._kb

    def _get_event_bus(self):
        """懒加载 EventBus. 没注册 handler 时返回 None, 调用方判空跳过."""
        if self._event_bus is not None:
            return self._event_bus
        try:
            from huginn.plugins.event_bus import EventBus

            self._event_bus = EventBus()
        except Exception:
            return None
        return self._event_bus

    async def _dispatch_stage_event(
        self,
        event_type: EventType,
        stage_name: str,
        duration_sec: float = 0.0,
        error: str | None = None,
    ) -> None:
        """向 EventBus 发一个 WorkflowStageEvent. 没总线或没 handler 时静默跳过."""
        bus = self._get_event_bus()
        if bus is None:
            return
        idx = self._phase_order.index(stage_name) + 1 if stage_name in self._phase_order else 0
        event = WorkflowStageEvent(
            type=event_type,
            workflow_name="autoloop",
            stage_name=stage_name,
            stage_index=idx,
            duration_sec=duration_sec,
            error=error,
        )
        try:
            result = await bus.dispatch(event)
            if result.executed == 0 and result.failed == 0:
                logger.debug(
                    "stage event %s.%s had no handlers",
                    event_type.name, stage_name,
                )
        except Exception:
            logger.warning("error in _dispatch_stage_event: bus.dispatch failed", exc_info=True)

    def _build_kb_text(self, query: str) -> str:
        """检索领域知识库, 把命中 chunk 拼成 prompt 上下文块. KB 没装、
        空、查询失败都返回空串, 不影响 loop."""
        if not query:
            return ""
        kb = self._get_kb()
        if kb is None:
            return ""
        try:
            if kb.count() == 0:
                return ""
            chunks = kb.query(query, top_k=5)
            if not chunks:
                return ""
            lines = []
            for i, c in enumerate(chunks, 1):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                if len(text) > 800:
                    text = text[:800] + "…"
                lines.append(f"[{i}] {text}")
            if not lines:
                return ""
            body = "\n".join(lines)
            return (
                "### Domain Knowledge Context\n"
                "The following first-principles reference chunks may ground your "
                "hypothesis and plan. Cite source numbers when relevant.\n"
                f"{body}\n"
                "### End Domain Knowledge Context"
            )
        except Exception:
            return ""

    def _build_kg_text(self, query: str) -> str:
        """检索知识图谱, 把相关实体+关系拼成 prompt 上下文块.
        KG 没建、空、查询失败都返回空串. 这是 KG 读回闭环的关键 —
        _learn 写入的实体, _hypothesize/_plan 要能检索到."""
        if not query:
            return ""
        kg = getattr(self, "kg", None)
        if kg is None:
            return ""
        try:
            result = kg.query(query, depth=1, top_k=8)
            nodes = result.get("nodes") or []
            if not nodes:
                return ""
            lines = []
            for node in nodes[:8]:
                data = node.get("data", node)
                label = data.get("label", node.get("id", ""))
                etype = data.get("type", "")
                conf = data.get("confidence", 0)
                lines.append(f"- [{etype}] {label} (conf={conf:.2f})")
                # 把出边也带上
                for edge in (node.get("edges") or [])[:3]:
                    rel = edge.get("relation", "→")
                    dst = edge.get("dst_label", edge.get("dst", ""))
                    lines.append(f"  {rel} → {dst}")
            if not lines:
                return ""
            body = "\n".join(lines)
            return (
                "### Knowledge Graph Context\n"
                "Previously discovered entities and relations from prior runs:\n"
                f"{body}\n"
                "### End Knowledge Graph Context"
            )
        except Exception:
            return ""

    def _build_memory_text(self, query: str) -> str:
        """检索长期记忆, 把跨会话的教训/发现拼成 prompt 上下文块.
        Memory 之前只写不读 — _learn 写入的迭代记录和失败教训,
        下轮 hypothesize/plan 完全看不到. 这个函数闭合了 memory 读回环.
        查询失败/空结果返回空串, 不影响 prompt."""
        if not query:
            return ""
        mem = getattr(self, "memory", None)
        if mem is None:
            return ""
        try:
            return mem.recall_for_prompt(query, max_entries=3)
        except Exception:
            return ""

    # 上下文预算: 防止 prompt block 累积超过 token 上限.
    # 优先级: context > math > kg > visual > kb > mem > hint > pipeline > composite
    # 低优先级 block 先被裁剪. ponytail: 字符级近似, 不算 token, 够用.
    _PROMPT_BUDGET = 12000  # chars, 约 3K tokens

    def _trim_to_budget(self, blocks: list[tuple[str, str]]) -> str:
        """按优先级拼接 blocks, 超预算时从低优先级开始裁剪."""
        total = sum(len(v) for _, v in blocks)
        if total <= self._PROMPT_BUDGET:
            return "".join(v for _, v in blocks)
        # 从尾部 (低优先级) 开始删
        kept = [v for _, v in blocks]
        for i in range(len(kept) - 1, -1, -1):
            total -= len(kept[i])
            kept[i] = ""
            if total <= self._PROMPT_BUDGET:
                break
        return "".join(kept)

    def _persona_system_prompt(self, persona_name: str | None) -> str:
        """取 persona 的 system prompt. 找不到就返回空串, 不报错."""
        if not persona_name:
            return ""
        try:
            persona = self._get_persona_manager().get(persona_name)
            return persona.system_prompt or ""
        except Exception:
            return ""

    @staticmethod
    def _phase_persona(phase_name: str) -> str | None:
        """查表拿阶段对应的 persona 名."""
        return _PHASE_PERSONAS.get(phase_name)

    # ──────────────────────────────────────────────────────────────
    # Phase-gate
    # ──────────────────────────────────────────────────────────────

    def _check_gate(
        self, from_phase: str, to_phase: str, evidence: dict[str, Any]
    ) -> bool:
        """评估阶段转移门. 通过/已 override 返回 True; 阻断时把 feedback
        拼进 _speculator_hint (下轮 prompt 用) 并返回 False, caller 应
        continue 到下一轮迭代, 不推进到 to_phase.

        共享状态写一条记录进 history, 让 phase_tool 能查到最新门决策.
        """
        state = get_shared_phase_gate_state()
        # override 优先: 已强制放行的转移直接记一条 approved, 不再评估
        if (from_phase, to_phase) in state.overrides:
            state.history.append(
                PhaseGate(
                    from_phase=from_phase,
                    to_phase=to_phase,
                    status="approved",
                    required_evidence=self.phase_gate_hook.config.required_for(
                        from_phase, to_phase
                    ),
                    feedback="override 放行",
                )
            )
            state.pending_transition = (from_phase, to_phase)
            return True

        gate = self.phase_gate_hook.evaluate(from_phase, to_phase, evidence)
        state.history.append(gate)
        state.pending_transition = (from_phase, to_phase)

        if gate.is_blocked:
            fb = gate.feedback or (
                f"阶段转移 {from_phase}→{to_phase} 被阻断: 缺 {gate.missing_evidence}"
            )
            self._speculator_hint = (
                (self._speculator_hint + "\n" + fb).strip() if self._speculator_hint else fb
            )
            print(
                f"  → Gate blocked {from_phase}→{to_phase}: "
                f"missing {gate.missing_evidence}"
            )
            return False
        return True

    # ──────────────────────────────────────────────────────────────
    # Progressive budget
    # ──────────────────────────────────────────────────────────────

    def _check_budget(self, iteration: int, plan: dict[str, Any]) -> bool:
        """检查 plan 的 mode 是否在当前迭代预算允许范围内.

        通过返回 True (含 budget 未启用 / 已降级放行 / mode 允许三种情况).
        不通过时把"用哪个 mode 代替"的提示拼进 _speculator_hint, 下轮 prompt
        能看到, 返回 False 让 caller continue 到下一轮迭代.

        每个档位有 max_calls 次拒绝额度, 用尽后整条预算降级为放行, 避免
        LLM 反复提同样的 mode 把循环卡死.
        """
        if self._budget is None or self._budget_degraded:
            return True
        tier = self._budget.for_iteration(iteration)
        mode = plan.get("mode")
        if tier.allows(mode):
            # 这轮通过了就清掉该档位的拒绝计数, 下次重新数
            self._budget_rejects.pop(tier.label, None)
            return True

        rejects = self._budget_rejects.get(tier.label, 0) + 1
        self._budget_rejects[tier.label] = rejects
        if tier.max_calls is not None and rejects > tier.max_calls:
            # 拒绝额度用尽, 降级放行剩下的所有 mode, 不再卡
            self._budget_degraded = True
            print(
                f"  → Budget degraded at iter {iteration}: "
                f"{tier.label} reject cap {tier.max_calls} hit, allowing all modes"
            )
            return True

        allowed = ", ".join(tier.allowed_modes) if tier.allowed_modes else "any"
        fb = (
            f"迭代 {iteration} 预算档位 {tier.label}: mode={mode} 不被允许, "
            f"可用: {allowed}. 请改用允许的 mode 重新规划."
        )
        self._speculator_hint = (
            (self._speculator_hint + "\n" + fb).strip() if self._speculator_hint else fb
        )
        print(
            f"  → Budget rejected mode={mode} at iter {iteration} "
            f"(tier {tier.label}, reject {rejects}/{tier.max_calls})"
        )
        return False

    async def _drain_side_questions(self) -> int:
        """轮空时把 pending 侧边问题答掉. 返回答了几个.

        拿 shared SideChannel 的 pending 快照, 逐条调 model.ainvoke 出答案,
        再 channel.respond() 写回. 单条失败不阻塞其他条, 也不抛异常 ——
        侧边对话是次要任务, 不能影响主 loop.
        """
        if not self._side_channel_enabled:
            return 0
        from huginn.side_conversation import get_shared_side_channel

        channel = self._side_channel or get_shared_side_channel()
        pending = channel.drain()
        if not pending:
            return 0
        from langchain_core.messages import HumanMessage, SystemMessage

        # Side questions are low-priority — use a cheap model when available
        side_model = self.model
        router = getattr(self, 'model_router', None) or getattr(getattr(self, 'agent', None), 'model_router', None)
        if router is not None:
            try:
                side_model = router.select("cheap", prefer_cheap=True) or self.model
            except Exception:
                pass

        answered = 0
        for sq in pending:
            try:
                messages = [
                    SystemMessage(
                        content=(
                            "You are answering a side question while the main "
                            "research loop is idle. Keep it concise and direct."
                        )
                    ),
                    HumanMessage(content=sq.question),
                ]
                response = await side_model.ainvoke(messages)
                answer = str(response.content).strip()
                if answer:
                    channel.respond(sq.id, answer)
                    answered += 1
                    print(f"  → [side] answered {sq.id}: {answer[:80]}")
            except Exception as exc:
                # 单条失败不影响其他, 也不影响主 loop
                print(f"  → [side] failed to answer {sq.id}: {exc}")
        return answered

    def _get_clarification_manager(self):
        """懒加载 ClarificationManager. 不可用时返回 None, 调用方判空跳过."""
        if self._clarification_mgr is not None:
            return self._clarification_mgr
        try:
            from huginn.interaction.clarification import get_clarification_manager
            self._clarification_mgr = get_clarification_manager()
        except Exception:
            return None
        return self._clarification_mgr

    def _get_plan_store(self):
        """懒加载 PlanStore. 不可用时返回 None, 调用方判空走老的纯 dict 路径."""
        if self._plan_store is not None:
            return self._plan_store
        try:
            from huginn.autoloop.plan_store import PlanStore
            self._plan_store = PlanStore()
        except Exception:
            return None
        return self._plan_store

    def _get_refine_model(self):
        """获取 refine 用的 LLM model, 优先用验证模型 (便宜档).

        没有就用 None, hypothesis_graph.refine_failed 会走 findings 模板拼接.
        """
        try:
            from huginn.llm_config import get_verification_model
            return get_verification_model()
        except Exception:
            return None

    async def _maybe_clarify(
        self,
        checkpoint: str,
        phase_result: Any,
        thread_id: str = "autoloop",
    ) -> str | None:
        """在关键决策点检查是否需要向用户提问.

        checkpoint 取值:
        - "plan": 计划生成后, 高成本 mode (workflow/DFT) 时确认
        - "validation_fail": 验证失败后, 连续 3+ 次时问方向

        返回用户回答的字符串, 或 None (无需提问 / manager 不可用 / 超时走默认).

        非阻塞设计: 没有 async event loop 时直接返回 None, 不强制阻塞.
        autoloop 在 async 上下文里跑, 所以正常路径能拿到回答.
        """
        mgr = self._get_clarification_manager()
        if mgr is None:
            return None

        # 构建上下文
        if checkpoint == "plan":
            plan = phase_result or {}
            mode = plan.get("mode", "")
            # 只对高成本 mode 提问 (workflow=DFT/MD, 通常是几小时)
            expensive_modes = ("workflow", "dft", "md", "vasp", "lammps")
            if mode.lower() not in expensive_modes:
                return None

            ctx = {
                "thread_id": thread_id,
                "question_type": "cost_confirm",
                "phase": "plan",
                "summary": f"mode={mode}, desc={plan.get('description', '')[:200]}",
                "tool": mode,
                "cost_estimate_hours": 1.0,  # workflow 类至少 1h
            }
        elif checkpoint == "validation_fail":
            if self._consecutive_failures < 3:
                return None
            ctx = {
                "thread_id": thread_id,
                "question_type": "validation_fail",
                "phase": "validate",
                "summary": str(phase_result)[:300],
                "consecutive_failures": self._consecutive_failures,
            }
        else:
            return None

        if not mgr.should_ask_contextual(ctx.get("question_type", ""), ctx):
            return None

        # 生成提问
        question, options, default = mgr.generate_question(ctx, model=None)

        try:
            answer = await mgr.ask(
                thread_id=thread_id,
                question=question,
                options=options,
                context=ctx.get("summary", ""),
                default_answer=default,
                timeout=5,  # short timeout: if no human watching, proceed with default
                metadata={
                    "question_type": ctx.get("question_type", ""),
                    "checkpoint": checkpoint,
                    "iteration": self._iteration,
                },
            )
            print(f"  → [clarify] {checkpoint}: {answer[:80]}")
            return answer
        except Exception as exc:
            print(f"  → [clarify] {checkpoint} failed: {exc}")
            return None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    async def run(
        self,
        objective: str,
        max_iterations: int = 20,
        progressive_budget: bool = True,
        goal: Goal | None = None,
        max_refines: int = 8,
    ) -> AutoloopResult:
        """Run the full autonomous loop for the given objective.

        max_iterations 默认 20 (W2 R1 从 5 提到 20), 给阶段门重试和 agentic
        search / goal scheduling 留迭代额度. progressive_budget=True 时按
        迭代数收紧允许的 plan mode (见 ProgressiveBudget.default), False
        则全程放行, 行为跟提预算之前一致.

        max_refines 控制 refute→refine 循环次数上限, 默认 8.
        超过后不再生成修正假设, 避免在错误方向上反复迭代.

        goal 不为空时, 每轮 learn 后用 GoalScheduler.check_completion 查
        success_criteria 是否满足, 满足则提前停循环并在 scheduler 里标记
        completed. 没传 goal 行为不变.
        """
        self._max_refines = max_refines
        self._refine_count = 0
        run_id, provenance_record, run_collector = self._prepare_run(
            objective, progressive_budget, goal
        )
        tracker = get_progress_tracker()
        total_steps = max_iterations * 6 + 1
        progress_task_id = f"autoloop:{run_id}"
        tracker.start_task(
            task_id=progress_task_id,
            description=f"autoloop: {objective[:80]}",
            total_steps=total_steps,
            stage_labels=list(AUTOLOOP_PHASES),
            engine_kind="autoloop",
            metadata={"run_id": run_id, "objective": objective[:200]},
        )
        completed_steps = 0
        phases: list[LoopPhase] = []

        while self._iteration < max_iterations and not self._should_stop:
            self._iteration += 1
            print(f"\n[Autoloop] Iteration {self._iteration}/{max_iterations}: {objective}")
            # 发布 campaign.iteration 事件
            self._emit_campaign("campaign.iteration", {
                "iteration": self._iteration,
                "max": max_iterations,
                "objective": objective[:200],
            })

            # 1. Perceive
            phase = self._run_phase("perceive", self._perceive)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: perceive ({phase.status})")
            if not phase.result:
                print("  → No changes detected, waiting...")
                # 轮空时 drain 侧边对话: 有 pending 问题就顺手答掉, 不白等.
                await self._drain_side_questions()
                await asyncio.sleep(2)
                continue

            context = phase.result

            # 2. Hypothesize
            phase = await self._run_phase_async("hypothesize", self._hypothesize, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: hypothesize ({phase.status})")
            hypothesis = phase.result
            if not hypothesis:
                print("  → No hypothesis generated, skipping iteration")
                continue
            print(f"  → Hypothesis: {hypothesis}")
            # 发布 campaign.hypothesis 事件
            self._emit_campaign("campaign.hypothesis", {
                "iteration": self._iteration,
                "hypothesis": str(hypothesis)[:300],
            })
            # 把假设记进 hypothesis graph, 方便后续 support/refute 追踪
            _current_hyp_id = None
            try:
                _current_hyp_id = self.hypothesis_graph.add_hypothesis(
                    statement=hypothesis,
                    rationale=context.get("summary", ""),
                )
            except Exception:
                logger.warning("error in run: hypothesis_graph.add_hypothesis failed", exc_info=True)

            # 3. Plan
            phase = await self._run_phase_async("plan", self._plan, hypothesis, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: plan ({phase.status})")
            plan = phase.result
            if not plan:
                print("  → No plan generated, skipping iteration")
                continue
            print(f"  → Plan: {plan['mode']} | {plan['description']}")

            # 高成本 plan 时问用户确认. 有 plan_id 说明 _plan 里已经走过
            # PlanStore 确认门了, 别重复问; 没 plan_id (PlanStore 不可用)
            # 才走老的 fire-and-forget 提问
            if not plan.get("plan_id"):
                await self._maybe_clarify("plan", plan)

            # 预算: 后期迭代限制昂贵 mode. 拒绝时把可用 mode 写进 hint,
            # continue 到下一轮让 LLM 改提 plan. 降级后全程放行.
            if not self._check_budget(self._iteration, plan):
                continue

            # gate: plan→execute — 必须有 mode + description 才放行
            if not self._check_gate(
                "plan", "execute",
                {"mode": plan.get("mode"), "description": plan.get("description")},
            ):
                continue

            # 4. Execute
            # JEPA: stash plan's prediction for validate to compare against actual
            self._current_prediction = plan.get("expected_prediction", "")
            phase = await self._run_phase_async("execute", self._execute, plan, context)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: execute ({phase.status})")
            execution_result = phase.result
            if phase.error:
                print(f"  → Execution failed: {phase.error}")
                continue
            print(f"  → Execution complete: {execution_result}")

            # plan 跑完了, 标记 PlanStore 里的 plan 为 completed
            _plan_id = plan.get("plan_id") if isinstance(plan, dict) else None
            if _plan_id:
                try:
                    store = self._get_plan_store()
                    if store is not None:
                        store.complete_plan(_plan_id)
                except Exception:
                    logger.warning("error in run: store.complete_plan failed", exc_info=True)

            # gate: execute→validate — 必须有 mode (执行模式) 才放行
            if not self._check_gate(
                "execute", "validate",
                {"mode": plan.get("mode")},
            ):
                continue

            # 5. Validate
            phase = await self._run_phase_async("validate", self._validate, execution_result)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: validate ({phase.status})")
            validation = phase.result
            print(f"  → Validation: {validation}")

            # 更新假设图: tests_passed → support, 否则 → refute → refine
            try:
                tests_passed = validation.get("tests_passed", False)
                if _current_hyp_id is not None:
                    if tests_passed:
                        self.hypothesis_graph.support(_current_hyp_id, evidence=validation)
                    else:
                        self.hypothesis_graph.refute(
                            _current_hyp_id,
                            evidence={"errors": validation.get("errors", "tests failed")},
                        )
                        # refine 闭环: refute 后生成修正假设, 下轮迭代处理
                        if self._refine_count < self._max_refines:
                            try:
                                new_hyp = self.hypothesis_graph.refine_failed(
                                    _current_hyp_id,
                                    evidence={"errors": validation.get("errors", "tests failed")},
                                    model=self._get_refine_model(),
                                )
                                self._refine_count += 1
                                logger.info(
                                    "refine %d/%d: %s → %s",
                                    self._refine_count, self._max_refines,
                                    _current_hyp_id, new_hyp,
                                )
                                # 发布 campaign.refine 事件
                                self._emit_campaign("campaign.refine", {
                                    "iteration": self._iteration,
                                    "refine_count": self._refine_count,
                                    "max_refines": self._max_refines,
                                    "old_hypothesis": str(_current_hyp_id),
                                    "new_hypothesis": str(new_hyp)[:300] if new_hyp else "",
                                })
                            except Exception:
                                logger.warning("refine_failed failed", exc_info=True)
                        else:
                            # refine 次数耗尽 → 战略 pivot, 不再修参数, 换方向
                            try:
                                _obj = self._objective if hasattr(self, "_objective") else ""
                                new_hyp = self.hypothesis_graph.pivot(
                                    _current_hyp_id,
                                    evidence={"errors": validation.get("errors", "tests failed")},
                                    model=self._get_refine_model(),
                                    objective=_obj,
                                )
                                self._refine_count = 0  # reset: 新方向有新的 refine 预算
                                logger.info(
                                    "PIVOT: %s → %s (refine budget reset)",
                                    _current_hyp_id, new_hyp,
                                )
                                self._emit_campaign("campaign.refine", {
                                    "iteration": self._iteration,
                                    "pivot": True,
                                    "old_hypothesis": str(_current_hyp_id),
                                    "new_hypothesis": str(new_hyp)[:300] if new_hyp else "",
                                    "reason": "max_refines_reached",
                                })
                            except Exception:
                                logger.warning(
                                    "pivot failed for %s, giving up on this hypothesis",
                                    _current_hyp_id, exc_info=True,
                                )
            except Exception:
                logger.warning("error in run: hypothesis_graph support/refute update failed", exc_info=True)

            # 连续失败计数: 通过则清零, 不通过则累加并在阈值时问用户
            _tests_ok = _extract_tests_passed(validation)
            if _tests_ok:
                self._consecutive_failures = 0
            else:
                self._consecutive_failures += 1
                # 连续 3+ 次失败, 问用户方向 (非阻塞, 超时走默认继续)
                await self._maybe_clarify("validation_fail", validation)

            # gate: validate→learn — 要有 tests_passed 证据才放行.
            # tests 没过 = 没有"测试通过"的证据, 传空 dict 让门阻断,
            # 而不是传 tests_passed=False (hook 只查 key 存在性, False 会被当成有值)
            _gate_evidence: dict[str, Any] = (
                {"tests_passed": True} if _tests_ok else {}
            )
            # 透传数学证据 key (由 _validate 从 execution_result 收集):
            # conservation_law / dimensional_consistent / pde_classification /
            # sobol_top_features / constraint_check. math_checker 用 Dempster-
            # Shafer 合成这些 source, belief(pass) <= threshold 时阻断.
            for _mk in (
                "conservation_law",
                "dimensional_consistent",
                "pde_classification",
                "sobol_top_features",
                "constraint_check",
            ):
                if _mk in validation:
                    _gate_evidence[_mk] = validation[_mk]
            if not self._check_gate("validate", "learn", _gate_evidence):
                continue

            # 6. Learn
            phase = await self._run_phase_async("learn", self._learn, hypothesis, plan, validation)
            phases.append(phase)
            completed_steps += 1
            tracker.update(progress_task_id, current_step=completed_steps,
                           current_label=f"iter {self._iteration}: learn ({phase.status})")
            print(f"  → Learning complete")

            # Goal completion: success_criteria 全命中 → 提前停循环.
            # 没 goal 或 criteria 为空时 check_completion 返回 False, 不影响.
            if goal is not None and GoalScheduler.check_completion(goal, validation):
                print(f"  → Goal completed: {goal.objective}")
                goal.status = "completed"
                if self._goal_scheduler is not None:
                    self._goal_scheduler.complete_goal(goal.id)
                self._should_stop = True

            # GoalJudge 反馈: 每 3 轮或最后一轮做一次快速目标判定.
            # 没达成时把 gaps 拼进 _speculator_hint, 下轮 plan 会看到.
            if goal is not None and not self._should_stop:
                if self._iteration % 3 == 2 or self._iteration >= max_iterations - 1:
                    try:
                        from huginn.evaluation.goal_judge import GoalJudge
                        judge = GoalJudge(llm=None)  # rule-based in loop, LLM judge at exit
                        final_text = str(validation.get("summary") or
                                         validation.get("result_data") or
                                         execution_result.get("summary", ""))
                        gj = judge.judge(goal.objective, None, final_text)
                        if gj.get("achieved"):
                            print(f"  → GoalJudge: achieved (score={gj['score']})")
                            self._should_stop = True
                        elif gj.get("gaps"):
                            gap_hint = "; ".join(gj["gaps"][:3])
                            self._speculator_hint = (
                                (self._speculator_hint + "\n" + gap_hint).strip()
                                if self._speculator_hint else gap_hint
                            )
                            print(f"  → GoalJudge gaps: {gap_hint}")
                    except Exception:
                        logger.warning("error in run: GoalJudge evaluation failed", exc_info=True)

            # JEPA intrinsic motivation: 高 surprise = 预测误差大 = 这个方向
            # agent 的心智模型不准, 值得继续探索. 把 surprise 信号注入
            # _speculator_hint, 下轮 hypothesize/plan 会看到并优先关注.
            # MPC 的 receding horizon: 每轮用预测误差调整下一步, 不是固定计划.
            surprise = getattr(self, "_last_surprise", 0.0)
            if surprise > 0.5:
                surprise_hint = (
                    f"High prediction surprise ({surprise:.2f}) last iteration: "
                    "the actual result differed significantly from what was predicted. "
                    "This area is poorly understood — consider exploring why."
                )
                self._speculator_hint = (
                    (self._speculator_hint + "\n" + surprise_hint).strip()
                    if self._speculator_hint else surprise_hint
                )
                print(f"  → Surprise: {surprise:.2f} (high — exploring)")
            elif surprise > 0 and self._iteration > 1:
                print(f"  → Surprise: {surprise:.2f} (low — model matches reality)")

        # 7. Report + finalize
        return await self._finalize_run(
            objective, phases, run_id, provenance_record,
            run_collector, tracker, progress_task_id, completed_steps,
        )

    def _emit_campaign(self, event_type: str, data: dict) -> None:
        """发布 campaign.* 事件到 EventBus, fire-and-forget."""
        try:
            from huginn.events.integration import _publish
            import asyncio
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(_publish(event_type, data, source="autoloop"))
        except Exception:
            pass

    def _prepare_run(
        self, objective: str, progressive_budget: bool, goal: Goal | None
    ) -> tuple[str, Any, Any]:
        """Set up run state: provenance, telemetry, budget, speculator."""
        run_id = f"loop_{uuid.uuid4().hex[:8]}"
        self._run_start_time = time.time()
        self._objective = objective

        from huginn.provenance import ProvenanceLogger, ProvenanceRecord
        provenance_logger = ProvenanceLogger(
            self.workspace / ".huginn" / "provenance.jsonl"
        )
        provenance_record = ProvenanceRecord(
            run_id=run_id,
            objective=objective,
            timestamps={"start": datetime.now().isoformat()},
        )
        self._provenance_record = provenance_record
        self._provenance_logger = provenance_logger

        from huginn.telemetry import TelemetryCollector, set_telemetry_collector
        run_collector = TelemetryCollector()
        set_telemetry_collector(run_collector)

        self._iteration = 0
        self._should_stop = False
        self._consecutive_failures = 0

        if goal is not None and goal.status == "pending":
            goal.status = "active"
            if self._goal_scheduler is not None:
                self._goal_scheduler.update_goal(goal.id, status="active")

        self._budget = ProgressiveBudget.default() if progressive_budget else None
        self._budget_rejects: dict[str, int] = {}
        self._budget_degraded = False

        self._speculator_hint = ""
        self._last_visual_context = ""  # reset per run, stale data shapes mislead
        self._current_prediction = ""  # reset JEPA prediction buffer
        self._last_surprise = 0.0
        try:
            from huginn.agents.speculator import on_turn_start
            spec_result = on_turn_start(objective)
            self._speculator_hint = spec_result.get("hint", "")
            if spec_result.get("predictions"):
                print(f"[Autoloop] Speculator: {self._speculator_hint}")
        except Exception as exc:
            print(f"[Autoloop] Speculator skipped: {exc}")

        return run_id, provenance_record, run_collector

    async def _finalize_run(
        self, objective: str, phases: list[LoopPhase],
        run_id: str, provenance_record: Any,
        run_collector: Any, tracker: Any, progress_task_id: str,
        completed_steps: int,
    ) -> AutoloopResult:
        """Report, save trajectory, judge goal, write provenance + FAIR metadata."""
        total_time = time.time() - getattr(self, "_run_start_time", time.time())
        report_phase = await self._run_phase_async(
            "report", self._report, objective, phases, total_time
        )
        phases.append(report_phase)
        completed_steps += 1
        tracker.update(progress_task_id, current_step=completed_steps,
                       current_label=f"report ({report_phase.status})")

        if report_phase.status == "completed":
            tracker.complete(progress_task_id, result={"report_path": report_phase.result})
        else:
            tracker.fail(progress_task_id, f"report phase failed: {report_phase.error}")

        # session summary → long-term memory
        try:
            self.memory.promote_session_summary(tier="long")
        except Exception:
            logger.debug("session summary promotion failed", exc_info=True)

        # trajectory
        trajectory_path = None
        trajectory_data = None
        try:
            from huginn.telemetry import save_trajectory, load_trajectory
            traj_dir = self.workspace / ".huginn" / "trajectories"
            trajectory_path = traj_dir / f"{run_id}.json"
            save_trajectory(run_collector, trajectory_path, metadata={
                "run_id": run_id, "objective": objective[:200],
                "phases": [p.name for p in phases], "total_time": total_time,
            })
            trajectory_data = load_trajectory(trajectory_path)
        except Exception:
            trajectory_path = None

        # goal judgment
        goal_achieved = None
        goal_judgment = None
        try:
            from huginn.evaluation.goal_judge import GoalJudge
            final_output = str(report_phase.result or "")
            judge = GoalJudge(llm=self.verification_model or self.model)
            goal_judgment = judge.judge(
                objective=objective, trajectory=trajectory_data,
                final_output=final_output,
            )
            goal_achieved = goal_judgment.get("achieved")
        except Exception as e:
            print(f"[Autoloop] GoalJudge skipped: {e}")

        # provenance
        provenance_path = None
        try:
            provenance_record.timestamps["end"] = datetime.now().isoformat()
            self._provenance_logger.log(provenance_record)
            provenance_path = str(self._provenance_logger.path)
        except Exception:
            provenance_path = None

        # FAIR metadata
        try:
            from huginn.export.fair_metadata import generate_dataset_metadata, write_fair_jsonld
            run_results: dict[str, Any] = {}
            for ph in phases:
                if ph.result and isinstance(ph.result, dict):
                    run_results.update(ph.result)
            fair_metadata = generate_dataset_metadata(
                run_id=run_id, objective=objective, results=run_results,
                provenance={
                    "report_path": str(report_phase.result) if report_phase.result else None,
                    "trajectory_path": str(trajectory_path) if trajectory_path else None,
                    "provenance_path": provenance_path,
                    "start_time": provenance_record.timestamps.get("start"),
                    "end_time": provenance_record.timestamps.get("end"),
                },
            )
            jsonld_path = self.workspace / f"{run_id}_dataset.jsonld"
            write_fair_jsonld(fair_metadata, jsonld_path)
            logger.info("FAIR JSON-LD written to %s", jsonld_path)
        except Exception:
            logger.debug("FAIR metadata generation failed", exc_info=True)

        return AutoloopResult(
            run_id=run_id,
            objective=objective,
            phases=phases,
            success=all(p.status == "completed" for p in phases[-7:]),
            report_path=report_phase.result,
            total_time_seconds=total_time,
            trajectory_path=str(trajectory_path) if trajectory_path else None,
            goal_achieved=goal_achieved,
            goal_judgment=goal_judgment,
            provenance_path=provenance_path,
        )

    def stop(self) -> None:
        """Signal the loop to stop at the next safe point."""
        self._should_stop = True

    # ──────────────────────────────────────────────────────────────
    # Phase implementations
    # ──────────────────────────────────────────────────────────────

    def _perceive(self) -> dict[str, Any] | None:
        """Perceive the workspace using the multi-modal perception layer.

        The PerceptionLayer is now a long-lived member started in __init__,
        so background watchers and log tailers actually accumulate events
        between iterations. Previously we started+stopped it here, which
        killed the watcher threads before they could collect anything.
        """
        perception = self._get_perception()
        snapshot = perception.get_snapshot()
        context = snapshot.to_context()
        if not snapshot.has_activity():
            return None
        return context
    def _perceive_legacy(self) -> dict[str, Any] | None:
        """Legacy perceive (fallback)."""
        changed_files = []
        git_diff = ""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.workspace, capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed_files = [line.strip() for line in result.stdout.strip().split("\n")]
                git_diff = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=self.workspace, capture_output=True, text=True, timeout=10,
                ).stdout
        except Exception:
            logger.warning("error in _perceive_legacy: git diff collection failed", exc_info=True)
        error_patterns = []
        for log_file in self.workspace.rglob("*.log"):
            if log_file.stat().st_mtime > time.time() - 3600:
                try:
                    content = log_file.read_text(errors="ignore")
                    if "ERROR" in content or "FAIL" in content:
                        error_patterns.append(f"{log_file.name}: {content[:200]}")
                except Exception:
                    logger.warning("error in _perceive_legacy: log file error-pattern scan failed", exc_info=True)
        if not changed_files and not error_patterns:
            return None
        return {
            "changed_files": changed_files,
            "git_diff": git_diff,
            "error_patterns": error_patterns,
            "timestamp": datetime.now().isoformat(),
        }

    async def _hypothesize(self, context: dict[str, Any]) -> str | None:
        """Generate a hypothesis from perceived context."""
        # Use knowledge graph + LLM to generate hypothesis
        # symreg (async, up to 60s) and conjecture (sync, fast) are independent
        symreg_task = asyncio.create_task(self._symreg_hint(context))
        conjecture_hint = self._conjecture_hint(context)
        symreg_hint = await symreg_task
        prompt = self._build_hypothesis_prompt(context)
        if symreg_hint:
            prompt = f"{symreg_hint}\n{prompt}"
        if conjecture_hint:
            prompt = f"{conjecture_hint}\n{prompt}"
        # 按研究类型选 persona: MD 类用 md_expert, 默认走 dft_expert.
        # 这俩 persona 在 personas.py 内置, 直接取就行.
        persona_name = self._pick_hypothesis_persona(context)
        try:
            response = await self._llm_chat(prompt, persona_name=persona_name, task="reasoning")
            return response.strip()
        except Exception:
            return None

    def _conjecture_hint(self, context: dict[str, Any]) -> str:
        """跑 Moonshine 跨域猜想流水线, 返回注入 prompt 的 hint.

        从 context 提取源问题和领域, 调 ConjectureGenerator 生成跨域类比
        猜想. 失败返回空串, 不影响 hypothesize 主流程.
        """
        try:
            from huginn.autoloop.conjecture import get_conjecture_generator

            source_problem = context.get("goal") or context.get("observation") or ""
            if not source_problem or len(source_problem) < 10:
                return ""
            source_domain = context.get("domain") or "materials science"
            # 目标领域: 从 KG 拿当前研究方向, 默认用 "battery cathodes"
            target_domain = context.get("target_domain") or "battery cathodes"

            gen = get_conjecture_generator()
            result = gen.run(
                source_problem=str(source_problem)[:500],
                source_domain=str(source_domain),
                target_domain=str(target_domain),
                model=None,  # template mode, 不烧 token
            )
            conjecture = result.get("conjecture", {})
            statement = conjecture.get("statement", "")
            prediction = conjecture.get("prediction", "")
            if not statement:
                return ""
            return (
                f"[Cross-domain analogy hint]\n"
                f"Conjecture: {statement}\n"
                f"Prediction: {prediction}\n"
                f"(This is a template-based analogy for inspiration only.)"
            )
        except Exception:
            return ""

    async def _symreg_hint(self, context: dict[str, Any]) -> str:
        """从 observation_data 跑符号回归, 把最优解析表达式作为 hint 返回.

        PSE/PSRN 不可用、数据不全、搜索失败都返回空串, 不影响 hypothesize.
        time_limit 压到 60s 避免 hypothesize 阶段被符号回归卡死.

        A3: 同时查 KB 拿已知公式形式 (Arrhenius / Brillouin / Langmuir 等) 作为
        kb_candidate_forms 前缀, 跟 symreg 数据驱动候选一起注入 hypothesize prompt.
        KB 给先验形式, symreg 给数据拟合, 两边互补."""
        data = context.get("observation_data")
        if not isinstance(data, dict) or not data:
            return ""
        # A3: 先查 KB 拿已知公式形式, symreg 失败时 KB 候选仍能注入
        kb_forms = self._query_kb_known_forms(data)
        try:
            from huginn.tools.sci.symbolic_regression_tool import (
                SymbolicRegressionInput,
                SymbolicRegressionTool,
            )
            from huginn.types import ToolContext

            target = data.get("target_column") or data.get("target") or "y"
            if target not in data:
                return kb_forms
            tool = SymbolicRegressionTool()
            args = SymbolicRegressionInput(
                action="discover",
                data_json=data,
                target_column=target,
                time_limit=60,
                top_k=3,
            )
            ctx = ToolContext(
                session_id=f"symreg_{uuid.uuid4().hex[:8]}",
                workspace=str(self.workspace),
                config=self.settings,
            )
            vr = await tool.call(args, ctx)
            symreg_block = ""
            if vr.success and vr.data:
                cands = (
                    vr.data.get("candidates")
                    or vr.data.get("expressions")
                    or vr.data.get("pareto_front")
                    or []
                )
                if cands:
                    top = cands[0] if isinstance(cands, list) else cands
                    expr = top.get("expression") if isinstance(top, dict) else str(top)
                    if expr:
                        symreg_block = (
                            "### Data-driven candidate law (symbolic regression)\n"
                            f"Top recovered expression: {expr}\n"
                            "Use this as a data-driven candidate when forming the hypothesis.\n"
                            "### End candidate law"
                        )
            if symreg_block and kb_forms:
                return f"{kb_forms}\n{symreg_block}"
            return symreg_block or kb_forms
        except Exception:
            return kb_forms

    def _query_kb_known_forms(self, data: dict[str, Any]) -> str:
        """查 KB 拿已知公式形式 (Arrhenius / Brillouin / Langmuir 等) 作为
        symreg 先验. 失败/空都返回空串."""
        try:
            target = data.get("target_column") or data.get("target") or "y"
            feature_keys = [k for k in data.keys() if k != target and isinstance(data[k], list)]
            query = f"symbolic expression formula {target} {' '.join(feature_keys[:3])}"
            kb = self._get_kb()
            if kb is None or kb.count() == 0:
                return ""
            chunks = kb.query(query, top_k=2)
            if not chunks:
                return ""
            lines = []
            for i, c in enumerate(chunks, 1):
                text = (c.get("text") or "").strip()
                if text:
                    lines.append(f"[{i}] {text[:300]}")
            if not lines:
                return ""
            return (
                "### KB candidate forms (known first-principles expressions)\n"
                "The knowledge base suggests these known formula forms. Compare "
                "your data-driven candidate against them.\n"
                + "\n".join(lines)
                + "\n### End KB candidate forms"
            )
        except Exception:
            return ""

    def _pick_hypothesis_persona(self, context: dict[str, Any]) -> str:
        """根据 context + surprise 选择 persona.
        高 surprise → 切换到 reviewer persona, 更批判地审视上轮意外结果.
        否则按内容走 DFT/MD 专家."""
        # JEPA: 上轮预测误差大时, 用 reviewer persona 审视 —
        # 预测错了说明 agent 的心智模型不准, 需要更批判的视角.
        if getattr(self, "_last_surprise", 0.0) > 0.6:
            return "reviewer"
        blob = json.dumps(context, ensure_ascii=False).lower()
        md_markers = ("md", "lammps", "molecular dynamics", "nvt", "npt", "md_steps")
        if any(m in blob for m in md_markers):
            return "md_expert"
        return "dft_expert"

    async def _plan(self, hypothesis: str, context: dict[str, Any]) -> dict[str, Any] | None:
        """Generate a plan from hypothesis and persist it to PlanStore.

        以前只返回一个临时 dict, turn 结束就丢了. 现在往 PlanStore 落一份,
        跨会话可恢复, 用户也能 confirm/reject. PlanStore 不可用时退回老行为.
        """
        prompt = self._build_plan_prompt(hypothesis, context)
        try:
            response = await self._llm_chat(prompt, persona_name="default", task="reasoning")
            plan = self._parse_plan(response)
        except Exception:
            return None

        if not plan:
            return None

        # 落 PlanStore: 创建 plan → cost 确认门 → confirm/reject
        plan_store = self._get_plan_store()
        if plan_store is None:
            return plan

        try:
            from huginn.autoloop.plan_store import PlanStep

            steps = [
                PlanStep(
                    id="step_0",
                    description=plan.get("description", ""),
                    tool=plan.get("mode", ""),
                )
            ]
            persisted = plan_store.create_plan(
                objective=hypothesis,
                steps=steps,
                auto_confirm=False,
                metadata={"mode": plan.get("mode", ""), "source": "autoloop"},
            )

            # cost 确认门: None = 不用问 (非高成本 mode / manager 不可用),
            # 直接放行; 显式拒绝才拦. bool 和字符串都兼容 (测试 mock 常传 bool)
            answer = await self._maybe_clarify("plan", plan)
            if answer is None:
                should_confirm = True
            elif isinstance(answer, bool):
                should_confirm = answer
            elif isinstance(answer, str):
                should_confirm = answer.lower().strip() not in (
                    "no", "n", "cancel", "reject", "decline", "stop", "abort",
                )
            else:
                should_confirm = bool(answer)

            if should_confirm:
                plan_store.confirm_plan(persisted.id)
                plan_store.mark_executing(persisted.id)
            else:
                plan_store.reject_plan(persisted.id, reason="user declined")
                return None

            plan["plan_id"] = persisted.id
        except Exception as e:
            logger.warning("plan store persistence failed: %s", e)
            # PlanStore 挂了不阻塞执行, 退回老的纯 dict 路径

        return plan

    async def _execute(self, plan: dict[str, Any], context: dict[str, Any]) -> Any:
        """Execute the plan using the appropriate sub-engine."""
        mode = plan.get("mode", "coder")
        description = plan.get("description", "")

        if mode == "coder":
            # Use CoderRunner to modify code
            result = await self._execute_coder(description, context)
        elif mode == "workflow":
            # Use WorkflowEngine to run computational pipeline
            result = await self._execute_workflow(description, context)
            # On failure, try applying a learned heuristic fix before giving up
            if isinstance(result, dict) and not result.get("success", True):
                result = await self._try_evolved_fix(mode, description, result) or result
        elif mode == "dynamic_workflow":
            # A5: agent 写的并行 subtask 脚本, orchestrator 并发跑
            result = await self._execute_dynamic_workflow(plan, context)
        elif mode == "explore":
            # Use ExplorationOrchestrator to search design space
            result = await self._execute_explore(description, context)
        elif mode == "skill":
            # Run a pre-built composite skill pipeline
            result = await self._execute_skill(plan, context)
        else:
            raise ValueError(f"Unknown plan mode: {mode}")

        # provenance: 记一次 tool call, mode 当工具名, plan 当输入参数
        self._record_provenance(mode, plan, result)
        # 缓存给 _build_plan_prompt 的 pipeline suggest_next 用
        self._last_execution_result = {
            '_tool_name': mode,
            '_tool_input': plan,
            'result': result if isinstance(result, dict) else {'value': str(result)[:500]},
        }
        return result

    def _record_provenance(
        self, tool_name: str, input_params: dict[str, Any], output: Any
    ) -> None:
        """往当前 run 的 provenance record 追加一次 tool-call 快照.

        run() 启动时建好 self._provenance_record; 没建 (比如单测里直接调
        _execute) 就跳过, 不强求调用方先 setup. provenance 是 best-effort,
        快照挂了不能把 execute 带挂.
        """
        record = getattr(self, "_provenance_record", None)
        if record is None:
            return
        try:
            from huginn.provenance import capture
            record.add_snapshot(capture(tool_name, input_params, output=output))
        except Exception:
            logger.warning("error in _record_provenance: capture snapshot failed", exc_info=True)

    async def _try_evolved_fix(
        self, tool_name: str, tool_input: dict[str, Any], error_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Check if the evolution engine has a learned fix for this error.

        This is the other half of the Learn→Execute loop: when a tool fails,
        we ask evolution if it's seen this error before and has a fix.
        Returns a patched result dict on hit, None on miss.
        """
        try:
            evolution = self._get_evolution()
            error_str = str(error_result.get("error", ""))
            fix = evolution.apply_heuristic_fix(tool_name, tool_input, error_str)
            if fix:
                patched_desc = fix.get("description", str(tool_input))
                return await self._execute_workflow(
                    patched_desc, {"_evolved_fix": True}
                )
        except Exception:
            logger.warning("error in _try_evolved_fix: apply_heuristic_fix failed", exc_info=True)
        return None

    async def _execute_dynamic_workflow(
        self, plan: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """A5: 跑 agent 提交的并行工作流脚本.

        plan 里带 "script" 字段 (WorkflowScript.from_dict 的输入), 直接走
        WorkflowOrchestrator.run() 同步等完. 失败的 subtask 不炸整体,
        返回聚合结果让 validate/learn 阶段看.
        """
        from huginn.autoloop.dynamic_workflow import (
            WorkflowOrchestrator,
            WorkflowScript,
        )
        from huginn.types import ToolContext

        raw_script = plan.get("script") or {}
        if isinstance(raw_script, str):
            # agent 可能传 JSON 字符串
            import json
            try:
                raw_script = json.loads(raw_script)
            except json.JSONDecodeError:
                raw_script = {}
        script = WorkflowScript.from_dict(raw_script)
        if not script.subtasks:
            return {
                "mode": "dynamic_workflow",
                "success": False,
                "error": "脚本无有效 subtask",
            }
        orch = WorkflowOrchestrator(
            max_concurrent=script.max_concurrent,
        )
        ctx = ToolContext(
            session_id=f"dynwf_{script.id}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        result = await orch.run(script, ctx)
        return {
            "mode": "dynamic_workflow",
            "success": result.success,
            "workflow_id": result.id,
            "n_total": result.n_total,
            "n_completed": result.n_completed,
            "n_failed": result.n_failed,
            "summary": result.summary(),
        }

    async def _validate(self, execution_result: Any) -> dict[str, Any]:
        """Validate execution results using benchmarks and constraints."""
        results = {
            "tests_passed": False,
            "constraints_satisfied": False,
            "benchmarks": {},
        }

        if isinstance(execution_result, dict):
            # Extract visual primitives from tool output — the deictic pointers
            # that let the next iteration's hypothesis/plan reason about data
            # shape without needing image input (Mirage + Visual Primitives).
            visual_hint = execution_result.get("_visual_hint")
            if visual_hint:
                results["visual_primitives"] = visual_hint
                self._last_visual_context = visual_hint

            r_phys = execution_result.get("r_phys")
            if r_phys is None:
                result_type = execution_result.get("result_type")
                result_data = execution_result.get("result_data")
                if result_type and result_data:
                    try:
                        from huginn.tools.validate_tool import (
                            ValidateTool,
                            ValidateToolInput,
                        )

                        validator = ValidateTool()
                        tool_ctx = ToolContext(
                            session_id=f"validate_{uuid.uuid4().hex[:8]}",
                            workspace=str(self.workspace),
                            config=self.settings,
                        )
                        vr = await validator.call(
                            ValidateToolInput(
                                result_type=result_type,
                                result_data=result_data,
                            ),
                            tool_ctx,
                        )
                        if vr.success and vr.data:
                            r_phys = vr.data.get("r_phys")
                            results["physics_validation"] = vr.data
                    except Exception as e:
                        results["physics_validation_error"] = str(e)
            if r_phys is not None:
                results["r_phys"] = r_phys

        try:
            collapse = self._detect_thinking_collapse(execution_result)
            if collapse:
                results["thinking_collapse"] = collapse
        except Exception as e:
            results["thinking_collapse_error"] = str(e)

        # pytest, benchmark, math validation — sync, offload to thread
        py_test, bench_report, math_val = await asyncio.gather(
            self._run_pytest(),
            self._run_benchmark(),
            self._run_math_validation(execution_result),
            return_exceptions=True,
        )

        if isinstance(py_test, dict):
            results.update(py_test)
        elif isinstance(py_test, Exception):
            results["test_output"] = f"Test execution error: {py_test}"

        if isinstance(bench_report, dict):
            results["benchmarks"] = bench_report
        elif isinstance(bench_report, Exception):
            logger.warning("BenchmarkRunner failed", exc_info=True)

        if isinstance(math_val, dict):
            results["math_validation"] = math_val
        elif isinstance(math_val, Exception):
            results["math_validation_error"] = str(math_val)

        try:
            math_ev = await self._collect_math_evidence(
                execution_result, results.get("math_validation", {})
            )
            for _k, _v in math_ev.items():
                results[_k] = _v
        except Exception as e:
            results["math_evidence_error"] = str(e)

        # Conditional verification: run cheap generative_verify first,
        # only call expensive reviewer critique when score < 0.5.
        # ponytail: _generative_verify 依赖 results (被 math_evidence 修改过),
        # 不能和 _collect_math_evidence 并行. 但可以和 emergent_complexity/
        # literature_comparison 并行 — 它们在下面已经 gather 了.
        gen_verify = None
        try:
            gen_verify = await self._generative_verify(execution_result, results)
            if gen_verify:
                results["generative_verify"] = gen_verify
        except Exception as e:
            results["generative_verify_error"] = str(e)

        needs_review = gen_verify is None or gen_verify.get("score", 0.5) < 0.5
        if needs_review:
            try:
                reviewer_kb = self._build_kb_text(
                    query=self._summarize_for_kb(execution_result, results)
                )
                critique = await self._llm_chat(
                    self._build_reviewer_prompt(execution_result, results, reviewer_kb),
                    persona_name="reviewer",
                    model=self.verification_model,
                )
                if critique and critique.strip():
                    results["reviewer_critique"] = critique.strip()
            except Exception as e:
                results["reviewer_critique_error"] = str(e)

        # emergent complexity + literature + grader + eval — independent
        ec_task = asyncio.create_task(self._safe_emergent_complexity(execution_result, results))
        lit_task = asyncio.create_task(self._safe_literature_comparison(execution_result, results))
        await asyncio.gather(ec_task, lit_task)

        try:
            from huginn.validation.grader import default_registry
            reg = default_registry()
            merged: dict[str, Any] = {}
            if isinstance(execution_result, dict):
                merged.update(execution_result)
            merged.update(results)
            grader_list = reg.evaluate_all(merged)
            results["grader_scores"] = {
                gr.name: {
                    "score": gr.score,
                    "passed": gr.passed,
                    "message": gr.message,
                }
                for gr in grader_list
            }
            if grader_list:
                avg_score = sum(gr.score for gr in grader_list) / len(grader_list)
                results["grader_reward"] = round(avg_score, 4)
                try:
                    from huginn.events.integration import _publish
                    loop = asyncio.get_running_loop()
                    asyncio.ensure_future(_publish("quality.check", {
                        "iteration": self._iteration,
                        "graders": results["grader_scores"],
                        "reward": results.get("grader_reward", 0),
                    }, source="autoloop"))
                except Exception:
                    pass
        except Exception as e:
            results["grader_error"] = str(e)

        try:
            from huginn.evaluation.matworld_bench import MatWorldBench
            bench = MatWorldBench()
            exec_data = execution_result if isinstance(execution_result, dict) else {}
            eval_scores: list[dict] = []
            for task in bench.tasks:
                if task.category in ("structure", "thermo", "electronic"):
                    try:
                        br = bench.evaluate(task.id, exec_data)
                        eval_scores.append({
                            "task_id": task.id,
                            "category": task.category,
                            "passed": br.passed,
                            "score": br.score,
                        })
                    except Exception:
                        pass
            if eval_scores:
                passed = sum(1 for e in eval_scores if e["passed"])
                results["eval_summary"] = {
                    "bench_passed": passed,
                    "bench_total": len(eval_scores),
                    "bench_pass_rate": round(passed / len(eval_scores), 4),
                    "details": eval_scores,
                }
        except Exception as e:
            logger.debug(f"[validate] eval bench failed: {e}")

        # JEPA 式预测误差: 对比 plan 阶段的预测 vs 实际结果.
        # 预测误差高 = surprise = 值得探索的方向 (intrinsic motivation).
        # 低误差 = agent 对这类任务已有良好心智模型.
        prediction = getattr(self, "_current_prediction", "")
        if prediction:
            actual_text = self._extract_text(execution_result)[:500]
            surprise = self._compute_surprise(prediction, actual_text)
            results["prediction_error"] = {
                "predicted": prediction[:200],
                "actual": actual_text[:200],
                "surprise": round(surprise, 3),
            }
            self._last_surprise = surprise

        return results

    async def _run_pytest(self) -> dict[str, Any]:
        """Run pytest in workspace, return results dict."""
        import subprocess
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["python", "-m", "pytest", "-x", "-q", "--tb=line"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "tests_passed": result.returncode == 0,
                "test_output": result.stdout + result.stderr,
            }
        except Exception as e:
            return {"test_output": f"Test execution error: {e}"}

    async def _run_benchmark(self) -> dict[str, Any]:
        """Run BenchmarkRunner, return results dict."""
        try:
            from huginn.validation.benchmarks import BenchmarkRunner
            runner = BenchmarkRunner()
            report = await asyncio.to_thread(
                runner.run, categories=["math", "coding"]
            )
            return {
                "passed": report.passed,
                "failed": report.failed,
                "skipped": report.skipped,
            }
        except Exception:
            logger.warning("BenchmarkRunner failed", exc_info=True)
            return {}

    async def _safe_emergent_complexity(
        self, execution_result: Any, results: dict[str, Any]
    ) -> None:
        """Compute emergent complexity, mutate results in place."""
        try:
            from huginn.validation.emergent_complexity import compute_ec
            results["emergent_complexity"] = compute_ec(execution_result, results)
            ec_score = results["emergent_complexity"].get("ec_score", 0)
            if ec_score < 0.2 and self._iteration > 0:
                ec_hint = f"EC={ec_score:.2f}: low emergent complexity, try diverse tools or cross-domain reasoning"
                self._speculator_hint = (
                    (self._speculator_hint + "\n" + ec_hint).strip()
                    if self._speculator_hint else ec_hint
                )
        except Exception as e:
            results["emergent_complexity_error"] = str(e)

    async def _safe_literature_comparison(
        self, execution_result: Any, results: dict[str, Any]
    ) -> None:
        """Run literature comparison, mutate results in place."""
        try:
            lit_comp = await self._literature_comparison(execution_result)
            if lit_comp:
                results["literature_comparison"] = lit_comp
        except Exception as e:
            results["literature_comparison_error"] = str(e)

    async def _literature_comparison(self, execution_result: Any) -> dict[str, Any]:
        """Extract numeric results, look up literature benchmarks, run innovation
        signal detection. Best-effort — any failure just skips that property.

        Returns {property_key: InnovationSignal} for properties where literature
        data was found. Empty dict if nothing to compare.
        """
        if not isinstance(execution_result, dict):
            return {}

        result_data = (
            execution_result.get("result_data")
            or execution_result.get("parsed")
            or {}
        )
        if not isinstance(result_data, dict):
            return {}

        # system / formula: benchmark_lookup 必须知道查什么材料
        system = None
        for key in ("formula", "system", "material", "compound"):
            val = result_data.get(key) or execution_result.get(key)
            if val:
                system = str(val)
                break
        if not system:
            return {}

        # 抽数值: 扁平 key + 嵌套 lattice_params
        numerics: dict[str, float] = {}
        for key in _LIT_PROPERTY_MAP:
            val = result_data.get(key)
            if val is not None:
                try:
                    numerics[key] = float(val)
                except (TypeError, ValueError):
                    logger.warning("error in _literature_comparison: numeric property cast failed", exc_info=True)
        lattice = result_data.get("lattice_params") or {}
        if isinstance(lattice, dict):
            for param in ("a", "b", "c"):
                val = lattice.get(param)
                if val is not None:
                    try:
                        numerics[f"lattice_{param}"] = float(val)
                    except (TypeError, ValueError):
                        logger.warning("error in _literature_comparison: lattice param cast failed", exc_info=True)

        if not numerics:
            return {}

        try:
            from huginn.tools.literature import LiteratureInput, LiteratureTool
            from huginn.validation.innovation_signal import InnovationSignalDetector
        except ImportError:
            return {}

        tool = LiteratureTool()
        tool_ctx = ToolContext(
            session_id=f"litcmp_{uuid.uuid4().hex[:8]}",
            workspace=str(self.workspace),
            config=self.settings,
        )
        detector = InnovationSignalDetector()

        comparison: dict[str, Any] = {}
        for prop_key, agent_value in numerics.items():
            prop_name = _LIT_PROPERTY_MAP.get(prop_key, prop_key)
            try:
                res = await tool.call(
                    LiteratureInput(
                        action="benchmark_lookup",
                        system=system,
                        property=prop_name,
                        max_results=10,
                    ),
                    tool_ctx,
                )
                if not res.success or not res.data:
                    continue
                reported = res.data.get("reported_values") or []
                lit_values = [r["value"] for r in reported if "value" in r]
                if not lit_values:
                    continue
                signal = detector.detect(prop_key, agent_value, lit_values)
                comparison[prop_key] = signal
            except Exception:
                continue

        return comparison

    @staticmethod
    def _summarize_for_kb(execution_result: Any, results: dict[str, Any]) -> str:
        """把 execution_result + validation results 拍扁成短串当 KB query.
        给 reviewer 检索已知 first-principles 结论用, 失败无所谓."""
        try:
            parts: list[str] = []
            if isinstance(execution_result, dict):
                for k in ("result_type", "equations", "lagrangian", "summary"):
                    v = execution_result.get(k)
                    if v:
                        parts.append(str(v)[:120])
            for k in ("tests_passed", "constraints_satisfied"):
                v = results.get(k)
                if v is not None:
                    parts.append(f"{k}={v}")
            return " ".join(parts)[:400]
        except Exception:
            return ""

    # -- 思维坍塌检测 + 生成式验证 --

    def _detect_thinking_collapse(self, execution_result: Any) -> dict[str, Any] | None:
        """检查 LLM 输出是否陷入重复推理 / 发散 / 工具调用循环.

        三条规则, 纯文本分析不需要 LLM:
          1. 相同短语 (5 词 n-gram) 出现 3+ 次 → 重复推理路径
          2. 输出 > 200 词但 unique word ratio < 0.3 → 发散但不前进
          3. 相同工具 + 相同参数出现 2+ 次 → 工具循环

        检测到任一信号就返回 dict, 否则 None.
        """
        from collections import Counter

        text = self._extract_text(execution_result)
        if not text or len(text.strip()) < 20:
            return None

        signals: dict[str, Any] = {}

        # Rule 1: 重复短语 — 5 词 n-gram 出现 3+ 次
        words = text.lower().split()
        if len(words) >= 10:
            ngrams = [
                " ".join(words[i : i + 5])
                for i in range(len(words) - 4)
            ]
            counts = Counter(ngrams)
            repeated = [(p, c) for p, c in counts.items() if c >= 3]
            if repeated:
                repeated.sort(key=lambda x: -x[1])
                signals["repeated_phrases"] = repeated[:5]

        # Rule 2: 发散推理 — 长文本但词汇丰富度低
        if len(words) > 200:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                signals["divergent_reasoning"] = {
                    "word_count": len(words),
                    "unique_ratio": round(unique_ratio, 3),
                }

        # Rule 3: 工具调用循环 — 相同工具 + 相同参数 2+ 次
        if isinstance(execution_result, dict):
            loops = self._find_tool_call_loops(execution_result)
            if loops:
                signals["tool_call_loops"] = loops

        if not signals:
            return None

        # 严重度: 有重复短语或工具循环 = high, 只有发散 = medium
        has_loop = bool(signals.get("tool_call_loops"))
        has_repeat = bool(signals.get("repeated_phrases"))
        signals["severity"] = "high" if (has_loop or has_repeat) else "medium"
        return signals

    @staticmethod
    def _find_tool_call_loops(execution_result: dict) -> list[dict[str, Any]]:
        """从 execution_result 里找重复的工具调用 (同工具 + 同参数 2+ 次)."""
        import hashlib

        calls = (
            execution_result.get("tool_calls")
            or execution_result.get("steps")
            or execution_result.get("actions")
            or []
        )
        if not isinstance(calls, list):
            return []

        seen: dict[str, int] = {}
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("tool") or call.get("name") or call.get("action") or ""
            params = call.get("input") or call.get("params") or call.get("args") or {}
            try:
                payload = name + json.dumps(params, sort_keys=True, default=str)
            except Exception:
                payload = name + str(params)
            key = hashlib.sha256(payload.encode()).hexdigest()[:12]
            seen[key] = seen.get(key, 0) + 1

        return [
            {"call_hash": k, "count": c}
            for k, c in seen.items()
            if c >= 2
        ]

    @staticmethod
    def _extract_text(execution_result: Any) -> str:
        """从 execution_result 里抽文本, 给坍塌检测做分析用."""
        if execution_result is None:
            return ""
        if isinstance(execution_result, str):
            return execution_result
        if not isinstance(execution_result, dict):
            return str(execution_result)

        parts: list[str] = []
        for key in ("summary", "description", "result_data", "output",
                     "error", "reasoning", "plan", "hypothesis"):
            v = execution_result.get(key)
            if v:
                parts.append(str(v))
        # 嵌套的 steps / tool_calls 里的文本也抽出来
        for key in ("steps", "tool_calls", "actions"):
            items = execution_result.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        for sk in ("description", "output", "result", "error"):
                            sv = item.get(sk)
                            if sv:
                                parts.append(str(sv))
        return " ".join(parts)

    def _compute_surprise(self, prediction: str, actual: str) -> float:
        """JEPA 式预测误差: 预测文本 vs 实际文本的语义距离.

        ponytail: 用关键词 Jaccard 距离代替真正的嵌入余弦距离.
        纯文本操作, 零依赖, 零 LLM 调用. 对于"预测说了 energy, 实际也出了
        energy"这种常见场景已经够用. 升级路径: 用 sentence-transformers
        算 cosine distance, 或训练专门的 JEPA 编码器.
        """
        if not prediction or not actual:
            return 0.0
        # 提取关键词: 去标点, 小写, 过滤停用词和短词
        import re
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
                "in", "on", "at", "for", "and", "or", "not", "this", "that",
                "it", "with", "from", "by", "as", "will", "can", "may"}
        def keywords(text: str) -> set[str]:
            words = re.findall(r'[a-zA-Z_]\w{2,}', text.lower())
            return {w for w in words if w not in stop}
        pred_kw = keywords(prediction)
        actual_kw = keywords(actual)
        if not pred_kw and not actual_kw:
            return 0.0
        # Jaccard distance = 1 - intersection/union
        union = pred_kw | actual_kw
        if not union:
            return 0.0
        intersection = pred_kw & actual_kw
        return 1.0 - len(intersection) / len(union)

    async def _generative_verify(
        self, execution_result: Any, results: dict[str, Any]
    ) -> dict[str, Any] | None:
        """用 verification_model 给 agent 输出打 0-1 质量分.

        分数 < 0.5 标记 needs_retry. verification_model 不可用时返回
        None, 让上层降级到规则检查 (thinking_collapse 等).
        """
        if self.verification_model is None:
            return None

        text = self._extract_text(execution_result)
        if not text or len(text.strip()) < 10:
            return None

        # 截断防止 prompt 爆炸
        snippet = text[:2000]
        collapse = results.get("thinking_collapse", {})
        collapse_hint = ""
        if collapse:
            collapse_hint = (
                f"\nNote: automated checks detected: {json.dumps(collapse, default=str)[:300]}"
            )

        prompt = (
            "You are a verification model. Score the quality of this agent output "
            "from 0.0 to 1.0.\n"
            "1.0 = well-reasoned, complete, correct.\n"
            "0.5 = acceptable but has issues.\n"
            "0.0 = poor, incorrect, or incomplete.\n"
            f"{collapse_hint}\n\n"
            f"Agent output:\n{snippet}\n\n"
            "Respond with ONLY a JSON object: "
            '{"score": <float>, "reason": "<brief>"}'
        )

        resp = await self._llm_chat(prompt, model=self.verification_model)
        score, reason = self._parse_verify_score(resp)

        return {
            "score": score,
            "reason": reason,
            "needs_retry": score < 0.5,
        }

    @staticmethod
    def _parse_verify_score(resp: str) -> tuple[float, str]:
        """从 LLM 响应里抠出 score 和 reason, 容错解析."""
        import re

        if not resp:
            return 0.5, "empty response"

        # 先试 JSON 解析
        try:
            data = json.loads(resp.strip())
            return float(data.get("score", 0.5)), str(data.get("reason", ""))
        except (json.JSONDecodeError, ValueError):
            logger.warning("error in _parse_verify_score: JSON parse failed, falling back to regex", exc_info=True)

        # fallback: 正则抠数字
        m = re.search(r"([01]\.\d+|[01])\b", resp)
        if m:
            return float(m.group(1)), resp[:200]

        return 0.5, resp[:200]

    async def _run_math_validation(self, execution_result: Any) -> dict[str, Any]:
        """把执行结果里的数学结构抽出来, 用数学工具做形式化校验.

        三个独立子项, 互不影响:
          A. 守恒律 (BourbakiTool.check_conservation) — equations 非空时跑
          B. 变分原理 (LeanTool.constitutive/variational_principle) — lagrangian 非空时跑
          C. 自动微分 (AutoDiffTool.gradient) — function spec 齐全时跑

        工具懒加载, 任一缺失/报错只记 *_error, 不阻断其余子项与主 validate 流程.
        engine 没有自己的 tool_registry, 这里直接构造工具实例 (它们都是无状态轻量构造).
        """
        from huginn.types import ToolContext

        out: dict[str, Any] = {}
        if not isinstance(execution_result, dict):
            return out

        tool_ctx = ToolContext(
            session_id=f"mathval_{uuid.uuid4().hex[:8]}",
            workspace=str(self.workspace),
            config=self.settings,
        )

        equations = execution_result.get("equations") or ""
        lagrangian = execution_result.get("lagrangian") or ""
        coords = execution_result.get("coordinates") or []
        velocities = execution_result.get("velocities")
        domain = execution_result.get("conservation_domain") or "continuum_mechanics"
        if equations:
            try:
                from huginn.tools.bourbaki_tool import BourbakiTool

                tool = BourbakiTool()
                raw = await tool.call(
                    {
                        "task": "check_conservation",
                        "domain": domain,
                        "equations": equations,
                    },
                    tool_ctx,
                )
                # BourbakiTool.call 可能返回 dict 或 BourbakiResult; 统一成 dict
                if hasattr(raw, "model_dump"):
                    raw = raw.model_dump()
                out["conservation"] = {
                    "verified": raw.get("verified"),
                    "message": raw.get("message", ""),
                    "fallback": raw.get("fallback", False),
                    "method": "bourbaki",
                }
            except Exception as e:
                out["conservation_error"] = str(e)

        # A2: KB 交叉验证 — 把守恒律方程 + Lagrangian 关键词拿去查 KB, 命中的
        # first-principles 参考块作为 reference_principles 写回, 让下游 reviewer
        # 能对照已知结论. KB 不可用/空查询都不阻断, 只是不写该字段.
        kb_ref = self._query_kb_reference(equations, lagrangian)
        if kb_ref:
            out["reference_principles"] = kb_ref

        if lagrangian and coords:
            try:
                from huginn.tools.lean_tool import LeanTool, LeanToolInput

                tool = LeanTool()
                args = LeanToolInput(
                    action="constitutive",
                    sub_action="variational_principle",
                    lagrangian=lagrangian,
                    coordinates=list(coords),
                    velocities=velocities,
                )
                vr = await tool.call(args, tool_ctx)
                out["variational"] = {
                    "ok": bool(vr.success),
                    "data": vr.data,
                    "error": vr.error,
                    "method": "lean",
                }
            except Exception as e:
                out["variational_error"] = str(e)

        func_spec = execution_result.get("autodiff")
        if isinstance(func_spec, dict) and func_spec.get("function_type"):
            try:
                from huginn.tools.sci.autodiff_tool import (
                    AutoDiffInput,
                    AutoDiffTool,
                )

                tool = AutoDiffTool()
                args = AutoDiffInput(
                    action="gradient",
                    function_type=func_spec.get("function_type", "custom"),
                    function_params=func_spec.get("function_params", {}),
                    variables=func_spec.get("variables", {}),
                    target_variable=func_spec.get("target_variable"),
                )
                vr = await tool.call(args, tool_ctx)
                out["autodiff"] = {
                    "ok": bool(vr.success),
                    "data": vr.data,
                    "error": vr.error,
                }
            except Exception as e:
                out["autodiff_error"] = str(e)

        return out

    async def _collect_math_evidence(
        self, execution_result: Any, math_validation: dict
    ) -> dict[str, Any]:
        """从 execution_result + math_validation 抽 5 个数学证据 key,
        供 PhaseGate 的 MathEvidenceChecker 做 Dempster-Shafer 合成.

        证据来源:
          1. conservation_law — 从 math_validation["conservation"] 透传
          2. dimensional_consistent — execution_result 带 equation 时跑
             symbolic_math_tool action=dimensional_analysis
          3. pde_classification — execution_result 带 pde_coefficients +
             expected_pde_class 时跑 symbolic_math_tool action=pde_classify
          4. sobol_top_features — execution_result 带 sobol_data +
             hypothesis_features 时跑 symbolic_regression_tool action=sobol_indices
          5. constraint_check — execution_result 带 expression + constraints
             时跑 symbolic_regression_tool action=constraint_check

        每项 best-effort: 数据不全/工具报错就跳过, 不写 key (math_checker 忽略缺失).
        """
        evidence: dict[str, Any] = {}
        if not isinstance(execution_result, dict):
            return evidence

        # 1. conservation_law — 从已有 math_validation 透传
        cons = math_validation.get("conservation")
        if isinstance(cons, dict) and "verified" in cons:
            evidence["conservation_law"] = {
                "verified": bool(cons["verified"]),
                "current": cons.get("message", ""),
                "symmetry": cons.get("method", ""),
            }

        from huginn.types import ToolContext

        tool_ctx = ToolContext(
            session_id=f"mathevid_{uuid.uuid4().hex[:8]}",
            workspace=str(self.workspace),
            config=self.settings,
        )

        # 2. dimensional_consistent — 跑量纲分析, 所有 quantity 都能解析 → True
        equation = (
            execution_result.get("equation")
            or execution_result.get("equations")
            or ""
        )
        if equation:
            try:
                from huginn.tools.symbolic_math.tool import (
                    SymbolicMathInput,
                    SymbolicMathTool,
                )

                tool = SymbolicMathTool()
                args = SymbolicMathInput(
                    action="dimensional_analysis",
                    expression=str(equation),
                    target="validate_expression",
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    quantities = vr.data.get("quantities", [])
                    has_error = any("error" in q for q in quantities)
                    evidence["dimensional_consistent"] = (
                        len(quantities) > 0 and not has_error
                    )
            except Exception:
                logger.warning("error in _collect_math_evidence: dimensional_analysis failed", exc_info=True)

        # 3. pde_classification — 跑 pde_classify, 比对 expected vs actual
        pde_coeffs = execution_result.get("pde_coefficients")
        expected_class = execution_result.get("expected_pde_class")
        if pde_coeffs and expected_class:
            try:
                from huginn.tools.symbolic_math.tool import (
                    SymbolicMathInput,
                    SymbolicMathTool,
                )

                tool = SymbolicMathTool()
                args = SymbolicMathInput(
                    action="pde_classify",
                    expression=str(pde_coeffs),
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    actual = vr.data.get("classification", "")
                    evidence["pde_classification"] = {
                        "consistent": actual.lower() == str(expected_class).lower(),
                        "expected": str(expected_class),
                        "actual": actual,
                    }
            except Exception:
                logger.warning("error in _collect_math_evidence: pde_classify failed", exc_info=True)

        # 4. sobol_top_features — 跑 sobol_indices, top features (S_i>0.1) 必须
        # 被 hypothesis_features 覆盖
        sobol_data = execution_result.get("sobol_data")
        hypothesis_features = execution_result.get("hypothesis_features")
        if sobol_data and hypothesis_features:
            try:
                from huginn.tools.sci.symbolic_regression_tool import (
                    SymbolicRegressionInput,
                    SymbolicRegressionTool,
                )

                tool = SymbolicRegressionTool()
                target_col = (
                    sobol_data.get("target", "y")
                    if isinstance(sobol_data, dict)
                    else "y"
                )
                args = SymbolicRegressionInput(
                    action="sobol_indices",
                    data_json=sobol_data,
                    target_column=target_col,
                    n_sobol_samples=512,
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    first_order = vr.data.get("first_order", {})
                    if first_order:
                        top = [f for f, s in first_order.items() if s > 0.1]
                        evidence["sobol_top_features"] = {
                            "hypothesis_covers_top": set(top).issubset(
                                set(hypothesis_features)
                            ),
                            "top_features": top,
                            "hypothesis_features": list(hypothesis_features),
                        }
            except Exception:
                logger.warning("error in _collect_math_evidence: sobol_indices failed", exc_info=True)

        # 5. constraint_check — 跑 constraint_check, 所有先验通过 → all_passed
        expr = execution_result.get("expression")
        constraints = execution_result.get("constraints")
        if expr and constraints:
            try:
                from huginn.tools.sci.symbolic_regression_tool import (
                    SymbolicRegressionInput,
                    SymbolicRegressionTool,
                )

                tool = SymbolicRegressionTool()
                args = SymbolicRegressionInput(
                    action="constraint_check",
                    probe_expression=str(expr),
                    constraints=constraints,
                )
                vr = await tool.call(args, tool_ctx)
                if vr.success and vr.data:
                    evidence["constraint_check"] = {
                        "all_passed": vr.data.get("all_passed", False),
                        "violations": vr.data.get("violations", []),
                    }
            except Exception:
                logger.warning("error in _collect_math_evidence: constraint_check failed", exc_info=True)

        return evidence

    def _query_kb_reference(self, equations: str, lagrangian: str) -> list[dict]:
        """查 KB 拿 first-principles 参考块. 把 equations + lagrangian 拼成
        query 串, 命中返回 [{text, source}], 失败/空都返回 []."""
        query = " ".join(filter(None, [equations, lagrangian])).strip()
        if not query:
            return []
        kb = self._get_kb()
        if kb is None:
            return []
        try:
            if kb.count() == 0:
                return []
            chunks = kb.query(f"conservation law variational {query}", top_k=2)
            return [
                {"text": (c.get("text") or "")[:300], "source": c.get("source", "")}
                for c in chunks
                if c.get("text")
            ]
        except Exception:
            return []

    @staticmethod
    def _build_reviewer_prompt(
        execution_result: Any,
        results: dict[str, Any],
        kb_text: str = "",
    ) -> str:
        """构造让 reviewer persona 点评执行结果的 prompt."""
        try:
            exec_blob = json.dumps(execution_result, ensure_ascii=False, default=str)[:1500]
        except Exception:
            exec_blob = str(execution_result)[:1500]
        try:
            res_blob = json.dumps(results, ensure_ascii=False, default=str)[:1500]
        except Exception:
            res_blob = str(results)[:1500]
        kb_section = f"\n{kb_text}\n" if kb_text else ""
        return (
            "Below is the execution result and validation summary from an "
            "autonomous materials-science research loop iteration.\n\n"
            f"Execution result:\n{exec_blob}\n\n"
            f"Validation summary:\n{res_blob}\n"
            f"{kb_section}"
            "As a critical peer reviewer, point out:\n"
            "1. Any methodological weakness or missing convergence check.\n"
            "2. Whether the result is reproducible and benchmarked.\n"
            "3. Whether the result aligns with the domain knowledge context above "
            "(if any), or contradicts known first-principles.\n"
            "4. Concrete next-step improvements.\n"
            "Be concise and direct."
        )

    async def _learn(self, hypothesis: str, plan: dict[str, Any], validation: dict[str, Any]) -> None:
        """Learn from iteration results — update memory, knowledge graph, evolution rules."""
        r_phys = validation.get("r_phys") if isinstance(validation, dict) else None

        # Log to memory
        self.memory.add_message(
            "system",
            {
                "iteration": self._iteration,
                "hypothesis": hypothesis,
                "plan": plan,
                "validation": validation,
                "r_phys": r_phys,
            },
        )

        # Long-term memory: 把关键迭代写入 long-term, 下次 RAG 能检索到
        # 包含 visual primitives 和 surprise 分数, 跨会话完整恢复上下文
        try:
            mem_content = f"iter {self._iteration}: {hypothesis[:120]}"
            # Visual primitives 入 memory, 下次 recall_for_prompt 能检索到数据形状
            visual_ctx = validation.get("visual_primitives") if isinstance(validation, dict) else None
            if visual_ctx:
                mem_content += f"\nVisual: {visual_ctx[:200]}"
            # Surprise 入 memory, 下次能检索到"这类任务预测准不准"
            pred_err = validation.get("prediction_error", {}) if isinstance(validation, dict) else {}
            if pred_err:
                mem_content += f"\nSurprise: {pred_err.get('surprise', 0)} (predicted: {pred_err.get('predicted', '')[:80]})"
            self.memory.remember(
                content=mem_content,
                category="autoloop_iteration",
                importance=0.6 if r_phys is None else min(0.9, float(r_phys)),
                tier="mid",
            )
        except Exception:
            logger.warning("error in _learn: memory.remember iteration failed", exc_info=True)

        # 奖励回流: 把 R_phys 喂给 evolution engine, 驱动基于奖励的进化
        # 这是阶段4 单轨的核心闭环——物理校验分数真正影响 agent 后续行为
        if r_phys is not None:
            try:
                evolution = self._get_evolution()
                # 记录本次迭代的 reward, 供 evolve_from_rewards 消费
                evolution.logger.log_tool_call(
                    session_id=f"loop_{self._iteration}",
                    tool_name=plan.get("mode", "unknown"),
                    tool_input={"hypothesis": hypothesis, "plan": plan},
                    result=validation,
                    reward=r_phys,
                )
                reward_result = evolution.evolve_from_rewards()
                n_skills = len(reward_result["high_reward_skills"])
                n_patches = len(reward_result["low_reward_patches"])
                if n_skills or n_patches:
                    logger.info(
                        "reward evolution: +%d skills, +%d patches (R_phys=%.2f)",
                        n_skills, n_patches, r_phys,
                    )
            except Exception as e:
                logger.warning("reward evolution failed: %s", e)

        # KB 回写: 把本次实验结论存入知识库, 下次同类问题能从 KB 召回.
        # 不存原始数据 (太大), 只存 hypothesis + validation 摘要.
        # JEPA: 预测误差也写入, 下次同类任务能从 KB 检索到"这类任务
        # agent 的预测准不准", 帮助判断是否需要更多探索.
        try:
            kb = self._get_kb()
            if kb:
                pred_err = validation.get("prediction_error", {})
                surprise_line = ""
                if pred_err:
                    surprise_line = f"\nPrediction surprise: {pred_err.get('surprise', 0)}\nPredicted: {pred_err.get('predicted', '')[:100]}\nActual: {pred_err.get('actual', '')[:100]}"
                summary_text = (
                    f"Iteration {self._iteration}: {hypothesis[:200]}\n"
                    f"Mode: {plan.get('mode', 'unknown')}\n"
                    f"R_phys: {r_phys}\n"
                    f"Validation: {json.dumps(validation, default=str)[:500]}"
                    f"{surprise_line}"
                )
                kb.add_document(
                    filename=f"autoloop_iter_{self._iteration}.txt",
                    content=summary_text.encode("utf-8"),
                )
        except Exception:
            logger.warning("error in _learn: KB writeback failed", exc_info=True)

        # KG 回写: 把 hypothesis 作为 experiment 实体加入知识图,
        # 让 ProjectKnowledgeGraph 随实验增长而非只读展示.
        # 视觉基元 + surprise 都写入实体属性, 下次 KG 查询能检索到.
        try:
            kg_attrs: dict[str, Any] = {
                "iteration": self._iteration,
                "r_phys": r_phys,
            }
            visual_ctx = validation.get("visual_primitives") if isinstance(validation, dict) else None
            if visual_ctx:
                kg_attrs["visual_primitives"] = visual_ctx[:500]
            # JEPA: surprise 分数存入 KG, 下次查同类实验能看到"这类任务
            # agent 预测准不准", 帮助判断是否值得继续探索.
            pred_err = validation.get("prediction_error", {}) if isinstance(validation, dict) else {}
            if pred_err:
                kg_attrs["surprise"] = pred_err.get("surprise", 0)
                kg_attrs["predicted"] = pred_err.get("predicted", "")[:200]
            exp_id = self.kg.add_entity(
                label=hypothesis[:80],
                entity_type="experiment",
                source="autoloop",
                confidence=float(r_phys) if r_phys is not None else 0.5,
                **kg_attrs,
            )
            # Hyperedge: 把 hypothesis → plan_mode → validation 结果
            # 连成 n-ary 关系. 之前 add_hyperedge 是死代码, 现在接上.
            plan_id = self.kg.add_entity(
                label=f"plan_{plan.get('mode', 'unknown')}_iter{self._iteration}",
                entity_type="Method",
                source="autoloop",
            )
            result_label = "pass" if (validation.get("tests_passed") if isinstance(validation, dict) else False) else "fail"
            result_id = self.kg.add_entity(
                label=f"{result_label}_iter{self._iteration}",
                entity_type="Fact",
                source="autoloop",
                surprise=pred_err.get("surprise", 0) if pred_err else 0,
            )
            if exp_id and plan_id and result_id:
                self.kg.add_hyperedge(
                    [exp_id, plan_id, result_id],
                    relation="experiment_pipeline",
                    source="autoloop",
                    iteration=self._iteration,
                )
            self.kg.save()
        except Exception:
            logger.warning("error in _learn: KG add_entity failed", exc_info=True)

        # Benchmark 失败回写: 把验证失败写入 memory, 下次 _plan 能读到.
        if isinstance(validation, dict) and not validation.get("tests_passed", True):
            try:
                self.memory.remember(
                    content=(
                        f"Validation failure iter {self._iteration}: "
                        f"{json.dumps(validation, default=str)[:400]}"
                    ),
                    category="benchmark_failure",
                    tags=["autoloop", "validation"],
                    importance=0.7,
                    tier="mid",
                )
            except Exception:
                logger.warning("error in _learn: benchmark_failure memory writeback failed", exc_info=True)

        # 把 plan 进度存进 long-term memory, 下次会话能接续
        _plan_id = plan.get("plan_id") if isinstance(plan, dict) else None
        if _plan_id:
            try:
                store = self._get_plan_store()
                if store is not None:
                    persisted = store.get_plan(_plan_id)
                    if persisted is not None:
                        self.memory.store_plan_progress(
                            plan_id=persisted.id,
                            objective=persisted.objective,
                            step_index=len(
                                [s for s in persisted.steps if s.status == "done"]
                            ),
                            status=persisted.status,
                            l1_coordinates=f"autoloop: {persisted.objective[:100]}",
                        )
            except Exception:
                logger.warning("error in _learn: store_plan_progress writeback failed", exc_info=True)

    async def _report(self, objective: str, phases: list[LoopPhase], total_time: float) -> str | None:
        """Generate a final report summarizing the loop."""
        report_data = {
            "objective": objective,
            "run_id": f"loop_{uuid.uuid4().hex[:8]}",
            "total_time_seconds": total_time,
            "phases": [
                {
                    "name": p.name,
                    "status": p.status,
                    "duration": (p.end_time or 0) - (p.start_time or 0) if p.start_time and p.end_time else 0,
                    "error": p.error,
                }
                for p in phases
            ],
        }

        # Report 阶段接入 tutor persona: 让 LLM 用教学口吻写一段总结,
        # 帮助用户理解这轮 loop 做了什么、为什么这么做. 失败就退化为纯表格报告.
        # A4: 同时查 KB 拿 first-principles 文献块, 拼进 tutor prompt 让总结
        # 引用已知理论, 报告里会带 "Domain Knowledge References" 段落.
        kb_text = self._build_kb_text(query=objective)
        tutor_narrative = ""
        try:
            tutor_narrative = await self._llm_chat(
                self._build_tutor_report_prompt(report_data, kb_text),
                persona_name="tutor",
                task="summarize",  # ponytail: report 用便宜模型, 不需要强推理
            )
            tutor_narrative = (tutor_narrative or "").strip()
        except Exception:
            tutor_narrative = ""

        # Save markdown report to workspace
        report_path = self.workspace / f"huginn_autoloop_report_{report_data['run_id']}.md"
        report_content = self._render_report(report_data)
        if kb_text:
            report_content += "\n\n## Domain Knowledge References\n\n" + kb_text + "\n"
        if tutor_narrative:
            report_content += "\n\n## Tutor's Summary\n\n" + tutor_narrative + "\n"
        report_path.write_text(report_content, encoding="utf-8")

        return str(report_path)

    @staticmethod
    def _build_tutor_report_prompt(
        report_data: dict[str, Any], kb_text: str = ""
    ) -> str:
        """构造让 tutor persona 写教学口吻总结的 prompt."""
        try:
            phases_blob = json.dumps(report_data["phases"], ensure_ascii=False)[:1200]
        except Exception:
            phases_blob = str(report_data.get("phases", ""))[:1200]
        kb_section = f"\n{kb_text}\n" if kb_text else ""
        return (
            "You just supervised an autonomous research loop. Summarize for a "
            "graduate student what happened, in a patient, pedagogical tone.\n\n"
            f"Objective: {report_data['objective']}\n"
            f"Total time: {report_data['total_time_seconds']:.1f}s\n"
            f"Phases:\n{phases_blob}\n"
            f"{kb_section}"
            "Cover:\n"
            "- What the loop tried to achieve and why each phase matters.\n"
            "- Any phase that failed, and what a student should learn from it.\n"
            "- How the result relates to the domain knowledge context above "
            "(if any), citing source numbers when relevant.\n"
            "- One concrete suggestion for the next iteration.\n"
            "Keep it under 200 words."
        )

    # ──────────────────────────────────────────────────────────────
    # Execution helpers
    # ──────────────────────────────────────────────────────────────

    async def _execute_coder(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a coder task."""
        # Build a coding prompt from the description and context
        prompt = f"""Task: {description}

Context:
- Changed files: {context.get('changed_files', [])}
- Git diff: {context.get('git_diff', '')[:500]}

Please modify the code to address this task."""

        # Run CoderRunner
        # (Simplified — in production this would use the full CoderRunner loop)
        return {"mode": "coder", "prompt": prompt, "status": "submitted"}

    async def _execute_workflow(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute a workflow task."""
        # For now, use a standard DFT workflow as example
        # In production, dynamically select workflow template based on description
        try:
            # Find structure files in workspace
            structure_files = list(self.workspace.rglob("*.cif")) + list(self.workspace.rglob("*.poscar")) + list(self.workspace.rglob("*.vasp"))
            structure_path = str(structure_files[0]) if structure_files else "structure.cif"

            stages = standard_dft_workflow(structure_path, engine="vasp")
            tool_context = ToolContext(
                session_id=f"workflow_{uuid.uuid4().hex[:8]}",
                workspace=str(self.workspace),
                config=self.settings,
            )
            result = await self.workflow_engine.execute(stages, tool_context)
            return {"mode": "workflow", "success": result.success, "stages": len(stages)}
        except Exception as e:
            return {"mode": "workflow", "success": False, "error": str(e)}

    async def _execute_explore(self, description: str, context: dict[str, Any]) -> dict[str, Any]:
        """Execute an exploration task."""
        try:
            result = await self.explorer.explore(
                objective=description,
                initial_branches=[
                    {"name": "baseline", "hypothesis": f"Baseline for: {description}"}
                ],
                max_iterations=5,
            )
            return {
                "mode": "explore",
                "n_explored": result.n_branches_explored,
                "n_pruned": result.n_branches_pruned,
                "convergence": result.convergence_reason,
            }
        except Exception as e:
            return {"mode": "explore", "success": False, "error": str(e)}

    async def _execute_skill(self, plan: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Run a pre-built composite skill pipeline."""
        try:
            from huginn.skills.registry import SkillRegistry
            from huginn.skills.base import DeclarativeSkillExecutor
            from huginn.skills.composite import _ensure_registered
            _ensure_registered()

            skill_name = plan.get("skill", "")
            skill = SkillRegistry.get(skill_name)
            if not skill:
                # Fuzzy match if exact name missing
                matches = SkillRegistry.search(skill_name or plan.get("description", ""))
                skill = matches[0] if matches else None
            if not skill:
                return {"mode": "skill", "success": False,
                        "error": f"no matching skill for '{skill_name}'"}

            # Reuse the same tool registry as the rest of the engine
            from huginn.tools.registry import ToolRegistry
            executor = DeclarativeSkillExecutor(ToolRegistry)
            result = await executor.execute(skill, {}, context)
            return {"mode": "skill", "skill": skill.name, **result}
        except Exception as e:
            return {"mode": "skill", "success": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────────
    # LLM helpers
    # ──────────────────────────────────────────────────────────────

    async def _llm_chat(
        self,
        prompt: str,
        persona_name: str | None = None,
        model: Any = None,
        task: str | None = None,
    ) -> str:
        """Send a prompt to the LLM and return the response.

        persona_name 不为空时, 把对应 persona 的 system prompt 作为
        SystemMessage 插在最前, 实现"每阶段开始注入 persona system prompt".
        persona 找不到就退化为不注入, 行为跟改动前一致.

        model 不为空时用传入的模型 (用于三槽 verification), 否则用默认 self.model.

        task 不为空时, 优先从 model_router 路由 (team 模式):
        - "reasoning"/"science" → 强模型 (云端)
        - "summarize"/"format" → 便宜模型 (本地/小模型)
        - "verification" → 独立验证模型
        model 参数优先于 task — 显式指定的模型不被路由覆盖.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        # Team 模式: task 路由优先, 但显式 model 不被覆盖
        if model is None and task is not None:
            router = getattr(self, 'model_router', None)
            if router is not None:
                try:
                    routed = router.select(task, prefer_cheap=(task in ("summarize", "format", "archival")))
                    if routed is not None:
                        model = routed
                except Exception:
                    pass

        llm = model or self.model
        messages: list[Any] = []
        if persona_name:
            sys_prompt = self._persona_system_prompt(persona_name)
            if sys_prompt:
                sys_msg = SystemMessage(content=sys_prompt)
                # 静态 system prompt 跨调用不变, 给 Anthropic/Kimi 打 cache 标记
                _ident = f"{type(llm).__name__}{getattr(llm, 'model', '')}".lower()
                if any(k in _ident for k in ("anthropic", "claude", "kimi", "moonshot")):
                    sys_msg.additional_kwargs["cache_control"] = {"type": "ephemeral"}
                messages.append(sys_msg)
        messages.append(HumanMessage(content=prompt))
        response = await llm.ainvoke(messages)
        return str(response.content)

    def _build_hypothesis_prompt(self, context: dict[str, Any]) -> str:
        # 投机执行 hint: 基于历史预测的下一步意图, 注入给 LLM 参考
        # 预测只是 hint, LLM 可以无视, 不强制. 截断到 500 字符防止无界增长
        # — _speculator_hint 有 5 处 append, 不截断 20 轮后可能数 KB.
        hint_block = ""
        if self._speculator_hint:
            hint_block = f"\nSpeculator hint (advisory, may be ignored): {self._speculator_hint[:500]}\n"
        # 三路检索共用一个 query, 避免重复序列化 context 3 次
        ctx_query = json.dumps(context, ensure_ascii=False)[:500]
        # 领域知识库检索: 命中 first-principles 参考块就拼进 prompt
        kb_block = self._build_kb_text(query=ctx_query)
        if kb_block:
            kb_block = f"\n{kb_block}\n"
        # 知识图谱检索: 把之前 run 发现的实体和关系拉回来, 避免
        # 重复发现已有结论, 也让假设能建立在已有发现上
        kg_block = self._build_kg_text(query=ctx_query)
        if kg_block:
            kg_block = f"\n{kg_block}\n"
        # 长期记忆检索: 跨会话的失败教训和发现. 之前只写不读,
        # 现在闭合 — _learn 写入的迭代记录下次能检索到.
        mem_block = self._build_memory_text(query=ctx_query)
        if mem_block:
            mem_block = f"\n{mem_block}\n"
        # 视觉基元: 上一轮 tool 输出的数值指针 (峰值/趋势/异常),
        # 给 LLM 具体坐标锚定推理 — Thinking with Visual Primitives 的
        # "point while it reasons" 原则, Mirage 效应的文本路径
        visual_block = getattr(self, '_last_visual_context', '')
        if visual_block:
            visual_block = f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
        # 数学深度引导: 提醒 agent 优先识别 PDE / 变分原理 / 微分几何结构,
        # 并用符号回归 + Sobol 灵敏度 + 物理约束先验 反复试探.
        # 条件化: 只在 context 含数学信号时注入, coder-only 任务不需要.
        # 节省 ~150 tokens × 2 calls/iter × 20 iters = 6K tokens/run.
        ctx_blob = json.dumps(context, ensure_ascii=False).lower()
        _math_signals = ("equation", "lagrangian", "pde", "hamiltonian", "derivative",
                         "differential", "integral", "eigenvalue", "tensor", "manifold",
                         "symmetry", "conservation", "variational", "continuum",
                         "stress", "strain", "energy", "phonon", "band")
        math_block = self._MATH_DEPTH_PROMPT_BLOCK if any(s in ctx_blob for s in _math_signals) else ""
        # 按优先级拼接, 超预算自动裁剪低优先级 block
        return self._trim_to_budget([
            ("body", f"""You are an autonomous material science research agent.

Perceived context:
{json.dumps(context, indent=2, ensure_ascii=False)[:2000]}

Generate a single, testable hypothesis about what should be done next.
The hypothesis should be a single sentence, concrete and actionable.
Ground it in the domain knowledge context above when relevant.
Prefer hypotheses that can be expressed as governing PDEs, variational
principles, or conservation laws; identify the mathematical structure
before proposing numerical experiments.

Hypothesis:"""),
            ("math", math_block),
            ("kg", kg_block),
            ("visual", visual_block),
            ("kb", kb_block),
            ("mem", mem_block),
            ("hint", hint_block),
        ])

    # 数学深度提示块: 在 hypothesis / plan prompt 里持续提醒 agent 用
    # 符号数学工具把"现象"翻译成"PDE / 变分 / 几何 / 灵敏度"语言.
    # 用户偏好: 物理、化学本质上是数学的一部分 — 这里把那条原则落进 prompt.
    _MATH_DEPTH_PROMPT_BLOCK = """
Math depth guidance (treat physics/chemistry as mathematics):
- Identify the governing PDE: use symbolic_math_tool action=pde_classify
  (A;B;C discriminant) to classify elliptic/parabolic/hyperbolic, then
  pde_separation or pde_characteristics for analytic structure.
- If the phenomenon extremizes a functional, derive Euler-Lagrange:
  symbolic_math_tool action=euler_lagrange or action=derive (alias).
  Check symmetries with action=noether to predict conserved currents.
- For curved manifolds (defects, interfaces, crystal plasticity), compute
  Christoffel/Ricci via action=diffgeo_metric or diffgeo_curvature.
- Before fitting data, run symbolic_regression_tool action=sobol_indices
  to rank feature importance, then discover expressions with
  action=discover and validate candidates with action=constraint_check
  (positivity / monotonicity / finiteness priors).
"""

    def _build_plan_prompt(self, hypothesis: str, context: dict[str, Any]) -> str:
        # 同 hypothesize: 用 hypothesis 串检索 KB, 把参考块喂给 planner
        kb_block = self._build_kb_text(query=hypothesis)
        if kb_block:
            kb_block = f"\n{kb_block}\n"
        # KG 检索: 用 hypothesis 当 query, 看看已有实体里有没有相关的
        kg_block = self._build_kg_text(query=hypothesis)
        if kg_block:
            kg_block = f"\n{kg_block}\n"
        # 长期记忆检索 (同 hypothesize)
        mem_block = self._build_memory_text(query=hypothesis)
        if mem_block:
            mem_block = f"\n{mem_block}\n"
        # 视觉基元注入 (同 hypothesize)
        visual_block = getattr(self, '_last_visual_context', '')
        if visual_block:
            visual_block = f"\n### Visual Primitives (from last tool output)\n{visual_block}\n"
        # 条件化 math_block (同 hypothesize)
        hyp_blob = hypothesis.lower() + json.dumps(context, ensure_ascii=False).lower()[:500]
        math_block = self._MATH_DEPTH_PROMPT_BLOCK if any(s in hyp_blob for s in _math_signals) else ""

        # Inject learned skills + prompt patches from evolution engine.
        # This is the "use what you learned" half of the Learn→Plan loop.
        skill_hints = ""
        patch_hints = ""
        try:
            evolution = self._get_evolution()
            skills = evolution.get_relevant_skills(hypothesis)
            if skills:
                skill_lines = [f"  - {s.name}: {s.description}" for s in skills[:3]]
                skill_hints = "\nLearned skills (from past iterations):\n" + "\n".join(skill_lines) + "\n"
            patches = evolution.get_prompt_patches()
            if patches:
                patch_hints = "\nLearned patches:\n" + "\n".join(f"  - {p}" for p in patches[:3]) + "\n"
        except Exception:
            logger.warning("error in _build_plan_prompt: evolution skill/patch fetch failed", exc_info=True)

        # Inject matching composite skills — lets the LLM pick a pre-built
        # multi-tool pipeline instead of improvising from scratch.
        # 条件化: 只在 hypothesis 涉及仿真/计算/材料性质时注入, coder-only
        # 任务不需要 composite skill 列表. 节省 ~500 tokens.
        composite_block = ""
        hyp_lower = hypothesis.lower()
        _workflow_signals = ("workflow", "simulation", "band", "dos", "phonon",
                              "mechanical", "thermal", "optical", "dft", "vasp",
                              "lammps", "md ", "structure", "property", "energy",
                              "convergence", "optimize", "calc")
        if any(s in hyp_lower for s in _workflow_signals):
            try:
                from huginn.skills.registry import SkillRegistry
                from huginn.skills.composite import _ensure_registered
                _ensure_registered()
                matches = SkillRegistry.search(hypothesis)
                if not matches:
                    matches = SkillRegistry.get_all_definitions()
                if matches:
                    lines = [s.to_prompt() for s in matches[:4]]
                    composite_block = "\nAvailable composite skills (prefer these over manual workflow):\n" + "\n\n".join(lines) + "\n"
            except Exception:
                logger.debug("composite skill lookup failed", exc_info=True)

        # Pipeline 建议: 基于 provenance 规则推荐下一步工具.
        # 42 条领域规则, 零 LLM 调用. 让 plan 知道"这类任务通常下一步是 X".
        pipeline_block = ""
        try:
            from huginn.provenance.pipeline import SimulationPipeline
            pipeline = SimulationPipeline(self.kg.root if hasattr(self.kg, 'root') else None)
            # 用上一轮的 execution_result 触发 suggest_next
            last_result = getattr(self, '_last_execution_result', None)
            if last_result and isinstance(last_result, dict):
                tool_name = last_result.get('_tool_name', '')
                suggestions = pipeline.suggest_next(
                    tool_name=tool_name,
                    tool_input=last_result.get('_tool_input', {}),
                    tool_output=last_result.get('result', last_result),
                )
                if suggestions:
                    s_lines = [f"  - {s.tool_hint}: {s.description}" for s in suggestions[:3]]
                    pipeline_block = "\nPipeline suggestions (based on provenance):\n" + "\n".join(s_lines) + "\n"
        except Exception:
            pass  # pipeline 是 advisory, 失败不阻塞

        return self._trim_to_budget([
            ("body", f"""Given the hypothesis: "{hypothesis}"

Context:
{json.dumps(context, indent=2, ensure_ascii=False)[:1000]}

Choose ONE mode and describe the plan:
- coder: modify code/files to fix or improve something
- workflow: run a computational simulation pipeline
- explore: search a design space for optimal parameters
- skill: use a pre-built composite skill pipeline (band structure, mechanical properties, MD, etc.)

When the hypothesis involves a PDE / variational principle / curved
geometry, prefer the symbolic_math_tool actions listed in the math
depth block above before falling back to numerical solvers.

Respond in this exact format:
MODE: <coder|workflow|explore|skill>
DESCRIPTION: <brief description of what to do>
SKILL: <composite skill name, only if MODE is skill>
PREDICTION: <what you expect the result to look like — be specific: "energy ~ -X eV", "converges in ~N steps", "band gap ~X eV". This prediction will be compared against actual results to measure surprise.>
"""),
            ("math", math_block),
            ("kg", kg_block),
            ("visual", visual_block),
            ("kb", kb_block),
            ("mem", mem_block),
            ("skill", skill_hints + patch_hints),
            ("composite", composite_block),
            ("pipeline", pipeline_block),
        ])

    def _parse_plan(self, response: str) -> dict[str, Any]:
        """Parse LLM plan response."""
        mode = "coder"
        description = response.strip()
        skill_name = ""
        prediction = ""

        for line in response.split("\n"):
            if line.startswith("MODE:"):
                mode = line.replace("MODE:", "").strip().lower()
            elif line.startswith("DESCRIPTION:"):
                description = line.replace("DESCRIPTION:", "").strip()
            elif line.startswith("SKILL:"):
                skill_name = line.replace("SKILL:", "").strip()
            elif line.startswith("PREDICTION:"):
                prediction = line.replace("PREDICTION:", "").strip()

        plan = {"mode": mode, "description": description}
        if skill_name:
            plan["skill"] = skill_name
        if prediction:
            plan["expected_prediction"] = prediction
        return plan

    # ──────────────────────────────────────────────────────────────
    # Phase runner utilities
    # ──────────────────────────────────────────────────────────────

    def _run_phase(self, name: str, fn, *args) -> LoopPhase:
        """Run a synchronous phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        # 同步路径: 如果当前在 event loop 里, fire-and-forget 发开始事件.
        # 不 await, 因为 _run_phase 本身是同步的.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._dispatch_stage_event(
                    EventType.ON_WORKFLOW_STAGE_START, name
                )
            )
        except RuntimeError:
            logger.warning("error in _run_phase: stage-start event dispatch skipped (no running loop)", exc_info=True)
        # 包 telemetry span: 把 phase 级决策也记进轨迹, 回放时不止看 tool_call
        from huginn.telemetry import get_telemetry_collector
        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning("error in _run_phase: span metadata update failed", exc_info=True)
        phase.end_time = time.time()
        # fire-and-forget 发结束/失败事件
        try:
            loop = asyncio.get_running_loop()
            done_type = (
                EventType.ON_WORKFLOW_STAGE_DONE
                if phase.status == "completed"
                else EventType.ON_WORKFLOW_FAILED
            )
            loop.create_task(
                self._dispatch_stage_event(
                    done_type,
                    name,
                    duration_sec=phase.end_time - (phase.start_time or 0),
                    error=phase.error,
                )
            )
        except RuntimeError:
            logger.warning("error in _run_phase: stage-done event dispatch skipped (no running loop)", exc_info=True)
        return phase

    async def _run_phase_async(self, name: str, fn, *args) -> LoopPhase:
        """Run an async phase function."""
        phase = LoopPhase(name=name)
        phase.start_time = time.time()
        phase.status = "running"
        await self._dispatch_stage_event(
            EventType.ON_WORKFLOW_STAGE_START, name
        )
        from huginn.telemetry import get_telemetry_collector
        span_cm = get_telemetry_collector().span(f"phase:{name}")
        try:
            with span_cm as phase_span:
                phase.result = await fn(*args)
                phase.status = "completed"
                phase_span.metadata["status"] = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
            try:
                phase_span.metadata["status"] = "failed"
                phase_span.metadata["error"] = str(e)
            except Exception:
                logger.warning("error in _run_phase_async: span metadata update failed", exc_info=True)
        phase.end_time = time.time()
        if phase.status == "completed":
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_STAGE_DONE,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
            )
        else:
            await self._dispatch_stage_event(
                EventType.ON_WORKFLOW_FAILED,
                name,
                duration_sec=phase.end_time - (phase.start_time or 0),
                error=phase.error,
            )
        return phase

    # ──────────────────────────────────────────────────────────────
    # Report rendering
    # ──────────────────────────────────────────────────────────────

    def _render_report(self, data: dict[str, Any]) -> str:
        """Render a markdown report."""
        lines = [
            f"# Huginn Autoloop Report",
            f"",
            f"**Objective:** {data['objective']}",
            f"**Run ID:** {data['run_id']}",
            f"**Total Time:** {data['total_time_seconds']:.1f}s",
            f"",
            f"## Phases",
            f"",
            f"| Phase | Status | Duration (s) | Error |",
            f"|-------|--------|--------------|-------|",
        ]
        for p in data["phases"]:
            lines.append(f"| {p['name']} | {p['status']} | {p['duration']:.1f} | {p['error'] or ''} |")
        lines.append("")
        lines.append("---")
        lines.append("Generated by Huginn Autoloop Engine")
        return "\n".join(lines)
