"""HuginnAgent — core class with init, factory, graph, and dispatch methods.

The heavy logic lives in the mixin modules: context.py, streaming.py,
reflection.py, session.py, callbacks.py.  This file wires them together
and keeps the construction + graph-building + convenience methods.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage

from huginn.agent_config import (
    _UNSET_SENTINEL,
    AgentConfig,
    resolve_tool_budget,
)
from huginn.agent_modes import LONG_HORIZON_MODES, SETTABLE_MODES, AgentMode
from huginn.checkpointer import create_in_memory_checkpointer
from huginn.cognitive_engine import STATE_TO_MODEL_TASK
from huginn.context_manager import (
    get_context_window,
    reset_context_cache,
)
from huginn.feature_flags import FeatureFlags
from huginn.hooks import HookManager
from huginn.models.router import ModelRouter
from huginn.permissions import PermissionConfig
from huginn.phases import PhaseManager, ResearchPhase
from huginn.prompts import HUGINN_BASE_PROMPT
from huginn.telemetry import NullTelemetryCollector, TelemetryCollector
from huginn.tools.adapter import ToolAdapter
from huginn.tools.registry import ToolRegistry
from huginn.utils.conversation_tree import ConversationTree
from huginn.utils.prompt_cache import PromptCacheBuilder
from huginn.utils.session_context import get_user_message

from .callbacks import CallbackMixin
from .context import ContextMixin
from .middlewares import (
    DeliverableCoverageMiddleware,
    FixDanglingToolCallsMiddleware,
    RateLimitMiddleware,
)
from .reflection import ReflectionMixin
from .session import SessionMixin
from .streaming import StreamingMixin

logger = logging.getLogger(__name__)


# 负结果提示要扫的指标名关键词 (小写匹配).
_WEAK_METRIC_TOKENS = ("auc", "accuracy", "r2", "r_squared", "f1")


def _weak_signal_hint(outputs_dir: Path) -> str | None:
    """扫 outputs/*.json 找 < 0.6 的指标, 组装 negative-result 提示; 无则 None."""
    weak: list[str] = []
    for jp in outputs_dir.glob("*.json"):
        try:
            payload = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:  # 防御: weak-metric 扫描失败则忽略
            logger.debug("weak-metric scan failed", exc_info=True)
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if not any(t in str(key).lower() for t in _WEAK_METRIC_TOKENS):
                continue
            try:
                fv = float(value)
            except (TypeError, ValueError):
                logger.debug("best-effort op failed", exc_info=True)
                continue
            if 0 <= fv < 0.6:
                weak.append(f"{key}={fv:.3f}")
    if not weak:
        return None
    return (
        "⚠️ WEAK SIGNAL DETECTED in outputs/: "
        + ", ".join(weak[:5])
        + ". If model performance is below target, write a 'Negative Results' "
        "section in report.md explaining WHY signal is weak (data quantity, "
        "feature design, model architecture, compute budget). Honest negative "
        "results with root-cause analysis score better than silence."
    )


def _coverage_hints(
    cov_mw: DeliverableCoverageMiddleware,
    inst_text: str,
    rpt_text: str | None,
) -> list[str]:
    """按 report 是否已产出, 生成 planning 或 frontier 覆盖率提示."""
    quantities = cov_mw._extract_quantities(inst_text)
    if not quantities:
        return []
    if rpt_text is None:
        return [cov_mw._build_planning_msg(quantities)]
    hints: list[str] = []
    missing = cov_mw._check_coverage(inst_text, rpt_text)
    gaps = cov_mw._check_layer_gaps(inst_text, rpt_text)
    if missing:
        hints.append(cov_mw._build_frontier_msg(missing))
    if gaps:
        hints.append(cov_mw._build_layer_frontier_msg(gaps))
    return hints


def _collect_benchmark_hints(cov_mw: DeliverableCoverageMiddleware) -> list[str]:
    """时间感知 / benchmark / 负结果 / 覆盖率 四层 hint (读 cwd 下的产物)."""
    extra: list[str] = []
    cwd = Path.cwd()
    rpt_path = cwd / "report" / "report.md"

    # Benchmark Mode: 通用 env var 注入, agent 不感知具体 benchmark 框架.
    bm_prompt = os.environ.get("HUGINN_BENCHMARK_MODE_PROMPT", "")
    if bm_prompt:
        extra.append(bm_prompt)
    # 方案 C: Negative Result — 还没写报告且已有 outputs/ 时提醒低分指标.
    if not rpt_path.exists() and (cwd / "outputs").exists():
        hint = _weak_signal_hint(cwd / "outputs")
        if hint:
            extra.append(hint)
    # DeliverableCoverage: benchmark 路径没走 deepagents middleware 协议,
    # 手动复用纯函数注入 frontier/planning hint.
    inst = cwd / "INSTRUCTIONS.md"
    if inst.exists():
        inst_text = inst.read_text(encoding="utf-8")
        rpt_text = rpt_path.read_text(encoding="utf-8") if rpt_path.exists() else None
        extra.extend(_coverage_hints(cov_mw, inst_text, rpt_text))
    return extra


def _bandit_effort_hint() -> str | None:
    """v18 bandit effort controller 的 advisory 提示; 失败静默返回 None."""
    try:
        from huginn.agent.bandit_controller import EffortBandit

        return EffortBandit.get_instance().build_hint() or None
    except Exception:  # 防御: bandit 提示注入失败则忽略
        logger.debug("bandit effort hint failed", exc_info=True)
        return None


def _make_pre_model_hook(fix_mw: Any, cov_mw: Any) -> Any:
    """langgraph pre_model_hook: 修悬空 tool_calls + 注入时基/覆盖率/bandit hint.

    langgraph 1.2+ 的 pre_model_hook 支持 llm_input_messages key — 只作 LLM 输入,
    不经 add_messages reducer, 顺序保留. 比 wrap model.bind_tools 干净: 不用
    fake Runnable, create_react_agent 的 model 类型校验直接过.
    """

    def _pre_model_hook(state: Any) -> dict[str, Any]:
        msgs = state.get("messages", []) or []
        patched = fix_mw._patch_messages(list(msgs))
        # 时间感知 + 覆盖率 + 负结果三层 hint, 失败静默不阻塞 LLM 调用.
        # ponytail: 每次读 INSTRUCTIONS+report+outputs (~2ms IO), 相比
        # LLM 调用 1-5s 零开销. 升级: langgraph config 传 deadline.
        extra_hints: list[str] = []
        bandit_hint: str | None = None
        try:
            extra_hints = _collect_benchmark_hints(cov_mw)
            # v18: bandit effort controller hint (advisory).
            bandit_hint = _bandit_effort_hint()
        except Exception:  # 防御: pre-model 提示注入失败则忽略
            logger.debug("pre-model hint injection failed", exc_info=True)
        if extra_hints:
            patched = [SystemMessage(content="\n\n".join(extra_hints))] + patched
        if bandit_hint:
            patched = [SystemMessage(content=bandit_hint)] + patched
        return {"llm_input_messages": patched}

    return _pre_model_hook


def _unpack_agent_config(config: AgentConfig) -> tuple[Any, ...]:
    """按原顺序拆出 AgentConfig 的九个子配置段."""
    return (
        config.model,
        config.tools,
        config.memory,
        config.security,
        config.telemetry,
        config.context_budget,
        config.knowledge_graph,
        config.personalization,
        config.core,
    )


class HuginnAgent(
    ContextMixin,
    CallbackMixin,
    SessionMixin,
    ReflectionMixin,
    StreamingMixin,
):
    """Material Science specialized Agent.

    Wraps EvoScientist's model and infrastructure with huginn-specific
    tools, prompts, exploration engine, memory, and skills.
    """

    def __init__(
        self,
        model: Any | None = None,
        tools: list[Any] | None = None,
        system_prompt: str | None = None,
        enable_exploration: Any = _UNSET_SENTINEL,
        memory_manager: Any | None = None,
        skill_executor: Any | None = None,
        sandbox: Any | None = None,
        audit: Any | None = None,
        profile_id: Any = _UNSET_SENTINEL,
        thread_id: str | None = None,
        tool_filter: list[str] | None = None,
        agent_factory: Any | None = None,
        privacy_redact_secrets: bool | None = None,
        privacy_block_on_secrets: bool | None = None,
        max_tool_output_tokens: int | None = None,
        context_budget_tokens: int | None = None,
        begin_dialogs: list[tuple[str, str]] | None = None,
        prompt_cache_control: bool | None = None,
        model_router: ModelRouter | None = None,
        checkpointer: Any | None = None,
        checkpointer_path: str | None = None,
        memory_decay_enabled: bool | None = None,
        memory_decay_interval_turns: int | None = None,
        memory_decay_prune_threshold: float | None = None,
        workspace: Any = _UNSET_SENTINEL,
        kg_enabled: Any = _UNSET_SENTINEL,
        kg_depth: Any = _UNSET_SENTINEL,
        kg_top_k: Any = _UNSET_SENTINEL,
        kb_enabled: Any = _UNSET_SENTINEL,
        auto_approve: bool | None = None,
        compression_max_tokens: int | None = None,
        telemetry_enabled: bool | None = None,
        persona_name: str | None = None,
        emotion_tracker: Any | None = None,
        approval_callback: Callable[[str, str], bool] | None = None,
        break_after_tool: Any = _UNSET_SENTINEL,
        max_tool_calls: int | None = None,
        max_tool_calls_per_tool: int | None = None,
        style_learner: Any | None = None,
        scheduler: Any | None = None,
        *,
        config: AgentConfig | None = None,
    ):
        if config is None:
            config = AgentConfig.from_env()

        # 参数表本身就是覆盖集合 (45 个关键字逐个手抄 = 长函数); 具体应用见 _init_agent.
        self._init_agent(config, locals())

    def _init_agent(self, config: AgentConfig, p: dict[str, Any]) -> None:
        """装配入口: 应用显式覆盖 → _init_from_config."""
        self._apply_model_tool_memory_overrides(config, p)
        self._apply_security_persona_overrides(config, p)
        self._apply_core_overrides(config, p)
        self._init_from_config(
            config, scheduler=p["scheduler"], kb_enabled=p["kb_enabled"]
        )

    @staticmethod
    def _apply_overrides(target: Any, values: dict[str, Any]) -> None:
        """把非 None 的显式覆盖写入配置子结构 (None = 未指定)."""
        for name, value in values.items():
            if value is not None:
                setattr(target, name, value)

    @staticmethod
    def _apply_sentinels(target: Any, values: dict[str, Any]) -> None:
        """把非 _UNSET_SENTINEL 的覆盖写入 (哨兵 = 未指定; None 是合法值)."""
        for name, value in values.items():
            if value is not _UNSET_SENTINEL:
                setattr(target, name, value)

    @classmethod
    def _apply_model_tool_memory_overrides(
        cls, config: AgentConfig, p: dict[str, Any]
    ) -> None:
        """model / tools / memory 三组覆盖."""
        cls._apply_overrides(
            config.model,
            {
                "model": p["model"],
                "model_router": p["model_router"],
                "system_prompt": p["system_prompt"],
                "begin_dialogs": p["begin_dialogs"],
                "prompt_cache_control": p["prompt_cache_control"],
            },
        )
        cls._apply_overrides(
            config.tools,
            {
                "tools": p["tools"],
                "tool_filter": p["tool_filter"],
                "max_tool_output_tokens": p["max_tool_output_tokens"],
                "max_tool_calls": p["max_tool_calls"],
                "max_tool_calls_per_tool": p["max_tool_calls_per_tool"],
                "compression_max_tokens": p["compression_max_tokens"],
            },
        )
        cls._apply_overrides(
            config.memory,
            {
                "memory_manager": p["memory_manager"],
                "checkpointer": p["checkpointer"],
                "checkpointer_path": p["checkpointer_path"],
                "memory_decay_enabled": p["memory_decay_enabled"],
                "memory_decay_interval_turns": p["memory_decay_interval_turns"],
                "memory_decay_prune_threshold": p["memory_decay_prune_threshold"],
            },
        )

    @classmethod
    def _apply_security_persona_overrides(
        cls, config: AgentConfig, p: dict[str, Any]
    ) -> None:
        """security / telemetry / context_budget / personalization 覆盖."""
        cls._apply_overrides(
            config.security,
            {
                "sandbox": p["sandbox"],
                "audit": p["audit"],
                "privacy_redact_secrets": p["privacy_redact_secrets"],
                "privacy_block_on_secrets": p["privacy_block_on_secrets"],
                "auto_approve": p["auto_approve"],
                "approval_callback": p["approval_callback"],
            },
        )
        cls._apply_overrides(
            config.telemetry,
            {"telemetry_enabled": p["telemetry_enabled"]},
        )
        cls._apply_overrides(
            config.context_budget,
            {"context_budget_tokens": p["context_budget_tokens"]},
        )
        cls._apply_overrides(
            config.personalization,
            {
                "persona_name": p["persona_name"],
                "emotion_tracker": p["emotion_tracker"],
                "style_learner": p["style_learner"],
            },
        )

    @classmethod
    def _apply_core_overrides(cls, config: AgentConfig, p: dict[str, Any]) -> None:
        """core 组 + 哨兵语义组 (enable_exploration / workspace / kg_* / break)."""
        cls._apply_overrides(
            config.core,
            {
                "thread_id": p["thread_id"],
                "agent_factory": p["agent_factory"],
                "skill_executor": p["skill_executor"],
            },
        )
        cls._apply_sentinels(
            config.core,
            {
                "enable_exploration": p["enable_exploration"],
                "profile_id": p["profile_id"],
            },
        )
        cls._apply_sentinels(
            config.knowledge_graph,
            {
                "workspace": p["workspace"],
                "kg_enabled": p["kg_enabled"],
                "kg_depth": p["kg_depth"],
                "kg_top_k": p["kg_top_k"],
            },
        )
        cls._apply_sentinels(config.tools, {"break_after_tool": p["break_after_tool"]})

    def _init_from_config(
        self,
        config: AgentConfig,
        scheduler: Any = None,
        kb_enabled: Any = _UNSET_SENTINEL,
    ) -> None:
        """按阶段装配 agent, 各阶段保持原有初始化顺序与降级行为."""
        m, t, mem, sec, tel, cb, kg, pers, core = _unpack_agent_config(config)

        self._core_init_base_state(m, t, mem)

        self._core_init_cache_and_session(m, core)

        self._core_init_cognitive()

        self._core_init_scheduler_state(t, core, scheduler)

        # ponytail: CodeAct 双轨制 — 默认走原 langgraph tool_call 路径,
        # 设 "code_act" 后 chat() 早返回到 code_act_loop. 用户通过实例属性切换,
        # 不改 __init__ 签名. 升级: 加 config 字段 + AgentConfig.mode.
        self.mode: str = "tool_call"

        self._core_init_mode_and_permissions(mem, sec, t, tel)

        self._core_init_security(sec)

        self.max_tool_output_tokens, self.context_budget_tokens = (
            self._core_resolve_context_budget(m, t, cb)
        )
        self._context_max_messages, self._model_context_window = (
            self._core_resolve_message_limits(m)
        )

        self._core_init_knowledge(kg, kb_enabled)

        self._core_init_persona(pers)

        self._core_init_memory_and_skills(mem, core)

        self._core_init_tool_adapter()
        self._core_init_hooks(t)

        self._core_init_personalization(pers)

        self._core_init_internal_state()

        self._core_init_context_builder()

    def _core_init_base_state(self, m: Any, t: Any, mem: Any) -> None:
        """基础字段、checkpointer 与 phase manager."""
        self.model = m.model
        self.model_router = m.model_router
        self.langchain_tools = t.tools or []
        self._main_fallback_override: Any = None

        self._exit_stack = ExitStack()

        if mem.checkpointer is not None:
            self.checkpointer = mem.checkpointer
        elif mem.checkpointer_path:
            from huginn.checkpointer import persistent_checkpointer

            self.checkpointer = self._exit_stack.enter_context(
                persistent_checkpointer(mem.checkpointer_path)
            )
        else:
            self.checkpointer = create_in_memory_checkpointer()

        self.system_prompt = m.system_prompt or HUGINN_BASE_PROMPT
        self.begin_dialogs = m.begin_dialogs or []

        from huginn.phases import ResearchPhase

        self._phase_manager = PhaseManager(initial=ResearchPhase.OPEN)

    def _core_init_cache_and_session(self, m: Any, core: Any) -> None:
        """prompt cache、任务路由字段与 unified session state."""
        # 当前用户消息, 用于 query-aware tool retrieval
        self._current_user_message: str | None = None

        # P4 Task-Dynamic Tool Router: 当前 task 描述, 由 set_current_task 设.
        # 跟 _current_user_message 分开, 避免并发覆盖 (session_context.py 警告).
        # HUGINN_TASK_TOOL_ROUTER=1 时 register_tools_from_registry 会用它过滤工具子集.
        self._current_task: str | None = None
        # 上次 routing 用的 task, 跟 _current_task 不同才 refresh, 避免每次 chat
        # 都重建 graph. ponytail: 字符串比较, 假设相同 task 字符串路由结果一致.
        self._last_routed_task: str | None = None

        self.prompt_cache_control = m.prompt_cache_control

        self._cache_builder = PromptCacheBuilder(
            system_prompt=self.system_prompt,
            begin_dialogs=self.begin_dialogs,
            cache_control=self.prompt_cache_control,
        )
        self._cache_builder.set_provider(self._detect_provider())

        self.enable_exploration = core.enable_exploration
        self.profile_id = core.profile_id
        self.thread_id = core.thread_id

        from huginn.session_state import UnifiedSessionState

        self._session_state = UnifiedSessionState(
            session_id=self.thread_id,
            persona_name=getattr(self, "persona_name", "default"),
            persona_system_prompt=self.system_prompt,
        )

    def _core_init_cognitive(self) -> None:
        """task reflector 与认知状态机 (含事件日志绑定)."""
        from huginn.task_reflector import TaskReflector

        self._reflector = TaskReflector()

        from huginn.cognitive_engine import CognitiveStateMachine

        self._csm = CognitiveStateMachine()
        # H1: 把会话事件日志绑定到认知状态机 — 每次认知跃迁 append 事件,
        # 使状态机状态成为事件日志的投影 (可重放/可恢复/可分支).
        try:
            from huginn.events.session_log import SessionEventLog

            self._csm.attach_session_log(
                SessionEventLog.open(self.thread_id, load=True)
            )
        except Exception:  # 防御: 会话事件日志加载失败则忽略, 不阻塞 agent
            logger.debug(
                "best-effort op failed", exc_info=True
            )  # 无事件日志不阻塞 agent

    def _core_init_scheduler_state(self, t: Any, core: Any, scheduler: Any) -> None:
        """tool_filter/agent_factory/scheduler 与图相关初始字段."""
        self.tool_filter = set(t.tool_filter) if t.tool_filter else None
        self.agent_factory = core.agent_factory
        self.scheduler = scheduler
        if self.scheduler is None:
            try:
                from huginn.scheduling import AdmissionPolicy, ToolScheduler

                self.scheduler = ToolScheduler(policy=AdmissionPolicy.from_env())
            except Exception:  # 防御: ToolScheduler 构造失败则置空, 后续懒加载兜底
                self.scheduler = None
        self._agent_graph: Any | None = None
        self._tool_description_text: str | None = None
        self._last_cache_stats: dict[str, Any] = {}
        self._conversation_summary: str = ""

        self._conversation_tree = ConversationTree()
        self._thread_branch_roots: dict[str, str] = {}
        self._state_msg_offsets: dict[str, int] = {}

        self._telemetry_collector = TelemetryCollector()
        self._turn_count = 0

    def _core_init_mode_and_permissions(
        self, mem: Any, sec: Any, t: Any, tel: Any
    ) -> None:
        """mode、memory decay、权限与 telemetry 字段."""
        # ponytail: CodeAct 双轨制 — 默认走原 langgraph tool_call 路径,
        # 设 "code_act" 后 chat() 早返回到 code_act_loop. 用户通过实例属性切换,
        # 不改 __init__ 签名. 升级: 加 config 字段 + AgentConfig.mode.
        self.mode: str = "tool_call"

        self.memory_decay_enabled = mem.memory_decay_enabled
        self.memory_decay_interval_turns = mem.memory_decay_interval_turns
        self.memory_decay_prune_threshold = mem.memory_decay_prune_threshold

        self.auto_approve = sec.auto_approve
        self._permission_config = PermissionConfig(auto_approve_all=sec.auto_approve)
        self._approval_callback: Callable[[str, str], bool] | None = (
            sec.approval_callback
        )
        # P2-5: Inbox opt-in. HUGINN_APPROVAL_MODE=inbox 时, CodeAct approval
        # 走跨会话队列 (Unattended 场景). 默认 None → yield approval_request 事件.
        if os.environ.get("HUGINN_APPROVAL_MODE") == "inbox":
            from huginn.interaction.inbox import get_inbox_store, inbox_approval_fn

            self.approval_fn = inbox_approval_fn(
                get_inbox_store(), self.thread_id or "default"
            )
        else:
            self.approval_fn = None

        self.compression_max_tokens = t.compression_max_tokens

        self.telemetry_enabled = tel.telemetry_enabled
        self._telemetry_collector = (
            TelemetryCollector() if tel.telemetry_enabled else NullTelemetryCollector()
        )

    def _core_init_security(self, sec: Any) -> None:
        """sandbox/audit 与隐私脱敏开关."""
        self.sandbox = sec.sandbox
        self.audit = sec.audit

        self.privacy_redact_secrets = sec.privacy_redact_secrets
        self.privacy_block_on_secrets = sec.privacy_block_on_secrets

    def _core_resolve_context_budget(self, m: Any, t: Any, cb: Any) -> tuple[Any, Any]:
        """解析工具输出上限与上下文预算 (预算 <=0 时按模型窗口回填)."""
        max_tool_output_tokens = t.max_tool_output_tokens
        context_budget_tokens = cb.context_budget_tokens
        if (
            context_budget_tokens is not None
            and context_budget_tokens <= 0
            and m.model is not None
        ):
            model_name = getattr(m.model, "model_name", None) or getattr(
                m.model, "model", ""
            )
            if model_name:
                context_budget_tokens = get_context_window(str(model_name))
        return max_tool_output_tokens, context_budget_tokens

    def _core_resolve_message_limits(self, m: Any) -> tuple[int, int]:
        """按消息条数的压缩阈值与当前模型上下文窗口."""
        # C1b: 渐进式压缩阈值 — 消息条数超此值即触发压缩(即便未超 token 预算),
        # 让长程任务把上下文渐进维护在安全水位, 而非等逼近模型窗口才压缩.
        # env 可调; 默认 60 条. 0 = 关闭按条数触发.
        context_max_messages = int(os.environ.get("HUGINN_CONTEXT_MAX_MESSAGES", "60"))
        model_context_window = (
            get_context_window(
                str(
                    getattr(m.model, "model_name", None)
                    or getattr(m.model, "model", "")
                )
            )
            if m.model is not None
            else 0
        )
        return context_max_messages, model_context_window

    def _core_init_knowledge(self, kg: Any, kb_enabled: Any) -> None:
        """workspace/knowledge graph 字段与上下文缓存重置."""
        self.workspace = kg.workspace
        self.kg_enabled = kg.kg_enabled
        self.kg_depth = kg.kg_depth
        self.kg_top_k = kg.kg_top_k
        self._kg: Any | None = None
        self.kb_enabled = True if kb_enabled is _UNSET_SENTINEL else bool(kb_enabled)
        self._kb: Any | None = None
        reset_context_cache()

    def _core_init_persona(self, pers: Any) -> None:
        """persona 与情绪追踪器."""
        self.persona_name = pers.persona_name
        self.emotion_tracker = pers.emotion_tracker

    def _core_init_memory_and_skills(self, mem: Any, core: Any) -> None:
        """memory manager 与 skill executor (缺省时懒加载)."""
        memory_manager = mem.memory_manager
        if memory_manager is None:
            from huginn.memory.manager import MemoryManager

            memory_manager = MemoryManager()
        self.memory = memory_manager

        skill_executor = core.skill_executor
        if skill_executor is None:
            from huginn.skills.base import DeclarativeSkillExecutor

            skill_executor = DeclarativeSkillExecutor(ToolRegistry)
        self.skills = skill_executor

    def _core_init_tool_adapter(self) -> None:
        """tool adapter、boundary state 与 summarizer."""
        self._tool_adapter = ToolAdapter()
        # P1-1: constraints BoundaryState 之前生产路径不注入, 所有 boundary 守卫被 None 短路.
        # 现在挂 session 级 BoundaryState, SafetyOperator 评测出 block 级失败后能收紧 require_confirmation.
        from huginn.constraints.boundaries import BoundaryState

        self._boundary_state = BoundaryState()
        # 反向引用: 让 ToolAdapter._serialize 能把 _visual_base64 存到 agent 实例
        self._tool_adapter._agent_ref = self
        tool_summarizer = self._make_summarizer()
        if tool_summarizer is not None:
            self._tool_adapter.set_summarizer(tool_summarizer)

    def _core_init_hooks(self, t: Any) -> None:
        """hook manager、science hooks 与工具调用上限."""
        self.hook_manager = HookManager()
        self.hook_manager.agent = self

        try:
            from huginn.hooks.science_hooks import register_science_hooks

            register_science_hooks(self.hook_manager)
        except ImportError:
            logger.debug("science_hooks not available (non-fatal)")

        self._break_after_tool = t.break_after_tool
        self._break_flag = False

        self._max_tool_calls = t.max_tool_calls
        self._max_tool_calls_per_tool = t.max_tool_calls_per_tool

    def _core_init_personalization(self, pers: Any) -> None:
        """style learner 与共享实例注册."""
        self.style_learner = pers.style_learner
        if pers.style_learner is not None:
            from huginn.personalization import set_shared_style_learner

            set_shared_style_learner(pers.style_learner)

    def _core_init_internal_state(self) -> None:
        """内部模式标记."""
        self._mode: AgentMode = "chat"

        self._evolution_engine: Any | None = None

    def _core_init_context_builder(self) -> None:
        """上下文构建器 (依赖前面各阶段产物)."""
        from huginn.context_builder import ContextBuilder

        self._ctx_builder = ContextBuilder(
            memory_manager=self.memory,
            workspace=self.workspace,
            kg_enabled=self.kg_enabled,
            kb_enabled=self.kb_enabled,
            kg_depth=self.kg_depth,
            kg_top_k=self.kg_top_k,
            emotion_tracker=self.emotion_tracker,
            checkpointer=getattr(self, "checkpointer", None),
            conversation_tree=self._conversation_tree,
            cache_builder=self._cache_builder,
        )

    # ── Mode management ───────────────────────────────────────────

    # 长程模式: research / extreme 都是长程任务, 需要开 trajectory 召回 +
    # cycle 检测才不重复犯错. chat/plan 是短程, 不开.
    # 词表单一来源 (A3): 值域见 huginn.agent_modes.LONG_HORIZON_MODES.
    _LONG_HORIZON_MODES = LONG_HORIZON_MODES

    # tool 预算单一数据源在 agent_config.MODE_TOOL_BUDGET, 这里不再自持一份.
    # 科研长工具链吃单工具上限, 默认 5 次一碰就卡死 — mode 联动已在统一入口做.
    # 向后兼容别名: 旧版用 _LONG_HORIZON_TOOL_BUDGET 映射 mode→(calls, per_tool),
    # 现收敛到 agent_config. 保留别名防外部代码读取报 AttributeError.
    _LONG_HORIZON_TOOL_BUDGET: tuple[int, int] = (300, 100)
    # 上下文窗口已到 1M (deepseek-v4), 200K 时代的 100/300 明显偏小.

    def _set_long_horizon_env(self, enable: bool) -> None:
        """长程模式联动: 开/关 HUGINN_EXTREME_DISPATCH 并重置研究阶段."""
        import os

        if enable:
            # 长程模式联动: research/extreme 跑长程 (recursion_limit=500),
            # 需要开 trajectory 召回 + cycle 检测才不重复犯错. 保存原值便于退出恢复.
            if not hasattr(self, "_extreme_dispatch_saved"):
                self._extreme_dispatch_saved = os.environ.get("HUGINN_EXTREME_DISPATCH")
            os.environ["HUGINN_EXTREME_DISPATCH"] = "1"
            self._phase_manager.reset(ResearchPhase.LITERATURE)
            return
        # 退出长程模式: 恢复 HUGINN_EXTREME_DISPATCH 原值
        saved = getattr(self, "_extreme_dispatch_saved", None)
        if saved is None:
            os.environ.pop("HUGINN_EXTREME_DISPATCH", None)
        else:
            os.environ["HUGINN_EXTREME_DISPATCH"] = saved
        self._extreme_dispatch_saved = None
        self._phase_manager.reset(ResearchPhase.OPEN)

    def _delegate_mode_switch(self, mode: str) -> None:
        """R7: mode 切换 = cognitive switch, 委托 CSM S3_SWITCH.

        reflection.py 的 should_switch_mode 路径也走 set_mode → 这里一处覆盖两边.
        ponytail: getattr 兜底 + try/except 静默, 不破坏 turn. 升级: 显式 CSM 注入.
        """
        csm = getattr(self, "_csm", None)
        if csm is None:
            return
        try:
            from huginn.cognitive_engine import TransitionSignal

            csm.transition(TransitionSignal("user_confirmed", {"mode": mode}))
        except Exception:  # 防御: set_mode CSM 委托失败则忽略, state 保持原值
            logger.debug("set_mode CSM delegation failed", exc_info=True)

    def set_mode(self, mode: str) -> None:
        """切换 agent mode (chat/research/plan/extreme).

        内部委托 CSM S3_SWITCH 处理 (R7 减法: 合并 mode 切换和 CSM 控制).
        保留公开 API 向后兼容. CSM transition 是 advisory — 当前状态不允许转 S3
        时返回当前 state 不强制, 不破坏现有 mode 逻辑.
        """
        if mode not in SETTABLE_MODES:
            raise ValueError(
                f"unknown mode: {mode}. options: {', '.join(SETTABLE_MODES)}"
            )
        if mode == self._mode:
            return
        old = self._mode
        self._mode = mode
        self._agent_graph = None
        if mode in self._LONG_HORIZON_MODES:
            HuginnAgent._set_long_horizon_env(self, True)
        elif old in self._LONG_HORIZON_MODES:
            HuginnAgent._set_long_horizon_env(self, False)
        # plan mode 联动 permission_config: 写工具强制 ASK
        # 之前 /plan slash 只改 _mode 不改 permission_config, 导致 UI 提示与实际行为不一致.
        perm_cfg = getattr(self, "_permission_config", None)
        if perm_cfg is not None:
            perm_cfg.plan_mode = mode == "plan"
        logger.info("agent mode switched to '%s'", mode)
        HuginnAgent._delegate_mode_switch(self, mode)

    def get_mode(self) -> AgentMode:
        return self._mode

    def is_research_mode(self) -> bool:
        return self._mode == "research"

    def is_extreme_mode(self) -> bool:
        return self._mode == "extreme"

    def is_plan_mode(self) -> bool:
        # 兜底: plan execution 期间 permission_config.plan_mode 临时翻转但 _mode 不变,
        # 读 is_plan_mode() 的代码 (research_safety_hook / research_budget) 需要感知.
        # ponytail: or 短路, 不改 _mode 语义. 升级: 抽 enter/exit plan execution 显式状态.
        return self._mode == "plan" or bool(
            getattr(self._permission_config, "plan_mode", False)
        )

    def _effective_recursion_limit(self) -> int:
        """根据 mode 和 extreme dispatch 返回 recursion_limit.

        - research / extreme mode → 500 (长程探索)
        - extreme dispatch 开启但非长程 mode → 400 (极限模式跑长程, 250 不够)
        - 普通 chat/plan → 250 (短程, 省计算)
        ponytail: 三档够用, 不上配置. 升级: 暴露到 config.py.
        """

        if self._mode in self._LONG_HORIZON_MODES:
            return 500
        if FeatureFlags.shared().is_enabled("extreme_dispatch"):
            return 400
        return 250

    def _effective_tool_budget(self) -> tuple[int, int]:
        """tool 预算随 mode 联动, 与 _effective_recursion_limit 档位对齐.

        数值与长程地基都收敛在 agent_config.resolve_tool_budget, 这里只透传
        mode 与显式配置. 保持方法名不变, streaming 调用点无感.
        """
        return resolve_tool_budget(
            self._mode,
            self._max_tool_calls,
            self._max_tool_calls_per_tool,
        )

    def enter_plan_execution(self) -> None:
        """执行阶段临时开启 plan_mode (写工具强制 ASK), 不改 _mode 字段.

        与 set_mode("plan") 区别: set_mode 是用户级 mode 切换 (会重置 phase_manager /
        失效 graph); enter_plan_execution 是 plan 执行窗口内的权限加严, 保持 _mode
        不变, 避免把 chat/research mode 误覆盖成 plan mode.
        ponytail: 配对 exit_plan_execution restore. 升级: 用 contextmanager.
        """
        self._plan_exec_prev = getattr(self._permission_config, "plan_mode", False)
        if self._permission_config is not None:
            self._permission_config.plan_mode = True

    def exit_plan_execution(self) -> None:
        """配对 exit: 恢复 enter_plan_execution 之前的 plan_mode 状态."""
        if self._permission_config is not None:
            self._permission_config.plan_mode = getattr(self, "_plan_exec_prev", False)

    # ── Specialised model selection ───────────────────────────────
    # ponytail: select_verification_model/select_archival_model/has_dedicated_verification
    # 已删除 — 全代码库无调用方. ModelRouter.select_verification() 在 routes/eval.py
    # 直接调用, 不需要 core.py 的包装层. 如需恢复, git log 找到此 commit.

    def _detect_provider(self) -> str | None:
        if self.model is None:
            return None
        llm_type = getattr(self.model, "_llm_type", None)
        if isinstance(llm_type, str):
            return llm_type
        cls_name = self.model.__class__.__name__.lower()
        module_name = getattr(self.model, "__module__", "").lower()
        for hint in (module_name, cls_name):
            if "anthropic" in hint or "claude" in hint:
                return "anthropic"
            if "openai" in hint:
                return "openai"
            if "kimi" in hint or "moonshot" in hint:
                return "kimi"
            if "deepseek" in hint:
                return "deepseek"
            if "google" in hint or "genai" in hint:
                return "google"
        return None

    @classmethod
    def from_config(
        cls,
        config: Any,
        profile_id: str = "lead",
        **overrides: Any,
    ) -> HuginnAgent:
        kwargs = config.build_agent_kwargs(profile_id=profile_id)
        kwargs.update(overrides)
        return cls(**kwargs)

    # ── Tool registration ─────────────────────────────────────────

    def register_tool(self, tool: Any) -> None:
        from huginn.tools.base import HuginnTool

        if isinstance(tool, HuginnTool):
            lc_tool = self._tool_adapter.adapt(
                tool,
                max_tool_output_tokens=self.max_tool_output_tokens,
                compression_max_tokens=self.compression_max_tokens,
                permission_config=self._permission_config,
                approval_callback=self._approval_callback,
                boundary_state=self._boundary_state,
            )
        else:
            lc_tool = tool
        self.langchain_tools.append(self._wrap_tool_with_hooks(lc_tool))
        self._invalidate_tool_description_cache()

    def _effective_tool_filter(self) -> set[str] | None:
        """P4 Task-Dynamic Tool Router: 按 task 语义算有效子集.

        默认关 (HUGINN_TASK_TOOL_ROUTER=1 开启). 失败或返回空 → fallback 到
        原 self.tool_filter, 不破坏现有行为. 升级路径: LLM 版小模型打分.

        跟 self.tool_filter 取交集而非覆盖: caller (如 benchmark runner) 显式设的
        tool_filter 是硬约束 (限制 agent 只能用 code_tool/bash_tool), 不能被
        task router 冲掉. 交集为空 → 保留原 tool_filter (task router 误判时不破坏).
        """
        if not (
            FeatureFlags.shared().is_enabled("task_tool_router") and self._current_task
        ):
            return self.tool_filter
        try:
            from huginn.runtime.task_tool_router import compute_effective_subset

            candidate = compute_effective_subset(
                self._current_task, ToolRegistry.list_tools()
            )
        except Exception:  # 防御: task_tool_router 异常则回退 tool_filter
            logger.debug(
                "task_tool_router failed, fallback to tool_filter", exc_info=True
            )
            return self.tool_filter
        if not candidate:
            return self.tool_filter
        candidate_set = set(candidate)
        if self.tool_filter is None:
            return candidate_set
        inter = candidate_set & self.tool_filter
        # 交集空 → task router 误判, 保留原 tool_filter
        return inter if inter else self.tool_filter

    def _adapt_registry_tool(self, tool: Any, huginn_tool_cls: Any) -> Any:
        """registry 工具 → LangChain 工具 (非 HuginnTool 原样返回)."""
        if not isinstance(tool, huginn_tool_cls):
            return tool
        return self._tool_adapter.adapt(
            tool,
            memory_manager=self.memory,
            agent_factory=self.agent_factory,
            max_tool_output_tokens=self.max_tool_output_tokens,
            compression_max_tokens=self.compression_max_tokens,
            permission_config=self._permission_config,
            approval_callback=self._approval_callback,
            boundary_state=self._boundary_state,
        )

    def register_tools_from_registry(self) -> None:
        from huginn.tools.assembly import collect_denied_tool_names
        from huginn.tools.base import HuginnTool

        denied = collect_denied_tool_names(self._permission_config.rules)
        effective_filter = self._effective_tool_filter()

        tools = []
        # C: 懒加载边界 — 先按名过滤 (effective_filter/denied), 命中才 get()/adapt.
        # 之前对全部 132 工具先 get() 再过滤, 每次 task 变化都把重后端工具实例化并
        # 包进 langchain. 现在只在被选中子集内 get()/adapt, 未命中的重工具不实例化,
        # 与常驻 task_tool_router 的最小子集联动.
        for name in ToolRegistry.list_tools():
            if effective_filter is not None and name not in effective_filter:
                continue
            if name in denied:
                continue
            tool = ToolRegistry.get(name)
            if tool is None:
                continue
            tools.append(
                self._wrap_tool_with_hooks(self._adapt_registry_tool(tool, HuginnTool))
            )
        # replace 语义而非 extend: 避免 MCP 重连 / mock fallback 重复调用导致工具重复注册.
        # 之前 extend 会让 LLM 在工具 schema 里看到 N 份同名 tool, 浪费 context tokens.
        # ponytail: 直接覆盖, 假设所有工具都来自 registry. 升级: 保留非 registry 来源工具.
        self.langchain_tools = tools
        self._invalidate_tool_description_cache()

    def set_current_task(self, message: str | None) -> None:
        """记录当前 task 描述, 供 task-dynamic tool router (P4) 使用.

        HUGINN_TASK_TOOL_ROUTER=1 时, 下次 register_tools_from_registry
        会根据这个 message 路由工具子集. message 为 None 清空路由.

        ponytail: 跟 _current_user_message 分开存, 避免并发覆盖
        (session_context.py 警告的多 chat() 并发场景).
        升级路径: 直接读 session_state.current_objective, 不依赖 caller 显式 set.
        """
        self._current_task = message

    def refresh_tools_from_registry(self) -> None:
        """重连 MCP / 改了 tool_filter / mock fallback 后调一次.

        重新从 ToolRegistry 拉工具, 并失效已编译的 graph, 让下次 build_graph 用新工具集.
        之前 MCP reconnect 路由只刷 ToolRegistry, agent.langchain_tools / _agent_graph 保持旧的,
        LLM 看到 stale schema 还在调已经不存在的工具.
        ponytail: 调 register_tools_from_registry + 置 _agent_graph=None. 升级: 增量 diff.
        """
        self.register_tools_from_registry()
        self._agent_graph = None  # 下次 build_graph 重建
        logger.info("tools refreshed from registry (graph invalidated)")

    # ── Model selection ──────────────────────────────────────────

    def select_model(self, task: str = "agent") -> Any:
        if task == "agent" and self._main_fallback_override is not None:
            return self._main_fallback_override
        if self.model_router is not None:
            # 让 CSM state 覆盖 task: 构造用 reasoning, 验证/自修改用 verification
            # ponytail: 用 .get 避免 KeyError，未映射状态保留原 task；升级路径是加完整 CSM state → model task 映射
            try:
                csm = getattr(self, "_csm", None)
                if csm is not None:
                    task = STATE_TO_MODEL_TASK.get(csm.state, task)
            except Exception:  # 防御: CSM state 未匹配则保留原 task
                logger.debug(
                    "CSM state → model task mapping failed, fallback to original task",
                    exc_info=True,
                )
            return self.model_router.select(task)
        if self.model is None:
            raise RuntimeError("HuginnAgent has no model or model_router configured")
        return self.model

    def _select_main_fallback_model(self) -> Any:
        if self.model_router is None:
            return None
        if len(self.model_router.list_models()) < 2:
            return None
        try:
            primary = self.select_model("agent")
            cheap = self.model_router.select("cheap", prefer_cheap=True)
            if cheap is not primary:
                return cheap
        except Exception:  # 防御: 主备兜底模型选择失败则无备选
            logger.debug("pick main fallback model failed", exc_info=True)
        return None

    # ── Explainability ───────────────────────────────────────────

    def explain(
        self,
        goal: str | None = None,
        *,
        actor: str | None = None,
        tool: str | None = None,
        limit: int = 200,
    ) -> Any:
        """统一解释入口: 把 audit + provenance 观测拼成端到端解释.

        可解释性 facade (huginn.explainability.Explainer) 的便捷封装.
        默认用本 agent 的 audit logger + ProvenanceRegistry.shared() 作为
        provider, 无需调用方手动装配. 返回 Explanation (含 timeline /
        artifacts / edges / key_findings).
        """
        from huginn.explainability import Explainer
        from huginn.provenance.registry import ProvenanceRegistry

        provenance = None
        try:
            provenance = ProvenanceRegistry.shared()
        except Exception:  # 防御: ProvenanceRegistry 不可用则 provenance 置空
            logger.debug("ProvenanceRegistry.shared() unavailable", exc_info=True)
        return Explainer(
            audit=getattr(self, "audit", None), provenance=provenance
        ).explain(goal=goal, actor=actor, tool=tool, limit=limit)

    # ── Graph building ────────────────────────────────────────────

    def _deep_agent_backend(self) -> Any:
        """deepagents 内置 fs 工具的落盘后端 (LocalShellBackend).

        deepagents 内置 fs 工具 (ls/read_file/write_file/edit_file/execute) 默认走
        StateBackend — 虚拟内存 fs, write_file 返回成功但没落盘, agent 以为
        写了 report.md 实际没有 (σ₁₀). 换 LocalShellBackend 落真盘 +
        支持 execute 命令执行 (FilesystemBackend 不实现 SandboxBackendProtocol,
        导致 execute 工具报 "backend does not support command execution").
        virtual_mode=True 把 "/report.md" 映射进 root 且禁止路径逃逸.
        inherit_env=True 让 subprocess 有 PATH 等环境变量.
        """
        from deepagents.backends import LocalShellBackend

        fs_root = str(self.workspace) if self.workspace else None
        return LocalShellBackend(
            root_dir=fs_root,
            virtual_mode=True,
            inherit_env=True,
        )

    def _build_deep_agent(self) -> Any:
        """deepagents.create_deep_agent 建图 (含 middleware + recursion_limit)."""
        from deepagents import create_deep_agent

        system_message = self._effective_system_prompt()
        fs_backend = self._deep_agent_backend()
        _agent_model = self.select_model("agent")
        return create_deep_agent(
            name="HuginnAgent",
            model=_agent_model,
            tools=self._effective_tools(query=get_user_message()),
            system_prompt=system_message,
            checkpointer=self.checkpointer,
            backend=fs_backend,
            middleware=[
                FixDanglingToolCallsMiddleware(
                    model_name=getattr(_agent_model, "model_name", "")
                ),
                DeliverableCoverageMiddleware(),
                RateLimitMiddleware(),
            ],
        ).with_config(
            {
                # research mode → 500; extreme dispatch 开启但非 research → 400
                # (极限模式跑长程, 250 不够用). 普通 chat/plan 保持 250.
                "recursion_limit": self._effective_recursion_limit(),
            }
        )

    def build_graph(self) -> Any:
        if self._agent_graph is not None:
            return self._agent_graph

        try:
            self._agent_graph = self._build_deep_agent()
            return self._agent_graph
        except ImportError:
            return self._build_simple_graph()

    def _build_simple_graph(self) -> Any:
        """deepagents 未装时的 langgraph prebuilt 兜底图."""
        try:
            from langgraph.prebuilt import create_react_agent

            from huginn.agent.middlewares import (
                DeliverableCoverageMiddleware,
                FixDanglingToolCallsMiddleware,
            )

            _fix_mw = FixDanglingToolCallsMiddleware(
                model_name=getattr(self.select_model("agent"), "model_name", "")
            )
            agent = create_react_agent(
                model=self.select_model("agent"),
                tools=self._effective_tools(query=get_user_message()),
                prompt=SystemMessage(content=self._effective_system_prompt()),
                pre_model_hook=_make_pre_model_hook(
                    _fix_mw, DeliverableCoverageMiddleware()
                ),
                checkpointer=self.checkpointer,
            )

            self._agent_graph = agent
            return agent

        except ImportError as err:
            raise ImportError(
                "Neither deepagents nor langgraph prebuilt agents are available. "
                "Install one of them to use HuginnAgent."
            ) from err

    # ── Conversation branches ─────────────────────────────────────

    def fork_conversation(self, from_node_id: str | None = None) -> dict[str, Any]:
        if from_node_id is None:
            leaf = self._conversation_tree.get_node(
                self._conversation_tree.active_leaf_id
            )
            from_node_id = leaf.parent_id if leaf else self._conversation_tree.root_id

        node = self._conversation_tree.fork(from_node_id)
        if node is None:
            return {"success": False, "error": "Node not found"}
        return {
            "success": True,
            "forked_from": from_node_id,
            "active_leaf": self._conversation_tree.active_leaf_id,
        }

    def switch_branch(self, node_id: str) -> dict[str, Any]:
        ok = self._conversation_tree.set_active_leaf(node_id)
        return {
            "success": ok,
            "active_leaf": self._conversation_tree.active_leaf_id,
            "active_path": self._conversation_tree.active_path(),
        }

    def conversation_branches(self) -> dict[str, Any]:
        return {
            "summary": self._conversation_tree.summary(),
            "branches": self._conversation_tree.get_branches(),
            "active_path": self._conversation_tree.active_path(),
        }

    # ── Telemetry ─────────────────────────────────────────────────

    def telemetry_summary(self) -> dict[str, Any]:
        return self._telemetry_collector.summary()

    def telemetry_spans(self) -> list[dict[str, Any]]:
        return self._telemetry_collector.to_dict()

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        try:
            from huginn.privacy_guard import PrivacyGuard

            purged = PrivacyGuard.shared().purge_session()
            if purged:
                logger.info("purged %d ephemeral privacy entries", len(purged))
        except Exception:  # 防御: runtime 归档时隐私清理失败不阻塞关闭
            logger.warning("privacy purge failed", exc_info=True)
        self._exit_stack.close()

    def __enter__(self) -> HuginnAgent:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ── Synchronous invocation ────────────────────────────────────

    def invoke(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        async def _run():
            final_state = None
            async for state in self.chat(message, thread_id):
                if isinstance(state, dict) and state.get("tool_break"):
                    final_state = state.get("state", final_state)
                elif (
                    isinstance(state, dict)
                    and "messages" in state
                    or final_state is None
                ):
                    final_state = state
            return final_state

        # 用统一 helper: 已在 running loop 时跑独立线程, 否则直接 asyncio.run.
        # 之前 try/except 顺序写反 — run_until_complete 抛 RuntimeError 后
        # asyncio.run 仍会因 running loop 抛错, 错误透传给 caller.
        from huginn.utils.async_bridge import run_async

        return run_async(_run())

    # ── Memory shortcuts ──────────────────────────────────────────


# Short alias for convenience: from huginn.agent import Agent
Agent = HuginnAgent
