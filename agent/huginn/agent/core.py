"""HuginnAgent — core class with init, factory, graph, and dispatch methods.

The heavy logic lives in the mixin modules: context.py, streaming.py,
reflection.py, session.py, callbacks.py.  This file wires them together
and keeps the construction + graph-building + convenience methods.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack
from typing import Any

from langchain_core.messages import SystemMessage

from huginn.agent_config import _UNSET_SENTINEL, AgentConfig
from huginn.self_improvement import BenchmarkSuite
from huginn.checkpointer import create_in_memory_checkpointer
from huginn.context_manager import (
    get_context_window,
    reset_context_cache,
)
from huginn.cognitive_engine import STATE_TO_MODEL_TASK
from huginn.hooks import HookManager
from huginn.models.router import ModelRouter
from huginn.permissions import PermissionConfig
from huginn.phases import PhaseManager, ResearchPhase
from huginn.prompts import HUGINN_SYSTEM_PROMPT
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

        m, t, mem = config.model, config.tools, config.memory
        sec, tel, cb = config.security, config.telemetry, config.context_budget
        kg, pers, core = (
            config.knowledge_graph,
            config.personalization,
            config.core,
        )

        if model is not None:
            m.model = model
        if model_router is not None:
            m.model_router = model_router
        if system_prompt is not None:
            m.system_prompt = system_prompt
        if begin_dialogs is not None:
            m.begin_dialogs = begin_dialogs
        if prompt_cache_control is not None:
            m.prompt_cache_control = prompt_cache_control
        if tools is not None:
            t.tools = tools
        if tool_filter is not None:
            t.tool_filter = tool_filter
        if max_tool_output_tokens is not None:
            t.max_tool_output_tokens = max_tool_output_tokens
        if max_tool_calls is not None:
            t.max_tool_calls = max_tool_calls
        if max_tool_calls_per_tool is not None:
            t.max_tool_calls_per_tool = max_tool_calls_per_tool
        if compression_max_tokens is not None:
            t.compression_max_tokens = compression_max_tokens
        if memory_manager is not None:
            mem.memory_manager = memory_manager
        if checkpointer is not None:
            mem.checkpointer = checkpointer
        if checkpointer_path is not None:
            mem.checkpointer_path = checkpointer_path
        if memory_decay_enabled is not None:
            mem.memory_decay_enabled = memory_decay_enabled
        if memory_decay_interval_turns is not None:
            mem.memory_decay_interval_turns = memory_decay_interval_turns
        if memory_decay_prune_threshold is not None:
            mem.memory_decay_prune_threshold = memory_decay_prune_threshold
        if sandbox is not None:
            sec.sandbox = sandbox
        if audit is not None:
            sec.audit = audit
        if privacy_redact_secrets is not None:
            sec.privacy_redact_secrets = privacy_redact_secrets
        if privacy_block_on_secrets is not None:
            sec.privacy_block_on_secrets = privacy_block_on_secrets
        if auto_approve is not None:
            sec.auto_approve = auto_approve
        if approval_callback is not None:
            sec.approval_callback = approval_callback
        if telemetry_enabled is not None:
            tel.telemetry_enabled = telemetry_enabled
        if context_budget_tokens is not None:
            cb.context_budget_tokens = context_budget_tokens
        if persona_name is not None:
            pers.persona_name = persona_name
        if emotion_tracker is not None:
            pers.emotion_tracker = emotion_tracker
        if style_learner is not None:
            pers.style_learner = style_learner
        if thread_id is not None:
            core.thread_id = thread_id
        if agent_factory is not None:
            core.agent_factory = agent_factory
        if skill_executor is not None:
            core.skill_executor = skill_executor

        if enable_exploration is not _UNSET_SENTINEL:
            core.enable_exploration = enable_exploration
        if profile_id is not _UNSET_SENTINEL:
            core.profile_id = profile_id
        if workspace is not _UNSET_SENTINEL:
            kg.workspace = workspace
        if kg_enabled is not _UNSET_SENTINEL:
            kg.kg_enabled = kg_enabled
        if kg_depth is not _UNSET_SENTINEL:
            kg.kg_depth = kg_depth
        if kg_top_k is not _UNSET_SENTINEL:
            kg.kg_top_k = kg_top_k
        if break_after_tool is not _UNSET_SENTINEL:
            t.break_after_tool = break_after_tool

        self._init_from_config(config, scheduler=scheduler, kb_enabled=kb_enabled)

    def _init_from_config(
        self, config: AgentConfig, scheduler: Any = None, kb_enabled: Any = _UNSET_SENTINEL
    ) -> None:
        m = config.model
        t = config.tools
        mem = config.memory
        sec = config.security
        tel = config.telemetry
        cb = config.context_budget
        kg = config.knowledge_graph
        pers = config.personalization
        core = config.core

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

        self.system_prompt = m.system_prompt or HUGINN_SYSTEM_PROMPT
        self.begin_dialogs = m.begin_dialogs or []

        from huginn.phases import ResearchPhase

        self._phase_manager = PhaseManager(initial=ResearchPhase.OPEN)

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

        from huginn.task_reflector import TaskReflector

        self._reflector = TaskReflector()

        from huginn.cognitive_engine import CognitiveStateMachine

        self._csm = CognitiveStateMachine()

        self.tool_filter = set(t.tool_filter) if t.tool_filter else None
        self.agent_factory = core.agent_factory
        self.scheduler = scheduler
        if self.scheduler is None:
            try:
                from huginn.scheduling import AdmissionPolicy, ToolScheduler
                self.scheduler = ToolScheduler(policy=AdmissionPolicy.from_env())
            except Exception:
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

        self.compression_max_tokens = t.compression_max_tokens

        self.telemetry_enabled = tel.telemetry_enabled
        self._telemetry_collector = (
            TelemetryCollector()
            if tel.telemetry_enabled
            else NullTelemetryCollector()
        )

        self.sandbox = sec.sandbox
        self.audit = sec.audit

        self.privacy_redact_secrets = sec.privacy_redact_secrets
        self.privacy_block_on_secrets = sec.privacy_block_on_secrets

        max_tool_output_tokens = t.max_tool_output_tokens
        context_budget_tokens = cb.context_budget_tokens
        if context_budget_tokens is not None and context_budget_tokens <= 0 and m.model is not None:
            model_name = getattr(m.model, "model_name", None) or getattr(
                m.model, "model", ""
            )
            if model_name:
                context_budget_tokens = get_context_window(str(model_name))
        self.max_tool_output_tokens = max_tool_output_tokens
        self.context_budget_tokens = context_budget_tokens
        self._model_context_window = (
            get_context_window(
                str(
                    getattr(m.model, "model_name", None)
                    or getattr(m.model, "model", "")
                )
            )
            if m.model is not None
            else 0
        )

        self.workspace = kg.workspace
        self.kg_enabled = kg.kg_enabled
        self.kg_depth = kg.kg_depth
        self.kg_top_k = kg.kg_top_k
        self._kg: Any | None = None
        self.kb_enabled = True if kb_enabled is _UNSET_SENTINEL else bool(kb_enabled)
        self._kb: Any | None = None
        reset_context_cache()

        self.persona_name = pers.persona_name
        self.emotion_tracker = pers.emotion_tracker

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

        self._tool_adapter = ToolAdapter()
        # 反向引用: 让 ToolAdapter._serialize 能把 _visual_base64 存到 agent 实例
        self._tool_adapter._agent_ref = self
        tool_summarizer = self._make_summarizer()
        if tool_summarizer is not None:
            self._tool_adapter.set_summarizer(tool_summarizer)

        self.hook_manager = HookManager()

        try:
            from huginn.hooks.science_hooks import register_science_hooks
            register_science_hooks(self.hook_manager)
        except ImportError:
            logger.debug("science_hooks not available (non-fatal)")

        self._break_after_tool = t.break_after_tool
        self._break_flag = False

        self._max_tool_calls = t.max_tool_calls
        self._max_tool_calls_per_tool = t.max_tool_calls_per_tool

        self.style_learner = pers.style_learner
        if pers.style_learner is not None:
            from huginn.personalization import set_shared_style_learner
            set_shared_style_learner(pers.style_learner)

        self._mode: str = "chat"

        self._evolution_engine: Any | None = None

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

    def set_mode(self, mode: str) -> None:
        """切换 agent mode (chat/research/plan).

        内部委托 CSM S3_SWITCH 处理 (R7 减法: 合并 mode 切换和 CSM 控制).
        保留公开 API 向后兼容. CSM transition 是 advisory — 当前状态不允许转 S3
        时返回当前 state 不强制, 不破坏现有 mode 逻辑.
        """
        if mode not in ("chat", "research", "plan"):
            raise ValueError(f"unknown mode: {mode}. options: chat, research, plan")
        if mode != self._mode:
            old = self._mode
            self._mode = mode
            self._agent_graph = None
            if mode == "research":
                self._phase_manager.reset(ResearchPhase.LITERATURE)
            elif old == "research":
                self._phase_manager.reset(ResearchPhase.OPEN)
            # plan mode 联动 permission_config: 写工具强制 ASK
            # 之前 /plan slash 只改 _mode 不改 permission_config, 导致 UI 提示与实际行为不一致.
            perm_cfg = getattr(self, "_permission_config", None)
            if perm_cfg is not None:
                perm_cfg.plan_mode = (mode == "plan")
            logger.info("agent mode switched to '%s'", mode)

            # R7: mode 切换 = cognitive switch, 委托 CSM S3_SWITCH.
            # reflection.py 的 should_switch_mode 路径也走 set_mode → 这里一处覆盖两边.
            # ponytail: getattr 兜底 + try/except 静默, 不破坏 turn. 升级: 显式 CSM 注入.
            csm = getattr(self, "_csm", None)
            if csm is not None:
                try:
                    from huginn.cognitive_engine import TransitionSignal
                    csm.transition(TransitionSignal("user_confirmed", {"mode": mode}))
                except Exception:
                    logger.debug("set_mode CSM delegation failed", exc_info=True)

    def get_mode(self) -> str:
        return self._mode

    def is_research_mode(self) -> bool:
        return self._mode == "research"

    def is_plan_mode(self) -> bool:
        # 兜底: plan execution 期间 permission_config.plan_mode 临时翻转但 _mode 不变,
        # 读 is_plan_mode() 的代码 (research_safety_hook / research_budget) 需要感知.
        # ponytail: or 短路, 不改 _mode 语义. 升级: 抽 enter/exit plan execution 显式状态.
        return self._mode == "plan" or bool(
            getattr(self._permission_config, "plan_mode", False)
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

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def from_provider(
        cls,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        """Create a HuginnAgent from any supported provider."""
        # ponytail: preserves the original import/call name mismatch —
        # create_langchain_model is imported without underscore but called
        # with one. Not called in tests; fixing it is a separate concern.
        model_instance = _create_langchain_model(
            provider=provider,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
        )
        return cls(model=model_instance, **kwargs)

    @classmethod
    def from_anthropic(
        cls,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        return cls.from_provider("anthropic", model=model, api_key=api_key, **kwargs)

    @classmethod
    def from_openai(
        cls,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        return cls.from_provider(
            "openai", model=model, api_key=api_key, base_url=base_url, **kwargs
        )

    @classmethod
    def from_ollama(
        cls,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> HuginnAgent:
        return cls.from_provider("ollama", model=model, base_url=base_url, **kwargs)

    @classmethod
    def from_deepseek(
        cls,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        return cls.from_provider("deepseek", model=model, api_key=api_key, **kwargs)

    @classmethod
    def from_google(
        cls,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        return cls.from_provider("google-genai", model=model, api_key=api_key, **kwargs)

    @classmethod
    def from_evo_config(
        cls,
        model_name: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        try:
            from EvoScientist.llm.models import get_chat_model

            model = get_chat_model(model=model_name, provider=provider)
        except ImportError as err:
            raise ImportError("EvoScientist not installed.") from err
        return cls(model=model, **kwargs)

    @classmethod
    def from_model_router(
        cls,
        router: ModelRouter,
        **kwargs: Any,
    ) -> HuginnAgent:
        return cls(model_router=router, **kwargs)

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
            )
        else:
            lc_tool = tool
        self.langchain_tools.append(self._wrap_tool_with_hooks(lc_tool))
        self._invalidate_tool_description_cache()

    def register_tools_from_registry(self) -> None:
        from huginn.tools.assembly import collect_denied_tool_names
        from huginn.tools.base import HuginnTool

        denied = collect_denied_tool_names(self._permission_config.rules)

        # P4 Task-Dynamic Tool Router: 根据 task 语义动态过滤工具子集.
        # 默认关 (HUGINN_TASK_TOOL_ROUTER=1 开启). 失败或返回空 → fallback 到
        # 原 self.tool_filter, 不破坏现有行为. 升级路径: LLM 版小模型打分.
        #
        # 跟 self.tool_filter 取交集而非覆盖: caller (如 RCB runner) 显式设的
        # tool_filter 是硬约束 (限制 agent 只能用 code_tool/bash_tool), 不能被
        # task router 冲掉. 交集为空 → 保留原 tool_filter (task router 误判时不破坏).
        effective_filter: set[str] | None = self.tool_filter
        if (
            os.environ.get("HUGINN_TASK_TOOL_ROUTER", "0") == "1"
            and self._current_task
        ):
            try:
                from huginn.runtime.task_tool_router import route_tools
                available = ToolRegistry.list_tools()
                routed = route_tools(self._current_task, available)
                if routed:
                    routed_set = set(routed)
                    if self.tool_filter is not None:
                        candidate = routed_set & self.tool_filter
                        if candidate:
                            effective_filter = candidate
                        # else: 交集空 → task router 误判, 保留原 tool_filter
                    else:
                        effective_filter = routed_set
            except Exception:
                logger.debug(
                    "task_tool_router failed, fallback to tool_filter",
                    exc_info=True,
                )

        tools = []
        for name in ToolRegistry.list_tools():
            tool = ToolRegistry.get(name)
            if tool is None:
                continue
            if effective_filter is not None and name not in effective_filter:
                continue
            if name in denied:
                continue
            if isinstance(tool, HuginnTool):
                lc_tool = self._tool_adapter.adapt(
                    tool,
                    memory_manager=self.memory,
                    agent_factory=self.agent_factory,
                    max_tool_output_tokens=self.max_tool_output_tokens,
                    compression_max_tokens=self.compression_max_tokens,
                    permission_config=self._permission_config,
                    approval_callback=self._approval_callback,
                )
            else:
                lc_tool = tool
            tools.append(self._wrap_tool_with_hooks(lc_tool))
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
            except Exception:
                logger.debug("CSM state → model task mapping failed, fallback to original task", exc_info=True)
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
        except Exception:
            logger.debug("pick main fallback model failed", exc_info=True)
        return None

    # ── Graph building ────────────────────────────────────────────

    def build_graph(self) -> Any:
        if self._agent_graph is not None:
            return self._agent_graph

        try:
            from deepagents import create_deep_agent

            system_message = self._effective_system_prompt()

            # deepagents 内置 fs 工具 (ls/read_file/write_file/edit_file) 默认走
            # StateBackend — 虚拟内存 fs, write_file 返回成功但没落盘, agent 以为
            # 写了 report.md 实际没有 (σ₁₀). 换 FilesystemBackend 落真盘;
            # virtual_mode=True 把 "/report.md" 映射进 root 且禁止路径逃逸.
            from deepagents.backends import FilesystemBackend

            fs_root = str(self.workspace) if self.workspace else None
            fs_backend = FilesystemBackend(root_dir=fs_root, virtual_mode=True)

            self._agent_graph = create_deep_agent(
                name="HuginnAgent",
                model=self.select_model("agent"),
                tools=self._effective_tools(query=get_user_message()),
                system_prompt=system_message,
                checkpointer=self.checkpointer,
                backend=fs_backend,
                middleware=[
                    FixDanglingToolCallsMiddleware(),
                    DeliverableCoverageMiddleware(),
                    RateLimitMiddleware(),
                ],
            ).with_config({
                "recursion_limit": 500 if self.is_research_mode() else 250,
            })

            return self._agent_graph

        except ImportError:
            return self._build_simple_graph()

    def _build_simple_graph(self) -> Any:
        try:
            from langgraph.prebuilt import create_react_agent

            system_prompt = self._effective_system_prompt()

            # deepagents 没装时走 fallback. langgraph 1.2+ 的 pre_model_hook 支持
            # llm_input_messages key — 只作 LLM 输入, 不经 add_messages reducer,
            # 顺序保留. 比 wrap model.bind_tools 干净: 不用 fake Runnable,
            # create_react_agent 的 model 类型校验直接过.
            from huginn.agent.middlewares import (
                DeliverableCoverageMiddleware,
                FixDanglingToolCallsMiddleware,
            )
            _fix_mw = FixDanglingToolCallsMiddleware()
            _cov_mw = DeliverableCoverageMiddleware()

            def _pre_model_hook(state):
                msgs = state.get("messages", []) or []
                patched = _fix_mw._patch_messages(list(msgs))
                # DeliverableCoverage: RCB 路径没走 deepagents middleware 协议,
                # 这里手动复用它的纯函数注入 frontier/planning hint. 失败静默,
                # 不阻塞 LLM 调用. ponytail: 每次都读 INSTRUCTIONS+report 文件
                # (~1ms IO), 相比 LLM 调用 1-5s 零开销.
                try:
                    from pathlib import Path as _P
                    _cwd = _P.cwd()
                    _inst = _cwd / "INSTRUCTIONS.md"
                    _rpt = _cwd / "report" / "report.md"
                    if _inst.exists():
                        _inst_text = _inst.read_text(encoding="utf-8")
                        _qs = _cov_mw._extract_quantities(_inst_text)
                        if _qs:
                            if not _rpt.exists():
                                _hint = _cov_mw._build_planning_msg(_qs)
                                patched = [SystemMessage(content=_hint)] + patched
                            else:
                                _rpt_text = _rpt.read_text(encoding="utf-8")
                                _missing = _cov_mw._check_coverage(_inst_text, _rpt_text)
                                _gaps = _cov_mw._check_layer_gaps(_inst_text, _rpt_text)
                                if _missing or _gaps:
                                    _parts = []
                                    if _missing:
                                        _parts.append(_cov_mw._build_frontier_msg(_missing))
                                    if _gaps:
                                        _parts.append(_cov_mw._build_layer_frontier_msg(_gaps))
                                    patched = [SystemMessage(content="\n\n".join(_parts))] + patched
                except Exception:
                    pass
                return {"llm_input_messages": patched}

            agent = create_react_agent(
                model=self.select_model("agent"),
                tools=self._effective_tools(query=get_user_message()),
                prompt=SystemMessage(content=system_prompt),
                pre_model_hook=_pre_model_hook,
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
            PrivacyGuard.shared().purge_session()
        except Exception:
            logger.warning("privacy purge failed", exc_info=True)
        self._exit_stack.close()

    def __enter__(self) -> HuginnAgent:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ── Benchmark ─────────────────────────────────────────────────

    async def run_benchmark(
        self,
        suite: BenchmarkSuite | None = None,
        store_failures: bool = True,
    ) -> dict[str, Any]:
        from huginn.self_improvement import SelfImprovementLoop

        suite = suite or BenchmarkSuite().add_defaults()
        loop = SelfImprovementLoop(suite=suite, memory_manager=self.memory)
        return await loop.evaluate(self, store_failures=store_failures)

    # ── Exploration ───────────────────────────────────────────────

    async def explore(
        self,
        objective: str,
        max_iterations: int = 10,
        thread_id: str = "exploration",
    ) -> AsyncIterator[dict[str, Any]]:
        if not self.enable_exploration:
            raise RuntimeError("Exploration mode is disabled")

        from huginn.exploration.orchestrator import ExplorationOrchestrator

        orchestrator = ExplorationOrchestrator(max_parallel=3)

        initial_branches = [
            {
                "name": "baseline",
                "hypothesis": objective,
            }
        ]

        async for result in orchestrator.explore_stream(
            objective,
            initial_branches=initial_branches,
            max_iterations=max_iterations,
        ):
            yield result

    # ── Synchronous invocation ────────────────────────────────────

    def invoke(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        async def _run():
            final_state = None
            async for state in self.chat(message, thread_id):
                if isinstance(state, dict) and state.get("tool_break"):
                    final_state = state.get("state", final_state)
                elif isinstance(state, dict) and "messages" in state:
                    final_state = state
                elif final_state is None:
                    final_state = state
            return final_state

        # 用统一 helper: 已在 running loop 时跑独立线程, 否则直接 asyncio.run.
        # 之前 try/except 顺序写反 — run_until_complete 抛 RuntimeError 后
        # asyncio.run 仍会因 running loop 抛错, 错误透传给 caller.
        from huginn.utils.async_bridge import run_async
        return run_async(_run())

    # ── Memory shortcuts ──────────────────────────────────────────

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> str:
        return self.memory.remember(
            content, category=category, tags=tags, importance=importance
        )

    def recall(
        self, query: str, category: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        return self.memory.recall(query, category=category, top_k=top_k)

    # ── Skills ────────────────────────────────────────────────────

    async def execute_skill(
        self,
        skill_name: str,
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from huginn.skills.registry import SkillRegistry

        skill = SkillRegistry.get(skill_name)
        if skill is None:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        return await self.skills.execute(skill, params, context or {})

    def list_skills(self, category: str | None = None) -> list[str]:
        from huginn.skills.registry import SkillRegistry

        return SkillRegistry.list_skills(category=category)

    # ── Parallel tool execution ───────────────────────────────────

    async def execute_tools_parallel(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        from huginn.agents.parallel_executor import ParallelToolExecutor

        tools_by_name = {t.name: t for t in self._effective_tools()}

        async def _invoke(name: str, tool_input: dict[str, Any]) -> Any:
            tool = tools_by_name.get(name)
            if tool is None:
                raise KeyError(f"tool '{name}' not registered on this agent")
            return await tool.ainvoke(tool_input)

        executor = ParallelToolExecutor(
            invoke_fn=_invoke,
            max_concurrency=5,
        )
        return await executor.execute_parallel(calls)


# Short alias for convenience: from huginn.agent import Agent
Agent = HuginnAgent
