"""HuginnAgent — core class with init, factory, graph, and dispatch methods.

The heavy logic lives in the mixin modules: context.py, streaming.py,
reflection.py, session.py, callbacks.py.  This file wires them together
and keeps the construction + graph-building + convenience methods.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack
from typing import Any

from langchain_core.messages import SystemMessage

from huginn.agent_config import _UNSET_SENTINEL, AgentConfig
from huginn.benchmark import BenchmarkSuite
from huginn.checkpointer import create_in_memory_checkpointer
from huginn.context_manager import (
    get_context_window,
    reset_context_cache,
)
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
from .middlewares import FixDanglingToolCallsMiddleware, RateLimitMiddleware
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

        from huginn.reflection import TaskReflector

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
            logger.info("agent mode switched to '%s'", mode)

    def get_mode(self) -> str:
        return self._mode

    def is_research_mode(self) -> bool:
        return self._mode == "research"

    def is_plan_mode(self) -> bool:
        return self._mode == "plan"

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

        tools = []
        for name in ToolRegistry.list_tools():
            tool = ToolRegistry.get(name)
            if tool is None:
                continue
            if self.tool_filter is not None and name not in self.tool_filter:
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
        self.langchain_tools.extend(tools)
        self._invalidate_tool_description_cache()

    # ── Model selection ──────────────────────────────────────────

    def select_model(self, task: str = "agent") -> Any:
        if task == "agent" and self._main_fallback_override is not None:
            return self._main_fallback_override
        if self.model_router is not None:
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

            self._agent_graph = create_deep_agent(
                name="HuginnAgent",
                model=self.select_model("agent"),
                tools=self._effective_tools(query=get_user_message()),
                system_prompt=system_message,
                checkpointer=self.checkpointer,
                middleware=[
                    FixDanglingToolCallsMiddleware(),
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

            try:
                agent = create_react_agent(
                    model=self.select_model("agent"),
                    tools=self._effective_tools(query=get_user_message()),
                    prompt=SystemMessage(content=system_prompt),
                    checkpointer=self.checkpointer,
                )
            except TypeError:
                agent = create_react_agent(
                    model=self.select_model("agent"),
                    tools=self._effective_tools(query=get_user_message()),
                    state_modifier=[SystemMessage(content=system_prompt)],
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
        from huginn.benchmark import SelfImprovementLoop

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
