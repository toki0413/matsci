"""HuginnAgent — the core Agent runtime.

Integrates with EvoScientist's model configuration and MCP infrastructure,
while using our own huginn tools, system prompts, exploration engine,
memory system, and skills framework.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from huginn.benchmark import BenchmarkSuite
from huginn.checkpointer import create_in_memory_checkpointer
from huginn.context_manager import (
    calculate_context_usage,
    format_context_usage,
    get_context_window,
    get_system_context,
    get_user_context,
    reset_context_cache,
)
from huginn.hooks import (
    HookContext,
    HookManager,
    POST_TOOL_USE,
    PRE_COMPACT,
    PRE_TOOL_USE,
    STOP,
    USER_PROMPT_SUBMIT,
)
from huginn.interaction.interrupt import (
    InterruptCancelled,
    get_interrupt_manager,
)
from huginn.project_memory import load_agents_md
from huginn.llm_retry import (
    FallbackTriggeredError,
    _get_retry_after,
    _is_context_overflow,
    _is_overloaded,
    _is_rate_limit,
    _is_transient_network,
    _jitter,
    _exponential_backoff,
    parse_context_overflow,
    with_retry,
)
from huginn.models.registry import create_langchain_model
from huginn.models.router import ModelRouter
from huginn.agent_config import (
    AgentConfig,
    HuginnAgentBuilder,
    _UNSET_SENTINEL,
)
from huginn.permissions import PermissionConfig
from huginn.personas import Persona


class FixDanglingToolCallsMiddleware(AgentMiddleware):
    """修补 summarization 压缩后产生的孤儿 tool_calls。

    deepagents 自带的 PatchToolCallsMiddleware 只有 before_agent,
    在 agent 开始前检查一次. 但 summarization middleware 的
    awrap_model_call 可能在 model_node 内部压缩消息时丢掉
    ToolMessage, 留下孤儿 AIMessage.tool_calls, 导致 DeepSeek
    报 400: "assistant message with 'tool_calls' must be followed
    by tool messages".

    这个 middleware 在 wrap_model_call / awrap_model_call 层拦截,
    检查 request.messages 里的孤儿 tool_calls, 补一个假的
    ToolMessage 填上, 让消息序列满足 OpenAI/DeepSeek 的格式约束.
    """

    def _patch_messages(self, messages: list) -> list:
        if not messages:
            return messages
        answered_ids = {
            getattr(msg, "tool_call_id", None)
            for msg in messages
            if hasattr(msg, "type") and msg.type == "tool"
        }
        has_orphan = any(
            tc.get("id") is not None and tc["id"] not in answered_ids
            for msg in messages
            if isinstance(msg, AIMessage)
            for tc in (*msg.tool_calls, *getattr(msg, "invalid_tool_calls", []))
        )
        if not has_orphan:
            return messages
        patched = list(messages)
        for msg in patched:
            if not isinstance(msg, AIMessage):
                continue
            for tc in (*msg.tool_calls, *getattr(msg, "invalid_tool_calls", [])):
                tc_id = tc.get("id")
                if tc_id is None or tc_id in answered_ids:
                    continue
                name = tc.get("name") or "unknown"
                content = (
                    f"Tool call {name} (id={tc_id}) was cancelled — "
                    f"summarization compaction removed its result."
                )
                patched.append(
                    ToolMessage(content=content, name=name, tool_call_id=tc_id)
                )
                answered_ids.add(tc_id)
        return patched

    def wrap_model_call(self, request, handler):
        request.messages = self._patch_messages(request.messages)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.messages = self._patch_messages(request.messages)
        return await handler(request)

    # deepagents middleware 协议要求这四个方法都存在, tool_call 层
    # 不需要修补孤儿, 直接 passthrough
    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


class RateLimitMiddleware(AgentMiddleware):
    """LLM 调用限流中间件, 在 model_call 层拦截.

    参考 Moonshine Voice 的 max_tokens_per_second 机制: 不让 agent 陷入
    无限生成烧 token. 调用前 check_allowed 拦截, 超限抛 RateLimitExceeded;
    调用后从返回的 AIMessage 上挖 usage, record_usage 记账.

    放在 middleware 链里而不是包 model, 是因为 bind_tools 返回的新对象
    不会带 guard, middleware 层拦更可靠.
    """

    def __init__(self) -> None:
        from huginn.security.rate_limiter import get_rate_limiter

        self._limiter = get_rate_limiter()

    def _estimate_tokens(self, messages: list) -> int:
        """粗估 messages 的 token 数, ~4 字符 / token."""
        total = 0
        for msg in messages or []:
            content = getattr(msg, "content", None) or str(msg)
            if not isinstance(content, str):
                content = str(content)
            total += len(content)
        return max(total // 4, 1)

    def _extract_usage(self, result: Any) -> tuple[int, int]:
        """从 model 返回结果里挖 input/output token."""
        from huginn.security.rate_limiter import _extract_usage as _extract

        return _extract(result)

    def wrap_model_call(self, request, handler):
        ok, reason = self._limiter.check_allowed(
            "agent", self._estimate_tokens(getattr(request, "messages", []))
        )
        if not ok:
            from huginn.security.rate_limiter import RateLimitExceeded

            raise RateLimitExceeded(reason, reason="limit_exceeded")
        result = handler(request)
        in_tok, out_tok = self._extract_usage(result)
        self._limiter.record_usage("agent", in_tok, out_tok)
        return result

    async def awrap_model_call(self, request, handler):
        ok, reason = self._limiter.check_allowed(
            "agent", self._estimate_tokens(getattr(request, "messages", []))
        )
        if not ok:
            from huginn.security.rate_limiter import RateLimitExceeded

            raise RateLimitExceeded(reason, reason="limit_exceeded")
        result = await handler(request)
        in_tok, out_tok = self._extract_usage(result)
        self._limiter.record_usage("agent", in_tok, out_tok)
        return result

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


from huginn.pet import PetMood, get_pet_bus
from huginn.phases import ResearchPhase
from huginn.privacy import redact_secrets, scan_for_secrets
from huginn.prompts import EXPLORATION_PROMPT, HUGINN_SYSTEM_PROMPT
from huginn.telemetry import NullTelemetryCollector, TelemetryCollector
from huginn.tools.adapter import ToolAdapter
from huginn.tools.registry import ToolRegistry
from huginn.utils.context import (
    compact_messages,
    estimate_message_tokens,
    summarize_compact_messages,
)
from huginn.utils.conversation_tree import ConversationTree
from huginn.utils.prompt_cache import PromptCacheBuilder
from huginn.utils.tokens import count_tokens, rough_token_count_for_text

logger = logging.getLogger(__name__)

# Prompt for the conversation summarizer — preserves research context.
_SUMMARY_SYSTEM_PROMPT = (
    "You are a research conversation summarizer. Condense the following "
    "conversation excerpt into a concise summary that preserves:\n"
    "1. Key decisions and their rationale\n"
    "2. Important numerical results (energies, parameters, convergence criteria)\n"
    "3. Failed approaches and why they failed\n"
    "4. Pending tasks and next steps\n"
    "5. Any file paths, structure IDs, or job IDs referenced\n"
    "Be terse — use bullet points. Do not include greetings or filler."
)

# Marker the LLM can embed in its response to request a phase transition.
# Matches things like [PHASE:VALIDATION] or [PHASE: EXECUTION].
_PHASE_MARKER = re.compile(r"\[PHASE:\s*(\w+)\s*\]", re.IGNORECASE)


class HuginnAgent:
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
        # 1) 确定 config 基底: 显式传入 > 环境变量默认
        if config is None:
            config = AgentConfig.from_env()

        # 2) 旧式扁平参数覆盖 config.
        #    - "None 表示未指定" 的参数: 非 None 才覆盖 (None 时保留 config 的 env 值)
        #    - "有真实默认值" 的参数: 非 _UNSET 才覆盖 (保持调用方显式传值优先)
        m, t, mem = config.model, config.tools, config.memory
        sec, tel, cb = config.security, config.telemetry, config.context_budget
        kg, pers, core = (
            config.knowledge_graph,
            config.personalization,
            config.core,
        )

        # None 表示未指定
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

        # 有真实默认值: 非 _UNSET 才覆盖
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

        # 3) 统一从 config 初始化
        self._init_from_config(config, scheduler=scheduler, kb_enabled=kb_enabled)

    def _init_from_config(
        self, config: AgentConfig, scheduler: Any = None, kb_enabled: Any = _UNSET_SENTINEL
    ) -> None:
        """从已解析的 AgentConfig 初始化所有实例状态.

        环境变量默认值已在 AgentConfig.from_env() 集中处理, 这里只做
        赋值与副作用, 保持与原 __init__ 完全一致的执行顺序.
        """
        m = config.model
        t = config.tools
        mem = config.memory
        sec = config.security
        tel = config.telemetry
        cb = config.context_budget
        kg = config.knowledge_graph
        pers = config.personalization
        core = config.core

        # 模型 / 路由 / 工具列表
        self.model = m.model
        self.model_router = m.model_router
        self.langchain_tools = t.tools or []

        # Resource management for persistent checkpointers and similar objects.
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

        # Research phase state machine — provides coarse-grained workflow
        # control on top of the fine-grained ReAct loop.
        from huginn.phases import PhaseManager, ResearchPhase

        self._phase_manager = PhaseManager(initial=ResearchPhase.OPEN)

        self.prompt_cache_control = m.prompt_cache_control

        self._cache_builder = PromptCacheBuilder(
            system_prompt=self.system_prompt,
            begin_dialogs=self.begin_dialogs,
            cache_control=self.prompt_cache_control,
        )
        # Detect provider so prompt-cache markers can be provider-specific.
        self._cache_builder.set_provider(self._detect_provider())

        self.enable_exploration = core.enable_exploration
        self.profile_id = core.profile_id
        self.thread_id = core.thread_id
        self.tool_filter = set(t.tool_filter) if t.tool_filter else None
        self.agent_factory = core.agent_factory
        # Central tool scheduler — shared across parent + sub-agents when injected
        # by AgentFactory. None here means the caller didn't wire one; we fall back
        # to a per-instance in-process scheduler so old callers keep working and
        # at least get the heavy/light concurrency caps locally.
        self.scheduler = scheduler
        if self.scheduler is None:
            try:
                from huginn.scheduling import AdmissionPolicy, ToolScheduler
                self.scheduler = ToolScheduler(policy=AdmissionPolicy.from_env())
            except Exception:
                # Scheduling package unavailable (import error) — degrade to no
                # admission control. _invoke_with_hooks checks for None.
                self.scheduler = None
        self._agent_graph: Any | None = None
        self._tool_description_text: str | None = None
        self._last_cache_stats: dict[str, Any] = {}
        # Accumulated conversation summary for summarization-based compaction.
        self._conversation_summary: str = ""

        # Conversation branch tree for fork/backtrack support.
        self._conversation_tree = ConversationTree()
        self._thread_branch_roots: dict[str, str] = {}

        # Telemetry is scoped to the agent instance so multi-tenant servers can
        # keep traces separate.
        self._telemetry_collector = TelemetryCollector()
        self._turn_count = 0

        self.memory_decay_enabled = mem.memory_decay_enabled
        self.memory_decay_interval_turns = mem.memory_decay_interval_turns
        self.memory_decay_prune_threshold = mem.memory_decay_prune_threshold

        # Approval / automation
        self.auto_approve = sec.auto_approve
        self._permission_config = PermissionConfig(auto_approve_all=sec.auto_approve)
        # Optional interactive approval hook — invoked by the tool adapter
        # whenever a tool lands in ASK mode. Set via constructor or
        # set_approval_callback() before registering tools.
        self._approval_callback: Callable[[str, str], bool] | None = (
            sec.approval_callback
        )

        # Tool output compression budget (separate from truncation budget)
        self.compression_max_tokens = t.compression_max_tokens

        # Telemetry
        self.telemetry_enabled = tel.telemetry_enabled
        self._telemetry_collector = (
            TelemetryCollector()
            if tel.telemetry_enabled
            else NullTelemetryCollector()
        )

        # Security layer
        self.sandbox = sec.sandbox
        self.audit = sec.audit

        # Privacy controls
        self.privacy_redact_secrets = sec.privacy_redact_secrets
        self.privacy_block_on_secrets = sec.privacy_block_on_secrets

        # Context/output budgets
        max_tool_output_tokens = t.max_tool_output_tokens
        context_budget_tokens = cb.context_budget_tokens
        # context_budget_tokens<=0 时自动按模型名推断上下文窗口
        if context_budget_tokens is not None and context_budget_tokens <= 0 and m.model is not None:
            model_name = getattr(m.model, "model_name", None) or getattr(
                m.model, "model", ""
            )
            if model_name:
                context_budget_tokens = get_context_window(str(model_name))
        self.max_tool_output_tokens = max_tool_output_tokens
        self.context_budget_tokens = context_budget_tokens
        # 记录模型上下文窗口上限, 用于实时使用率计算
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

        # Knowledge graph integration
        self.workspace = kg.workspace
        self.kg_enabled = kg.kg_enabled
        self.kg_depth = kg.kg_depth
        self.kg_top_k = kg.kg_top_k
        self._kg: Any | None = None
        # Domain knowledge base (first-principles seed docs). 默认开,
        # 可由构造器显式关掉. 懒加载避免实例化时就拉 ChromaDB.
        self.kb_enabled = True if kb_enabled is _UNSET_SENTINEL else bool(kb_enabled)
        self._kb: Any | None = None
        # 新实例清空系统上下文缓存, 避免 workspace 切换后拿到旧的 git 状态
        reset_context_cache()

        # Persona emotional trajectory
        self.persona_name = pers.persona_name
        self.emotion_tracker = pers.emotion_tracker

        # Memory integration
        memory_manager = mem.memory_manager
        if memory_manager is None:
            from huginn.memory.manager import MemoryManager

            memory_manager = MemoryManager()
        self.memory = memory_manager

        # Skills integration
        skill_executor = core.skill_executor
        if skill_executor is None:
            from huginn.skills.base import DeclarativeSkillExecutor

            skill_executor = DeclarativeSkillExecutor(ToolRegistry)
        self.skills = skill_executor

        # Each agent owns its own ToolAdapter so the summarizer isn't shared
        # across instances (which caused overwrites in multi-agent setups).
        self._tool_adapter = ToolAdapter()
        tool_summarizer = self._make_summarizer()
        if tool_summarizer is not None:
            self._tool_adapter.set_summarizer(tool_summarizer)

        # PreToolUse / PostToolUse 钩子管理器. 工具注册时会被包一层,
        # 钩子在外部通过 register_hook() 注入.
        self.hook_manager = HookManager()

        # 单步执行: 工具结果出来后暂停 chat 流, 让调用方检查 / 决定要不要继续.
        # _break_after_tool 是开关, _break_flag 是 _process_stream_state 置位、
        # chat() 消费的一次性信号.
        self._break_after_tool = t.break_after_tool
        self._break_flag = False

        # 工具调用预算: 限制单轮 chat 最多调几次工具、同一工具最多调几次.
        # None 表示不限制. agent profile 可以按需覆盖.
        self._max_tool_calls = t.max_tool_calls
        self._max_tool_calls_per_tool = t.max_tool_calls_per_tool

        # 个人定制: 学习用户语言偏好, 逐步定制通信风格.
        # None 时不启用, set_style_learner() 可后续注入.
        self.style_learner = pers.style_learner
        if pers.style_learner is not None:
            from huginn.personalization import set_shared_style_learner
            set_shared_style_learner(pers.style_learner)

        # 运行模式: "chat" (默认, 快速响应) 或 "research" (深度探索).
        # research 模式: 更高 recursion limit, 启用研究日志, 验证用独立 LLM.
        self._mode: str = "chat"

    def _make_summarizer(self):
        """Create an async callable for conversation summarization.

        Prefers the model router's cheap/summarize model to avoid burning
        expensive main-model tokens on compaction. Falls back to the main
        model, or None if no model is available.
        """
        model = None
        if self.model_router is not None:
            try:
                model = self.model_router.select("summarize", prefer_cheap=True)
            except Exception:
                pass
        if model is None:
            model = self.model
        if model is None:
            return None

        async def _summarize(transcript: str):
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=_SUMMARY_SYSTEM_PROMPT),
                HumanMessage(content=transcript),
            ]
            # Use async invocation if available, else sync
            if hasattr(model, "ainvoke"):

                async def _call():
                    return await model.ainvoke(messages)

                return await with_retry(_call, source="summarize")
            return model.invoke(messages)

        return _summarize

    # ── 运行模式 ──────────────────────────────────────────────

    def set_mode(self, mode: str) -> None:
        """切换运行模式: 'chat' 或 'research'.

        research 模式提升 recursion limit, 适合长链推理和多步验证.
        切换后需要 rebuild graph (下次 chat 时自动).
        """
        if mode not in ("chat", "research"):
            raise ValueError(f"未知模式: {mode}. 可选: chat, research")
        if mode != self._mode:
            self._mode = mode
            self._agent_graph = None  # force rebuild
            logger.info("agent mode switched to '%s'", mode)

    def get_mode(self) -> str:
        return self._mode

    def is_research_mode(self) -> bool:
        return self._mode == "research"

    # ── 专精模型 ─────────────────────────────────────────────

    def select_verification_model(self) -> Any:
        """选验证用 LLM. 有独立 verification 模型就用, 没有就退回主模型.

        参考 Moonshine 三槽: main 生成假设, verification 用不同 LLM 独立验证,
        避免"模型自己生成自己验证"的确认偏差.
        """
        if self.model_router is not None:
            try:
                return self.model_router.select_verification()
            except Exception:
                pass
        return self.select_model("reasoning")

    def has_dedicated_verification(self) -> bool:
        """是否注册了独立的验证模型."""
        if self.model_router is None:
            return False
        return self.model_router.has_dedicated_verification()

    def select_archival_model(self) -> Any:
        """选归档用 LLM. 优先便宜模型, 降低归档成本."""
        if self.model_router is not None:
            try:
                return self.model_router.select_archival()
            except Exception:
                pass
        return self.select_model("cheap")

    def _detect_provider(self) -> str | None:
        """Detect the LLM provider from the configured model."""
        if self.model is None:
            return None
        # LangChain models often expose _llm_type.
        llm_type = getattr(self.model, "_llm_type", None)
        if isinstance(llm_type, str):
            return llm_type
        # Fallback to module/class name.
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

    def _find_local_model(self) -> Any | None:
        """从 model_router 找一个本地 provider (ollama/vllm/local) 的模型.

        PrivacyGuard 的 local_only 模式用. 没配本地模型返回 None.
        """
        if self.model_router is None:
            return None
        # _models 是 router 内部字典, 这里直接读一下找本地 provider
        for entry in getattr(self.model_router, "_models", {}).values():
            m = entry.model
            llm_type = (getattr(m, "_llm_type", "") or "").lower()
            cls_name = m.__class__.__name__.lower()
            if any(
                k in llm_type or k in cls_name
                for k in ("ollama", "vllm", "local")
            ):
                return m
        return None

    @classmethod
    def from_provider(
        cls,
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        """Create a HuginnAgent from any supported provider.

        Unified factory that handles all supported LLM providers directly
        via their langchain integrations — no EvoScientist required.
        """
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
        """Create via Anthropic Claude."""
        return cls.from_provider("anthropic", model=model, api_key=api_key, **kwargs)

    @classmethod
    def from_openai(
        cls,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        """Create via OpenAI."""
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
        """Create via local Ollama."""
        return cls.from_provider("ollama", model=model, base_url=base_url, **kwargs)

    @classmethod
    def from_deepseek(
        cls,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        """Create via DeepSeek."""
        return cls.from_provider("deepseek", model=model, api_key=api_key, **kwargs)

    @classmethod
    def from_google(
        cls,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        """Create via Google GenAI (Gemini)."""
        return cls.from_provider("google-genai", model=model, api_key=api_key, **kwargs)

    @classmethod
    def from_evo_config(
        cls,
        model_name: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> HuginnAgent:
        """Create a HuginnAgent using EvoScientist's model configuration."""
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
        """Create a HuginnAgent backed by a multi-model router."""
        return cls(model_router=router, **kwargs)

    @classmethod
    def from_config(
        cls,
        config: Any,
        profile_id: str = "lead",
        **overrides: Any,
    ) -> HuginnAgent:
        """Build a HuginnAgent from a HuginnConfig.

        This is the recommended construction path: it wires up the model router,
        checkpointer, privacy settings, and tool filter from a single config
        object instead of a long list of constructor arguments.
        """
        kwargs = config.build_agent_kwargs(profile_id=profile_id)
        kwargs.update(overrides)
        return cls(**kwargs)

    def register_tool(self, tool: Any) -> None:
        """Register a HuginnTool or LangChain tool."""
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
            # Assume it's already a LangChain tool
            lc_tool = tool
        self.langchain_tools.append(self._wrap_tool_with_hooks(lc_tool))
        self._invalidate_tool_description_cache()

    def register_tools_from_registry(self) -> None:
        """Register tools from the global ToolRegistry, optionally filtered by name.

        deny 工具在这里就过滤掉, 避免主 agent 路径绕过 assembly.py 的 deny
        规则. tool_filter 白名单仍然生效, deny 是黑名单, 两者叠加.
        """
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
                # deny 规则在装配阶段就该过滤掉, 别让 LLM 看到这些工具
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

    def set_approval_callback(
        self, callback: Callable[[str, str], bool] | None
    ) -> None:
        """Set the interactive approval callback.

        Must be called before ``register_tools_from_registry`` so the
        callback is captured by each adapted tool. Calling it after tools
        are already registered has no effect on those tools — re-register
        if you need the callback to apply retroactively.
        """
        self._approval_callback = callback

    def set_style_learner(self, learner: Any | None) -> None:
        """注入 StyleLearner, 之后 chat() 会用它定制通信风格.

        同时注册到 personalization 模块的共享单例, 让 tool / route
        也能访问同一个 learner 实例. 传 None 关闭个人定制.
        """
        self.style_learner = learner
        if learner is not None:
            from huginn.personalization import set_shared_style_learner
            set_shared_style_learner(learner)

    def register_hook(self, event: str, callback: Any) -> None:
        """注册一个钩子.

        event 取 huginn.hooks 里定义的任意事件常量 (PRE_TOOL_USE /
        POST_TOOL_USE / SESSION_START / STOP / PRE_COMPACT 等).
        callback 是 async callable, 签名见 hooks.HookCallback.
        钩子在工具注册时通过 _wrap_tool_with_hooks 接入, 所以建议在
        register_tool* 之前注册; 后注册的钩子对已包装的工具同样生效
        (包装层按调用时读取 self.hook_manager).
        """
        self.hook_manager.register(event, callback)

    def _wrap_tool_with_hooks(self, tool: Any) -> Any:
        """把一个 LangChain StructuredTool 包一层 pre/post 钩子.

        异步路径(langgraph 默认)走完整钩子流程; 同步路径在没有运行中
        事件循环时也跑钩子, 否则直接转发避免嵌套 loop 报错.
        返回的是新的 StructuredTool, 原工具不动.
        """
        from langchain_core.tools import StructuredTool

        original = tool
        tool_name = getattr(original, "name", "")
        hm = self.hook_manager

        async def _invoke_with_hooks(input_data: dict) -> Any:
            allowed, modified, pre_ctx = await hm.run_pre(
                tool_name,
                input_data,
                thread_id=getattr(self, "thread_id", None),
            )
            if not allowed:
                # 钩子可在 metadata.block_reason 里塞具体原因, 喂回 LLM
                # 让它知道为什么被拦, 避免盲目重试.
                reason = pre_ctx.metadata.get("block_reason") if pre_ctx else None
                return {
                    "error": "blocked by pre_tool_use hook",
                    "block_reason": reason or "blocked by pre_tool_use hook",
                }
            if isinstance(modified, dict):
                input_data = modified
            # Scheduler admission: gate every tool call by cost_tier so the
            # heavy/light semaphores arbitrate concurrency across tools and
            # across agents sharing this scheduler. ResourceExhausted (cpu/gpu
            # budget) is surfaced like a pre_tool_use block so the LLM switches
            # to a light alternative instead of bare-retrying.
            admission = None
            sched = getattr(self, "scheduler", None)
            if sched is not None:
                cost_tier, cost = self._scheduler_cost(tool_name, input_data)
                try:
                    admission = await sched.acquire(tool_name, cost_tier, cost)
                except Exception as exc:
                    return {
                        "error": "resource_exhausted",
                        "block_reason": str(exc),
                    }
            start = time.time()
            error: BaseException | None = None
            result: Any = None
            try:
                result = await original.ainvoke(input_data)
                return result
            except Exception as exc:
                error = exc
                raise
            finally:
                if admission is not None and sched is not None:
                    try:
                        sched.release(admission)
                    except Exception:
                        # Release failure must not mask the real result/error.
                        logger.warning("scheduler release failed for %s", tool_name)
                duration_ms = (time.time() - start) * 1000
                await hm.run_post(
                    tool_name, input_data, result, error, duration_ms,
                    thread_id=getattr(self, "thread_id", None),
                    user_message=getattr(self, "_current_user_message", None),
                )

        async def hooked_coroutine(**kwargs: Any) -> Any:
            return await _invoke_with_hooks(kwargs)

        def hooked_func(**kwargs: Any) -> Any:
            # 同步调用: 没有运行中的事件循环才跑钩子(用 asyncio.run),
            # 否则直接转发 —— 在事件循环里同步调工具属于边缘场景.
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_invoke_with_hooks(kwargs))
            return original.invoke(kwargs)

        return StructuredTool.from_function(
            name=tool_name or "unnamed_tool",
            description=getattr(original, "description", "") or "",
            args_schema=getattr(original, "args_schema", None),
            coroutine=hooked_coroutine,
            func=hooked_func,
            return_direct=getattr(original, "return_direct", False),
        )

    def _scheduler_cost(
        self, tool_name: str, input_data: dict
    ) -> tuple[str, dict[str, float] | None]:
        """Best-effort (cost_tier, estimate_cost) for scheduler admission.

        Looks the tool up in the live ToolRegistry. If the tool isn't registered
        or estimate_cost raises, returns ("none", None) so the call is admitted
        without gating — the scheduler degrades gracefully rather than blocking
        on unknown tools.
        """
        try:
            from huginn.tools.registry import ToolRegistry

            t = ToolRegistry.get(tool_name)
            if t is None:
                return "none", None
            cost_tier = t.cost_tier
            try:
                cost = t.estimate_cost(input_data)
            except Exception:
                cost = None
            return cost_tier, cost
        except Exception:
            return "none", None

    def select_model(self, task: str = "agent") -> Any:
        """Select the active model for a task.

        Uses the model router when available; otherwise falls back to the
        single model configured at construction time.
        """
        if self.model_router is not None:
            return self.model_router.select(task)
        if self.model is None:
            raise RuntimeError("HuginnAgent has no model or model_router configured")
        return self.model

    def build_graph(self) -> Any:
        """Build the LangGraph agent graph."""
        if self._agent_graph is not None:
            return self._agent_graph

        # Try deepagents first (for full middleware support)
        try:
            from deepagents import create_deep_agent

            # Use the effective system prompt (base + phase prefix) as the
            # system message. Dynamic memory and the current user message
            # are injected in chat() so the cached prefix stays stable.
            system_message = self._effective_system_prompt()

            self._agent_graph = create_deep_agent(
                name="HuginnAgent",
                model=self.select_model("agent"),
                tools=self._effective_tools(),
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
        """Build a simple ReAct agent without deepagents."""
        try:
            from langgraph.prebuilt import create_react_agent

            # Use the effective prompt (base + phase prefix) as the state
            # modifier so phase-specific guidance is included.
            messages = [SystemMessage(content=self._effective_system_prompt())]

            agent = create_react_agent(
                model=self.select_model("agent"),
                tools=self._effective_tools(),
                state_modifier=messages,
                checkpointer=self.checkpointer,
            )

            self._agent_graph = agent
            return agent

        except ImportError as err:
            raise ImportError(
                "Neither deepagents nor langgraph prebuilt agents are available. "
                "Install one of them to use HuginnAgent."
            ) from err

    def _build_memory_text(self, query: str | None = None) -> str:
        """Recall relevant long-term memory formatted for the prompt tail.

        The query defaults to the current user message so recalled facts are
        actually relevant. Keeping this text out of the system prompt keeps
        the static prefix stable and improves LLM prompt/KV-cache hit rates.

        Also injects verified conjectures from the research log, so the LLM
        can see its own past hypothesis evolution tree.
        """
        if not query:
            query = "materials science computation"
        parts: list[str] = []
        try:
            mem = self.memory.recall_for_prompt(query, max_entries=3)
            if mem:
                parts.append(mem)
        except Exception:
            pass
        # 注入研究日志: 最近已验证/进行中的猜想, 让 LLM 看到自己的演化树
        try:
            from huginn.research_log import get_research_log
            log = get_research_log()
            # 拿最近 3 条 verified + 2 条 in_progress 的猜想
            verified = log.list_by_status("verified", limit=3)
            in_progress = log.list_by_status("in_progress", limit=2)
            if verified or in_progress:
                lines = ["### Research Log (recent conjectures)"]
                for r in verified:
                    lines.append(f"- [verified] {r.title}")
                for r in in_progress:
                    lines.append(f"- [in_progress] {r.title}")
                lines.append("### End Research Log")
                parts.append("\n".join(lines))
        except Exception:
            pass
        return "\n\n".join(parts) if parts else ""

    def _build_kg_text(self, query: str) -> str:
        """Query the project knowledge graph and format results for the prompt."""
        if not self.kg_enabled:
            return ""
        try:
            from huginn.kg.graph import ProjectKnowledgeGraph

            if self._kg is None:
                self._kg = ProjectKnowledgeGraph(Path(self.workspace) / ".huginn")
            result = self._kg.query(query, depth=self.kg_depth, top_k=self.kg_top_k)
            nodes = {n["id"] for n in result.get("nodes", [])}
            if not nodes:
                return ""
            text = self._kg.to_text(nodes)
            if not text:
                return ""
            return (
                "### Project Knowledge Context\n"
                "The following project-specific facts and relationships may help:\n"
                f"{text}\n"
                "### End Project Knowledge Context"
            )
        except Exception:
            return ""

    def _build_kb_text(self, query: str) -> str:
        """Query the domain knowledge base (first-principles seed docs) and
        format the retrieved chunks for the prompt. 镜像 _build_kg_text, 但走
        ChromaDB 向量检索. 任何异常都吞掉返回空串, 不影响主对话."""
        if not self.kb_enabled:
            return ""
        try:
            if self._kb is None:
                from huginn.knowledge.store import get_knowledge_base

                self._kb = get_knowledge_base(str(self.workspace))
            if self._kb.count() == 0:
                return ""
            chunks = self._kb.query(query, top_k=5)
            if not chunks:
                return ""
            lines = []
            for i, c in enumerate(chunks, 1):
                # 截断超长 chunk, 避免单条把上下文撑爆
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
                "The following first-principles reference chunks may ground your answer. "
                "Cite the source numbers when relevant.\n"
                f"{body}\n"
                "### End Domain Knowledge Context"
            )
        except Exception:
            return ""

    def _build_state_modifier(self) -> list[SystemMessage]:
        """Static system message used as the graph state modifier."""
        return self._cache_builder.build_state_modifier()

    def _build_input_messages(
        self,
        message: str,
        memory_text: str | None = None,
        kg_text: str | None = None,
        include_history: bool = True,
        kb_text: str | None = None,
    ) -> list[Any]:
        """Dynamic input messages: conversation history + memory + KG + KB + current user."""
        if memory_text is None:
            memory_text = self._build_memory_text(query=message)
        if kg_text is None:
            kg_text = self._build_kg_text(query=message)
        if kb_text is None:
            kb_text = self._build_kb_text(query=message)

        history_messages: list[Any] | None = None
        if include_history:
            history_messages = self._conversation_tree_history_to_messages()

        messages = self._cache_builder.build_input_messages(
            memory_text,
            message,
            kg_text=kg_text,
            history_messages=history_messages,
            kb_text=kb_text,
        )
        emotion_text = self._build_emotion_text(message)
        if emotion_text:
            # Insert the mood context just before the current user message so it
            # remains dynamic and does not invalidate the cached static prefix.
            messages.insert(-1, SystemMessage(content=emotion_text))
        return messages

    def _conversation_tree_history_to_messages(self) -> list[Any]:
        """Convert the active conversation path (excluding the latest user turn) to LC messages."""
        messages: list[Any] = []
        # Skip the last node because it is the current user message being handled.
        path = self._conversation_tree.active_path()
        for node_id in path[:-1]:
            node = self._conversation_tree.get_node(node_id)
            if node is None:
                continue
            meta = node.metadata or {}
            if node.role == "user":
                messages.append(HumanMessage(content=node.content))
            elif node.role == "assistant":
                tool_calls = meta.get("tool_calls")
                if tool_calls:
                    messages.append(
                        AIMessage(content=node.content, tool_calls=tool_calls)
                    )
                else:
                    messages.append(AIMessage(content=node.content))
            elif node.role == "system":
                messages.append(SystemMessage(content=node.content))
            elif node.role == "tool":
                # Reconstruct the ToolMessage from stored metadata so the LLM
                # sees the tool result paired with the originating tool_call_id.
                messages.append(
                    ToolMessage(
                        content=node.content,
                        tool_call_id=meta.get("tool_call_id", ""),
                        name=meta.get("name"),
                    )
                )
        return messages

    def _build_emotion_text(self, message: str) -> str | None:
        """Update the persona's emotional trajectory and return mood context."""
        if self.emotion_tracker is None:
            return None
        self.emotion_tracker.update_from_message(message, source="user")
        return self.emotion_tracker.context_prompt()

    def _rebuild_cache_builder(self) -> None:
        """Recreate the cache builder when the system prompt changes."""
        self._cache_builder = PromptCacheBuilder(
            system_prompt=self.system_prompt,
            begin_dialogs=self.begin_dialogs,
            cache_control=self.prompt_cache_control,
        )
        self._cache_builder.set_provider(self._detect_provider())

    def set_persona(
        self,
        persona: Persona | None = None,
        system_prompt: str | None = None,
        begin_dialogs: list[tuple[str, str]] | None = None,
        emotion_tracker: Any | None = None,
    ) -> None:
        """Switch the agent's active persona at runtime.

        Rebuilds the prompt-cache builder and invalidates the compiled graph so
        the new system prompt and begin dialogs take effect on the next turn.
        Memory and knowledge-graph state are preserved.
        """
        if persona is not None:
            self.persona_name = persona.name
            self.system_prompt = system_prompt or persona.system_prompt
            self.begin_dialogs = (
                begin_dialogs
                if begin_dialogs is not None
                else [
                    (d.get("role", "user"), d.get("content", ""))
                    for d in persona.begin_dialogs
                ]
            )
        else:
            if system_prompt is not None:
                self.system_prompt = system_prompt
            if begin_dialogs is not None:
                self.begin_dialogs = begin_dialogs
        if emotion_tracker is not None:
            self.emotion_tracker = emotion_tracker
        self._rebuild_cache_builder()
        self._agent_graph = None

    # ── Research phase management ────────────────────────────────────

    @property
    def phase(self) -> str:
        """Current research phase as a string."""
        return self._phase_manager.phase.value

    @property
    def phase_history(self) -> list[str]:
        return [p.value for p in self._phase_manager.history]

    def set_phase(self, phase: str) -> bool:
        """Transition to a new research phase.

        Returns True if the transition was allowed, False otherwise.
        Rebuilds the graph so the phase-specific prompt prefix takes effect.
        """
        from huginn.phases import ResearchPhase

        try:
            target = ResearchPhase(phase)
        except ValueError:
            return False
        if not self._phase_manager.transition(target):
            return False
        # Invalidate the graph so the new phase prompt is picked up
        self._agent_graph = None
        # phase 切换会改变 _effective_tools(), 工具 schema 缓存得跟着失效,
        # 否则 token 估算会用旧 phase 的工具集, compact 触发时机不准
        self._invalidate_tool_description_cache()
        logger.info("Research phase → %s", target.value)
        return True

    def transition_phase(self, target_phase: ResearchPhase) -> bool:
        """Transition to *target_phase* and invalidate the cached agent graph.

        Takes a ResearchPhase enum directly — use set_phase when you only
        have the phase name as a string. The compiled graph is cleared here
        so the next chat() call rebuilds it with the new phase's tool
        filter and prompt prefix.
        """
        if not self._phase_manager.transition(target_phase):
            return False
        self._agent_graph = None
        self._invalidate_tool_description_cache()
        logger.info("Research phase → %s", target_phase.value)
        return True

    def _check_phase_transition(self, ai_content: str) -> ResearchPhase | None:
        """Extract a phase transition request from the LLM's output.

        Looks for ``[PHASE:NAME]`` markers. Returns the matching
        ResearchPhase, or None if the content has no marker or the name
        is not a valid phase.
        """
        match = _PHASE_MARKER.search(ai_content)
        if not match:
            return None
        phase_name = match.group(1).upper()
        try:
            return ResearchPhase[phase_name]
        except KeyError:
            return None

    @staticmethod
    def _extract_last_ai_content(state: dict[str, Any]) -> str:
        """Pull the text of the most recent assistant message from a graph state."""
        msgs = state.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str):
                    return content
                # Some providers return a list of content blocks; join the
                # text ones so the marker scan still works.
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and "text" in block
                ]
                return "".join(parts)
        return ""

    def _effective_system_prompt(self) -> str:
        """Base system prompt + current phase prefix + transition hint + env context."""
        prefix = self._phase_manager.prompt_prefix()
        base = f"{prefix}\n\n{self.system_prompt}" if prefix else self.system_prompt
        # Tell the LLM it can drive the phase state machine itself by
        # emitting a marker. We list the valid destinations up front so
        # it does not have to guess.
        base = (
            f"{base}\n\n"
            "You can request a phase transition by including "
            "[PHASE:TARGET_PHASE] in your response. "
            "Available phases: LITERATURE, HYPOTHESIS, PLANNING, "
            "EXECUTION, VALIDATION, REPORTING."
        )
        # 注入缓存的系统上下文(日期 + git 状态)和项目上下文(.huginn.md / AGENTS.md)
        # 对话期间只计算一次, 避免每轮重复拉取 git
        workspace = self.workspace or os.getcwd()
        try:
            sys_ctx = get_system_context(str(workspace))
            if sys_ctx:
                ctx_lines = [f"{k}: {v}" for k, v in sys_ctx.items()]
                base = f"{base}\n\n# Environment\n" + "\n".join(ctx_lines)
            user_ctx = get_user_context(str(workspace))
            if user_ctx:
                base = (
                    f"{base}\n\n# Project Context\n"
                    + "\n".join(f"{k}: {v}" for k, v in user_ctx.items())
                )
            # AGENTS.md 契约 — 作为项目记忆独立注入, 与 .huginn.md 互补
            agents_md = load_agents_md(str(workspace))
            if agents_md:
                base = f"{base}\n\n# Project Memory\n{agents_md}"
        except Exception:
            # 上下文注入失败不影响主流程
            pass
        # 用户 taste profile — 问卷填完后持久化到 taste_profile.json,
        # 每轮注入让 agent 按用户思维偏好调整回答风格. 没填过返回空串.
        try:
            from huginn.personalization import get_taste_directive

            taste = get_taste_directive()
            if taste:
                base = f"{base}\n\n# User Taste Profile\n{taste}"
        except Exception:
            pass
        return base

    def _effective_tools(self) -> list[Any]:
        """Return the tool list filtered by the current research phase.

        When the phase has a tool filter (non-None), only tools whose names
        appear in the filter are exposed to the LLM. This keeps the tool
        description list short and focused, reducing prompt tokens and
        steering the model toward phase-appropriate actions.
        """
        phase_tools = self._phase_manager.tool_filter()
        if phase_tools is None:
            return self.langchain_tools
        return [
            t for t in self.langchain_tools if t.name in phase_tools
        ]

    def _tool_names_for_validation(self) -> set[str]:
        """收集当前 agent 所有工具名, 给 ToolNameValidationHook 做校验用.

        从 _effective_tools() 拿当前可见工具, 提取每个工具的 .name.
        出任何异常都返回空 set, 让钩子直接跳过, 不拖垮 chat() 主流程.
        """
        try:
            tools = self._effective_tools()
            names: set[str] = set()
            for t in tools:
                # 有的工具对象可能没 name 属性, 单个跳过别影响整体
                try:
                    name = t.name
                except Exception:
                    continue
                if isinstance(name, str) and name:
                    names.add(name)
            return names
        except Exception:
            # 拿工具列表都不该挂, 真挂了返回空 set 让钩子静默跳过
            logger.warning("_tool_names_for_validation raised", exc_info=True)
            return set()

    # ── Conversation branch management ─────────────────────────────────

    def fork_conversation(self, from_node_id: str | None = None) -> dict[str, Any]:
        """Fork the conversation from a node and return the new branch info.

        If ``from_node_id`` is None, forks from the current active leaf's
        parent, creating a sibling branch at the latest turn.
        """
        if from_node_id is None:
            # Fork from the current leaf's parent so the next message creates
            # a sibling of the current leaf.
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
        """Switch the active conversation path to end at ``node_id``."""
        ok = self._conversation_tree.set_active_leaf(node_id)
        return {
            "success": ok,
            "active_leaf": self._conversation_tree.active_leaf_id,
            "active_path": self._conversation_tree.active_path(),
        }

    def conversation_branches(self) -> dict[str, Any]:
        """Return a summary of all branches in the conversation tree."""
        return {
            "summary": self._conversation_tree.summary(),
            "branches": self._conversation_tree.get_branches(),
            "active_path": self._conversation_tree.active_path(),
        }

    def telemetry_summary(self) -> dict[str, Any]:
        """Return a coarse summary of agent telemetry spans."""
        return self._telemetry_collector.summary()

    def telemetry_spans(self) -> list[dict[str, Any]]:
        """Return all recorded telemetry spans as dicts."""
        return self._telemetry_collector.to_dict()

    def close(self) -> None:
        """Release resources held by the agent (checkpointer, etc.).

        也触发隐私数据清除: purge_session() 清理临时数据注册表,
        让 ephemeral tier 的数据不会在会话结束后残留.
        """
        # 隐私生命周期收尾: 清除临时数据
        try:
            from huginn.privacy_guard import PrivacyGuard
            PrivacyGuard.shared().purge_session()
        except Exception:
            pass
        self._exit_stack.close()

    def __enter__(self) -> HuginnAgent:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    async def run_benchmark(
        self,
        suite: BenchmarkSuite | None = None,
        store_failures: bool = True,
    ) -> dict[str, Any]:
        """Run a benchmark suite and optionally memorize failures."""
        from huginn.benchmark import SelfImprovementLoop

        suite = suite or BenchmarkSuite().add_defaults()
        loop = SelfImprovementLoop(suite=suite, memory_manager=self.memory)
        return await loop.evaluate(self, store_failures=store_failures)

    def _get_tool_description_text(self) -> str:
        """拼接所有生效工具的完整 JSON schema, 用于 token 估算.

        必须序列化完整 schema (name + description + parameters), 只拼
        description 会低估约 10 倍, compact 触发时机过晚, 长对话直接撞上
        context_length_exceeded. _effective_tools() 已经按 phase 过滤过,
        拿到的就是 LLM 实际看到的工具列表.
        """
        if self._tool_description_text is None:
            import json as _json
            parts = []
            for t in self._effective_tools():
                args_schema = getattr(t, "args_schema", None)
                if args_schema is not None and hasattr(
                    args_schema, "model_json_schema"
                ):
                    params = args_schema.model_json_schema()
                else:
                    params = {}
                schema = {
                    "name": getattr(t, "name", ""),
                    "description": getattr(t, "description", "") or "",
                    "parameters": params,
                }
                parts.append(_json.dumps(schema, ensure_ascii=False))
            self._tool_description_text = "\n".join(parts)
        return self._tool_description_text

    def _invalidate_tool_description_cache(self) -> None:
        """Invalidate cached tool descriptions when tools change."""
        self._tool_description_text = None

    def _extract_cache_stats(self, messages: list[Any]) -> dict[str, Any]:
        """Extract provider cache-hit telemetry from the latest assistant turn."""
        stats: dict[str, Any] = {}
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            meta = getattr(msg, "response_metadata", None) or {}
            for key in (
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "input_tokens",
                "output_tokens",
            ):
                if key in meta:
                    stats[key] = meta[key]
            usage = meta.get("usage")
            if isinstance(usage, dict):
                for k, v in usage.items():
                    stats[f"usage_{k}"] = v
            if stats:
                break
        return stats

    def _process_stream_state(
        self,
        state: dict[str, Any],
        turn_span: Any,
        thread_id: str,
        pet: Any,
    ) -> None:
        """Update memory, branch tree, telemetry, and pet status from one graph state."""
        msgs = state.get("messages", [])
        for msg in msgs:
            if isinstance(msg, AIMessage):
                self.memory.add_message("assistant", msg.content)
                # Stash tool_calls in node metadata so history rebuild can
                # reconstruct the AIMessage with its tool calls intact.
                meta: dict[str, Any] = {}
                if getattr(msg, "tool_calls", None):
                    meta["tool_calls"] = msg.tool_calls
                self._conversation_tree.add_message(
                    "assistant", msg.content, metadata=meta
                )
                # Track tool calls
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        self.memory.add_tool_call(
                            tool_name=tc.get("name", "unknown"),
                            input_args=tc.get("args", {}),
                        )
            elif isinstance(msg, ToolMessage):
                self.memory.add_message("tool", msg.content)
                self._conversation_tree.add_message(
                    "tool",
                    msg.content,
                    metadata={
                        "tool_call_id": msg.tool_call_id,
                        "name": getattr(msg, "name", None),
                    },
                )
                # 单步模式: 工具结果落袋后置位, 让 chat() 在 yield 完这轮 state
                # 之后停下来, 把控制权交回调用方.
                if self._break_after_tool:
                    self._break_flag = True
        # Extract provider cache-hit telemetry from the latest turn.
        cache_stats = self._extract_cache_stats(msgs)
        if cache_stats:
            self._last_cache_stats = cache_stats
            turn_span.metadata.update(cache_stats)
            pet.publish(
                PetMood.SUCCESS,
                "Turn complete",
                {"thread_id": thread_id, **cache_stats},
            )

    def _extract_usage_tokens(self) -> dict[str, int]:
        """从最后一次 LLM 调用的 cache_stats 里抽 token 用量.

        Anthropic 风格的 metadata 把 input_tokens 放顶层,
        OpenAI 风格塞在 usage 子 dict 里 (会被 _extract_cache_stats 加 usage_ 前缀).
        两种都兼容.
        """
        stats = self._last_cache_stats or {}
        usage: dict[str, int] = {}
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            if key in stats:
                usage[key] = int(stats[key] or 0)
            elif f"usage_{key}" in stats:
                usage[key] = int(stats[f"usage_{key}"] or 0)
        return usage

    async def _check_loop_interrupt(self, thread_id: str) -> dict[str, Any] | None:
        """在 chat 循环每轮 state 之间检查用户干预.

        返回值:
        - None: 没有干预, 继续往下走
        - {"cancelled": True, "reason": str}: 用户取消, 调用方应抛 InterruptCancelled
        - {"modified": True, "message": str}: 用户修改, 消息已存到 memory,
          下一轮 LLM 调用会通过 memory recall 看到

        pause 不在这里返回, 而是直接 await wait_if_paused 阻塞. 没干预时
        只多一次 dict 查询, 零开销.
        """
        try:
            mgr = get_interrupt_manager()
        except Exception:
            # interaction 模块加载失败不影响主流程
            return None
        # 先处理 pause (阻塞到 resume), 再查 cancel/modify
        await mgr.wait_if_paused(thread_id)
        evt = mgr.check_interrupt(thread_id)
        if evt is None:
            return None
        if evt.type == "cancel":
            return {"cancelled": True, "reason": evt.message or "cancelled by user"}
        if evt.type == "modify":
            # 把用户的修改意见存到 memory, 下一轮 chat 时 _build_memory_text
            # 会 recall 到, LLM 能看到. 不直接改 graph state (改不了).
            try:
                self.memory.add_message("user", f"[中途修改] {evt.message}")
                self._conversation_tree.add_message(
                    "user", f"[中途修改] {evt.message}"
                )
            except Exception:
                pass
            return {"modified": True, "message": evt.message}
        return None

    async def _maybe_auto_compact(
        self,
        final_state: dict[str, Any],
        turn_span: Any,
        thread_id: str,
    ) -> None:
        """检测上下文使用率, 超 70% 触发 PRE_COMPACT 钩子 + promote session summary.

        compact 本身不删消息列表 (那是 summarize_compact_messages 的活儿,
        在下一轮 chat 开头做), 这里只是把会话摘要存到长期记忆,
        并刷新 self._conversation_summary 让下一轮压缩更激进.
        """
        if self._model_context_window <= 0:
            return

        usage = self._extract_usage_tokens()
        if not any(usage.values()):
            return

        before = calculate_context_usage(usage, self._model_context_window)
        if before["used"] <= 70:
            return

        logger.info(
            "Context usage %d%%, triggering auto-compact",
            before["used"],
        )

        # 压缩前先让钩子知道, 给它们机会记日志 / 做准备
        pre_ctx = HookContext(
            tool_name="context_compact",
            metadata={
                "before_pct": before["used"],
                "usage": usage,
                "thread_id": thread_id,
            },
        )
        try:
            await self.hook_manager.trigger(PRE_COMPACT, pre_ctx)
        except Exception:
            logger.warning("PRE_COMPACT hook raised", exc_info=True)

        # 做 compact: promote session summary 到长期记忆
        try:
            summary = self.memory.promote_session_summary()
            # 把摘要并到 _conversation_summary, 下一轮 summarize_compact_messages
            # 会带着这个 existing_summary 压缩, 保留上下文连续性
            if summary:
                self._conversation_summary = (
                    f"{self._conversation_summary}\n{summary}".strip()
                    if self._conversation_summary
                    else summary
                )
        except Exception:
            logger.warning("promote_session_summary failed", exc_info=True)

        # 估算 compact 后的使用率: 只剩 system prompt + tool schema + summary
        try:
            after_tokens = (
                count_tokens(self.system_prompt)
                + count_tokens(self._get_tool_description_text())
                + count_tokens(self._conversation_summary)
            )
            after = calculate_context_usage(
                {"input_tokens": after_tokens},
                self._model_context_window,
            )
            after_pct = after["used"]
        except Exception:
            after_pct = 0

        # Belief Entropy 自适应: 读上次压缩的 h_belief, 调整下一轮参数
        try:
            from huginn.utils.belief_entropy import get_belief_entropy
            be = get_belief_entropy()
            last = getattr(be, "_last_result", None)
            if last is not None:
                if last.adaptive_keep_last_n is not None:
                    self._adaptive_keep_last_n = max(2, (
                        getattr(self, "_adaptive_keep_last_n", 6)
                        + last.adaptive_keep_last_n
                    ))
                if last.adaptive_budget_ratio is not None:
                    base_budget = getattr(self, "_adaptive_budget_ratio", 1.0)
                    self._adaptive_budget_ratio = max(
                        0.5, min(2.0, base_budget * last.adaptive_budget_ratio)
                    )
                # 高熵时额外触发 memory promote, 防丢关键信息
                if last.h_belief >= be.config.threshold_high:
                    logger.warning(
                        "high belief entropy (%.3f) after compaction, "
                        "promoting extra memory to long-term",
                        last.h_belief,
                    )
                    try:
                        self.memory.promote_session_summary(tier="long")
                    except Exception:
                        pass
        except Exception:
            pass

        logger.info(
            "Context compacted (%d%% → %d%%)",
            before["used"],
            after_pct,
        )
        turn_span.metadata["compact_before_pct"] = before["used"]
        turn_span.metadata["compact_after_pct"] = after_pct
        get_pet_bus().publish(
            PetMood.SUCCESS,
            f"Context compacted ({before['used']}% → {after_pct}%)",
            {"thread_id": thread_id},
        )

    async def chat(
        self,
        message: str,
        thread_id: str = "default",
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message to the Agent and stream responses.

        Stores messages in session memory and tracks tool calls for
        auto-promotion to long-term memory.
        """
        # 更新当前会话的 thread_id, 让 _wrap_tool_with_hooks 闭包和
        # post hook 能拿到, PRT Level 1 写 AnomalyLog 时要靠它关联会话
        self.thread_id = thread_id
        # 同时存当前用户消息, PRT Level 1 的 LLM 判定钩子要靠它识别
        # "用户给的值"和"工具返回的值"是否冲突 (DATA_CONFLICT 类别)
        self._current_user_message = message
        # Privacy scan on the raw user message.
        if self.privacy_block_on_secrets:
            found = scan_for_secrets(message)
            if found:
                labels = ", ".join(m.label for m in found)
                yield {
                    "messages": [
                        HumanMessage(content=message),
                        AIMessage(
                            content=f"I can't send this message because it may contain sensitive data: {labels}. Please remove the secrets and try again."
                        ),
                    ]
                }
                return

        if self.privacy_redact_secrets:
            message = redact_secrets(message)

        # Store user message in memory and in the branch tree.
        self.memory.add_message("user", message)
        self._conversation_tree.add_message("user", message)

        pet = get_pet_bus()
        pet.publish(PetMood.THINKING, "Thinking…", {"thread_id": thread_id})

        # Bind the per-agent telemetry collector for this turn so tool calls
        # are recorded under the right agent instance.
        from huginn.telemetry import set_telemetry_collector

        set_telemetry_collector(self._telemetry_collector)

        # 新一轮 turn: 重置单轮限流计数, 全局累计不动
        from huginn.security.rate_limiter import get_rate_limiter
        get_rate_limiter().reset_turn()

        with self._telemetry_collector.span(
            "agent_turn", thread_id=thread_id
        ) as turn_span:
            graph = self.build_graph()

            # USER_PROMPT_SUBMIT 钩子: 让外部规则在 LLM 看到用户消息前
            # 有机会注入引导(比如命中"验证/计算"关键词时强制要求走工具).
            # 钩子把引导词写进 ctx.metadata['prompt_guidance'], 这里取出来
            # 拼成 SystemMessage 插在用户消息前面. 任何钩子抛异常都不影响主流程.
            prompt_ctx = HookContext(
                tool_name="user_prompt",
                metadata={
                    "user_message": message,
                    "thread_id": thread_id,
                    # 给 ToolNameValidationHook 用: 校验用户点名的工具是否真实存在
                    "available_tools": self._tool_names_for_validation(),
                },
            )
            try:
                prompt_ctx = await self.hook_manager.trigger(
                    USER_PROMPT_SUBMIT, prompt_ctx
                )
            except Exception:
                logger.warning(
                    "USER_PROMPT_SUBMIT hook raised", exc_info=True
                )

            # Questions 机制: 信息不足时 ClarifyQuestionsHook 会在
            # metadata 里塞 clarify_questions. 此时直接 yield 追问消息
            # 不进 agent 循环, 等用户回答后再来一轮. 同 thread 只追问一次,
            # 避免循环. 用户回答后下一轮 hook 命中 _asked_threads 直接 passthrough.
            clarify_questions = prompt_ctx.metadata.get("clarify_questions")
            if clarify_questions:
                q_text = "\n".join(
                    f"{i + 1}. {q}" for i, q in enumerate(clarify_questions)
                )
                clarify_content = f"请先回答以下问题以对齐意图:\n{q_text}"
                self.memory.add_message("assistant", clarify_content)
                # 构造含 messages 的 state, 让 chat API 端点能从
                # messages[-1].content 读出追问内容, 跟正常回复路径一致.
                from langchain_core.messages import AIMessage as _AIMsg
                yield {
                    "messages": [_AIMsg(content=clarify_content)],
                    "clarify_questions": clarify_questions,
                    "needs_clarification": True,
                }
                return

            prompt_guidance = prompt_ctx.metadata.get("prompt_guidance")

            # Build input stream: begin-dialogs (static), recalled memory
            # (dynamic), then the current user message. The static system prompt
            # lives in the graph state modifier so the provider can cache the
            # prefix even when memory changes.
            memory_text = self._build_memory_text(query=message)
            kb_text = self._build_kb_text(query=message)
            messages = self._build_input_messages(
                message, memory_text=memory_text, kb_text=kb_text
            )

            # 引导词作为 SystemMessage 插在用户消息前. 多个钩子可能往
            # metadata 里塞多段, list 就用换行拼起来. 位置跟 emotion_text
            # 一致(-1 之前), 顺序保持 system / emotion / guidance / user.
            if prompt_guidance:
                guidance_text = (
                    "\n\n".join(prompt_guidance)
                    if isinstance(prompt_guidance, list)
                    else prompt_guidance
                )
                messages.insert(-1, SystemMessage(content=guidance_text))

            # 个人定制: confidence > 0.3 才注入 style directive, 避免早期瞎猜.
            # directive 作为 SystemMessage 插在用户消息前, 跟 prompt_guidance 同位置.
            if self.style_learner is not None:
                try:
                    profile = self.style_learner.get_profile()
                    if profile.confidence > 0.3:
                        directive = self.style_learner.get_style_directive()
                        if directive:
                            messages.insert(-1, SystemMessage(content=directive))
                except Exception:
                    logger.warning(
                        "style directive injection failed", exc_info=True
                    )

            # Privacy guard: 发云端前脱敏 / local_only 时切本地模型.
            # 用 try/except 包住, guard 自己挂了不能带挂 LLM 调用.
            # RuntimeError (没本地模型) 单独往上抛, 让用户看到.
            try:
                from huginn.privacy_guard import PrivacyGuard

                _pg = PrivacyGuard.shared()
                if not _pg.should_send_to_cloud():
                    # local_only: 当前 provider 是云端就切本地
                    _prov = self._detect_provider()
                    if _pg.should_use_local(_prov):
                        _local = self._find_local_model()
                        if _local is None:
                            raise RuntimeError(
                                "PrivacyGuard 处于 local_only 模式但未找到本地模型. "
                                "请用 config_wizard_tool 的 setup_local_model 配置 "
                                "ollama/vllm/llama.cpp, 或调 set_privacy "
                                "level='off'/'redact' 允许走云端."
                            )
                        self.model = _local
                        # 让 build_graph 用新模型重建, 不然还是旧的图
                        self._agent_graph = None
                        graph = self.build_graph()
                        logger.info(
                            "local_only: 切到本地模型 %s",
                            type(_local).__name__,
                        )
                else:
                    # redact / off: 给 messages 脱敏 (off 级别 redact_messages_for_cloud 直接透传)
                    messages = _pg.redact_messages_for_cloud(messages)
            except RuntimeError:
                raise
            except Exception:
                logger.warning(
                    "PrivacyGuard hook failed", exc_info=True
                )

            inputs = {"messages": messages}

            # Compact initial messages if a context budget is configured.
            # Use summarization-based compaction to preserve research context,
            # falling back to drop-oldest when no model is available.
            if self.context_budget_tokens > 0:
                summarizer = self._make_summarizer()
                if summarizer is not None:
                    # Belief Entropy 自适应: 用上一轮的熵信号调 keep_last_n
                    adaptive_kln = getattr(self, "_adaptive_keep_last_n", 4)
                    # budget 也按自适应比例缩放
                    adaptive_budget = int(
                        self.context_budget_tokens
                        * getattr(self, "_adaptive_budget_ratio", 1.0)
                    )
                    inputs["messages"], self._conversation_summary = (
                        await summarize_compact_messages(
                            inputs["messages"],
                            adaptive_budget,
                            keep_last_n=adaptive_kln,
                            summarizer=summarizer,
                            existing_summary=self._conversation_summary,
                        )
                    )
                else:
                    inputs["messages"] = compact_messages(
                        inputs["messages"],
                        self.context_budget_tokens,
                        keep_last_n=1,
                    )
                # Pre-flight estimate including system prompt and tool schemas.
                estimated = (
                    count_tokens(self.system_prompt)
                    + estimate_message_tokens(inputs["messages"])
                    + count_tokens(self._get_tool_description_text())
                )
                if estimated > self.context_budget_tokens:
                    get_pet_bus().publish(
                        PetMood.ERROR,
                        f"Context budget warning: ~{estimated} tokens",
                        {"budget": self.context_budget_tokens},
                    )
                # 用 context_manager 格式化使用率, 方便排障
                if self._model_context_window > 0:
                    logger.info(
                        "context usage: %s",
                        format_context_usage(
                            {"input_tokens": estimated},
                            self._model_context_window,
                        ),
                    )

            config = {
                "configurable": {"thread_id": thread_id},
                # deepagents 多层 middleware (subagents/summarization/filesystem)
                # 每个 turn 会消耗多个 graph step, 默认 100 不够用, 提到 250
                # 否则复杂推理场景会 GraphRecursionError 中断
                "recursion_limit": 250,
            }

            # The synchronous SqliteSaver does not implement async checkpoint
            # methods, so we fall back to the synchronous graph.stream() path
            # for that backend.
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver

                use_sync_stream = isinstance(self.checkpointer, SqliteSaver)
            except Exception:
                use_sync_stream = False

            # 给本轮 chat 挂上工具调用预算，防 agent 在某个工具上死循环
            from huginn.agents.loop_detector import LoopDetector
            from huginn.agents.tool_budget import ToolCallBudget
            from huginn.agents.tool_call_router import ToolCallRouter

            turn_budget = ToolCallBudget(
                max_calls=self._max_tool_calls,
                max_per_tool=self._max_tool_calls_per_tool,
            )
            self._tool_adapter.set_budget(turn_budget)
            # 同轮挂上最简路径路由: 重型工具调用前 sanity check 轻量路径
            turn_router = ToolCallRouter(budget=turn_budget)
            self._tool_adapter.set_router(turn_router)
            # 同轮挂上循环检测器: 抓同工具同参数连调的死循环, 跟 budget 互补
            turn_loop_detector = LoopDetector()
            self._tool_adapter.set_loop_detector(turn_loop_detector)

            # Retry the graph invocation for transient API failures. We only
            # retry before any state has been yielded — once output starts
            # flowing to the caller, re-running would duplicate it.
            # 3 次: 退避最多 1+2=3s, 比 5 次(1+2+4+8=15s) 少很多, 避免长推理场景超时
            max_retries = 3
            states_yielded = 0
            final_state: dict[str, Any] | None = None
            try:
                for attempt in range(max_retries):
                    try:
                        if use_sync_stream:
                            states = await asyncio.to_thread(
                                lambda: list(
                                    graph.stream(
                                        inputs, config, stream_mode="values"
                                    )
                                )
                            )
                            for state in states:
                                self._process_stream_state(
                                    state, turn_span, thread_id, pet
                                )
                                states_yielded += 1
                                final_state = state
                                yield state
                                # 单步模式: 工具结果出来后暂停, 把控制权交回调用方
                                if self._break_flag:
                                    self._break_flag = False
                                    yield {"tool_break": True, "state": final_state}
                                    break
                                # 中途干预检查: 每轮 state 之间看用户有没有 pause/cancel/modify
                                interrupt = await self._check_loop_interrupt(thread_id)
                                if interrupt and interrupt.get("cancelled"):
                                    raise InterruptCancelled(interrupt.get("reason", ""))
                        else:
                            async for state in graph.astream(
                                inputs, config, stream_mode="values"
                            ):
                                self._process_stream_state(
                                    state, turn_span, thread_id, pet
                                )
                                states_yielded += 1
                                final_state = state
                                yield state
                                # 单步模式: 工具结果出来后暂停, 把控制权交回调用方
                                if self._break_flag:
                                    self._break_flag = False
                                    yield {"tool_break": True, "state": final_state}
                                    break
                                # 中途干预检查: 每轮 state 之间看用户有没有 pause/cancel/modify
                                interrupt = await self._check_loop_interrupt(thread_id)
                                if interrupt and interrupt.get("cancelled"):
                                    raise InterruptCancelled(interrupt.get("reason", ""))
                        break
                    except Exception as exc:
                        # 用户主动取消不重试, 直接往上抛, 让调用方知道是 cancel
                        if isinstance(exc, InterruptCancelled):
                            raise
                        # Don't retry once we've started yielding — partial
                        # output has already been committed to the caller.
                        if states_yielded > 0:
                            raise
                        # 用 llm_retry 的分级判断替代字符串匹配, 准确识别
                        # 429/529/网络抖动/上下文溢出, 并尊重 retry-after 头
                        retryable = (
                            _is_rate_limit(exc)
                            or _is_overloaded(exc)
                            or _is_transient_network(exc)
                            or _is_context_overflow(exc)
                        )
                        if not retryable or attempt == max_retries - 1:
                            raise
                        # 429 优先用服务端给的 retry-after, 其它走指数退避+抖动
                        if _is_rate_limit(exc):
                            wait = _get_retry_after(exc)
                            if wait is None:
                                wait = _jitter(_exponential_backoff(attempt + 1))
                            else:
                                wait = _jitter(wait, jitter_ratio=0.1)
                        else:
                            wait = _jitter(_exponential_backoff(attempt + 1))
                        logger.warning(
                            "Graph invocation failed (attempt %d/%d), "
                            "retrying in %.2fs: %s",
                            attempt + 1,
                            max_retries,
                            wait,
                            exc,
                        )
                        await asyncio.sleep(wait)

                # Scan the final assistant message for a phase-transition
                # marker emitted by the LLM and apply it if valid.
                if final_state is not None:
                    # 上下文超 70% 就自动 compact, 触发 PRE_COMPACT 钩子
                    await self._maybe_auto_compact(
                        final_state, turn_span, thread_id
                    )
                    ai_content = self._extract_last_ai_content(final_state)
                    if ai_content:
                        # 个人定制: 基于实际对话学习用户语言偏好, 不瞎猜.
                        # observe 失败不影响主流程, profile 更新是 best-effort.
                        if self.style_learner is not None:
                            try:
                                self.style_learner.observe(message, ai_content)
                            except Exception:
                                logger.warning(
                                    "style_learner.observe failed",
                                    exc_info=True,
                                )
                        phase_target = self._check_phase_transition(ai_content)
                        if phase_target is not None:
                            self.transition_phase(phase_target)
                            logger.info(
                                "Phase auto-transitioned to %s",
                                phase_target.value,
                            )
            except Exception as exc:
                pet.publish(PetMood.ERROR, f"Error: {exc}", {"thread_id": thread_id})
                raise
            finally:
                # 清掉本轮的工具调用预算和路由，避免影响下一轮
                self._tool_adapter.set_budget(None)
                self._tool_adapter.set_router(None)
                # 同样清掉循环检测器
                self._tool_adapter.set_loop_detector(None)
                # STOP 事件: 一轮回复结束, 给 test_stop_hook 之类的钩子用
                try:
                    stop_ctx = HookContext(
                        tool_name="agent_turn",
                        metadata={
                            "thread_id": thread_id,
                            "workspace": self.workspace,
                        },
                    )
                    await self.hook_manager.trigger(STOP, stop_ctx)
                except Exception:
                    logger.warning("STOP hook raised", exc_info=True)
                pet.publish(PetMood.IDLE, "Ready", {"thread_id": thread_id})
                self._turn_count += 1
                if (
                    self.memory_decay_enabled
                    and self.memory_decay_interval_turns > 0
                    and self._turn_count % self.memory_decay_interval_turns == 0
                ):
                    try:
                        summary = self.memory.maintenance(
                            prune_threshold=self.memory_decay_prune_threshold
                        )
                        pet.publish(
                            PetMood.SUCCESS,
                            "Memory maintenance",
                            {"summary": summary},
                        )
                    except Exception as exc:
                        # Maintenance failures should not break the chat flow,
                        # but should be logged for observability.
                        logger.warning("Memory maintenance failed: %s", exc, exc_info=True)

    async def explore(
        self,
        objective: str,
        max_iterations: int = 10,
        thread_id: str = "exploration",
    ) -> AsyncIterator[dict[str, Any]]:
        """Run systematic design-space search via ExplorationOrchestrator.

        Delegates to the real orchestrator instead of just swapping the
        system prompt. The orchestrator drives branch creation, pruning,
        and Pareto-front convergence; we stream its progress updates back
        to the caller.
        """
        if not self.enable_exploration:
            raise RuntimeError("Exploration mode is disabled")

        from huginn.exploration.orchestrator import ExplorationOrchestrator

        orchestrator = ExplorationOrchestrator(max_parallel=3)

        # Seed the search with a single baseline branch derived from the
        # objective. The orchestrator's strategy will expand from here.
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

    def invoke(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        """Synchronous single-turn invocation."""
        import asyncio

        async def _run():
            final_state = None
            async for state in self.chat(message, thread_id):
                # 单步模式会 yield {"tool_break": True, "state": ...},
                # 拆出真实 state, 别把 marker dict 当成最终结果返回
                if isinstance(state, dict) and state.get("tool_break"):
                    final_state = state.get("state", final_state)
                else:
                    final_state = state
            return final_state

        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(_run())
        except RuntimeError:
            return asyncio.run(_run())

    # --- Memory shortcuts ---

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        importance: float = 0.5,
    ) -> str:
        """Explicitly store a fact in long-term memory."""
        return self.memory.remember(
            content, category=category, tags=tags, importance=importance
        )

    def recall(
        self, query: str, category: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Search long-term memory."""
        return self.memory.recall(query, category=category, top_k=top_k)

    # --- Skills ---

    async def execute_skill(
        self,
        skill_name: str,
        params: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a preset skill by name."""
        from huginn.skills.registry import SkillRegistry

        skill = SkillRegistry.get(skill_name)
        if skill is None:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}

        return await self.skills.execute(skill, params, context or {})

    def list_skills(self, category: str | None = None) -> list[str]:
        """List available skills."""
        from huginn.skills.registry import SkillRegistry

        return SkillRegistry.list_skills(category=category)

    # --- 并行工具执行 ---

    async def execute_tools_parallel(
        self, calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """并行执行多个独立的工具调用.

        LLM 在一次 chat 里返回多个 tool_call 时, langgraph 自带的
        ToolNode 已经会并发 dispatch —— 这条路径走 chat()/graph, 不需要
        调这个方法. 这个方法给程序化批量调用用 (比如外部脚本一次查
        10 个 mp_id), 内部用 ParallelToolExecutor + Semaphore 限流,
        受 max_concurrency 控制, 不会把外部 API 打爆.

        Args:
            calls: [{"tool_name": str, "tool_input": dict}, ...]

        Returns:
            [{"tool_name", "result", "error", "dt"}, ...], 跟 calls 等长,
            单个失败不影响其它, 错误塞进 error 字段.

        注意:
            调用方自己保证 calls 之间没有依赖 (后一个 call 的 input 不
            引用前一个的 output). 有依赖的调用应该串行跑, 或者用
            ``ParallelToolExecutor.split_by_dependency`` 先切批次.
        """
        from huginn.agents.parallel_executor import ParallelToolExecutor

        # 工具名 -> langchain tool 的查找表, 只暴露当前生效工具
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


def retry_llm_call(max_retries: int = 3, backoff: float = 1.0):
    """Decorator for retrying LLM calls with exponential backoff.

    Catches rate-limit, timeout, and transient API errors.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    retryable = any(
                        tag in msg
                        for tag in (
                            "rate limit",
                            "ratelimit",
                            "429",
                            "too many requests",
                            "timeout",
                            "temporarily unavailable",
                            "503",
                            "502",
                            "500",
                            "connection error",
                            "eof occurred",
                        )
                    )
                    if not retryable or attempt == max_retries:
                        raise
                    wait = backoff * (2 ** (attempt - 1))
                    time.sleep(wait)
            raise last_exc

        return wrapper

    return decorator


def _create_langchain_model(
    provider: str,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> Any:
    """Backwards-compatible wrapper around the model registry factory."""
    return create_langchain_model(
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        thinking=kwargs.get("thinking"),
        max_tokens=kwargs.get("max_tokens"),
    )


# 短名别名, 方便外部 from huginn.agent import Agent
Agent = HuginnAgent
