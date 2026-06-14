"""HuginnAgent — the core Agent runtime.

Integrates with EvoScientist's model configuration and MCP infrastructure,
while using our own huginn tools, system prompts, exploration engine,
memory system, and skills framework.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from huginn.prompts import HUGINN_SYSTEM_PROMPT, EXPLORATION_PROMPT
from huginn.tools.registry import ToolRegistry
from huginn.tools.adapter import ToolAdapter
from huginn.models.registry import create_langchain_model
from huginn.models.router import ModelRouter
from huginn.pet import get_pet_bus, PetMood
from huginn.privacy import redact_secrets, scan_for_secrets
from huginn.utils.context import compact_messages, estimate_message_tokens
from huginn.utils.tokens import rough_token_count_for_text
from huginn.utils.prompt_cache import PromptCacheBuilder


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
        enable_exploration: bool = True,
        memory_manager: Any | None = None,
        skill_executor: Any | None = None,
        sandbox: Any | None = None,
        audit: Any | None = None,
        profile_id: str = "default",
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
    ):
        self.model = model
        self.model_router = model_router
        self.langchain_tools = tools or []
        self.system_prompt = system_prompt or HUGINN_SYSTEM_PROMPT
        self.begin_dialogs = begin_dialogs or []

        if prompt_cache_control is None:
            prompt_cache_control = (
                os.environ.get("HUGINN_PROMPT_CACHE_CONTROL", "1") != "0"
            )
        self.prompt_cache_control = prompt_cache_control

        self._cache_builder = PromptCacheBuilder(
            system_prompt=self.system_prompt,
            begin_dialogs=self.begin_dialogs,
            cache_control=self.prompt_cache_control,
        )
        self.enable_exploration = enable_exploration
        self.profile_id = profile_id
        self.thread_id = thread_id
        self.tool_filter = set(tool_filter) if tool_filter else None
        self.agent_factory = agent_factory
        self._agent_graph: Any | None = None
        self._tool_description_text: str | None = None
        self._last_cache_stats: dict[str, Any] = {}

        # Security layer
        self.sandbox = sandbox
        self.audit = audit

        # Privacy controls (default to env vars if not explicitly passed)
        if privacy_redact_secrets is None:
            privacy_redact_secrets = os.environ.get("HUGINN_PRIVACY_REDACT_SECRETS", "1") != "0"
        if privacy_block_on_secrets is None:
            privacy_block_on_secrets = os.environ.get("HUGINN_PRIVACY_BLOCK_ON_SECRETS", "0") == "1"
        self.privacy_redact_secrets = privacy_redact_secrets
        self.privacy_block_on_secrets = privacy_block_on_secrets

        # Context/output budgets
        if max_tool_output_tokens is None:
            max_tool_output_tokens = int(os.environ.get("HUGINN_MAX_TOOL_OUTPUT_TOKENS", "25000"))
        if context_budget_tokens is None:
            context_budget_tokens = int(os.environ.get("HUGINN_CONTEXT_BUDGET_TOKENS", "0"))
        self.max_tool_output_tokens = max_tool_output_tokens
        self.context_budget_tokens = context_budget_tokens

        # Memory integration
        if memory_manager is None:
            from huginn.memory.manager import MemoryManager
            memory_manager = MemoryManager()
        self.memory = memory_manager

        # Skills integration
        if skill_executor is None:
            from huginn.skills.base import DeclarativeSkillExecutor
            skill_executor = DeclarativeSkillExecutor(ToolRegistry)
        self.skills = skill_executor

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
        return cls.from_provider("openai", model=model, api_key=api_key, base_url=base_url, **kwargs)
    
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
        except ImportError:
            raise ImportError("EvoScientist not installed.")
        return cls(model=model, **kwargs)

    @classmethod
    def from_model_router(
        cls,
        router: ModelRouter,
        **kwargs: Any,
    ) -> HuginnAgent:
        """Create a HuginnAgent backed by a multi-model router."""
        return cls(model_router=router, **kwargs)
    
    def register_tool(self, tool: Any) -> None:
        """Register a HuginnTool or LangChain tool."""
        from huginn.tools.base import HuginnTool

        if isinstance(tool, HuginnTool):
            self.langchain_tools.append(
                ToolAdapter.adapt(tool, max_tool_output_tokens=self.max_tool_output_tokens)
            )
        else:
            # Assume it's already a LangChain tool
            self.langchain_tools.append(tool)
        self._invalidate_tool_description_cache()

    def register_tools_from_registry(self) -> None:
        """Register tools from the global ToolRegistry, optionally filtered by name."""
        from huginn.tools.base import HuginnTool

        tools = []
        for name in ToolRegistry.list_tools():
            tool = ToolRegistry.get(name)
            if tool is None:
                continue
            if self.tool_filter is not None and name not in self.tool_filter:
                continue
            if isinstance(tool, HuginnTool):
                tools.append(
                    ToolAdapter.adapt(
                        tool,
                        memory_manager=self.memory,
                        agent_factory=self.agent_factory,
                        max_tool_output_tokens=self.max_tool_output_tokens,
                    )
                )
            else:
                tools.append(tool)
        self.langchain_tools.extend(tools)
        self._invalidate_tool_description_cache()

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

            # Use the static system prompt as the system message. Dynamic
            # memory and the current user message are injected in chat() so
            # the cached prefix stays stable across turns.
            system_message = self._build_state_modifier()[0]

            self._agent_graph = create_deep_agent(
                name="HuginnAgent",
                model=self.select_model("agent"),
                tools=self.langchain_tools,
                system_prompt=system_message,
            ).with_config({"recursion_limit": 100})

            return self._agent_graph

        except ImportError:
            return self._build_simple_graph()
    
    def _build_simple_graph(self) -> Any:
        """Build a simple ReAct agent without deepagents."""
        try:
            from langgraph.prebuilt import create_react_agent

            # Keep the state modifier static (system prompt only). Dynamic
            # memory and the current user message are supplied per-request in
            # chat() so the cached prefix is not invalidated by new facts.
            messages = self._build_state_modifier()

            agent = create_react_agent(
                model=self.select_model("agent"),
                tools=self.langchain_tools,
                state_modifier=messages,
            )

            self._agent_graph = agent
            return agent

        except ImportError:
            raise ImportError(
                "Neither deepagents nor langgraph prebuilt agents are available. "
                "Install one of them to use HuginnAgent."
            )
    
    def _build_memory_text(self, query: str | None = None) -> str:
        """Recall relevant long-term memory formatted for the prompt tail.

        The query defaults to the current user message so recalled facts are
        actually relevant. Keeping this text out of the system prompt keeps
        the static prefix stable and improves LLM prompt/KV-cache hit rates.
        """
        if not query:
            query = "materials science computation"
        try:
            return self.memory.recall_for_prompt(query, max_entries=3)
        except Exception:
            return ""

    def _build_state_modifier(self) -> list[SystemMessage]:
        """Static system message used as the graph state modifier."""
        return self._cache_builder.build_state_modifier()

    def _build_input_messages(
        self, message: str, memory_text: str | None = None
    ) -> list[Any]:
        """Dynamic input messages: memory context + current user."""
        if memory_text is None:
            memory_text = self._build_memory_text(query=message)
        return self._cache_builder.build_input_messages(memory_text, message)

    def _rebuild_cache_builder(self) -> None:
        """Recreate the cache builder when the system prompt changes."""
        self._cache_builder = PromptCacheBuilder(
            system_prompt=self.system_prompt,
            begin_dialogs=self.begin_dialogs,
            cache_control=self.prompt_cache_control,
        )

    def _get_tool_description_text(self) -> str:
        """Cached concatenation of tool descriptions for token estimation."""
        if self._tool_description_text is None:
            self._tool_description_text = " ".join(
                getattr(t, "description", "") for t in self.langchain_tools
            )
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
    
    async def chat(
        self,
        message: str,
        thread_id: str = "default",
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message to the Agent and stream responses.

        Stores messages in session memory and tracks tool calls for
        auto-promotion to long-term memory.
        """
        # Privacy scan on the raw user message.
        if self.privacy_block_on_secrets:
            found = scan_for_secrets(message)
            if found:
                labels = ", ".join(m.label for m in found)
                yield {
                    "messages": [
                        HumanMessage(content=message),
                        AIMessage(content=f"I can't send this message because it may contain sensitive data: {labels}. Please remove the secrets and try again."),
                    ]
                }
                return

        if self.privacy_redact_secrets:
            message = redact_secrets(message)

        # Store user message in memory
        self.memory.add_message("user", message)

        pet = get_pet_bus()
        pet.publish(PetMood.THINKING, "Thinking…", {"thread_id": thread_id})

        graph = self.build_graph()

        # Build input stream: begin-dialogs (static), recalled memory
        # (dynamic), then the current user message. The static system prompt
        # lives in the graph state modifier so the provider can cache the
        # prefix even when memory changes.
        memory_text = self._build_memory_text(query=message)
        messages = self._build_input_messages(message, memory_text=memory_text)

        inputs = {
            "messages": messages
        }

        # Compact initial messages if a context budget is configured.
        if self.context_budget_tokens > 0:
            inputs["messages"] = compact_messages(
                inputs["messages"],
                self.context_budget_tokens,
                keep_last_n=1,
            )
            # Pre-flight estimate including system prompt and tool schemas.
            estimated = (
                rough_token_count_for_text(self.system_prompt)
                + estimate_message_tokens(inputs["messages"])
                + rough_token_count_for_text(self._get_tool_description_text())
            )
            if estimated > self.context_budget_tokens:
                get_pet_bus().publish(
                    PetMood.ERROR,
                    f"Context budget warning: ~{estimated} tokens",
                    {"budget": self.context_budget_tokens},
                )

        config = {
            "configurable": {"thread_id": thread_id},
        }

        try:
            async for state in graph.astream(inputs, config, stream_mode="values"):
                # Track assistant/tool messages in memory
                msgs = state.get("messages", [])
                for msg in msgs:
                    if isinstance(msg, AIMessage):
                        self.memory.add_message("assistant", msg.content)
                        # Track tool calls
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                self.memory.add_tool_call(
                                    tool_name=tc.get("name", "unknown"),
                                    input_args=tc.get("args", {}),
                                )
                # Extract provider cache-hit telemetry from the latest turn.
                cache_stats = self._extract_cache_stats(msgs)
                if cache_stats:
                    self._last_cache_stats = cache_stats
                    pet.publish(
                        PetMood.SUCCESS,
                        "Turn complete",
                        {"thread_id": thread_id, **cache_stats},
                    )
                yield state
        except Exception as exc:
            pet.publish(PetMood.ERROR, f"Error: {exc}", {"thread_id": thread_id})
            raise
        finally:
            pet.publish(PetMood.IDLE, "Ready", {"thread_id": thread_id})
    
    async def explore(
        self,
        objective: str,
        thread_id: str = "exploration",
    ) -> AsyncIterator[dict[str, Any]]:
        """Enter exploration mode for systematic design-space search."""
        if not self.enable_exploration:
            raise RuntimeError("Exploration mode is disabled")
        
        exploration_prompt = self.system_prompt + "\n\n" + EXPLORATION_PROMPT

        original_prompt = self.system_prompt
        self.system_prompt = exploration_prompt
        self._rebuild_cache_builder()
        self._agent_graph = None

        try:
            async for state in self.chat(objective, thread_id):
                yield state
        finally:
            self.system_prompt = original_prompt
            self._rebuild_cache_builder()
            self._agent_graph = None
    
    def invoke(self, message: str, thread_id: str = "default") -> dict[str, Any]:
        """Synchronous single-turn invocation."""
        import asyncio
        
        async def _run():
            final_state = None
            async for state in self.chat(message, thread_id):
                final_state = state
            return final_state
        
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(_run())
        except RuntimeError:
            return asyncio.run(_run())
    
    # --- Memory shortcuts ---
    
    def remember(self, content: str, category: str = "fact", tags: list[str] | None = None, importance: float = 0.5) -> str:
        """Explicitly store a fact in long-term memory."""
        return self.memory.remember(content, category=category, tags=tags, importance=importance)
    
    def recall(self, query: str, category: str | None = None, top_k: int = 5) -> list[dict[str, Any]]:
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
) -> Any:
    """Backwards-compatible wrapper around the model registry factory."""
    return create_langchain_model(
        provider=provider,  # type: ignore[arg-type]
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
