"""MatSciAgent — the core Agent runtime.

Integrates with EvoScientist's model configuration and MCP infrastructure,
while using our own MatSci tools, system prompts, exploration engine,
memory system, and skills framework.
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from matsci_agent.prompts import MATSCI_SYSTEM_PROMPT, EXPLORATION_PROMPT
from matsci_agent.tools.registry import ToolRegistry
from matsci_agent.tools.adapter import ToolAdapter
from matsci_agent.models.registry import create_langchain_model
from matsci_agent.pet import get_pet_bus, PetMood
from matsci_agent.privacy import redact_secrets, scan_for_secrets
from matsci_agent.utils.context import compact_messages, estimate_message_tokens
from matsci_agent.utils.tokens import rough_token_count_for_text


class MatSciAgent:
    """Material Science specialized Agent.
    
    Wraps EvoScientist's model and infrastructure with MatSci-specific
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
    ):
        self.model = model
        self.langchain_tools = tools or []
        self.system_prompt = system_prompt or MATSCI_SYSTEM_PROMPT
        self.enable_exploration = enable_exploration
        self.profile_id = profile_id
        self.thread_id = thread_id
        self.tool_filter = set(tool_filter) if tool_filter else None
        self.agent_factory = agent_factory
        self._agent_graph: Any | None = None

        # Security layer
        self.sandbox = sandbox
        self.audit = audit

        # Privacy controls (default to env vars if not explicitly passed)
        if privacy_redact_secrets is None:
            privacy_redact_secrets = os.environ.get("MATSCI_PRIVACY_REDACT_SECRETS", "1") != "0"
        if privacy_block_on_secrets is None:
            privacy_block_on_secrets = os.environ.get("MATSCI_PRIVACY_BLOCK_ON_SECRETS", "0") == "1"
        self.privacy_redact_secrets = privacy_redact_secrets
        self.privacy_block_on_secrets = privacy_block_on_secrets

        # Context/output budgets
        if max_tool_output_tokens is None:
            max_tool_output_tokens = int(os.environ.get("MATSCI_MAX_TOOL_OUTPUT_TOKENS", "25000"))
        if context_budget_tokens is None:
            context_budget_tokens = int(os.environ.get("MATSCI_CONTEXT_BUDGET_TOKENS", "0"))
        self.max_tool_output_tokens = max_tool_output_tokens
        self.context_budget_tokens = context_budget_tokens

        # Memory integration
        if memory_manager is None:
            from matsci_agent.memory.manager import MemoryManager
            memory_manager = MemoryManager()
        self.memory = memory_manager

        # Skills integration
        if skill_executor is None:
            from matsci_agent.skills.base import DeclarativeSkillExecutor
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
    ) -> MatSciAgent:
        """Create a MatSciAgent from any supported provider.
        
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
    ) -> MatSciAgent:
        """Create via Anthropic Claude."""
        return cls.from_provider("anthropic", model=model, api_key=api_key, **kwargs)
    
    @classmethod
    def from_openai(
        cls,
        model: str = "gpt-5.4",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> MatSciAgent:
        """Create via OpenAI."""
        return cls.from_provider("openai", model=model, api_key=api_key, base_url=base_url, **kwargs)
    
    @classmethod
    def from_ollama(
        cls,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> MatSciAgent:
        """Create via local Ollama."""
        return cls.from_provider("ollama", model=model, base_url=base_url, **kwargs)
    
    @classmethod
    def from_deepseek(
        cls,
        model: str = "deepseek-chat",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> MatSciAgent:
        """Create via DeepSeek."""
        return cls.from_provider("deepseek", model=model, api_key=api_key, **kwargs)
    
    @classmethod
    def from_google(
        cls,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> MatSciAgent:
        """Create via Google GenAI (Gemini)."""
        return cls.from_provider("google-genai", model=model, api_key=api_key, **kwargs)
    
    @classmethod
    def from_evo_config(
        cls,
        model_name: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> MatSciAgent:
        """Create a MatSciAgent using EvoScientist's model configuration."""
        try:
            from EvoScientist.llm.models import get_chat_model
            model = get_chat_model(model=model_name, provider=provider)
        except ImportError:
            raise ImportError("EvoScientist not installed.")
        return cls(model=model, **kwargs)
    
    def register_tool(self, tool: Any) -> None:
        """Register a MatSciTool or LangChain tool."""
        from matsci_agent.tools.base import MatSciTool
        
        if isinstance(tool, MatSciTool):
            self.langchain_tools.append(
                ToolAdapter.adapt(tool, max_tool_output_tokens=self.max_tool_output_tokens)
            )
        else:
            # Assume it's already a LangChain tool
            self.langchain_tools.append(tool)
    
    def register_tools_from_registry(self) -> None:
        """Register tools from the global ToolRegistry, optionally filtered by name."""
        from matsci_agent.tools.base import MatSciTool

        tools = []
        for name in ToolRegistry.list_tools():
            tool = ToolRegistry.get(name)
            if tool is None:
                continue
            if self.tool_filter is not None and name not in self.tool_filter:
                continue
            if isinstance(tool, MatSciTool):
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
    
    def build_graph(self) -> Any:
        """Build the LangGraph agent graph."""
        if self._agent_graph is not None:
            return self._agent_graph
        
        # Try deepagents first (for full middleware support)
        try:
            from deepagents import create_deep_agent
            
            # Inject memory context into system prompt if available
            prompt = self._build_prompt_with_memory()
            
            self._agent_graph = create_deep_agent(
                name="MatSciAgent",
                model=self.model,
                tools=self.langchain_tools,
                system_prompt=prompt,
            ).with_config({"recursion_limit": 100})
            
            return self._agent_graph
            
        except ImportError:
            return self._build_simple_graph()
    
    def _build_simple_graph(self) -> Any:
        """Build a simple ReAct agent without deepagents."""
        try:
            from langgraph.prebuilt import create_react_agent
            
            prompt = self._build_prompt_with_memory()
            messages = [SystemMessage(content=prompt)]
            
            agent = create_react_agent(
                model=self.model,
                tools=self.langchain_tools,
                state_modifier=messages,
            )
            
            self._agent_graph = agent
            return agent
            
        except ImportError:
            raise ImportError(
                "Neither deepagents nor langgraph prebuilt agents are available. "
                "Install one of them to use MatSciAgent."
            )
    
    def _build_prompt_with_memory(self) -> str:
        """Build system prompt augmented with relevant long-term memory."""
        prompt = self.system_prompt
        
        # Try to recall relevant facts (simple heuristic: use last 3 words)
        try:
            recall = self.memory.recall_for_prompt("materials science computation", max_entries=2)
            if recall:
                prompt += f"\n\n{recall}"
        except Exception:
            pass
        
        return prompt
    
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

        inputs = {
            "messages": [HumanMessage(content=message)]
        }

        # Compact initial messages if a context budget is configured.
        if self.context_budget_tokens > 0:
            inputs["messages"] = compact_messages(
                inputs["messages"],
                self.context_budget_tokens,
                keep_last_n=1,
            )
            # Pre-flight estimate including system prompt and tool schemas.
            tool_desc = " ".join(getattr(t, "description", "") for t in self.langchain_tools)
            estimated = (
                rough_token_count_for_text(self.system_prompt)
                + estimate_message_tokens(inputs["messages"])
                + rough_token_count_for_text(tool_desc)
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
                    # Note: ToolMessage handling depends on langchain version
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
        self._agent_graph = None
        
        try:
            async for state in self.chat(objective, thread_id):
                yield state
        finally:
            self.system_prompt = original_prompt
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
        from matsci_agent.skills.registry import SkillRegistry
        
        skill = SkillRegistry.get(skill_name)
        if skill is None:
            return {"success": False, "error": f"Skill '{skill_name}' not found"}
        
        return await self.skills.execute(skill, params, context or {})
    
    def list_skills(self, category: str | None = None) -> list[str]:
        """List available skills."""
        from matsci_agent.skills.registry import SkillRegistry
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
