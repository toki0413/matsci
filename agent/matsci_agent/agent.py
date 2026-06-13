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
    ):
        self.model = model
        self.langchain_tools = tools or []
        self.system_prompt = system_prompt or MATSCI_SYSTEM_PROMPT
        self.enable_exploration = enable_exploration
        self._agent_graph: Any | None = None
        
        # Security layer
        self.sandbox = sandbox
        self.audit = audit
        
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
            self.langchain_tools.append(ToolAdapter.adapt(tool))
        else:
            # Assume it's already a LangChain tool
            self.langchain_tools.append(tool)
    
    def register_tools_from_registry(self) -> None:
        """Register all tools from the global ToolRegistry."""
        adapted = ToolAdapter.adapt_registry(ToolRegistry)
        self.langchain_tools.extend(adapted)
    
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
        # Store user message in memory
        self.memory.add_message("user", message)
        
        graph = self.build_graph()
        
        inputs = {
            "messages": [HumanMessage(content=message)]
        }
        
        config = {
            "configurable": {"thread_id": thread_id},
        }
        
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


def _is_local_url(url: str | None) -> bool:
    """Check if a URL points to a local endpoint (localhost, 127.*, ::1, 0.0.0.0)."""
    if not url:
        return False
    host = url.split("://")[-1].split(":")[0].lower()
    return host in ("localhost", "0.0.0.0", "::1") or host.startswith("127.")


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
    """Create a LangChain chat model instance for the given provider."""
    provider = provider.lower().strip()
    
    defaults = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai": "gpt-4o",
        "ollama": "qwen2.5:14b",
        "deepseek": "deepseek-chat",
        "google-genai": "gemini-2.5-pro",
        "openrouter": "anthropic/claude-sonnet-4",
        "nvidia": "meta/llama-3.1-405b-instruct",
        "vllm": None,
        "local": None,
    }
    model = model_name or defaults.get(provider)
    if model is None:
        raise ValueError(
            f"Provider '{provider}' requires an explicit model name. "
            "Use --model / MATSCI_MODEL to set it."
        )
    
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("pip install langchain-anthropic")
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        return ChatAnthropic(model=model, api_key=key, temperature=temperature)
    
    if provider in ("openai", "vllm", "local"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("pip install langchain-openai")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        # Local endpoints (vLLM, LM Studio, TGI, etc.) often accept dummy/empty keys
        if not key and not _is_local_url(base_url):
            raise ValueError("OPENAI_API_KEY not set (required for non-local endpoints)")
        return ChatOpenAI(
            model=model,
            api_key=key or "not-needed",
            temperature=temperature,
            base_url=base_url,
        )
    
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("pip install langchain-ollama")
        return ChatOllama(model=model, base_url=base_url or "http://localhost:11434", temperature=temperature)
    
    if provider == "deepseek":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("pip install langchain-openai")
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url or "https://api.deepseek.com",
            temperature=temperature,
        )
    
    if provider == "google-genai":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("pip install langchain-google-genai")
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(model=model, api_key=key, temperature=temperature)
    
    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("pip install langchain-openai")
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")
        return ChatOpenAI(
            model=model,
            api_key=key,
            base_url=base_url or "https://openrouter.ai/api/v1",
            temperature=temperature,
        )
    
    if provider == "nvidia":
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError:
            raise ImportError("pip install langchain-nvidia-ai-endpoints")
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        return ChatNVIDIA(model=model, api_key=key, temperature=temperature)
    
    raise ValueError(
        f"Unsupported provider: {provider}. "
        "Supported: anthropic, openai, ollama, deepseek, google-genai, openrouter, nvidia, vllm, local"
    )
