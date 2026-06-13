"""Model registry — multi-provider model pool with alias lookup and failover.

Inspired by OpenClaw's `provider/model` refs and Hermes' multi-model profiles.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from matsci_agent.config import MatSciConfig, ModelConfig

ProviderT = Literal[
    "anthropic", "openai", "ollama", "deepseek",
    "google-genai", "openrouter", "nvidia", "vllm", "local", "default"
]

_PROVIDER_DEFAULTS: dict[ProviderT, str | None] = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
    "ollama": "qwen2.5:14b",
    "deepseek": "deepseek-chat",
    "google-genai": "gemini-2.5-pro",
    "openrouter": "anthropic/claude-sonnet-4",
    "nvidia": "meta/llama-3.1-405b-instruct",
    "vllm": None,
    "local": None,
    "default": None,
}

_PROVIDER_KEY_ENV: dict[ProviderT, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "vllm": "OPENAI_API_KEY",
    "local": "OPENAI_API_KEY",
    "ollama": "",
    "default": "",
}


def _is_local_url(url: str | None) -> bool:
    if not url:
        return False
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", ":11434"))


def create_langchain_model(
    provider: ProviderT,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
) -> Any:
    """Create a LangChain chat model instance for the given provider."""
    provider = provider.lower().strip()  # type: ignore[assignment]
    model = model_name or _PROVIDER_DEFAULTS.get(provider)
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
        f"Supported: {', '.join(_PROVIDER_DEFAULTS)}"
    )


def resolve_provider_key(provider: ProviderT, api_key: str | None) -> str | None:
    """Resolve API key using env fallback if not provided explicitly."""
    if api_key:
        return api_key
    env_var = _PROVIDER_KEY_ENV.get(provider)
    return os.environ.get(env_var) if env_var else None


@dataclass
class ModelRef:
    """Lightweight record returned by ModelRegistry.list()."""
    alias: str
    provider: str
    model: str | None
    enabled: bool


class ModelRegistry:
    """Registry of configured LLM models.

    Supports alias lookup and `provider/model` string resolution.
    """

    def __init__(self, models: list[ModelConfig] | None = None):
        self._models: dict[str, ModelConfig] = {}
        self._cache: dict[str, Any] = {}
        if models:
            for m in models:
                self.register(m)

    def register(self, model: ModelConfig) -> None:
        if not model.alias:
            raise ValueError("ModelConfig must have an alias")
        self._models[model.alias] = model
        self._cache.pop(model.alias, None)

    def list(self) -> list[ModelRef]:
        return [
            ModelRef(alias=m.alias, provider=m.provider, model=m.model, enabled=m.enabled)
            for m in self._models.values()
        ]

    def get(self, alias: str) -> Any:
        """Return cached LangChain model instance by alias."""
        if alias in self._cache:
            return self._cache[alias]
        cfg = self._models.get(alias)
        if not cfg or not cfg.enabled:
            raise ValueError(f"Model alias '{alias}' not found or disabled")
        instance = create_langchain_model(
            provider=cfg.provider,  # type: ignore[arg-type]
            model_name=cfg.model,
            api_key=resolve_provider_key(cfg.provider, cfg.api_key),  # type: ignore[arg-type]
            base_url=cfg.base_url,
            temperature=cfg.temperature,
        )
        self._cache[alias] = instance
        return instance

    def resolve(self, ref: str) -> Any:
        """Resolve either an alias or a `provider/model` string to a LangChain model."""
        if ref in self._models:
            return self.get(ref)
        if "/" in ref:
            provider, model = ref.split("/", 1)
            return create_langchain_model(
                provider=provider.strip(),  # type: ignore[arg-type]
                model_name=model.strip(),
            )
        raise ValueError(f"Cannot resolve model reference: {ref}")

    def default_alias(self) -> str | None:
        """Return the first enabled alias, preferring 'default' or 'lead'."""
        for preferred in ("lead", "default"):
            if preferred in self._models and self._models[preferred].enabled:
                return preferred
        for alias, cfg in self._models.items():
            if cfg.enabled:
                return alias
        return None

    @classmethod
    def from_config(cls, config: MatSciConfig) -> "ModelRegistry":
        return cls(models=config.models)
