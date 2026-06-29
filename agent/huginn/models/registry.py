"""Model registry — multi-provider model pool with alias lookup and failover.

Inspired by OpenClaw's `provider/model` refs and Hermes' multi-model profiles.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal

from huginn.config import HuginnConfig, ModelConfig, ThinkingIntensity

ProviderT = Literal[
    "anthropic",
    "openai",
    "ollama",
    "deepseek",
    "google-genai",
    "openrouter",
    "nvidia",
    "vllm",
    "local",
    "default",
    # Domestic / OpenAI-compatible providers
    "siliconflow",
    "moonshot",
    "zhipu",
    "baichuan",
    "dashscope",
    "qianfan",
    "doubao",
    "hunyuan",
    "openai-compatible",
]


# 单次 LLM API 请求超时 (秒). 不设的话 openai SDK 默认 600s, DeepSeek
# 长输出推理慢时会把整个 agent invoke timeout 吃完, 早 fail 早 retry 更稳.
# 用 HUGINN_LLM_REQUEST_TIMEOUT 覆盖, 默认 120s.
def _llm_request_timeout() -> float:
    raw = os.environ.get("HUGINN_LLM_REQUEST_TIMEOUT", "120")
    try:
        v = float(raw)
        return v if v > 0 else 120.0
    except (TypeError, ValueError):
        return 120.0


_INTENSITY_TO_ANTHROPIC_BUDGET: dict[ThinkingIntensity, int] = {
    "low": 4096,
    "medium": 16000,
    "high": 32000,
}


def _anthropic_thinking(
    thinking: ThinkingIntensity | dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Map a thinking intensity to Anthropic's thinking dict."""
    if thinking is None:
        return None
    if isinstance(thinking, dict):
        return thinking
    budget = _INTENSITY_TO_ANTHROPIC_BUDGET.get(thinking)
    if budget is None:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _is_openai_reasoning_model(model_name: str | None) -> bool:
    """Return True if the model name looks like an OpenAI reasoning model."""
    if not model_name:
        return False
    name = model_name.lower()
    for prefix in ("o1", "o3"):
        if name.startswith(prefix) or f"/{prefix}" in name:
            return True
    return False


def _apply_thinking_kwargs(
    provider: str,
    model_name: str | None,
    kwargs: dict[str, Any],
    thinking: ThinkingIntensity | dict[str, Any] | None,
    max_tokens: int | None,
) -> None:
    """Mutate ``kwargs`` to include provider-specific reasoning parameters."""
    if thinking is None:
        return

    if provider == "anthropic":
        anthropic_thinking = _anthropic_thinking(thinking)
        if anthropic_thinking:
            kwargs["thinking"] = anthropic_thinking
            budget = anthropic_thinking["budget_tokens"]
            if max_tokens is None or max_tokens <= budget:
                kwargs["max_tokens"] = budget + 4096
            else:
                kwargs["max_tokens"] = max_tokens
        return

    if provider in ("openai", "vllm", "local", "openrouter"):
        if _is_openai_reasoning_model(model_name) and isinstance(thinking, str):
            kwargs["reasoning_effort"] = thinking
        return

    # DeepSeek, Google, Ollama, NVIDIA, domestic: no standard mapping yet.


@dataclass
class ModelCaps:
    """模型能力声明, 4 个槽位.

    参考 claude-code-haha 的能力路由设计, 上层按能力筛选模型
    (带 vision 的才能看图, 带 tools 的才能调函数, 以此类推).
    未知名返回全 False (fail-closed), 避免把不支持工具调用的模型
    当成支持的.
    """

    vision: bool = False
    tools: bool = False
    reasoning: bool = False
    streaming: bool = False


# 已知模型能力表. 维护时按 provider 分组, 新增模型记得补一条.
# 未列出的模型 get_model_capabilities() 会做前缀模糊匹配, 仍然命中不了
# 就返回全 False.
MODEL_CAPABILITIES: dict[str, ModelCaps] = {
    # ── Anthropic ──────────────────────────────────────────────
    "claude-sonnet-4-20250514": ModelCaps(
        vision=True, tools=True, reasoning=True, streaming=True
    ),
    "claude-sonnet-4-6": ModelCaps(
        vision=True, tools=True, reasoning=True, streaming=True
    ),
    "claude-3-5-sonnet-20241022": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
    "claude-3-5-sonnet": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
    "claude-3-opus": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
    "claude-3-haiku": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
    # ── OpenAI ─────────────────────────────────────────────────
    "gpt-4o": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "gpt-4o-mini": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "gpt-4-turbo": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "gpt-4": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "gpt-3.5-turbo": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    # o 系列推理模型目前不支持原生 function calling
    "o1": ModelCaps(vision=False, tools=False, reasoning=True, streaming=False),
    "o3": ModelCaps(vision=False, tools=False, reasoning=True, streaming=False),
    "o1-mini": ModelCaps(vision=False, tools=False, reasoning=True, streaming=False),
    "o3-mini": ModelCaps(vision=False, tools=False, reasoning=True, streaming=False),
    # ── DeepSeek ───────────────────────────────────────────────
    "deepseek-chat": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    "deepseek-coder": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    "deepseek-reasoner": ModelCaps(
        vision=False, tools=False, reasoning=True, streaming=True
    ),
    # ── Google Gemini ──────────────────────────────────────────
    "gemini-2.5-pro": ModelCaps(
        vision=True, tools=True, reasoning=True, streaming=True
    ),
    "gemini-2.0-flash": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
    "gemini-1.5-pro": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
    "gemini-1.5-flash": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
    # ── Qwen / 通义 ────────────────────────────────────────────
    "qwen-max": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "qwen2.5:14b": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    # ── Moonshot / Kimi ───────────────────────────────────────
    "moonshot-v1-8k": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    "moonshot-v1-32k": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    # ── GLM ───────────────────────────────────────────────────
    "glm-4": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "glm-4-flash": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
}


def get_model_capabilities(model_name: str) -> ModelCaps:
    """查模型能力. 未知模型返回全 False (fail-closed).

    先精确匹配; 命中不了再按前缀模糊匹配, 处理带日期后缀 / 版本号
    的变体 (比如 "gpt-4o-2024-08-06" 命中 "gpt-4o").
    """
    if not model_name:
        return ModelCaps()
    name = model_name.strip()
    if name in MODEL_CAPABILITIES:
        return MODEL_CAPABILITIES[name]
    lower = name.lower()
    for key, caps in MODEL_CAPABILITIES.items():
        if lower.startswith(key.lower()):
            return caps
    return ModelCaps()


#: OpenAI-compatible domestic providers with default base URLs and env keys.
_DOMESTIC_OPENAI_COMPATIBLE: dict[str, dict[str, str | None]] = {
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
    },
    "siliconflow": {
        "env": "SILICONFLOW_API_KEY",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "deepseek-ai/DeepSeek-V3",
    },
    "moonshot": {
        "env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
    },
    "zhipu": {
        "env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-4-flash",
    },
    "baichuan": {
        "env": "BAICHUAN_API_KEY",
        "base_url": "https://api.baichuan-ai.com/v1",
        "default_model": "Baichuan4",
    },
    "dashscope": {
        "env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-max",
    },
    "qianfan": {
        "env": "QIANFAN_API_KEY",
        "base_url": "https://qianfan.baidubce.com/v2",
        "default_model": "ernie-4.0-turbo-8k",
    },
    "doubao": {
        "env": "DOUBAO_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-pro-32k",
    },
    "hunyuan": {
        "env": "HUNYUAN_API_KEY",
        "base_url": "https://api.hunyuan.tencentcloudapi.com/v1",
        "default_model": "hunyuan-turbo",
    },
    "openai-compatible": {
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": None,
    },
}

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
    "siliconflow": "deepseek-ai/DeepSeek-V3",
    "moonshot": "moonshot-v1-8k",
    "zhipu": "glm-4-flash",
    "baichuan": "Baichuan4",
    "dashscope": "qwen-max",
    "qianfan": "ernie-4.0-turbo-8k",
    "doubao": "doubao-pro-32k",
    "hunyuan": "hunyuan-turbo",
    "openai-compatible": None,
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
    "siliconflow": "SILICONFLOW_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "baichuan": "BAICHUAN_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "qianfan": "QIANFAN_API_KEY",
    "doubao": "DOUBAO_API_KEY",
    "hunyuan": "HUNYUAN_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
}


def _is_local_url(url: str | None) -> bool:
    if not url:
        return False
    return any(h in url for h in ("localhost", "127.0.0.1", "::1", "0.0.0.0", ":11434"))


_CLOUD_PROVIDERS: set[str] = {
    "anthropic",
    "openai",
    "deepseek",
    "google-genai",
    "openrouter",
    "nvidia",
    "siliconflow",
    "moonshot",
    "zhipu",
    "baichuan",
    "dashscope",
    "qianfan",
    "doubao",
    "hunyuan",
}


def is_local_provider(provider: str, base_url: str | None = None) -> bool:
    """Return True if the provider/base_url combination is local-only."""
    provider = provider.lower().strip()
    if provider == "ollama":
        return True
    if provider in ("vllm", "local"):
        return _is_local_url(base_url) if base_url else True
    if provider == "openai-compatible":
        # Generic OpenAI-compatible endpoint is local only when it points to localhost.
        return _is_local_url(base_url) if base_url else False
    if provider in _CLOUD_PROVIDERS:
        return _is_local_url(base_url)
    return False


def _create_openai_compatible(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    thinking: ThinkingIntensity | dict[str, Any] | None,
    max_tokens: int | None,
) -> Any:
    """Create a ChatOpenAI instance for an OpenAI-compatible provider."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as err:
        raise ImportError("pip install langchain-openai") from err

    cfg = _DOMESTIC_OPENAI_COMPATIBLE.get(provider, {})
    env_var = cfg.get("env") or "OPENAI_API_KEY"
    default_base_url = cfg.get("base_url")
    resolved_base_url = base_url or default_base_url
    if provider == "openai-compatible" and not resolved_base_url:
        raise ValueError(
            "Provider 'openai-compatible' requires an explicit base_url. "
            "Use --base-url / HUGINN_BASE_URL to set it."
        )

    key = api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(f"{env_var} not set")

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": key,
        "base_url": resolved_base_url,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    _apply_thinking_kwargs(provider, model, kwargs, thinking, max_tokens)
    kwargs["request_timeout"] = _llm_request_timeout()
    return ChatOpenAI(**kwargs)


def create_langchain_model(
    provider: ProviderT,
    model_name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.7,
    thinking: ThinkingIntensity | dict[str, Any] | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Create a LangChain chat model instance for the given provider."""
    provider = provider.lower().strip()  # type: ignore[assignment]

    # openai-compatible requires explicit model + base_url.
    if provider == "openai-compatible":
        if not model_name:
            raise ValueError(
                "Provider 'openai-compatible' requires an explicit model name. "
                "Use --model / HUGINN_MODEL to set it."
            )
        return _create_openai_compatible(
            provider=provider,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
        )

    model = model_name or _PROVIDER_DEFAULTS.get(provider)
    if model is None:
        raise ValueError(
            f"Provider '{provider}' requires an explicit model name. "
            "Use --model / HUGINN_MODEL to set it."
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as err:
            raise ImportError("pip install langchain-anthropic") from err
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": key,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _apply_thinking_kwargs(provider, model, kwargs, thinking, max_tokens)
        return ChatAnthropic(**kwargs)

    if provider in ("openai", "vllm", "local"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as err:
            raise ImportError("pip install langchain-openai") from err
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key and not _is_local_url(base_url):
            raise ValueError(
                "OPENAI_API_KEY not set (required for non-local endpoints)"
            )
        kwargs = {
            "model": model,
            "api_key": key or "not-needed",
            "temperature": temperature,
            "base_url": base_url,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _apply_thinking_kwargs(provider, model, kwargs, thinking, max_tokens)
        kwargs["request_timeout"] = _llm_request_timeout()
        return ChatOpenAI(**kwargs)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as err:
            raise ImportError("pip install langchain-ollama") from err
        return ChatOllama(
            model=model,
            base_url=base_url or "http://localhost:11434",
            temperature=temperature,
        )

    if provider == "deepseek":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as err:
            raise ImportError("pip install langchain-openai") from err
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not set")
        kwargs = {
            "model": model,
            "api_key": key,
            "base_url": base_url or "https://api.deepseek.com",
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _apply_thinking_kwargs(provider, model, kwargs, thinking, max_tokens)
        kwargs["request_timeout"] = _llm_request_timeout()
        return ChatOpenAI(**kwargs)

    if provider == "google-genai":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as err:
            raise ImportError("pip install langchain-google-genai") from err
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        kwargs = {"model": model, "api_key": key, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _apply_thinking_kwargs(provider, model, kwargs, thinking, max_tokens)
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "openrouter":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as err:
            raise ImportError("pip install langchain-openai") from err
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY not set")
        kwargs = {
            "model": model,
            "api_key": key,
            "base_url": base_url or "https://openrouter.ai/api/v1",
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _apply_thinking_kwargs(provider, model, kwargs, thinking, max_tokens)
        kwargs["request_timeout"] = _llm_request_timeout()
        return ChatOpenAI(**kwargs)

    if provider in _DOMESTIC_OPENAI_COMPATIBLE:
        return _create_openai_compatible(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            thinking=thinking,
            max_tokens=max_tokens,
        )

    if provider == "nvidia":
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
        except ImportError as err:
            raise ImportError("pip install langchain-nvidia-ai-endpoints") from err
        key = api_key or os.environ.get("NVIDIA_API_KEY")
        kwargs = {"model": model, "api_key": key, "temperature": temperature}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        _apply_thinking_kwargs(provider, model, kwargs, thinking, max_tokens)
        return ChatNVIDIA(**kwargs)

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
    thinking: ThinkingIntensity | dict[str, Any] | None = None
    max_tokens: int | None = None


class ModelRegistry:
    """Registry of configured LLM models.

    Supports alias lookup and `provider/model` string resolution.
    """

    def __init__(
        self,
        models: list[ModelConfig] | None = None,
        local_only_mode: bool = False,
    ):
        self._models: dict[str, ModelConfig] = {}
        # LRU cache for instantiated model clients — bounded to prevent
        # unbounded memory growth in long-running servers.
        from collections import OrderedDict

        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_max: int = 32
        self._local_only_mode = local_only_mode
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
            ModelRef(
                alias=m.alias,
                provider=m.provider,
                model=m.model,
                enabled=m.enabled,
                thinking=m.thinking,
                max_tokens=m.max_tokens,
            )
            for m in self._models.values()
        ]

    def _check_local_only(self, provider: str, base_url: str | None = None) -> None:
        if self._local_only_mode and not is_local_provider(provider, base_url):
            raise ValueError(
                f"Local-only mode is enabled, but '{provider}' is not a local provider. "
                "Allowed: ollama, vllm/local with loopback URL."
            )

    @staticmethod
    def _cache_key(
        alias: str,
        thinking: ThinkingIntensity | dict[str, Any] | None,
        max_tokens: int | None,
    ) -> str:
        """Include thinking/max_tokens in the cache key to avoid sharing instances."""
        parts = [alias]
        if thinking is not None:
            parts.append(
                str(thinking)
                if isinstance(thinking, str)
                else json.dumps(thinking, sort_keys=True)
            )
        if max_tokens is not None:
            parts.append(f"max_tokens={max_tokens}")
        return "|".join(parts)

    def get(
        self,
        alias: str,
        thinking: ThinkingIntensity | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Return cached LangChain model instance by alias.

        ``thinking`` and ``max_tokens`` override the configured model values.
        """
        cache_key = self._cache_key(alias, thinking, max_tokens)
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        cfg = self._models.get(alias)
        if not cfg or not cfg.enabled:
            raise ValueError(f"Model alias '{alias}' not found or disabled")
        self._check_local_only(cfg.provider, cfg.base_url)
        effective_thinking = thinking if thinking is not None else cfg.thinking
        effective_max_tokens = max_tokens if max_tokens is not None else cfg.max_tokens
        instance = create_langchain_model(
            provider=cfg.provider,  # type: ignore[arg-type]
            model_name=cfg.model,
            api_key=resolve_provider_key(cfg.provider, cfg.api_key),  # type: ignore[arg-type]
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            thinking=effective_thinking,
            max_tokens=effective_max_tokens,
        )
        self._cache[cache_key] = instance
        # Evict oldest entries if over capacity
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return instance

    def resolve(
        self,
        ref: str,
        thinking: ThinkingIntensity | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Resolve either an alias or a `provider/model` string to a LangChain model."""
        if ref in self._models:
            return self.get(ref, thinking=thinking, max_tokens=max_tokens)
        if "/" in ref:
            provider, model = ref.split("/", 1)
            provider = provider.strip()
            self._check_local_only(provider)
            return create_langchain_model(
                provider=provider,  # type: ignore[arg-type]
                model_name=model.strip(),
                thinking=thinking,
                max_tokens=max_tokens,
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
    def from_config(cls, config: HuginnConfig) -> ModelRegistry:
        local_only = config.local_only_mode
        if local_only is None:
            local_only = os.environ.get("HUGINN_LOCAL_ONLY", "0") == "1"
        return cls(models=config.models, local_only_mode=local_only)
