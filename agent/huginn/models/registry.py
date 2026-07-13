"""Model registry — multi-provider model pool with alias lookup and failover.

Inspired by OpenClaw's `provider/model` refs and Hermes' multi-model profiles.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from typing import Any, Literal

from huginn.config import HuginnConfig, ModelConfig, ThinkingIntensity
import logging
logger = logging.getLogger(__name__)


# ── API key rotation ────────────────────────────────────────────────
# Supports multiple keys per provider via env vars like DEEPSEEK_API_KEY_2,
# DEEPSEEK_API_KEY_3, or comma-separated DEEPSEEK_API_KEYS="k1,k2,k3".
# Round-robin per process; on rate-limit errors the caller can call
# next_key() to skip to the next one.

_KEY_LOCK = threading.Lock()
_KEY_INDEX: dict[str, int] = {}


def _collect_provider_keys(provider: str, explicit: str | None = None) -> list[str]:
    """Gather all API keys for a provider from env vars.

    Priority: explicit > comma-separated env > individual env vars.
    """
    keys: list[str] = []
    if explicit:
        keys.append(explicit)

    env_base = _PROVIDER_KEY_ENV.get(provider, "")
    if not env_base:
        return keys

    # comma-separated form: PROVIDER_KEYS="k1,k2,k3"
    plural = env_base + "S"  # e.g. DEEPSEEK_API_KEYS
    multi = os.environ.get(plural, "")
    if multi:
        for k in multi.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    # numbered form: PROVIDER_KEY_2, PROVIDER_KEY_3, ...
    for i in range(2, 6):
        k = os.environ.get(f"{env_base}_{i}", "")
        if k and k not in keys:
            keys.append(k)

    # always include the base key last if not already present
    base = os.environ.get(env_base, "")
    if base and base not in keys:
        keys.append(base)

    return keys


def pick_api_key(provider: str, explicit: str | None = None) -> str | None:
    """Return the next API key for the provider (round-robin)."""
    keys = _collect_provider_keys(provider, explicit)
    if not keys:
        return None
    if len(keys) == 1:
        return keys[0]
    with _KEY_LOCK:
        idx = _KEY_INDEX.get(provider, 0)
        _KEY_INDEX[provider] = (idx + 1) % len(keys)
        return keys[idx]


def list_provider_keys(provider: str, explicit: str | None = None) -> list[str]:
    """Return all available keys for the provider (for diagnostics)."""
    return _collect_provider_keys(provider, explicit)


def _patch_langchain_reasoning_content():
    """Forward DeepSeek's reasoning_content from OpenAI SDK delta to
    AIMessageChunk.additional_kwargs.

    LangChain's _convert_delta_to_message_chunk only extracts standard
    OpenAI fields and silently drops DeepSeek-specific reasoning_content.
    This patch intercepts the conversion and preserves the field.
    """
    try:
        from langchain_openai.chat_models import base as _base
        if getattr(_base, "_reasoning_patched", False):
            return
        _original = _base._convert_delta_to_message_chunk

        def _patched(delta, *args, **kwargs):
            chunk = _original(delta, *args, **kwargs)
            # DeepSeek puts reasoning_content in the delta dict (streaming)
            # or in model_extra (non-streaming). Check both.
            reasoning = ""
            if isinstance(delta, dict):
                reasoning = delta.get("reasoning_content", "")
            else:
                reasoning = getattr(delta, "reasoning_content", "") or ""
                if not reasoning and hasattr(delta, "model_extra"):
                    reasoning = delta.model_extra.get("reasoning_content", "")
            if reasoning:
                if chunk.additional_kwargs is None:
                    chunk.additional_kwargs = {}
                chunk.additional_kwargs["reasoning_content"] = reasoning
            return chunk

        _base._convert_delta_to_message_chunk = _patched
        _base._reasoning_patched = True
        logger.info("patched _convert_delta_to_message_chunk for reasoning_content")
    except Exception as e:
        logger.warning("could not patch reasoning_content: %s", e)


_patch_langchain_reasoning_content()


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
    "minimax",
    "openai-compatible",
    # Local LLM presets (all OpenAI-compatible)
    "lm-studio",
    "llama-cpp",
    "sglang",
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

    # DeepSeek and Qwen/DashScope support enable_thinking via extra_body
    if provider in ("deepseek", "dashscope") and isinstance(thinking, str):
        if thinking != "off":
            kwargs["extra_body"] = {"enable_thinking": True}
        return

    # Local OpenAI-compatible presets — pass enable_thinking in extra_body.
    # Most local servers (lm-studio, llama.cpp, sglang) expose thinking via
    # the same OpenAI extra_body convention, so we reuse it here.
    if provider in _LOCAL_PRESETS and isinstance(thinking, str):
        if thinking != "off":
            kwargs["extra_body"] = {"enable_thinking": True}
        return


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
    # structured_output: response_format json_object / with_structured_output
    structured_output: bool = False
    # parallel_tool_calls: model can return multiple tool_calls in one response
    parallel_tool_calls: bool = False


# 已知模型能力表. 维护时按 provider 分组, 新增模型记得补一条.
# 未列出的模型 get_model_capabilities() 会做前缀模糊匹配, 仍然命中不了
# 就返回全 False.
MODEL_CAPABILITIES: dict[str, ModelCaps] = {
    # ── Anthropic ──────────────────────────────────────────────
    # Sonnet 5: "最强 agentic Sonnet", 200K, agent 编程 63.2%
    "claude-sonnet-5": ModelCaps(
        vision=True, tools=True, reasoning=True, streaming=True
    ),
    "claude-opus-4-8": ModelCaps(
        vision=True, tools=True, reasoning=True, streaming=True
    ),
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
    # GPT-5.6 (Sol/Terra/Luna, 2026-07): 1.5M context, native agent
    "gpt-5.6": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "gpt-5.2": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "gpt-5": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "gpt-4o": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "gpt-4o-mini": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "gpt-4-turbo": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "gpt-4": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "gpt-3.5-turbo": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    # o 系列推理模型 — 2025 起支持 function calling
    "o1": ModelCaps(vision=False, tools=True, reasoning=True, streaming=False),
    "o3": ModelCaps(vision=False, tools=True, reasoning=True, streaming=False),
    "o1-mini": ModelCaps(vision=False, tools=False, reasoning=True, streaming=False),
    "o3-mini": ModelCaps(vision=False, tools=True, reasoning=True, streaming=False),
    # ── DeepSeek ───────────────────────────────────────────────
    # V4-Pro: MoE 1.6T/49B active, 1M context, multimodal
    "deepseek-v4-pro": ModelCaps(
        vision=True, tools=True, reasoning=True, streaming=True
    ),
    "deepseek-v4-flash": ModelCaps(
        vision=False, tools=True, reasoning=True, streaming=True
    ),
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
    # Gemini 3.0 Pro (2025-11): 2M context, 强多模态
    "gemini-3.0-pro": ModelCaps(
        vision=True, tools=True, reasoning=True, streaming=True
    ),
    "gemini-3.0-flash": ModelCaps(
        vision=True, tools=True, reasoning=False, streaming=True
    ),
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
    # ── Qwen / 通义 (DashScope) ───────────────────────────────
    "qwen-max": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "qwen3-max": ModelCaps(vision=False, tools=True, reasoning=True, streaming=True),
    "qwen3.5-plus": ModelCaps(vision=False, tools=True, reasoning=True, streaming=True),
    "qwen3.6-max-preview": ModelCaps(vision=False, tools=True, reasoning=True, streaming=True),
    "qwen3.6-flash": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "qwen-long": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
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
    "moonshot-v1-128k": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    "kimi-k2.5": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "kimi-k2.6": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "kimi-k2.7": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "kimi-k2-thinking": ModelCaps(vision=False, tools=True, reasoning=True, streaming=True),
    "kimi-k2-turbo-preview": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    # ── GLM (智谱) ────────────────────────────────────────────
    "glm-4": ModelCaps(vision=True, tools=True, reasoning=False, streaming=True),
    "glm-4-flash": ModelCaps(
        vision=False, tools=True, reasoning=False, streaming=True
    ),
    "glm-4.7": ModelCaps(vision=False, tools=True, reasoning=True, streaming=True),
    "glm-4.7-flash": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "glm-5": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "glm-5.1": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    "glm-5.2": ModelCaps(vision=True, tools=True, reasoning=True, streaming=True),
    # ── MiniMax ───────────────────────────────────────────────
    "MiniMax-M2.7": ModelCaps(vision=False, tools=True, reasoning=True, streaming=True),
    "MiniMax-M2.7-highspeed": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "MiniMax-M2.5": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "MiniMax-M2": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    # ── 本地多模态模型 (Ollama / vLLM / LM Studio) ────────────
    # vision=True 标记让 VisionRouter 走 NATIVE_LLM / BOTH 路径
    # 前缀模糊匹配会自动覆盖 :7b, :32b, :latest, -q4_0 等标签变体
    "qwen2.5-vl": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "qwen2-vl": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "qwen-vl": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "llava": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "llama3.2-vision": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "minicpm-v": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "minicpm-v2.6": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "internvl2": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "internvl": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "mllama": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "phi3.5-vision": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    "pixtral": ModelCaps(vision=True, tools=False, reasoning=False, streaming=True),
    # ── 本地文本模型 ──────────────────────────────────────────
    "qwen2.5": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "qwen2": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "llama3.1": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "llama3": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
    "deepseek-r1": ModelCaps(vision=False, tools=False, reasoning=True, streaming=True),
    "deepseek-v3": ModelCaps(vision=False, tools=True, reasoning=False, streaming=True),
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
        caps = MODEL_CAPABILITIES[name]
    else:
        lower = name.lower()
        for key, caps in MODEL_CAPABILITIES.items():
            if lower.startswith(key.lower()):
                break
        else:
            return ModelCaps()
    # Derive: models that support tools almost always support structured output
    # and parallel tool calls too. Avoids updating 50+ entries manually.
    # ponytail: derive in one place rather than per-entry; override explicitly
    # in MODEL_CAPABILITIES when a model deviates.
    if caps.tools and not caps.structured_output:
        caps.structured_output = True
    if caps.tools and not caps.parallel_tool_calls:
        caps.parallel_tool_calls = True
    return caps


#: OpenAI-compatible domestic providers with default base URLs and env keys.
_DOMESTIC_OPENAI_COMPATIBLE: dict[str, dict[str, str | None]] = {
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
    },
    "siliconflow": {
        "env": "SILICONFLOW_API_KEY",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "deepseek-ai/DeepSeek-V3",
    },
    "moonshot": {
        "env": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "kimi-k2.6",
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
        "default_model": "qwen3.5-plus",
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
    "minimax": {
        "env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimaxi.com/v1",
        "default_model": "MiniMax-M2.7",
    },
    "openai-compatible": {
        "env": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": None,
    },
}

#: Local LLM presets — all OpenAI-compatible, all run on localhost.
_LOCAL_PRESETS: dict[str, dict[str, str]] = {
    "lm-studio": {
        "base_url": "http://localhost:1234/v1",
        "default_model": "local-model",
    },
    "llama-cpp": {
        "base_url": "http://localhost:8080/v1",
        "default_model": "local-model",
    },
    "sglang": {
        "base_url": "http://localhost:30000/v1",
        "default_model": "default",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "default_model": "default",
    },
}

_PROVIDER_DEFAULTS: dict[ProviderT, str | None] = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o",
    "ollama": "qwen2.5:14b",
    "deepseek": "deepseek-v4-flash",
    "google-genai": "gemini-2.5-pro",
    "openrouter": "anthropic/claude-sonnet-4",
    "nvidia": "meta/llama-3.1-405b-instruct",
    "vllm": None,
    "local": None,
    "default": None,
    "siliconflow": "deepseek-ai/DeepSeek-V3",
    "moonshot": "kimi-k2.6",
    "zhipu": "glm-4-flash",
    "baichuan": "Baichuan4",
    "dashscope": "qwen3.5-plus",
    "qianfan": "ernie-4.0-turbo-8k",
    "doubao": "doubao-pro-32k",
    "hunyuan": "hunyuan-turbo",
    "minimax": "MiniMax-M2.7",
    "openai-compatible": None,
    "lm-studio": "local-model",
    "llama-cpp": "local-model",
    "sglang": "default",
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
    "minimax": "MINIMAX_API_KEY",
    "openai-compatible": "OPENAI_API_KEY",
    # Local presets don't need real keys — fill any non-empty string
    "lm-studio": "",
    "llama-cpp": "",
    "sglang": "",
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
    "minimax",
}


def is_local_provider(provider: str, base_url: str | None = None) -> bool:
    """Return True if the provider/base_url combination is local-only."""
    provider = provider.lower().strip()
    if provider == "ollama":
        return True
    if provider in ("vllm", "local"):
        return _is_local_url(base_url) if base_url else True
    if provider in _LOCAL_PRESETS:
        return True
    if provider == "openai-compatible":
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
        # Speculative decoding — vLLM only, passed through OpenAI extra_body.
        # create_langchain_model has no cfg handle, so we read the same env
        # vars that HuginnConfig.from_env uses (see config.py).
        if provider == "vllm" and os.environ.get(
            "HUGINN_SPECULATIVE_ENABLED", ""
        ).lower() == "true":
            extra_body = kwargs.get("extra_body") or {}
            extra_body["speculative_model"] = (
                os.environ.get("HUGINN_SPECULATIVE_MODEL", "") or "auto"
            )
            extra_body["num_speculative_tokens"] = int(
                os.environ.get("HUGINN_SPECULATIVE_DRAFT_TOKENS", "5")
            )
            kwargs["extra_body"] = extra_body
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
        key = pick_api_key("deepseek", api_key)
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

    if provider in _LOCAL_PRESETS:
        preset = _LOCAL_PRESETS[provider]
        resolved_base_url = base_url or preset["base_url"]
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as err:
            raise ImportError("pip install langchain-openai") from err
        return ChatOpenAI(
            model=model,
            api_key=api_key or "not-needed",
            base_url=resolved_base_url,
            temperature=temperature,
            request_timeout=_llm_request_timeout(),
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
    """Resolve API key using env fallback if not provided explicitly.

    When multiple keys are available (via _KEY_2, _KEY_3, or _KEYS env vars),
    rotates between them round-robin to distribute rate-limit load.
    """
    if api_key:
        return api_key
    return pick_api_key(provider, None)


def list_providers() -> list[dict[str, Any]]:
    """Return all supported providers with metadata for UI rendering.

    Each entry: {id, label, base_url, default_model, env_var, local, needs_key}
    """
    entries: list[dict[str, Any]] = []

    # Cloud providers with dedicated handling
    cloud_meta: list[tuple[str, str, str | None, str | None, str]] = [
        ("anthropic", "Anthropic (Claude)", None, "claude-3-5-sonnet-20241022", "ANTHROPIC_API_KEY"),
        ("openai", "OpenAI (GPT)", None, "gpt-4o", "OPENAI_API_KEY"),
        ("google-genai", "Google (Gemini)", None, "gemini-2.5-pro", "GOOGLE_API_KEY"),
        ("openrouter", "OpenRouter (聚合)", None, "anthropic/claude-sonnet-4", "OPENROUTER_API_KEY"),
        ("nvidia", "NVIDIA NIM", None, "meta/llama-3.1-405b-instruct", "NVIDIA_API_KEY"),
    ]
    for pid, label, url, model, env in cloud_meta:
        entries.append({
            "id": pid, "label": label, "base_url": url,
            "default_model": model, "env_var": env,
            "local": False, "needs_key": True,
        })

    # Domestic OpenAI-compatible providers
    for pid, cfg in _DOMESTIC_OPENAI_COMPATIBLE.items():
        if pid == "openai-compatible":
            continue
        label_map = {
            "deepseek": "DeepSeek (深度求索)",
            "siliconflow": "硅基流动 (SiliconFlow)",
            "moonshot": "Kimi (月之暗面)",
            "zhipu": "智谱 GLM",
            "baichuan": "百川",
            "dashscope": "阿里通义 (DashScope)",
            "qianfan": "百度千帆",
            "doubao": "字节豆包",
            "hunyuan": "腾讯混元",
            "minimax": "MiniMax (稀宇)",
        }
        entries.append({
            "id": pid,
            "label": label_map.get(pid, pid),
            "base_url": cfg.get("base_url"),
            "default_model": cfg.get("default_model"),
            "env_var": cfg.get("env"),
            "local": False,
            "needs_key": True,
        })

    # Local LLM presets
    entries.append({
        "id": "ollama", "label": "Ollama (本地)",
        "base_url": "http://localhost:11434",
        "default_model": "qwen2.5:14b",
        "env_var": "", "local": True, "needs_key": False,
    })
    entries.append({
        "id": "vllm", "label": "vLLM (本地)",
        "base_url": "http://localhost:8000/v1",
        "default_model": None, "env_var": "", "local": True, "needs_key": False,
    })
    for pid, preset in _LOCAL_PRESETS.items():
        label_map = {
            "lm-studio": "LM Studio (本地)",
            "llama-cpp": "llama.cpp (本地)",
            "sglang": "SGLang (本地)",
        }
        entries.append({
            "id": pid,
            "label": label_map.get(pid, pid),
            "base_url": preset["base_url"],
            "default_model": preset["default_model"],
            "env_var": "", "local": True, "needs_key": False,
        })

    # Generic OpenAI-compatible (user provides everything)
    entries.append({
        "id": "openai-compatible", "label": "自定义 (OpenAI 兼容)",
        "base_url": None, "default_model": None,
        "env_var": "OPENAI_API_KEY", "local": False, "needs_key": True,
    })

    return entries


def get_provider_info(provider: str) -> dict[str, Any] | None:
    """Get info for a single provider by id. Returns None if not found."""
    for entry in list_providers():
        if entry["id"] == provider:
            return entry
    return None


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

        resolved_key = resolve_provider_key(cfg.provider, cfg.api_key)
        # Fallback: if no key in config but credential_id is set, use CredentialStore
        if not resolved_key and cfg.credential_id:
            try:
                from huginn.security.credential_store import get_credential_store

                cred_info = get_credential_store().to_llm_info(cfg.credential_id)
                if cred_info:
                    resolved_key = cred_info.get("api_key")
                    # Credential may also override empty fields
                    if not cfg.model and cred_info.get("model"):
                        cfg = replace(cfg, model=cred_info["model"])
                    if not cfg.base_url and cred_info.get("base_url"):
                        cfg = replace(cfg, base_url=cred_info["base_url"])
            except Exception:
                logger.debug("get failed", exc_info=True)  # CredentialStore not available, fall through

        instance = create_langchain_model(
            provider=cfg.provider,  # type: ignore[arg-type]
            model_name=cfg.model,
            api_key=resolved_key,  # type: ignore[arg-type]
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
