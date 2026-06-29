"""Health and info endpoints.

配置类路由已迁移到 ``huginn.routes.config``, 这里只保留健康检查与引导。
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter

from huginn import __version__
from huginn.config import HuginnConfig
from huginn.server_core import _check_ollama_available, get_context

router = APIRouter(tags=["health"])

# Provider → expected API key env var (matches models/registry.py)
_PROVIDER_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google-genai": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zhipu": "ZHIPU_API_KEY",
    "baichuan": "BAICHUAN_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "qianfan": "QIANFAN_API_KEY",
    "doubao": "DOUBAO_API_KEY",
    "hunyuan": "HUNYUAN_API_KEY",
}


def _is_configured(cfg: HuginnConfig) -> bool:
    """Determine whether the system has a usable LLM configuration.

    Handles three cases:
    1. Ollama (keyless): configured if provider is ollama
    2. Multi-model pool: configured if any model has a resolved key
    3. Legacy single-model: configured if provider != default and key exists
    """
    # Ollama needs no API key
    if cfg.provider == "ollama":
        return True

    # Multi-model pool
    if cfg.models:
        for m in cfg.models:
            if m.provider == "ollama":
                return True
            if m.enabled and (m.api_key or _PROVIDER_KEY_ENV.get(m.provider) and os.environ.get(_PROVIDER_KEY_ENV[m.provider])):
                return True

    # Legacy single-model
    return cfg.provider != "default" and bool(cfg.resolved_api_key)


@router.get("/health")
async def health() -> dict[str, Any]:
    cfg = HuginnConfig.from_env()
    configured = _is_configured(cfg)

    result: dict[str, Any] = {
        "status": "ok" if configured else "unconfigured",
        "version": __version__,
        "provider": cfg.provider,
        "model": cfg.model,
        "configured": configured,
    }

    # Add model pool info if available
    if cfg.models:
        result["model_pool"] = len([m for m in cfg.models if m.enabled])

    # Add MCP server status
    mgr = get_context().mcp_manager
    if mgr:
        result["mcp_servers"] = mgr.get_server_status()

    return result


@router.get("/health/guidance")
async def health_guidance() -> dict[str, Any]:
    """Detect available providers and return configuration recommendations.

    Scans environment variables for API keys and probes local Ollama.
    The frontend can use this to guide the user through initial setup.
    """
    # Scan for available API keys
    available_providers: list[dict[str, Any]] = []
    keyless_providers: list[dict[str, Any]] = []

    for provider, env_var in _PROVIDER_KEY_ENV.items():
        if os.environ.get(env_var):
            available_providers.append({
                "provider": provider,
                "env_var": env_var,
                "key_detected": True,
            })

    # Ollama is always keyless — check if it's reachable
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_ok = await _check_ollama_available(ollama_host)
    if ollama_ok:
        keyless_providers.append({
            "provider": "ollama",
            "host": ollama_host,
            "available": True,
        })
    else:
        keyless_providers.append({
            "provider": "ollama",
            "host": ollama_host,
            "available": False,
        })

    # Build recommendation
    current = HuginnConfig.from_env()
    recommendation: dict[str, Any] = {}
    if not _is_configured(current):
        if available_providers:
            # Recommend the first detected cloud provider
            top = available_providers[0]
            recommendation = {
                "action": "set_provider",
                "provider": top["provider"],
                "reason": f"API key detected in {top['env_var']}",
            }
        elif ollama_ok:
            recommendation = {
                "action": "set_provider",
                "provider": "ollama",
                "reason": f"Ollama is running at {ollama_host}",
            }
        else:
            recommendation = {
                "action": "manual_setup",
                "reason": "No API keys detected and Ollama is not available",
                "suggestion": "Set an API key via the config page or start Ollama locally",
            }

    return {
        "configured": _is_configured(current),
        "current_provider": current.provider,
        "available_providers": available_providers,
        "keyless_providers": keyless_providers,
        "recommendation": recommendation,
        "supported_providers": sorted(set(_PROVIDER_KEY_ENV.keys()) | {"ollama", "vllm", "local", "openai-compatible"}),
    }


@router.get("/health/rust")
async def health_rust() -> dict[str, Any]:
    """Report whether the Rust acceleration extension is available."""
    try:
        import huginn_ext

        functions = [name for name in dir(huginn_ext) if not name.startswith("_")]
        return {
            "available": True,
            "module": "huginn_ext",
            "functions": functions,
        }
    except Exception as e:
        return {
            "available": False,
            "module": "huginn_ext",
            "error": str(e),
            "functions": [],
        }


# NOTE: /config, /config/encrypt 已迁移到 huginn.routes.config,
# 这里不再注册, 避免路由冲突。前端 POST /config 仍可用, 由 config router 接管。
