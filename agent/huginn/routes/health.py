"""Health, configuration, and info endpoints."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from huginn import __version__
from huginn.config import HuginnConfig
from huginn.pet import configure_pet
from huginn.security.auth import require_admin_key
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


@router.get("/config", dependencies=[Depends(require_admin_key)])
async def get_config() -> dict[str, Any]:
    """Return current server-side configuration (API key masked)."""
    return HuginnConfig.from_env().to_dict(mask_key=True)


_ALLOWED_CONFIG_KEYS = frozenset({
    "provider", "model", "api_key", "base_url", "ollama_host",
    "persona", "persona_auto_route_threshold",
    "rag_enabled", "team_mode_enabled", "max_concurrent_subagents",
    "models", "agents",
    "pet_name", "pet_personality", "pet_accessories",
    "encrypt_config", "encryption_password", "encryption_key_file",
})


@router.post("/config", dependencies=[Depends(require_admin_key)])
async def update_config(params: dict[str, Any]) -> dict[str, Any]:
    """Update server-side configuration and reset the agent so changes take effect."""

    unknown = set(params.keys()) - _ALLOWED_CONFIG_KEYS
    if unknown:
        return {
            "success": False,
            "error": f"Unknown config keys: {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(_ALLOWED_CONFIG_KEYS))}",
        }

    if "provider" in params:
        os.environ["HUGINN_PROVIDER"] = str(params["provider"])
    if "model" in params:
        os.environ["HUGINN_MODEL"] = str(params["model"])
    if "api_key" in params:
        if params["api_key"]:
            os.environ["HUGINN_API_KEY"] = str(params["api_key"])
        else:
            os.environ.pop("HUGINN_API_KEY", None)
    if "base_url" in params:
        if params["base_url"]:
            os.environ["HUGINN_BASE_URL"] = str(params["base_url"])
        else:
            os.environ.pop("HUGINN_BASE_URL", None)
    if "ollama_host" in params:
        os.environ["OLLAMA_HOST"] = str(params["ollama_host"])
    if "persona" in params:
        os.environ["HUGINN_PERSONA"] = str(params["persona"])
    if "persona_auto_route_threshold" in params:
        os.environ["HUGINN_PERSONA_AUTO_ROUTE_THRESHOLD"] = str(
            params["persona_auto_route_threshold"]
        )
    if "rag_enabled" in params:
        os.environ["HUGINN_RAG_ENABLED"] = "true" if params["rag_enabled"] else "false"
    if "team_mode_enabled" in params:
        os.environ["HUGINN_TEAM_MODE"] = (
            "true" if params["team_mode_enabled"] else "false"
        )
    if "max_concurrent_subagents" in params:
        os.environ["HUGINN_MAX_CONCURRENT_SUBAGENTS"] = str(
            params["max_concurrent_subagents"]
        )
    if "models" in params:
        os.environ["HUGINN_MODELS"] = json.dumps(params["models"])
    if "agents" in params:
        os.environ["HUGINN_AGENTS"] = json.dumps(params["agents"])
    if "pet_name" in params:
        os.environ["HUGINN_PET_NAME"] = str(params["pet_name"])
    if "pet_personality" in params:
        os.environ["HUGINN_PET_PERSONALITY"] = str(params["pet_personality"])
    if "pet_accessories" in params:
        os.environ["HUGINN_PET_ACCESSORIES"] = json.dumps(params["pet_accessories"])
    if "encrypt_config" in params:
        os.environ["HUGINN_ENCRYPT_CONFIG"] = (
            "true" if params["encrypt_config"] else "false"
        )
    if "encryption_password" in params:
        pw = params["encryption_password"]
        if pw:
            os.environ["HUGINN_ENCRYPTION_PASSWORD"] = str(pw)
        else:
            os.environ.pop("HUGINN_ENCRYPTION_PASSWORD", None)
    if "encryption_key_file" in params:
        kf = params["encryption_key_file"]
        if kf:
            os.environ["HUGINN_ENCRYPTION_KEY_FILE"] = str(kf)
        else:
            os.environ.pop("HUGINN_ENCRYPTION_KEY_FILE", None)

    get_context().agent = None
    get_context().agent_factory = None
    get_context().planner_agent = None
    get_context().orchestrator = None
    cfg = HuginnConfig.from_env()
    configure_pet(cfg.pet_name, cfg.pet_personality)
    return {"success": True, "config": cfg.to_dict(mask_key=True)}


@router.post("/config/encrypt", dependencies=[Depends(require_admin_key)])
async def encrypt_config_endpoint(params: dict[str, Any]) -> dict[str, Any]:
    """Encrypt the current or provided configuration file."""
    path = params.get("path", "huginn.toml")
    password = params.get("password")
    if not password:
        return {"success": False, "error": "password is required"}

    try:
        target = Path(path)
        cfg = HuginnConfig.load(path) if target.exists() else HuginnConfig.from_env()
        cfg.encrypt_config = True
        cfg.encryption_password = password
        out = (
            target
            if str(target).endswith(".enc")
            else target.with_suffix(target.suffix + ".enc")
        )
        cfg.save(out, format="json")
        return {"success": True, "path": str(out)}
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}
