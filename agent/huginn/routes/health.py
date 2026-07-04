"""Health and info endpoints.

配置类路由已迁移到 ``huginn.routes.config``, 这里只保留健康检查与引导。
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Response

from huginn import __version__
from huginn.config import HuginnConfig
from huginn.server_core import _check_ollama_available, get_context

logger = logging.getLogger(__name__)

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
    """Determine whether the system has a usable LLM configuration."""
    if cfg.provider == "ollama":
        return True
    if cfg.models:
        for m in cfg.models:
            if m.provider == "ollama":
                return True
            if m.enabled and (m.api_key or _PROVIDER_KEY_ENV.get(m.provider) and os.environ.get(_PROVIDER_KEY_ENV[m.provider])):
                return True
    return cfg.provider != "default" and bool(cfg.resolved_api_key)


# ── Liveness ──────────────────────────────────────────────────────


@router.get("/health/live")
async def health_live() -> dict[str, Any]:
    """Liveness probe — "is the process alive?".

    Returns 200 unconditionally as long as the event loop can respond.
    Kubernetes should use this for liveness probes — it must NEVER
    return non-200 due to dependency failures, otherwise the pod gets
    killed and restarted (which doesn't fix the dependency).
    """
    cfg = HuginnConfig.from_env()
    return {
        "status": "alive",
        "version": __version__,
        "provider": cfg.provider,
        "model": cfg.model,
    }


# ── Readiness ──────────────────────────────────────────────────────


@router.get("/health/ready")
@router.get("/ready")
async def health_ready(response: Response) -> dict[str, Any]:
    """Readiness probe — "can I handle requests right now?".

    Checks actual dependency reachability:
      * SQLite — can we read/write?
      * LLM — is a provider configured?
      * MCP — are configured servers connected?

    Returns 200 when all checks pass, 503 when any fail.
    Each check includes an ``error`` field on failure for diagnostics.
    """
    checks: dict[str, dict[str, Any]] = {}

    # -- SQLite -------------------------------------------------------
    try:
        from huginn.research_log import get_research_log

        log = get_research_log()
        with log._lock:
            log._conn.execute("SELECT 1").fetchone()
        checks["sqlite"] = {"status": "ok"}
    except Exception as e:
        logger.warning("/ready sqlite check failed", exc_info=True)
        checks["sqlite"] = {"status": "fail", "error": str(e)[:200]}

    # -- LLM provider ------------------------------------------------
    try:
        cfg = HuginnConfig.from_env()
        if _is_configured(cfg):
            checks["llm"] = {"status": "ok", "provider": cfg.provider}
        else:
            checks["llm"] = {"status": "fail", "error": "No provider configured"}
    except Exception as e:
        logger.warning("/ready llm check failed", exc_info=True)
        checks["llm"] = {"status": "fail", "error": str(e)[:200]}

    # -- MCP servers --------------------------------------------------
    try:
        mgr = get_context().mcp_manager
        if mgr is None:
            checks["mcp"] = {"status": "ok", "note": "not configured"}
        else:
            status = mgr.get_server_status()
            if not status:
                checks["mcp"] = {"status": "ok", "note": "no servers"}
            elif all(s.get("connected") for s in status.values()):
                checks["mcp"] = {"status": "ok", "servers": len(status)}
            else:
                disconnected = [
                    name for name, s in status.items() if not s.get("connected")
                ]
                checks["mcp"] = {
                    "status": "fail",
                    "error": f"Servers not connected: {', '.join(disconnected)}",
                }
    except Exception as e:
        logger.warning("/ready mcp check failed", exc_info=True)
        checks["mcp"] = {"status": "fail", "error": str(e)[:200]}

    all_ok = all(c["status"] == "ok" for c in checks.values())
    if not all_ok:
        response.status_code = 503
    return {"ready": all_ok, "checks": checks}


# ── Legacy /health (backward compat) ─────────────────────────────


@router.get("/health")
async def health() -> dict[str, Any]:
    """Legacy health endpoint — returns basic config info.

    .. deprecated::
        Use ``/health/live`` for liveness and ``/health/ready`` for
        readiness. This endpoint is kept for backward compatibility.
    """
    cfg = HuginnConfig.from_env()
    configured = _is_configured(cfg)

    result: dict[str, Any] = {
        "status": "ok" if configured else "unconfigured",
        "version": __version__,
        "provider": cfg.provider,
        "model": cfg.model,
        "configured": configured,
        "deprecation": "Use /health/live (liveness) and /health/ready (readiness)",
    }

    if cfg.models:
        result["model_pool"] = len([m for m in cfg.models if m.enabled])

    mgr = get_context().mcp_manager
    if mgr:
        result["mcp_servers"] = mgr.get_server_status()

    return result


@router.get("/health/guidance")
async def health_guidance() -> dict[str, Any]:
    """Detect available providers and return configuration recommendations."""
    available_providers: list[dict[str, Any]] = []
    keyless_providers: list[dict[str, Any]] = []

    for provider, env_var in _PROVIDER_KEY_ENV.items():
        if os.environ.get(env_var):
            available_providers.append({
                "provider": provider,
                "env_var": env_var,
                "key_detected": True,
            })

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

    current = HuginnConfig.from_env()
    recommendation: dict[str, Any] = {}
    if not _is_configured(current):
        if available_providers:
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
