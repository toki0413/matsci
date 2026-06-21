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
from huginn.server_core import get_context

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, Any]:
    cfg = HuginnConfig.from_env()
    return {
        "status": "ok",
        "version": __version__,
        "provider": cfg.provider,
        "model": cfg.model,
        "configured": cfg.provider != "default" and bool(cfg.resolved_api_key),
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


@router.post("/config", dependencies=[Depends(require_admin_key)])
async def update_config(params: dict[str, Any]) -> dict[str, Any]:
    """Update server-side configuration and reset the agent so changes take effect."""

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
