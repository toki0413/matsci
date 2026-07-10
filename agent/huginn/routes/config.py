"""运行时配置路由组 —— 多模型池管理、连通性测试、加密切换。

设计要点:
- 复用 ``HuginnConfig.save/load`` 做持久化, 不重复造原子写入/加密轮子;
- API key 返回时一律脱敏 (前4 + **** + 后4), 原文绝不离开后端;
- 测试连通性走 ``create_langchain_model`` 发一条 "Hello", 10s 超时兜底;
- 配置文件路径优先 ``HUGINN_CONFIG_FILE``, 没设就退回工作目录下的 huginn.toml。
"""

from __future__ import annotations

import asyncio
import os
import time
import logging
import traceback
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from huginn.config import HuginnConfig, ModelConfig, get_config as get_cached_config
from huginn.models.registry import (
    _DOMESTIC_OPENAI_COMPATIBLE,
    _PROVIDER_DEFAULTS,
    _PROVIDER_KEY_ENV,
    create_langchain_model,
    resolve_provider_key,
)
from huginn.pet import configure_pet
from huginn.security.auth import clear_api_key_cache, require_admin_key
from huginn.server_core import get_context

router = APIRouter(tags=["config"])

logger = logging.getLogger(__name__)

# Runtime override for the config file path, set by encrypt/decrypt endpoints.
# Kept separate from os.environ so tests don't leak state across requests.
_config_path_override: Path | None = None


# ── 内部工具 ────────────────────────────────────────────────────


def _config_path() -> Path:
    """配置文件路径: 优先运行时覆盖, 其次 HUGINN_CONFIG_FILE, 否则工作目录下 huginn.toml."""
    if _config_path_override is not None:
        return _config_path_override
    raw = os.environ.get("HUGINN_CONFIG_FILE")
    if raw:
        return Path(raw)
    return Path(os.environ.get("HUGINN_WORKSPACE", ".")) / "huginn.toml"


def _load_runtime_config() -> HuginnConfig:
    """加载当前生效的配置: 有文件就 load, 没有就 from_env。"""
    target = _config_path()
    if target.exists():
        try:
            return HuginnConfig.load(target)
        except Exception:
            # 文件损坏退回 env, 不让用户彻底卡住
            return HuginnConfig.from_env()
    return HuginnConfig.from_env()


def _persist_config(cfg: HuginnConfig) -> None:
    """落盘 + 同步内存缓存 + 重置 agent 工厂, 让新配置立刻生效。"""
    target = _config_path()
    cfg.save(target, format="toml")
    # 配置变更后清掉 API key SWR 缓存, 否则老 key 还会被复用
    clear_api_key_cache()
    get_cached_config(force_reload=True)
    # 重置 agent / factory / orchestrator, 下次请求会用新配置重建
    ctx = get_context()
    ctx.agent = None
    ctx.agent_factory = None
    ctx.planner_agent = None
    ctx.orchestrator = None
    configure_pet(cfg.pet_name, cfg.pet_personality)


def _mask_api_key(raw: str | None) -> str | None:
    """脱敏: 前4 + **** + 后4; 不足8位全掩; None 透传。"""
    if raw is None:
        return None
    # env:/keyring: 引用直接标掩, 不暴露具体变量名
    if raw.startswith("env:") or raw.startswith("keyring:"):
        return "********"
    if len(raw) < 8:
        return "********"
    return f"{raw[:4]}****{raw[-4:]}"


def _model_to_dict(m: ModelConfig) -> dict[str, Any]:
    """单个 ModelConfig 转可序列化 dict, api_key 脱敏 + 标记是否有 key。"""
    resolved = resolve_provider_key(m.provider, m.api_key)  # type: ignore[arg-type]
    # credential_id 也能间接提供 key, 单独探一下, 让前端状态灯更准
    has_cred_key = False
    if m.credential_id:
        try:
            from huginn.security.credential_store import get_credential_store
            cred = get_credential_store().to_llm_info(m.credential_id)
            has_cred_key = bool(cred and cred.get("api_key"))
        except Exception:
            # 凭据 store 没初始化 / 记录不存在 — 当作没 key, 不影响主流程
            pass
    return {
        "alias": m.alias,
        "provider": m.provider,
        "model": m.model,
        "api_key": _mask_api_key(m.api_key),
        "credential_id": m.credential_id,
        "base_url": m.base_url,
        "temperature": m.temperature,
        "enabled": m.enabled,
        "thinking": m.thinking,
        "max_tokens": m.max_tokens,
        # has_key=True 表示实际能拿到 key (含 env/keyring 引用),
        # 前端据此显示状态灯, 比"有没有填字符串"更准确
        "has_key": bool(resolved) or has_cred_key,
    }


def _provider_info(provider: str) -> dict[str, Any]:
    """组装单个 provider 的元信息: 默认 base_url / 默认 model / 是否需要 key。"""
    domestic = _DOMESTIC_OPENAI_COMPATIBLE.get(provider, {})
    base_url = domestic.get("base_url") if domestic else None
    needs_key = bool(_PROVIDER_KEY_ENV.get(provider))
    default_model = _PROVIDER_DEFAULTS.get(provider)

    # ollama 是无 key 本地方案, 单独标一下方便前端引导
    if provider == "ollama":
        base_url = base_url or "http://localhost:11434"
        needs_key = False
    # vllm / local 默认走本地, 也不强求 key
    if provider in ("vllm", "local"):
        needs_key = False
    # local LLM presets — no key needed, preset base_url
    if provider in ("lm-studio", "llama-cpp", "sglang"):
        needs_key = False
        presets = {
            "lm-studio": "http://localhost:1234/v1",
            "llama-cpp": "http://localhost:8080/v1",
            "sglang": "http://localhost:30000/v1",
        }
        base_url = presets.get(provider)

    return {
        "provider": provider,
        "default_base_url": base_url,
        "default_model": default_model,
        "needs_api_key": needs_key,
        "env_var": _PROVIDER_KEY_ENV.get(provider) or None,
    }


# ── /config 顶层 ─────────────────────────────────────────────────


@router.get("/config", dependencies=[Depends(require_admin_key)])
async def get_config_endpoint() -> dict[str, Any]:
    """返回当前配置, api_key 全部脱敏。"""
    cfg = _load_runtime_config()
    data = cfg.to_dict(mask_key=True)
    # models 列表替换成带 has_key 的版本, 前端能显示状态灯
    data["models"] = [_model_to_dict(m) for m in cfg.models]
    data["config_file"] = str(_config_path())
    data["config_file_exists"] = _config_path().exists()
    return data


# 兼容老前端的 POST /config: 单 provider 字段直接落到 env + 文件,
# 不破坏现有 saveSettings() 调用链
_ALLOWED_LEGACY_KEYS = frozenset({
    "provider", "model", "api_key", "base_url", "ollama_host",
    "persona", "persona_auto_route_threshold",
    "rag_enabled", "team_mode_enabled", "max_concurrent_subagents",
    "models", "agents",
    "pet_name", "pet_personality",
    "encrypt_config", "encryption_password", "encryption_key_file",
    "privacy_redact_secrets", "privacy_block_on_secrets", "local_only_mode",
    "max_tool_output_tokens", "context_budget_tokens", "pet_accessories",
})


@router.post("/config", dependencies=[Depends(require_admin_key)])
async def update_config_endpoint(params: dict[str, Any]) -> dict[str, Any]:
    """兼容老前端的整配置更新: 落 env + 持久化到文件 + 重置 agent。"""
    unknown = set(params.keys()) - _ALLOWED_LEGACY_KEYS
    if unknown:
        return {
            "success": False,
            "error": (
                f"Unknown config keys: {', '.join(sorted(unknown))}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_LEGACY_KEYS))}"
            ),
        }

    # 先同步到 env, from_env() 才能拼出 HuginnConfig
    _apply_legacy_params_to_env(params)

    cfg = HuginnConfig.from_env()
    try:
        _persist_config(cfg)
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": f"持久化失败: {e}"}

    return {
        "success": True,
        "config": cfg.to_dict(mask_key=True),
    }


def _apply_legacy_params_to_env(params: dict[str, Any]) -> None:
    """把老式 flat config 字段刷到环境变量, 给 from_env() 用。"""
    mapping = {
        "provider": "HUGINN_PROVIDER",
        "model": "HUGINN_MODEL",
        "api_key": "HUGINN_API_KEY",
        "base_url": "HUGINN_BASE_URL",
        "ollama_host": "OLLAMA_HOST",
        "persona": "HUGINN_PERSONA",
        "persona_auto_route_threshold": "HUGINN_PERSONA_AUTO_ROUTE_THRESHOLD",
        "rag_enabled": "HUGINN_RAG_ENABLED",
        "team_mode_enabled": "HUGINN_TEAM_MODE",
        "max_concurrent_subagents": "HUGINN_MAX_CONCURRENT_SUBAGENTS",
        "pet_name": "HUGINN_PET_NAME",
        "pet_personality": "HUGINN_PET_PERSONA",
        "encrypt_config": "HUGINN_ENCRYPT_CONFIG",
        "encryption_password": "HUGINN_ENCRYPTION_PASSWORD",
        "encryption_key_file": "HUGINN_ENCRYPTION_KEY_FILE",
    }
    import json as _json

    for key, env_name in mapping.items():
        if key not in params:
            continue
        val = params[key]
        if val is None or val == "":
            os.environ.pop(env_name, None)
            continue
        if isinstance(val, bool):
            os.environ[env_name] = "true" if val else "false"
        elif isinstance(val, (dict, list)):
            os.environ[env_name] = _json.dumps(val)
        else:
            os.environ[env_name] = str(val)

    # models / agents 是 list, 单独走 JSON 序列化
    if "models" in params and params["models"] is not None:
        os.environ["HUGINN_MODELS"] = _json.dumps(params["models"])
    if "agents" in params and params["agents"] is not None:
        os.environ["HUGINN_AGENTS"] = _json.dumps(params["agents"])


# ── /config/models ──────────────────────────────────────────────


@router.get("/config/models", dependencies=[Depends(require_admin_key)])
async def list_models_config() -> dict[str, Any]:
    """列出模型池里所有 model (api_key 脱敏 + has_key 标记)。"""
    cfg = _load_runtime_config()
    return {"models": [_model_to_dict(m) for m in cfg.models]}


@router.post("/config/models", dependencies=[Depends(require_admin_key)])
async def add_model_config(params: dict[str, Any]) -> dict[str, Any]:
    """添加一个新 model 到模型池。body 字段对齐 ModelConfig。"""
    alias = params.get("alias")
    provider = params.get("provider")
    if not alias or not provider:
        return {"success": False, "error": "alias 和 provider 必填"}

    cfg = _load_runtime_config()
    # alias 去重, 避免覆盖已有配置
    if any(m.alias == alias for m in cfg.models):
        return {"success": False, "error": f"alias '{alias}' 已存在, 请换一个或用 PUT 更新"}

    try:
        model = _build_model_config(params)
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"参数无效: {e}"}

    cfg.models.append(model)
    try:
        _persist_config(cfg)
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": f"持久化失败: {e}"}

    return {"success": True, "model": _model_to_dict(model)}


@router.put("/config/models/{alias}", dependencies=[Depends(require_admin_key)])
async def update_model_config(alias: str, params: dict[str, Any]) -> dict[str, Any]:
    """部分更新某个 model 配置。api_key 留空表示保留原值。"""
    cfg = _load_runtime_config()
    idx = next((i for i, m in enumerate(cfg.models) if m.alias == alias), None)
    if idx is None:
        return {"success": False, "error": f"alias '{alias}' 不存在"}

    existing = cfg.models[idx]
    # api_key 留空 → 保留原 key, 避免 UI 提交时把 key 误清空
    if not params.get("api_key"):
        params["api_key"] = existing.api_key

    try:
        updated = _build_model_config({**existing.__dict__, **params, "alias": alias})
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"参数无效: {e}"}

    cfg.models[idx] = updated
    try:
        _persist_config(cfg)
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": f"持久化失败: {e}"}

    return {"success": True, "model": _model_to_dict(updated)}


@router.delete("/config/models/{alias}", dependencies=[Depends(require_admin_key)])
async def delete_model_config(alias: str) -> dict[str, Any]:
    """从模型池删除一个 model。"""
    cfg = _load_runtime_config()
    before = len(cfg.models)
    cfg.models = [m for m in cfg.models if m.alias != alias]
    if len(cfg.models) == before:
        return {"success": False, "error": f"alias '{alias}' 不存在"}

    try:
        _persist_config(cfg)
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": f"持久化失败: {e}"}

    return {"success": True, "deleted": alias}


@router.post("/config/models/{alias}/test", dependencies=[Depends(require_admin_key)])
async def test_model_config(alias: str) -> dict[str, Any]:
    """测试模型池里某个 model 的连通性: 发 "Hello", max_tokens=10, 10s 超时。"""
    cfg = _load_runtime_config()
    target = next((m for m in cfg.models if m.alias == alias), None)
    if target is None:
        return {"success": False, "error": f"alias '{alias}' 不存在"}

    return await _run_connectivity_test(target)


@router.post("/config/models/test", dependencies=[Depends(require_admin_key)])
async def test_model_inline(params: dict[str, Any]) -> dict[str, Any]:
    """不落盘直接测一组 ModelConfig 字段, 添加前先验证用。"""
    try:
        model = _build_model_config(params)
    except (TypeError, ValueError) as e:
        return {"success": False, "error": f"参数无效: {e}"}
    return await _run_connectivity_test(model)


async def _run_connectivity_test(model: ModelConfig) -> dict[str, Any]:
    """实际跑连通性: 建客户端 → 发 Hello → 测延迟。"""
    start = time.perf_counter()
    try:
        resolved_key = resolve_provider_key(model.provider, model.api_key)  # type: ignore[arg-type]
        # 本地 provider 没 key 也得给个占位, 不然 ChatOpenAI 会拒
        if not resolved_key and model.provider in ("ollama", "vllm", "local"):
            resolved_key = "not-needed"

        client = create_langchain_model(
            provider=model.provider,  # type: ignore[arg-type]
            model_name=model.model,
            api_key=resolved_key,
            base_url=model.base_url,
            temperature=0.0,
            max_tokens=16,
        )

        # 用 invoke 跑同步调用, 包到 to_thread 里避免阻塞事件循环
        # 10s 超时: 够覆盖建连 + 一次小推理, 又不至于让用户等太久
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.invoke,
                "Hello",
                config={"max_tokens": 16} if model.provider in ("openai", "vllm", "local", "deepseek", "openrouter") else None,
            ),
            timeout=10.0,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = ""
        if hasattr(response, "content"):
            text = str(response.content)
        elif isinstance(response, str):
            text = response
        return {
            "success": True,
            "latency_ms": latency_ms,
            "error": None,
            "model_response": text[:200],
        }
    except asyncio.TimeoutError:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": "请求超时 (10s), 检查 base_url 是否可达 / 模型是否加载完成",
            "model_response": "",
        }
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.error("unexpected error", exc_info=True)
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": str(e),
            "model_response": "",
        }


def _build_model_config(params: dict[str, Any]) -> ModelConfig:
    """从 dict 构造 ModelConfig, 只取已知字段, 多余的丢掉。"""
    known = {f for f in ModelConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in params.items() if k in known}
    return ModelConfig(**filtered)


# ── /config/providers ───────────────────────────────────────────


@router.get("/config/providers", dependencies=[Depends(require_admin_key)])
async def list_providers() -> dict[str, Any]:
    """列出支持的 19 个 provider + 默认 base_url + 是否需要 key。"""
    # 顺序固定, 前端展示更稳定
    order = [
        "anthropic", "openai", "deepseek", "google-genai", "openrouter",
        "nvidia", "ollama", "vllm", "local", "siliconflow",
        "moonshot", "zhipu", "baichuan", "dashscope", "qianfan",
        "doubao", "hunyuan", "minimax",
        "lm-studio", "llama-cpp", "sglang",
        "openai-compatible", "default",
    ]
    return {"providers": [_provider_info(p) for p in order], "count": len(order)}


# ── /config/local-models ────────────────────────────────────────


@router.get("/config/local-models", dependencies=[Depends(require_admin_key)])
async def discover_local_models(provider: str = "ollama", base_url: str = "") -> dict[str, Any]:
    """探一下本地推理服务 (ollama/vllm/local) 上挂了哪些模型。

    ollama 走 /api/tags, 其余 OpenAI 兼容服务走 /v1/models。5s 超时,
    连不上就回 success=False + error, 不抛异常给前端。

    只允许访问 loopback 地址, 防止 SSRF。
    """
    import ipaddress
    import json as _json
    import socket
    import urllib.request
    from urllib.parse import urlparse

    # 各 provider 的默认本地地址, 没传 base_url 时兜底
    defaults = {
        "ollama": "http://localhost:11434",
        "vllm": "http://localhost:8000",
        "local": "http://localhost:8000",
    }
    url = base_url or defaults.get(provider, "http://localhost:8000")

    # SSRF 防护: 只允许 loopback / 链路本地地址
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"success": False, "models": [], "error": f"scheme '{parsed.scheme}' not allowed"}
    hostname = parsed.hostname or ""
    if not hostname:
        return {"success": False, "models": [], "error": "invalid URL: no hostname"}
    # 允许的 hostname: localhost, 127.0.0.1, ::1, .local 后缀
    allowed_names = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    is_loopback_name = hostname.lower() in allowed_names or hostname.endswith(".local")
    # 如果是 IP 地址, 检查是否 loopback / link-local
    try:
        ip = ipaddress.ip_address(hostname)
        is_loopback_ip = ip.is_loopback or ip.is_link_local or ip.is_private
    except ValueError:
        is_loopback_ip = False
    if not (is_loopback_name or is_loopback_ip):
        return {
            "success": False,
            "models": [],
            "error": f"blocked: '{hostname}' is not a loopback/local address — SSRF protection",
        }

    try:
        if provider == "ollama":
            api_url = f"{url.rstrip('/')}/api/tags"
        else:
            # vLLM 和其它 OpenAI 兼容服务都用 /v1/models
            api_url = f"{url.rstrip('/')}/v1/models"

        req = urllib.request.Request(api_url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = _json.loads(resp.read().decode())

        if provider == "ollama":
            models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        else:
            models = [m.get("id", "") for m in data.get("data", []) if m.get("id")]

        return {"success": True, "models": models, "provider": provider, "base_url": url}
    except Exception as e:
        # 不回传完整异常信息, 避免泄露内部路径
        return {"success": False, "models": [], "error": type(e).__name__}


# ── /config/active-model ────────────────────────────────────────


@router.get("/config/active-model", dependencies=[Depends(require_admin_key)])
async def get_active_model() -> dict[str, Any]:
    """返回当前活跃 model (lead agent 绑定的 alias, 没有就退回第一个 enabled)。"""
    cfg = _load_runtime_config()
    lead = next((a for a in cfg.agents if a.id == "lead" and a.enabled), None)
    alias = lead.model_alias if lead else None
    if not alias and cfg.models:
        for m in cfg.models:
            if m.enabled:
                alias = m.alias
                break
    if not alias:
        return {"active_alias": None, "model": None}

    target = next((m for m in cfg.models if m.alias == alias), None)
    return {
        "active_alias": alias,
        "model": _model_to_dict(target) if target else None,
    }


@router.post("/config/active-model", dependencies=[Depends(require_admin_key)])
async def set_active_model(params: dict[str, Any]) -> dict[str, Any]:
    """切换活跃 model: 改 lead agent 的 model_alias。"""
    alias = params.get("alias")
    if not alias:
        return {"success": False, "error": "alias 必填"}

    cfg = _load_runtime_config()
    if not any(m.alias == alias and m.enabled for m in cfg.models):
        return {"success": False, "error": f"alias '{alias}' 不存在或未启用"}

    # 找 lead agent, 没有就现建一个指向这个 alias
    lead = next((a for a in cfg.agents if a.id == "lead"), None)
    if lead is None:
        from huginn.config import AgentProfileConfig
        cfg.agents.append(
            AgentProfileConfig(id="lead", name="Lead", model_alias=alias)
        )
    else:
        lead.model_alias = alias

    try:
        _persist_config(cfg)
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": f"持久化失败: {e}"}

    return {"success": True, "active_alias": alias}


# ── /config/encrypt · /config/decrypt ───────────────────────────


@router.post("/config/encrypt", dependencies=[Depends(require_admin_key)])
async def encrypt_config_endpoint(params: dict[str, Any]) -> dict[str, Any]:
    """启用加密: 用 password 把当前配置写到 .enc 文件。"""
    password = params.get("password")
    if not password:
        return {"success": False, "error": "password is required"}

    target = _config_path()
    cfg = _load_runtime_config()
    cfg.encrypt_config = True
    cfg.encryption_password = password

    # .enc 后缀走 EncryptedConfig 路径, save() 内部会识别
    enc_path = target if str(target).endswith(".enc") else target.with_suffix(
        target.suffix + ".enc"
    )
    try:
        cfg.save(enc_path, format="json")
        # 切到加密文件后, 后续 load 都走这个路径
        global _config_path_override
        _config_path_override = enc_path
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}

    return {"success": True, "path": str(enc_path)}


@router.post("/config/decrypt", dependencies=[Depends(require_admin_key)])
async def decrypt_config_endpoint(params: dict[str, Any]) -> dict[str, Any]:
    """解密: 用 password 读 .enc 文件, 写回明文 toml, 关掉加密开关。"""
    password = params.get("password")
    if not password:
        return {"success": False, "error": "password is required"}

    enc_path = _config_path()
    if not str(enc_path).endswith(".enc"):
        # 当前不是 .enc, 找一下同目录的 .enc
        candidate = enc_path.with_suffix(enc_path.suffix + ".enc")
        if candidate.exists():
            enc_path = candidate
        else:
            return {"success": False, "error": "当前配置文件未加密"}

    if not enc_path.exists():
        return {"success": False, "error": f"加密文件不存在: {enc_path}"}

    try:
        cfg = HuginnConfig.load(enc_path, password=password)
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": f"解密失败 (密码错?): {e}"}

    cfg.encrypt_config = False
    cfg.encryption_password = None
    # 写回明文 toml
    plain_path = enc_path.with_suffix("")
    if plain_path.suffix != ".toml":
        plain_path = plain_path.with_suffix(".toml")
    try:
        cfg.save(plain_path, format="toml")
        global _config_path_override
        _config_path_override = plain_path
        # 解密成功后删掉 .enc, 避免下次又走加密路径
        try:
            enc_path.unlink()
        except OSError:
            pass
    except Exception as e:
        logger.error("unexpected error", exc_info=True)
        return {"success": False, "error": str(e)}

    return {"success": True, "path": str(plain_path)}


# ── /config/features ────────────────────────────────────────────


@router.get("/config/features", dependencies=[Depends(require_admin_key)])
async def list_feature_flags() -> dict[str, Any]:
    """列出所有 feature flag 的当前状态 (名称/是否开启/描述/默认值)."""
    from huginn.feature_flags import FeatureFlags

    flags = FeatureFlags.shared().list_flags()
    return {"features": flags, "count": len(flags)}


@router.post("/config/features/{feature}", dependencies=[Depends(require_admin_key)])
async def toggle_feature_flag(feature: str, params: dict[str, Any]) -> dict[str, Any]:
    """运行时开关某个 feature flag. body: {"enabled": bool}.

    改动只在内存生效, 不写盘. 要持久化得调 config wizard 或直接改配置文件.
    """
    from huginn.feature_flags import FeatureFlags

    if "enabled" not in params:
        return {"success": False, "error": "body 需要 enabled 字段"}
    enabled = bool(params["enabled"])

    ff = FeatureFlags.shared()
    # 未知 feature 直接报错, 别让前端以为开关成功了
    known = {f["name"] for f in ff.list_flags()}
    if feature not in known:
        return {
            "success": False,
            "error": f"未知 feature: {feature}. 可用: {sorted(known)}",
        }

    ff.toggle(feature, enabled)
    new_state = ff.is_enabled(feature)
    return {
        "success": True,
        "feature": feature,
        "enabled": new_state,
        "requested": enabled,
        "applied": new_state == enabled,
    }


# ── /config/health ──────────────────────────────────────────────


@router.get("/config/health", dependencies=[Depends(require_admin_key)])
async def tool_health_endpoint() -> dict[str, Any]:
    """所有工具的健康报告 + 总体统计 + 系统资源指标.

    返回 {tools: [...], summary: {...}, system: {...}}. tools 只包含有调用
    记录的工具, summary 给整体成功率/工具数, system 给当前 CPU/内存/磁盘
    负载和最近异常事件. 前端顶部条用.
    """
    from huginn.agents.health_dashboard import HealthDashboard
    from huginn.diagnostics.system_health import SystemHealthMonitor
    from huginn.feature_flags import FeatureFlags

    dash = HealthDashboard.shared()
    tools = dash.get_all()
    summary = dash.summary()

    system: dict[str, Any] = {"enabled": False}
    if FeatureFlags.shared().is_enabled("system_health_monitor"):
        monitor = SystemHealthMonitor.shared()
        system = {
            "enabled": True,
            "metrics": monitor.snapshot().to_dict(),
            "running": monitor.is_running(),
            "recent_anomalies": [
                ev.to_dict() for ev in monitor.recent_anomalies(limit=5)
            ],
        }

    return {
        "tools": tools,
        "summary": summary,
        "count": len(tools),
        "system": system,
    }


# ── /config/circuit ─────────────────────────────────────────────


@router.get("/config/circuit", dependencies=[Depends(require_admin_key)])
async def circuit_state_endpoint() -> dict[str, Any]:
    """所有工具的熔断状态快照, 给前端仪表盘用."""
    from huginn.agents.circuit_breaker import CircuitBreaker

    states = CircuitBreaker.shared().list_all()
    return {"breakers": states, "count": len(states)}


@router.post("/config/circuit/{tool}/reset", dependencies=[Depends(require_admin_key)])
async def reset_circuit_endpoint(tool: str) -> dict[str, Any]:
    """手动重置某个工具的熔断器. 重置后状态回 closed, 失败计数清零."""
    from huginn.agents.circuit_breaker import CircuitBreaker

    breaker = CircuitBreaker.shared()
    # 先看一下有没有这个工具的记录, 没有也允许重置 (幂等), 但提示一下
    before = breaker.get_stats(tool)
    breaker.reset(tool)
    after = breaker.get_stats(tool)
    return {
        "success": True,
        "tool": tool,
        "before": before["state"],
        "after": after["state"],
        "note": "已重置" if before["state"] != "closed" or before["consecutive_failures"] > 0 else "本就为 closed, 无需重置",
    }
