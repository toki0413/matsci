"""凭据管理路由组 —— SSH 连接与 LLM API key 的长时存储 / 编辑 / 删除 / 测试。

端点一览 (全部走 require_admin_key 保护, dev 模式自动放行):
    GET    /credentials                 列表 (?kind=ssh|llm 可选过滤)
    POST   /credentials                 新建
    GET    /credentials/{cid}           单条详情 (脱敏)
    PUT    /credentials/{cid}           更新 (部分字段)
    DELETE /credentials/{cid}           删除
    POST   /credentials/{cid}/set-default   设为同 kind 默认
    POST   /credentials/{cid}/test          测试连通性 (SSH 探活 / LLM 发 Hello)
    GET    /credentials/defaults        取 ssh + llm 两类当前默认

设计要点:
- 明文 secret 永远不出后端; 列表 / 详情只返回 secret_masked + has_secret;
- 测试连通性复用现有 HPCClient 与 create_langchain_model, 不另起一套;
- 借鉴 AstrBot 的 RESTful 风格, 但用 PUT 做部分更新更贴合前端 fetch 调用。
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any

from fastapi import APIRouter, Depends

from huginn.hpc.client import HPCClient
from huginn.models.registry import create_langchain_model
from huginn.security.auth import require_admin_key
from huginn.security.credential_store import (
    CRED_KIND_LLM,
    CRED_KIND_SSH,
    get_credential_store,
)

router = APIRouter(tags=["credentials"])


def _store():
    # 包一层方便测试 monkeypatch
    return get_credential_store()


# ── 列表 / 详情 / 默认 ─────────────────────────────────────────


@router.get("/credentials", dependencies=[Depends(require_admin_key)])
async def list_credentials(kind: str | None = None) -> dict[str, Any]:
    """列出凭据 (脱敏)。?kind=ssh 或 ?kind=llm 过滤; 不传返回全部。"""
    store = _store()
    if kind and kind not in (CRED_KIND_SSH, CRED_KIND_LLM):
        return {"success": False, "error": f"kind 必须是 ssh 或 llm, 得到 {kind!r}"}
    return {"credentials": store.list(kind=kind)}


@router.get("/credentials/defaults", dependencies=[Depends(require_admin_key)])
async def get_default_credentials() -> dict[str, Any]:
    """取 ssh 与 llm 两类各自的默认凭据 (脱敏), 方便前端展示当前生效项。"""
    store = _store()
    return {
        "ssh": store.get_default(CRED_KIND_SSH),
        "llm": store.get_default(CRED_KIND_LLM),
    }


@router.get("/credentials/{cid}", dependencies=[Depends(require_admin_key)])
async def get_credential(cid: str) -> dict[str, Any]:
    """单条凭据详情 (脱敏)。"""
    rec = _store().get(cid)
    if rec is None:
        return {"success": False, "error": f"凭据 {cid} 不存在"}
    return {"credential": rec}


# ── 增删改 ─────────────────────────────────────────────────────


@router.post("/credentials", dependencies=[Depends(require_admin_key)])
async def create_credential(params: dict[str, Any]) -> dict[str, Any]:
    """新建凭据。

    body 字段:
        kind        str   "ssh" | "llm" (必填)
        name        str   可读名称, 如 "实验室集群" / "DeepSeek 主 key" (必填)
        metadata    dict  非敏感参数 (host/port/username/provider/model/...)
        secret      str   敏感值 (SSH 密码 / LLM api_key), 可空 (密钥认证时)
        is_default  bool  是否设为同 kind 默认, 省略时首条自动默认
    """
    kind = params.get("kind")
    name = params.get("name")
    if not kind or not name:
        return {"success": False, "error": "kind 和 name 必填"}

    try:
        rec = _store().create(
            kind=kind,
            name=name,
            metadata=params.get("metadata") or {},
            secret=params.get("secret") or "",
            is_default=params.get("is_default"),
        )
    except ValueError as e:
        return {"success": False, "error": str(e)}
    return {"success": True, "credential": rec}


@router.put("/credentials/{cid}", dependencies=[Depends(require_admin_key)])
async def update_credential(cid: str, params: dict[str, Any]) -> dict[str, Any]:
    """部分更新凭据。

    可更新字段: name / metadata / secret。
    secret 传 None 或不传 = 不改密钥; 传 "" = 清空密钥。
    """
    try:
        rec = _store().update(
            cid,
            name=params.get("name"),
            metadata=params.get("metadata"),
            secret=params.get("secret"),
        )
    except ValueError as e:
        return {"success": False, "error": str(e)}
    if rec is None:
        return {"success": False, "error": f"凭据 {cid} 不存在"}
    return {"success": True, "credential": rec}


@router.delete("/credentials/{cid}", dependencies=[Depends(require_admin_key)])
async def delete_credential(cid: str) -> dict[str, Any]:
    """删除凭据。若删的是默认, 同 kind 自动提升最早一条为默认。"""
    ok = _store().delete(cid)
    if not ok:
        return {"success": False, "error": f"凭据 {cid} 不存在"}
    return {"success": True}


@router.post(
    "/credentials/{cid}/set-default", dependencies=[Depends(require_admin_key)]
)
async def set_default_credential(cid: str) -> dict[str, Any]:
    """把指定凭据设为同 kind 的默认。"""
    ok = _store().set_default(cid)
    if not ok:
        return {"success": False, "error": f"凭据 {cid} 不存在"}
    return {"success": True}


# ── 连通性测试 ─────────────────────────────────────────────────


@router.post("/credentials/{cid}/test", dependencies=[Depends(require_admin_key)])
async def test_credential(cid: str) -> dict[str, Any]:
    """测试凭据连通性。

    SSH: 连上后跑 hostname, 15s 超时;
    LLM: 建客户端发 "Hello", 10s 超时, 返回首段回复与延迟。
    """
    store = _store()
    rec = store.get(cid)
    if rec is None:
        return {"success": False, "error": f"凭据 {cid} 不存在"}

    if rec["kind"] == CRED_KIND_SSH:
        return await _test_ssh(store, cid)
    if rec["kind"] == CRED_KIND_LLM:
        return await _test_llm(store, cid)
    return {"success": False, "error": f"未知凭据类型 {rec['kind']!r}"}


async def _test_ssh(store, cid: str) -> dict[str, Any]:
    """SSH 探活: 建连 → hostname → 报状态。同步 paramiko 包到 to_thread。"""
    cfg = store.to_hpc_config(cid)
    if cfg is None:
        return {"success": False, "error": "凭据不是 SSH 类型"}
    if not cfg.host or not cfg.username:
        return {"success": False, "error": "host 和 username 必填"}

    def _probe():
        with HPCClient(cfg) as client:
            stdout, stderr, rc = client._exec("hostname")
            return stdout, stderr, rc

    start = time.perf_counter()
    try:
        stdout, stderr, rc = await asyncio.wait_for(
            asyncio.to_thread(_probe), timeout=15.0
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        if rc == 0:
            return {
                "success": True,
                "latency_ms": latency_ms,
                "hostname": (stdout or "").strip(),
                "scheduler": cfg.scheduler,
            }
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": stderr or "连接失败 (rc!=0)",
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": "SSH 连接超时 (15s), 检查 host/端口/网络/密钥",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": str(e),
        }


async def _test_llm(store, cid: str) -> dict[str, Any]:
    """LLM 连通性: 建客户端发 Hello, 复用 registry 的 create_langchain_model。"""
    info = store.to_llm_info(cid)
    if info is None:
        return {"success": False, "error": "凭据不是 LLM 类型"}
    if not info.get("provider") or not info.get("model"):
        return {"success": False, "error": "provider 和 model 必填"}

    provider = info["provider"]
    api_key = info.get("api_key") or ""
    # 本地 provider 没 key 也得给占位, 否则 ChatOpenAI 直接拒
    if not api_key and provider in ("ollama", "vllm", "local"):
        api_key = "not-needed"

    start = time.perf_counter()
    try:
        client = create_langchain_model(
            provider=provider,
            model_name=info["model"],
            api_key=api_key,
            base_url=info.get("base_url"),
            temperature=0.0,
            max_tokens=16,
        )
        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.invoke,
                "Hello",
                config=(
                    {"max_tokens": 16}
                    if provider in ("openai", "vllm", "local", "deepseek", "openrouter")
                    else None
                ),
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
            "model_response": text[:200],
        }
    except asyncio.TimeoutError:
        return {
            "success": False,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": "请求超时 (10s), 检查 base_url / 模型是否加载完成",
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "success": False,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": str(e),
        }


# ── 凭据 ↔ 模型配置 桥接 ──────────────────────────────────────


@router.post("/credentials/import-from-config")
async def import_credentials_from_config() -> dict[str, Any]:
    """扫一遍 huginn.toml 里的模型池, 把明文 api_key 批量导入 CredentialStore。

    方便老用户从配置文件内联 key 迁移到凭据管理。返回 {alias: credential_id}
    映射, 前端可以再据此调 link-model 把模型挂上去。
    """
    import os
    from pathlib import Path

    from huginn.config import HuginnConfig
    from huginn.security.credential_store import get_credential_store

    # 优先用 HUGINN_CONFIG_FILE 指的文件, 没有就退回环境变量拼配置
    config_file = Path(os.environ.get("HUGINN_CONFIG_FILE", "huginn.toml"))
    if config_file.exists():
        cfg = HuginnConfig.load(config_file)
    else:
        cfg = HuginnConfig.from_env()

    store = get_credential_store()
    mapping = store.import_from_config(cfg)

    return {
        "success": True,
        "imported": mapping,
        "count": len(mapping),
    }


@router.post("/credentials/{cid}/link-model/{alias}")
async def link_credential_to_model(cid: str, alias: str) -> dict[str, Any]:
    """把一条凭据挂到某个 ModelConfig 上: 设 credential_id, 清掉明文 api_key。

    之后 agent 跑推理时会从凭据 store 取 key, 配置文件里不再留明文。
    """
    from dataclasses import replace

    # 惰性导入, 避免和 routes.config 形成循环依赖
    from huginn.routes.config import _load_runtime_config, _persist_config

    # 先确认凭据确实存在, 别把无效 id 写进配置
    store = get_credential_store()
    rec = store.get_record(cid)
    if rec is None:
        return {"success": False, "error": f"credential '{cid}' not found"}

    cfg = _load_runtime_config()
    idx = next((i for i, m in enumerate(cfg.models) if m.alias == alias), None)
    if idx is None:
        return {"success": False, "error": f"model alias '{alias}' not found"}

    # 挂上 credential_id, 同时清掉明文 api_key (改走凭据)
    cfg.models[idx] = replace(cfg.models[idx], credential_id=cid, api_key=None)

    try:
        _persist_config(cfg)
    except Exception as e:
        return {"success": False, "error": f"persist failed: {e}"}

    return {"success": True, "alias": alias, "credential_id": cid}
