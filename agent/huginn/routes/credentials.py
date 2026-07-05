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
import logging
import traceback
from typing import Any

from fastapi import APIRouter, Depends

from huginn.hpc.client import HPCClient
from huginn.models.registry import create_langchain_model
from huginn.security.auth import require_admin_key
from huginn.security.credential_store import (
    CRED_KIND_LLM,
    CRED_KIND_SSH,
    SUPPORTED_SERVICES,
    get_credential_store,
    get_service_credential_store,
)

router = APIRouter(tags=["credentials"])

logger = logging.getLogger(__name__)


def _store():
    # 包一层方便测试 monkeypatch
    return get_credential_store()


def _svc_store():
    # 包一层方便测试 monkeypatch — 与 _store() 对应, 指 service API key 存储
    return get_service_credential_store()


# ── 列表 / 详情 / 默认 ─────────────────────────────────────────


@router.get("/credentials", dependencies=[Depends(require_admin_key)])
async def list_credentials(kind: str | None = None) -> dict[str, Any]:
    """列出凭据 (脱敏)。?kind=ssh 或 ?kind=llm 过滤; 不传返回全部。

    返回体里同时带上 ``services`` — 已配置的外部服务 API key 清单 (只有
    service 名 + metadata + has_key, 永不回明文/密文 key)。这样前端一个
    请求就能同时拿到 SSH/LLM 凭据和外部服务 key 的状态。
    """
    store = _store()
    if kind and kind not in (CRED_KIND_SSH, CRED_KIND_LLM):
        return {"success": False, "error": f"kind 必须是 ssh 或 llm, 得到 {kind!r}"}
    return {
        "credentials": store.list(kind=kind),
        "services": _svc_store().list_services(),
    }


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
    """删除凭据。

    双模式分发: 路径参数既可能是外部服务名 (openai / materials_project ...),
    也可能是 SSH/LLM 凭据的 hex id。服务名命中 SUPPORTED_SERVICES 时按 service
    凭据删, 否则按 cid 走原来的 SQLite 删除 (删默认则同 kind 自动提升最早一条)。
    前端传的 id 是 hex token, 不会和服务名撞, 所以可以放心合在一个端点里。
    """
    if cid in SUPPORTED_SERVICES:
        ok = _svc_store().delete_credential(cid)
        if not ok:
            return {"success": False, "error": f"服务 {cid} 未配置凭据"}
        return {"success": True, "service": cid}
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
        logger.error("unexpected error", exc_info=True)
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
        logger.error("unexpected error", exc_info=True)
        return {
            "success": False,
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": str(e),
        }


# ── 凭据 ↔ 模型配置 桥接 ──────────────────────────────────────


@router.post("/credentials/import-from-config", dependencies=[Depends(require_admin_key)])
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


@router.post("/credentials/{cid}/link-model/{alias}", dependencies=[Depends(require_admin_key)])
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


# ════════════════════════════════════════════════════════════════════
# 外部服务 API Key 管理 (service-keyed, 加密 JSON 落盘)
# ════════════════════════════════════════════════════════════════════
#
# 与上面的 SSH/LLM 凭据 CRUD 并列, 这一组面向"外部数据源 / 出版商 /
# LLM provider"的 API key: 一份 service 一份 key。端点:
#     POST   /credentials/{service}        新增 / 更新一份 service 凭据
#     GET    /credentials/{service}/test   探活: 拿 key 去服务端问一句
# 列表和删除复用上面已注册的 GET /credentials / DELETE /credentials/{cid}
# — service 名会命中 SUPPORTED_SERVICES, 由那两个端点做双模式分发。


@router.post("/credentials/{service}", dependencies=[Depends(require_admin_key)])
async def set_service_credential(
    service: str, params: dict[str, Any]
) -> dict[str, Any]:
    """新增 / 更新一份外部服务 API key。

    body 字段:
        api_key    str   必填, 明文 (落盘前会 Fernet 加密)
        metadata   dict  非敏感附加信息 (base_url / owner / 备注 ...), 可选

    service 必须在 SUPPORTED_SERVICES 里, 否则拒掉 — 避免任意字符串当
    service 名写进存储。
    """
    if service not in SUPPORTED_SERVICES:
        return {
            "success": False,
            "error": (
                f"不支持的服务 {service!r}, 可选: {SUPPORTED_SERVICES}"
            ),
        }
    api_key = params.get("api_key")
    if not api_key or not str(api_key).strip():
        return {"success": False, "error": "api_key 必填且不能为空"}
    metadata = params.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {"success": False, "error": "metadata 必须是对象"}

    try:
        _svc_store().set_credential(service, str(api_key), metadata)
    except Exception as e:
        logger.error("写入 service 凭据失败: %s", service, exc_info=True)
        return {"success": False, "error": str(e)}
    return {"success": True, "service": service}


@router.get(
    "/credentials/{service}/test", dependencies=[Depends(require_admin_key)]
)
async def test_service_credential(service: str) -> dict[str, Any]:
    """探活外部服务凭据: 取明文 key 去服务端发一个最小请求, 看 401/403 还是 200。

    返回 ``{valid: bool, error: str?}``。没有配 key / 服务名不支持 /
    请求失败都会回 valid=False 并带上原因; 只有真返回 2xx 才算 valid=True。
    网络不通时不算 key 无效, 会把网络错误回传让前端区分提示。
    """
    if service not in SUPPORTED_SERVICES:
        return {
            "valid": False,
            "error": (
                f"不支持的服务 {service!r}, 可选: {SUPPORTED_SERVICES}"
            ),
        }
    store = _svc_store()
    if not store.has_credential(service):
        return {"valid": False, "error": f"未配置 {service} 的凭据"}

    api_key = store.get_credential(service)
    tester = _SERVICE_TESTERS.get(service) or _format_check
    try:
        valid, error = await tester(api_key)
    except Exception as e:
        logger.error("service 凭据探活异常: %s", service, exc_info=True)
        return {"valid": False, "error": str(e)}
    return {"valid": bool(valid), "error": error}


# ── 服务探活实现 ───────────────────────────────────────────────
#
# 能做"真问一句"的服务走 HTTP 探活 (aiohttp, 10s 超时); 没有公开探活端点
# 或需要 OAuth 的 (google_ai / wiley / chemspider ...) 退回格式校验 —
# 至少挡掉明显的空 key / 短 key, 不假装能验。

_SERVICE_TEST_TIMEOUT = 10.0


def _http_tester(url, *, build_headers=None, build_params=None, ok_status=200):
    """造一个"GET 一发看状态码"的 async 探活函数。

    build_headers / build_params 都是 ``api_key -> dict`` 的 callable,
    把 key 拼进 header (Bearer / X-API-KEY) 或 query (apikey=)。
    ok_status 默认 200; 有些服务 204 也算正常, 调用时覆盖即可。
    """

    async def _tester(api_key: str) -> tuple[bool, str | None]:
        try:
            import aiohttp
        except ImportError:  # pragma: no cover - aiohttp 是硬依赖
            return False, "服务器缺少 aiohttp, 无法探活"
        headers = build_headers(api_key) if build_headers else {}
        params = build_params(api_key) if build_params else None
        timeout = aiohttp.ClientTimeout(total=_SERVICE_TEST_TIMEOUT)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    url, headers=headers, params=params
                ) as resp:
                    if resp.status == ok_status:
                        return True, None
                    body = await resp.text()
                    # 401/403 是 key 问题, 其它状态码也算探活失败但带原文
                    return False, f"HTTP {resp.status}: {body[:160]}"
        except asyncio.TimeoutError:
            return False, f"请求超时 ({int(_SERVICE_TEST_TIMEOUT)}s)"
        except Exception as e:  # noqa: BLE001 - 探活要把任何异常透传给前端
            return False, f"{type(e).__name__}: {e}"

    return _tester


async def _format_check(api_key: str) -> tuple[bool, str | None]:
    """无公开探活端点的服务: 退回格式校验。

    只挡明显的空 / 过短 key; 没法验真伪时按"格式 OK"放行, 由前端展示
    "未做活体验证"提示。Google AI / Wiley / ChemSpider 这类要 OAuth 或
    私有协议的都走这里。
    """
    if not api_key or len(api_key) < 8:
        return False, "key 过短或为空 (少于 8 字符)"
    return True, None


# service → async tester(api_key) -> (valid, error)
# 没列在这里的 (google_ai / wiley / elsevier_science_direct / arxiv /
# nist_webbook / pubchem / chemspider) 走 _format_check 兜底。
_SERVICE_TESTERS = {
    "openai": _http_tester(
        "https://api.openai.com/v1/models",
        build_headers=lambda k: {"Authorization": f"Bearer {k}"},
    ),
    "anthropic": _http_tester(
        "https://api.anthropic.com/v1/models?limit=1",
        build_headers=lambda k: {
            "x-api-key": k,
            "anthropic-version": "2023-06-01",
        },
    ),
    "deepseek": _http_tester(
        "https://api.deepseek.com/models",
        build_headers=lambda k: {"Authorization": f"Bearer {k}"},
    ),
    "qwen": _http_tester(
        # DashScope 的 OpenAI 兼容端点
        "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
        build_headers=lambda k: {"Authorization": f"Bearer {k}"},
    ),
    "materials_project": _http_tester(
        "https://api.materialsproject.org/rest/v2/api_check",
        build_headers=lambda k: {"X-API-KEY": k},
    ),
    "scopus": _http_tester(
        # Elsevier API: apikey 走 query
        "https://api.elsevier.com/authenticate",
        build_params=lambda k: {"apikey": k},
    ),
    "springer_nature": _http_tester(
        "https://api.springernature.com/metadata/json",
        build_params=lambda k: {"q": "doi:1", "api_key": k},
    ),
    "semantic_scholar": _http_tester(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        build_headers=lambda k: {"x-api-key": k},
        build_params=lambda _k: {"query": "test", "limit": 1},
    ),
}
