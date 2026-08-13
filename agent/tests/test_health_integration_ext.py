"""health.py 集成路径测试 — live/ready/legacy/guidance 全分支.

覆盖 `_is_configured` 各判定、`health_ready` 的 sqlite/llm/mcp 三路检查
(含成功/失败/异常分支与 503 汇聚), legacy `/health`(configured/unconfigured/
model_pool/mcp_servers), 以及 `/health/guidance` 的 key/ollama/recommendation
分支。直接调用路由函数并注入 fake config / context / research_log / ollama 检查。

注: `/health/rust` 已在 test_health_rust_ext.py 覆盖, 此处不重复。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import Response

import huginn.routes.health as H
from huginn.config import HuginnConfig, ModelConfig


# ── helpers ────────────────────────────────────────────────────────


def _cfg(**kw) -> HuginnConfig:
    defaults: dict = {
        "provider": "default",
        "api_key": None,
        "model": "test-model",
        "models": [],
        "resolved_api_key": None,
    }
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _patch_config(monkeypatch, cfg):
    monkeypatch.setattr(H, "get_config", lambda: cfg)


def _patch_context(monkeypatch, mcp_manager=None):
    ctx = SimpleNamespace(mcp_manager=mcp_manager)
    monkeypatch.setattr(H, "get_context", lambda: ctx)


def _patch_ollama(monkeypatch, result):
    async def _check(host, timeout=2.0):
        return result

    monkeypatch.setattr(H, "_check_ollama_available", _check)


# ── _is_configured ─────────────────────────────────────────────────


def test_is_configured_ollama_provider():
    assert H._is_configured(_cfg(provider="ollama")) is True


def test_is_configured_models_ollama():
    m = SimpleNamespace(provider="ollama", enabled=True, api_key=None)
    assert H._is_configured(_cfg(models=[m])) is True


def test_is_configured_model_enabled_with_api_key():
    m = SimpleNamespace(provider="openai", enabled=True, api_key="sk-abc")
    assert H._is_configured(_cfg(models=[m])) is True


def test_is_configured_model_env_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    m = SimpleNamespace(provider="openai", enabled=True, api_key=None)
    assert H._is_configured(_cfg(models=[m])) is True


def test_is_configured_disabled_model_no_key():
    m = SimpleNamespace(provider="openai", enabled=False, api_key=None)
    assert H._is_configured(_cfg(models=[m])) is False


def test_is_configured_unknown_provider_env_missing():
    m = SimpleNamespace(provider="openai", enabled=True, api_key=None)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        assert H._is_configured(_cfg(models=[m])) is False
    finally:
        monkeypatch.undo()


def test_is_configured_resolved_key():
    assert H._is_configured(_cfg(provider="deepseek", resolved_api_key="key")) is True


def test_is_configured_default_no_key():
    assert H._is_configured(_cfg(provider="default", resolved_api_key=None)) is False


# ── /health/live ───────────────────────────────────────────────────


@pytest.mark.anyio
async def test_health_live(monkeypatch):
    _patch_config(monkeypatch, _cfg(provider="ollama", model="qwen3"))
    res = await H.health_live()
    assert res["status"] == "alive"
    assert res["provider"] == "ollama"
    assert res["model"] == "qwen3"
    assert "version" in res


# ── /health/ready ──────────────────────────────────────────────────


class _CtxLock:
    def __enter__(self):
        return self

    def __exit__(self, *a, **k):
        return False


def _fake_log(raise_on_execute=False):
    if raise_on_execute:

        class _Conn:
            def execute(self, *a, **k):
                raise RuntimeError("sqlite exploded")

        return SimpleNamespace(_lock=_CtxLock(), _conn=_Conn())
    return SimpleNamespace(
        _lock=_CtxLock(),
        _conn=SimpleNamespace(execute=lambda *a, **k: SimpleNamespace(fetchone=lambda: (1,))),
    )


def _patch_log(monkeypatch, log):
    import huginn.research_log as rl

    monkeypatch.setattr(rl, "get_research_log", lambda: log)


@pytest.mark.anyio
async def test_ready_all_ok(monkeypatch):
    _patch_log(monkeypatch, _fake_log())
    # LLM ok via ollama provider
    _patch_config(monkeypatch, _cfg(provider="ollama"))
    # MCP: manager with all connected
    mgr = SimpleNamespace(get_server_status=lambda: {"srv": {"connected": True}})
    _patch_context(monkeypatch, mcp_manager=mgr)
    resp = Response()
    data = await H.health_ready(resp)
    assert data["ready"] is True
    assert data["checks"]["sqlite"]["status"] == "ok"
    assert data["checks"]["llm"]["status"] == "ok"
    assert data["checks"]["mcp"]["status"] == "ok"
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_ready_sqlite_fail(monkeypatch):
    _patch_log(monkeypatch, _fake_log(raise_on_execute=True))
    _patch_config(monkeypatch, _cfg(provider="ollama"))
    _patch_context(monkeypatch, mcp_manager=None)
    resp = Response()
    data = await H.health_ready(resp)
    assert data["ready"] is False
    assert data["checks"]["sqlite"]["status"] == "fail"
    assert "error" in data["checks"]["sqlite"]
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_ready_llm_unconfigured(monkeypatch):
    _patch_log(monkeypatch, _fake_log())
    _patch_config(monkeypatch, _cfg(provider="default", resolved_api_key=None))
    _patch_context(monkeypatch, mcp_manager=None)
    resp = Response()
    data = await H.health_ready(resp)
    assert data["ready"] is False
    assert data["checks"]["llm"] == {"status": "fail", "error": "No provider configured"}


@pytest.mark.anyio
async def test_ready_llm_exception(monkeypatch):
    _patch_log(monkeypatch, _fake_log())

    def _boom():
        raise RuntimeError("cfg boom")

    monkeypatch.setattr(H, "get_config", _boom)
    _patch_context(monkeypatch, mcp_manager=None)
    resp = Response()
    data = await H.health_ready(resp)
    assert data["checks"]["llm"]["status"] == "fail"
    assert "error" in data["checks"]["llm"]


@pytest.mark.anyio
async def test_ready_mcp_not_configured(monkeypatch):
    _patch_log(monkeypatch, _fake_log())
    _patch_config(monkeypatch, _cfg(provider="ollama"))
    _patch_context(monkeypatch, mcp_manager=None)
    resp = Response()
    data = await H.health_ready(resp)
    assert data["checks"]["mcp"] == {"status": "ok", "note": "not configured"}


@pytest.mark.anyio
async def test_ready_mcp_no_servers(monkeypatch):
    _patch_log(monkeypatch, _fake_log())
    _patch_config(monkeypatch, _cfg(provider="ollama"))
    _patch_context(monkeypatch, mcp_manager=SimpleNamespace(get_server_status=lambda: {}))
    resp = Response()
    data = await H.health_ready(resp)
    assert data["checks"]["mcp"] == {"status": "ok", "note": "no servers"}


@pytest.mark.anyio
async def test_ready_mcp_disconnected(monkeypatch):
    _patch_log(monkeypatch, _fake_log())
    _patch_config(monkeypatch, _cfg(provider="ollama"))
    mgr = SimpleNamespace(get_server_status=lambda: {"a": {"connected": True}, "b": {"connected": False}})
    _patch_context(monkeypatch, mcp_manager=mgr)
    resp = Response()
    data = await H.health_ready(resp)
    assert data["ready"] is False
    assert data["checks"]["mcp"]["status"] == "fail"
    assert "b" in data["checks"]["mcp"]["error"]


@pytest.mark.anyio
async def test_ready_mcp_exception(monkeypatch):
    _patch_log(monkeypatch, _fake_log())
    _patch_config(monkeypatch, _cfg(provider="ollama"))

    def _boom():
        raise RuntimeError("mcp boom")

    _patch_context(monkeypatch, mcp_manager=SimpleNamespace(get_server_status=_boom))
    resp = Response()
    data = await H.health_ready(resp)
    assert data["checks"]["mcp"]["status"] == "fail"
    assert "error" in data["checks"]["mcp"]


# ── legacy /health ─────────────────────────────────────────────────


@pytest.mark.anyio
async def test_health_legacy_configured(monkeypatch):
    _patch_config(monkeypatch, _cfg(provider="openai", resolved_api_key="key", models=[]))
    _patch_context(monkeypatch, mcp_manager=None)
    res = await H.health()
    assert res["status"] == "ok"
    assert res["configured"] is True
    assert "model_pool" not in res
    assert "mcp_servers" not in res


@pytest.mark.anyio
async def test_health_legacy_unconfigured(monkeypatch):
    _patch_config(monkeypatch, _cfg(provider="default"))
    _patch_context(monkeypatch, mcp_manager=None)
    res = await H.health()
    assert res["status"] == "unconfigured"
    assert res["configured"] is False


@pytest.mark.anyio
async def test_health_legacy_model_pool_and_mcp(monkeypatch):
    config = _cfg(provider="openai", resolved_api_key="key")
    config.models = [
        SimpleNamespace(enabled=True, provider="openai", api_key="k", alias="a"),
        SimpleNamespace(enabled=False, provider="openai", api_key="k", alias="b"),
    ]
    _patch_config(monkeypatch, config)
    _patch_context(monkeypatch, mcp_manager=SimpleNamespace(get_server_status=lambda: {"srv": {"connected": True}}))
    res = await H.health()
    assert res["model_pool"] == 1
    assert res["mcp_servers"] == {"srv": {"connected": True}}


# ── /health/guidance ───────────────────────────────────────────────


@pytest.mark.anyio
async def test_guidance_key_detected_recommendation(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    _patch_config(monkeypatch, _cfg(provider="default"))
    _patch_ollama(monkeypatch, False)
    res = await H.health_guidance()
    providers = [p["provider"] for p in res["available_providers"]]
    assert "openai" in providers
    assert res["configured"] is False
    assert res["recommendation"]["action"] == "set_provider"
    assert res["recommendation"]["provider"] == "openai"


@pytest.mark.anyio
async def test_guidance_ollama_recommendation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _patch_config(monkeypatch, _cfg(provider="default"))
    _patch_ollama(monkeypatch, True)
    res = await H.health_guidance()
    # ollama 在 keyless 列表且 available
    keyless = {k["provider"]: k["available"] for k in res["keyless_providers"]}
    assert keyless["ollama"] is True
    assert res["recommendation"]["action"] == "set_provider"
    assert res["recommendation"]["provider"] == "ollama"


@pytest.mark.anyio
async def test_guidance_manual_setup(monkeypatch):
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    _patch_config(monkeypatch, _cfg(provider="default"))
    _patch_ollama(monkeypatch, False)
    res = await H.health_guidance()
    assert res["configured"] is False
    assert res["recommendation"]["action"] == "manual_setup"
    assert "suggestion" in res["recommendation"]


@pytest.mark.anyio
async def test_guidance_ollama_unavailable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _patch_config(monkeypatch, _cfg(provider="default"))
    _patch_ollama(monkeypatch, False)
    res = await H.health_guidance()
    keyless = {k["provider"]: k["available"] for k in res["keyless_providers"]}
    assert "ollama" in keyless
    assert keyless["ollama"] is False


@pytest.mark.anyio
async def test_guidance_already_configured_no_recommendation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    _patch_config(monkeypatch, _cfg(provider="openai", resolved_api_key="key"))
    _patch_ollama(monkeypatch, False)
    res = await H.health_guidance()
    assert res["configured"] is True
    assert res["recommendation"] == {}