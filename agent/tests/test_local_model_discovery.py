"""本地模型发现 + 凭据/配置桥接接口的测试。

覆盖:
- GET  /config/local-models              (ollama / vllm / 连接失败)
- _model_to_dict 带 credential_id
- POST /credentials/import-from-config
- POST /credentials/{cid}/link-model/{alias}
"""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from huginn.config import HuginnConfig, ModelConfig
from huginn.routes.config import _model_to_dict
from huginn.routes.config import router as config_router
from huginn.routes.credentials import router as credentials_router


@pytest.fixture()
def client():
    """挂上 config + credentials 两个路由的精简 app。

    conftest 里开了 HUGINN_DEV_MODE, require_admin_key 会直接放行,
    所以测试不用带 admin key。
    """
    app = FastAPI()
    app.include_router(config_router)
    app.include_router(credentials_router)
    with TestClient(app) as c:
        yield c


def _fake_urlopen(payload: dict):
    """造一个能进 with 语句、read() 返回 payload JSON 的假 urlopen 返回值。"""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


# ── /config/local-models ───────────────────────────────────────


def test_local_models_ollama(client):
    """ollama 走 /api/tags, 模型列表在 models[].name 里。"""
    payload = {"models": [{"name": "llama3:8b"}, {"name": "qwen2.5:14b"}]}
    with patch("urllib.request.urlopen", return_value=_fake_urlopen(payload)):
        r = client.get("/config/local-models", params={"provider": "ollama"})

    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["models"] == ["llama3:8b", "qwen2.5:14b"]
    assert data["provider"] == "ollama"
    # 没传 base_url 时用 ollama 默认地址
    assert data["base_url"] == "http://localhost:11434"


def test_local_models_vllm(client):
    """vllm / OpenAI 兼容服务走 /v1/models, 模型列表在 data[].id 里。"""
    payload = {"data": [{"id": "meta-llama/Llama-3-8B"}, {"id": "Qwen/Qwen2-7B"}]}
    with patch("urllib.request.urlopen", return_value=_fake_urlopen(payload)):
        r = client.get("/config/local-models", params={"provider": "vllm"})

    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["models"] == ["meta-llama/Llama-3-8B", "Qwen/Qwen2-7B"]
    assert data["provider"] == "vllm"
    assert data["base_url"] == "http://localhost:8000"


def test_local_models_connection_error(client):
    """本地服务没起 / 连不上时, 应回 success=False + error, 而不是抛 500。"""
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        r = client.get("/config/local-models", params={"provider": "ollama"})

    assert r.status_code == 200
    data = r.json()
    assert data["success"] is False
    assert data["models"] == []
    assert "error" in data
    assert data["error"]  # 错误信息非空


# ── _model_to_dict ──────────────────────────────────────────────


def test_model_dict_includes_credential_id(monkeypatch):
    """_model_to_dict 要带上 credential_id, 且凭据能提供 key 时 has_key=True。"""
    # stub 凭据 store: to_llm_info 返回带 key 的 dict, 模拟凭据里存了 api_key
    fake_store = MagicMock()
    fake_store.to_llm_info.return_value = {"api_key": "sk-from-cred"}
    import huginn.security.credential_store as cs_mod

    monkeypatch.setattr(cs_mod, "get_credential_store", lambda: fake_store)

    # 没填明文 api_key, 全靠 credential_id 提供凭据
    m = ModelConfig(
        alias="m1",
        provider="openai",
        model="gpt-4o",
        credential_id="cred-1",
        api_key=None,
    )
    d = _model_to_dict(m)

    assert "credential_id" in d
    assert d["credential_id"] == "cred-1"
    # 明文 key 没填, 但凭据里有 -> has_key 仍为 True
    assert d["has_key"] is True
    assert d["api_key"] is None  # None 脱敏后还是 None


# ── /credentials/import-from-config ────────────────────────────


def test_import_from_config_endpoint(client, monkeypatch, tmp_path):
    """import-from-config 扫配置里的明文 key 导入凭据 store, 回 {alias: cid} 映射。"""
    mapping = {"gpt4o": "cid-1", "claude": "cid-2"}

    class FakeStore:
        def import_from_config(self, cfg):
            # 真实实现会扫 cfg.models, 这里只验证接口契约: 原样回 mapping
            return mapping

    import huginn.security.credential_store as cs_mod

    monkeypatch.setattr(cs_mod, "get_credential_store", lambda: FakeStore())
    monkeypatch.setattr(
        "huginn.routes.credentials.get_credential_store", lambda: FakeStore()
    )

    # 指向一个不存在的文件, 走 from_env() 分支, 不依赖磁盘上的 huginn.toml
    monkeypatch.setenv("HUGINN_CONFIG_FILE", str(tmp_path / "no-such-file.toml"))

    r = client.post("/credentials/import-from-config")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["imported"] == mapping
    assert data["count"] == 2


# ── /credentials/{cid}/link-model/{alias} ─────────────────────


def test_link_model_endpoint(client, monkeypatch, tmp_path):
    """link-model: 设 credential_id, 清明文 api_key, 并触发落盘。"""
    # 先写一份带明文 key 的配置到 tmp
    cfg = HuginnConfig(
        models=[
            ModelConfig(
                alias="gpt4o",
                provider="openai",
                model="gpt-4o",
                api_key="sk-original",
            )
        ]
    )
    config_path = tmp_path / "huginn.toml"
    cfg.save(config_path, format="toml")
    monkeypatch.setenv("HUGINN_CONFIG_FILE", str(config_path))

    # 凭据存在: get_record 返回非 None 即可
    fake_store = MagicMock()
    fake_store.get_record.return_value = MagicMock(name="cred-record")
    import huginn.security.credential_store as cs_mod

    monkeypatch.setattr(cs_mod, "get_credential_store", lambda: fake_store)
    monkeypatch.setattr(
        "huginn.routes.credentials.get_credential_store", lambda: fake_store
    )

    # 拦截 _persist_config: 避免真落盘的副作用 (重置 agent factory / pet 等),
    # 顺便把传进去的 cfg 截下来好断言
    captured: dict = {}
    monkeypatch.setattr(
        "huginn.routes.config._persist_config",
        lambda c: captured.__setitem__("cfg", c),
    )

    r = client.post("/credentials/cid-1/link-model/gpt4o")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["alias"] == "gpt4o"
    assert data["credential_id"] == "cid-1"

    updated = captured["cfg"].models[0]
    assert updated.credential_id == "cid-1"
    assert updated.api_key is None
