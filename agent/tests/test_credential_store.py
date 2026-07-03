"""Tests for the encrypted credential store (SSH + LLM API keys).

数据层用真实 Fernet + 临时 DB 验证加密/脱敏/CRUD/默认提升;
路由层用一个只挂 credentials_router 的最小 FastAPI app 跑端到端。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from huginn.routes.credentials import router as credentials_router
from huginn.security import credential_store as cs_mod
from huginn.security.credential_store import (
    CRED_KIND_LLM,
    CRED_KIND_SSH,
    CredentialStore,
    _cred_dir,
    _get_fernet,
    get_credential_store,
    mask_secret,
    reset_credential_store,
)


@pytest.fixture()
def store(tmp_path):
    """隔离的 CredentialStore: 真实 Fernet key + 临时 DB, 不碰 ~/.huginn。"""
    fernet = Fernet(Fernet.generate_key())
    return CredentialStore(tmp_path / "cred.sqlite", fernet=fernet)


@pytest.fixture()
def client(store, monkeypatch):
    """最小 FastAPI app 只挂 credentials 路由, _store() 指向隔离 store。"""
    app = FastAPI()
    app.include_router(credentials_router)
    # 路由里通过 _store() 拿单例; 这里替换成隔离实例, 避免污染全局
    monkeypatch.setattr(cs_mod, "get_credential_store", lambda: store)
    monkeypatch.setattr(
        "huginn.routes.credentials.get_credential_store", lambda: store
    )
    return TestClient(app)


# ── 脱敏 ─────────────────────────────────────────────────────


def test_mask_secret_basic():
    assert mask_secret(None) is None
    assert mask_secret("") == ""
    assert mask_secret("abc") == "********"  # 不足 8 位全掩
    assert mask_secret("sk-abcdef123456") == "sk-a****3456"


# ── CRUD ─────────────────────────────────────────────────────


def test_create_and_get_ssh(store):
    rec = store.create(
        CRED_KIND_SSH,
        "lab-cluster",
        metadata={"host": "hpc.lab.edu", "username": "wz", "port": 22, "scheduler": "slurm"},
        secret="s3cretpass",
    )
    assert rec["kind"] == "ssh"
    assert rec["name"] == "lab-cluster"
    assert rec["is_default"] is True  # 首条自动默认
    assert rec["has_secret"] is True
    assert rec["secret_masked"] == "s3cr****pass"
    # 明文绝不出现在脱敏视图里
    assert "s3cretpass" not in str(rec)
    # get_secret 才拿得到明文
    assert store.get_secret(rec["id"]) == "s3cretpass"


def test_create_llm(store):
    rec = store.create(
        CRED_KIND_LLM,
        "deepseek-main",
        metadata={"provider": "deepseek", "model": "deepseek-chat"},
        secret="sk-deepseek-xxx",
    )
    assert rec["kind"] == "llm"
    assert rec["has_secret"] is True
    assert store.get_secret(rec["id"]) == "sk-deepseek-xxx"


def test_list_filters_by_kind(store):
    store.create(CRED_KIND_SSH, "s1", metadata={"host": "h1", "username": "u"}, secret="p1")
    store.create(CRED_KIND_LLM, "l1", metadata={"provider": "openai", "model": "gpt"}, secret="k1")
    assert len(store.list(kind=CRED_KIND_SSH)) == 1
    assert len(store.list(kind=CRED_KIND_LLM)) == 1
    assert len(store.list()) == 2


def test_update_partial(store):
    rec = store.create(
        CRED_KIND_SSH, "c1", metadata={"host": "h", "username": "u"}, secret="oldpass"
    )
    # 只改 name + metadata, 不传 secret => 密钥保持不变
    updated = store.update(
        rec["id"], name="c1-renamed", metadata={"host": "h2", "username": "u2"}
    )
    assert updated["name"] == "c1-renamed"
    assert updated["metadata"]["host"] == "h2"
    assert store.get_secret(rec["id"]) == "oldpass"
    # 改 secret
    store.update(rec["id"], secret="newpass")
    assert store.get_secret(rec["id"]) == "newpass"
    # 清空 secret (空串语义)
    store.update(rec["id"], secret="")
    assert store.get_secret(rec["id"]) == ""
    assert store.get(rec["id"])["has_secret"] is False


def test_update_nonexistent(store):
    assert store.update("nope", name="x") is None


def test_delete(store):
    rec = store.create(
        CRED_KIND_SSH, "c1", metadata={"host": "h", "username": "u"}, secret="p"
    )
    assert store.delete(rec["id"]) is True
    assert store.get(rec["id"]) is None
    assert store.delete(rec["id"]) is False  # 已删


def test_create_validates_kind_and_name(store):
    with pytest.raises(ValueError):
        store.create("bad-kind", "x", secret="p")
    with pytest.raises(ValueError):
        store.create(CRED_KIND_SSH, "", secret="p")


# ── 默认凭据 ─────────────────────────────────────────────────


def test_first_entry_auto_default(store):
    r1 = store.create(CRED_KIND_SSH, "s1", metadata={"host": "h1", "username": "u"}, secret="p1")
    r2 = store.create(CRED_KIND_SSH, "s2", metadata={"host": "h2", "username": "u"}, secret="p2")
    assert r1["is_default"] is True
    assert r2["is_default"] is False
    assert store.get_default(CRED_KIND_SSH)["id"] == r1["id"]
    # 切默认
    assert store.set_default(r2["id"]) is True
    assert store.get_default(CRED_KIND_SSH)["id"] == r2["id"]
    assert store.get(r1["id"])["is_default"] is False


def test_delete_default_promotes_survivor(store):
    r1 = store.create(CRED_KIND_SSH, "s1", metadata={"host": "h1", "username": "u"}, secret="p1")
    r2 = store.create(CRED_KIND_SSH, "s2", metadata={"host": "h2", "username": "u"}, secret="p2")
    # r1 是默认, 删掉后 r2 应被提升为默认
    store.delete(r1["id"])
    assert store.get_default(CRED_KIND_SSH)["id"] == r2["id"]


def test_default_isolated_per_kind(store):
    s = store.create(CRED_KIND_SSH, "s1", metadata={"host": "h", "username": "u"}, secret="p")
    l = store.create(CRED_KIND_LLM, "l1", metadata={"provider": "openai", "model": "gpt"}, secret="k")
    assert store.get_default(CRED_KIND_SSH)["id"] == s["id"]
    assert store.get_default(CRED_KIND_LLM)["id"] == l["id"]


# ── 适配层: 转现有系统配置对象 ───────────────────────────────


def test_to_hpc_config(store):
    rec = store.create(
        CRED_KIND_SSH,
        "cluster",
        metadata={
            "host": "hpc.x.edu",
            "username": "wz",
            "port": 2222,
            "scheduler": "pbs",
            "remote_work_dir": "/scratch/wz",
        },
        secret="sshpass",
    )
    cfg = store.to_hpc_config(rec["id"])
    assert cfg is not None
    assert cfg.host == "hpc.x.edu"
    assert cfg.username == "wz"
    assert cfg.port == 2222
    assert cfg.scheduler == "pbs"
    assert cfg.password == "sshpass"  # 解密后的明文
    assert cfg.remote_work_dir == "/scratch/wz"


def test_to_hpc_config_wrong_kind(store):
    rec = store.create(
        CRED_KIND_LLM, "l", metadata={"provider": "openai", "model": "gpt"}, secret="k"
    )
    assert store.to_hpc_config(rec["id"]) is None


def test_to_llm_info(store):
    rec = store.create(
        CRED_KIND_LLM,
        "ds",
        metadata={
            "provider": "deepseek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
        },
        secret="sk-ds",
    )
    info = store.to_llm_info(rec["id"])
    assert info is not None
    assert info["provider"] == "deepseek"
    assert info["model"] == "deepseek-chat"
    assert info["api_key"] == "sk-ds"


# ── 加密落盘 ─────────────────────────────────────────────────


def test_secret_encrypted_at_rest(store, tmp_path):
    rec = store.create(
        CRED_KIND_LLM,
        "l",
        metadata={"provider": "openai", "model": "gpt"},
        secret="sk-super-secret",
    )
    # 直接读 sqlite 原始密文, 确保明文不在磁盘上
    conn = sqlite3.connect(tmp_path / "cred.sqlite")
    row = conn.execute(
        "SELECT secret_enc FROM credentials WHERE id=?", (rec["id"],)
    ).fetchone()
    conn.close()
    assert row[0] != "sk-super-secret"
    assert "sk-super-secret" not in row[0]


def test_persistence_across_instances(tmp_path):
    """同一 fernet key 的两个 store 实例能互相读到加密数据。"""
    key = Fernet.generate_key()
    s1 = CredentialStore(tmp_path / "c.db", fernet=Fernet(key))
    rec = s1.create(CRED_KIND_SSH, "s", metadata={"host": "h", "username": "u"}, secret="pw")
    # 新实例同 key 同 db => 能解出明文
    s2 = CredentialStore(tmp_path / "c.db", fernet=Fernet(key))
    assert s2.get_secret(rec["id"]) == "pw"


def test_wrong_key_cannot_decrypt(tmp_path):
    """换了 fernet key, 旧密文解不出来 (抛 RuntimeError 而非泄漏密文)。"""
    s1 = CredentialStore(tmp_path / "c.db", fernet=Fernet(Fernet.generate_key()))
    rec = s1.create(CRED_KIND_SSH, "s", metadata={"host": "h", "username": "u"}, secret="pw")
    s2 = CredentialStore(tmp_path / "c.db", fernet=Fernet(Fernet.generate_key()))
    with pytest.raises(RuntimeError):
        s2.get_secret(rec["id"])


# ── 路径隔离: 尊重 HUGINN_CACHE_DIR ──────────────────────────


def test_cred_dir_respects_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    assert _cred_dir() == tmp_path


def test_get_fernet_uses_cache_dir(monkeypatch, tmp_path):
    """无主密码时, cred.key 应落到 HUGINN_CACHE_DIR 下, 不污染 ~/.huginn。"""
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HUGINN_ENCRYPTION_PASSWORD", raising=False)
    monkeypatch.delenv("HUGINN_CREDENTIAL_KEY_FILE", raising=False)
    reset_credential_store()
    _get_fernet()
    assert (tmp_path / "cred.key").exists()


def test_singleton_respects_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HUGINN_CREDENTIAL_DB", raising=False)
    reset_credential_store()
    s = get_credential_store()
    assert str(s.db_path).startswith(str(tmp_path))
    reset_credential_store()


# ── 路由层端到端 ─────────────────────────────────────────────


def test_route_list_empty(client):
    r = client.get("/credentials")
    assert r.status_code == 200
    assert r.json()["credentials"] == []


def test_route_create_list_setdefault_delete(client):
    # 创建
    r = client.post(
        "/credentials",
        json={
            "kind": "ssh",
            "name": "lab",
            "metadata": {"host": "h", "username": "u"},
            "secret": "topsecret",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    cid = body["credential"]["id"]
    # 明文不出现在 HTTP 响应里
    assert "topsecret" not in r.text
    # 列表
    r = client.get("/credentials?kind=ssh")
    assert len(r.json()["credentials"]) == 1
    # 设默认
    r = client.post(f"/credentials/{cid}/set-default")
    assert r.json()["success"] is True
    # defaults
    r = client.get("/credentials/defaults")
    assert r.json()["ssh"]["id"] == cid
    # 删除
    r = client.delete(f"/credentials/{cid}")
    assert r.json()["success"] is True
    assert client.get("/credentials").json()["credentials"] == []


def test_route_update_does_not_leak_plaintext(client):
    r = client.post(
        "/credentials",
        json={
            "kind": "llm",
            "name": "ds",
            "metadata": {"provider": "deepseek", "model": "deepseek-chat"},
            "secret": "sk-original",
        },
    )
    cid = r.json()["credential"]["id"]
    # 更新密钥
    r = client.put(
        f"/credentials/{cid}",
        json={"secret": "sk-rotated-newkey"},
    )
    assert r.status_code == 200
    assert "sk-rotated-newkey" not in r.text
    assert r.json()["credential"]["has_secret"] is True


def test_route_create_validates(client):
    r = client.post("/credentials", json={"kind": "ssh", "name": ""})
    assert r.json()["success"] is False
    r = client.post("/credentials", json={"kind": "bad", "name": "x"})
    assert r.json()["success"] is False


def test_route_test_nonexistent(client):
    r = client.post("/credentials/nope/test")
    assert r.json()["success"] is False
