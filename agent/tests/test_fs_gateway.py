"""ADR-0001 文件 I/O 归口后端：/v1/fs/* 端点测试。

验证后端提供 cwd / list / read / write，且继承 Tauri 原有的路径安全语义：
敏感目录（.ssh、ProgramData 等）与其他用户 profile 被拦截，用户可写区域放行。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from huginn import server


@pytest.fixture(scope="module")
def client():
    """Context-managed TestClient over server.app (TestClient hygiene guard 合规)."""
    with TestClient(server.app) as c:
        yield c


def _enable_restricted(monkeypatch):
    monkeypatch.delenv("HUGINN_ALLOW_UNRESTRICTED_READ", raising=False)


def test_fs_cwd(client):
    r = client.get("/v1/fs/cwd")
    assert r.status_code == 200
    body = r.json()
    assert body["path"] == os.getcwd()


def test_fs_list_roundtrip(client, tmp_path: Path):
    d = tmp_path / "sub"
    d.mkdir()
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    r = client.get("/v1/fs/list", params={"path": str(tmp_path)})
    assert r.status_code == 200
    entries = {e["name"]: e for e in r.json()["entries"]}
    assert "sub" in entries and entries["sub"]["is_dir"] is True
    assert "a.txt" in entries and entries["a.txt"]["is_dir"] is False
    # 目录排前面
    names = [e["name"] for e in r.json()["entries"]]
    dirs = [n for n in names if entries[n]["is_dir"]]
    files = [n for n in names if not entries[n]["is_dir"]]
    assert dirs + files == names


def test_fs_read_write(client, tmp_path: Path):
    target = tmp_path / "notes.txt"
    r = client.put("/v1/fs/write", json={"path": str(target), "content": "line1\nline2"})
    assert r.status_code == 200
    r = client.get("/v1/fs/read", params={"path": str(target)})
    assert r.status_code == 200
    assert r.json()["content"] == "line1\nline2"


def test_fs_write_creates_parent(tmp_path: Path):
    target = tmp_path / "nested" / "deep" / "f.txt"
    r = client.put("/v1/fs/write", json={"path": str(target), "content": "x"})
    assert r.status_code == 200
    assert target.exists()


def test_fs_blocks_sensitive_paths(monkeypatch, tmp_path: Path):
    """敏感路径（如 .ssh）在受限模式下被 403 拦截。"""
    _enable_restricted(monkeypatch)
    r = client.get("/v1/fs/list", params={"path": str(tmp_path / ".ssh")})
    # 即使目录不存在，也应按安全策略拒绝而非暴露
    assert r.status_code == 403


def test_fs_blocks_other_profile(monkeypatch, tmp_path: Path):
    """其他用户 profile（home 的父目录下、非当前 home）被拦截。"""
    _enable_restricted(monkeypatch)
    home = os.path.expanduser("~")
    other = str(Path(home).parent / "someone_else")
    r = client.get("/v1/fs/list", params={"path": other})
    assert r.status_code == 403


def test_fs_read_missing_returns_400(tmp_path: Path):
    r = client.get("/v1/fs/read", params={"path": str(tmp_path / "nope.txt")})
    assert r.status_code == 400
