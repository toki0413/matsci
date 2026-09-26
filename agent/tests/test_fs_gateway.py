"""ADR-0001 文件 I/O 归口后端：/v1/fs/* 端点测试。

验证后端提供 cwd / list / read / write，且继承 Tauri 原有的路径安全语义：
敏感目录（.ssh、ProgramData 等）与其他用户 profile 被拦截，用户可写区域放行。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def client(app_client):
    """Alias 到 conftest 的 context-managed app_client (TestClient hygiene guard 合规).

    server 由 conftest.shared_huginn_app 提供, 与 server.app 同一实例.
    """
    return app_client


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


def test_fs_write_creates_parent(client, tmp_path: Path):
    target = tmp_path / "nested" / "deep" / "f.txt"
    r = client.put("/v1/fs/write", json={"path": str(target), "content": "x"})
    assert r.status_code == 200
    assert target.exists()


def test_fs_blocks_sensitive_paths(client, monkeypatch, tmp_path: Path):
    """敏感路径（如 .ssh）在受限模式下被 403 拦截。"""
    _enable_restricted(monkeypatch)
    r = client.get("/v1/fs/list", params={"path": str(tmp_path / ".ssh")})
    # 即使目录不存在，也应按安全策略拒绝而非暴露
    assert r.status_code == 403


def test_fs_blocks_other_profile(client, monkeypatch, tmp_path: Path):
    """其他用户 profile（home 的父目录下、非当前 home）被拦截。"""
    _enable_restricted(monkeypatch)
    home = os.path.expanduser("~")
    other = str(Path(home).parent / "someone_else")
    r = client.get("/v1/fs/list", params={"path": other})
    assert r.status_code == 403


def test_fs_read_missing_returns_400(client, tmp_path: Path):
    r = client.get("/v1/fs/read", params={"path": str(tmp_path / "nope.txt")})
    assert r.status_code == 400


# ── 附件上传 (浏览器拖拽二进制 → 工作区) ────────────────────────────


def test_fs_upload_writes_into_given_dir(client, tmp_path: Path):
    r = client.post(
        "/v1/fs/upload",
        files={"file": ("pic.bin", b"\x00\x01\x02")},
        data={"dir": str(tmp_path)},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["size"] == 3
    assert Path(body["path"]) == tmp_path / "pic.bin"
    assert (tmp_path / "pic.bin").read_bytes() == b"\x00\x01\x02"


def test_fs_upload_defaults_to_workspace_huginn_uploads(client, tmp_path: Path, monkeypatch):
    """不传 dir 时落 `<cwd>/.huginn/uploads/`, 不散落到项目根."""
    monkeypatch.chdir(tmp_path)
    r = client.post("/v1/fs/upload", files={"file": ("drop.png", b"png")})
    assert r.status_code == 200
    assert Path(r.json()["path"]) == tmp_path / ".huginn" / "uploads" / "drop.png"
    assert not (tmp_path / "drop.png").exists()


def test_fs_upload_strips_path_from_filename(client, tmp_path: Path):
    """文件名只取 basename: `../../evil.txt` 不能穿越出目标目录."""
    r = client.post(
        "/v1/fs/upload",
        files={"file": ("../../evil.txt", b"x")},
        data={"dir": str(tmp_path)},
    )
    assert r.status_code == 200
    assert Path(r.json()["path"]) == tmp_path / "evil.txt"


def test_fs_upload_avoids_overwriting_same_name(client, tmp_path: Path):
    (tmp_path / "a.txt").write_text("original", encoding="utf-8")
    r = client.post(
        "/v1/fs/upload",
        files={"file": ("a.txt", b"new")},
        data={"dir": str(tmp_path)},
    )
    assert r.status_code == 200
    assert Path(r.json()["path"]) == tmp_path / "a-1.txt"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "original"


def test_fs_upload_rejects_oversize(client, tmp_path: Path, monkeypatch):
    monkeypatch.setattr("huginn.routes.fs._UPLOAD_MAX_BYTES", 4)
    r = client.post(
        "/v1/fs/upload",
        files={"file": ("big.bin", b"0123456789")},
        data={"dir": str(tmp_path)},
    )
    assert r.status_code == 413
    # 超限的半截文件要清掉, 不留垃圾
    assert list(tmp_path.iterdir()) == []


def test_fs_upload_rejects_non_dir_target(client, tmp_path: Path):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    r = client.post(
        "/v1/fs/upload", files={"file": ("b.bin", b"x")}, data={"dir": str(target)}
    )
    assert r.status_code == 400
