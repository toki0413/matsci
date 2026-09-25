"""浏览器形状的文件传输端点：POST /transfer/web/upload 与 GET /transfer/web/download。

FilesPanel 的「上传/下载」按钮拿到的是浏览器 File / Blob, 套不上 `/transfer/upload|
download` 的 `local_path` 契约, 故新增这两个浏览器形状的端点。这里用内存版 SFTP
替身接管 HPCClient, 只验证端点自身的接线: 落盘字节、basename 取法、父目录创建、
大小上限、路径安全、404。
"""

from __future__ import annotations

import pytest

from huginn.routes import transfer


@pytest.fixture(scope="module")
def client(app_client):
    """Alias 到 conftest 的 context-managed app_client (TestClient hygiene guard 合规)."""
    return app_client


class _FakeAttr:
    def __init__(self, size: int = 0) -> None:
        self.st_size = size


class _FakeRemoteFile:
    """内存远端文件句柄: 写时缓冲, 退出上下文才落进 files (模拟关句柄才生效)。"""

    def __init__(self, sftp: _FakeSFTP, path: str, mode: str) -> None:
        self._sftp = sftp
        self._path = path
        self._mode = mode
        self._buf = bytearray()
        self._pos = 0

    def __enter__(self) -> _FakeRemoteFile:
        return self

    def __exit__(self, *exc) -> bool:
        if "w" in self._mode:
            self._sftp.files[self._path] = bytes(self._buf)
        return False

    def write(self, data: bytes) -> int:
        self._buf += data
        return len(data)

    def read(self, n: int = -1) -> bytes:
        data = self._sftp.files.get(self._path, b"")
        if n is None or n < 0:
            out, self._pos = data[self._pos :], len(data)
            return out
        out = data[self._pos : self._pos + n]
        self._pos += len(out)
        return out


class _FakeSFTP:
    """最小 SFTP 替身: 覆盖端点用到的 normalize/stat/mkdir/file/remove。"""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files: dict[str, bytes] = dict(files or {})
        self.dirs: set[str] = set()
        self.removed: list[str] = []

    def normalize(self, path: str) -> str:
        if path in ("~", "."):
            return "/home/u"
        if path.startswith("~/"):
            return "/home/u/" + path[2:]
        return path

    def stat(self, path: str) -> _FakeAttr:
        if path in self.files:
            return _FakeAttr(len(self.files[path]))
        if path in self.dirs:
            return _FakeAttr(0)
        raise FileNotFoundError(path)

    def mkdir(self, path: str) -> None:
        self.dirs.add(path)

    def file(self, path: str, mode: str) -> _FakeRemoteFile:
        return _FakeRemoteFile(self, path, mode)

    def remove(self, path: str) -> None:
        self.removed.append(path)
        self.files.pop(path, None)


class _FakeConfig:
    host = "cluster.example"


@pytest.fixture
def remote(monkeypatch):
    """把 HPCClient / _build_cfg 换成内存替身, 返回可检查的远端 SFTP 状态。"""
    sftp = _FakeSFTP()

    class _FakeClient:
        def __init__(self, cfg):
            self.config = cfg
            self._sftp = sftp

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def _ensure_connected(self):
            return None

    monkeypatch.setattr(transfer, "HPCClient", _FakeClient)
    monkeypatch.setattr(transfer, "_build_cfg", lambda body: (_FakeConfig(), None))
    return sftp


def test_web_upload_writes_remote_file(client, remote):
    r = client.post(
        "/v1/transfer/web/upload",
        files={"file": ("data.bin", b"\x00\x01\x02\x03")},
        data={"credential_id": "abc123", "remote_dir": "/home/u/files"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["size"] == 4
    assert body["remote_path"] == "/home/u/files/data.bin"
    assert remote.files["/home/u/files/data.bin"] == b"\x00\x01\x02\x03"


def test_web_upload_expands_home_and_creates_parent_dir(client, remote):
    r = client.post(
        "/v1/transfer/web/upload",
        files={"file": ("a.txt", b"x")},
        data={"credential_id": "abc123", "remote_dir": "~/jobs/new"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["remote_path"] == "/home/u/jobs/new/a.txt"
    assert "/home/u/jobs/new" in remote.dirs


def test_web_upload_strips_path_from_filename(client, remote):
    """文件名只取 basename: `../../evil.sh` 不能逃出目标目录."""
    r = client.post(
        "/v1/transfer/web/upload",
        files={"file": ("../../evil.sh", b"rm -rf /")},
        data={"credential_id": "abc123", "remote_dir": "/home/u/files"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["remote_path"] == "/home/u/files/evil.sh"
    assert "/home/u/files/evil.sh" in remote.files


def test_web_upload_rejects_oversize(client, remote, monkeypatch):
    monkeypatch.setattr(transfer, "_WEB_UPLOAD_MAX_BYTES", 4)
    r = client.post(
        "/v1/transfer/web/upload",
        files={"file": ("big.bin", b"0123456789")},
        data={"credential_id": "abc123", "remote_dir": "/home/u/files"},
    )
    assert r.status_code == 413
    # 超限时不写远端, 不留垃圾
    assert remote.files == {}


def test_web_upload_rejects_traversal_dir(client, remote):
    r = client.post(
        "/v1/transfer/web/upload",
        files={"file": ("a.txt", b"x")},
        data={"credential_id": "abc123", "remote_dir": "/home/u/../etc"},
    )
    assert r.status_code == 400
    assert remote.files == {}


def test_web_upload_requires_credential_id(client, remote):
    r = client.post(
        "/v1/transfer/web/upload",
        files={"file": ("a.txt", b"x")},
        data={"remote_dir": "/home/u/files"},
    )
    assert r.status_code == 422


def test_web_download_streams_remote_file(client, remote):
    remote.files["/home/u/files/report.txt"] = b"hello remote"
    r = client.get(
        "/v1/transfer/web/download",
        params={"credential_id": "abc123", "path": "/home/u/files/report.txt"},
    )
    assert r.status_code == 200, r.text
    assert r.content == b"hello remote"
    assert r.headers["content-type"] == "application/octet-stream"
    assert "report.txt" in r.headers["content-disposition"]


def test_web_download_missing_returns_404(client, remote):
    r = client.get(
        "/v1/transfer/web/download",
        params={"credential_id": "abc123", "path": "/home/u/files/nope.txt"},
    )
    assert r.status_code == 404


def test_web_download_rejects_traversal_path(client, remote):
    remote.files["/etc/passwd"] = b"root:x:0:0"
    r = client.get(
        "/v1/transfer/web/download",
        params={"credential_id": "abc123", "path": "/home/u/../etc/passwd"},
    )
    assert r.status_code == 400


def test_sanitize_header_filename(remote):
    assert transfer._sanitize_header_filename('a"b\\c\nd.txt') == "abcd.txt"
    assert transfer._sanitize_header_filename("") == "download"
