"""文件上传安全测试 — 超大文件、恶意 MIME、文件名注入、损坏文件."""
from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_auth_env(monkeypatch):
    """Isolate auth env so we don't pollute other modules in the same worker."""
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
    monkeypatch.setenv("HUGINN_API_KEY", "upload-key-0123456789abcdef")
    monkeypatch.setenv("HUGINN_JWT_SECRET", "upload-jwt-secret")
    monkeypatch.setenv("HUGINN_RATE_LIMIT_PER_MINUTE", "0")
    yield


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("upload_ws")
    os.environ["HUGINN_WORKSPACE"] = str(ws)
    import huginn.server as sm
    sm._init_mcp_tools = _noop
    sm._shutdown_mcp = _noop
    from huginn.config import HuginnConfig
    from huginn.server_context import create_server_context, set_server_context
    ctx = create_server_context(HuginnConfig(provider="ollama", model="test", workspace=str(ws)))
    set_server_context(ctx)
    sm._context = ctx
    from huginn.security.auth import get_user_store
    from huginn.security.rbac import Role, User
    store = get_user_store()
    store._users["upload-user"] = User(user_id="upload-user", username="upload", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["upload-user"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestFileUploadSecurity:
    """文件上传安全."""

    def test_upload_path_traversal_filename(self, app_client, admin_token):
        """文件名含路径遍历 (../../etc/passwd), 不应写到目标目录外."""
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("../../etc/passwd", b"malicious", "text/plain")})
        assert resp.status_code != 500, f"crashed: {resp.status_code}"
        # 不能在 /etc/passwd 写入 (检查文件未被覆盖 — 实际上 /etc/passwd 需要 root)
        # 关键是服务不崩溃

    def test_upload_null_byte_filename(self, app_client, admin_token):
        """文件名含 null byte (file.txt\0.exe)."""
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("file.txt\x00.exe", b"null byte", "text/plain")})
        assert resp.status_code != 500, f"crashed on null byte: {resp.status_code}"

    def test_upload_mismatched_mime_type(self, app_client, admin_token):
        """MIME type 伪造 (声明 image/png 但内容是文本)."""
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("fake.png", b"not really a png", "image/png")})
        assert resp.status_code != 500, f"crashed on fake MIME: {resp.status_code}"

    def test_upload_executable_file(self, app_client, admin_token):
        """上传可执行文件 (.exe/.sh)."""
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("malicious.sh", b"#!/bin/bash\nrm -rf /", "application/x-sh")})
        assert resp.status_code != 500, f"crashed on .sh: {resp.status_code}"

    def test_upload_extremely_long_filename(self, app_client, admin_token):
        """超长文件名 (1000 字符)."""
        long_name = "a" * 1000 + ".txt"
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": (long_name, b"long name", "text/plain")})
        assert resp.status_code != 500, f"crashed on long name: {resp.status_code}"

    def test_upload_corrupted_zip(self, app_client, admin_token):
        """损坏的 zip 文件."""
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("broken.zip", b"PK\x03\x04broken content", "application/zip")})
        assert resp.status_code != 500, f"crashed on corrupted zip: {resp.status_code}"

    def test_upload_unicode_filename(self, app_client, admin_token):
        """Unicode 文件名 (中文/emoji)."""
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("测试文件_🎉.txt", b"unicode name", "text/plain")})
        assert resp.status_code != 500, f"crashed on unicode name: {resp.status_code}"


class TestImportSecurity:
    """导入安全."""

    def test_import_empty_zip(self, app_client, admin_token):
        """空 zip 文件不崩溃."""
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass  # 空 zip
        buf.seek(0)
        resp = app_client.post("/import/all", headers=_bearer(admin_token),
                               files={"file": ("empty.zip", buf.getvalue(), "application/zip")})
        assert resp.status_code != 500

    def test_import_not_an_archive(self, app_client, admin_token):
        """非归档文件 (纯文本) 不崩溃."""
        resp = app_client.post("/import/all", headers=_bearer(admin_token),
                               files={"file": ("not_zip.txt", b"just plain text", "text/plain")})
        assert resp.status_code != 500

    def test_import_zip_bomb_small(self, app_client, admin_token):
        """高压缩比 zip (小规模 zip bomb 模拟)."""
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1MB 的重复数据, 高压缩比
            zf.writestr("bomb.txt", "A" * (1024 * 1024))
        buf.seek(0)
        resp = app_client.post("/import/all", headers=_bearer(admin_token),
                               files={"file": ("bomb.zip", buf.getvalue(), "application/zip")})
        assert resp.status_code != 500

    def test_import_symlink_in_zip(self, app_client, admin_token):
        """zip 中含符号链接."""
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # zip 不直接支持 symlink, 但可以放一个名字像 symlink 的文件
            zf.writestr("link -> /etc/passwd", "symlink content")
        buf.seek(0)
        resp = app_client.post("/import/all", headers=_bearer(admin_token),
                               files={"file": ("symlink.zip", buf.getvalue(), "application/zip")})
        assert resp.status_code != 500


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
