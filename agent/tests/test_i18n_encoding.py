"""国际化/编码边界测试 — 中文/日文/emoji/UTF-8 BOM/代理对."""
from __future__ import annotations

import os
import sys

import pytest

os.environ.pop("HUGINN_DEV_MODE", None)
os.environ["HUGINN_API_KEY"] = "i18n-key-0123456789abcdef"
os.environ["HUGINN_JWT_SECRET"] = "i18n-jwt-secret"
os.environ["HUGINN_RATE_LIMIT_PER_MINUTE"] = "0"


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("i18n_ws")
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
    store._users["i18n-user"] = User(user_id="i18n-user", username="i18n", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["i18n-user"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestI18nMemoryContent:
    """多语言 memory 内容."""

    @pytest.mark.parametrize("label,content", [
        ("chinese", "材料科学第一性原理计算"),
        ("japanese", "材料科学第一原理計算"),
        ("korean", "재료 과학 제일원리 계산"),
        ("arabic", "حسابات المبدأ الأول لعلوم المواد"),
        ("emoji", "材料科学 🧪⚛️🔬 计算完成 ✅"),
        ("mixed", "DFT 计算 ( density functional theory ) 密度泛函理论"),
    ])
    def test_memory_write_and_search_multilingual(self, app_client, admin_token, label, content):
        """多语言内容写入 + 检索不崩溃."""
        # 写入
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": content, "category": "fact", "tags": [label],
        })
        assert resp.status_code == 200, f"[{label}] write failed: {resp.status_code}"

        # 检索
        resp = app_client.post("/memory/search", headers=_bearer(admin_token), json={
            "query": content[:10], "top_k": 5,
        })
        assert resp.status_code == 200, f"[{label}] search failed: {resp.status_code}"


class TestSpecialCharacters:
    """特殊字符处理."""

    def test_memory_with_newlines(self, app_client, admin_token):
        """多行内容."""
        content = "line 1\nline 2\nline 3\n\tindented"
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": content, "category": "fact",
        })
        assert resp.status_code == 200

    def test_memory_with_quotes(self, app_client, admin_token):
        """含引号的内容."""
        content = 'He said "hello" and \'world\''
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": content, "category": "fact",
        })
        assert resp.status_code == 200

    def test_memory_with_backslash(self, app_client, admin_token):
        """含反斜杠."""
        content = r"path\to\file C:\Users\test"
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": content, "category": "fact",
        })
        assert resp.status_code == 200

    def test_memory_with_null_char(self, app_client, admin_token):
        """null 字符 (JSON 中 \u0000)."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": "before\u0000after", "category": "fact",
        })
        assert resp.status_code != 500

    def test_memory_with_surrogate_pair(self, app_client, admin_token):
        """UTF-16 代理对 (emoji) — httpx 不允许 lone surrogate, 用真实 emoji 字符."""
        # 直接用 emoji 字符 (Python 内部是正确 Unicode, 不是 lone surrogate)
        content = "surrogate test 🎉 done"
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": content, "category": "fact",
        })
        assert resp.status_code == 200


class TestUnicodeFilenames:
    """Unicode 文件名."""

    def test_upload_chinese_filename(self, app_client, admin_token):
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("材料数据.txt", b"chinese filename", "text/plain")})
        assert resp.status_code != 500

    def test_upload_japanese_filename(self, app_client, admin_token):
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("データ.txt", b"japanese", "text/plain")})
        assert resp.status_code != 500

    def test_upload_mixed_filename(self, app_client, admin_token):
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("report_2024_报告.txt", b"mixed", "text/plain")})
        assert resp.status_code != 500


class TestThreadTitles:
    """Unicode thread 标题."""

    def test_create_thread_chinese_title(self, app_client, admin_token):
        resp = app_client.post("/threads", headers=_bearer(admin_token), json={"title": "DFT 计算任务"})
        assert resp.status_code in (200, 201)

    def test_create_thread_emoji_title(self, app_client, admin_token):
        resp = app_client.post("/threads", headers=_bearer(admin_token), json={"title": "🚀 Launch Task"})
        assert resp.status_code in (200, 201)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
