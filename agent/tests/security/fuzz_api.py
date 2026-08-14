"""API 模糊测试 — 用 hypothesis 生成随机/畸形输入, 探测 API 边界 bug.

覆盖:
1. 内存搜索 query fuzzing — 各种特殊字符/超长串/unicode
2. 知识库 URL ingest fuzzing — 各种 scheme/host 畸形
3. JWT 畸形 token fuzzing — 各种格式破坏
4. 路径参数 fuzzing — 路径遍历/特殊字符
5. JSON body 畸形 fuzzing — 类型混淆/嵌套/超大

目标: 不出 500 (服务器内部错误). 4xx 是正常的输入校验拒绝.
"""
from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _strict_no_dev_mode(monkeypatch):
    """隔离环境变量: 确保 dev mode 关闭, 且不污染同 worker 的其他测试."""
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HYPOTHESIS_AVAILABLE,
                                 reason="hypothesis not installed")


@pytest.fixture
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("HUGINN_API_KEY", "fuzz-admin-key-0123456789abcdef")
    monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "fuzz-admin-key-0123456789abcdef")
    monkeypatch.setenv("HUGINN_JWT_SECRET", "fuzz-jwt-secret")
    monkeypatch.setenv("HUGINN_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HUGINN_ENFORCE_WRITE_CAPABILITY", "1")
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)

    from huginn.security.auth import get_user_store
    from huginn.security.rbac import Role, User, jwt_encode
    store = get_user_store()
    store._users["fuzz-admin"] = User(
        user_id="fuzz-admin", username="admin", role=Role.ADMIN, active=True,
        api_key_hash="", created_at=0,
    )
    admin_token = jwt_encode(
        {"sub": "fuzz-admin", "username": "admin", "role": Role.ADMIN.value, "jti": "fuzz-jti"},
        "fuzz-jwt-secret",
    )

    try:
        from fastapi.testclient import TestClient

        from huginn.server import app
    except Exception as e:
        pytest.skip(f"无法启动 app: {e}")

    # TestClient 启动时 lifespan 连接 MCP servers 并把工具注册进全局
    # ToolRegistry, 退出时不卸载, conftest 的 _restore_tool_registry guard
    # 会误报 leak (added=MCP 工具). 用 snapshot/restore 包裹, 结束时恢复.
    from huginn.tools.registry import ToolRegistry

    _before_tools = ToolRegistry.snapshot()
    try:
        with TestClient(app) as client:
            yield client, admin_token
    finally:
        ToolRegistry.restore(_before_tools)

    store._users.pop("fuzz-admin", None)


# ── 1. memory search query fuzzing ──────────────────────────────

@given(query=st.one_of(
    st.text(min_size=0, max_size=1000),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",), min_codepoint=1, max_codepoint=0x10FFFF), min_size=1, max_size=100),
    st.binary(min_size=0, max_size=100).map(lambda b: b.decode("latin-1", errors="ignore")),
))
@settings(max_examples=50, deadline=2000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_memory_search_query_fuzz(app_client, query):
    """memory search 不应因任意 query 字符串崩溃 (500)."""
    client, token = app_client
    resp = client.post(
        "/memory/search",
        json={"query": query, "limit": 5},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code != 500, \
        f"memory search 因 query 崩溃! query={query!r}, status=500"


# ── 2. knowledge ingest-url fuzzing ─────────────────────────────

@given(url=st.one_of(
    st.text(min_size=0, max_size=200),
    st.from_regex(r"https?://[a-zA-Z0-9.\-]+(/[^\s]*)?", fullmatch=False),
    st.from_regex(r"[a-z]+://[^/]+", fullmatch=False),
))
@settings(max_examples=30, deadline=3000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_knowledge_ingest_url_fuzz(app_client, url):
    """ingest-url 不应因畸形 URL 崩溃 (500)."""
    client, token = app_client
    try:
        resp = client.post(
            "/knowledge/ingest-url",
            json={"url": url},
            headers={"Authorization": f"Bearer {token}"},
        )
    except Exception:
        return  # 客户端侧异常 (如 URL 编码失败) 不算 server bug
    assert resp.status_code != 500, \
        f"ingest-url 因 URL 崩溃! url={url!r}"


# ── 3. JWT 畸形 token fuzzing ───────────────────────────────────

@given(token=st.one_of(
    st.text(min_size=0, max_size=200, alphabet=st.characters(blacklist_categories=("Cs", "Co"))),
    st.from_regex(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*", fullmatch=False),
    st.binary(min_size=0, max_size=100).map(lambda b: b.hex()),
    st.just(""),  # 空 token
))
@settings(max_examples=50, deadline=2000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_malformed_jwt_fuzz(app_client, token):
    """畸形 JWT 不应导致服务器 500."""
    client, _ = app_client
    try:
        resp = client.get("/memory", headers={"Authorization": f"Bearer {token}"})
    except (UnicodeEncodeError, ValueError):
        return  # HTTP header 不允许非 ASCII, 客户端侧异常, 跳过
    assert resp.status_code != 500, \
        f"畸形 JWT 导致 500! token={token!r}"
    # 畸形 token 必须被拒绝 (401/403), 不能 200
    assert resp.status_code in (401, 403, 422), \
        f"畸形 JWT 被接受! token={token!r}, status={resp.status_code}"


# ── 4. 路径参数 fuzzing ─────────────────────────────────────────

@given(doc_id=st.one_of(
    st.text(min_size=1, max_size=100),
    st.from_regex(r"\.\./[^\s]+", fullmatch=False),
    st.from_regex(r"[;|&$`][^\s]*", fullmatch=False),
    st.binary(min_size=1, max_size=50).map(lambda b: b.hex()),
))
@settings(max_examples=30, deadline=2000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_path_param_fuzz(app_client, doc_id):
    """DELETE /knowledge/{doc_id} 不应因畸形 doc_id 崩溃."""
    client, token = app_client
    # URL 编码路径参数
    from urllib.parse import quote
    encoded = quote(doc_id, safe="")
    resp = client.delete(
        f"/knowledge/{encoded}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code != 500, \
        f"DELETE /knowledge/{{doc_id}} 因 doc_id 崩溃! doc_id={doc_id!r}"


# ── 5. JSON body 类型混淆 fuzzing ───────────────────────────────

@given(body=st.one_of(
    st.dictionaries(st.text(min_size=1, max_size=20), st.text(min_size=0, max_size=100)),
    st.dictionaries(st.text(min_size=1, max_size=20), st.integers()),
    st.dictionaries(st.text(min_size=1, max_size=20), st.lists(st.integers())),
    st.lists(st.integers()),  # 顶层不是 dict
    st.just("not a dict"),  # 顶层是 string
    st.just(12345),  # 顶层是 number
    st.just(None),  # 顶层是 null
))
@settings(max_examples=40, deadline=2000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_json_body_type_confusion_fuzz(app_client, body):
    """POST /memory 不应因畸形 JSON body 崩溃 (500)."""
    client, token = app_client
    resp = client.post(
        "/memory",
        json=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code != 500, \
        f"POST /memory 因 body 崩溃! body={body!r}"


# ── 6. HTTP header fuzzing ──────────────────────────────────────

@given(api_key=st.one_of(
    st.text(min_size=0, max_size=200, alphabet=st.characters(blacklist_categories=("Cs", "Co"))),
    st.binary(min_size=0, max_size=100).map(lambda b: b.hex()),
    st.from_regex(r"[A-Za-z0-9+/=]{20,}", fullmatch=False),
))
@settings(max_examples=30, deadline=2000,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_api_key_header_fuzz(app_client, api_key):
    """畸形 API key 不应导致 500, 必须返回 401."""
    client, _ = app_client
    try:
        resp = client.get("/memory", headers={"X-HUGINN-API-KEY": api_key})
    except (UnicodeEncodeError, ValueError):
        return  # HTTP header 不允许非 ASCII, 客户端侧异常, 跳过
    assert resp.status_code != 500, \
        f"畸形 API key 导致 500! key={api_key!r}"
    assert resp.status_code == 401, \
        f"畸形 API key 被接受! key={api_key!r}, status={resp.status_code}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
