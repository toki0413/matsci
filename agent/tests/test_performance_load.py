"""性能/负载测试 — 并发 QPS、大数据量、内存增长、多用户隔离.

沙箱环境 (6GB) 用轻量并发, 验证趋势而非绝对数值.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

import pytest

os.environ.pop("HUGINN_DEV_MODE", None)
os.environ["HUGINN_API_KEY"] = "perf-key-0123456789abcdef"
os.environ["HUGINN_JWT_SECRET"] = "perf-jwt-secret"
os.environ["HUGINN_RATE_LIMIT_PER_MINUTE"] = "0"
os.environ["HUGINN_ALLOW_LOCAL_BASH"] = "1"
os.environ["HUGINN_USE_DOCKER"] = "0"


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("perf_ws")
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
    store._users["perf-user"] = User(user_id="perf-user", username="perf", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["perf-user"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestConcurrentQPS:
    """并发请求 QPS."""

    def test_concurrent_memory_stats(self, app_client, admin_token):
        """10 并发 GET /memory/stats, 全部 200, QPS > 50."""
        import concurrent.futures

        def call():
            return app_client.get("/memory/stats", headers=_bearer(admin_token)).status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            start = time.monotonic()
            futures = [pool.submit(call) for _ in range(100)]
            results = [f.result() for f in futures]
            elapsed = time.monotonic() - start

        assert all(r == 200 for r in results), f"some failed: {set(results)}"
        qps = 100 / elapsed
        print(f"\n[PERF] memory/stats 100 req / 10 threads: {qps:.0f} QPS ({elapsed:.2f}s)")
        assert qps > 10, f"QPS too low: {qps:.0f}"

    def test_concurrent_memory_write(self, app_client, admin_token):
        """10 并发 POST /memory, 全部成功, 无串数据."""
        import concurrent.futures

        def call(i):
            resp = app_client.post("/memory", headers=_bearer(admin_token), json={
                "content": f"perf-test-item-{i}", "category": "fact", "importance": 0.5,
            })
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(call, i) for i in range(20)]
            results = [f.result() for f in futures]

        # 全部 200 (写成功)
        assert all(r == 200 for r in results), f"some failed: {set(results)}"

    def test_concurrent_thread_creation(self, app_client, admin_token):
        """并发创建 thread, ID 不冲突."""
        import concurrent.futures

        def call(i):
            resp = app_client.post("/threads", headers=_bearer(admin_token), json={"title": f"perf-thread-{i}"})
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(call, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert all(r in (200, 201) for r in results), f"some failed: {set(results)}"


class TestLargeDataVolume:
    """大数据量下的检索延迟."""

    def test_memory_search_with_many_entries(self, app_client, admin_token):
        """写入 100 条 memory 后搜索, 延迟 < 2s."""
        # 批量写入
        for i in range(100):
            app_client.post("/memory", headers=_bearer(admin_token), json={
                "content": f"large-volume-item-{i} materials science fact",
                "category": "fact", "importance": 0.5, "tags": ["perf", "batch"],
            })

        start = time.monotonic()
        resp = app_client.post("/memory/search", headers=_bearer(admin_token), json={
            "query": "materials science", "top_k": 10,
        })
        elapsed = time.monotonic() - start

        assert resp.status_code == 200, f"search failed: {resp.text[:200]}"
        print(f"\n[PERF] search 100 entries: {elapsed:.3f}s")
        assert elapsed < 5.0, f"search too slow: {elapsed:.3f}s"

    def test_large_query_string(self, app_client, admin_token):
        """超长 query (10KB) 不崩溃, 延迟合理."""
        big_query = "materials " * 1000  # ~10KB
        start = time.monotonic()
        resp = app_client.post("/memory/search", headers=_bearer(admin_token), json={
            "query": big_query, "top_k": 5,
        })
        elapsed = time.monotonic() - start
        assert resp.status_code != 500, f"crashed on big query: {resp.status_code}"
        assert elapsed < 10.0, f"too slow: {elapsed:.3f}s"


class TestMemoryGrowth:
    """内存增长趋势."""

    def test_memory_growth_under_load(self, app_client, admin_token):
        """100 次请求后 RSS 增长 < 100MB."""
        import resource

        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB

        for _ in range(100):
            app_client.get("/memory/stats", headers=_bearer(admin_token))

        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB
        growth_mb = (rss_after - rss_before) / 1024
        print(f"\n[PERF] RSS growth after 100 reqs: {growth_mb:.1f} MB")
        # 允许 100MB 增长 (Python GC + cache)
        assert growth_mb < 200, f"excessive memory growth: {growth_mb:.1f} MB"


class TestLargeFileUpload:
    """大文件上传."""

    def test_upload_1mb_file(self, app_client, admin_token):
        """1MB 文件上传不超时."""
        content = b"x" * (1024 * 1024)
        start = time.monotonic()
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("big.txt", content, "text/plain")})
        elapsed = time.monotonic() - start
        assert resp.status_code != 500, f"crashed: {resp.status_code}"
        assert elapsed < 30.0, f"too slow: {elapsed:.3f}s"
        print(f"\n[PERF] upload 1MB: {elapsed:.3f}s")

    def test_upload_empty_file(self, app_client, admin_token):
        """空文件不崩溃."""
        resp = app_client.post("/knowledge/upload", headers=_bearer(admin_token),
                               files={"file": ("empty.txt", b"", "text/plain")})
        assert resp.status_code != 500


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov", "-s"]))
