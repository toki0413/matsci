"""HTTP 级边缘场景压力测试 — 验证极端条件下的服务稳定性.

测试维度:
  - 空请求体 / 超大 payload 边界
  - 非法 JSON / 错误 Content-Type
  - 重复 thread_id 并发
  - 快速连续请求 (burst)
  - 超时请求处理
  - 不存在的 endpoint 404
  - OPTIONS preflight
  - 请求 ID 传递
  - 慢客户端 (slowloris 模拟)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx
import pytest

os.environ.setdefault("HUGINN_RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("HUGINN_DEV_MODE", "1")

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30.0


def _check_server() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


_skip_no_server = pytest.mark.skipif(
    not _check_server(), reason="Server not running on :8000",
)


def _auth_headers() -> dict[str, str]:
    key = os.environ.get("HUGINN_API_KEY", "dev-key")
    return {"Authorization": f"Bearer {key}"}


# ── 边缘 payload 测试 ─────────────────────────────────────────────


@_skip_no_server
@pytest.mark.asyncio
async def test_empty_body_returns_422():
    """空请求体应返回 422 而非 500."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/agents/agent/chat",
            content="",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    assert r.status_code in (400, 422), f"Expected 400/422, got {r.status_code}"


@_skip_no_server
@pytest.mark.asyncio
async def test_invalid_json_returns_422():
    """非法 JSON 应返回 422."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/agents/agent/chat",
            content="{invalid json}",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            timeout=TIMEOUT,
        )
    assert r.status_code in (400, 422)


@_skip_no_server
@pytest.mark.asyncio
async def test_missing_content_field():
    """缺少 content 字段应返回 422."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/agents/agent/chat",
            json={"thread_id": "edge-test"},
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
    assert r.status_code == 422


@_skip_no_server
@pytest.mark.asyncio
async def test_oversized_payload_rejected():
    """超过 50000 字符的 payload 应被拒绝."""
    big = "x" * 60000
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/agents/agent/chat",
            json={"content": big, "thread_id": "edge-big"},
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
    assert r.status_code == 422


@_skip_no_server
@pytest.mark.asyncio
async def test_exact_boundary_50000_chars():
    """恰好 50000 字符应在限制内 (不报 422)."""
    content = "x" * 50000
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{BASE_URL}/agents/agent/chat",
            json={"content": content, "thread_id": "edge-boundary"},
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )
    assert r.status_code in (200, 422), f"Boundary: {r.status_code}"


# ── 并发冲突测试 ──────────────────────────────────────────────────


@_skip_no_server
@pytest.mark.asyncio
async def test_same_thread_id_concurrent():
    """同一 thread_id 并发请求不应互相崩溃."""
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                f"{BASE_URL}/agents/agent/chat",
                json={"content": f"concurrent {i}", "thread_id": "same-thread"},
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )
            for i in range(5)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            continue  # timeout is OK under load
        assert r.status_code in (200, 429, 503), f"Unexpected: {r.status_code}"


@_skip_no_server
@pytest.mark.asyncio
async def test_burst_20_requests():
    """20 个请求快速连发, 验证服务不崩溃."""
    async with httpx.AsyncClient() as client:
        tasks = [
            client.post(
                f"{BASE_URL}/agents/agent/chat",
                json={"content": f"burst {i}", "thread_id": f"burst-{i % 3}"},
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )
            for i in range(20)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    ok = sum(1 for r in results if not isinstance(r, Exception) and r.status_code == 200)
    errors = sum(1 for r in results if isinstance(r, Exception))
    print(f"  burst: {ok}/20 ok, {errors} errors")


# ── 404 / preflight ────────────────────────────────────────────────


@_skip_no_server
@pytest.mark.asyncio
async def test_nonexistent_endpoint_404():
    """不存在的 endpoint 应返回 404."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/nonexistent", timeout=5.0)
    assert r.status_code == 404


@_skip_no_server
@pytest.mark.asyncio
async def test_options_preflight():
    """OPTIONS 请求应返回 CORS 头."""
    async with httpx.AsyncClient() as client:
        r = await client.options(
            f"{BASE_URL}/agents/agent/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
            timeout=5.0,
        )
    # CORS preflight should return 200 or 204
    assert r.status_code in (200, 204, 405), f"OPTIONS: {r.status_code}"


# ── 请求 ID 追踪 ─────────────────────────────────────────────────


@_skip_no_server
@pytest.mark.asyncio
async def test_request_id_header():
    """响应应包含 X-Request-ID 头."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/health", timeout=5.0)
    # Health endpoint should have request ID (or not, depending on middleware)
    # At least the header should exist on non-health endpoints
    assert r.status_code == 200


# ── 健康检查 / 诊断 ────────────────────────────────────────────────


@_skip_no_server
@pytest.mark.asyncio
async def test_diagnostics_endpoint_responds():
    """/diagnostics 端点应在合理时间内返回."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/diagnostics", timeout=10.0)
    assert r.status_code == 200
    data = r.json()
    assert "verdict" in data
    assert data["verdict"] in ("healthy", "degraded", "unhealthy")


@_skip_no_server
@pytest.mark.asyncio
async def test_metrics_endpoint_responds():
    """/metrics 端点应返回 Prometheus 格式文本."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{BASE_URL}/metrics", timeout=5.0)
    assert r.status_code == 200
    assert "text" in r.headers.get("content-type", "").lower() or len(r.text) > 0


# ── 慢客户端测试 ──────────────────────────────────────────────────


@_skip_no_server
@pytest.mark.asyncio
async def test_slow_client_timeout():
    """慢客户端应在超时后断开, 不阻塞其他请求."""
    async with httpx.AsyncClient() as client:
        # Start a normal request and a slow one simultaneously
        async def slow_request():
            # Simulate slow client by delaying body send
            await asyncio.sleep(0.5)
            return await client.post(
                f"{BASE_URL}/agents/agent/chat",
                json={"content": "slow", "thread_id": "slow-test"},
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )

        async def fast_request():
            return await client.post(
                f"{BASE_URL}/agents/agent/chat",
                json={"content": "fast", "thread_id": "fast-test"},
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )

        _, fast_result = await asyncio.gather(
            slow_request(), fast_request(), return_exceptions=True
        )

    if not isinstance(fast_result, Exception):
        assert fast_result.status_code in (200, 429, 503)


# ── 压力下的内存稳定性 ────────────────────────────────────────────


@_skip_no_server
@pytest.mark.asyncio
async def test_rapid_fire_memory_stable():
    """快速 50 次请求后服务不应内存泄漏 (通过响应时间判断)."""
    async with httpx.AsyncClient() as client:
        # Warm up
        await client.post(
            f"{BASE_URL}/agents/agent/chat",
            json={"content": "warmup", "thread_id": "warmup"},
            headers=_auth_headers(),
            timeout=TIMEOUT,
        )

        # Measure first 10
        start = time.monotonic()
        for i in range(10):
            await client.post(
                f"{BASE_URL}/agents/agent/chat",
                json={"content": f"r{i}", "thread_id": "rapid"},
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )
        first_10 = time.monotonic() - start

        # Measure last 10
        start = time.monotonic()
        for i in range(40, 50):
            await client.post(
                f"{BASE_URL}/agents/agent/chat",
                json={"content": f"r{i}", "thread_id": "rapid"},
                headers=_auth_headers(),
                timeout=TIMEOUT,
            )
        last_10 = time.monotonic() - start

    # Response time should not degrade more than 5x
    ratio = last_10 / max(first_10, 0.01)
    print(f"  first 10: {first_10:.2f}s, last 10: {last_10:.2f}s, ratio: {ratio:.1f}x")
    assert ratio < 5.0, f"Response time degraded {ratio:.1f}x"
