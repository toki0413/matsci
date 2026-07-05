"""Async stress tests for the Huginn HTTP + WebSocket surface.

Run with: STRESS_TEST_URL=http://localhost:8999 pytest -q test_http_stress.py

These hit a *running* server (started by CI or by you). If nothing is
listening on /health/live the whole module skips rather than failing.
"""
from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

BASE_URL = os.environ.get("STRESS_TEST_URL", "http://localhost:8999").rstrip("/")
# The server reads X-HUGINN-API-KEY. Original ask was X-API-Key; flip this one
# constant if an alias gets added on the server side.
API_KEY_HEADER = "X-HUGINN-API-KEY"
API_KEY = os.environ.get("HUGINN_API_KEY", "test-key-12345")
WS_URL = BASE_URL.replace("http://", "ws://", 1).replace("https://", "wss://", 1) + "/ws/agent"

pytestmark = pytest.mark.asyncio


def _server_up() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get(f"{BASE_URL}/health/live").status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _require_server():
    if not _server_up():
        pytest.skip(f"no server at {BASE_URL} (set STRESS_TEST_URL)")
    yield


async def test_health_live_concurrent_50():
    async with httpx.AsyncClient(timeout=10.0) as c:
        results = await asyncio.gather(
            *(c.get(f"{BASE_URL}/health/live") for _ in range(50))
        )
    statuses = [r.status_code for r in results]
    assert all(s == 200 for s in statuses), f"non-200 in batch: {statuses}"


async def test_tools_concurrent_20():
    headers = {API_KEY_HEADER: API_KEY}
    async with httpx.AsyncClient(timeout=10.0) as c:
        results = await asyncio.gather(
            *(c.get(f"{BASE_URL}/tools", headers=headers) for _ in range(20))
        )
    statuses = [r.status_code for r in results]
    assert all(s == 200 for s in statuses), f"non-200 in batch: {statuses}"


async def test_rate_limit_burst():
    # Fire a burst past the default 120 req/min ceiling and check the server
    # stays sane: either it throttles (429) or the limiter is disabled (all
    # 200). Anything else (5xx, 401) is a real failure.
    headers = {API_KEY_HEADER: API_KEY}
    async with httpx.AsyncClient(timeout=10.0) as c:
        results = await asyncio.gather(
            *(c.get(f"{BASE_URL}/tools", headers=headers) for _ in range(150))
        )
    statuses = [r.status_code for r in results]
    ok = sum(1 for s in statuses if s == 200)
    throttled = sum(1 for s in statuses if s == 429)
    unexpected = [s for s in statuses if s not in (200, 429)]
    assert not unexpected, f"unexpected statuses under burst: {unexpected}"
    print(f"burst result: {ok} ok, {throttled} throttled")


def _ws_connect(uri: str, headers: dict[str, str]):
    # websockets renamed extra_headers -> additional_headers in v11; accept
    # whichever the installed version speaks so this stays portable.
    import websockets

    try:
        return websockets.connect(uri, additional_headers=headers)
    except TypeError:
        return websockets.connect(uri, extra_headers=headers)


async def test_websocket_stability_5x10s():
    headers = {API_KEY_HEADER: API_KEY}

    async def _hold_one() -> bool:
        async with _ws_connect(WS_URL, headers) as conn:
            # Round-trip a ping to prove the connection is live, then just
            # hold it open for the hold window.
            await conn.send(json.dumps({"type": "ping"}))
            raw = await asyncio.wait_for(conn.recv(), timeout=5.0)
            got_pong = json.loads(raw).get("type") == "pong"
            await asyncio.sleep(10.0)
            return got_pong

    results = await asyncio.gather(
        *(_hold_one() for _ in range(5)), return_exceptions=True
    )
    for r in results:
        assert not isinstance(r, Exception), f"WS connection failed: {r}"
    assert all(results), "at least one connection did not pong"


if __name__ == "__main__":
    # Standalone smoke: 20 concurrent health pings, no pytest needed.
    if not _server_up():
        raise SystemExit(f"no server at {BASE_URL} (set STRESS_TEST_URL)")

    async def _smoke() -> None:
        async with httpx.AsyncClient(timeout=10.0) as c:
            rs = await asyncio.gather(
                *(c.get(f"{BASE_URL}/health/live") for _ in range(20))
            )
        print("smoke statuses:", sorted({r.status_code for r in rs}))

    asyncio.run(_smoke())
