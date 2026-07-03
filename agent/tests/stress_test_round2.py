"""Round 2 stress tests: API throughput, WebSocket stability, tool registry concurrency.

Runs against http://localhost:8000. Reports per-endpoint p50/p95/p99 latency,
success rate, and total throughput.
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HUGINN_CACHE_DIR", str(Path(__file__).parent.parent / ".test_cache"))

BASE = "http://localhost:8000"
N_CONCURRENT = 50
N_REQUESTS = 500


def _post(path: str, payload: dict | None = None) -> tuple[int, float, bool]:
    url = f"{BASE}{path}"
    data = json.dumps(payload or {}).encode() if payload else None
    headers = {"Content-Type": "application/json"} if data else {}
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
        resp = urllib.request.urlopen(req, timeout=15)
        elapsed = time.perf_counter() - t0
        return resp.status, elapsed, True
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return 0, elapsed, False


def _get(path: str, payload: dict | None = None) -> tuple[int, float, bool]:
    url = f"{BASE}{path}"
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(url, method="GET")
        resp = urllib.request.urlopen(req, timeout=15)
        elapsed = time.perf_counter() - t0
        return resp.status, elapsed, True
    except Exception:
        elapsed = time.perf_counter() - t0
        return 0, elapsed, False


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def run_endpoint(name: str, method: str, path: str, payload: dict | None = None) -> dict:
    latencies: list[float] = []
    successes = 0
    errors = 0
    func = _post if method == "POST" else _get

    with ThreadPoolExecutor(max_workers=N_CONCURRENT) as pool:
        futures = [pool.submit(func, path, payload) for _ in range(N_REQUESTS)]
        for f in as_completed(futures):
            status, elapsed, ok = f.result()
            latencies.append(elapsed)
            if ok and 200 <= status < 400:
                successes += 1
            else:
                errors += 1

    ok_lat = [l for l, s, ok in zip(latencies, [True]*len(latencies), [True]*len(latencies))]
    return {
        "endpoint": name,
        "method": method,
        "path": path,
        "total": N_REQUESTS,
        "success": successes,
        "errors": errors,
        "success_rate": f"{successes / N_REQUESTS * 100:.1f}%",
        "p50_ms": f"{percentile(latencies, 50) * 1000:.1f}",
        "p95_ms": f"{percentile(latencies, 95) * 1000:.1f}",
        "p99_ms": f"{percentile(latencies, 99) * 1000:.1f}",
        "throughput_rps": f"{N_REQUESTS / sum(latencies):.1f}",
    }


async def ws_stability_test() -> dict:
    """Test WebSocket connection stability with multiple concurrent connections."""
    try:
        import websockets
    except ImportError:
        return {"test": "websocket_stability", "status": "skipped", "reason": "websockets not installed"}

    ws_url = "ws://localhost:8000/ws"
    n_connections = 20
    messages_per_conn = 10
    results = {"connected": 0, "messages_sent": 0, "messages_received": 0, "errors": 0}

    async def single_conn(idx: int) -> None:
        try:
            async with websockets.connect(ws_url, close_timeout=5) as ws:
                results["connected"] += 1
                for i in range(messages_per_conn):
                    msg = json.dumps({"type": "ping", "session_id": f"stress-{idx}", "data": f"msg-{i}"})
                    await ws.send(msg)
                    results["messages_sent"] += 1
                    try:
                        resp = await asyncio.wait_for(ws.recv(), timeout=5)
                        results["messages_received"] += 1
                    except asyncio.TimeoutError:
                        results["errors"] += 1
        except Exception:
            results["errors"] += 1

    await asyncio.gather(*[single_conn(i) for i in range(n_connections)])
    return {
        "test": "websocket_stability",
        "connections": n_connections,
        "connected": results["connected"],
        "messages_sent": results["messages_sent"],
        "messages_received": results["messages_received"],
        "errors": results["errors"],
        "status": "pass" if results["errors"] == 0 else "partial",
    }


def tool_registry_concurrency() -> dict:
    """Test tool registry thread-safety under concurrent access."""
    from huginn.tools.registry import ToolRegistry

    reg = ToolRegistry()
    errors = []
    n_threads = 20
    n_iterations = 50

    def worker(tid: int) -> None:
        try:
            for i in range(n_iterations):
                tools = reg.list_tools()
                schemas = reg.get_all_schemas()
                _ = len(tools)
                _ = len(schemas) if schemas else 0
        except Exception as e:
            errors.append(f"thread-{tid}: {e}")

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(worker, i) for i in range(n_threads)]
        for f in as_completed(futures):
            f.result()

    return {
        "test": "tool_registry_concurrency",
        "threads": n_threads,
        "iterations_per_thread": n_iterations,
        "total_operations": n_threads * n_iterations,
        "errors": len(errors),
        "status": "pass" if not errors else "fail",
    }


def main() -> None:
    print("=" * 70)
    print("STRESS TEST ROUND 2")
    print("=" * 70)

    # Health check
    status, _, ok = _get("/health")
    if not ok:
        print("Backend is not running on :8000. Aborting API tests.")
        print("Running tool-level tests only...")
    else:
        print(f"Backend health: {status}\n")

        # API endpoint stress tests
        # Only /health and /health/guidance are public (no auth required).
        # Other endpoints (/tools, /config, /sessions, etc.) need HUGINN_API_KEY
        # or HUGINN_DEV_MODE=1 — tested separately in authenticated stress tests.
        endpoints = [
            ("health", "GET", "/health"),
            ("health_guidance", "GET", "/health/guidance"),
        ]

        print(f"--- API Stress: {N_REQUESTS} requests x {N_CONCURRENT} concurrent ---\n")
        results = []
        for name, method, path in endpoints:
            print(f"  Testing {name} ({method} {path})...", end="", flush=True)
            r = run_endpoint(name, method, path)
            results.append(r)
            print(f" {r['success_rate']} p95={r['p95_ms']}ms")
            if int(r["errors"]) > 0:
                print(f"    ⚠ {r['errors']} errors")

        print(f"\n{'Endpoint':<20} {'Success':<10} {'p50':<10} {'p95':<10} {'p99':<10} {'RPS':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['endpoint']:<20} {r['success_rate']:<10} {r['p50_ms']:<10} {r['p95_ms']:<10} {r['p99_ms']:<10} {r['throughput_rps']:<10}")

    # WebSocket stability
    print("\n--- WebSocket Stability ---")
    ws_result = asyncio.run(ws_stability_test())
    print(f"  {ws_result}")

    # Tool registry concurrency
    print("\n--- Tool Registry Concurrency ---")
    reg_result = tool_registry_concurrency()
    print(f"  {reg_result}")

    print("\n" + "=" * 70)
    print("STRESS TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
