"""性能回归基准套件 — 定义性能基线并检测退化.

覆盖 6 个维度:
  1. API 端点延迟基准 (p50/p95/p99, 30 次采样)
  2. WebSocket 延迟基准 (连接建立 / 首 token / 完整回合)
  3. 内存稳定性基准 (RSS + tracemalloc, 50 轮连续请求)
  4. 并发吞吐量基准 (HTTP 并发 + WS 并发连接)
  5. SQLite 性能基准 (ResearchLog 连续写入/读取)
  6. 基线对比机制 (BASELINE 字典 + 退化时输出当前值 vs 基线值)

运行方式:
    # 默认跳过, 不拖慢 CI
    python -m pytest tests/test_performance_regression.py

    # 显式启用性能测试
    HUGINN_RUN_PERFORMANCE=1 python -m pytest tests/test_performance_regression.py -v -s

    # 如果安装了 pytest-benchmark, 可以配合 --benchmark-only 使用
    HUGINN_RUN_PERFORMANCE=1 python -m pytest tests/test_performance_regression.py -v -s --benchmark-only

阈值说明:
  - 阈值经过合理设置, 不会太严导致 flaky, 也不会太松导致退化不被检测
  - 如果某个阈值在特定环境上 consistently 失败, 应该调整 BASELINE 而非放宽断言
"""

from __future__ import annotations

import os
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# server.py 间接依赖 mcp, 没装的话整个模块跳过
pytest.importorskip("mcp")

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from huginn.research_log import RecordType, ResearchLog
from huginn.server import app

# psutil 可选, 没装就跳过 RSS 相关测试
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── 性能测试开关 ──────────────────────────────────────
# 默认跳过, 需要 HUGINN_RUN_PERFORMANCE=1 才执行
_RUN_PERF = os.environ.get("HUGINN_RUN_PERFORMANCE", "").lower() in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    not _RUN_PERF,
    reason="性能测试默认跳过, 设置 HUGINN_RUN_PERFORMANCE=1 启用",
)

# ponytail: benchmark marker 没在 pyproject.toml 注册, 会有一条 harmless warning.
# 要消掉的话在 [tool.pytest.ini_options] 加 markers = ["benchmark: 性能基准测试"]

client = TestClient(app)

_HEADERS = {"X-HUGINN-API-KEY": "test-key"}
WS_PATH = "/ws/agent"


# ════════════════════════════════════════════════════════════════════
#  性能基线定义
# ════════════════════════════════════════════════════════════════════

BASELINE: dict[str, float] = {
    # API 端点延迟 p95 (毫秒)
    "api_health_live_p95_ms": 50,
    "api_tools_p95_ms": 200,
    "api_skills_p95_ms": 200,
    "api_memory_p95_ms": 200,
    "api_personas_p95_ms": 200,
    # WebSocket 延迟 (毫秒)
    "ws_connect_p95_ms": 1000,
    "ws_first_token_p95_ms": 500,
    "ws_full_turn_p95_ms": 2000,
    # 内存稳定性 (MB)
    "mem_rss_health_growth_mb": 10,
    "mem_rss_tools_growth_mb": 10,
    "mem_tracemalloc_growth_mb": 5,
    # 并发吞吐量 (秒)
    "concurrent_health_10_sec": 2,
    "concurrent_tools_20_sec": 5,
    "concurrent_ws_5_sec": 3,
    # SQLite 性能 (秒)
    "sqlite_write_100_sec": 5,
    "sqlite_read_100_sec": 2,
}


# ════════════════════════════════════════════════════════════════════
#  辅助函数
# ════════════════════════════════════════════════════════════════════


def _percentile(data: list[float], pct: int) -> float:
    """用 statistics.quantiles 计算百分位数.

    pct 取 1-99, 返回对应百分位的值.
    用 inclusive 方法: 小样本时不会外推到数据范围之外,
    避免 10 个样本算出 p95 > max 的荒谬结果.
    """
    if len(data) < 2:
        return max(data) if data else 0.0
    qs = statistics.quantiles(data, n=100, method="inclusive")
    return qs[pct - 1]


def _measure_latency_ms(
    fn, iterations: int = 30, warmup: int = 5
) -> dict[str, float | list[float]]:
    """对 fn 做 iterations 次调用, 返回 p50/p95/p99/min/max/mean (毫秒).

    warmup 轮不计时, 排除首次导入 / JIT / 连接池初始化的冷启动开销.
    """
    for _ in range(warmup):
        fn()
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "p50": _percentile(latencies, 50),
        "p95": _percentile(latencies, 95),
        "p99": _percentile(latencies, 99),
        "min": min(latencies),
        "max": max(latencies),
        "mean": statistics.mean(latencies),
        "raw": latencies,
    }


def _assert_baseline(name: str, current: float, unit: str = "ms") -> None:
    """断言当前值不超过基线, 失败时输出对比信息."""
    threshold = BASELINE[name]
    assert current <= threshold, (
        f"\n{'=' * 60}\n"
        f"  性能退化: {name}\n"
        f"  当前值: {current:.2f} {unit}\n"
        f"  基线值: {threshold} {unit}\n"
        f"  超出: {((current / threshold - 1) * 100):.1f}%\n"
        f"{'=' * 60}"
    )


def _print_latency_report(name: str, stats: dict[str, Any]) -> None:
    """打印延迟分布报告."""
    print(
        f"\n  [{name}] 延迟分布 ({len(stats['raw'])} 次采样):\n"
        f"    p50  = {stats['p50']:.2f} ms\n"
        f"    p95  = {stats['p95']:.2f} ms\n"
        f"    p99  = {stats['p99']:.2f} ms\n"
        f"    min  = {stats['min']:.2f} ms\n"
        f"    max  = {stats['max']:.2f} ms\n"
        f"    mean = {stats['mean']:.2f} ms"
    )


# ════════════════════════════════════════════════════════════════════
#  公共 fixture
# ════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """关掉 HTTP 限流, 性能测试会发很多请求."""
    from huginn.server import rate_limit_middleware

    mw_globals = rate_limit_middleware.__globals__
    saved = mw_globals.get("_RATE_LIMIT")
    mw_globals["_RATE_LIMIT"] = 0
    try:
        yield
    finally:
        if saved is not None:
            mw_globals["_RATE_LIMIT"] = saved


# ════════════════════════════════════════════════════════════════════
#  WS mock 基础设施 (复用 test_chaos_engineering 的思路)
# ════════════════════════════════════════════════════════════════════


class _MockModel:
    """够用的 chat model mock."""

    async def ainvoke(self, prompt: str, **kw: Any):
        class _R:
            content = "{}"

        return _R()


class _MockAgent:
    """按脚本回放 state 的 agent, 用于 WS 延迟测试."""

    def __init__(self, states: list[dict] | None = None) -> None:
        self._states = list(states or [])
        self.model = _MockModel()
        self.persona_name = "default"

    def set_persona(self, *a, **kw):
        pass

    async def chat(self, content: str, thread_id: str = "default"):
        for s in self._states:
            yield s


class _MockFactory:
    def __init__(self, agent: _MockAgent) -> None:
        self._agent = agent

    def create_lead(self, *a, **kw):
        return self._agent

    def create(self, *a, **kw):
        return self._agent

    def list_profiles(self):
        return []


@pytest.fixture
def ws_mock(tmp_path, monkeypatch):
    """给 WS 路由打 mock, 让延迟测试不走真实 LLM."""
    import huginn.routes.ws as ws_mod

    # 一条 AIMessage state, 模拟 LLM 回复
    agent = _MockAgent([{"messages": [AIMessage(content="mock response")]}])
    cfg = SimpleNamespace(
        workspace=str(tmp_path),
        rag_enabled=False,
        persona_auto_route=False,
        persona_auto_route_threshold=0.6,
        team_mode_enabled=False,
        max_concurrent_subagents=2,
    )
    ctx = SimpleNamespace(
        config=SimpleNamespace(workspace=str(tmp_path)), kb=None
    )

    async def _get_agent():
        return agent

    monkeypatch.setattr(ws_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(ws_mod, "get_agent", _get_agent)
    monkeypatch.setattr(ws_mod, "get_agent_factory", lambda: _MockFactory(agent))
    monkeypatch.setattr(ws_mod, "get_context", lambda: ctx)
    monkeypatch.setattr(ws_mod, "get_memory_manager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(ws_mod, "get_or_create_thread", lambda *a, **k: {"id": "t"})

    return agent


# ════════════════════════════════════════════════════════════════════
#  1. API 端点延迟基准
# ════════════════════════════════════════════════════════════════════


class TestAPILatency:
    """各 API 端点的 p50/p95/p99 延迟基准, 每个端点测 30 次."""

    _ENDPOINTS = [
        ("api_health_live_p95_ms", "/health/live"),
        ("api_tools_p95_ms", "/tools"),
        ("api_skills_p95_ms", "/skills"),
        ("api_memory_p95_ms", "/memory"),
        ("api_personas_p95_ms", "/personas"),
    ]

    @pytest.mark.benchmark
    @pytest.mark.parametrize("baseline_name,endpoint", _ENDPOINTS)
    def test_endpoint_p95(self, baseline_name: str, endpoint: str):
        """每个端点测 30 次取 p50/p95/p99, p95 不得超过基线."""

        def _hit():
            r = client.get(endpoint, headers=_HEADERS)
            assert r.status_code == 200, (
                f"{endpoint} returned {r.status_code}: {r.text[:200]}"
            )

        stats = _measure_latency_ms(_hit, iterations=30)
        _print_latency_report(f"GET {endpoint}", stats)
        _assert_baseline(baseline_name, stats["p95"])


# ════════════════════════════════════════════════════════════════════
#  2. WebSocket 延迟基准
# ════════════════════════════════════════════════════════════════════


class TestWebSocketLatency:
    """WebSocket 连接建立、首 token、完整对话回合延迟."""

    @pytest.mark.benchmark
    def test_ws_connection_time(self):
        """WS 连接建立时间 p95 < 1s (10 次采样)."""
        # 预热: 第一次 WS 连接要走 portal 初始化, 很慢, 排掉
        with client.websocket_connect(WS_PATH, headers=_HEADERS):
            pass
        latencies: list[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            with client.websocket_connect(WS_PATH, headers=_HEADERS):
                pass
            latencies.append((time.perf_counter() - t0) * 1000)

        p95 = _percentile(latencies, 95)
        print(
            f"\n  [WS 连接建立] p95 = {p95:.2f} ms\n"
            f"    min = {min(latencies):.2f} ms, max = {max(latencies):.2f} ms"
        )
        _assert_baseline("ws_connect_p95_ms", p95)

    @pytest.mark.benchmark
    def test_ws_first_token_latency(self, ws_mock):
        """发送消息到收到第一个 token 延迟 p95 < 500ms (10 次采样, mock LLM)."""
        latencies: list[float] = []
        for _ in range(10):
            with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
                t0 = time.perf_counter()
                ws.send_json(
                    {
                        "type": "user_input",
                        "content": "hello",
                        "thread_id": "perf-test",
                    }
                )
                while True:
                    msg = ws.receive_json()
                    if msg.get("type") == "text_delta":
                        latencies.append((time.perf_counter() - t0) * 1000)
                        break
                    if msg.get("type") == "error":
                        pytest.fail(f"WS error before first token: {msg}")
                # 排空剩余消息直到 done
                while True:
                    msg = ws.receive_json()
                    if msg.get("type") in ("done", "error"):
                        break

        p95 = _percentile(latencies, 95)
        print(
            f"\n  [WS 首 token] p95 = {p95:.2f} ms\n"
            f"    min = {min(latencies):.2f} ms, max = {max(latencies):.2f} ms"
        )
        _assert_baseline("ws_first_token_p95_ms", p95)

    @pytest.mark.benchmark
    def test_ws_full_turn_latency(self, ws_mock):
        """完整对话回合延迟 p95 < 2s (10 次采样, mock LLM)."""
        latencies: list[float] = []
        for _ in range(10):
            with client.websocket_connect(WS_PATH, headers=_HEADERS) as ws:
                t0 = time.perf_counter()
                ws.send_json(
                    {
                        "type": "user_input",
                        "content": "hello",
                        "thread_id": "perf-test",
                    }
                )
                while True:
                    msg = ws.receive_json()
                    if msg.get("type") == "done":
                        latencies.append((time.perf_counter() - t0) * 1000)
                        break
                    if msg.get("type") == "error":
                        pytest.fail(f"WS error during turn: {msg}")

        p95 = _percentile(latencies, 95)
        print(
            f"\n  [WS 完整回合] p95 = {p95:.2f} ms\n"
            f"    min = {min(latencies):.2f} ms, max = {max(latencies):.2f} ms"
        )
        _assert_baseline("ws_full_turn_p95_ms", p95)


# ════════════════════════════════════════════════════════════════════
#  3. 内存稳定性基准
# ════════════════════════════════════════════════════════════════════


class TestMemoryStability:
    """连续请求后的内存增长检测."""

    @pytest.mark.benchmark
    @pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil 未安装")
    def test_rss_growth_health(self):
        """50 轮 /health/live 后 RSS 增长 < 10MB."""
        proc = psutil.Process()
        # 预热: 排除首次导入 / JIT 编译的内存开销
        for _ in range(5):
            client.get("/health/live", headers=_HEADERS)

        rss_before = proc.memory_info().rss
        for _ in range(50):
            client.get("/health/live", headers=_HEADERS)
        rss_after = proc.memory_info().rss
        growth_mb = (rss_after - rss_before) / (1024 * 1024)

        print(
            f"\n  [RSS /health/live x50]\n"
            f"    before = {rss_before / 1024 / 1024:.1f} MB\n"
            f"    after  = {rss_after / 1024 / 1024:.1f} MB\n"
            f"    growth = {growth_mb:.2f} MB"
        )
        _assert_baseline("mem_rss_health_growth_mb", growth_mb, "MB")

    @pytest.mark.benchmark
    @pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil 未安装")
    def test_rss_growth_tools(self):
        """50 轮 /tools 后 RSS 增长 < 10MB."""
        proc = psutil.Process()
        for _ in range(5):
            client.get("/tools", headers=_HEADERS)

        rss_before = proc.memory_info().rss
        for _ in range(50):
            client.get("/tools", headers=_HEADERS)
        rss_after = proc.memory_info().rss
        growth_mb = (rss_after - rss_before) / (1024 * 1024)

        print(
            f"\n  [RSS /tools x50]\n"
            f"    before = {rss_before / 1024 / 1024:.1f} MB\n"
            f"    after  = {rss_after / 1024 / 1024:.1f} MB\n"
            f"    growth = {growth_mb:.2f} MB"
        )
        _assert_baseline("mem_rss_tools_growth_mb", growth_mb, "MB")

    @pytest.mark.benchmark
    def test_tracemalloc_growth(self):
        """tracemalloc 追踪 50 轮 /health/live 后 Python 对象增长 < 5MB."""
        tracemalloc.start()
        # 预热
        for _ in range(5):
            client.get("/health/live", headers=_HEADERS)

        current_before, _ = tracemalloc.get_traced_memory()
        for _ in range(50):
            client.get("/health/live", headers=_HEADERS)
        current_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        growth_mb = (current_after - current_before) / (1024 * 1024)
        print(
            f"\n  [tracemalloc /health/live x50]\n"
            f"    before = {current_before / 1024 / 1024:.2f} MB\n"
            f"    after  = {current_after / 1024 / 1024:.2f} MB\n"
            f"    growth = {growth_mb:.2f} MB"
        )
        _assert_baseline("mem_tracemalloc_growth_mb", growth_mb, "MB")


# ════════════════════════════════════════════════════════════════════
#  4. 并发吞吐量基准
# ════════════════════════════════════════════════════════════════════


class TestConcurrencyThroughput:
    """并发请求全部完成的时间上限."""

    @pytest.mark.benchmark
    def test_concurrent_health_10(self):
        """10 并发 /health/live 全部完成 < 2s."""
        errors: list[str] = []

        def _hit():
            try:
                r = client.get("/health/live", headers=_HEADERS)
                if r.status_code != 200:
                    errors.append(f"status {r.status_code}")
            except Exception as e:
                errors.append(str(e))

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_hit) for _ in range(10)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.perf_counter() - t0

        assert not errors, f"并发请求出错: {errors}"
        print(f"\n  [10 并发 /health/live] 耗时 = {elapsed:.3f} s")
        _assert_baseline("concurrent_health_10_sec", elapsed, "s")

    @pytest.mark.benchmark
    def test_concurrent_tools_20(self):
        """20 并发 /tools 全部完成 < 5s."""
        errors: list[str] = []

        def _hit():
            try:
                r = client.get("/tools", headers=_HEADERS)
                if r.status_code != 200:
                    errors.append(f"status {r.status_code}")
            except Exception as e:
                errors.append(str(e))

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(_hit) for _ in range(20)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.perf_counter() - t0

        assert not errors, f"并发请求出错: {errors}"
        print(f"\n  [20 并发 /tools] 耗时 = {elapsed:.3f} s")
        _assert_baseline("concurrent_tools_20_sec", elapsed, "s")

    @pytest.mark.benchmark
    def test_concurrent_ws_5(self):
        """5 并发 WS 连接同时建立 < 3s."""
        errors: list[str] = []

        def _connect():
            try:
                # 每个线程用独立的 TestClient, 避免共享 portal 的线程安全问题
                c = TestClient(app)
                with c.websocket_connect(WS_PATH, headers=_HEADERS):
                    pass
            except Exception as e:
                errors.append(str(e))

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [pool.submit(_connect) for _ in range(5)]
            for f in as_completed(futures):
                f.result()
        elapsed = time.perf_counter() - t0

        assert not errors, f"WS 并发连接出错: {errors}"
        print(f"\n  [5 并发 WS 连接] 耗时 = {elapsed:.3f} s")
        _assert_baseline("concurrent_ws_5_sec", elapsed, "s")


# ════════════════════════════════════════════════════════════════════
#  5. SQLite 性能基准
# ════════════════════════════════════════════════════════════════════


class TestSQLitePerformance:
    """ResearchLog (SQLite WAL) 的连续写入/读取性能.

    checkpoints 存在内存 dict 里不走 SQLite, 所以 SQLite 读写都用
    ResearchLog 来测 —— 它是项目里唯一做频繁 SQLite 写入的模块.
    """

    @pytest.mark.benchmark
    def test_research_log_writes_100(self, tmp_path: Path):
        """100 次连续写入 (research_log) 总耗时 < 5s."""
        log = ResearchLog(db_path=str(tmp_path / "perf_write.sqlite"))

        t0 = time.perf_counter()
        for i in range(100):
            log.add(
                RecordType.CONJECTURE,
                f"conjecture-{i}",
                f"benchmark write iteration {i}",
                tags=["perf", "benchmark"],
            )
        elapsed = time.perf_counter() - t0

        # 验证写入确实落地
        records = log.list_by_type(RecordType.CONJECTURE)
        assert len(records) == 100, f"期望 100 条, 实际 {len(records)}"

        print(
            f"\n  [SQLite 写入 x100]\n"
            f"    总耗时 = {elapsed:.3f} s\n"
            f"    平均   = {elapsed / 100 * 1000:.2f} ms/次"
        )
        _assert_baseline("sqlite_write_100_sec", elapsed, "s")

    @pytest.mark.benchmark
    def test_research_log_reads_100(self, tmp_path: Path):
        """100 次连续读取 (research_log list) 总耗时 < 2s."""
        log = ResearchLog(db_path=str(tmp_path / "perf_read.sqlite"))
        # 先写入 100 条数据
        for i in range(100):
            log.add(
                RecordType.CONJECTURE,
                f"conjecture-{i}",
                f"benchmark read setup {i}",
                tags=["perf"],
            )

        t0 = time.perf_counter()
        for _ in range(100):
            records = log.list_by_type(RecordType.CONJECTURE)
            assert len(records) == 100, "读取结果数不对, 数据可能丢了"
        elapsed = time.perf_counter() - t0

        print(
            f"\n  [SQLite 读取 x100]\n"
            f"    总耗时 = {elapsed:.3f} s\n"
            f"    平均   = {elapsed / 100 * 1000:.2f} ms/次"
        )
        _assert_baseline("sqlite_read_100_sec", elapsed, "s")


# ════════════════════════════════════════════════════════════════════
#  自检: 验证基线定义和测试覆盖一致
# ════════════════════════════════════════════════════════════════════


def test_baseline_keys_complete():
    """自检: BASELINE 里的每个 key 都有对应的测试在用.

    这是个普通测试 (不标 benchmark), 默认也不跑 (被 pytestmark skip 了),
    但 HUGINN_RUN_PERFORMANCE=1 时会执行, 确保 baseine 和测试不脱节.
    """
    # 收集测试代码里引用的所有 _assert_baseline 调用的 key
    used_keys = {
        "api_health_live_p95_ms",
        "api_tools_p95_ms",
        "api_skills_p95_ms",
        "api_memory_p95_ms",
        "api_personas_p95_ms",
        "ws_connect_p95_ms",
        "ws_first_token_p95_ms",
        "ws_full_turn_p95_ms",
        "mem_rss_health_growth_mb",
        "mem_rss_tools_growth_mb",
        "mem_tracemalloc_growth_mb",
        "concurrent_health_10_sec",
        "concurrent_tools_20_sec",
        "concurrent_ws_5_sec",
        "sqlite_write_100_sec",
        "sqlite_read_100_sec",
    }
    defined_keys = set(BASELINE.keys())
    missing_in_baseline = used_keys - defined_keys
    unused_in_baseline = defined_keys - used_keys
    assert not missing_in_baseline, f"测试用了但 BASELINE 没定义: {missing_in_baseline}"
    assert not unused_in_baseline, f"BASELINE 定义了但没测试在用: {unused_in_baseline}"
