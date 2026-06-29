"""工具健康仪表盘 —— 聚合每个工具的成功率/耗时/缓存命中/熔断状态。

telemetry 模块记的是 span 树，查询时得遍历整棵树才能算出 per-tool 的
成功率，实时性差。这里自己维护一个滚动窗口（最近 100 次），调用方每次
工具执行完调 record_call 喂数据，仪表盘负责聚合给 verdict。

verdict 判定：
    circuit_open  熔断器开着
    unhealthy     成功率 < 50%
    degraded      成功率 < 80%
    healthy       其它
    unknown       还没数据

纯内存，重启清零。线程安全。不依赖 numpy。
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone

# 滚动窗口大小：只看最近这么多调用，老数据自动挤出
_WINDOW_SIZE = 100


class _ToolRecord:
    """单次工具调用记录。"""

    __slots__ = ("ts", "success", "duration", "cache_hit", "error")

    def __init__(
        self,
        ts: float,
        success: bool,
        duration: float,
        cache_hit: bool,
        error: str | None,
    ) -> None:
        self.ts = ts
        self.success = success
        self.duration = duration
        self.cache_hit = cache_hit
        self.error = error


def _percentile(sorted_vals: list[float], p: float) -> float:
    """ nearest-rank 百分位，纯 Python，不用 numpy。"""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = int(n * p / 100)
    if idx >= n:
        idx = n - 1
    return sorted_vals[idx]


class HealthDashboard:
    """每个工具一份滚动窗口的健康报告，线程安全。

    用法::

        dash = HealthDashboard.shared()
        dash.record_call("vasp_tool", success=True, duration_sec=12.3)
        report = dash.get_health("vasp_tool")
        if report["verdict"] == "unhealthy":
            # 降权或换工具
            ...
    """

    _singleton_lock = threading.Lock()
    _singleton: HealthDashboard | None = None

    def __init__(self, window_size: int = _WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._lock = threading.RLock()
        # tool_name -> 滚动窗口
        self._records: dict[str, deque[_ToolRecord]] = {}

    @classmethod
    def shared(cls) -> HealthDashboard:
        """进程级单例。"""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # ---- 内部 ----

    def _get_window(self, tool_name: str) -> deque[_ToolRecord]:
        w = self._records.get(tool_name)
        if w is None:
            w = deque(maxlen=self._window_size)
            self._records[tool_name] = w
        return w

    def _verdict(self, success_rate: float, total: int, circuit_state: str) -> str:
        if total == 0:
            return "unknown"
        if circuit_state == "open":
            return "circuit_open"
        if success_rate < 0.5:
            return "unhealthy"
        if success_rate < 0.8:
            return "degraded"
        return "healthy"

    # ---- 公开 API ----

    def record_call(
        self,
        tool_name: str,
        success: bool,
        duration_sec: float,
        cache_hit: bool = False,
        error: str | None = None,
    ) -> None:
        """记一次工具调用。成功/失败都记，缓存命中也记（算 cache_hit_rate）。"""
        with self._lock:
            w = self._get_window(tool_name)
            w.append(
                _ToolRecord(
                    ts=time.time(),
                    success=success,
                    duration=duration_sec,
                    cache_hit=cache_hit,
                    error=error,
                )
            )

    def get_health(self, tool_name: str) -> dict:
        """单个工具的健康报告。"""
        with self._lock:
            w = self._records.get(tool_name)
            if w is None or len(w) == 0:
                # 没数据也报一下熔断状态，方便排查
                circuit_state = _safe_circuit_state(tool_name)
                return {
                    "tool": tool_name,
                    "total_calls": 0,
                    "success_rate": 0.0,
                    "avg_duration_sec": 0.0,
                    "p95_duration_sec": 0.0,
                    "cache_hit_rate": 0.0,
                    "circuit_state": circuit_state,
                    "last_error": None,
                    "last_call": None,
                    "verdict": self._verdict(0.0, 0, circuit_state),
                }

            records = list(w)
            total = len(records)
            successes = sum(1 for r in records if r.success)
            durations = [r.duration for r in records]
            cache_hits = sum(1 for r in records if r.cache_hit)
            last = records[-1]

            success_rate = successes / total
            avg_duration = sum(durations) / total
            p95 = _percentile(sorted(durations), 95)
            cache_hit_rate = cache_hits / total
            circuit_state = _safe_circuit_state(tool_name)

            return {
                "tool": tool_name,
                "total_calls": total,
                "success_rate": round(success_rate, 3),
                "avg_duration_sec": round(avg_duration, 3),
                "p95_duration_sec": round(p95, 3),
                "cache_hit_rate": round(cache_hit_rate, 3),
                "circuit_state": circuit_state,
                "last_error": last.error,
                "last_call": _iso_utc(last.ts),
                "verdict": self._verdict(success_rate, total, circuit_state),
            }

    def get_all(self) -> list[dict]:
        """所有有记录的工具的健康报告。"""
        with self._lock:
            return [self.get_health(name) for name in self._records]

    def get_unhealthy(self) -> list[dict]:
        """成功率 < 80% 或熔断中的工具，给 agent 推荐降权用。"""
        with self._lock:
            out = []
            for name in self._records:
                h = self.get_health(name)
                if h["verdict"] in ("unhealthy", "degraded", "circuit_open"):
                    out.append(h)
            return out

    def summary(self) -> dict:
        """总体统计。"""
        with self._lock:
            all_reports = [self.get_health(name) for name in self._records]
            total_calls = sum(r["total_calls"] for r in all_reports)
            total_successes = sum(
                int(r["total_calls"] * r["success_rate"]) for r in all_reports
            )
            by_verdict: dict[str, int] = {}
            for r in all_reports:
                v = r["verdict"]
                by_verdict[v] = by_verdict.get(v, 0) + 1
            return {
                "total_tools": len(all_reports),
                "total_calls": total_calls,
                "overall_success_rate": round(
                    total_successes / total_calls, 3
                ) if total_calls else 0.0,
                "by_verdict": by_verdict,
            }

    def reset(self, tool_name: str | None = None) -> None:
        """清记录。传 None 清全部，传 tool_name 只清一个。"""
        with self._lock:
            if tool_name is None:
                self._records.clear()
            else:
                self._records.pop(tool_name, None)


def _safe_circuit_state(tool_name: str) -> str:
    """best-effort 查熔断状态，熔断器没装或没数据就当 closed。"""
    try:
        from huginn.agents.circuit_breaker import CircuitBreaker

        return CircuitBreaker.shared().get_state(tool_name)
    except Exception:
        return "closed"


def _iso_utc(ts: float) -> str:
    """unix 时间戳转 ISO8601 UTC 字符串。"""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
    except Exception:
        return ""
