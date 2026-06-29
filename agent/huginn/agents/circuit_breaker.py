"""工具熔断器 —— 连续失败的工具自动熔断，避免雪崩。

某个工具接连失败（比如外部 API 挂了、HPC 调度卡死），如果不拦住，
agent 会一遍遍重试把整轮推理拖垮。这里用经典三态熔断器：

    closed    正常放行，累计连续失败数
    open      连续失败达阈值，直接拒绝请求，等冷却时间
    half_open 冷却到了放一个试探请求，成功转 closed，失败转 open

纯内存，重启清零 —— 上次坏状态不该卡住新进程。线程安全。
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Generator


class _BreakerState:
    """单个工具的熔断状态。"""

    __slots__ = (
        "state",
        "consecutive_failures",
        "last_failure_time",
        "last_error",
        "half_open_trials",
    )

    def __init__(self) -> None:
        self.state: str = "closed"
        self.consecutive_failures: int = 0
        self.last_failure_time: float | None = None
        self.last_error: str = ""
        self.half_open_trials: int = 0


class CircuitBreaker:
    """每个 tool_name 一个独立熔断器，线程安全。

    用法::

        breaker = CircuitBreaker.shared()
        if not breaker.can_call("vasp_tool"):
            return {"error": "circuit_open", "tool": "vasp_tool"}
        try:
            result = run_tool()
            breaker.record_success("vasp_tool")
        except Exception as exc:
            breaker.record_failure("vasp_tool", str(exc))
            raise
    """

    _singleton_lock = threading.Lock()
    _singleton: CircuitBreaker | None = None

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 60.0,
        half_open_max: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._half_open_max = half_open_max
        self._lock = threading.RLock()
        self._states: dict[str, _BreakerState] = {}

    @classmethod
    def shared(cls) -> CircuitBreaker:
        """进程级单例。所有工具共用一份熔断状态。"""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # ---- 内部 ----

    def _get_state(self, tool_name: str) -> _BreakerState:
        st = self._states.get(tool_name)
        if st is None:
            st = _BreakerState()
            self._states[tool_name] = st
        return st

    def _refresh(self, st: _BreakerState) -> None:
        """冷却到期就把 open 转成 half_open，准备放试探请求。"""
        if st.state == "open" and st.last_failure_time is not None:
            if time.time() - st.last_failure_time >= self._cooldown:
                st.state = "half_open"
                st.half_open_trials = 0

    # ---- 公开 API ----

    def can_call(self, tool_name: str) -> bool:
        """能不能调这个工具。closed/half_open(限额内) 放行，open 拦下。"""
        with self._lock:
            st = self._get_state(tool_name)
            self._refresh(st)
            if st.state == "closed":
                return True
            if st.state == "half_open":
                # 半开状态下只放有限的试探请求，多了直接拒
                if st.half_open_trials < self._half_open_max:
                    st.half_open_trials += 1
                    return True
                return False
            # open：还在冷却里
            return False

    def record_success(self, tool_name: str) -> None:
        """记录一次成功，清零失败计数，回 closed。"""
        with self._lock:
            st = self._get_state(tool_name)
            st.consecutive_failures = 0
            st.half_open_trials = 0
            st.state = "closed"

    def record_failure(self, tool_name: str, error: str = "") -> None:
        """记录一次失败。累计到阈值就 open，半开状态下失败直接回 open。"""
        with self._lock:
            st = self._get_state(tool_name)
            st.consecutive_failures += 1
            st.last_failure_time = time.time()
            if error:
                st.last_error = error

            if st.state == "half_open":
                # 试探请求失败，重新熔断，冷却时间重算
                st.state = "open"
                st.half_open_trials = 0
                return

            if st.consecutive_failures >= self._failure_threshold:
                st.state = "open"

    def get_state(self, tool_name: str) -> str:
        """返回 closed / open / half_open。会顺带做冷却到期转换。"""
        with self._lock:
            st = self._get_state(tool_name)
            self._refresh(st)
            return st.state

    def get_stats(self, tool_name: str) -> dict:
        """单个工具的熔断详情，含剩余冷却时间。"""
        with self._lock:
            st = self._get_state(tool_name)
            self._refresh(st)
            retry_after = 0.0
            if st.state == "open" and st.last_failure_time is not None:
                remaining = self._cooldown - (time.time() - st.last_failure_time)
                retry_after = max(0.0, remaining)
            return {
                "tool": tool_name,
                "state": st.state,
                "consecutive_failures": st.consecutive_failures,
                "failure_threshold": self._failure_threshold,
                "last_failure_time": st.last_failure_time,
                "last_error": st.last_error,
                "half_open_trials": st.half_open_trials,
                "retry_after": round(retry_after, 2),
            }

    def list_all(self) -> list[dict]:
        """所有工具的熔断状态快照，方便仪表盘/调试用。"""
        with self._lock:
            return [self.get_stats(name) for name in self._states]

    def reset(self, tool_name: str | None = None) -> None:
        """手动重置。传 None 重置全部，传 tool_name 只重置一个。"""
        with self._lock:
            if tool_name is None:
                self._states.clear()
            else:
                self._states.pop(tool_name, None)


@contextmanager
def circuit_guard(
    tool_name: str,
    breaker: CircuitBreaker | None = None,
) -> Generator[dict, None, None]:
    """上下文管理器：自动判熔断 + 记成功/失败。

    用法::

        with circuit_guard("vasp_tool") as ctx:
            if ctx["blocked"]:
                return ctx["error_result"]
            result = run_tool()
            ctx["result"] = result

    退出时根据有没有抛异常自动 record_success / record_failure。
    """
    b = breaker or CircuitBreaker.shared()
    ctx: dict = {
        "blocked": False,
        "error_result": None,
        "result": None,
    }
    if not b.can_call(tool_name):
        stats = b.get_stats(tool_name)
        ctx["blocked"] = True
        ctx["error_result"] = {
            "error": "circuit_open",
            "tool": tool_name,
            "retry_after": stats.get("retry_after", 0),
        }
        yield ctx
        return
    try:
        yield ctx
        b.record_success(tool_name)
    except Exception as exc:
        b.record_failure(tool_name, str(exc))
        raise
