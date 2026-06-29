"""工具调用预算 —— 限制单轮 agent 调工具的次数，防死循环。

agent 有时候会卡在某个工具上反复调（比如查结构查不到就一遍遍换参数
重查），把预算耗光之前主动停掉，比让它自己跑满 recursion_limit 省
时间也省 token。
"""

from __future__ import annotations

import threading
import time
from typing import Any


class ToolCallBudget:
    """跟踪单次 agent 调用的工具次数，线程安全。

    用法::

        budget = ToolCallBudget(max_calls=15, max_per_tool=5)
        if not budget.record("vasp_tool"):
            # 超预算了，该停
            stop, reason = budget.should_stop()
    """

    def __init__(
        self,
        max_calls: int = 15,
        max_per_tool: int = 5,
        max_walltime_sec: float | None = None,
    ) -> None:
        self.max_calls = max_calls
        self.max_per_tool = max_per_tool
        # 总耗时上限 (秒), None 表示不限. 首次 record/check 时开始计时.
        self.max_walltime_sec = max_walltime_sec
        self._lock = threading.RLock()
        self._total: int = 0
        # tool_name -> 调用次数
        self._per_tool: dict[str, int] = {}
        # 首次调用的时间戳, 用来算 walltime. None 表示还没开始计时.
        self._start_time: float | None = None

    @property
    def _elapsed_sec(self) -> float:
        """从首次调用到现在的耗时，没开始计时返回 0。"""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def record(self, tool_name: str) -> bool:
        """记录一次工具调用，返回 True 表示还能继续，False 表示超预算了。

        不管返回啥都会把这次调用记上（方便事后排查），调用方拿到 False
        就该停止后续工具调用。
        """
        with self._lock:
            # 首次调用开始计时
            if self._start_time is None:
                self._start_time = time.time()

            self._total += 1
            self._per_tool[tool_name] = self._per_tool.get(tool_name, 0) + 1

            if self._total > self.max_calls:
                return False
            if self._per_tool[tool_name] > self.max_per_tool:
                return False
            # walltime 超了也算超预算
            if self.max_walltime_sec is not None:
                if (time.time() - self._start_time) > self.max_walltime_sec:
                    return False
            return True

    def should_stop(self) -> tuple[bool, str]:
        """返回 (是否该停, 原因)。"""
        with self._lock:
            if self._total > self.max_calls:
                return True, (
                    f"工具调用总数 {self._total} 超过预算 {self.max_calls}"
                )
            # 找有没有单个工具超限
            for name, count in self._per_tool.items():
                if count > self.max_per_tool:
                    return True, (
                        f"工具 {name} 调用 {count} 次超过单工具上限 {self.max_per_tool}"
                    )
            # walltime 检查
            if self.max_walltime_sec is not None and self._start_time is not None:
                elapsed = time.time() - self._start_time
                if elapsed > self.max_walltime_sec:
                    return True, (
                        f"walltime {elapsed:.1f}s 超过预算 {self.max_walltime_sec}s"
                    )
            return False, ""

    def status(self) -> dict[str, Any]:
        """返回当前预算使用情况。"""
        with self._lock:
            elapsed = self._elapsed_sec
            walltime_exceeded = (
                self.max_walltime_sec is not None
                and elapsed > self.max_walltime_sec
            )
            return {
                "total_calls": self._total,
                "max_calls": self.max_calls,
                "remaining": max(self.max_calls - self._total, 0),
                "max_per_tool": self.max_per_tool,
                "per_tool": dict(self._per_tool),
                "exceeded": self._total > self.max_calls,
                "max_walltime_sec": self.max_walltime_sec,
                "elapsed_sec": elapsed,
                "walltime_exceeded": walltime_exceeded,
            }

    def check(self, tool_name: str) -> dict[str, Any]:
        """检查并记录一次工具调用，返回带 allowed/reason 的详细 dict。

        跟 record() 的区别在于返回值是结构化的，调用方不用再单独调
        should_stop() 拿原因，直接看 reason 字段就行。
        """
        with self._lock:
            # 首次 check 开始计时
            if self._start_time is None:
                self._start_time = time.time()

            # 先把这次调用记上
            self._total += 1
            self._per_tool[tool_name] = self._per_tool.get(tool_name, 0) + 1

            # 总次数超限
            if self._total > self.max_calls:
                return {
                    "allowed": False,
                    "reason": "max_calls_exceeded",
                    "total": self._total,
                    "max_calls": self.max_calls,
                }
            # 单工具次数超限
            if self._per_tool[tool_name] > self.max_per_tool:
                return {
                    "allowed": False,
                    "reason": "per_tool_exceeded",
                    "tool": tool_name,
                    "count": self._per_tool[tool_name],
                    "max_per_tool": self.max_per_tool,
                }
            # walltime 超限
            if self.max_walltime_sec is not None:
                elapsed = time.time() - self._start_time
                if elapsed > self.max_walltime_sec:
                    return {
                        "allowed": False,
                        "reason": "walltime_exceeded",
                        "elapsed": elapsed,
                        "budget": self.max_walltime_sec,
                    }

            return {"allowed": True, "reason": None}

    def remaining(self) -> dict[str, Any]:
        """返回剩余预算（次数 + walltime）。"""
        with self._lock:
            calls_remaining = max(self.max_calls - self._total, 0)
            walltime_remaining = None
            if self.max_walltime_sec is not None:
                # 已超时会返回负数，调用方自己判断
                walltime_remaining = self.max_walltime_sec - self._elapsed_sec
            return {
                "calls_remaining": calls_remaining,
                "walltime_remaining_sec": walltime_remaining,
            }

    def reset(self) -> None:
        """清零计数（下一轮 agent 调用开始时用）。"""
        with self._lock:
            self._total = 0
            self._per_tool.clear()
            self._start_time = None

    def __repr__(self) -> str:
        with self._lock:
            wt = ""
            if self.max_walltime_sec is not None:
                wt = f", elapsed={self._elapsed_sec:.1f}/{self.max_walltime_sec}s"
            return (
                f"ToolCallBudget(total={self._total}/{self.max_calls}{wt}, "
                f"per_tool={self._per_tool})"
            )
