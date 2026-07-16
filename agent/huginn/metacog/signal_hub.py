"""SignalHub — 外部信号到 CSM TransitionSignal 的统一路由.

v4 spec Task 1 (G17). 把 perception / event / belief / skill / evolution 等
外部信号源映射到 TransitionSignal.signal_type, 由调用方决定是否真的 transition.

不持有 CSM 引用 — route() 只构造 TransitionSignal, 让上层 (agent.py / engine.py)
去调 csm.transition(). 这样 SignalHub 可以在无 CSM 的环境里自检.
"""

from __future__ import annotations

import threading
from typing import Any

from huginn.cognitive_engine import TransitionSignal


# 源 → signal_type 映射. 模块级常量, 运行时可经 register() 扩展.
_SOURCE_TO_SIGNAL_TYPE: dict[str, str] = {
    "perception_converged": "tool_success",
    "perception_conflict": "gap_found",
    "perception_error": "tool_failure",
    "event_overflow": "context_overflow",
    "event_tool_burst": "tool_failure",
    "belief_high": "belief_high",
    "skill_failure": "tool_failure",
    "evolution_rule": "evolution_rule_learned",
}


class SignalHub:
    """外部信号 → TransitionSignal 的路由器.

    单例 — 用 shared() 拿全局实例. 也可直接 SignalHub() 用于测试隔离.
    多线程下 register() 加锁, route() 走快路径 (无锁读 dict).
    """

    _instance: "SignalHub | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        # 拷贝默认表, 避免外部 register 改到模块常量
        self._table: dict[str, str] = dict(_SOURCE_TO_SIGNAL_TYPE)

    @classmethod
    def shared(cls) -> "SignalHub":
        # ponytail: 双检锁单例. 当前 metacog 并发量低, 简单 lock 已足够.
        # 升级路径: 若 route() 热路径出现锁竞争, 改 threading.local 或无锁读.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def route(
        self,
        signal_source: str,
        payload: dict[str, Any] | None = None,
    ) -> TransitionSignal | None:
        """把外部信号源映射成 TransitionSignal.

        未映射的 source 返回 None, 调用方自己决定 fallback (丢弃 / 记日志 / 抛错).
        payload 透传到 TransitionSignal.data.
        """
        sig_type = self._table.get(signal_source)
        if sig_type is None:
            return None
        return TransitionSignal(signal_type=sig_type, data=dict(payload or {}))

    def register(self, source: str, signal_type: str) -> None:
        """运行时扩展映射表 (覆盖已有 source)."""
        with self._lock:
            self._table[source] = signal_type


# ── 自检 ─────────────────────────────────────────────────────────
# ponytail: 非平凡逻辑留一个 runnable check. 验证默认映射 + unknown 退化 + payload 透传.

def _selfcheck() -> None:
    hub = SignalHub()
    s = hub.route("perception_converged")
    assert s is not None and s.signal_type == "tool_success"
    assert hub.route("unknown_source") is None
    # payload 透传
    s2 = hub.route("perception_error", {"tool": "vasp", "err": "edddav"})
    assert s2 is not None and s2.data.get("tool") == "vasp"
    # register 生效
    hub.register("custom_signal", "gap_found")
    assert hub.route("custom_signal") is not None
    print("signal_hub selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
