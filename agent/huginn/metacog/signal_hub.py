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
    # v6 G50: 结构主义信号 — validate_structure_preservation 失败 / sorry 填充 / sorry 不可实现
    "structure_violation": "gap_found",
    "sorry_filled": "belief_high",
    "sorry_impossible": "gap_found",
}


class SignalHub:
    """外部信号 → TransitionSignal 的路由器.

    单例 — 用 shared() 拿全局实例. 也可直接 SignalHub() 用于测试隔离.
    多线程下 register() 加锁, route() 走快路径 (无锁读 dict).

    H1 修复: 两类调用方分走两个方法, 避免 drain 重复 transition:
    - route(): 有 csm 引用的调用方用, 返回信号让上层自己 transition. 不 enqueue.
    - emit(): 无 csm 引用的 emit 方 (e.g. FailureModeRegistry) 用, 只 enqueue.
      reflection._run_post_turn_reflection 末尾 drain_pending 拉 emit 的信号.
    """

    _instance: "SignalHub | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        # 拷贝默认表, 避免外部 register 改到模块常量
        self._table: dict[str, str] = dict(_SOURCE_TO_SIGNAL_TYPE)
        # pending queue: emit() 时 enqueue, drain_pending() 消费.
        # ponytail: list + lock, 不上 queue.Queue 避免线程模型耦合.
        self._pending: list[TransitionSignal] = []
        self._pending_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "SignalHub":
        # ponytail: 双检锁单例. 当前 metacog 并发量低, 简单 lock 已足够.
        # 升级路径: 若 route() 热路径出现锁竞争, 改 threading.local 或无锁读.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _build(
        self, signal_source: str, payload: dict[str, Any] | None,
    ) -> TransitionSignal | None:
        sig_type = self._table.get(signal_source)
        if sig_type is None:
            return None
        return TransitionSignal(signal_type=sig_type, data=dict(payload or {}))

    def route(
        self,
        signal_source: str,
        payload: dict[str, Any] | None = None,
    ) -> TransitionSignal | None:
        """有 csm 引用的调用方用: 返回信号, 上层自己 transition. 不 enqueue."""
        return self._build(signal_source, payload)

    def emit(
        self,
        signal_source: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """无 csm 引用的 emit 方用: enqueue 到 pending, 由 reflection drain.

        H1: FailureModeRegistry.record_observation 调这个, 之前调 route() 返回值
        被丢弃, 信号永远不进 CSM. 现在 enqueue, reflection 末尾 drain 并 transition.
        """
        sig = self._build(signal_source, payload)
        if sig is None:
            return
        with self._pending_lock:
            self._pending.append(sig)

    def drain_pending(self) -> list[TransitionSignal]:
        """取出并清空 pending 信号. reflection._run_post_turn_reflection 调."""
        with self._pending_lock:
            queued = list(self._pending)
            self._pending.clear()
        return queued

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
    # v6 G50: 结构主义信号
    assert hub.route("structure_violation") is not None
    assert hub.route("structure_violation").signal_type == "gap_found"
    assert hub.route("sorry_filled") is not None
    assert hub.route("sorry_filled").signal_type == "belief_high"
    assert hub.route("sorry_impossible") is not None
    assert hub.route("sorry_impossible").signal_type == "gap_found"

    # H1: emit (enqueue) + drain_pending + route 不 enqueue
    hub2 = SignalHub()
    assert hub2.drain_pending() == [], "初始 pending 应为空"
    # route 不 enqueue (有 csm 引用方自己 transition, 不该被 drain 重复)
    hub2.route("perception_converged", {"x": 1})
    assert hub2.drain_pending() == [], "route 不应 enqueue"
    # emit enqueue
    hub2.emit("skill_failure", {"mode_id": "runtime_error", "tool_name": "vasp"})
    queued = hub2.drain_pending()
    assert len(queued) == 1, f"emit 应 enqueue 1 个, got {len(queued)}"
    assert queued[0].signal_type == "tool_failure", \
        f"skill_failure → tool_failure, got {queued[0].signal_type}"
    assert queued[0].data.get("tool_name") == "vasp"
    # drain 后清空
    assert hub2.drain_pending() == [], "drain 后应清空"
    # unknown source emit 不 enqueue
    hub2.emit("unknown_source", {"x": 1})
    assert hub2.drain_pending() == [], "unknown source 不该 enqueue"

    print("signal_hub selfcheck OK (含 H1 emit/drain)")


if __name__ == "__main__":
    _selfcheck()
