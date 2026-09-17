"""EngineSignals —— AutoloopEngine 的纯环信号收敛点（EngineSignals 重构核心）.

背景: AutoloopEngine 的 ~29 个**纯环信号**字段(cognitive_loop 失败/回退计数、
surprise / darwin 演化信号、plan_check 状态、迭代/点名状态)被 10 个 mixin 与
外部(runtime/engine_state 等)交错读写(实测 ~112 处/15 文件)。把这些信号收进一个
显式 dataclass, 让"单一事实源 + 可序列化 snapshot"成立, 并通过引擎上的属性桥
保持既有 ``self._<field>`` 读点稳定。

本模块只放**纯数据 + 序列化**, 无任何逻辑分支 / IO / LLM。基建态(model/kg/memory/
budget/mcmc/persona 等)留在引擎, 不进这里。

``SIGNAL_NAMES`` 是本结构唯一事实源: `engine.py` 的属性桥与 `runtime/engine_state.py`
的持久化都从这里取字段名, 杜绝字段名在不同文件重复声明。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineSignals:
    """AutoloopEngine 的纯环信号集合.

    字段名对齐既有 ``self._<field>``(下划线前缀), 便于属性桥 get/set 直接映射。
    默认值与重构前 engine ``_init_*`` 分片里的初始值一致。
    """

    # ── G1 失败/回退 ─────────────────────────────────────────────
    _consecutive_failures: int = 0
    _consecutive_failures_by_type: dict[str, int] = field(default_factory=dict)
    _validate_window: list[bool] = field(default_factory=list)
    _refine_count: int = 0
    _pivot_count: int = 0
    _next_phase_hint: str | None = None
    _refined_hypothesis: str | None = None

    # ── G2 演化信号 ──────────────────────────────────────────────
    _last_surprise: float = 0.0
    _surprise_history: list[tuple[float, float]] = field(default_factory=list)
    _darwin_best_score: float = 0.0
    _darwin_stagnation: int = 0
    _darwin_last_score: float = 0.0
    _darwin_belief_mu: float = 0.0
    _darwin_belief_sigma2: float = 100.0
    _last_hypothesis_confidence: float = 0.0
    _last_hypothesis_evidence_strength: float = 0.0
    _evals_history: list[Any] = field(default_factory=list)
    _scene_tag_extra_keywords: dict[str, set[str]] = field(default_factory=dict)

    # ── G3 计划检查 ──────────────────────────────────────────────
    _plan_check_history: list[dict[str, Any]] = field(default_factory=list)
    _plan_check_last_result: dict[str, Any] | None = None
    _plan_check_warnings: list[str] = field(default_factory=list)
    _plan_check_patterns: list[dict[str, Any]] = field(default_factory=list)

    # ── G4 迭代 / 点名状态 ───────────────────────────────────────
    _iteration: int = 0
    _should_stop: bool = False
    _current_phase: str = ""
    _grill_active: bool = False
    _grill_turns: int = 0
    _last_visual_context: str = ""
    _last_rule_hit_id: str = ""

    # ── 序列化 ──────────────────────────────────────────────────

    def to_snapshot(self) -> dict[str, Any]:
        """JSON 可序列化的信号快照.

        把嵌套 set(dict[str,set[str]]) 转成排序 list, 保证 json.dumps 可过;
        tuple(如 _surprise_history 条目) json 会自然转 list, 由 from_snapshot 还原。
        """
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if f.name == "_scene_tag_extra_keywords" and not isinstance(val, dict):
                val = {}
            if f.name == "_scene_tag_extra_keywords":
                val = {k: sorted(v) for k, v in val.items()}
            out[f.name] = val
        return out

    @classmethod
    def from_snapshot(cls, d: dict[str, Any]) -> EngineSignals:
        """从快照重建，容忍缺键(走默认值)与 set/tuple 的 JSON 变形."""
        obj = cls()
        for f in dataclasses.fields(cls):
            n = f.name
            if n not in d or d[n] is None:
                continue
            val = d[n]
            if n == "_scene_tag_extra_keywords":
                val = {k: set(v) for k, v in val.items()}
            elif n == "_surprise_history":
                try:
                    val = [
                        tuple(float(x) for x in pair)
                        for pair in (val or [])
                    ]
                except (TypeError, ValueError):
                    continue
            setattr(obj, n, val)
        return obj

    def apply_to_engine(self, engine: Any) -> None:
        """把信号挂到引擎(经属性桥). 引擎 ``_init_*`` 会在此之后填充信号字段."""
        engine.signals = self


# ── 字段名单一事实源 ─────────────────────────────────────────────
SIGNAL_NAMES: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(EngineSignals))


def _selfcheck() -> None:
    s = EngineSignals()
    s._iteration = 7
    s._last_surprise = 0.42
    s._validate_window = [True, False]
    s._scene_tag_extra_keywords = {"dft": {"iso", "relax"}}
    s._surprise_history = [(0.1, 0.2), (0.3, 0.4)]
    snap = s.to_snapshot()
    import json

    json.loads(json.dumps(snap))  # 必须 JSON 可序列化
    s2 = EngineSignals.from_snapshot(snap)
    assert s2._iteration == 7
    assert s2._scene_tag_extra_keywords == {"dft": {"iso", "relax"}}
    assert s2._surprise_history == [(0.1, 0.2), (0.3, 0.4)]
    empty = EngineSignals.from_snapshot({})
    assert empty._iteration == 0 and empty._scene_tag_extra_keywords == {}
    assert len(SIGNAL_NAMES) >= 28
    print(f"OK EngineSignals self-check passed ({len(SIGNAL_NAMES)} fields)")


if __name__ == "__main__":
    _selfcheck()
