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
import os
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


def routing_surprise(engine: Any) -> float:
    """路由 / 决策 / 记忆 / 展示**统一**的 surprise 信号(秩归一, [0,1]).

    v31: 全系统一律读**相对秩** `_last_surprise_rel`(由 engine_reflect 的
    `_relative_surprise` 产出), 不再各处读原始 `_last_surprise`. 原始值在"语义
    embedder + JEPA predictor 双缺失"时回落 jaccard 且 `worst` 饱和在 1.0
    (run65 实测) ⇒ 恒 >0.9 会每轮强制 explore / 报告恒写 "1=unexpected" /
    episodic 相对秩因并列恒 1.0. 秩信号把"当前值相对历史分布的位置"映射到 [0,1]:
    恒定 → ~0.5(中性), 只有真正相对异常才逼近 1.0.

    用 getattr 容错: 协作对象 (EngineReflect / PlanCheck / HypothesisLoop /
    EngineObserve) 经 ``__getattr__`` 转发到 engine, stub 缺字段时拾取默认值,
    不做硬依赖; 秩未产出时回落原始值以保旧行为.
    """
    rel = getattr(engine, "_last_surprise_rel", None)
    if rel is not None:
        return float(rel)
    return float(getattr(engine, "_last_surprise", 0.0))


# ── Ataraxos 式强度调度 (explore 超参自适应) ────────────────────────
# Ataraxos (arXiv:2511.07312) 的胜负手不是单点超参, 而是"正则强度 / 策略更新
# 规模 / 策略强度"三者的**协调**: 策略弱时强正则 + 大步 (激进探索), 策略强时
# 弱正则 + 小步 (局部精修). 映射到本引擎 —— 用既有纯环信号估一个标量
# strength∈[0,1], 再把原先写死的探索超参改成它的单调函数. 各调度在
# strength=0.5 处**恰好回到旧默认值**, 所以开/关之间平滑, 不引入新状态.
#
# 关闭开关: HUGINN_STRENGTH_SCHEDULE=0 → 各调用点用旧常量, 行为 100% 不变.


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def hypothesis_strength(engine: Any) -> float:
    """[0,1] 标量: 当前假设/策略的"强度". 弱→探索, 强→收敛.

    纯信号合成 (无新状态 / 无 IO), 用既有字段:
      - 0.5 · (1 - surprise)             越不意外越强 (主项)
      - 0.3 · _validate_window 近窗成功率  实验验证通过率 (空窗回落 0.5 中性)
      - 0.2 · (_darwin_best_score/10)     演化质量分 (0-10 → 0-1)
      - 停滞惩罚: 连续无增益每轮 -0.05, 最多 -0.2
    结果 clamp 到 [0,1]. 全字段缺失 (stub) 时 ≈ 0.4 (偏探索), 不误判为强.

    surprise 取秩归一 routing_surprise(); 但"尚无 history"时 (rel 与 raw 都缺)
    回落中性 0.5 —— 否则 raw 默认 0.0 会被读成"毫无意外 = 强", 让起步期偏向
    收敛而非探索.
    """
    _rel = getattr(engine, "_last_surprise_rel", None)
    _raw = float(getattr(engine, "_last_surprise", 0.0) or 0.0)
    surprise = 0.5 if (_rel is None and _raw <= 0.0) else routing_surprise(engine)
    s = 0.5 * (1.0 - surprise)

    window = list(getattr(engine, "_validate_window", None) or [])[-5:]
    success = (
        sum(1 for ok in window if ok) / len(window) if window else 0.5
    )
    s += 0.3 * success

    best = float(getattr(engine, "_darwin_best_score", 0.0) or 0.0)
    s += 0.2 * _clamp01(best / 10.0)

    stag = int(getattr(engine, "_darwin_stagnation", 0) or 0)
    s -= min(0.2, 0.05 * stag)
    return _clamp01(s)


def strength_global_proposal_prob(strength: float, base: float = 0.3) -> float:
    """弱→多全局跳 (逃尖锐后验), 强→少全局跳 (局部精修). s=0.5 回 base."""
    return _clamp01(base + 0.4 * (0.5 - strength))


def strength_temperature(strength: float, base: float = 1.0) -> float:
    """弱→高温宽松接受 (探索), 强→低温锁定 MAP (收敛). s=0.5 回 base."""
    return base + 0.8 * (0.5 - strength)


def strength_branch_depth(strength: float, base: int = 2) -> int:
    """弱→深搜/广探索, 强→浅搜/局部. s=0.5 回 base, 范围 [1,3]."""
    return max(1, min(3, base + round(2.0 * (0.5 - strength))))


def strength_stagnation_limit(strength: float, base: int = 5) -> int:
    """弱→更早 pivot (激进换向), 强→容忍更久 (稳健微调). s=0.5 回 base."""
    return max(2, min(base + 4, base + round(4.0 * (strength - 0.5))))


def strength_schedule_enabled() -> bool:
    """HUGINN_STRENGTH_SCHEDULE (默认 on). off → 调用点用旧常量, 行为不变."""
    return os.environ.get("HUGINN_STRENGTH_SCHEDULE", "1") != "0"


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

    # 强度调度: s=0.5 必须回到旧默认 (平滑), s 单调 (弱↑探索 / 强↑收敛)
    assert abs(strength_global_proposal_prob(0.5) - 0.3) < 1e-9
    assert abs(strength_temperature(0.5) - 1.0) < 1e-9
    assert strength_branch_depth(0.5) == 2
    assert strength_stagnation_limit(0.5) == 5
    assert strength_global_proposal_prob(0.0) > strength_global_proposal_prob(1.0)
    assert strength_temperature(0.0) > strength_temperature(1.0)
    assert strength_branch_depth(0.0) >= 2 >= strength_branch_depth(1.0)
    assert strength_stagnation_limit(0.0) <= 5 <= strength_stagnation_limit(1.0)
    # stub (全字段缺失) → 中性 ~0.5, 且各调度回默认
    class _Stub:
        pass

    _sb = hypothesis_strength(_Stub())
    assert 0.35 < _sb < 0.6, _sb
    assert abs(strength_global_proposal_prob(_sb) - 0.3) < 0.05
    print(f"OK EngineSignals self-check passed ({len(SIGNAL_NAMES)} fields)")


if __name__ == "__main__":
    _selfcheck()
