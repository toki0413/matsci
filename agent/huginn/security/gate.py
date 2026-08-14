"""统一门禁链 — 用时空可组合性 (CoEffectRegistry) 统一各门禁的激活与聚合.

背景: 代码库存在多个异构门禁 (CompletionGate 三审、SanityGate 机械检查、
phase_gate / completion_gate / adoption_gate / sanity_gate ...), 各自实现
"什么时候该跑、结果怎么判"。本模块给出一个**统一层**:

- **激活统一由依赖推导** (空间可组合): 每个门禁声明 requires (前置条件),
  依赖不满足 → 门禁自动停用 (skip), 不重复"手写 if 前置在不在".
- **决策聚合统一**: evaluate_all 只跑"激活的门禁", 归一化为 GateResult,
  all_pass 聚合最终放行与否.

不重写任何门禁内部检查逻辑, 只统一"激活推导 + 决策聚合"两层, 安全可迁移.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from huginn.security.coeffect import CoEffectRegistry

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """一个门禁的归一化决策."""

    gate: str
    status: str  # pass | block | skip
    reason: str = ""


class GateAdapter(Protocol):
    """门禁适配器: 声明依赖 + 提供归一化评估."""

    def gate_id(self) -> str: ...

    def gate_requires(self) -> set[str]: ...

    def evaluate(self, ctx: Any) -> GateResult: ...


@dataclass
class FuncGate:
    """把 (id, requires, evaluator) 函数包成一个 GateAdapter 的最小实现."""

    id: str
    requires: set[str] = field(default_factory=set)
    evaluator: Callable[[Any], GateResult] = lambda ctx: GateResult("", "pass")

    def gate_id(self) -> str:
        return self.id

    def gate_requires(self) -> set[str]:
        return set(self.requires)

    def evaluate(self, ctx: Any) -> GateResult:
        r = self.evaluator(ctx)
        # evaluator 返回 (status, reason) 元组, 或 GateResult 对象.
        if isinstance(r, GateResult):
            return GateResult(gate=self.id, status=r.status, reason=r.reason)
        status, reason = r
        return GateResult(gate=self.id, status=status, reason=reason)


class GateChain:
    """统一门禁链: CoEffectRegistry 推导激活, 统一聚合决策."""

    def __init__(self) -> None:
        self._reg = CoEffectRegistry()
        self._adapters: dict[str, GateAdapter] = {}

    def register(self, adapter: GateAdapter) -> GateAdapter:
        """注册一个门禁, 用其 requires 声明依赖 (空间可组合)."""
        self._adapters[adapter.gate_id()] = adapter
        self._reg.declare(adapter.gate_id(), requires=adapter.gate_requires())
        return adapter

    def set_available(self, key: str, available: bool) -> None:
        """设置前置条件可用性, 驱动门禁激活/停用."""
        self._reg.set_available(key, available)

    def is_active(self, gate_id: str) -> bool:
        """该门禁是否激活 (所有 requires 满足)."""
        return self._reg.is_active(gate_id)

    def evaluate_all(self, ctx: Any) -> list[GateResult]:
        """只评估激活的门禁; 依赖不满足的门禁返回 skip."""
        out: list[GateResult] = []
        for gid, adapter in self._adapters.items():
            if not self.is_active(gid):
                out.append(GateResult(gid, "skip", "前置条件未满足, 门禁停用"))
                continue
            out.append(adapter.evaluate(ctx))
        return out

    @staticmethod
    def all_pass(results: list[GateResult]) -> bool:
        """聚合: 所有结果均为 pass (skip 视为不阻断 = 通过)."""
        return all(r.status in ("pass", "skip") for r in results)


# ── 适配器 ───────────────────────────────────────────────────────
def _map_decision_status(status: str) -> str:
    """CompletionGate.GateDecision.status → GateResult.status.

    pass/pending 不阻断 → pass; block / gaps_hint 阻断 → block.
    """
    return "block" if status in ("block", "gaps_hint") else "pass"


def completion_adapter(
    gate: Any,
    requires: set[str],
) -> FuncGate:
    """把 CompletionGate.review 适配为 FuncGate.

    ``gate`` 有 ``review(goal, ctx) -> GateDecision(status, reason)``.
    """

    def evaluator(ctx: Any) -> tuple[str, str]:
        goal, gate_ctx = ctx  # ctx = (goal, GateContext)
        decision = gate.review(goal, gate_ctx)
        return (_map_decision_status(decision.status), decision.reason)

    return FuncGate(id="completion_gate", requires=requires, evaluator=evaluator)


def sanity_adapter(checker: Callable[[Any], dict[str, Any]], requires: set[str]) -> FuncGate:
    """把 check_sanity(workspace)->{passed,...} 适配为 FuncGate.

    ``checker`` 传入的目标是 workspace 路径.
    """

    def evaluator(ctx: Any) -> tuple[str, str]:
        result = checker(ctx)
        passed = bool(result.get("passed", True))
        return ("pass" if passed else "block", result.get("reason", ""))

    return FuncGate(id="sanity_gate", requires=requires, evaluator=evaluator)


def phase_gate_adapter(
    hook: Any,
    from_phase: str,
    to_phase: str,
    requires: set[str],
) -> FuncGate:
    """把 PhaseGateHook.evaluate(from, to, evidence) 适配为 FuncGate.

    ``hook`` 有 ``evaluate(from_phase, to_phase, evidence) -> PhaseGate(status, feedback)``.
    ctx 约定为 evidence dict; 阻断状态 (blocked/rejected) 归一化为 block.
    """

    def evaluator(ctx: Any) -> tuple[str, str]:
        gate = hook.evaluate(from_phase, to_phase, ctx)
        status = "block" if gate.is_blocked else "pass"
        return (status, gate.feedback)

    return FuncGate(
        id=f"phase_gate:{from_phase}->{to_phase}",
        requires=requires,
        evaluator=evaluator,
    )


def adoption_adapter(
    gate: Any,
    config_id: str,
    requires: set[str],
) -> FuncGate:
    """把 AdoptionGate.should_adopt(config_id) 适配为 FuncGate.

    ``gate`` 有 ``decide(config_id) -> AdoptionDecision(status, adopt)``.
    YELLOW/RED (adopt=False) 归一化为 block, 只有 GREEN 放行.
    """

    def evaluator(ctx: Any) -> tuple[str, str]:
        decision = gate.decide(config_id)
        adopted = bool(decision.adopt)
        return ("pass" if adopted else "block", f"{decision.status}: {decision.reason}")

    return FuncGate(id="adoption_gate", requires=requires, evaluator=evaluator)