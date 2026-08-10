"""DecisionArbiter — 统一决策仲裁层.

之前 huginn 有三套独立决策器无共享上下文:
  1. CognitiveStateMachine (CSM) — 状态转移决策 (S0→S7)
  2. EffortBandit — 努力分配决策 (continue/switch/requery)
  3. CompletionGate — 完成放行决策 (pass/block/gaps_hint)

三者各自维护独立状态, Bandit 建议 explore 时 Gate 可能已 pass, 反之亦然,
无仲裁层. DecisionArbiter 收敛为唯一决策出口.

使用方式:
    arbiter = DecisionArbiter()
    ctx = arbiter.build_context(csm_state, bandit, gate_decision, ...)
    decision = arbiter.evaluate(ctx)
    if decision.action == "stop":
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DecisionContext:
    """决策时可见的统一上下文 — 所有决策器共享."""

    # 认知状态
    csm_state: str = ""           # CognitiveState 枚举值
    l1_coordinates: str = ""
    phase: str = ""               # ResearchPhase 枚举值

    # Bandit 信号
    bandit_advice: str = "continue"  # continue/switch/requery
    effort_budget_remaining: int = 0

    # CompletionGate 信号
    gate_status: str = "pending"     # pass/block/gaps_hint/pending
    gate_should_stop: bool = False
    gate_reason: str = ""

    # 感知信号 (来自 SignalHub)
    active_signals: list[str] = field(default_factory=list)

    # 运行时上下文
    iteration: int = 0
    max_iterations: int = 0
    turns_count: int = 0
    tool_calls_count: int = 0
    last_errors: list[str] = field(default_factory=list)


@dataclass
class Decision:
    """统一决策结果 — 唯一决策出口."""

    action: str  # stop / continue / explore / switch_tool / requery
    reason: str = ""
    source: str = ""  # gate / bandit / arbiter / budget_exhausted
    confidence: float = 1.0

    @property
    def should_stop(self) -> bool:
        """是否应该停止."""
        return self.action == "stop"


class DecisionArbiter:
    """统一决策仲裁 — 替代各决策器各自为政.

    优先级:
        1. Gate pass + should_stop → stop (已完成)
        2. 预算耗尽 (iteration >= max) → stop (预算用完)
        3. Gate block + bandit continue → continue (未完成, 继续努力)
        4. Gate pass + bandit switch → switch_tool (完成但有更优工具)
        5. Gate gaps_hint + bandit continue → continue (有 gap, 继续)
        6. Gate pass + bandit continue → continue (正常继续)
        7. 冲突 (Gate pass + bandit requery) → 信任 Gate, continue
        8. 默认 → continue
    """

    def evaluate(self, ctx: DecisionContext) -> Decision:
        """返回最终决策."""
        # 1. Gate 判定完成 → stop
        if ctx.gate_status == "pass" and ctx.gate_should_stop:
            return Decision(
                action="stop",
                reason=f"CompletionGate passed: {ctx.gate_reason}",
                source="gate",
                confidence=0.95,
            )

        # 2. 预算耗尽 → stop
        if ctx.max_iterations > 0 and ctx.iteration >= ctx.max_iterations:
            return Decision(
                action="stop",
                reason=f"Budget exhausted: {ctx.iteration}/{ctx.max_iterations}",
                source="budget_exhausted",
                confidence=1.0,
            )

        # 3. Gate block → 继续努力 (不管 bandit 说什么)
        if ctx.gate_status == "block":
            if ctx.bandit_advice == "switch":
                return Decision(
                    action="switch_tool",
                    reason=f"Gate blocked ({ctx.gate_reason}), bandit suggests switch",
                    source="arbiter",
                    confidence=0.7,
                )
            return Decision(
                action="continue",
                reason=f"Gate blocked: {ctx.gate_reason}",
                source="gate",
                confidence=0.8,
            )

        # 4. Gate gaps_hint → 继续, gap 作为 hint
        if ctx.gate_status == "gaps_hint":
            return Decision(
                action="continue",
                reason=f"Gaps: {ctx.gate_reason}",
                source="gate",
                confidence=0.75,
            )

        # 5. Gate pass (无 should_stop) + bandit 建议
        if ctx.gate_status == "pass":
            if ctx.bandit_advice == "switch":
                return Decision(
                    action="switch_tool",
                    reason="Gate passed, bandit suggests switch",
                    source="bandit",
                    confidence=0.6,
                )
            if ctx.bandit_advice == "requery":
                # 冲突: Gate 说 pass 但 bandit 说 requery
                # 仲裁: 信任 Gate 的收敛判断, 但允许 requery
                return Decision(
                    action="requery",
                    reason="Gate passed but bandit suggests requery",
                    source="arbiter",
                    confidence=0.5,
                )
            # bandit continue + gate pass → 正常继续
            return Decision(
                action="continue",
                reason="Gate passed, bandit continue",
                source="arbiter",
                confidence=0.85,
            )

        # 6. Gate pending → 信任 bandit
        if ctx.bandit_advice == "switch":
            return Decision(
                action="switch_tool",
                reason="Bandit suggests switch (gate pending)",
                source="bandit",
                confidence=0.6,
            )
        if ctx.bandit_advice == "requery":
            return Decision(
                action="requery",
                reason="Bandit suggests requery (gate pending)",
                source="bandit",
                confidence=0.6,
            )

        # 7. 默认: continue
        return Decision(
            action="continue",
            reason="Default: continue",
            source="arbiter",
            confidence=0.5,
        )

    def build_context(
        self,
        csm_state: str = "",
        bandit: Any = None,
        gate_decision: Any = None,
        *,
        l1_coordinates: str = "",
        phase: str = "",
        active_signals: list[str] | None = None,
        iteration: int = 0,
        max_iterations: int = 0,
        turns_count: int = 0,
        tool_calls_count: int = 0,
        last_errors: list[str] | None = None,
    ) -> DecisionContext:
        """从各决策器收集状态, 构建统一上下文."""
        # Bandit
        bandit_advice = "continue"
        if bandit is not None:
            try:
                bandit_advice = bandit.policy()
            except Exception:
                logger.debug("bandit policy failed, default continue", exc_info=True)

        # Gate
        gate_status = "pending"
        gate_should_stop = False
        gate_reason = ""
        if gate_decision is not None:
            gate_status = getattr(gate_decision, "status", "pending")
            gate_should_stop = getattr(gate_decision, "should_stop", False)
            gate_reason = getattr(gate_decision, "reason", "")

        return DecisionContext(
            csm_state=csm_state,
            l1_coordinates=l1_coordinates,
            phase=phase,
            bandit_advice=bandit_advice,
            gate_status=gate_status,
            gate_should_stop=gate_should_stop,
            gate_reason=gate_reason,
            active_signals=active_signals or [],
            iteration=iteration,
            max_iterations=max_iterations,
            turns_count=turns_count,
            tool_calls_count=tool_calls_count,
            last_errors=last_errors or [],
        )


# ── 自检 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    arbiter = DecisionArbiter()

    # 1. Gate pass + should_stop → stop
    ctx = DecisionContext(gate_status="pass", gate_should_stop=True)
    d = arbiter.evaluate(ctx)
    assert d.action == "stop", f"gate pass+stop → stop, got {d.action}"

    # 2. 预算耗尽 → stop
    ctx = DecisionContext(iteration=10, max_iterations=10)
    d = arbiter.evaluate(ctx)
    assert d.action == "stop", f"budget exhausted → stop, got {d.action}"

    # 3. Gate block + bandit continue → continue
    ctx = DecisionContext(gate_status="block", gate_reason="effort low", bandit_advice="continue")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue", f"gate block → continue, got {d.action}"

    # 4. Gate block + bandit switch → switch_tool
    ctx = DecisionContext(gate_status="block", bandit_advice="switch")
    d = arbiter.evaluate(ctx)
    assert d.action == "switch_tool", f"gate block + switch → switch_tool, got {d.action}"

    # 5. Gate gaps_hint → continue
    ctx = DecisionContext(gate_status="gaps_hint", gate_reason="missing X")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue", f"gaps_hint → continue, got {d.action}"

    # 6. Gate pass (no stop) + bandit continue → continue
    ctx = DecisionContext(gate_status="pass", bandit_advice="continue")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue", f"pass+continue → continue, got {d.action}"

    # 7. Gate pass + bandit switch → switch_tool
    ctx = DecisionContext(gate_status="pass", bandit_advice="switch")
    d = arbiter.evaluate(ctx)
    assert d.action == "switch_tool", f"pass+switch → switch_tool, got {d.action}"

    # 8. Gate pending + bandit continue → continue
    ctx = DecisionContext(gate_status="pending", bandit_advice="continue")
    d = arbiter.evaluate(ctx)
    assert d.action == "continue", f"pending+continue → continue, got {d.action}"

    # 9. build_context
    class _MockBandit:
        def policy(self): return "switch"
    class _MockGate:
        status = "pass"
        should_stop = False
        reason = "ok"
    ctx = arbiter.build_context(
        csm_state="S4_CONSTRUCT",
        bandit=_MockBandit(),
        gate_decision=_MockGate(),
        iteration=3,
        max_iterations=10,
    )
    assert ctx.bandit_advice == "switch"
    assert ctx.gate_status == "pass"
    d = arbiter.evaluate(ctx)
    assert d.action == "switch_tool"

    print("DecisionArbiter self-checks passed")


__all__ = ["DecisionArbiter", "DecisionContext", "Decision"]
