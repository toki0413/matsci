"""统一门禁链测试 — 验证 CoEffectRegistry 驱动的门禁激活与决策聚合.

覆盖四个适配器 (completion / sanity / phase_gate / adoption) 的激活推导、
归一化与 all_pass 聚合, 以及依赖不满足时门禁自动停用 (skip).
"""
from __future__ import annotations

from huginn.security.gate import (
    GateChain,
    adoption_adapter,
    completion_adapter,
    phase_gate_adapter,
    sanity_adapter,
)


class _FakeGate:
    """模拟 CompletionGate.review(goal, ctx) -> decision(status, reason)."""

    def __init__(self, status: str = "pass", reason: str = ""):
        self._status = status
        self._reason = reason

    def review(self, goal, ctx):
        return type("D", (), {"status": self._status, "reason": self._reason})()


class _FakePhaseGate:
    """模拟 PhaseGateHook.evaluate(from, to, evidence) -> PhaseGate(is_blocked)."""

    def __init__(self, blocked: bool = False, feedback: str = ""):
        self._blocked = blocked
        self._feedback = feedback

    def evaluate(self, from_phase, to_phase, evidence):
        return type(
            "PG",
            (),
            {"is_blocked": self._blocked, "feedback": self._feedback},
        )()


class _FakeAdoption:
    """模拟 AdoptionGate.decide(config_id) -> AdoptionDecision(status, adopt)."""

    def __init__(self, status: str = "green", adopt: bool = True):
        self._status = status
        self._adopt = adopt

    def decide(self, config_id):
        return type(
            "AD",
            (),
            {"status": self._status, "adopt": self._adopt, "reason": "fake"},
        )()


def _fake_sanity(passed: bool = True, reason: str = ""):
    def checker(ctx):
        return {"passed": passed, "reason": reason}

    return checker


def test_gate_already_registered_requires_ok():
    chain = GateChain()
    chain.register(completion_adapter(_FakeGate("pass"), requires={"criteria"}))
    chain.set_available("criteria", True)
    results = chain.evaluate_all((None, None))
    assert results[0].status == "pass"


def test_gate_skipped_when_prerequisite_missing():
    chain = GateChain()
    chain.register(completion_adapter(_FakeGate("block"), requires={"evidence"}))
    # 不 set_available("evidence", True) → 依赖不满足 → skip
    results = chain.evaluate_all((None, None))
    assert results[0].status == "skip"
    assert GateChain.all_pass(results) is True  # skip 视为通过


def test_block_blocks_aggregation():
    chain = GateChain()
    chain.register(completion_adapter(_FakeGate("block"), requires=set()))
    results = chain.evaluate_all((None, None))
    assert results[0].status == "block"
    assert GateChain.all_pass(results) is False


def test_phase_gate_adapter_blocked():
    chain = GateChain()
    chain.register(
        phase_gate_adapter(
            _FakePhaseGate(blocked=True, feedback="missing evidence"),
            "plan",
            "execute",
            requires=set(),
        )
    )
    results = chain.evaluate_all({"mode": "x"})
    assert results[0].status == "block"
    assert "missing evidence" in results[0].reason


def test_phase_gate_adapter_pass():
    chain = GateChain()
    chain.register(
        phase_gate_adapter(
            _FakePhaseGate(blocked=False),
            "validate",
            "learn",
            requires={"math"},
        )
    )
    chain.set_available("math", True)
    results = chain.evaluate_all({})
    assert results[0].status == "pass"


def test_adoption_adapter_green_passes_red_blocks():
    chain = GateChain()
    chain.register(adoption_adapter(_FakeAdoption("green", True), "cfg", requires=set()))
    assert chain.evaluate_all(None)[0].status == "pass"
    chain2 = GateChain()
    chain2.register(adoption_adapter(_FakeAdoption("red", False), "cfg", requires=set()))
    assert chain2.evaluate_all(None)[0].status == "block"


def test_sanity_adapter():
    chain = GateChain()
    chain.register(sanity_adapter(_fake_sanity(passed=False, reason="dedup"), requires=set()))
    results = chain.evaluate_all("workspace")
    assert results[0].status == "block"
    assert results[0].reason == "dedup"


def test_mixed_chain_all_pass_requires_all_gates_pass():
    chain = GateChain()
    chain.register(completion_adapter(_FakeGate("pass"), requires=set()))
    chain.register(sanity_adapter(_fake_sanity(passed=True), requires=set()))
    chain.register(
        phase_gate_adapter(_FakePhaseGate(blocked=False), "p", "e", requires=set())
    )
    results = chain.evaluate_all((None, None))
    assert GateChain.all_pass(results) is True

    chain2 = GateChain()
    chain2.register(completion_adapter(_FakeGate("pass"), requires=set()))
    chain2.register(sanity_adapter(_fake_sanity(passed=False), requires=set()))
    results2 = chain2.evaluate_all((None, None))
    assert GateChain.all_pass(results2) is False
