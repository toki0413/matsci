"""Sanity check for human-in-the-loop features."""
from huginn.agent.code_act_loop import (
    _assess_risk,
    _mark_auto_approved,
    _is_auto_approved,
    reset_auto_approvals,
)
from huginn.memory.intuition import detect_intuition
from huginn.interaction.clarification import should_ask_clarification
from huginn.autoloop.phase_gate import PhaseGateState


class _FakeTool:
    destructive = False


class _DestTool:
    destructive = True


def test_risk_assessment():
    tools = {"rag_tool": _FakeTool(), "delete_tool": _DestTool()}
    assert _assess_risk("x = 1 + 2\nprint(x)", tools)[0] == "low"
    assert _assess_risk('result = rag_tool(query="GaN")', tools)[0] == "medium"
    assert _assess_risk('delete_tool(doc_ids=["x"])', tools)[0] == "high"
    assert _assess_risk('plt.savefig("out.png")', tools)[0] == "high"


def test_auto_approve():
    reset_auto_approvals("test")
    assert not _is_auto_approved("test", "medium")
    _mark_auto_approved("test", "medium")
    assert _is_auto_approved("test", "medium")
    assert not _is_auto_approved("test", "high")


def test_intuition_detection():
    sig = detect_intuition("这个体系的能带就像石墨烯的狄拉克锥")
    assert sig is not None
    assert sig["kind"] == "cross_domain_analogy"
    assert "analogy" in sig

    sig2 = detect_intuition("直觉是 ENCUT 应该设高一点")
    assert sig2 is not None
    assert sig2["kind"] == "intuition"

    assert detect_intuition("请计算 GaN 的能带") is None


def test_clarification_trigger():
    c = should_ask_clarification("这就像拓扑绝缘体的边缘态")
    assert c is not None
    assert c["reason"] == "cross_domain_analogy"

    c2 = should_ask_clarification("GaN 的性质")
    assert c2 is not None
    assert c2["reason"] == "no_action_verb"

    assert should_ask_clarification("计算 GaN 的能带结构") is None

    c4 = should_ask_clarification("帮我查一下 TiO2", session_history=[])
    assert c4 is not None
    assert c4["reason"] == "new_material"
    assert c4["material"] == "TiO2"


def test_phase_gate_checkpoint():
    state = PhaseGateState()
    state.human_checkpoint_phases.add(("plan", "execute"))
    assert state.needs_human_checkpoint("plan", "execute") is True
    assert state.needs_human_checkpoint("hypothesize", "plan") is False
    state.overrides.add(("plan", "execute"))
    assert state.needs_human_checkpoint("plan", "execute") is False


if __name__ == "__main__":
    test_risk_assessment()
    print("PASS: risk_assessment")
    test_auto_approve()
    print("PASS: auto_approve")
    test_intuition_detection()
    print("PASS: intuition_detection")
    test_clarification_trigger()
    print("PASS: clarification_trigger")
    test_phase_gate_checkpoint()
    print("PASS: phase_gate_checkpoint")
    print("ALL CHECKS PASSED")
