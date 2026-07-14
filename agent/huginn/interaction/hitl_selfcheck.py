"""Sanity check for human-in-the-loop features."""
from huginn.agent.code_act_loop import (
    _assess_risk,
    _mark_auto_approved,
    _is_auto_approved,
    reset_auto_approvals,
    _compute_risk_threshold,
    _apply_dynamic_threshold,
    _trust_scores,
    set_suggest_mode,
    _is_suggest_mode,
    _suggest_modes,
    resume_suggest,
    _active_agents,
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


def test_dynamic_risk_threshold():
    # high trust + no errors → lenient threshold → medium 降级为 low
    _trust_scores["test_hri"] = 0.85
    t = _compute_risk_threshold("test_hri", error_streak=0)
    assert t > 0.7, f"expected lenient threshold, got {t}"
    new_risk, reason = _apply_dynamic_threshold("medium", t)
    assert new_risk == "low", f"expected medium→low, got {new_risk}"
    assert "lenient" in reason

    # low trust → strict threshold → medium 升级为 high
    _trust_scores["test_hri"] = 0.15
    t = _compute_risk_threshold("test_hri", error_streak=0)
    assert t < 0.3, f"expected strict threshold, got {t}"
    new_risk, reason = _apply_dynamic_threshold("medium", t)
    assert new_risk == "high", f"expected medium→high, got {new_risk}"
    assert "strict" in reason

    # errors 拉低阈值: trust=0.5 + 4 errors → threshold ~0.3
    _trust_scores["test_hri"] = 0.5
    t = _compute_risk_threshold("test_hri", error_streak=4)
    assert t <= 0.3, f"expected errors to pull threshold down, got {t}"

    # low/high 不受阈值影响
    _trust_scores["test_hri"] = 0.9
    t = _compute_risk_threshold("test_hri", 0)
    r, _ = _apply_dynamic_threshold("low", t)
    assert r == "low"
    r, _ = _apply_dynamic_threshold("high", t)
    assert r == "high"

    del _trust_scores["test_hri"]


def test_suggest_mode():
    _suggest_modes.pop("test_hri", None)
    assert not _is_suggest_mode("test_hri")
    set_suggest_mode("test_hri", True)
    assert _is_suggest_mode("test_hri")
    set_suggest_mode("test_hri", False)
    assert not _is_suggest_mode("test_hri")

    # resume_suggest 没有 agent 注册时返回 False
    _active_agents.pop("test_hri", None)
    assert resume_suggest("test_hri", "approve") is False


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
    test_dynamic_risk_threshold()
    print("PASS: dynamic_risk_threshold")
    test_suggest_mode()
    print("PASS: suggest_mode")
    print("ALL CHECKS PASSED")
