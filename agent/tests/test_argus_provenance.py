"""ARGUS influence provenance MVP 测试.

覆盖 5 项改动:
  * MVP 1: 消息打标 — user_input (memory/manager) / external_content (provenance_enhancer)
  * MVP 2: PhaseGate 降级 feedback (含 external_content 时 approved + 降级提示)
  * MVP 3: RedTeamFinding 加 source_class 字段
  * MVP 4: _classify_failure 加 prompt_injection_suspect

adapter._serialize 的 _source_class 标记是 closure, 留给集成测试覆盖.
"""

from __future__ import annotations

from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.phase_gate import PhaseGateHook, _has_external_source
from huginn.autoloop.red_team import RedTeamFinding
from huginn.memory.manager import MemoryManager
from huginn.memory.session import SessionContext
from huginn.rag.provenance_enhancer import enhance_rag_results


# ── MVP 1: 消息打标 ───────────────────────────────────────────────


def test_add_message_user_marks_source_class():
    """user 消息入口自动标 source_class=user_input."""
    mgr = MemoryManager(session=SessionContext())
    mgr.add_message("user", "帮我算带隙")
    msg = mgr.session.messages[-1]
    assert msg.metadata.get("source_class") == "user_input"


def test_add_message_assistant_no_source_class():
    """assistant/system 消息不自动标 source_class (留给 tool/hook 自己标)."""
    mgr = MemoryManager(session=SessionContext())
    mgr.add_message("assistant", "好的, 我来算")
    msg = mgr.session.messages[-1]
    assert "source_class" not in msg.metadata


def test_enhance_rag_results_marks_external_content():
    """RAG 增强结果统一标 source_class=external_content."""
    results = [
        {"text": "TiO2 band gap is 3.2 eV", "score": 0.9},
        {"text": "anatase phase stable at room temp", "score": 0.8},
    ]
    enhanced = enhance_rag_results("band gap", results)
    assert len(enhanced) == 2
    for item in enhanced:
        assert item["source_class"] == "external_content"
        # 不动原始 dict
        assert "score" in item


# ── MVP 2: PhaseGate 降级 feedback ───────────────────────────────


def test_phase_gate_feedback_external_content():
    """evidence 含 external_content 时 status=approved 但 feedback 加降级提示."""
    hook = PhaseGateHook()
    evidence = {
        "summary": "TiO2 带隙 3.2 eV",
        "validation_result": {"source_class": "external_content", "value": "3.2 eV"},
    }
    # learn→report required_evidence 为空, 直接走 approved 路径
    gate = hook.evaluate("learn", "report", evidence)
    assert gate.status == "approved"  # 不阻断
    assert gate.feedback != ""
    assert "ARGUS" in gate.feedback
    assert "external_content" in gate.feedback
    assert gate.reviewer == "argus_provenance"


def test_phase_gate_no_feedback_when_no_external():
    """evidence 不含 external_content 时 feedback 为空."""
    hook = PhaseGateHook()
    evidence = {
        "summary": "TiO2 带隙 3.2 eV",
        "validation_result": {"source_class": "tool_output", "value": "3.2 eV"},
    }
    gate = hook.evaluate("learn", "report", evidence)
    assert gate.status == "approved"
    assert gate.feedback == ""
    assert gate.reviewer is None


def test_has_external_source_nested():
    """_has_external_source 递归扫描嵌套结构."""
    # 顶层 dict
    assert _has_external_source({"source_class": "external_content"})
    # 嵌套 dict
    assert _has_external_source({
        "a": {"b": {"source_class": "external_content"}}
    })
    # list 里的 dict
    assert _has_external_source({
        "results": [{"source_class": "external_content"}]
    })
    # 不含
    assert not _has_external_source({"source_class": "tool_output"})
    assert not _has_external_source({"hypothesis": "test"})
    assert not _has_external_source("plain string")
    assert not _has_external_source(42)


# ── MVP 3: RedTeamFinding source_class 字段 ───────────────────────


def test_redteam_finding_source_class_default():
    """RedTeamFinding 默认 source_class 为空串 (未声明)."""
    f = RedTeamFinding(
        category="methodology_gap",
        description="未控制温度",
        severity="high",
    )
    assert f.source_class == ""


def test_redteam_finding_source_class_to_dict():
    """to_dict 包含 source_class 字段."""
    f = RedTeamFinding(
        category="confounder",
        description="气压未控制",
        severity="medium",
        source_class="external_content",
    )
    d = f.to_dict()
    assert d["source_class"] == "external_content"
    assert d["category"] == "confounder"


# ── MVP 4: _classify_failure prompt_injection_suspect ─────────────


def test_classify_failure_prompt_injection_suspect():
    """validation 含 external_content + 失败 → prompt_injection_suspect."""
    validation = {
        "errors": "result mismatch",
        "result": {"source_class": "external_content", "value": "wrong"},
    }
    failure_type = AutoloopEngine._classify_failure(validation)
    assert failure_type == "prompt_injection_suspect"


def test_classify_failure_tool_error_takes_priority():
    """tool_error 优先级高于 prompt_injection_suspect (技术故障与来源无关)."""
    validation = {
        "errors": "tool timed out after 30s",
        "result": {"source_class": "external_content"},
    }
    failure_type = AutoloopEngine._classify_failure(validation)
    assert failure_type == "tool_error"


def test_classify_failure_no_external_returns_normal():
    """validation 不含 external_content → 走原有分类逻辑."""
    validation = {
        "errors": "invalid parameter: shape mismatch",
        "result": {"source_class": "tool_output"},
    }
    failure_type = AutoloopEngine._classify_failure(validation)
    assert failure_type == "param_error"


def test_classify_failure_nested_external():
    """validation 深层嵌套 external_content 也能识别."""
    validation = {
        "errors": "test failed",
        "result": {
            "data": {
                "sources": [
                    {"source_class": "external_content", "text": "malicious"}
                ]
            }
        },
    }
    failure_type = AutoloopEngine._classify_failure(validation)
    assert failure_type == "prompt_injection_suspect"


# ── 升级 1: depth 50 ──────────────────────────────────────────────


def test_has_external_source_deep_nesting():
    """depth 50 支持深嵌套 (原来 5 会漏)."""
    # 构造 10 层嵌套, depth=5 会漏, depth=50 能找到
    obj = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": {"j": {
        "source_class": "external_content"
    }}}}}}}}}}}
    assert _has_external_source(obj)


# ── 升级 2: DS 数值化 confidence ──────────────────────────────────


def test_argus_confidence_external_content():
    """external_content dominant → confidence 低 (< 0.5)."""
    from huginn.autoloop.phase_gate import _argus_confidence
    evidence = {"result": {"source_class": "external_content"}}
    confidence, dominant = _argus_confidence(evidence)
    assert dominant == "external_content"
    assert confidence < 0.5  # external_content 先验 m_pass=0.3


def test_argus_confidence_user_input():
    """user_input dominant → confidence 高 (> 0.5)."""
    from huginn.autoloop.phase_gate import _argus_confidence
    evidence = {"result": {"source_class": "user_input"}}
    confidence, dominant = _argus_confidence(evidence)
    assert dominant == "user_input"
    assert confidence > 0.5  # user_input 先验 m_pass=0.7


def test_argus_confidence_mixed_sources():
    """混合来源 → DS 合成, confidence 介于两者之间."""
    from huginn.autoloop.phase_gate import _argus_confidence
    evidence = {
        "a": {"source_class": "user_input"},
        "b": {"source_class": "external_content"},
    }
    confidence, dominant = _argus_confidence(evidence)
    # 2 个 source, dominant 取第一个出现 (Counter 同计数时)
    assert dominant in ("user_input", "external_content")
    # DS 合成: (0.7,0.05,0.25) + (0.3,0.4,0.3) → m_pass 应该在 0.4-0.7 之间
    assert 0.4 < confidence < 0.75


def test_argus_confidence_no_source_class():
    """无 source_class → confidence=1.0, dominant 空串."""
    from huginn.autoloop.phase_gate import _argus_confidence
    confidence, dominant = _argus_confidence({"hypothesis": "test"})
    assert confidence == 1.0
    assert dominant == ""


def test_argus_feedback_external_shows_confidence():
    """external_content dominant → feedback 含 DS confidence 数值."""
    from huginn.autoloop.phase_gate import _argus_feedback
    feedback = _argus_feedback({"result": {"source_class": "external_content"}})
    assert "DS confidence=" in feedback
    assert "external_content" in feedback


def test_argus_feedback_tool_output_silent():
    """tool_output dominant + confidence>=0.5 → 不加 feedback (噪声抑制)."""
    from huginn.autoloop.phase_gate import _argus_feedback
    feedback = _argus_feedback({"result": {"source_class": "tool_output"}})
    assert feedback == ""


# ── 升级 3: RedTeamFinding 从 evidence 自动派生 ──────────────────


def test_redteam_finding_effective_source_class():
    """source_class 空串 → effective_source_class 兜底为 agent_generated."""
    f = RedTeamFinding(
        category="methodology_gap",
        description="test",
        severity="low",
    )
    assert f.source_class == ""
    assert f.effective_source_class == "agent_generated"
    assert f.to_dict()["source_class"] == "agent_generated"


def test_redteam_finding_explicit_source_class_preserved():
    """显式标了 source_class → effective 返回显式值."""
    f = RedTeamFinding(
        category="confounder",
        description="test",
        severity="high",
        source_class="external_content",
    )
    assert f.effective_source_class == "external_content"
    assert f.to_dict()["source_class"] == "external_content"


def test_redteam_llm_findings_auto_derive_source_class():
    """LLM 没标 source_class → 从 evidence dominant 自动填."""
    from unittest.mock import MagicMock
    from huginn.autoloop.red_team import RedTeamReviewer

    # mock LLM 返回 2 条 finding, 都没标 source_class
    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = (
        '[{"category":"methodology_gap","description":"d1","severity":"high"},'
        '{"category":"confounder","description":"d2","severity":"medium"}]'
    )
    mock_model.invoke.return_value = mock_resp
    mock_model._mock_name = "test"  # 跳过 _is_real_model

    reviewer = RedTeamReviewer(model=mock_model)
    evidence = {
        "result": {"source_class": "external_content", "value": "test"},
    }
    findings = reviewer._llm_findings("validate", "learn", evidence)
    assert len(findings) == 2
    for f in findings:
        # LLM 没标, 自动从 evidence dominant 派生
        assert f.source_class == "external_content"
        assert f.effective_source_class == "external_content"


# ── 升级 1: _collect_source_classes ──────────────────────────────


def test_collect_source_classes():
    """_collect_source_classes 收集所有 source_class 值."""
    from huginn.autoloop.phase_gate import _collect_source_classes
    obj = {
        "a": {"source_class": "user_input"},
        "b": [{"source_class": "tool_output"}, {"source_class": "external_content"}],
        "c": "no source",
    }
    classes = _collect_source_classes(obj)
    assert sorted(classes) == ["external_content", "tool_output", "user_input"]
