"""completion_evidence 单元测试: 对象级取证 + 显式豁免决策档 + 门禁不变量.

覆盖三条沉淀 (AICC/OpenAI4S 启发):
  1) attach_evidence / make_evidence — 数值绑定来源论文的强引用证据标签
  2) build_waivers / DecisionLedger — 补后仍缺的自由度落显式豁免决策档 (哈希链)
  3) assess_gate / critical_missing — pass 禁带未豁免的关键缺度
"""

from __future__ import annotations

from pathlib import Path

from huginn.tools.literature.completion_evidence import (
    DecisionLedger,
    assess_gate,
    attach_evidence,
    build_waivers,
    critical_missing,
    make_evidence,
    sha256_text,
)


def test_make_evidence_fid_and_sha256_and_snippet():
    ev = make_evidence("10.9/bb.comp", "  DFT gives 5.80 eV.  ", index=1)
    assert ev["fid"] == "10.9/bb.comp#v1"
    assert ev["source_id"] == "10.9/bb.comp"
    # 原文取到 → sha256 非空; snippet 去空白截断
    assert len(ev["sha256"]) == 64
    assert ev["sha256"] == sha256_text("DFT gives 5.80 eV.")
    assert ev["snippet"] == "DFT gives 5.80 eV."


def test_make_evidence_no_text_marks_honest():
    # 拿不到原文 → sha256 空串, 不伪造（"provenance 错比缺更糟"）
    ev = make_evidence("10.9/x", "", index=1)
    assert ev["sha256"] == ""
    assert ev["snippet"] == ""


def test_attach_evidence_binds_to_source_paper():
    reported = [
        {"value": 5.80, "unit": "eV", "doi": "10.9/bb.comp", "source_paper": "Li2O comp"},
    ]
    papers = [
        {"doi": "10.9/bb.comp", "title": "Li2O comp", "abstract": "measured 5.80 eV"},
        {"doi": "10.9/other", "title": "noise", "abstract": "irrelevant"},
    ]
    out = attach_evidence(reported, papers)
    assert len(out) == 1
    ev = out[0]["evidence"]
    assert ev["source_id"] == "10.9/bb.comp"
    assert ev["sha256"] == sha256_text("measured 5.80 eV")
    # 原 reported 字段保留
    assert out[0]["value"] == 5.80


def test_attach_evidence_missing_source_marks_honest():
    reported = [{"value": 1.0, "doi": "10.9/unknown"}]
    out = attach_evidence(reported, [])
    # 无 papers 可匹配 → 原文未取到, sha256 空串
    assert out[0]["evidence"]["sha256"] == ""
    assert out[0]["evidence"]["fid"] == "10.9/unknown#v1"


def test_build_waivers_is_idempotent_and_self_referencing():
    a = build_waivers(["temperature", "method_family"], reason_var="sys=X,prop=g")
    b = build_waivers(["method_family", "temperature"], reason_var="sys=X,prop=g")
    # 同输入 → 同决策 id (幂等), 排序无关
    assert [w["dim"] for w in a] == ["method_family", "temperature"]
    assert a == b
    done = {w["decision_id"] for w in a}
    assert len(done) == 2  # 每条豁免唯一决策 id
    for w in a:
        assert w["kind"] == "waiver"
        assert w["decision"] == "waived"
        assert w["version"] == 1


def test_decision_ledger_append_idempotent_and_verify(tmp_path):
    ledger = DecisionLedger(tmp_path / "decisions.jsonl")
    w = build_waivers(["temperature"], reason_var="r")[0]
    did = ledger.append(w)
    # 幂等: 同 id 不重复追加
    ledger.append(w)
    ok, problems = ledger.verify()
    assert ok, problems
    assert not problems
    assert did.startswith("w-")


def test_decision_ledger_verify_detects_tamper(tmp_path):
    path = tmp_path / "decisions.jsonl"
    ledger = DecisionLedger(path)
    ledger.append(build_waivers(["temperature"])[0])
    # 篡改已落盘的记录 → 哈希链断
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"dim"', '"dmm"'), encoding="utf-8")
    ok, problems = DecisionLedger(path).verify()
    assert not ok
    assert any("hash mismatch" in p for p in problems)


def test_decision_ledger_verify_detects_broken_chain_and_duplicate(tmp_path):
    path = tmp_path / "decisions.jsonl"
    ledger = DecisionLedger(path)
    ledger.append(build_waivers(["temperature"])[0])
    ledger.append(build_waivers(["functional"])[0])
    lines = path.read_text(encoding="utf-8").splitlines()
    # 独立实例重读 (排除内存状态), 再篡改第二条的 _prev_hash → 断链
    assert string_verify(path)
    import json
    rec2 = json.loads(lines[1])
    rec2["_prev_hash"] = "f" * 64
    rec2["decision_id"] = "w-broken-chain"
    path.write_text(lines[0] + "\n" + json.dumps(rec2, sort_keys=True) + "\n", encoding="utf-8")
    ok, problems = DecisionLedger(path).verify()
    assert not ok
    assert any("broken chain" in p for p in problems)
    assert any("duplicate decision_id" not in p for p in problems)  # 不误报重复


def string_verify(path: Path):
    """独立 ledger 实例验链 (避免测试依赖原实例内存缓存)."""
    return (DecisionLedger(path).verify()[0])


def test_critical_missing_filters_by_urgency():
    urgency = {"temperature": 3, "functional": 2, "method_family": 1}
    assert critical_missing(
        ["method_family", "functional", "temperature"], urgency
    ) == ["functional", "temperature"]  # urgency>=2


def test_assess_gate_pass_without_blocking():
    g = assess_gate([], [], verdict="all_compatible")
    assert g["status"] == "pass"
    assert g["issues"] == []


def test_assess_gate_needs_waiver_when_uncovered_blocking():
    g = assess_gate(["temperature"], [], verdict="mixed")
    assert g["status"] == "needs_waiver"
    assert "temperature" in g["blocking_dims"]
    assert g["issues"]


def test_assess_gate_pass_with_waiver():
    g = assess_gate(["temperature"], ["temperature"], verdict="mixed")
    assert g["status"] == "pass_with_waiver"
    assert g["issues"] == []
    assert g["waived_dims"] == ["temperature"]


def test_assess_gate_partial_waiver_still_needs_waiver():
    # 只豁免了 functional, 温度仍缺 → 仍需 waiver
    g = assess_gate(["temperature", "functional"], ["functional"], verdict="mixed")
    assert g["status"] == "needs_waiver"
    assert g["blocking_dims"] == ["temperature"]
