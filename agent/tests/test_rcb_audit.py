"""rcb/audit.py 评测裁决纯函数族测试.

Covers the pure adjudication functions migrated out of rcb_runner.py:
- _rcb_drift_check (v16 window-2 drift)
- _extract_exact_components / _scan_implementation_traces / _parse_substitute_headers
- _count_failed_attempts
- _step2_substitution_audit (A3 silent substitution interception)
- _scan_real_metrics (A2 product gate)
- _lint_report_markers (B4 numeric marker lint)
- _step2_outputs_gate (A2 blocker remediation)

assert-based, runnable standalone or via pytest.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from huginn.cli.rcb.audit import (
    _count_failed_attempts,
    _extract_exact_components,
    _lint_report_markers,
    _parse_substitute_headers,
    _rcb_drift_check,
    _scan_implementation_traces,
    _scan_real_metrics,
    _step2_outputs_gate,
    _step2_substitution_audit,
)


def _run(coro):
    return asyncio.run(coro)


# --- drift check ------------------------------------------------------------
def test_drift_check_short_history():
    assert _rcb_drift_check([]) == (False, "")
    assert _rcb_drift_check([SimpleNamespace(on_track="true")]) == (False, "")


def test_drift_check_fires_on_two_unsure_low_evidence():
    h = [
        SimpleNamespace(on_track="unsure", evidence_quality="low"),
        SimpleNamespace(on_track="false", evidence_quality="low"),
    ]
    fired, msg = _rcb_drift_check(h)
    assert fired is True
    assert "RCB drift" in msg


def test_drift_check_no_fire_when_evidence_ok():
    h = [
        SimpleNamespace(on_track="unsure", evidence_quality="high"),
        SimpleNamespace(on_track="false", evidence_quality="high"),
    ]
    assert _rcb_drift_check(h) == (False, "")


# --- exact components / traces / substitute headers -------------------------
def test_extract_exact_components():
    # 正则按 ; 或换行分界, "." 不截断 — 用 ; 分隔才是正确边界
    text = "[EXACT] GVAE encoder; [EXACT] C2ST classifier; [EXACT] gvae encoder(dup)"
    comps = _extract_exact_components(text)
    assert "GVAE encoder" in comps
    assert "C2ST classifier" in comps
    # dedupe case-insensitive
    assert (
        len([c for c in comps if "GVAE" in c])
        == 1
        == len([c for c in comps if "gvae" in c])
    )


def test_extract_exact_components_empty():
    assert _extract_exact_components("") == []
    assert _extract_exact_components("no markers here") == []


def test_scan_implementation_traces(tmp_path: Path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "model.py").write_text(
        "# GVAE encoder implementation\nclass Model:\n    pass\n", encoding="utf-8"
    )
    (tmp_path / ".huginn").mkdir()
    # .huginn/ 内是观测而非产物, 应被跳过
    (tmp_path / ".huginn" / "trace.jsonl").write_text("GVAE encoder", encoding="utf-8")
    traces = _scan_implementation_traces(tmp_path, ["GVAE encoder", "C2ST classifier"])
    assert traces.get("GVAE encoder") is True
    assert traces.get("C2ST classifier") is False


def test_scan_implementation_traces_empty_components(tmp_path: Path):
    assert _scan_implementation_traces(tmp_path, []) == {}


def test_parse_substitute_headers(tmp_path: Path):
    report = tmp_path / "report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        "METHOD SUBSTITUTE: GVAE replaced Heuristic because no data\n"
        "Some body text\n",
        encoding="utf-8",
    )
    subs = _parse_substitute_headers(report)
    assert len(subs) == 1
    assert subs[0]["replaced"] == "GVAE"
    assert "no data" in subs[0]["reason"]


def test_parse_substitute_headers_missing(tmp_path: Path):
    assert _parse_substitute_headers(tmp_path / "nope.md") == []


# --- count failed attempts --------------------------------------------------
def test_count_failed_attempts(tmp_path: Path):
    evals = [SimpleNamespace(on_track="false", attempted="tried GVAE encoder")]
    assert _count_failed_attempts(tmp_path, evals, "GVAE encoder") == 1


def test_count_failed_attempts_from_trace(tmp_path: Path):
    trace_dir = tmp_path / ".huginn"
    trace_dir.mkdir()
    (trace_dir / "meta_trace.jsonl").write_text(
        json.dumps({"on_track": "false", "attempted": "GVAE encoder attempt"}) + "\n",
        encoding="utf-8",
    )
    assert _count_failed_attempts(tmp_path, [], "GVAE encoder") == 1


# --- substitution audit (A3) ------------------------------------------------
def test_substitution_audit_no_components(tmp_path: Path):
    async def _chat(*_a, **_k):
        raise AssertionError("should not chat with no components")

    r = _run(_step2_substitution_audit(tmp_path, "no exact markers", [], _chat))
    assert r["exact_components"] == []
    assert r["missing"] == []


def test_substitution_audit_blocks_silent_substitution(tmp_path: Path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "impl.py").write_text("print(1)", encoding="utf-8")
    checklist = "[EXACT] GVAE encoder; [EXACT] RealThing"
    called = []

    async def _chat(prompt, ctx="x"):
        called.append(prompt)
        # simulate remediation: write a trace file for the missing component
        (tmp_path / "code" / "real.py").write_text(
            "class RealThing: pass", encoding="utf-8"
        )

    r = _run(_step2_substitution_audit(tmp_path, checklist, [], _chat))
    # GVAE missing silently -> blocked & remediated; RealThing not present initially
    assert "GVAE encoder" in [m["component"] for m in r["missing"]]
    assert len(called) >= 1
    assert "SUBSTITUTION AUDIT FAILED" in called[0]


# --- real metrics scan / marker lint (A2, B4) -------------------------------
def test_scan_real_metrics(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "r2.json").write_text('{"r2": 0.79}', encoding="utf-8")
    (out / "placeholder.txt").write_text("expected todo", encoding="utf-8")
    (out / "empty.csv").write_text("", encoding="utf-8")
    files = _scan_real_metrics(tmp_path)
    names = {p.name for p in files}
    assert "r2.json" in names
    assert "placeholder.txt" not in names
    assert "empty.csv" not in names


def test_scan_real_metrics_missing_dir(tmp_path: Path):
    assert _scan_real_metrics(tmp_path) == []


def test_lint_report_markers(tmp_path: Path):
    report = tmp_path / "report.md"
    report.write_text(
        "R2=0.79 [EXECUTED]\n"
        "loss 0.3 [EXPECTED]\n"
        "acc 0.85 (no marker)\n"
        "year 2024 context\n",
        encoding="utf-8",
    )
    r = _lint_report_markers(report)
    assert r["total_numbers"] >= 3
    assert r["untagged"] >= 1
    assert r["marker_counts"]["[EXECUTED]"] == 1
    assert r["marker_counts"]["[EXPECTED]"] == 1


def test_lint_report_markers_missing(tmp_path: Path):
    r = _lint_report_markers(tmp_path / "nope.md")
    assert r["total_numbers"] == 0


# --- outputs gate (A2 blocker) ----------------------------------------------
def test_outputs_gate_passes_with_metrics(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "m.json").write_text('{"x": 1}', encoding="utf-8")

    async def _chat(*_a, **_k):
        raise AssertionError("should not remediate when metrics exist")

    r = _run(_step2_outputs_gate(tmp_path, _chat))
    assert r["has_real_metrics"] is True
    assert r["blocker"] is False


def test_outputs_gate_triggers_remediation(tmp_path: Path):
    async def _chat(prompt, ctx="x"):
        out = tmp_path / "outputs"
        out.mkdir(exist_ok=True)
        (out / "m.json").write_text('{"loss": 0.5}', encoding="utf-8")

    r = _run(_step2_outputs_gate(tmp_path, _chat))
    assert r["remediated"] is True
    assert r["blocker"] is False


def test_outputs_gate_blocker_when_no_remediation(tmp_path: Path):
    async def _chat(*_a, **_k):
        pass  # does nothing

    r = _run(_step2_outputs_gate(tmp_path, _chat))
    assert r["blocker"] is True
    assert r["has_real_metrics"] is False


if __name__ == "__main__":
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        test_drift_check_fires_on_two_unsure_low_evidence()
        test_extract_exact_components()
        test_scan_implementation_traces(base)
        test_parse_substitute_headers(base)
        test_count_failed_attempts(base)
        test_substitution_audit_no_components(base)
        test_scan_real_metrics(base)
        test_lint_report_markers(base)
        test_outputs_gate_passes_with_metrics(base)
        test_outputs_gate_triggers_remediation(base)
        test_outputs_gate_blocker_when_no_remediation(base)
    print("rcb/audit tests OK")
    sys.exit(0)
