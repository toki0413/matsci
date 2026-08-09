"""rcb_runner 主循环裁决判定函数测试.

Covers the pure decision functions that drive the RCB main loop (run/_step2/_step3):
- _should_retry_execute (Step3→Step2 回退触发, v14 拓扑许可)
- _derive_gap_type (从 critique dict 推断 gap 类型)
- _infer_beta_1_simple (β_1 简易推断)
- _write_directive_rejection (回退上限留痕)
- _recompute_report_metrics (G28 report↔outputs 数值交叉校验)

assert-based, runnable standalone or via pytest.
"""

from __future__ import annotations

import json
from pathlib import Path

from huginn.cli.rcb_runner import (
    _derive_gap_type,
    _infer_beta_1_simple,
    _recompute_report_metrics,
    _should_retry_execute,
    _write_directive_rejection,
)


# --- _should_retry_execute (Step3→Step2 回退触发) ---------------------------
def test_retry_trigger_pass_not_retried():
    # verdict=pass 永不回退
    assert _should_retry_execute("pass", 1, "numeric_recompute") is False


def test_retry_trigger_fix_needed_with_beta_and_gap():
    assert _should_retry_execute("fix_needed", 1, "numeric_recompute") is True
    assert _should_retry_execute("fail", 1, "exact_component_missing") is True


def test_retry_trigger_requires_beta():
    # β_1<=0 禁止回退 (拓扑无循环路径)
    assert _should_retry_execute("fix_needed", 0, "numeric_recompute") is False
    assert _should_retry_execute("fix_needed", -1, "numeric_recompute") is False


def test_retry_trigger_gap_type_whitelist():
    # text_description 不回退 (文字在 Step 3 内 OVERWRITE 即可)
    assert _should_retry_execute("fix_needed", 1, "text_description") is False
    assert _should_retry_execute("fix_needed", 1, "none") is False


# --- _derive_gap_type -------------------------------------------------------
def test_gap_type_numeric():
    assert _derive_gap_type({"recomputed_red_flags": [1]}) == "numeric_recompute"
    assert _derive_gap_type({"implausible_metrics": [1]}) == "numeric_recompute"


def test_gap_type_component():
    assert _derive_gap_type({"silent_substitutions": [1]}) == "exact_component_missing"
    assert _derive_gap_type({"missing_components": [1]}) == "exact_component_missing"


def test_gap_type_text_and_none():
    assert _derive_gap_type({"overall_verdict": "fix_needed"}) == "text_description"
    assert _derive_gap_type({}) == "none"
    assert _derive_gap_type(None) == "none"


# --- _infer_beta_1_simple ---------------------------------------------------
def test_beta1_missing_trace(tmp_path: Path):
    assert _infer_beta_1_simple(tmp_path) == 0


def test_beta1_short_trace(tmp_path: Path):
    trace = tmp_path / ".huginn" / "meta_trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("a\nb\n", encoding="utf-8")  # 2 行 < 3
    assert _infer_beta_1_simple(tmp_path) == 0


def test_beta1_reaches_threshold(tmp_path: Path):
    trace = tmp_path / ".huginn" / "meta_trace.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text("a\nb\nc\n", encoding="utf-8")  # 3 行 >= 3
    assert _infer_beta_1_simple(tmp_path) == 1


# --- _write_directive_rejection ---------------------------------------------
def test_directive_rejection_writes_entry(tmp_path: Path):
    _write_directive_rejection(tmp_path, "numeric_recompute", "fix_needed", 2)
    rej = tmp_path / ".huginn" / "directive_rejections.jsonl"
    assert rej.exists()
    entry = json.loads(rej.read_text(encoding="utf-8").strip().split("\n")[0])
    assert entry["reason"] == "step3_retry_limit_reached"
    assert entry["retry_count"] == 2
    assert entry["final_verdict"] == "fix_needed"
    assert entry["gap_type"] == "numeric_recompute"
    assert isinstance(entry["ts"], (int, float))


def test_directive_rejection_appends(tmp_path: Path):
    _write_directive_rejection(tmp_path, "a", "fail", 1)
    _write_directive_rejection(tmp_path, "b", "fix_needed", 2)
    rej = tmp_path / ".huginn" / "directive_rejections.jsonl"
    lines = [l for l in rej.read_text(encoding="utf-8").strip().split("\n") if l]
    assert len(lines) == 2


# --- _recompute_report_metrics (G28) ----------------------------------------
def test_recompute_no_metrics(tmp_path: Path):
    assert _recompute_report_metrics("no numbers here", tmp_path) == []


def test_recompute_matches_ok(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "r2.json").write_text('{"R2": 0.79}', encoding="utf-8")
    flags = _recompute_report_metrics("R2=0.78", tmp_path)
    # 0.78 vs 0.79 → dev ~1.3% < 10% → 无 flag
    assert flags == []


def test_recompute_flags_high_deviation(tmp_path: Path):
    out = tmp_path / "outputs"
    out.mkdir()
    (out / "mae.txt").write_text("MAE=0.1", encoding="utf-8")
    flags = _recompute_report_metrics("MAE=0.5", tmp_path)
    # 0.5 vs 0.1 → dev 400% > 10% → flag
    assert len(flags) == 1
    assert flags[0]["metric"] == "MAE"
    assert flags[0]["deviation_pct"] > 10


def test_recompute_skips_missing_outputs(tmp_path: Path):
    # outputs/ 不存在 → 无法交叉校验 → 不 flag
    assert _recompute_report_metrics("R2=0.99", tmp_path) == []


if __name__ == "__main__":
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        test_retry_trigger_pass_not_retried()
        test_retry_trigger_fix_needed_with_beta_and_gap()
        test_retry_trigger_requires_beta()
        test_retry_trigger_gap_type_whitelist()
        test_gap_type_numeric()
        test_gap_type_component()
        test_gap_type_text_and_none()
        test_beta1_missing_trace(base)
        test_beta1_short_trace(base)
        test_beta1_reaches_threshold(base)
        test_directive_rejection_writes_entry(base)
        test_directive_rejection_appends(base)
        test_recompute_no_metrics(base)
        test_recompute_matches_ok(base)
        test_recompute_flags_high_deviation(base)
        test_recompute_skips_missing_outputs(base)
    print("rcb_runner main-loop decision tests OK")
    sys.exit(0)
