"""执行观测层单测: exit_class 粗粒度分类 + 自演探针 (纯只读)."""

from __future__ import annotations

from huginn.autoloop.exec_observation import (
    EXIT_CLASSES,
    classify_exit_class,
    detect_soliloquy,
    exit_class_counts,
)
from huginn.autoloop.engine_act import EngineAct


# ── exit_class 分类 ────────────────────────────────────────────────────────


def test_evidence_present_is_ok_even_if_negative_result():
    # 跑完但实验自身判定负结果 (success=False) 仍是 ok —— 不能记成工具故障.
    out = {"mode": "code_lab", "success": False, "objectives": {"gap": 1.2}}
    assert classify_exit_class(out) == "ok"


def test_timeout_marker_wins():
    assert classify_exit_class({"error": "执行超时, 算力预算耗尽"}) == "timeout"


def test_empty_code_is_tool_error_not_env():
    out = {"mode": "code_lab", "success": False, "error": "书生未产出可解析的实验代码"}
    assert classify_exit_class(out) == "tool_error"


def test_missing_module_without_evidence_is_env_error():
    out = "Traceback ... ModuleNotFoundError: No module named 'scipy'"
    assert classify_exit_class(out) == "env_error"


def test_isolation_failure_is_env_error():
    assert classify_exit_class({"error": "隔离执行无结果(退出码 1)"}) == "env_error"


def test_no_signal_is_empty():
    assert classify_exit_class("") == "empty"
    assert classify_exit_class({"success": True}) == "empty"


def test_timeout_beats_evidence_and_error_ordering():
    # 带证据 + 超时字样 → 仍判 timeout (优先级最高).
    out = {"objectives": {"x": 1.0}, "error": "timeout"}
    assert classify_exit_class(out) == "timeout"
    assert set(EXIT_CLASSES) == {"ok", "empty", "tool_error", "timeout", "env_error"}


# ── 台账聚合 ───────────────────────────────────────────────────────────────


def test_exit_class_counts_and_legacy_fallback():
    ledger = [
        {"exit_class": "ok", "result": "{}"},
        {"exit_class": "timeout", "result": "{}"},
        # 老台账缺 exit_class → 现场按 result 判定 (含 error → tool_error)
        {"result": '{"error": "boom"}'},
    ]
    counts = exit_class_counts(ledger)
    assert counts["ok"] == 1
    assert counts["timeout"] == 1
    assert counts["tool_error"] == 1
    assert counts["env_error"] == 0


# ── 自演探针 ───────────────────────────────────────────────────────────────


def test_soliloquy_flagged_when_claimed_with_no_ok_receipt():
    ledger = [{"exit_class": "timeout", "result": "{}"}]
    got = detect_soliloquy("本实验执行后得到如下结果", ledger)
    # 超时不构成"观测到了" → 声称执行却无成功回执, 判为疑似自演.
    assert got["claimed"] is True
    assert got["receipts"] == 0
    assert got["flagged"] is True


def test_soliloquy_not_flagged_with_ok_receipt():
    ledger = [{"exit_class": "ok", "result": '{"objectives": {"a": 1}}'}]
    got = detect_soliloquy("我们执行了实验并观测到结果", ledger)
    assert got["receipts"] == 1
    assert got["flagged"] is False


def test_soliloquy_not_flagged_without_claim():
    got = detect_soliloquy("本报告仅讨论方法论, 未涉数值", [])
    assert got["claimed"] is False
    assert got["flagged"] is False


def test_soliloquy_on_empty_narrative_is_safe():
    got = detect_soliloquy("", [])
    assert got["flagged"] is False


# ── 接入: 台账写入带 exit_class (只加字段, 不改控制流) ────────────────────────


class _Stub:
    """最小替身: 只有 _append_execution_ledger 需要的字段."""

    def __init__(self) -> None:
        self._execution_ledger: list[dict] = []
        self._current_hyp_id_for_plan = "h1"


def test_append_execution_ledger_stamps_exit_class():
    stub = _Stub()
    EngineAct._append_execution_ledger(stub, "code_lab", {"objectives": {"x": 3.0}})
    entry = stub._execution_ledger[0]
    assert entry["exit_class"] == "ok"
    assert entry["intent"] == "h1"

    EngineAct._append_execution_ledger(stub, "code_lab", {"error": "执行超时"})
    assert stub._execution_ledger[1]["exit_class"] == "timeout"