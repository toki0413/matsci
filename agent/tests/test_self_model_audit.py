"""self_model.extract_step_result_from_audit: audit.jsonl → per-tool 调用记录.

brooks-sweep R1 修复 (抽取 _matches_step / _consume_tool_event 降嵌套) 的行为护栏:
call/result/error 配对、step 过滤、残缺 pending 计失败、时间戳缺失不伪造.
"""

from __future__ import annotations

import json

from huginn.metacog.self_model import extract_step_result_from_audit


def _write_audit(tmp_path, records):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


def test_missing_or_empty_audit_returns_empty_list(tmp_path):
    assert extract_step_result_from_audit(None, 0) == []
    assert extract_step_result_from_audit(tmp_path / "nope.jsonl", 0) == []


def test_pairs_call_with_result_and_measures_duration(tmp_path):
    path = _write_audit(tmp_path, [
        {"type": "tool.call", "step_id": 2, "data": {"tool": "bash_tool", "input": "ls"},
         "timestamp": 100.0},
        {"type": "tool.result", "step_id": 2, "data": {"tool": "bash_tool"}, "timestamp": 100.25},
    ])

    calls = extract_step_result_from_audit(path, 2)

    assert len(calls) == 1
    call = calls[0]
    assert call["tool_name"] == "bash_tool"
    assert call["success"] is True
    assert call["duration"] == 0.25
    assert "_call_ts" not in call, "内部字段必须清理掉"


def test_error_event_marks_failure_with_message(tmp_path):
    path = _write_audit(tmp_path, [
        {"type": "tool.call", "step_id": 0, "data": {"tool": "vasp_tool"}},
        {"type": "tool.error", "step_id": 0, "data": {"tool": "vasp_tool", "error": "boom"}},
    ])

    calls = extract_step_result_from_audit(path, 0)

    assert len(calls) == 1
    assert calls[0]["success"] is False
    assert calls[0]["error_message"] == "boom"
    assert calls[0]["duration"] == 0.0, "缺时间戳时不伪造 duration"


def test_orphan_call_counts_as_failure(tmp_path):
    path = _write_audit(tmp_path, [
        {"type": "tool.call", "step_id": 1, "data": {"tool": "python_tool"}},
    ])

    calls = extract_step_result_from_audit(path, 1)

    assert len(calls) == 1
    assert calls[0]["success"] is False
    assert calls[0]["error_message"] == "no result/error event"


def test_records_of_other_steps_are_skipped(tmp_path):
    path = _write_audit(tmp_path, [
        {"type": "tool.call", "step_id": 1, "data": {"tool": "a"}},
        {"type": "tool.result", "step_id": 1, "data": {"tool": "a"}},
        {"type": "tool.call", "step_id": 2, "data": {"tool": "b"}},
        {"type": "tool.result", "step_id": 2, "data": {"tool": "b"}},
    ])

    calls = extract_step_result_from_audit(path, 2)

    assert [c["tool_name"] for c in calls] == ["b"]


def test_records_without_step_field_are_all_kept(tmp_path):
    path = _write_audit(tmp_path, [
        {"type": "tool.call", "data": {"tool": "legacy"}},
        {"type": "tool.result", "data": {"tool": "legacy"}},
    ])

    calls = extract_step_result_from_audit(path, 7)

    assert len(calls) == 1
    assert calls[0]["success"] is True


def test_malformed_lines_are_ignored(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        'not json\n{"type": "tool.call", "step_id": 0, "data": {"tool": "x"}}\n'
        '{"type": "tool.result", "step_id": 0, "data": {"tool": "x"}}\n',
        encoding="utf-8",
    )

    calls = extract_step_result_from_audit(path, 0)

    assert len(calls) == 1
    assert calls[0]["tool_name"] == "x"
