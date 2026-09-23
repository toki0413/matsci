"""Tests for the task board / expert-panel canvas projection.

The board is a read-model over raw session events (``tool_call`` /
``tool_result``), so every test drives a real ``SessionEventLog`` and asserts
the derived graph — no mocks of the projection itself.
"""

from __future__ import annotations

import json

from huginn.agents.board import build_board
from huginn.events.session_log import (
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    SessionEventLog,
)


def _log(tmp_path, name: str = "s1") -> SessionEventLog:
    return SessionEventLog(name, tmp_path / f"{name}.jsonl", load=False)


def _call(log, tid, args, name="subagent_tool"):
    log.append(EVENT_TOOL_CALL, {"tool_call_id": tid, "name": name, "args": args})


def _result(log, tid, payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    log.append(EVENT_TOOL_RESULT, {"tool_call_id": tid, "content": content})


class TestEmpty:
    def test_no_events_yields_empty_board(self, tmp_path):
        board = build_board("s1", log=_log(tmp_path))
        assert board["exists"] is True
        assert board["nodes"] == []
        assert board["edges"] == []
        assert board["stats"]["total"] == 0

    def test_unreadable_log_is_fail_open(self, monkeypatch):
        # Opening the session log blows up → empty board, never raises.
        def _boom(cls, *a, **k):
            raise OSError("boom")

        monkeypatch.setattr(SessionEventLog, "open", classmethod(_boom))
        board = build_board("s1", log=None)
        assert board["exists"] is False
        assert board["nodes"] == []

    def test_non_dispatch_tools_ignored(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "run"}, name="bash_tool")
        _call(log, "t2", {"action": "list_types"}, name="subagent_tool")
        board = build_board("s1", log=log)
        assert board["nodes"] == []


class TestSingleDispatch:
    def test_running_when_no_result(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "dispatch", "spec_name": "explore", "task": "find X"})
        board = build_board("s1", log=log)
        assert len(board["nodes"]) == 1
        node = board["nodes"][0]
        assert node["expert"] == "explore"
        assert node["task"] == "find X"
        assert node["status"] == "running"
        assert node["success"] is None

    def test_done_after_success_result(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "dispatch", "spec_name": "coder", "task": "write Y"})
        _result(log, "t1", {"success": True, "summary": "wrote Y"})
        board = build_board("s1", log=log)
        node = board["nodes"][0]
        assert node["status"] == "done"
        assert node["success"] is True
        assert node["summary"] == "wrote Y"
        assert board["stats"] == {"total": 1, "done": 1, "failed": 0, "running": 0}

    def test_failed_after_error_result(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "dispatch", "spec_name": "analyst", "task": "analyze"})
        _result(log, "t1", {"success": False, "error": "OOM"})
        board = build_board("s1", log=log)
        node = board["nodes"][0]
        assert node["status"] == "failed"
        assert node["success"] is False
        assert node["summary"] == "OOM"

    def test_non_json_result_falls_back_to_done(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "dispatch", "spec_name": "explore", "task": "q"})
        _result(log, "t1", "plain prose summary, not json")
        board = build_board("s1", log=log)
        node = board["nodes"][0]
        assert node["status"] == "done"
        assert node["success"] is True
        assert "plain prose" in node["summary"]


class TestParallelDag:
    def test_layers_edges_and_width(self, tmp_path):
        log = _log(tmp_path)
        tasks = [
            {"spec_name": "explore", "task": "A"},
            {"spec_name": "coder", "task": "B"},
            {"spec_name": "analyst", "task": "C"},
        ]
        deps = [["explore:A", "coder:B"], ["explore:A", "analyst:C"]]
        _call(log, "t1", {"action": "dispatch_parallel", "tasks": tasks, "dependencies": deps})
        board = build_board("s1", log=log)

        assert len(board["nodes"]) == 3
        # A first, then B and C in parallel.
        assert board["layers"] == [["explore:A"], ["coder:B", "analyst:C"]]
        assert board["width"] == 2
        assert len(board["edges"]) == 2
        edge_pairs = {(e["from"], e["to"]) for e in board["edges"]}
        assert edge_pairs == {("explore:A", "coder:B"), ("explore:A", "analyst:C")}
        # Layer numbers stamped onto nodes.
        layer_of = {n["id"]: n["layer"] for n in board["nodes"]}
        assert layer_of["explore:A"] == 0
        assert layer_of["coder:B"] == 1

    def test_parallel_results_matched_by_index(self, tmp_path):
        log = _log(tmp_path)
        tasks = [{"spec_name": "explore", "task": "A"}, {"spec_name": "coder", "task": "B"}]
        _call(log, "t1", {"action": "dispatch_parallel", "tasks": tasks})
        _result(log, "t1", {
            "results": [
                {"success": True, "summary": "A ok"},
                {"success": False, "error": "B bad"},
            ]
        })
        board = build_board("s1", log=log)
        by_id = {n["id"]: n for n in board["nodes"]}
        assert by_id["explore:A"]["status"] == "done"
        assert by_id["coder:B"]["status"] == "failed"
        assert board["stats"]["done"] == 1
        assert board["stats"]["failed"] == 1

    def test_unknown_dep_reference_degrades_gracefully(self, tmp_path):
        log = _log(tmp_path)
        tasks = [{"spec_name": "explore", "task": "A"}]
        _call(log, "t1", {
            "action": "dispatch_parallel",
            "tasks": tasks,
            "dependencies": [["ghost:X", "explore:A"]],
        })
        board = build_board("s1", log=log)
        # Ghost edge filtered out; single node still lands in one layer.
        assert board["edges"] == []
        assert board["layers"] == [["explore:A"]]

    def test_duplicate_task_ids_get_unique_suffix(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "dispatch", "spec_name": "explore", "task": "same task"})
        _call(log, "t2", {"action": "dispatch", "spec_name": "explore", "task": "same task"})
        board = build_board("s1", log=log)
        ids = [n["id"] for n in board["nodes"]]
        assert len(ids) == 2
        assert len(set(ids)) == 2


class TestExperts:
    def test_idle_experts_from_roster_are_included(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "dispatch", "spec_name": "explore", "task": "q"})
        _result(log, "t1", {"success": True, "summary": "done"})
        roster = [
            {"name": "explore", "description": "read-only search"},
            {"name": "coder", "description": "writes code"},
        ]
        board = build_board("s1", log=log, expert_roster=roster)
        lanes = {e["name"]: e for e in board["experts"]}
        assert set(lanes) == {"explore", "coder"}
        assert lanes["explore"]["total"] == 1
        assert lanes["explore"]["done"] == 1
        assert lanes["explore"]["description"] == "read-only search"
        # Idle expert: no dispatches but still on the canvas.
        assert lanes["coder"]["total"] == 0

    def test_expert_lanes_counts(self, tmp_path):
        log = _log(tmp_path)
        _call(log, "t1", {"action": "dispatch", "spec_name": "explore", "task": "A"})
        _call(log, "t2", {"action": "dispatch", "spec_name": "explore", "task": "B"})
        _result(log, "t1", {"success": True, "summary": "ok"})
        _result(log, "t2", {"success": False, "error": "nope"})
        board = build_board("s1", log=log)
        lane = next(e for e in board["experts"] if e["name"] == "explore")
        assert lane["total"] == 2
        assert lane["done"] == 1
        assert lane["failed"] == 1
        assert lane["running"] == 0
