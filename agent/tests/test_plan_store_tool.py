"""PlanStoreTool 单元测试 — 所有 action 的行为 + 只读判定 + 错误路径."""

from __future__ import annotations

import asyncio

import pytest

from huginn.autoloop.plan_store import PlanStore
from huginn.tools.plan_store_tool import PlanStoreInput, PlanStoreTool


def _make_tool(tmp_path) -> PlanStoreTool:
    store = PlanStore(path=tmp_path / "plans.json")
    return PlanStoreTool(store=store)


def _propose(tool: PlanStoreTool, objective: str = "obj", steps: list | None = None) -> dict:
    if steps is None:
        steps = [
            {"id": "s1", "description": "第一步", "tool": "web_search", "parameters": {}},
            {"id": "s2", "description": "第二步", "dependencies": ["s1"]},
        ]
    result = asyncio.run(
        tool.call({"action": "propose", "objective": objective, "steps": steps})
    )
    assert result.success, result.error
    return result.data


# ── propose ───────────────────────────────────────────────────────────────────


class TestPropose:
    def test_propose_returns_plan_id_draft(self, tmp_path):
        tool = _make_tool(tmp_path)
        data = _propose(tool)
        assert data["plan_id"].startswith("plan_")
        assert data["status"] == "draft"
        assert len(data["steps"]) == 2

    def test_propose_auto_confirm(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = asyncio.run(
            tool.call({
                "action": "propose",
                "objective": "auto",
                "steps": [{"id": "s1", "description": "x"}],
                "auto_confirm": True,
            })
        )
        assert result.success
        assert result.data["status"] == "confirmed"

    def test_propose_missing_objective_fails_validation(self, tmp_path):
        tool = _make_tool(tmp_path)
        vr = asyncio.run(
            tool.validate_input(PlanStoreInput(action="propose", steps=[{"id": "s1"}]))
        )
        assert not vr.result
        assert "objective" in vr.message

    def test_propose_missing_steps_fails_validation(self, tmp_path):
        tool = _make_tool(tmp_path)
        vr = asyncio.run(
            tool.validate_input(PlanStoreInput(action="propose", objective="o"))
        )
        assert not vr.result
        assert "steps" in vr.message


# ── confirm / reject ──────────────────────────────────────────────────────────


class TestConfirmReject:
    def test_confirm_draft_to_confirmed(self, tmp_path):
        tool = _make_tool(tmp_path)
        data = _propose(tool)
        result = asyncio.run(
            tool.call({"action": "confirm", "plan_id": data["plan_id"]})
        )
        assert result.success
        assert result.data["status"] == "confirmed"

    def test_confirm_missing_plan(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = asyncio.run(
            tool.call({"action": "confirm", "plan_id": "plan_nope"})
        )
        assert not result.success
        assert "不存在" in result.error

    def test_reject_with_reason(self, tmp_path):
        tool = _make_tool(tmp_path)
        data = _propose(tool)
        result = asyncio.run(
            tool.call({
                "action": "reject",
                "plan_id": data["plan_id"],
                "reject_reason": "too vague",
            })
        )
        assert result.success
        assert result.data["status"] == "abandoned"
        assert result.data["reject_reason"] == "too vague"


# ── status / list_pending / list_all / get ────────────────────────────────────


class TestReadActions:
    def test_status_empty(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = asyncio.run(tool.call({"action": "status"}))
        assert result.success
        assert result.data["total"] == 0

    def test_status_with_plans(self, tmp_path):
        tool = _make_tool(tmp_path)
        _propose(tool, "a")
        _propose(tool, "b")
        result = asyncio.run(tool.call({"action": "status"}))
        assert result.data["total"] == 2
        assert result.data["by_status"].get("draft") == 2
        assert len(result.data["pending"]) == 2

    def test_list_pending(self, tmp_path):
        tool = _make_tool(tmp_path)
        _propose(tool, "draft1")
        result = asyncio.run(tool.call({"action": "list_pending"}))
        assert len(result.data["pending"]) == 1
        assert result.data["pending"][0]["objective"] == "draft1"

    def test_list_all(self, tmp_path):
        tool = _make_tool(tmp_path)
        _propose(tool, "a")
        _propose(tool, "b")
        result = asyncio.run(tool.call({"action": "list_all"}))
        assert len(result.data["plans"]) == 2

    def test_get_existing(self, tmp_path):
        tool = _make_tool(tmp_path)
        data = _propose(tool, "fetch me")
        result = asyncio.run(
            tool.call({"action": "get", "plan_id": data["plan_id"]})
        )
        assert result.success
        assert result.data["objective"] == "fetch me"
        assert len(result.data["steps"]) == 2

    def test_get_missing_plan(self, tmp_path):
        tool = _make_tool(tmp_path)
        result = asyncio.run(
            tool.call({"action": "get", "plan_id": "plan_nope"})
        )
        assert not result.success


# ── advance_step ──────────────────────────────────────────────────────────────


class TestAdvanceStep:
    def test_advance_step_status(self, tmp_path):
        tool = _make_tool(tmp_path)
        data = _propose(tool)
        result = asyncio.run(
            tool.call({
                "action": "advance_step",
                "plan_id": data["plan_id"],
                "step_id": "s1",
                "step_status": "done",
                "step_result": "ok",
            })
        )
        assert result.success
        assert result.data["status"] == "done"
        assert result.data["result"] == "ok"

    def test_advance_step_error(self, tmp_path):
        tool = _make_tool(tmp_path)
        data = _propose(tool)
        result = asyncio.run(
            tool.call({
                "action": "advance_step",
                "plan_id": data["plan_id"],
                "step_id": "s2",
                "step_status": "error",
                "step_error": "boom",
            })
        )
        assert result.success
        assert result.data["status"] == "error"
        assert result.data["error"] == "boom"

    def test_advance_step_unknown_step(self, tmp_path):
        tool = _make_tool(tmp_path)
        data = _propose(tool)
        result = asyncio.run(
            tool.call({
                "action": "advance_step",
                "plan_id": data["plan_id"],
                "step_id": "s99",
                "step_status": "done",
            })
        )
        assert not result.success

    def test_advance_step_missing_plan_id_validation(self, tmp_path):
        tool = _make_tool(tmp_path)
        vr = asyncio.run(
            tool.validate_input(
                PlanStoreInput(action="advance_step", step_id="s1")
            )
        )
        assert not vr.result
        assert "plan_id" in vr.message


# ── is_read_only ──────────────────────────────────────────────────────────────


class TestReadOnly:
    @pytest.mark.parametrize(
        "action,expected",
        [
            ("status", True),
            ("list_pending", True),
            ("list_all", True),
            ("get", True),
            ("propose", False),
            ("confirm", False),
            ("reject", False),
            ("advance_step", False),
        ],
    )
    def test_read_only_classification(self, action, expected):
        tool = PlanStoreTool()
        args = PlanStoreInput(action=action, plan_id="plan_x", step_id="s1")
        assert tool.is_read_only(args) is expected
