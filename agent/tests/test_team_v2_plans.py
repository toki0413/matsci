"""Tests for /team/v2/plans endpoints — PlanStore lifecycle over HTTP."""

from __future__ import annotations

import pytest

pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

from fastapi.testclient import TestClient

from huginn.autoloop.plan_store import PlanStep
from huginn.server import app
from huginn.server_core import get_plan_store

client = TestClient(app)


def _make_plan(objective: str = "test objective", n_steps: int = 2):
    """直接往 PlanStore 塞一个 plan, 绕过 LLM. 测试完记得删."""
    steps = [
        PlanStep(id=f"s{i+1}", description=f"step {i+1}", agent_id="lead")
        for i in range(n_steps)
    ]
    store = get_plan_store()
    plan = store.create_plan(objective, steps)
    return plan.id


def _cleanup(*plan_ids: str) -> None:
    store = get_plan_store()
    for pid in plan_ids:
        try:
            store.delete_plan(pid)
        except Exception:
            pass


class TestListAndGet:
    """只读 endpoint, 不需要 LLM."""

    def test_list_plans_returns_list(self):
        r = client.get("/team/v2/plans")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert isinstance(data["plans"], list)

    def test_list_plans_with_status_filter(self):
        r = client.get("/team/v2/plans?status=draft")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert isinstance(data["plans"], list)
        for p in data["plans"]:
            assert p["status"] == "draft"

    def test_get_plan_not_found(self):
        r = client.get("/team/v2/plans/nonexistent-plan-id")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "not found" in data["error"]

    def test_get_plan_existing(self):
        pid = _make_plan("get me")
        try:
            r = client.get(f"/team/v2/plans/{pid}")
            assert r.status_code == 200
            data = r.json()
            assert data["success"] is True
            assert data["plan"]["id"] == pid
            assert data["plan"]["objective"] == "get me"
            assert data["plan"]["status"] == "draft"
        finally:
            _cleanup(pid)


class TestConfirmReject:
    """draft → confirmed / abandoned, 不需要 LLM."""

    def test_confirm_plan(self):
        pid = _make_plan("confirm me")
        try:
            r = client.post(f"/team/v2/plans/{pid}/confirm")
            assert r.status_code == 200
            data = r.json()
            assert data["success"] is True
            assert data["plan"]["status"] == "confirmed"
            assert data["plan"]["confirmed_at"] is not None
        finally:
            _cleanup(pid)

    def test_confirm_plan_not_found(self):
        r = client.post("/team/v2/plans/no-such-plan/confirm")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False

    def test_reject_plan_with_reason(self):
        pid = _make_plan("reject me")
        try:
            r = client.post(f"/team/v2/plans/{pid}/reject", json={"reason": "bad plan"})
            assert r.status_code == 200
            data = r.json()
            assert data["success"] is True
            assert data["plan"]["status"] == "abandoned"
            assert data["plan"]["reject_reason"] == "bad plan"
        finally:
            _cleanup(pid)

    def test_reject_plan_no_body(self):
        pid = _make_plan("reject no body")
        try:
            r = client.post(f"/team/v2/plans/{pid}/reject")
            assert r.status_code == 200
            data = r.json()
            assert data["success"] is True
            assert data["plan"]["status"] == "abandoned"
        finally:
            _cleanup(pid)


class TestDelete:
    def test_delete_plan(self):
        pid = _make_plan("delete me")
        r = client.delete(f"/team/v2/plans/{pid}")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["deleted"] == pid
        # 确认真的删了
        r2 = client.get(f"/team/v2/plans/{pid}")
        assert r2.json()["success"] is False

    def test_delete_plan_not_found(self):
        r = client.delete("/team/v2/plans/no-such-plan")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False


class TestCreateEndpoint:
    """POST /team/v2/plans 走 orchestrator.plan() → LLM.

    测试环境没配 LLM 时会返回 error, 这里只检查 endpoint 不崩.
    """

    def test_create_missing_objective(self):
        r = client.post("/team/v2/plans", json={})
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "objective" in data["error"]

    def test_create_returns_success_or_llm_error(self):
        r = client.post("/team/v2/plans", json={"objective": "test create"})
        assert r.status_code == 200
        data = r.json()
        # 没配 LLM 时 success=False, 配了的话 success=True + plan
        assert "success" in data
        if data["success"]:
            assert "plan" in data
            _cleanup(data["plan"]["id"])


class TestExecuteEndpoint:
    """POST /team/v2/plans/{id}/execute 走 orchestrator.execute() → LLM.

    测试环境没配 LLM, 执行会失败, 但 endpoint 本身不该崩.
    """

    def test_execute_not_found(self):
        r = client.post("/team/v2/plans/no-such-plan/execute")
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False

    def test_execute_draft_rejected(self):
        """draft 状态的计划不能执行, orchestrator 会拒绝."""
        pid = _make_plan("draft exec")
        try:
            r = client.post(f"/team/v2/plans/{pid}/execute")
            assert r.status_code == 200
            data = r.json()
            assert data["success"] is False
            assert "confirmed" in data["error"]
        finally:
            _cleanup(pid)

    def test_execute_confirmed_returns_result_or_error(self):
        """confirmed 计划可以执行, 但没 LLM 会失败. 检查 endpoint 不崩."""
        pid = _make_plan("confirmed exec")
        try:
            store = get_plan_store()
            store.confirm_plan(pid)
            r = client.post(f"/team/v2/plans/{pid}/execute")
            assert r.status_code == 200
            data = r.json()
            assert "success" in data
            # 没配 LLM 时 success=False, 配了的话检查 result
            if data["success"]:
                assert "result" in data
            else:
                assert "error" in data
        finally:
            _cleanup(pid)


class TestFullLifecycle:
    """完整生命周期: 创建(手动) → 确认 → 执行(会失败) → 查状态."""

    def test_draft_confirm_execute_flow(self):
        pid = _make_plan("lifecycle test", n_steps=3)
        try:
            # 1. 初始状态 draft
            r = client.get(f"/team/v2/plans/{pid}")
            assert r.json()["plan"]["status"] == "draft"

            # 2. 确认
            r = client.post(f"/team/v2/plans/{pid}/confirm")
            assert r.json()["plan"]["status"] == "confirmed"

            # 3. 执行 (没 LLM 会失败, 但 plan 状态应该变成 failed)
            r = client.post(f"/team/v2/plans/{pid}/execute")
            assert r.status_code == 200

            # 4. 检查最终状态
            r = client.get(f"/team/v2/plans/{pid}")
            final_status = r.json()["plan"]["status"]
            # 没配 LLM 时执行失败 → failed; 配了 LLM 时 → completed
            assert final_status in ("failed", "completed")
        finally:
            _cleanup(pid)
