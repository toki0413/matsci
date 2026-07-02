"""PlanStore 单元测试 — 持久化 / CRUD / 线程安全 / 损坏恢复."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from huginn.autoloop.plan_store import Plan, PlanStep, PlanStore


def _make_steps(n: int = 2) -> list[PlanStep]:
    return [
        PlanStep(
            id=f"s{i+1}",
            description=f"step {i+1}",
            tool="web_search" if i == 0 else None,
            parameters={"q": f"query{i+1}"} if i == 0 else {},
            dependencies=[] if i == 0 else [f"s{i}"],
        )
        for i in range(n)
    ]


# ── 创建 / 读取 ──────────────────────────────────────────────────────────────


class TestCreateAndGet:
    def test_create_plan_returns_id(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("objective", _make_steps())
        assert plan.id.startswith("plan_")
        assert plan.status == "draft"
        assert plan.created_at != ""
        assert len(plan.steps) == 2

    def test_get_plan_not_found(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        assert store.get_plan("plan_nope") is None

    def test_get_plan_returns_stored(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        fetched = store.get_plan(plan.id)
        assert fetched is not None
        assert fetched.id == plan.id
        assert fetched.objective == "obj"
        assert len(fetched.steps) == 2
        assert fetched.steps[0].id == "s1"
        assert fetched.steps[1].dependencies == ["s1"]


# ── 状态转换 ──────────────────────────────────────────────────────────────────


class TestStatusTransitions:
    def test_confirm_plan_draft_to_confirmed(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        confirmed = store.confirm_plan(plan.id)
        assert confirmed.status == "confirmed"
        assert confirmed.confirmed_at is not None
        assert confirmed.confirmed_at != ""

    def test_reject_plan_sets_reason(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        rejected = store.reject_plan(plan.id, reason="steps too vague")
        assert rejected.status == "abandoned"
        assert rejected.reject_reason == "steps too vague"

    def test_mark_executing(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        store.confirm_plan(plan.id)
        executing = store.mark_executing(plan.id)
        assert executing.status == "executing"

    def test_complete_plan_sets_completed_at(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        store.confirm_plan(plan.id)
        store.mark_executing(plan.id)
        completed = store.complete_plan(plan.id)
        assert completed.status == "completed"
        assert completed.completed_at is not None

    def test_fail_plan_records_reason(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        store.confirm_plan(plan.id)
        store.mark_executing(plan.id)
        failed = store.fail_plan(plan.id, reason="step s2 crashed")
        assert failed.status == "failed"
        assert failed.metadata.get("failure_reason") == "step s2 crashed"


# ── 步骤更新 ──────────────────────────────────────────────────────────────────


class TestUpdateStep:
    def test_update_step_status(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        updated = store.update_step(plan.id, "s1", status="done", result="found 3 papers")
        step = next(s for s in updated.steps if s.id == "s1")
        assert step.status == "done"
        assert step.result == "found 3 papers"

    def test_update_step_error(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        updated = store.update_step(
            plan.id, "s2", status="error", error="timeout"
        )
        step = next(s for s in updated.steps if s.id == "s2")
        assert step.status == "error"
        assert step.error == "timeout"

    def test_update_step_unknown_step_no_crash(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        # 不存在的 step_id 不报错, 计划原样返回
        updated = store.update_step(plan.id, "s99", status="done")
        assert updated.id == plan.id


# ── 列表 / 删除 ───────────────────────────────────────────────────────────────


class TestListAndDelete:
    def test_list_plans_all(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        store.create_plan("a", _make_steps())
        store.create_plan("b", _make_steps())
        assert len(store.list_plans()) == 2

    def test_list_plans_filter_by_status(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        p1 = store.create_plan("a", _make_steps())
        p2 = store.create_plan("b", _make_steps())
        store.confirm_plan(p2.id)
        drafts = store.list_plans(status="draft")
        confirmed = store.list_plans(status="confirmed")
        assert len(drafts) == 1
        assert drafts[0].id == p1.id
        assert len(confirmed) == 1
        assert confirmed[0].id == p2.id

    def test_delete_plan(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        plan = store.create_plan("obj", _make_steps())
        assert store.delete_plan(plan.id) is True
        assert store.get_plan(plan.id) is None
        assert store.delete_plan(plan.id) is False


# ── 持久化 ────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "plans.json"
        store1 = PlanStore(path=path)
        plan = store1.create_plan("survive restart", _make_steps())
        store1.confirm_plan(plan.id)

        store2 = PlanStore(path=path)
        fetched = store2.get_plan(plan.id)
        assert fetched is not None
        assert fetched.objective == "survive restart"
        assert fetched.status == "confirmed"
        assert len(fetched.steps) == 2
        assert fetched.steps[1].dependencies == ["s1"]

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "plans.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json {{{", encoding="utf-8")
        store = PlanStore(path=path)
        # 损坏文件不应崩溃, 从空开始
        assert store.list_plans() == []

    def test_save_writes_valid_json(self, tmp_path):
        path = tmp_path / "plans.json"
        store = PlanStore(path=path)
        store.create_plan("obj", _make_steps())
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "plans" in data
        assert len(data["plans"]) == 1


# ── 线程安全 ──────────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_creates_no_lost_writes(self, tmp_path):
        store = PlanStore(path=tmp_path / "plans.json")
        n_threads = 8
        n_per_thread = 5

        def worker(tid: int) -> None:
            for i in range(n_per_thread):
                store.create_plan(f"t{tid}-obj{i}", _make_steps(1))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(store.list_plans()) == n_threads * n_per_thread


# ── 默认路径 ──────────────────────────────────────────────────────────────────


class TestDefaultPath:
    def test_default_path_uses_cache_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
        path = PlanStore._default_path()
        assert path == tmp_path / "plans.json"

    def test_default_path_fallback(self, monkeypatch):
        monkeypatch.delenv("HUGINN_CACHE_DIR", raising=False)
        path = PlanStore._default_path()
        assert path == Path(".huginn") / "plans.json"


# ── 序列化 round-trip ────────────────────────────────────────────────────────


class TestSerialization:
    def test_plan_round_trip(self):
        plan = Plan(
            id="plan_test",
            objective="round trip",
            steps=_make_steps(3),
            status="confirmed",
            auto_confirm=True,
        )
        d = plan.to_dict()
        restored = Plan.from_dict(d)
        assert restored.id == plan.id
        assert restored.objective == "round trip"
        assert len(restored.steps) == 3
        assert restored.status == "confirmed"
        assert restored.auto_confirm is True

    def test_step_round_trip_with_all_fields(self):
        step = PlanStep(
            id="s1",
            description="full",
            tool="vasp_tool",
            parameters={"encut": 520},
            dependencies=[],
            agent_id="dft_expert",
            status="done",
            result="converged",
            error=None,
        )
        d = step.to_dict()
        restored = PlanStep.from_dict(d)
        assert restored.tool == "vasp_tool"
        assert restored.parameters == {"encut": 520}
        assert restored.agent_id == "dft_expert"
        assert restored.status == "done"
