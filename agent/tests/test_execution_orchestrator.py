"""Unit tests for huginn/execution/orchestrator.py."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import BaseModel

from huginn.execution.orchestrator import ExecutionOrchestrator, StageResult


class _SimpleModel(BaseModel):
    value: int


def _sync_dummy(action, x=0):
    return {"x": x}


async def _async_dummy(action, x=0):
    return {"x": x}


class TestExecutionOrchestratorRun:
    def test_run_simple_sync_tool(self, tmp_path):
        orch = ExecutionOrchestrator(
            working_dir=str(tmp_path),
            tool_registry={"dummy": _sync_dummy},
        )
        record = asyncio.run(
            orch.run(
                [{"id": "s1", "tool": "dummy", "action": "go", "params": {"x": 7}}],
                workflow_name="wf1",
            )
        )
        assert record.overall_success is True
        assert len(record.stage_results) == 1
        assert record.stage_results[0].output_data["x"] == 7

    def test_run_simple_async_tool(self, tmp_path):
        orch = ExecutionOrchestrator(
            working_dir=str(tmp_path),
            tool_registry={"dummy": _async_dummy},
        )
        record = asyncio.run(
            orch.run(
                [{"id": "s1", "tool": "dummy", "action": "go", "params": {"x": 3}}],
            )
        )
        assert record.overall_success is True
        assert record.stage_results[0].output_data["x"] == 3

    def test_run_dependent_stages(self, tmp_path):
        orch = ExecutionOrchestrator(
            working_dir=str(tmp_path),
            tool_registry={"dummy": _sync_dummy},
        )
        stages = [
            {"id": "s1", "tool": "dummy", "action": "go", "params": {"x": 1}},
            {
                "id": "s2",
                "tool": "dummy",
                "action": "go",
                "params": {"x": "${s1.x}"},
                "depends_on": ["s1"],
            },
        ]
        record = asyncio.run(orch.run(stages, workflow_name="dep"))
        assert record.overall_success is True
        s2 = next(r for r in record.stage_results if r.stage_id == "s2")
        assert s2.output_data["x"] == 1

    def test_missing_tool(self, tmp_path):
        orch = ExecutionOrchestrator(working_dir=str(tmp_path))
        record = asyncio.run(
            orch.run([{"id": "s1", "tool": "missing", "action": "go"}])
        )
        assert record.overall_success is False
        assert "not found" in record.stage_results[0].error_message


class TestAutofix:
    def test_attempt_autofix_and_retry(self, tmp_path):
        calls = {"count": 0}

        def vasp_tool(action, **params):
            calls["count"] += 1
            if "__auto_fix" in params:
                return {"converged": True}
            raise RuntimeError("ZBRENT: fatal error in bracketing")

        orch = ExecutionOrchestrator(
            working_dir=str(tmp_path),
            tool_registry={"vasp_tool": vasp_tool},
            enable_autofix=True,
            max_retries=2,
        )
        record = asyncio.run(
            orch.run(
                [
                    {
                        "id": "relax",
                        "tool": "vasp_tool",
                        "action": "relax",
                        "params": {"ALGO": "Fast"},
                    }
                ]
            )
        )
        assert record.overall_success is True
        result = record.stage_results[0]
        assert result.auto_fixed is True
        assert result.retry_count == 1
        assert calls["count"] == 2


class TestHelpers:
    def test_resolve_param_refs(self):
        orch = ExecutionOrchestrator()
        previous = {
            "s1": StageResult(
                "s1", "", "", True, output_data={"x": 9, "nested": {"y": 2}}
            )
        }
        params = {
            "a": "${s1.x}",
            "b": "${s1}",
            "c": "${missing.x}",
            "d": 4,
        }
        resolved = orch._resolve_param_refs(params, previous)
        assert resolved["a"] == 9
        assert resolved["b"] == {"x": 9, "nested": {"y": 2}}
        assert resolved["c"] == "${missing.x}"
        assert resolved["d"] == 4

    def test_build_dependency_graph(self):
        orch = ExecutionOrchestrator()
        stages = [
            {"id": "a", "depends_on": []},
            {"id": "b", "depends_on": "a"},
            {"id": "c", "depends_on": ["a", "b"]},
        ]
        graph = orch._build_dependency_graph(stages)
        assert graph == {"a": [], "b": ["a"], "c": ["a", "b"]}

    def test_serialize_output_dict(self):
        orch = ExecutionOrchestrator()
        assert orch._serialize_output({"x": 1}) == {"x": 1}

    def test_serialize_output_pydantic(self):
        orch = ExecutionOrchestrator()
        assert orch._serialize_output(_SimpleModel(value=5)) == {"value": 5}

    def test_serialize_output_object(self):
        class Obj:
            def __init__(self):
                self.x = 10

        orch = ExecutionOrchestrator()
        assert orch._serialize_output(Obj()) == {"x": 10}

    def test_serialize_output_raw(self):
        orch = ExecutionOrchestrator()
        assert orch._serialize_output(42) == {"raw": "42"}


class TestCycleDetection:
    def test_cycle_marks_failed(self, tmp_path):
        orch = ExecutionOrchestrator(
            working_dir=str(tmp_path),
            tool_registry={"dummy": _sync_dummy},
        )
        stages = [
            {"id": "a", "tool": "dummy", "depends_on": ["b"]},
            {"id": "b", "tool": "dummy", "depends_on": ["a"]},
        ]
        record = asyncio.run(orch.run(stages))
        assert record.overall_success is False
        assert all(not r.success for r in record.stage_results)
        assert "Dependencies unresolved" in record.stage_results[0].error_message


class TestCheckpoint:
    def test_save_checkpoint(self, tmp_path):
        orch = ExecutionOrchestrator(
            working_dir=str(tmp_path),
            tool_registry={"dummy": _sync_dummy},
        )
        asyncio.run(orch.run([{"id": "s1", "tool": "dummy"}]))

        checkpoints = orch.list_checkpoints()
        assert len(checkpoints) == 1
        loaded = orch.load_checkpoint(checkpoints[0])
        assert loaded is not None
        assert loaded.workflow_name == "unnamed_workflow"
        assert loaded.overall_success is True
