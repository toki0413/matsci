"""Tests for the Transolver++ PDE surrogate tool.

These cover tool metadata, the torch-free list_models path, graceful
degradation when torch/transolver are absent, cost estimation, and input
schema validation. The predict/train happy paths require torch + the
transolver package and are skipped when those aren't importable.
"""

import asyncio
import importlib.util
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from huginn.tools.sim.transolver_tool import (
    TransolverTool,
    TransolverToolInput,
    TransolverToolOutput,
)
from huginn.types import ToolContext, ToolResult


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


def _run(tool, args):
    return asyncio.run(tool.call(args, _ctx()))


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# ── metadata ────────────────────────────────────────────────────


def test_tool_metadata():
    tool = TransolverTool()
    assert tool.name == "transolver_tool"
    assert tool.category == "sim"
    assert tool.profile.cost_tier == "heavy"
    # only training is genuinely expensive; predict / list_models are cheap
    assert tool.profile.heavy_actions == frozenset({"train"})
    assert "numerical_tool" in tool.profile.light_alternatives
    # EXECUTION-phase tool by design
    from huginn.tools.base import ResearchPhase

    assert ResearchPhase.EXECUTION in tool.profile.phases


def test_input_and_output_schemas():
    inp = TransolverToolInput(action="predict", coords=[[0.0, 0.0]], features=[[1.0]])
    assert inp.action == "predict"
    assert inp.space_dim == 3
    assert inp.epochs == 10
    out = TransolverToolOutput(status="ok", predictions=[[0.5]])
    assert out.predictions == [[0.5]]


# ── list_models works without torch ─────────────────────────────


def test_list_models_empty_dir():
    tool = TransolverTool()
    with tempfile.TemporaryDirectory() as d:
        args = TransolverToolInput(action="list_models", checkpoint_dir=d)
        res = _run(tool, args)
    assert res.success
    assert res.data["status"] == "no_models"
    assert res.data["available_models"] == []


def test_list_models_finds_checkpoints():
    tool = TransolverTool()
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "elastic.pt").write_bytes(b"")
        (Path(d) / "fluid.pth").write_bytes(b"")
        (Path(d) / "ignore.txt").write_text("nope")  # not a checkpoint
        args = TransolverToolInput(action="list_models", checkpoint_dir=d)
        res = _run(tool, args)
    assert res.success
    assert res.data["status"] == "ok"
    assert res.data["available_models"] == ["elastic", "fluid"]


def test_list_models_uses_workspace_relative_dir():
    # When constructed with a workspace, checkpoints live under
    # <workspace>/.huginn/models/transolver/.
    with tempfile.TemporaryDirectory() as ws:
        tool = TransolverTool(workspace=ws)
        (Path(ws) / ".huginn" / "models" / "transolver").mkdir(parents=True)
        (Path(ws) / ".huginn" / "models" / "transolver" / "beam.pt").write_bytes(b"")
        res = _run(tool, TransolverToolInput(action="list_models"))
    assert res.success
    assert res.data["available_models"] == ["beam"]


# ── graceful degradation ─────────────────────────────────────────


@pytest.mark.skipif(_TORCH_AVAILABLE, reason="torch is installed; install-hint path skipped")
def test_predict_returns_install_hint_without_torch():
    tool = TransolverTool()
    args = TransolverToolInput(
        action="predict",
        coords=[[0.0, 0.0, 0.0]],
        features=[[1.0]],
    )
    res = _run(tool, args)
    assert not res.success
    assert "Transolver++" in res.error
    assert "github.com/thuml/Transolver_plus" in res.error


@pytest.mark.skipif(_TORCH_AVAILABLE, reason="torch is installed; install-hint path skipped")
def test_train_returns_install_hint_without_torch():
    tool = TransolverTool()
    args = TransolverToolInput(
        action="train",
        coords=[[0.0, 0.0, 0.0]],
        features=[[1.0]],
        target=[[2.0]],
        epochs=1,
    )
    res = _run(tool, args)
    assert not res.success
    assert "Transolver++" in res.error


# ── cost estimation ──────────────────────────────────────────────


def test_estimate_cost():
    tool = TransolverTool()
    train_cost = tool.estimate_cost(TransolverToolInput(action="train", epochs=4))
    assert train_cost is not None
    assert train_cost["gpu_hours"] == 4 * 0.05

    pred_cost = tool.estimate_cost(TransolverToolInput(action="predict"))
    assert pred_cost is not None and pred_cost["gpu_hours"] > 0

    # list_models is free
    assert tool.estimate_cost(TransolverToolInput(action="list_models")) is None


# ── input validation ──────────────────────────────────────────────


def test_input_schema_rejects_bad_values():
    with pytest.raises(ValidationError):
        TransolverToolInput(action="not_a_real_action")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TransolverToolInput(action="train", epochs=0)  # ge=1
    with pytest.raises(ValidationError):
        TransolverToolInput(action="train", learning_rate=0.0)  # gt=0
    with pytest.raises(ValidationError):
        TransolverToolInput(action="train", space_dim=0)  # ge=1
