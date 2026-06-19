"""Tests for the autoresearch plugin."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from huginn.plugins.autoresearch import AutoresearchInput, AutoresearchTool
from huginn.types import ToolContext


@pytest.fixture
def tool() -> AutoresearchTool:
    return AutoresearchTool()


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(session_id="test", workspace=str(tmp_path))


def _make_workspace(
    ws: Path, train_py: str | None = None, program: str | None = None
) -> None:
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "train.py").write_text(train_py or "print('baseline')\n", encoding="utf-8")
    (ws / "program.md").write_text(program or "# autoresearch\n", encoding="utf-8")


def _metrics_stdout(val_bpb: float = 0.99) -> str:
    return (
        "---\n"
        f"val_bpb: {val_bpb}\n"
        "training_seconds: 300.1\n"
        "total_seconds: 325.9\n"
        "peak_vram_mb: 45060.2\n"
        "mfu_percent: 39.80\n"
        "total_tokens_M: 499.6\n"
        "num_steps: 953\n"
        "num_params_M: 50.3\n"
        "depth: 8\n"
    )


@pytest.fixture
def fake_run_command_factory():
    """Return a factory that creates an async fake runner with canned output."""

    def _factory(train_val_bpb: float = 0.99):
        async def _fake(tool_self, args, cwd, timeout, capture_output=True, env=None):
            command_str = " ".join(args)
            if "train.py" in args:
                return {
                    "command": args,
                    "returncode": 0,
                    "stdout": _metrics_stdout(train_val_bpb),
                    "stderr": "",
                }
            if "prepare.py" in args:
                return {
                    "command": args,
                    "returncode": 0,
                    "stdout": "data prepared",
                    "stderr": "",
                }
            # Git-like commands always succeed with empty output.
            return {"command": args, "returncode": 0, "stdout": "", "stderr": ""}

        return _fake

    return _factory


@pytest.mark.asyncio
async def test_init_workspace_skip_git(
    tool: AutoresearchTool, ctx: ToolContext, tmp_path: Path
) -> None:
    ws = tmp_path / "ar"
    _make_workspace(ws)
    inp = AutoresearchInput(
        action="init_workspace",
        workspace=str(ws),
        skip_git=True,
    )
    result = await tool.call(inp, ctx)
    assert result.success
    assert result.data["workspace"] == str(ws)
    assert (ws / "results.tsv").exists()
    header = (ws / "results.tsv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "commit\tval_bpb\tmemory_gb\tstatus\tdescription"


@pytest.mark.asyncio
async def test_run_experiment_parses_metrics(
    tool: AutoresearchTool,
    ctx: ToolContext,
    tmp_path: Path,
    fake_run_command_factory,
    monkeypatch,
) -> None:
    ws = tmp_path / "ar"
    _make_workspace(ws)
    monkeypatch.setattr(
        AutoresearchTool, "_run_command", fake_run_command_factory(train_val_bpb=0.987)
    )
    result = await tool.call(
        AutoresearchInput(action="run_experiment", workspace=str(ws), skip_git=True),
        ctx,
    )
    assert result.success
    assert result.data["metrics"]["val_bpb"] == pytest.approx(0.987)
    assert result.data["metrics"]["peak_vram_mb"] == pytest.approx(45060.2)
    assert (ws / "run.log").exists()


@pytest.mark.asyncio
async def test_results_reads_tsv(
    tool: AutoresearchTool, ctx: ToolContext, tmp_path: Path
) -> None:
    ws = tmp_path / "ar"
    _make_workspace(ws)
    (ws / "results.tsv").write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "abc1234\t0.990000\t44.0\tkeep\tbaseline\n",
        encoding="utf-8",
    )
    result = await tool.call(
        AutoresearchInput(action="results", workspace=str(ws), skip_git=True), ctx
    )
    assert result.success
    assert result.data["count"] == 1
    assert result.data["rows"][0]["commit"] == "abc1234"


@pytest.mark.asyncio
async def test_step_manual_train_py_keeps_improvement(
    tool: AutoresearchTool,
    ctx: ToolContext,
    tmp_path: Path,
    fake_run_command_factory,
    monkeypatch,
) -> None:
    ws = tmp_path / "ar"
    _make_workspace(ws)
    # Seed a baseline result so the new experiment is an improvement.
    (ws / "results.tsv").write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "baseline\t1.000000\t44.0\tkeep\tbaseline\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        AutoresearchTool, "_run_command", fake_run_command_factory(train_val_bpb=0.950)
    )
    new_train = "# improved train.py\nprint('improved')\n"
    result = await tool.call(
        AutoresearchInput(
            action="step",
            workspace=str(ws),
            train_py=new_train,
            description="increase lr",
            skip_git=True,
        ),
        ctx,
    )
    assert result.success
    assert result.data["status"] == "keep"
    assert result.data["kept"] is True
    assert (ws / "train.py").read_text(encoding="utf-8") == new_train
    rows = tool._results(AutoresearchInput(workspace=str(ws), skip_git=True)).data[
        "rows"
    ]
    assert any(r["description"] == "increase lr" for r in rows)


@pytest.mark.asyncio
async def test_step_reverts_regression(
    tool: AutoresearchTool,
    ctx: ToolContext,
    tmp_path: Path,
    fake_run_command_factory,
    monkeypatch,
) -> None:
    ws = tmp_path / "ar"
    original_train = "# original\nprint('original')\n"
    _make_workspace(ws, train_py=original_train)
    (ws / "results.tsv").write_text(
        "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"
        "baseline\t0.900000\t44.0\tkeep\tbaseline\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        AutoresearchTool, "_run_command", fake_run_command_factory(train_val_bpb=0.950)
    )
    result = await tool.call(
        AutoresearchInput(
            action="step",
            workspace=str(ws),
            train_py="# worse\nprint('worse')\n",
            description="bad idea",
            skip_git=True,
        ),
        ctx,
    )
    assert result.success
    assert result.data["status"] == "discard"
    assert result.data["kept"] is False
    # With skip_git=True the file is left as the proposed edit; ensure we know.
    assert result.data["commit"] == "no-git"


@pytest.mark.asyncio
async def test_propose_edit_uses_llm(
    tool: AutoresearchTool, ctx: ToolContext, tmp_path: Path, monkeypatch
) -> None:
    ws = tmp_path / "ar"
    _make_workspace(ws, train_py="# original\n")

    class FakeMessage:
        content = json.dumps(
            {
                "hypothesis": "bigger is better",
                "description": "increase depth",
                "train_py": "# new train.py\n",
            }
        )

    def fake_get_model(*, config=None, temperature=None, max_tokens=None):
        class FakeModel:
            def invoke(self, messages):
                return FakeMessage()

        return FakeModel()

    monkeypatch.setattr("huginn.llm.get_model", fake_get_model)
    result = await tool.call(
        AutoresearchInput(
            action="propose_edit",
            workspace=str(ws),
            user_hint="try deeper",
            skip_git=True,
        ),
        ctx,
    )
    assert result.success
    assert result.data["hypothesis"] == "bigger is better"
    assert result.data["description"] == "increase depth"
    assert "# new train.py" in result.data["train_py"]


@pytest.mark.asyncio
async def test_status_skip_git(
    tool: AutoresearchTool, ctx: ToolContext, tmp_path: Path
) -> None:
    ws = tmp_path / "ar"
    _make_workspace(ws)
    result = await tool.call(
        AutoresearchInput(action="status", workspace=str(ws), skip_git=True), ctx
    )
    assert result.success
    assert result.data["git"] is False
    assert result.data["train_py_exists"] is True
