"""Tests for the LAMMPS equilibrium_check action."""

from pathlib import Path

import pytest

from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")


def _make_log(tmp_path: Path, temps: list[float], presses: list[float] | None = None,
              step_size: int = 100) -> Path:
    """Write a minimal LAMMPS log with thermo data for the given temp/press values."""
    log = tmp_path / "log.lammps"
    lines = ["LAMMPS run", "Step Temp Press TotEng"]
    presses = presses if presses is not None else [0.0] * len(temps)
    for i, (t, p) in enumerate(zip(temps, presses)):
        step = i * step_size
        e = -100.0 - i * 0.001
        lines.append(f"{step} {t:.4f} {p:.4f} {e:.4f}")
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


def test_equilibrated_system(tmp_path: Path):
    """Temp stays within 5% of target and drift is low -> equilibrated."""
    temps = [300.0] * 100
    log = _make_log(tmp_path, temps)
    tool = LammpsTool(lammps_executable=None)
    args = LammpsToolInput(
        action="equilibrium_check",
        log_file_path=str(log),
        target_temp=300.0,
        window=30.0,
    )
    result = tool._run_equilibrium_check(args)
    assert result.success
    assert result.data["equilibrated"] is True
    assert abs(result.data["avg_temp"] - 300.0) < 0.01
    assert "Proceed with production" in result.data["recommendation"]


def test_not_equilibrated_temp_drift(tmp_path: Path):
    """Temperature drifting upward -> not equilibrated."""
    temps = [300.0 + i * 2.0 for i in range(100)]  # 300 -> 498 K
    log = _make_log(tmp_path, temps)
    tool = LammpsTool(lammps_executable=None)
    args = LammpsToolInput(
        action="equilibrium_check",
        log_file_path=str(log),
        target_temp=300.0,
        window=30.0,
    )
    result = tool._run_equilibrium_check(args)
    assert result.success
    assert result.data["equilibrated"] is False
    assert "drift" in result.data["recommendation"].lower()


def test_pressure_check(tmp_path: Path):
    """Pressure far from target should appear in recommendation."""
    temps = [300.0] * 100
    presses = [500.0] * 100
    log = _make_log(tmp_path, temps, presses)
    tool = LammpsTool(lammps_executable=None)
    args = LammpsToolInput(
        action="equilibrium_check",
        log_file_path=str(log),
        target_temp=300.0,
        target_pressure=1.0,
        window=30.0,
    )
    result = tool._run_equilibrium_check(args)
    assert result.success
    # temp is fine, pressure is off
    assert result.data["avg_pressure"] is not None
    assert abs(result.data["avg_pressure"] - 500.0) < 1.0


def test_empty_log_handling(tmp_path: Path):
    """An empty or non-existent log should return a useful error."""
    tool = LammpsTool(lammps_executable=None)
    # no log_file_path and no working_dir
    args = LammpsToolInput(action="equilibrium_check")
    result = tool._run_equilibrium_check(args)
    assert result.success is False
    assert "No log file" in result.error

    # empty log file
    empty_log = tmp_path / "empty.log"
    empty_log.write_text("", encoding="utf-8")
    args2 = LammpsToolInput(action="equilibrium_check", log_file_path=str(empty_log))
    result2 = tool._run_equilibrium_check(args2)
    assert result2.success
    assert result2.data["equilibrated"] is False
    assert "no thermo data" in result2.data["recommendation"].lower()


def test_recommendation_generation(tmp_path: Path):
    """When not equilibrated, the recommendation should suggest extending steps."""
    temps = [300.0 + i * 5.0 for i in range(100)]  # strong drift
    log = _make_log(tmp_path, temps)
    tool = LammpsTool(lammps_executable=None)
    args = LammpsToolInput(
        action="equilibrium_check",
        log_file_path=str(log),
        target_temp=300.0,
        window=30.0,
    )
    result = tool._run_equilibrium_check(args)
    assert result.success
    rec = result.data["recommendation"]
    assert "Extend run" in rec or "reduce timestep" in rec


def test_window_percentage(tmp_path: Path):
    """The window parameter should control how many trailing steps are used."""
    temps = [300.0 + (1 if i > 70 else 0) for i in range(100)]
    log = _make_log(tmp_path, temps)
    tool = LammpsTool(lammps_executable=None)
    args = LammpsToolInput(
        action="equilibrium_check",
        log_file_path=str(log),
        target_temp=300.0,
        window=10.0,  # last 10 steps only
    )
    result = tool._run_equilibrium_check(args)
    assert result.success
    assert result.data["window_steps"] == 10
    assert result.data["total_steps"] == 100


def test_via_call_method(tmp_path: Path):
    """equilibrium_check should work through the async call() entry point."""
    import asyncio

    temps = [300.0] * 50
    log = _make_log(tmp_path, temps)
    tool = LammpsTool(lammps_executable=None)
    args = LammpsToolInput(
        action="equilibrium_check",
        log_file_path=str(log),
        target_temp=300.0,
        window=30.0,
    )
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(tool.call(args, CTX))
    finally:
        loop.close()
    assert result.success
    assert result.data["equilibrated"] is True
