"""lammps_tool.py 集成路径补测 — 覆盖 __init__/_find_lammps/estimate_cost/
validate_input/call(analyze_trajectory·mock·缺可执行)/poll·wait_job/_parse_log/
_run_equilibrium_check/_apply_script_fixes/_read_script_params/_try_autofix/
_is_float/_to_float_or_str/_uq_hint.

配合 test_lammps_ext.py (_parse_trajectory_python/parse_trajectory),
把 lammps_tool.py 覆盖率从 40% 提升到 85%+.
"""

from __future__ import annotations

import shutil
import sys
import time
import types
from pathlib import Path

import pytest

from huginn.tools.sim import executable_resolver as er
from huginn.tools.sim import lammps_tool as lt

TRJ_PATH = Path(__file__).parent.parent / "lammps_traj_test" / "traj.lammpstrj"

pytestmark = pytest.mark.anyio


def _tool(**kw):
    return lt.LammpsTool(**kw)


def _args(**kw):
    base = {"action": "run", "input_script": "run 0"}
    base.update(kw)
    return lt.LammpsToolInput(**base)


def _ctx():
    return type("C", (), {"workspace": "."})()


# ── __init__ / _find_lammps ──────────────────────────────────────────────


def test_find_lammps_env(monkeypatch, tmp_path):
    exe = tmp_path / "lmp"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("LAMMPS_EXECUTABLE", str(exe))
    assert _tool()._find_lammps() == str(exe)


def test_find_lammps_env_not_exists(monkeypatch):
    monkeypatch.setenv("LAMMPS_EXECUTABLE", "/no/lmp")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert _tool()._find_lammps() is None


def test_find_lammps_path(monkeypatch):
    monkeypatch.delenv("LAMMPS_EXECUTABLE", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/lmp" if name == "lmp" else None)
    assert _tool()._find_lammps() == "/usr/bin/lmp"


def test_find_lammps_which_raises(monkeypatch):
    monkeypatch.delenv("LAMMPS_EXECUTABLE", raising=False)

    def boom(name):
        raise OSError("boom")

    monkeypatch.setattr(shutil, "which", boom)
    # glob 无匹配 → None
    assert _tool()._find_lammps() is None


# ── estimate_cost ─────────────────────────────────────────────────────────


def test_estimate_cost_poll_wait_none():
    tool = _tool()
    assert tool.estimate_cost(_args(action="poll_job", job_id="x")) is None
    assert tool.estimate_cost(_args(action="wait_job", job_id="x")) is None


def test_estimate_cost_run():
    tool = _tool()
    c = tool.estimate_cost(_args(action="run"))
    assert c == {"cpu_hours": 2, "walltime_hours": 2}


# ── validate_input ────────────────────────────────────────────────────────


async def _val(tool, args, context):
    return await tool.validate_input(args, context)


async def test_validate_poll_wait_pass():
    tool = _tool()
    r = await _val(tool, _args(action="poll_job", job_id="x"), _ctx())
    assert r.result is True


async def test_validate_analyze_no_traj():
    tool = _tool()
    r = await _val(tool, _args(action="analyze_trajectory", input_script=""), _ctx())
    assert r.result is False
    assert r.error_code == 400


async def test_validate_analyze_traj_missing():
    tool = _tool()
    r = await _val(tool, _args(action="analyze_trajectory", trajectory_file="/no/x"), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_analyze_traj_ok():
    tool = _tool()
    r = await _val(tool, _args(action="analyze_trajectory", trajectory_file=str(TRJ_PATH)), _ctx())
    assert r.result is True


async def test_validate_structure_missing(tmp_path):
    tool = _tool()
    r = await _val(tool, _args(action="run", structure_file="/no/struct"), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_input_script_missing(tmp_path):
    tool = _tool()
    r = await _val(tool, _args(action="run", input_script="/no/script.in"), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_potential_missing(tmp_path):
    tool = _tool()
    r = await _val(tool, _args(action="run", potentials=["/no/pot"]), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_success(tmp_path):
    tool = _tool()
    r = await _val(tool, _args(action="run", input_script="run 0"), _ctx())
    assert r.result is True


# ── call(): analyze_trajectory ───────────────────────────────────────────


async def test_call_analyze_trajectory_success(monkeypatch):
    tool = _tool()
    res = await tool.call(
        _args(action="analyze_trajectory", trajectory_file=str(TRJ_PATH)), _ctx()
    )
    assert res.success is True
    assert res.data["n_frames"] == 3
    assert res.data["uq_hint"]["tool"] == "gp_tool"


async def test_call_analyze_trajectory_missing_file():
    tool = _tool()
    res = await tool.call(_args(action="analyze_trajectory"), _ctx())
    assert res.success is False
    assert "not specified or not found" in res.error


# ── call(): 缺可执行文件 ─────────────────────────────────────────────────


async def test_call_no_executable(monkeypatch):
    monkeypatch.setattr(
        er, "resolve_executable",
        lambda name: types.SimpleNamespace(
            install_hint="Install LAMMPS via your package manager.",
            to_dict=lambda: {"name": name},
        ),
    )
    tool = _tool(lammps_executable=None)
    res = await tool.call(_args(action="run"), _ctx())
    assert res.success is False
    assert "LAMMPS executable not found" in res.error


# ── call(): _run_lammps 成功/失败 (mock sandbox) ─────────────────────────


def _install_auditor(monkeypatch):
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")
    class _Audit:
        def __init__(self, has_errors=False):
            self.has_errors = has_errors
            self.findings = []

        def to_dict(self):
            return {"has_errors": self.has_errors}
    class _Auditor:
        def audit(self, *a, **k):
            return _Audit()
    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)


def _install_provenance(monkeypatch):
    prov_mod = types.ModuleType("huginn.provenance")
    class _Cap:
        def to_dict(self):
            return {"p": True}
    prov_mod.capture = lambda *a, **k: _Cap()
    monkeypatch.setitem(sys.modules, "huginn.provenance", prov_mod)


async def test_call_run_success(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch)
    _install_provenance(monkeypatch)

    class _Sb:
        def run(self, *a, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(action="run", input_script="run 0", working_dir=str(d)), _ctx())
    assert res.success is True
    assert res.data["final_energy"] is None  # 无 log → 无能量
    assert res.data["uq_hint"]["tool"] == "gp_tool"


async def test_call_run_hard_failure(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _install_provenance(monkeypatch)

    class _Sb:
        def run(self, *a, **k):
            return types.SimpleNamespace(returncode=1, stderr="atom count mismatch", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(action="run", working_dir=str(d)), _ctx())
    assert res.success is False
    assert "atom count mismatch" in res.error


async def test_call_run_timeout(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()

    class _Sb:
        def run(self, *a, **k):
            import subprocess
            raise subprocess.TimeoutExpired("cmd", 3600)

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(action="run", working_dir=str(d)), _ctx())
    assert res.success is False
    assert "timed out" in res.error


async def test_call_run_exception(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()

    class _Sb:
        def run(self, *a, **k):
            raise RuntimeError("boom")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(action="run", working_dir=str(d)), _ctx())
    assert res.success is False
    assert "LAMMPS execution failed" in res.error


# ── poll_job / wait_job ──────────────────────────────────────────────────


async def test_poll_unknown_job():
    tool = _tool()
    res = await tool.call(_args(action="poll_job", working_dir=".", job_id="nope"), _ctx())
    assert res.success is False
    assert "Unknown job_id" in res.error


async def test_poll_known_job():
    tool = _tool()
    job_id = "j1"
    lt.LammpsTool._async_jobs[job_id] = {
        "status": "done", "started_at": time.time() - 3, "result": {"energy": -1.0},
        "error": None, "compute_action": "run",
    }
    try:
        res = await tool.call(_args(action="poll_job", working_dir=".", job_id=job_id), _ctx())
        assert res.success is True
        assert res.data["status"] == "done"
        assert res.data["progress"] == 100
    finally:
        lt.LammpsTool._async_jobs.pop(job_id, None)


async def test_wait_task_none():
    tool = _tool()
    job_id = "j2"
    lt.LammpsTool._async_jobs[job_id] = {
        "status": "running", "started_at": time.time(), "result": None,
        "error": None, "compute_action": "run", "task": None,
    }
    try:
        res = await tool.call(_args(action="wait_job", working_dir=".", job_id=job_id), _ctx())
        assert res.success is True
    finally:
        lt.LammpsTool._async_jobs.pop(job_id, None)


# ── _parse_log ───────────────────────────────────────────────────────────


def test_parse_log_missing(tmp_path):
    tool = _tool()
    thermo, energy, warns = tool._parse_log(tmp_path / "nope.log")
    assert thermo == {}
    assert energy is None
    assert "Log file not found" in warns


def test_parse_log_thermo(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text(
        "Step Temp Press TotEng\n"
        "0 300 1.0 -10.5\n"
        "10 305 1.1 -10.4\n"
        "WARNING: some warning\n"
        "ERROR: some error\n",
        encoding="utf-8",
    )
    tool = _tool()
    thermo, energy, warns = tool._parse_log(log)
    assert thermo["step"] == [0.0, 10.0]
    assert thermo["temp"] == [300.0, 305.0]
    assert energy == pytest.approx(-10.4)
    assert any("WARNING" in w for w in warns)
    assert any("ERROR" in w for w in warns)


def test_parse_log_read_exception(tmp_path, monkeypatch):
    from pathlib import Path

    log = tmp_path / "log.lammps"
    log.write_text("Step Temp\n0 300\n", encoding="utf-8")
    tool = _tool()

    def boom(self):
        raise OSError("io")

    monkeypatch.setattr(Path, "read_text", boom)
    thermo, energy, warns = tool._parse_log(log)
    assert any("Failed to parse log" in w for w in warns)


# ── _run_equilibrium_check ───────────────────────────────────────────────


def test_equilibrium_check_no_log(tmp_path):
    tool = _tool()
    res = tool._run_equilibrium_check(_args(action="equilibrium_check", working_dir=str(tmp_path)))
    assert res.success is False
    assert "No log file found" in res.error


def test_equilibrium_check_no_thermo(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text("just text no data\n", encoding="utf-8")
    tool = _tool()
    res = tool._run_equilibrium_check(
        _args(action="equilibrium_check", log_file_path=str(log))
    )
    assert res.success is True
    assert res.data["equilibrated"] is False


def test_equilibrium_check_equilibrated(tmp_path):
    log = tmp_path / "log.lammps"
    lines = ["Step Temp Press", *[f"{i} {300 + i} {10 + i}" for i in range(20)]]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tool = _tool()
    res = tool._run_equilibrium_check(
        _args(action="equilibrium_check", log_file_path=str(log), target_temp=300)
    )
    assert res.success is True
    assert "recommendation" in res.data


# ── _apply_script_fixes / _read_script_params ────────────────────────────


def test_apply_script_fixes():
    tool = _tool()
    script = "# comment\ntimestep 1.0\npair_style lj/cut 2.5\n"
    out = tool._apply_script_fixes(script, {"timestep": "0.5"})
    assert "timestep 0.5" in out
    assert "pair_style lj/cut 2.5" in out
    assert "# comment" in out


def test_apply_script_fixes_appends_new():
    tool = _tool()
    script = "run 0\n"
    out = tool._apply_script_fixes(script, {"neighbor": "0.3"})
    assert "neighbor 0.3" in out


def test_read_script_params(tmp_path):
    p = tmp_path / "input.lammps"
    p.write_text("# c\ntimestep 0.5\nneighbor 2.0 bin\nunits real\n", encoding="utf-8")
    tool = _tool()
    params = tool._read_script_params(p)
    assert params["timestep"] == pytest.approx(0.5)
    assert params["neighbor"] == pytest.approx(2.0)


def test_read_script_params_missing(tmp_path):
    tool = _tool()
    assert tool._read_script_params(tmp_path / "nope") == {}


# ── _is_float / _to_float_or_str ─────────────────────────────────────────


def test_is_float():
    tool = _tool()
    assert tool._is_float("3.14") is True
    assert tool._is_float("abc") is False


def test_to_float_or_str():
    tool = _tool()
    assert tool._to_float_or_str("2.5") == 2.5
    assert tool._to_float_or_str("abc") == "abc"


# ── _uq_hint ─────────────────────────────────────────────────────────────


def test_uq_hint():
    tool = _tool()
    h = tool._uq_hint()
    assert h["tool"] == "gp_tool"


# ── call(): run-loop 剩余分支 ────────────────────────────────────────────


def _make_auditor(monkeypatch, has_errors=False, findings=None):
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Finding:
        def __init__(self, severity, message):
            self.severity = severity
            self.message = message

    class _Audit:
        def __init__(self, has_errors, findings):
            self.has_errors = has_errors
            self.findings = findings

        def to_dict(self):
            return {"has_errors": self.has_errors, "findings": len(self.findings)}

    class _Auditor:
        def audit(self, *a, **k):
            return _Audit(has_errors, list(findings or []))

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)


async def test_call_run_with_structure_fixes_and_mpiexec(monkeypatch, tmp_path):
    """input_script 为文件路径 + structure_file 存在 + fixes + potentials + mpiexec."""
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch)
    _install_provenance(monkeypatch)

    script = tmp_path / "in.lammps"
    script.write_text("run 0\n", encoding="utf-8")
    structure = tmp_path / "data.lmp"
    structure.write_text("data", encoding="utf-8")
    pot = tmp_path / "pot.eam"
    pot.write_text("pot", encoding="utf-8")

    calls = []

    class _Sb:
        def run(self, cmd, **k):
            calls.append(cmd)
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    args = _args(
        action="run",
        input_script=str(script),
        structure_file=str(structure),
        potentials=[str(pot)],
        num_processes=4,
        fixes={"timestep": "0.5"},
        working_dir=str(d),
    )
    res = await tool.call(args, _ctx())
    assert res.success is True
    # mpiexec 前缀生效
    assert calls[0][0] == "mpiexec"
    assert calls[0][1] == "-n"
    assert calls[0][2] == "4"
    # fixes 被写进 input.lammps
    written = (d / "input.lammps").read_text(encoding="utf-8")
    assert "0.5" in written
    # potential 被复制进工作目录
    assert (d / "pot.eam").exists()


async def test_call_run_phys_audit_soft_failure_and_autofix(monkeypatch, tmp_path):
    """物理审计报错 → 软失败 → autofix 重试 → 脚本修复后成功."""
    d = tmp_path / "wd"
    d.mkdir()
    _install_provenance(monkeypatch)

    # 有状态 auditor: 第一次报错 (温度爆炸), autofix 重写脚本后第二次通过
    audit_calls = {"n": 0}
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Finding:
        severity = "error"
        message = "temperature exploded"

    class _Audit:
        def __init__(self, has_errors):
            self.has_errors = has_errors
            self.findings = [_Finding()] if has_errors else []

        def to_dict(self):
            return {"has_errors": self.has_errors}

    class _Auditor:
        def audit(self, *a, **k):
            audit_calls["n"] += 1
            return _Audit(has_errors=audit_calls["n"] <= 1)

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)

    # autofix 返回修复 → 重写脚本 → 重试成功
    monkeypatch.setattr(
        lt.LammpsTool, "_try_autofix",
        lambda self, ip, err: {"fixes": {"timestep": "0.5"}, "reasoning": "reduce dt"},
    )

    class _Sb:
        def run(self, cmd, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(action="run", working_dir=str(d)), _ctx())
    assert res.success is True
    assert res.data["autoheal_attempts"][0]["fixes_applied"] == {"timestep": "0.5"}


async def test_call_run_phys_audit_hard_failure_fallback(monkeypatch, tmp_path):
    """硬失败 (returncode!=0) → 兜底审计补跑 → audit 异常被吞."""
    d = tmp_path / "wd"
    d.mkdir()
    _install_provenance(monkeypatch)

    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")
    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")
    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    # 不触发 autofix 重试, 直接硬失败结束
    monkeypatch.setattr(lt.LammpsTool, "_try_autofix", lambda self, ip, err: None)

    class _Sb:
        def run(self, cmd, **k):
            return types.SimpleNamespace(returncode=1, stderr="boom", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(action="run", working_dir=str(d)), _ctx())
    assert res.success is False
    assert "boom" in res.error


async def test_call_run_traj_found_and_provenance_exception(monkeypatch, tmp_path):
    """工作目录存在轨迹文件 → parse_trajectory 成功; provenance 异常被吞."""
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch)

    # provenance 抛异常
    prov_mod = types.ModuleType("huginn.provenance")
    def _boom(*a, **k):
        raise RuntimeError("prov boom")
    prov_mod.capture = _boom
    monkeypatch.setitem(sys.modules, "huginn.provenance", prov_mod)

    # 轨迹文件
    (d / "traj.lammpstrj").write_text("ITEM: TIMESTEP\n0\n", encoding="utf-8")
    monkeypatch.setattr(
        lt.LammpsTool, "parse_trajectory",
        lambda self, p: {"n_frames": 1, "msd": []},
    )

    class _Sb:
        def run(self, cmd, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(action="run", working_dir=str(d)), _ctx())
    assert res.success is True
    assert res.data["trajectory_analysis"]["n_frames"] == 1


# ── submit_async ─────────────────────────────────────────────────────────


def test_submit_async_schema_missing_compute_action():
    with pytest.raises(ValueError):
        lt.LammpsToolInput(action="submit_async", input_script="run 0")


def test_submit_async_schema_missing_input_script():
    with pytest.raises(ValueError):
        lt.LammpsToolInput(action="submit_async", compute_action="run")


def test_poll_wait_schema_requires_job_id():
    with pytest.raises(ValueError):
        lt.LammpsToolInput(action="poll_job")


async def test_submit_async_background(monkeypatch, tmp_path):
    """submit_async → 后台跑 run → poll 确认完成."""
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch)
    _install_provenance(monkeypatch)

    class _Sb:
        def run(self, cmd, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    args = lt.LammpsToolInput(
        action="submit_async",
        compute_action="run",
        input_script="run 0",
        working_dir=str(d),
    )
    res = await tool.call(args, _ctx())
    assert res.success is True
    job_id = res.data["job_id"]
    assert res.data["status"] == "running"

    # 等后台任务完成
    poll = await tool.call(
        lt.LammpsToolInput(action="wait_job", job_id=job_id, working_dir=".", timeout=5),
        _ctx(),
    )
    assert poll.success is True
    assert poll.data["status"] == "done"


# ── wait_job 分支 ────────────────────────────────────────────────────────


async def test_wait_job_already_done():
    tool = _tool()
    job_id = "j3"
    lt.LammpsTool._async_jobs[job_id] = {
        "status": "done", "started_at": time.time() - 3, "result": {"energy": -1.0},
        "error": None, "compute_action": "run", "task": object(),
    }
    try:
        res = await tool.call(
            _args(action="wait_job", working_dir=".", job_id=job_id), _ctx()
        )
        assert res.success is True
        assert res.data["status"] == "done"
    finally:
        lt.LammpsTool._async_jobs.pop(job_id, None)


async def test_wait_job_task_timeout(monkeypatch):
    """后台任务还在跑, wait 超时 → 返回当前 running 状态."""
    import asyncio

    async def _never():
        await asyncio.sleep(100)

    # 让 wait_for 立即抛 TimeoutError, 加速测试
    async def _fake_wait_for(coro, timeout):
        raise TimeoutError("timed out")
    monkeypatch.setattr(asyncio, "wait_for", _fake_wait_for)

    tool = _tool()
    job_id = "j4"
    lt.LammpsTool._async_jobs[job_id] = {
        "status": "running", "started_at": time.time(), "result": None,
        "error": None, "compute_action": "run", "task": _never(),
    }
    try:
        res = await tool.call(
            _args(action="wait_job", working_dir=".", job_id=job_id, timeout=1.0),
            _ctx(),
        )
        assert res.success is True
        assert res.data["status"] == "running"
    finally:
        lt.LammpsTool._async_jobs.pop(job_id, None)


async def test_wait_job_task_exception():
    """后台任务抛异常 → job 标记 failed."""
    tool = _tool()

    async def _boom():
        raise RuntimeError("task died")

    job_id = "j5"
    lt.LammpsTool._async_jobs[job_id] = {
        "status": "running", "started_at": time.time(), "result": None,
        "error": None, "compute_action": "run", "task": _boom(),
    }
    try:
        res = await tool.call(
            _args(action="wait_job", working_dir=".", job_id=job_id, timeout=5),
            _ctx(),
        )
        assert res.success is True
        assert res.data["status"] == "failed"
        assert "task died" in res.data["error"]
    finally:
        lt.LammpsTool._async_jobs.pop(job_id, None)


# ── equilibrium_check 未平衡分支 / recommendation ─────────────────────────


def test_equilibrium_recommendation_drift_and_temp():
    """未平衡: 温度偏差 + 漂移 → 列出原因."""
    tool = _tool()
    rec = tool._build_equilibrium_recommendation(
        equilibrated=False,
        avg_temp=350.0,
        target_temp=300.0,
        temp_drift=0.05,
        avg_press=1.0,
        target_pressure=1.0,
        n_tail=500,
        n_total=1000,
    )
    assert "Not equilibrated" in rec
    assert "temperature drift" in rec
    assert "deviates" in rec


def test_equilibrium_recommendation_pressure():
    """未平衡: 压力严重偏移 → 列入原因."""
    tool = _tool()
    rec = tool._build_equilibrium_recommendation(
        equilibrated=False,
        avg_temp=300.0,
        target_temp=300.0,
        temp_drift=0.001,
        avg_press=5000.0,
        target_pressure=100.0,
        n_tail=500,
        n_total=1000,
    )
    assert "Not equilibrated" in rec


def test_equilibrium_recommendation_equilibrated():
    tool = _tool()
    rec = tool._build_equilibrium_recommendation(
        equilibrated=True, avg_temp=300.0, target_temp=300.0, temp_drift=0.001,
        avg_press=1.0, target_pressure=1.0, n_tail=500, n_total=1000,
    )
    assert "reached equilibrium" in rec


def test_equilibrium_recommendation_no_reasons():
    """未平衡但无具体原因 → 'close to equilibrium'."""
    tool = _tool()
    rec = tool._build_equilibrium_recommendation(
        equilibrated=False, avg_temp=300.0, target_temp=None, temp_drift=0.001,
        avg_press=None, target_pressure=None, n_tail=500, n_total=1000,
    )
    assert "close to equilibrium" in rec


def test_equilibrium_check_no_temp(tmp_path):
    """thermo 有 step 但无 temp → 提示检查 thermo_style."""
    log = tmp_path / "log.lammps"
    log.write_text("Step Press\n0 1.0\n10 1.1\n", encoding="utf-8")
    tool = _tool()
    res = tool._run_equilibrium_check(
        lt.LammpsToolInput(action="equilibrium_check", log_file_path=str(log))
    )
    assert res.success is True
    assert "No temperature data" in res.data["recommendation"]


# ── DEM packing ──────────────────────────────────────────────────────────


async def test_dem_packing_no_executable(monkeypatch, tmp_path):
    """无 LAMMPS 可执行文件 → 只生成脚本, 标记 needs_resolution."""
    monkeypatch.setattr(
        er, "resolve_executable",
        lambda name: types.SimpleNamespace(
            install_hint="Install LAMMPS.", to_dict=lambda: {"name": name},
        ),
    )
    tool = _tool(lammps_executable=None)
    res = await tool.call(
        _args(action="dem_packing", working_dir=str(tmp_path)), _ctx()
    )
    assert res.success is True
    assert res.data["needs_resolution"] is True
    assert "Script generated only" in res.error
    assert (tmp_path / "input.dem.lammps").exists()


def test_generate_dem_input_script_polydisperse():
    """多分散粒径 + 标准 restitution → 脚本含 polydispersion block."""
    tool = _tool()
    args = lt.LammpsToolInput(
        action="dem_packing",
        dem_radius=5.0,
        dem_radius_std=1.0,
        dem_n_particles=100,
        dem_n_steps=1000,
    )
    script = tool._generate_dem_input_script(args)
    assert "variable       r_var normal" in script
    assert "set             type 1 diameter v_r_var" in script
    assert "pair_style      granular" in script
    assert "# no gravity" in script


def test_generate_dem_input_script_monodisperse_and_restitution():
    """单分散 + e=1 (完全弹性) 与 e<1 分支."""
    tool = _tool()
    args = lt.LammpsToolInput(
        action="dem_packing",
        dem_radius=5.0,
        dem_radius_std=0.0,
        dem_restitution=1.0,
        dem_gravity=9.8,
        dem_n_particles=100,
        dem_n_steps=1000,
    )
    script = tool._generate_dem_input_script(args)
    assert "# 单分散粒径" in script
    assert "fix            gravity all gravity 9.8" in script

    # 完全弹性 (e=1) → alpha=0, 脚本里阻尼系数为 0
    assert "damping tsuji 0.000000" in script

    # e 在 (0,1) → Tsuji 阻尼正常分支
    args2 = lt.LammpsToolInput(
        action="dem_packing",
        dem_radius=5.0,
        dem_restitution=0.8,
        dem_n_particles=100,
        dem_n_steps=1000,
    )
    script2 = tool._generate_dem_input_script(args2)
    assert "damping tsuji" in script2


async def test_dem_packing_run_success(monkeypatch, tmp_path):
    """有可执行文件 → 实际执行 → 返回 output_dir/status."""
    d = tmp_path / "wd"
    d.mkdir()
    _install_provenance(monkeypatch)

    class _Sb:
        def run(self, cmd, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="done")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(
        _args(action="dem_packing", working_dir=str(d)), _ctx()
    )
    assert res.success is True
    assert res.data["status"] == "completed"
    assert res.data["output_dir"] == str(d)


async def test_dem_packing_run_failure(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()

    class _Sb:
        def run(self, cmd, **k):
            return types.SimpleNamespace(returncode=1, stderr="bad", stdout="")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(
        _args(action="dem_packing", working_dir=str(d)), _ctx()
    )
    assert res.success is False
    assert res.data["status"] == "failed"
    assert "exited with code 1" in res.error


async def test_dem_packing_run_timeout(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()

    class _Sb:
        def run(self, cmd, **k):
            import subprocess
            raise subprocess.TimeoutExpired("cmd", 3600)

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(
        _args(action="dem_packing", working_dir=str(d)), _ctx()
    )
    assert res.success is False
    assert "timed out" in res.error


async def test_dem_packing_run_exception(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()

    class _Sb:
        def run(self, cmd, **k):
            raise RuntimeError("dem boom")

    tool = _tool(lammps_executable="/usr/bin/lmp")
    tool.sandbox = _Sb()
    res = await tool.call(
        _args(action="dem_packing", working_dir=str(d)), _ctx()
    )
    assert res.success is False
    assert "DEM execution failed" in res.error
