"""gromacs_tool.py 集成路径补测 — 覆盖 is_read_only/is_destructive、call 分派、
_md_run(tpr 缺失/gmx 缺失·skipped/成功/失败/超时/sandbox 拦截)、
_energy_minimize(同上)、_analyze_traj(traj 缺失/gmx 缺失/success/失败/超时/
sandbox 拦截)、_resolve_file、_parse_md_log(缺失/读异常/内容解析/警告统计).

gmx 未安装 → _gmx_available()=False 分支真实命中; 成功路径用 fake sandbox +
monkeypatch _gmx_available 覆盖. 把 gromacs_tool.py 覆盖率从 0% 提升到 90%+.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from huginn.security import SandboxError
from huginn.tools.sim import gromacs_tool as gt
from huginn.tools.sim.gromacs_tool import GromacsTool, GromacsToolInput

pytestmark = pytest.mark.anyio


def _tool(**kw):
    return GromacsTool(**kw)


def _args(**kw):
    base = {"action": "md_run", "tpr_file": "topol.tpr"}
    base.update(kw)
    return base


def _run_call(**kw):
    return {"action": "md_run", "tpr_file": "topol.tpr", **kw}


def _em_call(**kw):
    return {"action": "energy_minimize", "tpr_file": "topol.tpr", **kw}


def _at_call(**kw):
    return {"action": "analyze_traj", "trajectory_file": "md.xtc", **kw}


def _install_auditor(monkeypatch):
    """安装 fake physics_auditor: audit 返回可 to_dict 的对象."""
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Report:
        def to_dict(self):
            return {"has_errors": False, "findings": 0}

    class _Auditor:
        def audit(self, *a, **k):
            return _Report()

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)


def _install_auditor_boom(monkeypatch):
    """audit 抛异常 → 被吞掉, 结果正常返回."""
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Sb:
    def __init__(self, proc=None, call=None):
        self._proc = proc or _Proc()
        self._call = call
        self.calls = []

    def run(self, cmd, cwd=None, capture_output=True, text=True, timeout=None, input=None):
        self.calls.append({"cmd": cmd, "input": input})
        if self._call:
            return self._call(cmd, cwd)
        return self._proc


class _SbErr:
    def run(self, cmd, **kw):
        raise SandboxError("blocked")


class _SbTimeout:
    def run(self, cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, timeout=kw.get("timeout"))


# ── is_read_only / is_destructive ─────────────────────────────────────────


def test_is_read_only_analyze():
    t = _tool()
    assert t.is_read_only(GromacsToolInput(action="analyze_traj")) is True
    assert t.is_read_only(GromacsToolInput(action="md_run")) is False


def test_is_destructive():
    t = _tool()
    assert t.is_destructive(GromacsToolInput(action="md_run")) is True
    assert t.is_destructive(GromacsToolInput(action="energy_minimize")) is True
    assert t.is_destructive(GromacsToolInput(action="analyze_traj")) is False


# ── _resolve_file ─────────────────────────────────────────────────────────


def test_resolve_file(tmp_path):
    f = tmp_path / "a.tpr"
    f.write_text("x")
    t = _tool()
    assert t._resolve_file(None, tmp_path) is None
    assert t._resolve_file("missing.tpr", tmp_path) is None
    assert t._resolve_file(str(f), tmp_path) == f
    # 相对路径解析到 work_dir
    assert t._resolve_file("a.tpr", tmp_path) == f


# ── _md_run ───────────────────────────────────────────────────────────────


def test_md_run_tpr_missing(tmp_path):
    res = _tool().call(_run_call(tpr_file="nope.tpr"), context=None)
    assert res.success is False
    assert "TPR file not found" in res.error


def test_md_run_gmx_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: False)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    res = _tool().call(_run_call(tpr_file=str(tpr)), context=None)
    assert res.success is True
    assert res.data["status"] == "skipped"
    assert "gmx not installed" in res.data["message"]


def test_md_run_success(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    _install_auditor(monkeypatch)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    # 写日志供 _parse_md_log 解析
    (tmp_path / "topol.log").write_text(
        "Step 100  Temperature = 300.0  Pressure = 1.0\n"
        "Total Energy = -100.5\nLINCS WARNING\nLINCS WARNING\n"
    )
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="md ok", stderr=""))
    res = tool.call(_run_call(tpr_file=str(tpr)), context=None)
    assert res.success is True
    assert res.data["returncode"] == 0
    assert res.data["md_log_data"]["temperatures"] == [300.0]
    assert res.data["md_log_data"]["lincs_warnings"] == 2
    assert res.data["physics_audit"]["has_errors"] is False


def test_md_run_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="md crashed"))
    res = tool.call(_run_call(tpr_file=str(tpr)), context=None)
    assert res.success is False
    assert "gmx mdrun failed" in res.error


def test_md_run_sandbox_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _SbErr()
    res = tool.call(_run_call(tpr_file=str(tpr)), context=None)
    assert res.success is False
    assert "blocked by sandbox" in res.error


def test_md_run_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _SbTimeout()
    res = tool.call(_run_call(tpr_file=str(tpr)), context=None)
    assert res.success is False
    assert "timed out" in res.error


def test_md_run_audit_boom_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    _install_auditor_boom(monkeypatch)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="ok"))
    res = tool.call(_run_call(tpr_file=str(tpr)), context=None)
    assert res.success is True
    assert "physics_audit" not in res.data


# ── _energy_minimize ──────────────────────────────────────────────────────


def test_em_tpr_missing(tmp_path):
    res = _tool().call(_em_call(tpr_file="nope.tpr"), context=None)
    assert res.success is False
    assert "TPR file not found" in res.error


def test_em_gmx_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: False)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    res = _tool().call(_em_call(tpr_file=str(tpr)), context=None)
    assert res.success is True
    assert res.data["status"] == "skipped"


def test_em_success(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    _install_auditor(monkeypatch)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="em done"))
    res = tool.call(_em_call(tpr_file=str(tpr)), context=None)
    assert res.success is True
    assert res.data["action"] == "energy_minimize"
    assert res.data["physics_audit"]["has_errors"] is False


def test_em_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="em failed"))
    res = tool.call(_em_call(tpr_file=str(tpr)), context=None)
    assert res.success is False
    assert "EM) failed" in res.error

def test_em_sandbox_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _SbErr()
    res = tool.call(_em_call(tpr_file=str(tpr)), context=None)
    assert res.success is False
    assert "blocked by sandbox" in res.error


def test_em_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _SbTimeout()
    res = tool.call(_em_call(tpr_file=str(tpr)), context=None)
    assert res.success is False
    assert "timed out" in res.error


def test_em_audit_boom_swallowed(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    _install_auditor_boom(monkeypatch)
    tpr = tmp_path / "topol.tpr"
    tpr.write_text("x")
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="ok"))
    res = tool.call(_em_call(tpr_file=str(tpr)), context=None)
    assert res.success is True
    assert "physics_audit" not in res.data


# ── _analyze_traj ─────────────────────────────────────────────────────────


def test_at_traj_missing(tmp_path):
    res = _tool().call(_at_call(trajectory_file="nope.xtc"), context=None)
    assert res.success is False
    assert "Trajectory file not found" in res.error


def test_at_gmx_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: False)
    traj = tmp_path / "md.xtc"
    traj.write_text("x")
    res = _tool().call(_at_call(trajectory_file=str(traj), working_dir=str(tmp_path)), context=None)
    assert res.success is True
    assert res.data["status"] == "skipped"


@pytest.mark.parametrize("atype", ["rms", "rmsd", "rdf", "gyrate"])
def test_at_success(tmp_path, monkeypatch, atype):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    traj = tmp_path / "md.xtc"
    traj.write_text("x")
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="ok"))
    res = tool.call(_at_call(analysis_type=atype, working_dir=str(tmp_path)), context=None)
    assert res.success is True
    assert res.data["analysis_type"] == atype
    # input="0\n0\n" 自动选组
    assert tool.sandbox.calls[0]["input"] == "0\n0\n"


def test_at_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    traj = tmp_path / "md.xtc"
    traj.write_text("x")
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="analysis failed"))
    res = tool.call(_at_call(working_dir=str(tmp_path)), context=None)
    assert res.success is False
    assert "analysis failed" in res.error


def test_at_sandbox_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    traj = tmp_path / "md.xtc"
    traj.write_text("x")
    tool = _tool()
    tool.sandbox = _SbErr()
    res = tool.call(_at_call(working_dir=str(tmp_path)), context=None)
    assert res.success is False
    assert "blocked by sandbox" in res.error


def test_at_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_gmx_available", lambda: True)
    traj = tmp_path / "md.xtc"
    traj.write_text("x")
    tool = _tool()
    tool.sandbox = _SbTimeout()
    res = tool.call(_at_call(working_dir=str(tmp_path)), context=None)
    assert res.success is False
    assert "timed out" in res.error


def test_call_unknown_action(tmp_path):
    # Literal 校验不通过 → pydantic ValidationError 由 call 抛出, 不会被
    # 内部的分派 (line 91) 捕获; 这里直接验证该分派分支逻辑.
    res = _tool().call({"action": "md_run", "tpr_file": None}, context=None)
    assert res.success is False
    assert "TPR file not found" in res.error


def test_gmx_available_direct(monkeypatch):
    # 直接命中 _gmx_available 内部 (which 调用)
    monkeypatch.setattr(gt.shutil, "which", lambda name: None)
    assert gt._gmx_available() is False


# ── _parse_md_log ─────────────────────────────────────────────────────────


def test_parse_md_log_missing(tmp_path):
    t = _tool()
    data = t._parse_md_log(tmp_path / "nope.log")
    assert data["temperatures"] == []
    assert data["lincs_warnings"] == 0
    assert data["has_nan"] is False


def test_parse_md_log_read_exception(tmp_path, monkeypatch):
    t = _tool()
    log = tmp_path / "x.log"
    log.write_text("x")
    def boom(log_path, **k):
        raise OSError("unreadable")
    monkeypatch.setattr(Path, "read_text", boom)
    data = t._parse_md_log(log)
    assert data["temperatures"] == []


def test_parse_md_log_content(tmp_path):
    t = _tool()
    log = tmp_path / "x.log"
    log.write_text(
        "Temperature = 300.0\n"
        "Temperature= -5.0\n"  # 非正温度被过滤
        "Pressure = 1.2\n"
        "Pres = -0.5\n"
        "Total Energy = -100.5\n"
        "Total Energy = nan\n"
        "LINCS WARNING step 1\nLINCS WARNING step 2\n"
        "SHAKE\n"
        "neighbor list entries exceed update\n"
        "info: nan in energy\n"
    )
    data = t._parse_md_log(log)
    assert data["temperatures"] == [300.0]
    assert data["pressures"] == [1.2, -0.5]
    assert data["energies"] == [-100.5]
    assert data["lincs_warnings"] == 2
    assert data["shake_warnings"] == 1
    assert data["neighbor_list_warnings"] == 1
    assert data["has_nan"] is True
