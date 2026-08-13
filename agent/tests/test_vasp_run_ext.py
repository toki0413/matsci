"""sim/vasp_tool.py `_run_vasp` 执行循环集成测试 — mock sandbox 覆盖
成功 / 硬失败 / SCF 未收敛软失败重试 / 物理审计报错 / 超时 / 异常.

配合 test_vasp_rust_ext.py + test_vasp_tool_integration_ext.py,
把 sim/vasp_tool.py 覆盖率从 61% 提升到 85%+.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from huginn.tools.sim import vasp_tool as vt


pytestmark = pytest.mark.anyio


SYNTH_OUTCAR = """VASP output
ENCUT  =  520.0 eV
ISPIN  =  2
NELM   =  100
FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  free  energy   TOTEN  =       -10.1234 eV
TOTAL-FORCE (eV/Angst)
  0.000  0.000  0.000    0.010  0.020  0.030
  1.750  1.750  0.000   -0.010 -0.020 -0.030
reached required accuracy - stopping structural energy minimisation
"""


def _install_auditors(monkeypatch, has_errors=False, findings=None):
    """注入 fake PhysicsAuditor."""
    import sys

    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Audit:
        def __init__(self, has_errors, findings):
            self.has_errors = has_errors
            self.findings = findings or []

        def to_dict(self):
            return {"has_errors": self.has_errors}

    class _Auditor:
        def audit(self, *a, **k):
            return _Audit(has_errors, findings)

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)


def _install_provenance(monkeypatch):
    prov_mod = types.ModuleType("huginn.provenance")
    class _Cap:
        def to_dict(self):
            return {"provenance": True}
    prov_mod.capture = lambda *a, **k: _Cap()
    import sys
    monkeypatch.setitem(sys.modules, "huginn.provenance", prov_mod)


def _setup_dir(tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    (d / "POSCAR").write_text("cell\n", encoding="utf-8")
    (d / "OUTCAR").write_text(SYNTH_OUTCAR, encoding="utf-8")
    (d / "INCAR").write_text("ENCUT = 400\n", encoding="utf-8")
    return d


def _args(**kw):
    base = {"action": "relax", "working_dir": "."}
    base.update(kw)
    return vt.VaspToolInput(**base)


def _ctx():
    return type("C", (), {"workspace": "."})()


# ── _run_vasp 成功 ───────────────────────────────────────────────────────


async def test_run_vasp_success(monkeypatch, tmp_path):
    d = _setup_dir(tmp_path)
    _install_auditors(monkeypatch, has_errors=False)
    _install_provenance(monkeypatch)

    class _Sb:
        def run(self, *a, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = vt.VaspTool(vasp_executable="/usr/bin/vasp")
    tool.sandbox = _Sb()
    res = await tool._run_vasp(_args(working_dir=str(d)), d)
    assert res.success is True
    assert res.data["status"] == "completed"
    assert res.data["physics_audit"] is not None
    assert res.data["uq_hint"]["tool"] == "gp_tool"


async def test_run_vasp_hard_failure(monkeypatch, tmp_path):
    d = _setup_dir(tmp_path)
    _install_provenance(monkeypatch)

    class _Sb:
        def run(self, *a, **k):
            return types.SimpleNamespace(returncode=1, stderr="some error", stdout="")

    tool = vt.VaspTool(vasp_executable="/usr/bin/vasp")
    tool.sandbox = _Sb()
    res = await tool._run_vasp(_args(working_dir=str(d)), d)
    assert res.success is False
    assert "some error" in res.error


async def test_run_vasp_soft_failure_retry(monkeypatch, tmp_path):
    """returncode=0 但 OUTCAR 未收敛 → 软失败重试."""
    d = tmp_path / "wd"
    d.mkdir()
    (d / "POSCAR").write_text("cell\n", encoding="utf-8")
    # 无收敛标记的 OUTCAR → converged=False
    (d / "OUTCAR").write_text(
        "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
        "  free  energy   TOTEN  =       -10.0 eV\n",
        encoding="utf-8",
    )
    (d / "INCAR").write_text("ENCUT = 400\n", encoding="utf-8")
    _install_provenance(monkeypatch)
    _install_auditors(monkeypatch, has_errors=False)

    calls = {"n": 0}
    class _Sb:
        def run(self, *a, **k):
            calls["n"] += 1
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = vt.VaspTool(vasp_executable="/usr/bin/vasp")
    tool.sandbox = _Sb()
    res = await tool._run_vasp(_args(working_dir=str(d)), d)
    assert res.success is False
    assert "converge" in res.error.lower() or "NELM" in res.error


async def test_run_vasp_physics_audit_error(monkeypatch, tmp_path):
    d = _setup_dir(tmp_path)
    _install_provenance(monkeypatch)
    _install_auditors(monkeypatch, has_errors=True, findings=[
        types.SimpleNamespace(message="unbound energy", severity="error")
    ])

    class _Sb:
        def run(self, *a, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = vt.VaspTool(vasp_executable="/usr/bin/vasp")
    tool.sandbox = _Sb()
    res = await tool._run_vasp(_args(working_dir=str(d)), d)
    assert res.success is False
    assert "unbound energy" in res.error


async def test_run_vasp_timeout(monkeypatch, tmp_path):
    d = _setup_dir(tmp_path)

    class _Sb:
        def run(self, *a, **k):
            import subprocess
            raise subprocess.TimeoutExpired("cmd", 60)

    tool = vt.VaspTool(vasp_executable="/usr/bin/vasp")
    tool.sandbox = _Sb()
    res = await tool._run_vasp(_args(working_dir=str(d)), d)
    assert res.success is False
    assert "timed out" in res.error


async def test_run_vasp_generic_exception(monkeypatch, tmp_path):
    d = _setup_dir(tmp_path)

    class _Sb:
        def run(self, *a, **k):
            raise RuntimeError("sandbox exploded")

    tool = vt.VaspTool(vasp_executable="/usr/bin/vasp")
    tool.sandbox = _Sb()
    res = await tool._run_vasp(_args(working_dir=str(d)), d)
    assert res.success is False
    assert "VASP execution failed" in res.error


async def test_call_runs_real_when_executable(tmp_path, monkeypatch):
    """call() 当 vasp_executable 存在时走 _run_vasp."""
    d = _setup_dir(tmp_path)
    _install_auditors(monkeypatch, has_errors=False)
    _install_provenance(monkeypatch)

    class _Sb:
        def run(self, *a, **k):
            return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    tool = vt.VaspTool(vasp_executable="/usr/bin/vasp")
    tool.sandbox = _Sb()
    res = await tool.call(_args(working_dir=str(d)), _ctx())
    assert res.success is True