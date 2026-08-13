"""elmer_tool.py 集成路径补测 — 覆盖 call 分派、is_read_only/is_destructive、
_resolve_sif(sif_content/sif_file/缺失)、_solve_sif(缺 sif/未装 Elmer 导出/
沙箱成功/失败/审计异常吞掉/沙箱拦截/超时)、_validate_sif(缺 sif/完整校验/
缺失区块)、_mesh_to_elmer(缺 mesh_dir/目录不存在/未装 ElmerGrid/成功/失败/
沙箱拦截/超时).

配合把 elmer_tool.py 覆盖率提升到 90%+.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from huginn.tools.sim.elmer_tool import (
    ElmerTool,
    ElmerToolInput,
    _elmer_available,
    _elmergrid_available,
)

pytestmark = pytest.mark.anyio


def _tool(**kw):
    return ElmerTool(**kw)


def _model(**kw):
    base = {"action": "solve_sif"}
    base.update(kw)
    return ElmerToolInput(**base)


def _install_auditor(monkeypatch, has_errors=False, findings=None):
    mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Audit:
        def __init__(self, has_errors, findings):
            self.has_errors = has_errors
            self.findings = findings

        def to_dict(self):
            return {"has_errors": self.has_errors, "findings": len(self.findings)}

    class _Auditor:
        def audit(self, *a, **k):
            return _Audit(has_errors, list(findings or []))

    mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", mod)


def _sb(returncode=0, stdout="", stderr=""):
    from types import SimpleNamespace

    proc = SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    class _Sb:
        def run(self, cmd, **kw):
            return proc

    return _Sb()


# ── is_read_only / is_destructive ────────────────────────────────────────


def test_flags():
    tool = _tool()
    assert tool.is_read_only(_model(action="validate_sif")) is True
    assert tool.is_read_only(_model(action="solve_sif")) is False
    assert tool.is_destructive(_model(action="solve_sif")) is True
    assert tool.is_destructive(_model(action="validate_sif")) is False


# ── _resolve_sif ─────────────────────────────────────────────────────────


def test_resolve_sif_content(tmp_path):
    tool = _tool()
    p = tool._resolve_sif(_model(sif_content="Header\nSimulation\n"), tmp_path)
    assert p.name == "case.sif"
    assert p.read_text(encoding="utf-8") == "Header\nSimulation\n"


def test_resolve_sif_file_absolute(tmp_path):
    f = tmp_path / "a.sif"
    f.write_text("x", encoding="utf-8")
    tool = _tool()
    assert tool._resolve_sif(_model(sif_file=str(f)), tmp_path) == f


def test_resolve_sif_file_relative(tmp_path):
    f = tmp_path / "b.sif"
    f.write_text("x", encoding="utf-8")
    tool = _tool()
    p = tool._resolve_sif(_model(sif_file="b.sif"), tmp_path)
    assert p == f


def test_resolve_sif_missing(tmp_path):
    tool = _tool()
    assert tool._resolve_sif(_model(sif_file=str(tmp_path / "nope.sif")), tmp_path) is None
    assert tool._resolve_sif(_model(), tmp_path) is None


# ── _solve_sif ───────────────────────────────────────────────────────────


def test_solve_sif_no_sif(tmp_path):
    res = _tool()._solve_sif(_model(), tmp_path)
    assert res.success is False
    assert "requires" in res.error


def test_solve_sif_not_installed_exports(tmp_path, monkeypatch):
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmer_available", lambda: False)
    res = _tool()._solve_sif(_model(sif_content="Header\n"), tmp_path)
    assert res.success is True
    assert res.data["status"] == "sif_exported"
    assert res.data["sif_path"].endswith("case.sif")


def test_solve_sif_success(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmer_available", lambda: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="done", stderr="")
    res = tool._solve_sif(_model(sif_content="Header\n"), tmp_path)
    assert res.success is True
    assert res.data["returncode"] == 0
    assert res.data["physics_audit"]["has_errors"] is False
    assert "completed" in res.data["message"]


def test_solve_sif_failure(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmer_available", lambda: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=2, stdout="", stderr="elmer error")
    res = tool._solve_sif(_model(sif_content="Header\n"), tmp_path)
    assert res.success is False
    assert "Elmer solve failed" in res.error


def test_solve_sif_audit_exception_swallowed(monkeypatch, tmp_path):
    mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", mod)
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmer_available", lambda: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="ok")
    res = tool._solve_sif(_model(sif_content="Header\n"), tmp_path)
    assert res.success is True
    assert "physics_audit" not in res.data


def test_solve_sif_sandbox_blocked(monkeypatch, tmp_path):
    from huginn.tools.sim import elmer_tool as et

    monkeypatch.setattr(et, "_elmer_available", lambda: True)

    class _Sb:
        def run(self, cmd, **kw):
            raise et.SandboxError("blocked")

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._solve_sif(_model(sif_content="Header\n"), tmp_path)
    assert res.success is False
    assert "blocked by sandbox" in res.error


def test_solve_sif_timeout(monkeypatch, tmp_path):
    from huginn.tools.sim import elmer_tool as et

    monkeypatch.setattr(et, "_elmer_available", lambda: True)

    class _Sb:
        def run(self, cmd, **kw):
            raise subprocess.TimeoutExpired("ElmerSolver", 600.0)

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._solve_sif(_model(sif_content="Header\n"), tmp_path)
    assert res.success is False
    assert "timed out" in res.error


# ── _validate_sif ────────────────────────────────────────────────────────


def test_validate_sif_no_sif(tmp_path):
    res = _tool()._validate_sif(_model(), tmp_path)
    assert res.success is False
    assert "requires" in res.error


def test_validate_sif_valid(tmp_path):
    content = (
        "Header\n"
        "Simulation\n"
        "Max Output Level = 5\n"
        "Steady State Max = 1\n"
        "Coordinate System = Cartesian\n"
        "Equation\n"
    )
    res = _tool()._validate_sif(_model(sif_content=content, action="validate_sif"), tmp_path)
    assert res.success is True
    assert res.data["valid"] is True
    assert res.data["issues"] == []


def test_validate_sif_issues(tmp_path):
    res = _tool()._validate_sif(_model(sif_content="just text", action="validate_sif"), tmp_path)
    assert res.success is True
    assert res.data["valid"] is False
    assert "Missing 'Header' section" in res.data["issues"]
    assert "Missing 'Simulation' section" in res.data["issues"]


# ── _mesh_to_elmer ───────────────────────────────────────────────────────


def test_mesh_to_elmer_no_mesh_dir(tmp_path):
    res = _tool()._mesh_to_elmer(_model(action="mesh_to_elmer"), tmp_path)
    assert res.success is False
    assert "requires 'mesh_dir'" in res.error


def test_mesh_to_elmer_dir_not_found(tmp_path):
    res = _tool()._mesh_to_elmer(
        _model(action="mesh_to_elmer", mesh_dir=str(tmp_path / "nope")), tmp_path
    )
    assert res.success is False
    assert "Mesh directory not found" in res.error


def test_mesh_to_elmer_no_elmergrid(tmp_path, monkeypatch):
    d = tmp_path / "mesh"
    d.mkdir()
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmergrid_available", lambda: False)
    res = _tool()._mesh_to_elmer(
        _model(action="mesh_to_elmer", mesh_dir=str(d)), tmp_path
    )
    assert res.success is True
    assert res.data["status"] == "skipped"


def test_mesh_to_elmer_success(tmp_path, monkeypatch):
    d = tmp_path / "mesh"
    d.mkdir()
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmergrid_available", lambda: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="ok")
    res = tool._mesh_to_elmer(
        _model(action="mesh_to_elmer", mesh_dir=str(d)), tmp_path
    )
    assert res.success is True
    assert res.data["returncode"] == 0
    assert "converted" in res.data["message"]


def test_mesh_to_elmer_failure(tmp_path, monkeypatch):
    d = tmp_path / "mesh"
    d.mkdir()
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmergrid_available", lambda: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=1, stderr="grid fail")
    res = tool._mesh_to_elmer(
        _model(action="mesh_to_elmer", mesh_dir=str(d)), tmp_path
    )
    assert res.success is False
    assert "ElmerGrid failed" in res.error


def test_mesh_to_elmer_sandbox_blocked(tmp_path, monkeypatch):
    from huginn.tools.sim import elmer_tool as et

    d = tmp_path / "mesh"
    d.mkdir()
    monkeypatch.setattr(et, "_elmergrid_available", lambda: True)

    class _Sb:
        def run(self, cmd, **kw):
            raise et.SandboxError("blocked")

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._mesh_to_elmer(
        _model(action="mesh_to_elmer", mesh_dir=str(d)), tmp_path
    )
    assert res.success is False
    assert "blocked by sandbox" in res.error


def test_mesh_to_elmer_timeout(tmp_path, monkeypatch):
    from huginn.tools.sim import elmer_tool as et

    d = tmp_path / "mesh"
    d.mkdir()
    monkeypatch.setattr(et, "_elmergrid_available", lambda: True)

    class _Sb:
        def run(self, cmd, **kw):
            raise subprocess.TimeoutExpired("ElmerGrid", 120.0)

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._mesh_to_elmer(
        _model(action="mesh_to_elmer", mesh_dir=str(d)), tmp_path
    )
    assert res.success is False
    assert "timed out" in res.error


# ── helpers ──────────────────────────────────────────────────────────────


def test_elmer_available(monkeypatch):
    monkeypatch.setattr("huginn.tools.sim.elmer_tool.shutil.which",
                        lambda cmd: "/usr/bin/ElmerSolver" if cmd == "ElmerSolver" else None)
    assert _elmer_available() is True
    monkeypatch.setattr("huginn.tools.sim.elmer_tool.shutil.which", lambda cmd: None)
    assert _elmer_available() is False
    assert _elmergrid_available() is False


# ── call 分派 ────────────────────────────────────────────────────────────


def test_call_dispatch_solve(monkeypatch, tmp_path):
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmer_available", lambda: False)
    res = _tool().call(
        {"action": "solve_sif", "sif_content": "Header\n", "working_dir": str(tmp_path)}
    )
    assert res.success is True
    assert res.data["status"] == "sif_exported"


def test_call_dispatch_validate(monkeypatch, tmp_path):
    res = _tool().call(
        {"action": "validate_sif", "sif_content": "Header\nSimulation\n", "working_dir": str(tmp_path)}
    )
    assert res.success is True
    assert res.data["action"] == "validate_sif"


def test_call_dispatch_mesh_relative_dir(monkeypatch, tmp_path):
    """mesh_dir 相对路径 → 拼到 work_dir."""
    d = tmp_path / "mesh"
    d.mkdir()
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmergrid_available", lambda: False)
    res = _tool().call(
        {"action": "mesh_to_elmer", "mesh_dir": "mesh", "working_dir": str(tmp_path)}
    )
    assert res.success is True
    assert res.data["status"] == "skipped"


def test_call_no_working_dir(tmp_path, monkeypatch):
    """无 working_dir → 用 cwd."""
    monkeypatch.setattr("huginn.tools.sim.elmer_tool._elmer_available", lambda: False)
    res = _tool().call({"action": "solve_sif", "sif_content": "Header\n"})
    assert res.success is True
