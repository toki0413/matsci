"""fenics_tool.py 集成路径补测 — 覆盖 is_read_only/is_destructive、call 分派、
_fenics_available(成功/沙箱拦截/其他异常)、_solve_pde(缺 script/无 FEniCS 生成脚本/
成功/失败/审计异常吞掉/沙箱拦截/超时)、_mesh_info(缺 mesh_file/相对路径解析/文件缺失/
成功·解析 int·读取异常/非零退出/沙箱拦截/超时)、_convergence_check(不足 2 文件/
成功多样性/相对 L2/error 行/未解析/超时/异常/审计异常吞掉).

配合把 fenics_tool.py 覆盖率提升到 90%+.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from huginn.tools.sim.fenics_tool import FenicsTool, FenicsToolInput, _fenics_available

pytestmark = pytest.mark.anyio


def _tool(**kw):
    return FenicsTool(**kw)


def _model(**kw):
    base = {"action": "solve_pde"}
    base.update(kw)
    return FenicsToolInput(**base)


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


# ── flags ────────────────────────────────────────────────────────────────


def test_flags():
    tool = _tool()
    assert tool.is_read_only(_model(action="mesh_info")) is True
    assert tool.is_read_only(_model(action="solve_pde")) is False
    assert tool.is_destructive(_model(action="solve_pde")) is True
    assert tool.is_destructive(_model(action="mesh_info")) is False


# ── _fenics_available ────────────────────────────────────────────────────


def test_fenics_available_true():
    class _Sb:
        def run(self, cmd, **kw):
            from types import SimpleNamespace
            return SimpleNamespace(returncode=0)

    assert _fenics_available(_Sb()) is True


def test_fenics_available_false():
    class _Sb:
        def run(self, cmd, **kw):
            from types import SimpleNamespace
            return SimpleNamespace(returncode=1)

    assert _fenics_available(_Sb()) is False


def test_fenics_available_sandbox_blocked():
    from huginn.tools.sim import fenics_tool as ft

    class _Sb:
        def run(self, cmd, **kw):
            raise ft.SandboxError("blocked")

    assert _fenics_available(_Sb()) is False


def test_fenics_available_exception():
    class _Sb:
        def run(self, cmd, **kw):
            raise RuntimeError("boom")

    assert _fenics_available(_Sb()) is False


# ── _solve_pde ───────────────────────────────────────────────────────────


def test_solve_pde_no_script(tmp_path):
    res = _tool()._solve_pde(_model(), tmp_path)
    assert res.success is False
    assert "requires 'script'" in res.error


def test_solve_pde_no_fenics_generates_script(monkeypatch, tmp_path):
    monkeypatch.setattr("huginn.tools.sim.fenics_tool._fenics_available", lambda s: False)
    res = _tool()._solve_pde(_model(script="print('hi')"), tmp_path)
    assert res.success is True
    assert res.data["status"] == "script_generated"
    assert res.data["script_path"].endswith("fenics_solve.py")


def test_solve_pde_success(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    monkeypatch.setattr("huginn.tools.sim.fenics_tool._fenics_available", lambda s: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="done")
    res = tool._solve_pde(_model(script="print('hi')"), tmp_path)
    assert res.success is True
    assert res.data["returncode"] == 0
    assert res.data["physics_audit"]["has_errors"] is False


def test_solve_pde_failure(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    monkeypatch.setattr("huginn.tools.sim.fenics_tool._fenics_available", lambda s: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=1, stderr="fenics err")
    res = tool._solve_pde(_model(script="print('hi')"), tmp_path)
    assert res.success is False
    assert "FEniCS solve failed" in res.error


def test_solve_pde_audit_exception_swallowed(monkeypatch, tmp_path):
    mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", mod)
    monkeypatch.setattr("huginn.tools.sim.fenics_tool._fenics_available", lambda s: True)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="ok")
    res = tool._solve_pde(_model(script="print('hi')"), tmp_path)
    assert res.success is True
    assert "physics_audit" not in res.data


def test_solve_pde_sandbox_blocked(monkeypatch, tmp_path):
    from huginn.tools.sim import fenics_tool as ft

    monkeypatch.setattr(ft, "_fenics_available", lambda s: True)

    class _Sb:
        def run(self, cmd, **kw):
            raise ft.SandboxError("blocked")

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._solve_pde(_model(script="print('hi')"), tmp_path)
    assert res.success is False
    assert "blocked by sandbox" in res.error


def test_solve_pde_timeout(monkeypatch, tmp_path):
    from huginn.tools.sim import fenics_tool as ft

    monkeypatch.setattr(ft, "_fenics_available", lambda s: True)

    class _Sb:
        def run(self, cmd, **kw):
            raise subprocess.TimeoutExpired("python", 300.0)

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._solve_pde(_model(script="print('hi')"), tmp_path)
    assert res.success is False
    assert "timed out" in res.error


# ── _mesh_info ───────────────────────────────────────────────────────────


def test_mesh_info_no_file(tmp_path):
    res = _tool()._mesh_info(_model(action="mesh_info"), tmp_path)
    assert res.success is False
    assert "requires 'mesh_file'" in res.error


def test_mesh_info_file_not_found(tmp_path):
    res = _tool()._mesh_info(
        _model(action="mesh_info", mesh_file=str(tmp_path / "nope.xml")), tmp_path
    )
    assert res.success is False
    assert "Mesh file not found" in res.error


def test_mesh_info_relative_resolution(tmp_path):
    f = tmp_path / "mesh.xml"
    f.write_text("x", encoding="utf-8")
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="dim 2\nnum_vertices 10\nnum_cells 8\n")
    res = tool._mesh_info(_model(action="mesh_info", mesh_file="mesh.xml"), tmp_path)
    assert res.success is True
    assert res.data["dim"] == 2
    assert res.data["num_vertices"] == 10
    assert res.data["num_cells"] == 8


def test_mesh_info_parse_int_value(tmp_path):
    f = tmp_path / "mesh.xml"
    f.write_text("x", encoding="utf-8")
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="dim 2\nnum_vertices 10\n")
    res = tool._mesh_info(_model(action="mesh_info", mesh_file=str(f)), tmp_path)
    assert isinstance(res.data["dim"], int)
    assert res.data["dim"] == 2


def test_mesh_info_nonzero_exit(tmp_path):
    f = tmp_path / "mesh.xml"
    f.write_text("x", encoding="utf-8")
    tool = _tool()
    tool.sandbox = _sb(returncode=1, stderr="mesh err")
    res = tool._mesh_info(_model(action="mesh_info", mesh_file=str(f)), tmp_path)
    assert res.success is False
    assert "mesh query failed" in res.error


def test_mesh_info_sandbox_blocked(tmp_path):
    from huginn.tools.sim import fenics_tool as ft

    f = tmp_path / "mesh.xml"
    f.write_text("x", encoding="utf-8")

    class _Sb:
        def run(self, cmd, **kw):
            raise ft.SandboxError("blocked")

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._mesh_info(_model(action="mesh_info", mesh_file=str(f)), tmp_path)
    assert res.success is False
    assert "blocked by sandbox" in res.error


def test_mesh_info_timeout(tmp_path):

    f = tmp_path / "mesh.xml"
    f.write_text("x", encoding="utf-8")

    class _Sb:
        def run(self, cmd, **kw):
            raise subprocess.TimeoutExpired("python", 30.0)

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._mesh_info(_model(action="mesh_info", mesh_file=str(f)), tmp_path)
    assert res.success is False
    assert "timed out" in res.error


# ── _convergence_check ───────────────────────────────────────────────────


def test_convergence_check_insufficient(tmp_path):
    res = _tool()._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf"]), tmp_path
    )
    assert res.success is False
    assert "at least 2" in res.error


def test_convergence_check_success_errornorm(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="diff 0.001\n")
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.success is True
    assert res.data["converged"] is True
    assert res.data["method"] == "errornorm"
    assert res.data["differences"] == [0.001]


def test_convergence_check_nonconverged(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="diff 0.05\n")
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.data["converged"] is False


def test_convergence_check_relative_l2(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="diff_rel 0.5\n")
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.data["method"] == "relative_l2"
    assert res.data["converged"] is False


def test_convergence_check_error_line(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="error could not read\n")
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.data["differences"][0] != res.data["differences"][0]  # NaN


def test_convergence_check_bad_float_line(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="diff notanum\n")
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    # 行解析失败 → 未解析 → NaN
    assert res.data["differences"][0] != res.data["differences"][0]


def test_convergence_check_unparsed(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="garbage no numbers\n")
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.data["differences"][0] != res.data["differences"][0]  # NaN appended


def test_convergence_check_timeout(tmp_path):

    class _Sb:
        def run(self, cmd, **kw):
            raise subprocess.TimeoutExpired("python", 60.0)

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.data["differences"][0] != res.data["differences"][0]  # NaN


def test_convergence_check_exception(tmp_path):
    class _Sb:
        def run(self, cmd, **kw):
            raise RuntimeError("boom")

    tool = _tool()
    tool.sandbox = _Sb()
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.data["differences"][0] != res.data["differences"][0]  # NaN


def test_convergence_check_audit_exception_swallowed(tmp_path):
    mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    mod.PhysicsAuditor = _Auditor
    sys.modules["huginn.execution.physics_auditor"] = mod
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="diff 0.001\n")
    res = tool._convergence_check(
        _model(action="convergence_check", solution_files=["a.xdmf", "b.xdmf"]),
        tmp_path,
    )
    assert res.success is True
    assert "physics_audit" not in res.data


def test_mesh_info_non_int_value(tmp_path):
    """int() 解析失败 → 保留字符串."""
    f = tmp_path / "mesh.xml"
    f.write_text("x", encoding="utf-8")
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="normalized 1.5\n")
    res = tool._mesh_info(_model(action="mesh_info", mesh_file=str(f)), tmp_path)
    assert res.success is True
    assert res.data["normalized"] == "1.5"


# ── call 分派 ────────────────────────────────────────────────────────────


def test_call_dispatch_solve_nofenics(monkeypatch, tmp_path):
    monkeypatch.setattr("huginn.tools.sim.fenics_tool._fenics_available", lambda s: False)
    res = _tool().call(
        {"action": "solve_pde", "script": "print('hi')", "working_dir": str(tmp_path)}
    )
    assert res.success is True
    assert res.data["status"] == "script_generated"


def test_call_dispatch_meshinfo(monkeypatch, tmp_path):
    f = tmp_path / "m.xml"
    f.write_text("x", encoding="utf-8")
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="dim 2\n")
    res = tool.call({"action": "mesh_info", "mesh_file": str(f), "working_dir": str(tmp_path)})
    assert res.success is True
    assert res.data["dim"] == 2


def test_call_dispatch_convergence(monkeypatch, tmp_path):
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool()
    tool.sandbox = _sb(returncode=0, stdout="diff 0.001\n")
    res = tool.call(
        {
            "action": "convergence_check",
            "solution_files": ["a.xdmf", "b.xdmf"],
            "working_dir": str(tmp_path),
        }
    )
    assert res.success is True
    assert res.data["converged"] is True


def test_call_no_working_dir(monkeypatch, tmp_path):
    monkeypatch.setattr("huginn.tools.sim.fenics_tool._fenics_available", lambda s: False)
    res = _tool().call({"action": "solve_pde", "script": "print('hi')"})
    assert res.success is True
