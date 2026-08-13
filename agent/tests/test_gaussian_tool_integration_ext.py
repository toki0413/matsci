"""gaussian_tool.py 集成路径补测 — 覆盖 _find_gaussian(env/which/无)、
validate_input(缺/成功)、call(缺 workdir/缺 gjf/route overrides/parse/缺 executable/
executable 解析/run)、_run_gaussian(成功/硬失败/SCF 未收敛软失败/opt 未收敛软失败/
物理审计报错/审计异常/autofix 重试/sandbox 异常)、_read_route_params、
_apply_route_overrides、_format_keyword、_find_gjf、_parse_and_return、
_mock_result、_get_returncode、_get_stderr、_parse_log 全分支.

配合已有测试, 把 gaussian_tool.py 覆盖率从 22% 提升到 85%+.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from huginn.tools.sim.gaussian_tool import GaussianTool, GaussianToolInput

pytestmark = pytest.mark.anyio


def _tool(**kw):
    return GaussianTool(**kw)


def _args(**kw):
    base = {"action": "sp", "working_dir": "."}
    base.update(kw)
    return GaussianToolInput(**base)


def _run(coro):
    return asyncio.run(coro)


def _install_auditor(monkeypatch, has_errors=False, findings=None):
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

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


def _no_autofix(monkeypatch):
    monkeypatch.setattr(GaussianTool, "_try_autofix", lambda self, p, e: None)


def _sb_fake(returncode=0, stderr="", log_content=None, log_path=None):
    class _Sb:
        def run(self, cmd, cwd=None, timeout=None):
            if log_content is not None and log_path is not None:
                log_path.write_text(log_content, encoding="utf-8")
            return types.SimpleNamespace(returncode=returncode, stderr=stderr)

    return _Sb()


@pytest.fixture
def workdir(tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    (d / "in.gjf").write_text("#p hf/6-31g\n\nmol\n\n0 1\nH 0 0 0\n", encoding="utf-8")
    return d


# ── _find_gaussian ───────────────────────────────────────────────────────


def test_find_gaussian_env_hit(monkeypatch, tmp_path):
    exe = tmp_path / "g16"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setenv("GAUSSIAN_EXECUTABLE", str(exe))
    assert _tool(gaussian_executable=None)._find_gaussian() == str(exe)


def test_find_gaussian_env_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GAUSSIAN_EXECUTABLE", str(tmp_path / "nope"))
    assert _tool(gaussian_executable=None)._find_gaussian() is None


def test_find_gaussian_which(monkeypatch):
    monkeypatch.delenv("GAUSSIAN_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        "huginn.tools.sim.gaussian_tool.shutil.which",
        lambda name: "/usr/local/bin/g16" if name == "g16" else None,
    )
    assert _tool(gaussian_executable=None)._find_gaussian() == "/usr/local/bin/g16"


def test_find_gaussian_none(monkeypatch):
    monkeypatch.delenv("GAUSSIAN_EXECUTABLE", raising=False)
    monkeypatch.setattr("huginn.tools.sim.gaussian_tool.shutil.which", lambda n: None)
    assert _tool(gaussian_executable=None)._find_gaussian() is None


# ── validate_input ───────────────────────────────────────────────────────


async def test_validate_input_missing_dir(tmp_path):
    tool = _tool(gaussian_executable=None)
    res = await tool.validate_input(
        _args(working_dir=str(tmp_path / "nope")), None
    )
    assert res.result is False
    assert res.error_code == 404


async def test_validate_input_ok(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    tool = _tool(gaussian_executable=None)
    res = await tool.validate_input(_args(working_dir=str(tmp_path)), None)
    assert res.result is True


# ── call 分支 ────────────────────────────────────────────────────────────


async def test_call_missing_workdir(tmp_path):
    tool = _tool(gaussian_executable=None)
    res = await tool.call(
        _args(working_dir=str(tmp_path / "nope")), None
    )
    assert res.success is False
    assert "Working directory not found" in res.error


async def test_call_no_gjf(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    tool = _tool(gaussian_executable=None)
    res = await tool.call(_args(working_dir=str(tmp_path)), None)
    assert res.success is False
    assert "No .gjf input file" in res.error


async def test_call_parse_action(workdir):
    (workdir / "in.log").write_text(
        "SCF Done:  E(RHF) =  -76.0\n     Normal termination\n", encoding="utf-8"
    )
    tool = _tool(gaussian_executable=None)
    res = await tool.call(_args(action="parse", working_dir=str(workdir)), None)
    assert res.success is True
    assert res.data["parsed"]["energy"] == pytest.approx(-76.0)
    assert res.data["parsed"]["normal_termination"] is True


async def test_call_route_overrides_applied(workdir):
    tool = _tool(gaussian_executable=None)
    await tool.call(
        _args(action="parse", working_dir=str(workdir), route_overrides={"scf": "xqc"}),
        None,
    )
    text = (workdir / "in.gjf").read_text(encoding="utf-8")
    assert "SCF=(XQC)" in text


async def test_call_executable_resolved_str(monkeypatch, workdir):
    _install_auditor(monkeypatch)
    resolver = types.ModuleType("huginn.tools.sim.executable_resolver")
    resolver.resolve_executable = lambda name: "/usr/bin/g16"
    monkeypatch.setitem(sys.modules, "huginn.tools.sim.executable_resolver", resolver)

    class _Sb:
        def run(self, cmd, cwd=None, timeout=None):
            (Path(cwd) / "in.log").write_text(
                "SCF Done:  E(RHF) =  -76.0\n     Normal termination\n",
                encoding="utf-8",
            )
            return types.SimpleNamespace(returncode=0, stderr="")

    tool = _tool(gaussian_executable=None)
    tool.sandbox = _Sb()
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is True
    assert tool.gaussian_executable == "/usr/bin/g16"


async def test_call_executable_not_resolved(monkeypatch, workdir):
    resolver = types.ModuleType("huginn.tools.sim.executable_resolver")
    resolver.resolve_executable = lambda name: types.SimpleNamespace(
        install_hint="Install g16.", to_dict=lambda: {"name": "g16"}
    )
    monkeypatch.setitem(sys.modules, "huginn.tools.sim.executable_resolver", resolver)
    tool = _tool(gaussian_executable=None)
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is False
    assert "Gaussian executable not found" in res.error
    assert res.metadata["needs_resolution"] is True


# ── _run_gaussian 分支 ───────────────────────────────────────────────────


async def test_run_gaussian_success(monkeypatch, workdir):
    _install_auditor(monkeypatch)
    tool = _tool(gaussian_executable="/usr/bin/g16")
    tool.sandbox = _sb_fake(
        returncode=0,
        log_content="SCF Done:  E(RHF) =  -76.0\n     Normal termination\n",
        log_path=workdir / "in.log",
    )
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is True
    assert res.data["parsed"]["energy"] == pytest.approx(-76.0)


async def test_run_gaussian_hard_failure(monkeypatch, workdir):
    _no_autofix(monkeypatch)
    tool = _tool(gaussian_executable="/usr/bin/g16")
    tool.sandbox = _sb_fake(returncode=1, stderr="link error")
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is False
    assert "link error" in res.error


async def test_run_gaussian_scf_not_converged(monkeypatch, workdir):
    _no_autofix(monkeypatch)
    tool = _tool(gaussian_executable="/usr/bin/g16")
    tool.sandbox = _sb_fake(
        returncode=0, log_content="Convergence failure\n", log_path=workdir / "in.log"
    )
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is False
    assert "SCF did not converge" in res.error


async def test_run_gaussian_opt_not_converged(monkeypatch, workdir):
    _no_autofix(monkeypatch)
    tool = _tool(gaussian_executable="/usr/bin/g16")
    tool.sandbox = _sb_fake(returncode=0, log_content="", log_path=workdir / "in.log")
    res = await tool.call(_args(action="opt", working_dir=str(workdir)), None)
    assert res.success is False
    assert "Optimization did not converge" in res.error


async def test_run_gaussian_audit_error(monkeypatch, workdir):
    _install_auditor(monkeypatch, has_errors=True, findings=[
        types.SimpleNamespace(severity="error", message="imaginary freq"),
    ])
    _no_autofix(monkeypatch)
    tool = _tool(gaussian_executable="/usr/bin/g16")
    tool.sandbox = _sb_fake(
        returncode=0,
        log_content="SCF Done:  E(RHF) =  -76.0\n     Normal termination\n",
        log_path=workdir / "in.log",
    )
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is False
    assert "Physics audit found errors" in res.error


async def test_run_gaussian_audit_exception(monkeypatch, workdir):
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    tool = _tool(gaussian_executable="/usr/bin/g16")
    tool.sandbox = _sb_fake(
        returncode=0,
        log_content="SCF Done:  E(RHF) =  -76.0\n     Normal termination\n",
        log_path=workdir / "in.log",
    )
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is True
    assert "physics_audit" not in res.data


async def test_run_gaussian_autofix_retry(monkeypatch, workdir):
    _install_auditor(monkeypatch)
    calls = {"n": 0}
    logs = {
        1: "Convergence failure\n",
        2: "SCF Done:  E(RHF) =  -76.0\n     Normal termination\n",
    }

    class _Sb:
        def run(self, cmd, cwd=None, timeout=None):
            calls["n"] += 1
            lf = Path(cwd) / "in.log"
            lf.write_text(logs[calls["n"]], encoding="utf-8")
            return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        GaussianTool, "_try_autofix",
        lambda self, p, e: {"fixes": {"scf": "xqc"}, "reasoning": "tighten"},
    )
    tool = _tool(gaussian_executable="/usr/bin/g16")
    tool.sandbox = _Sb()
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is True
    assert res.data["autoheal_attempts"][0]["fixes_applied"] == {"scf": "xqc"}
    assert calls["n"] == 2


async def test_run_gaussian_sandbox_exception(workdir):
    tool = _tool(gaussian_executable="/usr/bin/g16")

    class _Sb:
        def run(self, cmd, cwd=None, timeout=None):
            raise RuntimeError("boom")

    tool.sandbox = _Sb()
    res = await tool.call(_args(working_dir=str(workdir)), None)
    assert res.success is False
    assert "Gaussian execution failed" in res.error


# ── route 处理 ───────────────────────────────────────────────────────────


def test_read_route_params(tmp_path):
    gjf = tmp_path / "in.gjf"
    gjf.write_text("#p opt=(calcfc) hf/6-31g(d)\n\nmol\n", encoding="utf-8")
    tool = _tool(gaussian_executable=None)
    params = tool._read_route_params(gjf)
    assert params["method"] == "hf"
    assert params["basis"] == "6-31g(d)"
    assert params["opt"] == "calcfc"


def test_read_route_params_exception(tmp_path):
    tool = _tool(gaussian_executable=None)
    assert tool._read_route_params(tmp_path / "nope.gjf") == {}


def test_apply_route_overrides_replace_and_append(tmp_path):
    gjf = tmp_path / "in.gjf"
    gjf.write_text("#p hf/6-31g scf=conver\n\nmol\n", encoding="utf-8")
    tool = _tool(gaussian_executable=None)
    tool._apply_route_overrides(gjf, {"scf": "xqc", "integral": "grid=ultrafine"})
    text = gjf.read_text(encoding="utf-8")
    assert "SCF=(XQC)" in text
    assert "Int=(GRID=ULTRAFINE)" in text


def test_format_keyword():
    assert GaussianTool._format_keyword("SCF", True) == "SCF"
    assert GaussianTool._format_keyword("SCF", None) == "SCF"
    assert GaussianTool._format_keyword("SCF", "xqc") == "SCF=(XQC)"
    assert GaussianTool._format_keyword("MaxCycle", 100) == "MaxCycle=100"


def test_find_gjf_named(workdir):
    tool = _tool(gaussian_executable=None)
    p = tool._find_gjf(workdir, "in.gjf")
    assert p == workdir / "in.gjf"
    assert tool._find_gjf(workdir, "missing.gjf") is None


def test_find_gjf_autodetect_com(tmp_path):
    (tmp_path / "a.com").write_text("x", encoding="utf-8")
    tool = _tool(gaussian_executable=None)
    assert tool._find_gjf(tmp_path, None).name == "a.com"


def test_parse_and_return_no_log(workdir):
    tool = _tool(gaussian_executable=None)
    res = tool._parse_and_return(workdir, workdir / "in.gjf")
    assert res.success is False
    assert "No .log file found" in res.error


def test_parse_and_return_ok(workdir):
    (workdir / "in.log").write_text(
        "SCF Done:  E(RHF) =  -76.0\n     Normal termination\n", encoding="utf-8"
    )
    tool = _tool(gaussian_executable=None)
    res = tool._parse_and_return(workdir, workdir / "in.gjf")
    assert res.success is True
    assert res.data["status"] == "completed"


def test_parse_and_return_normal_termination_false(workdir):
    (workdir / "in.log").write_text("SCF Done:  E(RHF) =  -76.0\n", encoding="utf-8")
    tool = _tool(gaussian_executable=None)
    res = tool._parse_and_return(workdir, workdir / "in.gjf")
    assert res.data["status"] == "failed"


def test_mock_result(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)
    tool = _tool(gaussian_executable=None)
    res = tool._mock_result(_args(action="sp"), Path("."))
    assert res.success is True
    assert res.data["status"] == "mock"
    assert res.metadata["mock"] is True


# ── helper ───────────────────────────────────────────────────────────────


def test_get_returncode():
    assert GaussianTool._get_returncode(types.SimpleNamespace(returncode=3)) == 3
    assert GaussianTool._get_returncode({"returncode": 4}) == 4
    assert GaussianTool._get_returncode("x") == -1


def test_get_stderr():
    assert GaussianTool._get_stderr(types.SimpleNamespace(stderr="e")) == "e"
    assert GaussianTool._get_stderr({"stderr": "e"}) == "e"
    assert GaussianTool._get_stderr({}) == ""
    assert GaussianTool._get_stderr("x") == ""


# ── _parse_log 全分支 ────────────────────────────────────────────────────


def test_parse_log_full(tmp_path):
    log = tmp_path / "in.log"
    log.write_text(
        "Charge = 0 Multiplicity = 1\n"
        "SCF Done:  E(RHF) =  -76.1\n"
        "SCF Done:  E(RHF) =  -76.2\n"
        " Center    Atomic    Forces (Hartrees/Bohr)\n"
        "  1 1  H  0.001 0.002 0.003\n"
        "  2 1  H -0.001 -0.002 -0.003\n"
        " Frequencies --  123.4\n"
        " Frequencies -- -456.7\n"
        " Optimization completed.\n"
        " Normal termination\n",
        encoding="utf-8",
    )
    tool = _tool(gaussian_executable=None)
    parsed = tool._parse_log(log)
    assert parsed["energy"] == pytest.approx(-76.2)
    assert parsed["normal_termination"] is True
    assert parsed["converged"] is True
    assert parsed["optimization_completed"] is True
    assert len(parsed["forces"]) == 2
    assert parsed["frequencies"] == [123.4, -456.7]
    assert parsed["charge"] == 0
    assert parsed["multiplicity"] == 1


def test_parse_log_read_exception(tmp_path):
    tool = _tool(gaussian_executable=None)
    parsed = tool._parse_log(tmp_path / "nope.log")
    assert "parse_error" in parsed


def test_parse_log_convergence_failure(tmp_path):
    log = tmp_path / "in.log"
    log.write_text("Rules for a new−1lias:\nConvergence failure\n", encoding="utf-8")
    tool = _tool(gaussian_executable=None)
    parsed = tool._parse_log(log)
    assert parsed["scf_convergence_failure"] is True
    assert parsed["converged"] is False


# ── estimate_cost / _try_autofix ─────────────────────────────────────────


def test_estimate_cost():
    tool = _tool(gaussian_executable=None)
    assert tool.estimate_cost(_args())["cpu_hours"] == 1.0
    assert tool.estimate_cost(_args(timeout=1800))["walltime_hours"] == 0.5


def test_try_autofix_applies(monkeypatch, tmp_path):
    gjf = tmp_path / "in.gjf"
    gjf.write_text("#p hf/6-31g\n\nmol\n\n0 1\nH 0 0 0\n", encoding="utf-8")
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _AutoFix:
        def apply_fix(self, tool_name, error, current):
            return {"scf": "xqc", "integral": "grid=ultrafine", "__auto_fix": "fix"}

    autofix_mod.AutoFixLoop = _AutoFix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    tool = _tool(gaussian_executable=None)
    result = tool._try_autofix(gjf, "SCF convergence failure")
    assert result is not None
    assert result["fixes"] == {"scf": "xqc", "integral": "grid=ultrafine"}
    assert "SCF=(XQC)" in gjf.read_text(encoding="utf-8")


def test_try_autofix_no_route_fixes(monkeypatch, tmp_path):
    gjf = tmp_path / "in.gjf"
    gjf.write_text("#p hf/6-31g\n\nmol\n", encoding="utf-8")
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _AutoFix:
        def apply_fix(self, tool_name, error, current):
            return {"unmapped_key": 1, "__auto_fix": "x"}

    autofix_mod.AutoFixLoop = _AutoFix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    tool = _tool(gaussian_executable=None)
    assert tool._try_autofix(gjf, "err") is None


def test_try_autofix_no_fix(monkeypatch, tmp_path):
    gjf = tmp_path / "in.gjf"
    gjf.write_text("#p hf/6-31g\n\nmol\n", encoding="utf-8")
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _AutoFix:
        def apply_fix(self, *a, **k):
            return None

    autofix_mod.AutoFixLoop = _AutoFix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    tool = _tool(gaussian_executable=None)
    assert tool._try_autofix(gjf, "err") is None


def test_try_autofix_exception(monkeypatch, tmp_path):
    gjf = tmp_path / "in.gjf"
    gjf.write_text("#p hf/6-31g\n\nmol\n", encoding="utf-8")
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _AutoFix:
        def apply_fix(self, *a, **k):
            raise RuntimeError("boom")

    autofix_mod.AutoFixLoop = _AutoFix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    tool = _tool(gaussian_executable=None)
    assert tool._try_autofix(gjf, "err") is None
