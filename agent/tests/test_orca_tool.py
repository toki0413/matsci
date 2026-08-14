"""orca_tool.py 集成路径补测 — 覆盖 _find_orca(env/which)、estimate_cost、
validate_input(缺/成功)、call(缺 workdir/缺 inp/parse/有可执行/无可执行·
resolve str/无可执行·needs_resolution)、_run_orca(成功/硬失败/opt 未收敛软失败/
SCF 未收敛软失败/物理审计/兜底审计异常/autofix 重试/sandbox 异常/重试耗尽)、
_try_autofix、_read_input_params、_apply_input_overrides、_format_orca_token、
_parse_out 全字段、_find_inp、_parse_and_return、_mock_result、
_get_returncode/_get_stderr.

配合 tests/test_mock_simulation_tools.py, 把 orca_tool.py 覆盖率提升到 90%+.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from huginn.tools.sim.orca_tool import (
    OrcaTool,
    OrcaToolInput,
)

pytestmark = pytest.mark.anyio


def _tool(**kw):
    return OrcaTool(**kw)


def _args(**kw):
    base = {"action": "sp", "working_dir": "wd"}
    base.update(kw)
    return base


def _inp(path: Path, text: str | None = None) -> Path:
    p = path / "mol.inp"
    p.write_text(text or "! B3LYP def2-SVP\n* xyz 0 1\nC 0 0 0\n*", encoding="utf-8")
    return p


def _ctx():
    return type("C", (), {"workspace": "."})()


class _Proc:
    def __init__(self, returncode=0, stderr="", stdout=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


class _Sb:
    def __init__(self, proc=None, call=None):
        self._proc = proc or _Proc()
        self._call = call
        self.calls = []

    def run(self, cmd, cwd=None, timeout=None):
        self.calls.append(cmd)
        if self._call:
            return self._call(cmd, cwd)
        return self._proc


class _SbRaise:
    def run(self, cmd, cwd=None, timeout=None):
        raise RuntimeError("sandbox boom")


def _install_auditor(monkeypatch, has_errors=False, findings=None):
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")
    state = {"has_errors": has_errors, "findings": findings or []}

    class _Report:
        def __init__(self):
            self.has_errors = state["has_errors"]
            self.findings = state["findings"]

        def to_dict(self):
            return {"has_errors": self.has_errors, "findings": len(self.findings)}

    class _Auditor:
        def audit(self, *a, **k):
            return _Report()

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)


def _install_auditor_boom(monkeypatch):
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)


# ── _find_orca ────────────────────────────────────────────────────────────


def test_find_orca_env(monkeypatch, tmp_path):
    exe = tmp_path / "orca"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("ORCA_EXECUTABLE", str(exe))
    assert _tool()._find_orca() == str(exe)


def test_find_orca_env_not_exists(monkeypatch, tmp_path):
    monkeypatch.setenv("ORCA_EXECUTABLE", "/no/orca")
    monkeypatch.setattr("huginn.tools.sim.orca_tool.shutil.which", lambda c: None)
    assert _tool()._find_orca() is None


def test_find_orca_which(monkeypatch):
    monkeypatch.delenv("ORCA_EXECUTABLE", raising=False)
    monkeypatch.setattr("huginn.tools.sim.orca_tool.shutil.which", lambda c: "/usr/bin/orca")
    assert _tool()._find_orca() == "/usr/bin/orca"


# ── estimate_cost ─────────────────────────────────────────────────────────


def test_estimate_cost(tmp_path):
    tool = _tool()
    c = tool.estimate_cost(OrcaToolInput(action="sp", working_dir=str(tmp_path), timeout=3600))
    assert c["cpu_hours"] == 1.0
    assert c["walltime_hours"] == 1.0


# ── validate_input ────────────────────────────────────────────────────────


async def test_validate_missing_workdir():
    tool = _tool()
    r = await tool.validate_input(OrcaToolInput(action="sp", working_dir="/no/such"), _ctx())
    assert r.result is False
    assert r.error_code == 404


async def test_validate_ok(tmp_path):
    tool = _tool()
    r = await tool.validate_input(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.result is True


# ── call 基础分支 ─────────────────────────────────────────────────────────


async def test_call_workdir_missing():
    tool = _tool()
    r = await tool.call(OrcaToolInput(action="sp", working_dir="/no/such"), _ctx())
    assert r.success is False
    assert "Working directory not found" in r.error


async def test_call_no_inp(tmp_path):
    tool = _tool()
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is False
    assert "No .inp input file" in r.error


async def test_call_named_inp_missing(tmp_path):
    tool = _tool()
    r = await tool.call(
        OrcaToolInput(action="sp", working_dir=str(tmp_path), input_file="x.inp"), _ctx()
    )
    assert r.success is False


async def test_call_parse_ok(tmp_path):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8")
    tool = _tool()
    r = await tool.call(OrcaToolInput(action="parse", working_dir=str(tmp_path)), _ctx())
    assert r.success is True
    assert r.data["energy"] == pytest.approx(-76.5)


async def test_call_with_executable(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text(
        "FINAL SINGLE POINT ENERGY -76.5\nOPTIMIZATION HAS CONVERGED\n", encoding="utf-8"
    )
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=0))
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is True
    assert r.data["energy"] == pytest.approx(-76.5)
    assert r.data["status"] == "completed"


async def test_call_with_input_overrides(tmp_path, monkeypatch):
    inp = _inp(tmp_path, "! B3LYP SCF\n* xyz 0 1\n*")
    (tmp_path / "mol.out").write_text("FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8")
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=0))
    r = await tool.call(
        OrcaToolInput(
            action="sp", working_dir=str(tmp_path),
            input_overrides={"scf_conv": "tight", "grid": 7},
        ),
        _ctx(),
    )
    assert r.success is True
    assert "TightSCF" in inp.read_text(encoding="utf-8")


async def test_call_no_executable_resolve_str(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8")
    tool = _tool(orca_executable=None)
    monkeypatch.setattr(
        "huginn.tools.sim.executable_resolver.resolve_executable", lambda name: "/resolved/orca"
    )
    tool.sandbox = _Sb(_Proc(returncode=0))
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is True
    assert tool.orca_executable == "/resolved/orca"


async def test_call_no_executable_needs_resolution(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    tool = _tool(orca_executable=None)
    resolution = types.SimpleNamespace(
        install_hint="install orca",
        to_dict=lambda: {"exe": "orca"},
    )
    monkeypatch.setattr(
        "huginn.tools.sim.executable_resolver.resolve_executable", lambda name: resolution
    )
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is False
    assert "ORCA executable not found" in r.error
    assert r.metadata["needs_resolution"] is True


# ── _run_orca 分支 ────────────────────────────────────────────────────────


async def test_run_hard_failure(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="bad input"))
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is False
    assert "bad input" in r.error


async def test_run_opt_not_converged(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8")
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=0))
    monkeypatch.setattr(OrcaTool, "_try_autofix", lambda self, p, e: None)
    r = await tool.call(OrcaToolInput(action="opt", working_dir=str(tmp_path)), _ctx())
    assert r.success is False
    assert "did not converge" in r.error


async def test_run_scf_not_converged(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("some output without energy\n", encoding="utf-8")
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=0))
    monkeypatch.setattr(OrcaTool, "_try_autofix", lambda self, p, e: None)
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is False
    assert "SCF not converged" in r.error


async def test_run_physics_audit_errors(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8")
    _install_auditor(monkeypatch, has_errors=True, findings=[
        types.SimpleNamespace(severity="error", message="negative density"),
    ])
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=0))
    monkeypatch.setattr(OrcaTool, "_try_autofix", lambda self, p, e: None)
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is False
    assert "Physics audit found errors" in r.error


async def test_run_audit_boom_swallowed(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8")
    _install_auditor_boom(monkeypatch)
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=0))
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    # 审计挂了不影响最终结果: 兜底审计也挂 → 无 physics_audit 键, 仍成功
    assert r.success is True


async def test_run_autofix_retry(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    calls = {"n": 0}

    def _sb_run(cmd, cwd):
        calls["n"] += 1
        if calls["n"] == 1:
            # 第一次: 无 .out → SCF 未收敛软失败
            return _Proc(returncode=0)
        # 第二次: 生成 .out 带能量
        (tmp_path / "mol.out").write_text(
            "FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8"
        )
        return _Proc(returncode=0)

    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(call=_sb_run)
    monkeypatch.setattr(
        OrcaTool, "_try_autofix",
        lambda self, p, e: {"fixes": {"scf_conv": "tight"}, "reasoning": "tighten"},
    )
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is True
    assert r.data["autoheal_attempts"][0]["fixes_applied"] == {"scf_conv": "tight"}
    assert calls["n"] == 2


async def test_run_autofix_no_fix_breaks(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="boom"))
    monkeypatch.setattr(OrcaTool, "_try_autofix", lambda self, p, e: None)
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is False


async def test_run_retries_exhausted(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    calls = {"n": 0}

    def _sb_run(cmd, cwd):
        calls["n"] += 1
        return _Proc(returncode=0)

    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _Sb(call=_sb_run)
    monkeypatch.setattr(
        OrcaTool, "_try_autofix",
        lambda self, p, e: {"fixes": {"scf_conv": "tight"}, "reasoning": "x"},
    )
    r = await tool.call(
        OrcaToolInput(action="sp", working_dir=str(tmp_path), max_auto_retries=1), _ctx()
    )
    assert r.success is False
    assert calls["n"] == 2  # 初始 + 1 次重试后耗尽


async def test_run_sandbox_exception(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    tool = _tool(orca_executable="/usr/bin/orca")
    tool.sandbox = _SbRaise()
    r = await tool.call(OrcaToolInput(action="sp", working_dir=str(tmp_path)), _ctx())
    assert r.success is False
    assert "ORCA execution failed" in r.error


# ── _try_autofix ──────────────────────────────────────────────────────────


def test_try_autofix_none(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _Fix:
        def apply_fix(self, *a, **k):
            return {}

    autofix_mod.AutoFixLoop = _Fix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    assert _tool()._try_autofix(inp, "some error") is None


def test_try_autofix_applicable(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _Fix:
        def apply_fix(self, *a, **k):
            return {"scf_conv": "tight", "__auto_fix": "reason", "__auto_fix_patterns_matched": 1}

    autofix_mod.AutoFixLoop = _Fix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    res = _tool()._try_autofix(inp, "some error")
    assert res["fixes"] == {"scf_conv": "tight"}
    assert res["reasoning"] == "reason"


def test_try_autofix_no_applicable(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _Fix:
        def apply_fix(self, *a, **k):
            return {"some_unknown_key": "x"}

    autofix_mod.AutoFixLoop = _Fix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    assert _tool()._try_autofix(inp, "some error") is None


def test_try_autofix_exception(tmp_path, monkeypatch):
    inp = _inp(tmp_path)
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _Fix:
        def apply_fix(self, *a, **k):
            raise RuntimeError("boom")

    autofix_mod.AutoFixLoop = _Fix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    assert _tool()._try_autofix(inp, "some error") is None


# ── _read_input_params / _apply_input_overrides ───────────────────────────


def test_read_input_params(tmp_path):
    inp = _inp(tmp_path, "! B3LYP def2-SVP TightSCF\n* xyz\n*")
    params = _tool()._read_input_params(inp)
    assert params.get("b3lyp") is True
    assert params.get("def2-svp") is True


def test_read_input_params_exception(tmp_path, monkeypatch):
    inp = tmp_path / "mol.inp"
    inp.write_text("x")

    def boom(p, **k):
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", boom)
    assert _tool()._read_input_params(inp) == {}


def test_apply_overrides_replace_and_append(tmp_path):
    inp = _inp(tmp_path, "! B3LYP SCF\n* xyz\n*")
    tool = _tool()
    tool._apply_input_overrides(inp, {"scf_conv": "tight", "maxiter": 100, "grid": 7})
    text = inp.read_text(encoding="utf-8")
    assert "TightSCF" in text
    assert "Grid7" in text
    assert "MAXITER 100" in text


def test_apply_overrides_exception(tmp_path, monkeypatch):
    inp = _inp(tmp_path)

    def boom(p, **k):
        raise OSError("boom")

    monkeypatch.setattr(Path, "read_text", boom)
    # 不抛, 记 warning
    _tool()._apply_input_overrides(inp, {"scf_conv": "tight"})


def test_apply_overrides_nonbang_line(tmp_path):
    # 首行不是 ! 开头 → 跳过, 后续 ! 行仍被覆盖
    inp = tmp_path / "mol.inp"
    inp.write_text("# comment\n! B3LYP SCF\n* xyz\n*", encoding="utf-8")
    tool = _tool()
    tool._apply_input_overrides(inp, {"scf_conv": "tight"})
    assert "TightSCF" in inp.read_text(encoding="utf-8")


def test_apply_overrides_none_token(tmp_path):
    # format_orca_token 返回 None (如 maxiter 传字符串) → 跳过不写
    inp = _inp(tmp_path, "! B3LYP\n* xyz\n*")
    tool = _tool()
    tool._apply_input_overrides(inp, {"maxiter": "nope", "scf_conv": "tight"})
    text = inp.read_text(encoding="utf-8")
    assert "MAXITER" not in text
    assert "TightSCF" in text


def test_format_orca_token():
    t = OrcaTool._format_orca_token
    assert t("scf_conv", "tight") == "TightSCF"
    assert t("scf_conv", 5) == "TightSCF"
    assert t("grid", 7) == "Grid7"
    assert t("maxiter", 100) == "MAXITER 100"
    assert t("maxiter", "x") is None
    assert t("maxcore", "double") == "MAXCORE double"
    assert t("maxcore", 8) == "MAXCORE 8"
    assert t("some_key", True) == "SOME_KEY"
    assert t("some_key", "val") == "SOME_KEY val"


# ── _parse_out ────────────────────────────────────────────────────────────


def test_parse_out_full(tmp_path):
    p = tmp_path / "mol.out"
    p.write_text(
        "FINAL SINGLE POINT ENERGY -76.5\n"
        "FINAL SINGLE POINT ENERGY -76.4\n"
        "OPTIMIZATION HAS CONVERGED\n"
        "Geometry Optimization Cycle 1\n"
        "  0:  1200.00 cm\n"
        "  1:  320.00 cm\n"
        "Charge : 0\n"
        "Multiplicity = 1\n",
        encoding="utf-8",
    )
    r = _tool()._parse_out(p)
    assert r["energy"] == pytest.approx(-76.4)
    assert r["converged"] is True
    assert r["optimization_steps"] == 1
    assert r["frequencies"] == [1200.0, 320.0]
    assert r["charge"] == 0
    assert r["multiplicity"] == 1


def test_parse_out_total_energy_fallback(tmp_path):
    p = tmp_path / "mol.out"
    p.write_text("Total Energy : -76.3\n", encoding="utf-8")
    r = _tool()._parse_out(p)
    assert r["energy"] == pytest.approx(-76.3)


def test_parse_out_read_exception(tmp_path, monkeypatch):
    p = tmp_path / "mol.out"
    p.write_text("x")

    def boom(path, **k):
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_text", boom)
    r = _tool()._parse_out(p)
    assert "parse_error" in r


# ── _find_inp / _parse_and_return ─────────────────────────────────────────


def test_find_inp_named(tmp_path):
    _inp(tmp_path)
    assert _tool()._find_inp(tmp_path, "mol.inp").name == "mol.inp"
    assert _tool()._find_inp(tmp_path, "nope.inp") is None


def test_find_inp_glob(tmp_path):
    _inp(tmp_path)
    assert _tool()._find_inp(tmp_path, None).name == "mol.inp"


def test_parse_and_return_no_out(tmp_path):
    inp = _inp(tmp_path)
    r = _tool()._parse_and_return(tmp_path, inp)
    assert r.success is False
    assert "No .out file" in r.error


def test_parse_and_return_out_in_workdir(tmp_path):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("FINAL SINGLE POINT ENERGY -76.5\n", encoding="utf-8")
    r = _tool()._parse_and_return(tmp_path, inp)
    assert r.success is True
    assert r.data["energy"] == pytest.approx(-76.5)


def test_parse_and_return_failed_no_energy(tmp_path):
    inp = _inp(tmp_path)
    (tmp_path / "mol.out").write_text("no energy here\n", encoding="utf-8")
    r = _tool()._parse_and_return(tmp_path, inp)
    assert r.success is True
    assert r.data["status"] == "failed"
    assert r.data["energy"] is None


# ── _mock_result ──────────────────────────────────────────────────────────


def test_mock_result(tmp_path):
    tool = _tool()
    r = tool._mock_result(OrcaToolInput(action="opt", working_dir=str(tmp_path)), tmp_path)
    assert r.success is True
    assert r.data["status"] == "mock"
    assert r.data["optimization_steps"] == 3
    assert r.metadata["mock"] is True


# ── _get_returncode / _get_stderr ─────────────────────────────────────────


def test_get_returncode_variants():
    assert OrcaTool._get_returncode(_Proc(returncode=7)) == 7
    assert OrcaTool._get_returncode({"returncode": 3}) == 3
    assert OrcaTool._get_returncode({}) == -1
    assert OrcaTool._get_returncode("plain-string") == -1


def test_get_stderr_variants():
    assert OrcaTool._get_stderr(_Proc(stderr="err")) == "err"
    assert OrcaTool._get_stderr({"stderr": "dict-err"}) == "dict-err"
    assert OrcaTool._get_stderr({}) == ""
    assert OrcaTool._get_stderr(_Proc(stderr=None)) == ""
    assert OrcaTool._get_stderr("plain-string") == ""
