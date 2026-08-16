"""Tests for the CP2K tool."""

import sys
import types
from pathlib import Path

import pytest

from huginn.tools.sim.cp2k_tool import Cp2kTool, Cp2kToolInput


def test_cp2k_tool_generates_input(tmp_path: Path) -> None:
    """Cp2kTool should generate a CP2K input file."""
    tool = Cp2kTool(cp2k_executable=None)
    result = tool.call(
        {
            "action": "generate",
            "working_dir": str(tmp_path),
            "output_prefix": "si_dft",
        }
    )
    assert result.success is True
    input_path = Path(result.data["input_path"])
    assert input_path.exists()
    text = input_path.read_text(encoding="utf-8")
    assert "&GLOBAL" in text
    assert "RUN_TYPE ENERGY_FORCE" in text
    assert "&FORCE_EVAL" in text
    assert result.data["cp2k_available"] is False


def test_cp2k_tool_run_fallback(tmp_path: Path) -> None:
    """Run mode should fall back to input export when CP2K is missing."""
    tool = Cp2kTool(cp2k_executable=None)
    result = tool.call(
        {
            "action": "run",
            "run_type": "MD",
            "working_dir": str(tmp_path),
            "output_prefix": "si_md",
        }
    )
    assert result.success is True
    assert result.data["cp2k_available"] is False
    assert Path(result.data["input_path"]).exists()


def test_cp2k_tool_parse_output(tmp_path: Path) -> None:
    """Parse a synthetic CP2K output file."""
    tool = Cp2kTool()
    out_file = tmp_path / "cp2k.out"
    out_file.write_text(
        " ***  CP2K ***\n"
        "\n"
        " SCF iteration\n"
        " SCF iteration\n"
        " *** SCF run converged ***\n"
        "\n"
        "  Total energy:                                              -10.1234567890\n"
        "\n"
        " ATOMIC FORCES in [a.u.]\n"
        " # Atom   Kind   Element       X            Y            Z\n"
        "      1      1     Si          0.001000     0.002000     0.003000\n"
        "      2      1     Si         -0.001000    -0.002000    -0.003000\n"
        "\n"
        " STRESS|                        1          2          3\n"
        " STRESS|      1    -0.12345678   0.00000000   0.00000000\n"
        " STRESS|      2     0.00000000  -0.12345678   0.00000000\n"
        " STRESS|      3     0.00000000   0.00000000  -0.12345678\n",
        encoding="utf-8",
    )

    result = tool.call(
        {
            "action": "parse",
            "working_dir": str(tmp_path),
            "result_files": ["cp2k.out"],
        }
    )
    assert result.success is True
    parsed = result.data["results"]["cp2k.out"]
    assert parsed["energy"] == pytest.approx(-10.1234567890, abs=1e-9)
    assert parsed["converged"] is True
    assert parsed["n_scf_steps"] == 2
    assert len(parsed["forces"]) == 2
    assert parsed["forces"][0] == pytest.approx([0.001, 0.002, 0.003], abs=1e-9)
    assert len(parsed["stress"]) == 3


def test_cp2k_tool_input_schema() -> None:
    """Cp2kToolInput should accept valid parameters."""
    inp = Cp2kToolInput(
        action="run",
        run_type="GEO_OPT",
        cutoff=600,
    )
    assert inp.run_type == "GEO_OPT"
    assert inp.cutoff == 600


# ── 以下为原 tests/test_cp2k_tool_integration_ext.py 归并内容 ──────────────────
# cp2k_tool.py 集成路径补测 — 覆盖 _find_cp2k(env/which/无)、call(异常)、
# _run_cp2k(无可执行回退/成功/硬失败/审计异常吞掉)。


def _tool(**kw):
    return Cp2kTool(**kw)


def _args(**kw):
    base = {"action": "run", "run_type": "ENERGY_FORCE", "output_prefix": "cp2k_out"}
    base.update(kw)
    return base


# ── _find_cp2k ───────────────────────────────────────────────────────────


def test_find_cp2k_env_hit(monkeypatch, tmp_path):
    exe = tmp_path / "cp2k.popt"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CP2K_EXECUTABLE", str(exe))
    assert _tool(cp2k_executable=None)._find_cp2k() == str(exe)


def test_find_cp2k_env_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("CP2K_EXECUTABLE", str(tmp_path / "nope"))
    assert _tool(cp2k_executable=None)._find_cp2k() is None


def test_find_cp2k_which(monkeypatch):
    monkeypatch.delenv("CP2K_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        "huginn.tools.sim.cp2k_tool.shutil.which",
        lambda cmd: "/usr/bin/cp2k.popt" if cmd == "cp2k.popt" else None,
    )
    assert _tool(cp2k_executable=None)._find_cp2k() == "cp2k.popt"


def test_find_cp2k_none(monkeypatch):
    monkeypatch.delenv("CP2K_EXECUTABLE", raising=False)
    monkeypatch.setattr("huginn.tools.sim.cp2k_tool.shutil.which", lambda cmd: None)
    assert _tool(cp2k_executable=None)._find_cp2k() is None


# ── call 异常 ────────────────────────────────────────────────────────────


def test_call_exception(monkeypatch, tmp_path):
    tool = _tool(cp2k_executable=None)
    monkeypatch.setattr(
        Cp2kTool, "_generate_input",
        lambda self, a, w, p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    res = tool.call(_args(working_dir=str(tmp_path)))
    assert res.success is False
    assert "CP2K tool failed" in res.error


# ── _run_cp2k ────────────────────────────────────────────────────────────


def _sandbox_fake(returncode=0):
    class _Sb:
        def run(self, cmd, cwd=None, config=None):
            return {"returncode": returncode}

    return _Sb()


def test_run_cp2k_no_executable(tmp_path):
    tool = _tool(cp2k_executable=None)
    res = tool.call(_args(working_dir=str(tmp_path)))
    assert res.success is True
    assert res.data["cp2k_available"] is False
    assert "run manually" in res.data["message"]


def test_run_cp2k_success(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch)
    tool = _tool(cp2k_executable="/usr/bin/cp2k.popt")
    tool.sandbox = _sandbox_fake(returncode=0)
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is True
    assert res.data["cp2k_available"] is True
    assert res.data["physics_audit"]["has_errors"] is False


def test_run_cp2k_failure(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch)
    tool = _tool(cp2k_executable="/usr/bin/cp2k.popt")
    tool.sandbox = _sandbox_fake(returncode=1)
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is False
    assert "CP2K execution failed" in res.data["message"]


def test_run_cp2k_audit_exception(monkeypatch, tmp_path):
    """审计抛异常 → 被吞, 结果正常返回且无 physics_audit."""
    d = tmp_path / "wd"
    d.mkdir()
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    tool = _tool(cp2k_executable="/usr/bin/cp2k.popt")
    tool.sandbox = _sandbox_fake(returncode=0)
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is True
    assert "physics_audit" not in res.data


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
