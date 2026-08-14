"""Tests for the Quantum ESPRESSO tool."""

from pathlib import Path

import sys
import types

import pytest

from huginn.tools.sim.qe_tool import QuantumEspressoTool, QuantumEspressoToolInput

pytestmark = pytest.mark.anyio


def test_qe_tool_generates_input(tmp_path: Path) -> None:
    """QuantumEspressoTool should generate a pw.x input file."""
    tool = QuantumEspressoTool(qe_executable=None)
    result = tool.call(
        {
            "action": "generate",
            "working_dir": str(tmp_path),
            "output_prefix": "si_scf",
        }
    )
    assert result.success is True
    input_path = Path(result.data["input_path"])
    assert input_path.exists()
    text = input_path.read_text(encoding="utf-8")
    assert "&CONTROL" in text
    assert "ATOMIC_SPECIES" in text
    assert "K_POINTS" in text
    assert result.data["qe_available"] is False


def test_qe_tool_run_fallback(tmp_path: Path) -> None:
    """Run mode should fall back to input export when pw.x is missing."""
    tool = QuantumEspressoTool(qe_executable=None)
    result = tool.call(
        {
            "action": "run",
            "calculation": "scf",
            "working_dir": str(tmp_path),
            "output_prefix": "si_scf",
        }
    )
    assert result.success is True
    assert result.data["qe_available"] is False
    assert Path(result.data["input_path"]).exists()


def test_qe_tool_parse_output(tmp_path: Path) -> None:
    """Parse a synthetic QE output file."""
    tool = QuantumEspressoTool()
    out_file = tmp_path / "qe.out"
    out_file.write_text(
        "Program PWSCF v.7.0\n"
        "\n"
        "     iteration #  1\n"
        "     iteration #  2\n"
        "\n"
        "!    total energy              =     -10.12345678 Ry\n"
        "\n"
        "     convergence has been achieved\n"
        "\n"
        "     Forces acting on atoms (cartesian axes, Ry/au):\n"
        "\n"
        "     atom    1 type  1   force =     0.0010000    0.0020000    0.0030000\n"
        "     atom    2 type  1   force =    -0.0010000   -0.0020000   -0.0030000\n"
        "\n"
        "     total   stress  (Ry/bohr**3)                   (kbar)     P=       -0.12\n"
        "          -0.00123456   -0.00000000    0.00000000\n"
        "          -0.00000000   -0.00123456    0.00000000\n"
        "           0.00000000    0.00000000   -0.00123456\n",
        encoding="utf-8",
    )

    result = tool.call(
        {
            "action": "parse",
            "working_dir": str(tmp_path),
            "result_files": ["qe.out"],
        }
    )
    assert result.success is True
    parsed = result.data["results"]["qe.out"]
    assert parsed["energy"] == pytest.approx(-10.12345678, abs=1e-9)
    assert parsed["converged"] is True
    assert parsed["n_scf_steps"] == 2
    assert len(parsed["forces"]) == 2
    assert parsed["forces"][0] == pytest.approx([0.001, 0.002, 0.003], abs=1e-9)
    assert len(parsed["stress"]) == 3


def test_qe_tool_input_schema() -> None:
    """QuantumEspressoToolInput should accept valid parameters."""
    inp = QuantumEspressoToolInput(
        action="run",
        calculation="relax",
        ecutwfc=50.0,
    )
    assert inp.calculation == "relax"
    assert inp.ecutwfc == 50.0


# ── 以下为原 tests/test_qe_tool_integration_ext.py 归并内容 ──────────────────
# qe_tool.py 集成路径补测 — 覆盖 _find_qe(env/which)、call(异常)、
# _generate_input(relax/vc-relax/md + CELL 块)、_run_qe(成功/硬失败/SCF 未收敛
# 软失败/autofix 重试/物理审计/兜底审计)、_read_output_tail、_read_input_params、
# _apply_input_fixes、_try_autofix、_parse_output_file(缺失)、_parse_output 全分支、
# _parse_results.


def _tool(**kw):
    return QuantumEspressoTool(**kw)


def _args(**kw):
    base = {"action": "run", "calculation": "scf", "output_prefix": "qe_out"}
    base.update(kw)
    return base


def _model(**kw):
    return QuantumEspressoToolInput(**_args(**kw))


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


# ── _find_qe ─────────────────────────────────────────────────────────────


def test_find_qe_env_hit(monkeypatch, tmp_path):
    exe = tmp_path / "pw.x"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setenv("QE_EXECUTABLE", str(exe))
    tool = _tool(qe_executable=None)
    assert tool._find_qe() == str(exe)


def test_find_qe_env_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("QE_EXECUTABLE", str(tmp_path / "nope"))
    tool = _tool(qe_executable=None)
    assert tool._find_qe() is None


def test_find_qe_which(monkeypatch):
    monkeypatch.delenv("QE_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        "huginn.tools.sim.qe_tool.shutil.which",
        lambda cmd: "/usr/bin/pw.x" if cmd == "pw.x" else None,
    )
    tool = _tool(qe_executable=None)
    assert tool._find_qe() == "pw.x"


def test_find_qe_none(monkeypatch):
    monkeypatch.delenv("QE_EXECUTABLE", raising=False)
    monkeypatch.setattr(
        "huginn.tools.sim.qe_tool.shutil.which", lambda cmd: None
    )
    tool = _tool(qe_executable=None)
    assert tool._find_qe() is None


# ── call: 异常分支 ────────────────────────────────────────────────────────


def test_call_exception(tmp_path, monkeypatch):
    tool = _tool(qe_executable=None)
    # 让 _generate_input 抛异常, 验证 call 的 try/except
    monkeypatch.setattr(
        QuantumEspressoTool, "_generate_input",
        lambda self, a, w, p: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    res = tool.call(_args(working_dir=str(tmp_path)))
    assert res.success is False
    assert "QE tool failed" in res.error


# ── _generate_input: relax / vc-relax / md 分支 ──────────────────────────


def test_generate_input_relax(tmp_path):
    tool = _tool(qe_executable=None)
    args = _model(calculation="relax", ecutrho=80.0)
    p = tool._generate_input(args, Path(tmp_path), "prefix")
    text = p.read_text(encoding="utf-8")
    assert "&IONS" in text
    assert "&CELL" not in text
    assert "ecutrho = 80.0" in text


def test_generate_input_vc_relax(tmp_path):
    tool = _tool(qe_executable=None)
    args = _model(calculation="vc-relax")
    p = tool._generate_input(args, Path(tmp_path), "prefix")
    text = p.read_text(encoding="utf-8")
    assert "&IONS" in text
    assert "&CELL" in text


def test_generate_input_md(tmp_path):
    tool = _tool(qe_executable=None)
    args = _model(calculation="md")
    p = tool._generate_input(args, Path(tmp_path), "prefix")
    assert "&IONS" in p.read_text(encoding="utf-8")


def test_generate_input_unknown_pseudo_and_mass(tmp_path):
    """未知元素 → pseudo 默认 UPF, mass 默认 1.0."""
    tool = _tool(qe_executable=None)
    args = _model()
    args.structure = {
        "lattice": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "species": ["Xx"],
        "positions": [[0, 0, 0]],
    }
    args.pseudopotentials = {}
    p = tool._generate_input(args, Path(tmp_path), "prefix")
    text = p.read_text(encoding="utf-8")
    assert "Xx 1.0000 Xx.UPF" in text


# ── _run_qe 各分支 (mock sandbox, 返回 dict) ─────────────────────────────


def _sandbox_fake(returncode=0, output_text=None, output_path=None, converged=True):
    """QE sandbox fake: 每次 run 把 output 写进 stdout 文件."""

    class _Sb:
        def run(self, cmd, cwd=None, config=None, stdout=None, stderr=None):
            if output_text is not None:
                stdout.write(output_text)
                stdout.flush()
            return {"returncode": returncode}

    return _Sb()


def test_run_qe_success(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch, has_errors=False)
    tool = _tool(qe_executable="/usr/bin/pw.x")
    tool.sandbox = _sandbox_fake(
        returncode=0,
        output_text=(
            "!    total energy              =     -10.5 Ry\n"
            "     convergence has been achieved\n"
            "     iteration #  1\n"
            "     iteration #  2\n"
        ),
    )
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is True
    assert res.data["qe_available"] is True
    assert res.data["parsed"]["energy"] == pytest.approx(-10.5)
    assert res.data["parsed"]["converged"] is True
    assert res.data["parsed"]["n_scf_steps"] == 2


def test_run_qe_hard_failure(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    tool = _tool(qe_executable="/usr/bin/pw.x")
    tool.sandbox = _sandbox_fake(
        returncode=1, output_text="ERROR: something broke"
    )
    # 不触发 autofix 重试
    monkeypatch.setattr(QuantumEspressoTool, "_try_autofix", lambda self, p, e: None)
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is False
    assert "QE execution failed" in res.error


def test_run_qe_scf_not_converged_soft_fail(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    tool = _tool(qe_executable="/usr/bin/pw.x")
    # 输出不含 "convergence has been achieved" → 未收敛软失败
    tool.sandbox = _sandbox_fake(returncode=0, output_text="no convergence here")
    monkeypatch.setattr(QuantumEspressoTool, "_try_autofix", lambda self, p, e: None)
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is False
    assert "SCF did not converge" in res.error


def test_run_qe_autofix_retry(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch, has_errors=False)
    calls = {"n": 0}
    out_text = {
        1: "no convergence here",
        2: "!    total energy = -5.0 Ry\n     convergence has been achieved\n",
    }

    class _Sb:
        def run(self, cmd, cwd=None, config=None, stdout=None, stderr=None):
            calls["n"] += 1
            stdout.write(out_text[calls["n"]])
            stdout.flush()
            return {"returncode": 0}

    tool = _tool(qe_executable="/usr/bin/pw.x")
    tool.sandbox = _Sb()
    monkeypatch.setattr(
        QuantumEspressoTool, "_try_autofix",
        lambda self, p, e: {"fixes": {"mixing_beta": 0.3}, "reasoning": "reduce"},
    )
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is True
    assert res.data["autoheal_attempts"][0]["fixes_applied"] == {"mixing_beta": 0.3}
    assert calls["n"] == 2


def test_run_qe_phys_audit_error_soft_fail(monkeypatch, tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _install_auditor(monkeypatch, has_errors=True, findings=[
        types.SimpleNamespace(severity="error", message="negative density"),
    ])
    tool = _tool(qe_executable="/usr/bin/pw.x")
    tool.sandbox = _sandbox_fake(
        returncode=0, output_text="convergence has been achieved"
    )
    monkeypatch.setattr(QuantumEspressoTool, "_try_autofix", lambda self, p, e: None)
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is False
    assert "Physics audit found errors" in res.error


def test_run_qe_audit_fallback_exception(monkeypatch, tmp_path):
    """rc=0 收敛, 但兜底审计抛异常 → 被吞, 结果正常返回."""
    d = tmp_path / "wd"
    d.mkdir()
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")

    class _Auditor:
        def audit(self, *a, **k):
            raise RuntimeError("audit boom")

    auditor_mod.PhysicsAuditor = _Auditor
    monkeypatch.setitem(sys.modules, "huginn.execution.physics_auditor", auditor_mod)
    tool = _tool(qe_executable="/usr/bin/pw.x")
    tool.sandbox = _sandbox_fake(
        returncode=0, output_text="convergence has been achieved"
    )
    res = tool.call(_args(working_dir=str(d)))
    assert res.success is True
    assert "physics_audit" not in res.data


def test_run_qe_no_executable(tmp_path):
    tool = _tool(qe_executable=None)
    res = tool.call(_args(working_dir=str(tmp_path)))
    assert res.success is True
    assert res.data["qe_available"] is False
    assert "run manually" in res.data["message"]


# ── 解析器边界 ───────────────────────────────────────────────────────────


def test_parse_output_file_missing(tmp_path):
    tool = _tool(qe_executable=None)
    res = tool._parse_output_file(tmp_path / "nope.out")
    assert res["error"] == "Output file not found"


def test_parse_output_forces_total_force_and_stress(monkeypatch):
    """forces 块以 'Total force' 结束 + stress 块."""
    tool = _tool(qe_executable=None)
    content = (
        "     Forces acting on atoms (cartesian axes, Ry/au):\n"
        "     atom    1 type  1   force =     0.0010  0.0020  0.0030\n"
        "     atom    2 type  1   force =    -0.0010 -0.0020 -0.0030\n"
        "     Total force =     0.004    Total force =     0.004\n"
        "\n"
        "     total   stress  (Ry/bohr**3)                   (kbar)     P=       -0.12\n"
        "          -0.001  0.000  0.000\n"
        "           0.000 -0.001  0.000\n"
        "           0.000  0.000 -0.001\n"
    )
    parsed = tool._parse_output(content)
    assert len(parsed["forces"]) == 2
    assert parsed["forces"][0] == pytest.approx([0.001, 0.002, 0.003])
    assert len(parsed["stress"]) == 3


def test_parse_output_forces_blank_terminated():
    """forces 块以空行结束."""
    tool = _tool(qe_executable=None)
    content = (
        "     Forces acting on atoms:\n"
        "     atom    1 type  1   force =     1.0  2.0  3.0\n"
        "\n"
    )
    parsed = tool._parse_output(content)
    assert len(parsed["forces"]) == 1


def test_parse_output_trailing_current_block():
    """循环结束后仍有未终止的 force 块."""
    tool = _tool(qe_executable=None)
    content = (
        "     Forces acting on atoms:\n"
        "     atom    1 type  1   force =     1.0  2.0  3.0\n"
    )
    parsed = tool._parse_output(content)
    assert len(parsed["forces"]) == 1


def test_parse_output_bad_energy():
    """energy 行格式异常 → 不崩, energy 保持 None."""
    tool = _tool(qe_executable=None)
    parsed = tool._parse_output("!    total energy              =     garbage\n")
    assert parsed["energy"] is None


def test_parse_results_multiple(monkeypatch):
    tool = _tool(qe_executable=None)
    parsed = tool._parse_output("convergence has been achieved")
    monkeypatch.setattr(
        QuantumEspressoTool, "_parse_output_file",
        lambda self, p: parsed,
    )
    res = tool._parse_results(_model(action="parse", result_files=["a.out", "b.out"]), Path("."))
    assert res.success is True
    assert set(res.data["results"].keys()) == {"a.out", "b.out"}


# ── 脚本修复 / autofix ───────────────────────────────────────────────────


def test_read_input_params(monkeypatch, tmp_path):
    inp = tmp_path / "qe.in"
    inp.write_text(
        "&CONTROL\n  calculation = 'scf'\n  pseudo_dir = './'\n/\n"
        "&SYSTEM\n  ecutwfc = 40\n  degauss = 0.01\n/\n",
        encoding="utf-8",
    )
    tool = _tool(qe_executable=None)
    params = tool._read_input_params(inp)
    assert params["calculation"] == "scf"
    assert params["ecutwfc"] == 40
    assert params["degauss"] == 0.01


def test_apply_input_fixes_existing_and_new(tmp_path):
    inp = tmp_path / "qe.in"
    inp.write_text(
        "&ELECTRONS\n  mixing_beta = 0.7\n/\n", encoding="utf-8"
    )
    tool = _tool(qe_executable=None)
    tool._apply_input_fixes(inp, {"mixing_beta": 0.3, "conv_thr": 1e-6})
    text = inp.read_text(encoding="utf-8")
    assert "mixing_beta = 0.3" in text
    assert "conv_thr = 1e-06" in text


def test_apply_input_fixes_no_electrons_block(tmp_path):
    inp = tmp_path / "qe.in"
    inp.write_text("&SYSTEM\n  ecutwfc = 40\n/\n", encoding="utf-8")
    tool = _tool(qe_executable=None)
    tool._apply_input_fixes(inp, {"mixing_beta": 0.3})
    # 无 &ELECTRONS 块 → 不改动
    assert "mixing_beta" not in inp.read_text(encoding="utf-8")


def test_try_autofix_applies(monkeypatch, tmp_path):
    inp = tmp_path / "qe.in"
    inp.write_text("&ELECTRONS\n  mixing_beta = 0.7\n/\n", encoding="utf-8")

    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _AutoFix:
        def apply_fix(self, tool_name, error, current):
            return {"mixing_beta": 0.3, "__auto_fix": "reduce", "__auto_fix_patterns_matched": ["mixing"]}

    autofix_mod.AutoFixLoop = _AutoFix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)

    tool = _tool(qe_executable=None)
    result = tool._try_autofix(inp, "some error")
    assert result is not None
    assert result["fixes"] == {"mixing_beta": 0.3}
    assert result["reasoning"] == "reduce"
    assert "mixing_beta = 0.3" in inp.read_text(encoding="utf-8")


def test_try_autofix_no_fix(monkeypatch, tmp_path):
    inp = tmp_path / "qe.in"
    inp.write_text("&ELECTRONS\n/\n", encoding="utf-8")
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _AutoFix:
        def apply_fix(self, *a, **k):
            return None

    autofix_mod.AutoFixLoop = _AutoFix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    tool = _tool(qe_executable=None)
    assert tool._try_autofix(inp, "error") is None


def test_try_autofix_exception(monkeypatch, tmp_path):
    inp = tmp_path / "qe.in"
    inp.write_text("&ELECTRONS\n/\n", encoding="utf-8")
    autofix_mod = types.ModuleType("huginn.execution.autofix")

    class _AutoFix:
        def apply_fix(self, *a, **k):
            raise RuntimeError("boom")

    autofix_mod.AutoFixLoop = _AutoFix
    monkeypatch.setitem(sys.modules, "huginn.execution.autofix", autofix_mod)
    tool = _tool(qe_executable=None)
    assert tool._try_autofix(inp, "error") is None


def test_read_output_tail(tmp_path):
    out = tmp_path / "qe.out"
    out.write_text("A" * 3000, encoding="utf-8")
    tool = _tool(qe_executable=None)
    assert tool._read_output_tail(out) == "A" * 2000


def test_read_output_tail_exception(tmp_path):
    tool = _tool(qe_executable=None)
    assert tool._read_output_tail(tmp_path / "nope.out") == ""