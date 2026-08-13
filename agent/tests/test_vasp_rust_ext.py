"""vasp_tool.py 的 Rust `_parse_outcar` 桥接测试.

huginn_ext 未编译安装时, Rust accelerator 分支 (sim/vasp_tool.py `_parse_outcar`
722-731 行) 走不到. 这里 monkeypatch 模块级 `_HAS_HUGINN_EXT` + `huginn_ext`,
覆盖: relax 命中 Rust 成功 / scf 不信任 Rust 落 Python / Rust 返回 error /
Rust 返回 converged=False / Rust 抛异常 → 都降级到 Python parser.
"""

from __future__ import annotations

import types

import pytest

from huginn.tools.sim import vasp_tool as vt

_SYNTH_LINES = [
    "VASP output",
    "ENCUT  =  520.0 eV",
    "ISPIN  =  2",
    "NELM   =  100",
    "NELMIN =  3",
    "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
    "  free  energy   TOTEN  =       -10.1234 eV",
    "TOTAL-FORCE (eV/Angst)",
    "  0.000  0.000  0.000    0.010  0.020  0.030",
    "  1.750  1.750  0.000   -0.010 -0.020 -0.030",
    "reached required accuracy - stopping structural energy minimisation",
]


@pytest.fixture
def outcar(tmp_path):
    p = tmp_path / "OUTCAR"
    p.write_text("\n".join(_SYNTH_LINES), encoding="utf-8")
    return p


def _install_fake_ext(monkeypatch, parse_outcar=None):
    """把 fake huginn_ext 装进 sim.vasp_tool 模块全局, 并打开 Rust 开关."""
    ext = types.ModuleType("huginn_ext")
    ext.parse_outcar = parse_outcar
    monkeypatch.setattr(vt, "huginn_ext", ext)
    monkeypatch.setattr(vt, "_HAS_HUGINN_EXT", True)


def _python_baseline(outcar, action=None):
    return vt.VaspTool()._parse_outcar_python(outcar, action=action)


# ── Rust fast path ───────────────────────────────────────────────────────

def test_rust_relax_returns_rust_result(monkeypatch, outcar):
    def _rust(path):
        return {"energy": -10.5, "converged": True, "rust": True}

    _install_fake_ext(monkeypatch, parse_outcar=_rust)
    result = vt.VaspTool()._parse_outcar(outcar, action="relax")
    assert result["rust"] is True
    assert result["energy"] == -10.5


def test_rust_scf_not_trusted_falls_to_python(monkeypatch, outcar):
    """scf/band/dos 不信任 Rust converged 字段 → 直接落 Python."""
    called = {"n": 0}

    def _rust(path):
        called["n"] += 1
        return {"energy": -10.5, "converged": True}

    _install_fake_ext(monkeypatch, parse_outcar=_rust)
    result = vt.VaspTool()._parse_outcar(outcar, action="scf")
    assert called["n"] == 0  # Rust 没被调用
    assert result["energy"] == pytest.approx(-10.1234, abs=1e-9)
    # scf 电子收敛标记未出现在合成 OUTCAR → Python 判 not converged (符合预期)
    assert result["converged"] is False


def test_rust_returns_error_falls_to_python(monkeypatch, outcar):
    def _rust(path):
        return {"error": "unparseable"}

    _install_fake_ext(monkeypatch, parse_outcar=_rust)
    result = vt.VaspTool()._parse_outcar(outcar, action="relax")
    assert result["energy"] == pytest.approx(-10.1234, abs=1e-9)


def test_rust_not_converged_falls_to_python(monkeypatch, outcar):
    """Rust 说 not converged → Python 端 double-check (重新解析)."""
    def _rust(path):
        return {"energy": -10.5, "converged": False}

    _install_fake_ext(monkeypatch, parse_outcar=_rust)
    result = vt.VaspTool()._parse_outcar(outcar, action="relax")
    # synthetic OUTCAR 有 "reached required accuracy" → Python 判 converged
    assert result["converged"] is True
    assert result["energy"] == pytest.approx(-10.1234, abs=1e-9)


def test_rust_raises_falls_to_python(monkeypatch, outcar):
    def _rust(path):
        raise RuntimeError("rust crash")

    _install_fake_ext(monkeypatch, parse_outcar=_rust)
    result = vt.VaspTool()._parse_outcar(outcar, action="relax")
    assert result["energy"] == pytest.approx(-10.1234, abs=1e-9)


# ── Rust 关闭时 ──────────────────────────────────────────────────────────

def test_rust_disabled_uses_python(monkeypatch, outcar):
    """_HAS_HUGINN_EXT=False → 不管 fake 是否存在都走 Python."""
    called = {"n": 0}

    def _rust(path):
        called["n"] += 1
        return {"energy": -10.5, "converged": True}

    _install_fake_ext(monkeypatch, parse_outcar=_rust)
    monkeypatch.setattr(vt, "_HAS_HUGINN_EXT", False)
    result = vt.VaspTool()._parse_outcar(outcar, action="relax")
    assert called["n"] == 0
    assert result["energy"] == pytest.approx(-10.1234, abs=1e-9)