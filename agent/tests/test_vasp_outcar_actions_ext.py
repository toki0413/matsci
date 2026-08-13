"""sim/vasp_tool.py `_parse_outcar_python` action-aware 收敛判定 + eos 集成测试.

覆盖 scf/band/dos 电子收敛 vs relax/md/phonon 离子收敛分支、pymatgen 路径、
_eos 拟合 (mock numerical_tool). 把 sim/vasp_tool.py 覆盖率推到 85%+.
"""

from __future__ import annotations

import sys
import types

import pytest

from huginn.tools.sim import vasp_tool as vt


pytestmark = pytest.mark.anyio


def _write_outcar(tmp_path, lines):
    p = tmp_path / "OUTCAR"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_scf_ediiff_reached_converged(tmp_path):
    p = _write_outcar(tmp_path, [
        "free  energy   TOTEN  =       -10.0 eV",
        "EDIFF is reached",
    ])
    r = vt.VaspTool()._parse_outcar_python(p, action="scf")
    assert r["converged"] is True
    assert r["convergence_criterion"] == "electronic (action=scf)"


def test_scf_ediiff_not_reached_not_converged(tmp_path):
    p = _write_outcar(tmp_path, ["free  energy   TOTEN  =       -10.0 eV"])
    r = vt.VaspTool()._parse_outcar_python(p, action="scf")
    assert r["converged"] is False


def test_band_output_exists_converged(tmp_path):
    p = _write_outcar(tmp_path, [
        "free  energy   TOTEN  =       -10.0 eV",
        "E-fermi :   5.5000",
    ])
    r = vt.VaspTool()._parse_outcar_python(p, action="band")
    assert r["converged"] is True


def test_dos_no_output_not_converged(tmp_path):
    p = _write_outcar(tmp_path, ["just some text"])
    r = vt.VaspTool()._parse_outcar_python(p, action="dos")
    assert r["converged"] is False


def test_relax_no_ionic_marker_not_converged(tmp_path):
    p = _write_outcar(tmp_path, ["free  energy   TOTEN  =       -10.0 eV"])
    r = vt.VaspTool()._parse_outcar_python(p, action="relax")
    assert r["converged"] is False


def test_relax_ionic_marker_converged(tmp_path):
    p = _write_outcar(tmp_path, [
        "free  energy   TOTEN  =       -10.0 eV",
        "reached required accuracy - stopping structural energy minimisation",
    ])
    r = vt.VaspTool()._parse_outcar_python(p, action="relax")
    assert r["converged"] is True


def test_parse_fields_incar_params(tmp_path):
    p = _write_outcar(tmp_path, [
        "ENCUT  =  520.0 eV",
        "ISPIN  =  2",
        "NELM   =  100",
        "NELMIN =  3",
        "k-points in units of 2pi/SCALE and weight:",
        "  0.000  0.000  0.000  1.000",
        "direct lattice vectors                 reciprocal lattice vectors",
        "  3.500000000  0.000000000  0.000000000     0.285714286  0.000000000  0.000000000",
        "  0.000000000  3.500000000  0.000000000     0.000000000  0.285714286  0.000000000",
        "  0.000000000  0.000000000  3.500000000     0.000000000  0.000000000  0.285714286",
        "volume of cell :      42.875",
        "TOTAL-FORCE (eV/Angst)",
        "  0.000  0.000  0.000    0.010  0.020  0.030",
        "  1.750  1.750  0.000   -0.010 -0.020 -0.030",
        "",
        "",
    ])
    r = vt.VaspTool()._parse_outcar_python(p, action="band")
    assert r["encut"] == pytest.approx(520.0)
    assert r["ispin"] == 2
    assert r["nelm"] == 100
    assert r["nelmin"] == 3
    assert r["kpoints"] == "found"
    assert r["volume"] == pytest.approx(42.875)
    assert len(r["lattice_vectors"]) == 3
    assert len(r["forces"]) == 2


def test_parse_pymatgen_path(monkeypatch, tmp_path):
    p = _write_outcar(tmp_path, ["free  energy   TOTEN  =       -10.0 eV"])

    pymatgen = types.ModuleType("pymatgen")
    io_mod = types.ModuleType("pymatgen.io")
    vasp_mod = types.ModuleType("pymatgen.io.vasp")

    class _Outcar:
        def __init__(self, path):
            pass

        final_energy = -9.0
        forces = [[[0.1, 0.2, 0.3]]]
        magnetizations = [[0.5]]
        converged = True

    vasp_mod.Outcar = _Outcar
    io_mod.vasp = vasp_mod
    pymatgen.io = io_mod
    for name, mod in [("pymatgen", pymatgen), ("pymatgen.io", io_mod),
                      ("pymatgen.io.vasp", vasp_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)

    r = vt.VaspTool()._parse_outcar_python(p, action="relax")
    assert r["energy"] == pytest.approx(-9.0)
    assert r["parse_source"] == "pymatgen"
    assert r["converged"] is True
    assert r["magnetic_moments"] == [0.5]


# ── _eos ─────────────────────────────────────────────────────────────────


def _install_numerical_tool(monkeypatch, success=True, data=None):
    num_mod = types.ModuleType("huginn.tools.numerical_tool")
    class _NT:
        async def call(self, payload):
            return types.SimpleNamespace(
                success=success, data=data or {"eos": "ok"}, error=None
            )
    num_mod.NumericalTool = _NT
    monkeypatch.setitem(sys.modules, "huginn.tools.numerical_tool", num_mod)


def _eos_dir(tmp_path, n=5):
    d = tmp_path / "eos"
    d.mkdir()
    for i in range(n):
        sub = d / f"vol{i}"
        sub.mkdir()
        sub.joinpath("OUTCAR").write_text(
            f"volume of cell :      {10 + i}\n"
            f"free  energy   TOTEN  =       {-100.0 - i} eV\n",
            encoding="utf-8",
        )
    return d


async def test_eos_fits(monkeypatch, tmp_path):
    d = _eos_dir(tmp_path)
    _install_numerical_tool(monkeypatch, success=True, data={"a": 1})
    res = await vt.VaspTool()._eos(
        vt.VaspToolInput(action="eos", working_dir=str(d)), None
    )
    assert res.success is True
    assert res.data["n_points"] == 5
    assert len(res.data["ev_points"]) == 5


async def test_eos_not_enough_points(tmp_path):
    d = tmp_path / "eos"
    d.mkdir()
    # 只有 1 个 OUTCAR
    sub = d / "v0"
    sub.mkdir()
    sub.joinpath("OUTCAR").write_text(
        "volume of cell :      10\nfree  energy   TOTEN  =       -100 eV\n",
        encoding="utf-8",
    )
    res = await vt.VaspTool()._eos(
        vt.VaspToolInput(action="eos", working_dir=str(d)), None
    )
    assert res.success is False
    assert "needs ≥4 points" in res.error


async def test_eos_dir_missing(tmp_path):
    res = await vt.VaspTool()._eos(
        vt.VaspToolInput(action="eos", working_dir="/no/such/dir"), None
    )
    assert res.success is False
    assert "Working directory not found" in res.error