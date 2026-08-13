"""vina_tool.py 集成路径补测 — 覆盖 is_read_only/is_destructive、call 分派、
_dock(受体/配体缺失/resolve 缺 executable·needs_resolution/成功·解析+审计/
失败/sandbox 拦截/超时/审计异常吞掉)、_score_only(pose 缺失/resolve 缺/
成功·提取 affinity/失败/sandbox 拦截/超时)、_prepare_ligand(meeko 成功/meeko
缺 fallback obabel/都缺)、_prepare_with_meeko(缺 mk/缺 sdf/成功/失败)、
_prepare_with_obabel(缺 sdf/成功/失败)、_ensure_sdf(input_sdf/不存在/smiles·
rdkit 成功/smiles·import 失败/无源)、_resolve_file、_resolve_vina(python 包/
resolver)、_parse_vina_output(多 pose/无 pose/strong·moderate·weak·no_binding)、
_extract_score(命中/无).

配合 tests/test_bio_tools.py, 把 vina_tool.py 覆盖率提升到 90%+.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from huginn.security import SandboxError
from huginn.tools.sim.executable_resolver import (
    ResolutionRequest,
    ToolExecutableSpec,
)
from huginn.tools.sim.vina_tool import VinaTool, VinaToolInput


def _res_req():
    """Build a real ResolutionRequest for autodock_vina."""
    return ResolutionRequest(
        tool_name="autodock_vina",
        spec=ToolExecutableSpec(
            name="autodock_vina",
            env_vars=("VINA_EXECUTABLE",),
            basenames=("vina",),
            install_hint="install vina",
        ),
    )

pytestmark = pytest.mark.anyio


def _tool(**kw):
    return VinaTool(**kw)


def _dock_args(**kw):
    base = {"action": "dock", "receptor_pdbqt": "rec.pdbqt", "ligand_pdbqt": "lig.pdbqt"}
    base.update(kw)
    return base


def _score_args(**kw):
    base = {"action": "score_only", "pose_pdbqt": "pose.pdbqt"}
    base.update(kw)
    return base


def _prep_args(**kw):
    base = {"action": "prepare_ligand", "smiles": "CCO"}
    base.update(kw)
    return base


def _install_auditor(monkeypatch, payload=None):
    auditor_mod = types.ModuleType("huginn.execution.physics_auditor")
    state = {"payload": payload or {"has_errors": False}}

    class _Report:
        def to_dict(self):
            return state["payload"]

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


class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Sb:
    def __init__(self, proc=None, call=None):
        self._proc = proc or _Proc()
        self._call = call

    def run(self, cmd, cwd=None, capture_output=True, text=True, timeout=None):
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


def test_is_read_only():
    t = _tool()
    assert t.is_read_only(VinaToolInput(action="score_only")) is True
    assert t.is_read_only(VinaToolInput(action="dock")) is False


def test_is_destructive():
    t = _tool()
    assert t.is_destructive(VinaToolInput(action="dock")) is True
    assert t.is_destructive(VinaToolInput(action="prepare_ligand")) is True
    assert t.is_destructive(VinaToolInput(action="score_only")) is False


def test_call_unknown_action():
    res = _tool().call({"action": "dock", "receptor_pdbqt": None}, context=None)
    assert res.success is False
    assert "Receptor PDBQT not found" in res.error


# ── _dock ─────────────────────────────────────────────────────────────────


def test_dock_receptor_missing(tmp_path):
    res = _tool().call(_dock_args(receptor_pdbqt="nope.pdbqt"), context=None)
    assert res.success is False
    assert "Receptor PDBQT not found" in res.error


def test_dock_ligand_missing(tmp_path):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    res = _tool().call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt="nope.pdbqt"), context=None
    )
    assert res.success is False
    assert "Ligand PDBQT not found" in res.error


def test_dock_needs_resolution(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return _res_req()

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    monkeypatch.delitem(sys.modules, "vina", raising=False)
    res = _tool().call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.success is False
    assert "AutoDock Vina executable not found" in res.error
    assert res.metadata["needs_resolution"] is True


def test_dock_success(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    _install_auditor(monkeypatch)
    tool = _tool()
    stdout = "   1   -7.3   0.000   0.000\n   2   -6.5   0.100   0.200\n"
    tool.sandbox = _Sb(_Proc(returncode=0, stdout=stdout))
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.success is True
    assert res.data["n_poses"] == 2
    assert res.data["best_affinity"] == pytest.approx(-7.3)
    assert res.data["binding_strength"] == "moderate"
    assert res.data["physics_audit"]["has_errors"] is False


def test_dock_strong_binding(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="   1   -9.5   0.000   0.000\n"))
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.data["binding_strength"] == "strong"


def test_dock_weak_binding(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="   1   -1.0   0.000   0.000\n"))
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.data["binding_strength"] == "weak"


def test_dock_no_binding(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="no poses here\n"))
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.data["n_poses"] == 0
    assert res.data["binding_strength"] == "no_binding"


def test_dock_failure(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="dock crashed"))
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.success is False
    assert "Vina docking failed" in res.error


def test_dock_sandbox_blocked(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _SbErr()
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.success is False
    assert "Docking blocked by sandbox" in res.error


def test_dock_timeout(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _SbTimeout()
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.success is False
    assert "Docking timed out" in res.error


def test_dock_audit_boom_swallowed(tmp_path, monkeypatch):
    rec = tmp_path / "rec.pdbqt"
    rec.write_text("x")
    lig = tmp_path / "lig.pdbqt"
    lig.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    _install_auditor_boom(monkeypatch)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="   1   -7.3   0.000   0.000\n"))
    res = tool.call(
        _dock_args(receptor_pdbqt=str(rec), ligand_pdbqt=str(lig)), context=None
    )
    assert res.success is True
    assert "physics_audit" not in res.data


# ── _score_only ───────────────────────────────────────────────────────────


def test_score_pose_missing():
    res = _tool().call(_score_args(pose_pdbqt="nope.pdbqt"), context=None)
    assert res.success is False
    assert "Pose PDBQT not found" in res.error


def test_score_needs_resolution(tmp_path, monkeypatch):
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("x")
    monkeypatch.setattr(VinaTool, "_resolve_vina", staticmethod(lambda: _res_req()))
    res = _tool().call(
        _score_args(pose_pdbqt=str(pose)), context=None
    )
    assert res.success is False
    assert res.metadata["needs_resolution"] is True


def test_score_success(tmp_path, monkeypatch):
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="Affinity: -6.8 (kcal/mol)\n"))
    res = tool.call(_score_args(pose_pdbqt=str(pose)), context=None)
    assert res.success is True
    assert res.data["affinity"] == pytest.approx(-6.8)
    assert res.data["action"] == "score_only"


def test_score_no_affinity(tmp_path, monkeypatch):
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0, stdout="no affinity line\n"))
    res = tool.call(_score_args(pose_pdbqt=str(pose)), context=None)
    assert res.success is True
    assert res.data["affinity"] is None


def test_score_failure(tmp_path, monkeypatch):
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="boom"))
    res = tool.call(_score_args(pose_pdbqt=str(pose)), context=None)
    assert res.success is False
    assert "Scoring failed" in res.data["message"]


def test_score_sandbox_blocked(tmp_path, monkeypatch):
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _SbErr()
    res = tool.call(_score_args(pose_pdbqt=str(pose)), context=None)
    assert res.success is False
    assert "Scoring blocked by sandbox" in res.error


def test_score_timeout(tmp_path, monkeypatch):
    pose = tmp_path / "pose.pdbqt"
    pose.write_text("x")
    @staticmethod
    def _resolve_vina():
        return "/usr/bin/vina"

    monkeypatch.setattr(VinaTool, "_resolve_vina", _resolve_vina)
    tool = _tool()
    tool.sandbox = _SbTimeout()
    res = tool.call(_score_args(pose_pdbqt=str(pose)), context=None)
    assert res.success is False
    assert "Scoring timed out" in res.error


# ── _prepare_ligand ───────────────────────────────────────────────────────


def test_prepare_meeko_success(tmp_path, monkeypatch):
    monkeypatch.setattr(VinaTool, "_prepare_with_meeko", staticmethod(
        lambda i, o, w: (
            __import__("huginn.core_types", fromlist=["ToolResult"]).ToolResult(
                data={"preparer": "meeko"}, success=True
            )
        )
    ))
    res = _tool().call(_prep_args(), context=None)
    assert res.success is True


def test_prepare_meeko_missing_fallback_obabel(tmp_path, monkeypatch):
    from huginn.core_types import ToolResult

    def _m(self, i, o, w):
        raise FileNotFoundError("no meeko")

    def _o(self, i, o, w):
        return ToolResult(data={"preparer": "obabel"}, success=True)

    monkeypatch.setattr(VinaTool, "_prepare_with_meeko", _m)
    monkeypatch.setattr(VinaTool, "_prepare_with_obabel", _o)
    res = _tool().call(_prep_args(), context=None)
    assert res.success is True
    assert res.data["preparer"] == "obabel"


def test_prepare_both_missing(tmp_path, monkeypatch):

    def _m(self, i, o, w):
        raise FileNotFoundError("no meeko")

    def _o(self, i, o, w):
        raise FileNotFoundError("no obabel")

    monkeypatch.setattr(VinaTool, "_prepare_with_meeko", _m)
    monkeypatch.setattr(VinaTool, "_prepare_with_obabel", _o)
    res = _tool().call(_prep_args(), context=None)
    assert res.success is False
    assert "Neither meeko nor obabel found" in res.error


# ── _prepare_with_meeko ───────────────────────────────────────────────────


def test_meeko_no_binary(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: None)
    with pytest.raises(FileNotFoundError):
        _tool()._prepare_with_meeko(
            VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path / "l.pdbqt", tmp_path
        )


def test_meeko_no_sdf(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/mk")
    monkeypatch.setattr(VinaTool, "_ensure_sdf", staticmethod(lambda i, w: None))
    res = _tool()._prepare_with_meeko(
        VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path / "l.pdbqt", tmp_path
    )
    assert res.success is False
    assert "Could not obtain SDF" in res.error


def test_meeko_success(tmp_path, monkeypatch):
    sdf = tmp_path / "lig.sdf"
    sdf.write_text("x")
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/mk")
    monkeypatch.setattr(VinaTool, "_ensure_sdf", staticmethod(lambda i, w: sdf))
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0))
    res = tool._prepare_with_meeko(
        VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path / "l.pdbqt", tmp_path
    )
    assert res.success is True
    assert res.data["preparer"] == "meeko"


def test_meeko_failure(tmp_path, monkeypatch):
    sdf = tmp_path / "lig.sdf"
    sdf.write_text("x")
    monkeypatch.setattr("shutil.which", lambda n: "/usr/bin/mk")
    monkeypatch.setattr(VinaTool, "_ensure_sdf", staticmethod(lambda i, w: sdf))
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="mk failed"))
    res = tool._prepare_with_meeko(
        VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path / "l.pdbqt", tmp_path
    )
    assert res.success is False
    assert "meeko ligand preparation failed" in res.error


# ── _prepare_with_obabel ──────────────────────────────────────────────────


def test_obabel_no_sdf(tmp_path, monkeypatch):
    monkeypatch.setattr(VinaTool, "_ensure_sdf", lambda self, i, w: None)
    res = _tool()._prepare_with_obabel(
        VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path / "l.pdbqt", tmp_path
    )
    assert res.success is False


def test_obabel_success(tmp_path, monkeypatch):
    sdf = tmp_path / "lig.sdf"
    sdf.write_text("x")
    monkeypatch.setattr(VinaTool, "_ensure_sdf", lambda self, i, w: sdf)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=0))
    res = tool._prepare_with_obabel(
        VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path / "l.pdbqt", tmp_path
    )
    assert res.success is True
    assert res.data["preparer"] == "obabel"


def test_obabel_failure(tmp_path, monkeypatch):
    sdf = tmp_path / "lig.sdf"
    sdf.write_text("x")
    monkeypatch.setattr(VinaTool, "_ensure_sdf", lambda self, i, w: sdf)
    tool = _tool()
    tool.sandbox = _Sb(_Proc(returncode=1, stderr="obabel failed"))
    res = tool._prepare_with_obabel(
        VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path / "l.pdbqt", tmp_path
    )
    assert res.success is False
    assert "obabel ligand preparation failed" in res.error


# ── _ensure_sdf ───────────────────────────────────────────────────────────


def test_ensure_sdf_input_sdf(tmp_path):
    sdf = tmp_path / "in.sdf"
    sdf.write_text("x")
    r = _tool()._ensure_sdf(VinaToolInput(action="prepare_ligand", input_sdf=str(sdf)), tmp_path)
    assert r == sdf


def test_ensure_sdf_input_sdf_missing(tmp_path):
    r = _tool()._ensure_sdf(
        VinaToolInput(action="prepare_ligand", input_sdf="/no/x.sdf"), tmp_path
    )
    assert r is None


def test_ensure_sdf_smiles_rdkit(tmp_path, monkeypatch):
    rdkit_pkg = types.ModuleType("rdkit")
    chem_mod = types.ModuleType("rdkit.Chem")
    allchem_mod = types.ModuleType("rdkit.Chem.AllChem")

    class _Mol:
        pass

    class _Writer:
        def __init__(self, path):
            self.path = path

        def write(self, mol):
            Path(self.path).write_text("sdf-content")

        def close(self):
            pass

    chem_mod.MolFromSmiles = lambda s: _Mol()
    chem_mod.AddHs = lambda m: m
    chem_mod.SDWriter = _Writer

    class _AllChem:
        @staticmethod
        def EmbedMolecule(mol, useRandomCoords=True):
            pass

        @staticmethod
        def MMFFOptimizeMolecule(mol):
            return 0

    allchem_mod.EmbedMolecule = _AllChem.EmbedMolecule
    allchem_mod.MMFFOptimizeMolecule = _AllChem.MMFFOptimizeMolecule

    rdkit_pkg.Chem = chem_mod
    chem_mod.AllChem = allchem_mod
    monkeypatch.setitem(sys.modules, "rdkit", rdkit_pkg)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", chem_mod)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.AllChem", allchem_mod)

    r = _tool()._ensure_sdf(VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path)
    assert r == tmp_path / "ligand.sdf"
    assert r.read_text() == "sdf-content"


def test_ensure_sdf_smiles_invalid(tmp_path, monkeypatch):
    rdkit_pkg = types.ModuleType("rdkit")
    chem_mod = types.ModuleType("rdkit.Chem")
    chem_mod.MolFromSmiles = lambda s: None
    chem_mod.AddHs = lambda m: m
    chem_mod.SDWriter = lambda p: (_ for _ in ()).throw(AssertionError("unused"))
    chem_mod.AllChem = types.SimpleNamespace(
        EmbedMolecule=lambda m, useRandomCoords=True: None,
        MMFFOptimizeMolecule=lambda m: 0,
    )
    rdkit_pkg.Chem = chem_mod
    monkeypatch.setitem(sys.modules, "rdkit", rdkit_pkg)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", chem_mod)
    r = _tool()._ensure_sdf(VinaToolInput(action="prepare_ligand", smiles="bad"), tmp_path)
    assert r is None


def test_ensure_sdf_smiles_import_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rdkit", None)
    r = _tool()._ensure_sdf(VinaToolInput(action="prepare_ligand", smiles="CCO"), tmp_path)
    assert r is None


def test_ensure_sdf_no_source(tmp_path):
    r = _tool()._ensure_sdf(
        VinaToolInput(action="prepare_ligand", smiles=None, input_sdf=None), tmp_path
    )
    assert r is None


# ── _resolve_file / _resolve_vina ─────────────────────────────────────────


def test_resolve_file(tmp_path):
    f = tmp_path / "a.pdbqt"
    f.write_text("x")
    t = _tool()
    assert t._resolve_file(None, tmp_path) is None
    assert t._resolve_file("nope.pdbqt", tmp_path) is None
    assert t._resolve_file(str(f), tmp_path) == f
    assert t._resolve_file("a.pdbqt", tmp_path) == f


def test_resolve_vina_python_pkg(monkeypatch):
    vina_mod = types.ModuleType("vina")
    monkeypatch.setitem(sys.modules, "vina", vina_mod)
    assert VinaTool._resolve_vina() == "__python__"


def test_resolve_vina_fallback(monkeypatch):
    monkeypatch.delitem(sys.modules, "vina", raising=False)
    monkeypatch.setattr(
        "huginn.tools.sim.vina_tool.resolve_executable",
        lambda name: _res_req(),
    )
    assert isinstance(VinaTool._resolve_vina(), ResolutionRequest)


# ── _parse_vina_output ────────────────────────────────────────────────────


def test_parse_vina_output():
    stdout = "   1   -7.3   0.000   0.000\n   2   -6.5   0.100   0.200\n"
    info = VinaTool._parse_vina_output(stdout, Path("/tmp/nope.pdbqt"))
    assert info["n_poses"] == 2
    assert info["best_affinity"] == pytest.approx(-7.3)
    assert info["output_pdbqt"] is None  # 文件不存在
    assert info["binding_strength"] == "moderate"


def test_parse_vina_output_empty():
    info = VinaTool._parse_vina_output("nothing\n", Path("/tmp/nope.pdbqt"))
    assert info["n_poses"] == 0
    assert info["best_affinity"] is None
    assert info["binding_strength"] == "no_binding"


# ── _extract_score ────────────────────────────────────────────────────────


def test_extract_score_hit():
    assert VinaTool._extract_score("Affinity: -6.8 (kcal/mol)") == pytest.approx(-6.8)


def test_extract_score_miss():
    assert VinaTool._extract_score("no affinity") is None
