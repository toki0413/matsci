"""Tests for bio-pharma tools: RDKit, AutoDock Vina, OpenMM.

These tools depend on optional packages (rdkit, vina, openmm) and CLI
executables that are not available in CI. We mock the dependencies and
subprocess calls to exercise all code paths.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")

# ── RDKit ──

_RDKIT_AVAILABLE = True
try:
    import rdkit  # noqa: F401
except ImportError:
    _RDKIT_AVAILABLE = False


@pytest.mark.asyncio
@pytest.mark.skipif(not _RDKIT_AVAILABLE, reason="rdkit not installed")
class TestRDKitTool:
    async def test_smiles_to_mol(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(RDKitInput(action="smiles_to_mol", smiles="CCO"), CTX)
        assert result.success
        assert result.data["canonical_smiles"] == "CCO"
        assert result.data["num_heavy_atoms"] == 2
        assert "formula" in result.data

    async def test_descriptors(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(
            RDKitInput(action="descriptors", smiles="CC(=O)Oc1ccccc1C(=O)O"), CTX
        )
        assert result.success
        mol_data = result.data["molecules"][0]
        assert mol_data["molecular_weight"] > 170
        assert mol_data["logp"] > 0
        assert "lipinski_violations" in mol_data

    async def test_fingerprint(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(
            RDKitInput(action="fingerprint", smiles="CCO", fingerprint_type="morgan", radius=2, n_bits=1024),
            CTX,
        )
        assert result.success
        assert result.data["n_bits"] == 1024
        assert result.data["n_on_bits"] > 0
        assert result.data["density"] > 0

    async def test_similarity(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(
            RDKitInput(
                action="similarity",
                reference_smiles="CCO",
                query_smiles="CCO",
            ),
            CTX,
        )
        assert result.success
        assert result.data["tanimoto"] == 1.0
        assert result.data["interpretation"] == "highly similar"

    async def test_similarity_different(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(
            RDKitInput(
                action="similarity",
                reference_smiles="c1ccccc1",
                query_smiles="CCCCCCCC",
            ),
            CTX,
        )
        assert result.success
        assert result.data["tanimoto"] < 0.3

    async def test_substructure_search(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(
            RDKitInput(
                action="substructure_search",
                substructure="c1ccccc1",
                smiles_list=["CC(=O)Oc1ccccc1C(=O)O", "CCO", "c1ccccc1"],
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_hits"] == 2
        assert result.data["n_total"] == 3

    async def test_invalid_smiles(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(
            RDKitInput(action="smiles_to_mol", smiles="XYZNOTASMILES"), CTX
        )
        assert not result.success
        assert "Invalid SMILES" in result.error

    async def test_draw(self, tmp_path: Path):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        out = tmp_path / "mol.png"
        result = await tool.call(
            RDKitInput(action="draw", smiles="CCO", output_file=str(out)), CTX
        )
        assert result.success
        assert out.exists()
        assert result.data["image_file"] == str(out)

    async def test_conformers(self, tmp_path: Path):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        out = tmp_path / "confs.sdf"
        result = await tool.call(
            RDKitInput(action="conformers", smiles="CCO", n_conformers=3, output_file=str(out)),
            CTX,
        )
        assert result.success
        assert out.exists()
        assert result.data["n_conformers"] == 3
        assert len(result.data["rmsd_to_first"]) == 2

    async def test_smiles_to_sdf(self, tmp_path: Path):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        out = tmp_path / "mols.sdf"
        result = await tool.call(
            RDKitInput(
                action="smiles_to_sdf",
                smiles_list=["CCO", "c1ccccc1", "CC(=O)O"],
                output_file=str(out),
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_written"] == 3
        assert out.exists()


class TestRDKitToolNoDeps:
    """Test error paths when rdkit is not installed."""

    @pytest.mark.asyncio
    async def test_missing_smiles(self):
        from huginn.tools.sci.rdkit_tool import RDKitTool, RDKitInput

        tool = RDKitTool()
        result = await tool.call(RDKitInput(action="smiles_to_mol"), CTX)
        assert not result.success
        # Without rdkit we get an ImportError message; with rdkit but no smiles we get validation error
        assert "smiles is required" in result.error or "rdkit" in result.error.lower()


# ── AutoDock Vina ──


class TestVinaTool:
    def test_dock_no_receptor(self, tmp_path: Path):
        from huginn.tools.sim.vina_tool import VinaTool

        tool = VinaTool()
        result = tool.call(
            {
                "action": "dock",
                "working_dir": str(tmp_path),
                "receptor_pdbqt": "missing.pdbqt",
                "ligand_pdbqt": "missing_lig.pdbqt",
                "center_x": 0, "center_y": 0, "center_z": 0,
            },
            CTX,
        )
        assert not result.success
        assert "Receptor PDBQT not found" in result.error

    def test_dock_no_executable_returns_resolution(self, tmp_path: Path):
        from huginn.tools.sim.vina_tool import VinaTool

        # Create dummy PDBQT files
        (tmp_path / "receptor.pdbqt").write_text("ATOM  1  CA  ALA A  1\nEND")
        (tmp_path / "ligand.pdbqt").write_text("ATOM  1  C   LIG B  1\nEND")

        with patch("huginn.tools.sim.vina_tool.resolve_executable") as mock_resolve:
            from huginn.tools.sim.executable_resolver import ResolutionRequest, ToolExecutableSpec

            mock_resolve.return_value = ResolutionRequest(
                tool_name="autodock_vina",
                spec=ToolExecutableSpec(
                    name="autodock_vina",
                    env_vars=("VINA_EXECUTABLE",),
                    basenames=("vina",),
                    install_hint="conda install -c conda-forge autodock-vina",
                    conda_package="conda-forge/autodock-vina",
                ),
            )
            tool = VinaTool()
            result = tool.call(
                {
                    "action": "dock",
                    "working_dir": str(tmp_path),
                    "receptor_pdbqt": "receptor.pdbqt",
                    "ligand_pdbqt": "ligand.pdbqt",
                    "center_x": 10, "center_y": 20, "center_z": 30,
                },
                CTX,
            )
        assert not result.success
        assert result.metadata.get("needs_resolution") is True
        req = result.metadata["resolution_request"]
        assert req["tool_name"] == "autodock_vina"
        assert "提供本地安装路径" in req["options"]

    def test_score_only_no_file(self, tmp_path: Path):
        from huginn.tools.sim.vina_tool import VinaTool

        tool = VinaTool()
        result = tool.call(
            {"action": "score_only", "working_dir": str(tmp_path), "pose_pdbqt": "missing.pdbqt"},
            CTX,
        )
        assert not result.success
        assert "Pose PDBQT not found" in result.error

    def test_parse_vina_output(self):
        from huginn.tools.sim.vina_tool import VinaTool

        stdout = """\
1       -7.3      0.000      0.000
2       -6.8      1.234      2.345
3       -6.5      1.567      3.789
"""
        poses = VinaTool._parse_vina_output(stdout, Path("docked.pdbqt"))
        assert poses["n_poses"] == 3
        assert poses["best_affinity"] == -7.3
        assert poses["binding_strength"] == "moderate"

    def test_parse_vina_output_strong(self):
        from huginn.tools.sim.vina_tool import VinaTool

        stdout = "1       -9.5      0.000      0.000\n"
        poses = VinaTool._parse_vina_output(stdout, Path("docked.pdbqt"))
        assert poses["binding_strength"] == "strong"

    def test_extract_score(self):
        from huginn.tools.sim.vina_tool import VinaTool

        stdout = "Affinity: -8.5"
        assert VinaTool._extract_score(stdout) == -8.5

    def test_executable_resolver_has_vina(self):
        from huginn.tools.sim.executable_resolver import _REGISTRY

        assert "autodock_vina" in _REGISTRY
        spec = _REGISTRY["autodock_vina"]
        assert "vina" in spec.basenames
        assert "VINA_EXECUTABLE" in spec.env_vars
        assert not spec.license_required


# ── OpenMM ──


class TestOpenMMTool:
    def test_minimize_no_pdb(self, tmp_path: Path):
        from huginn.tools.sim.openmm_tool import OpenMMTool

        tool = OpenMMTool()
        result = tool.call(
            {"action": "energy_minimize", "working_dir": str(tmp_path), "pdb_file": "missing.pdb"},
            CTX,
        )
        assert not result.success
        assert "PDB file not found" in result.error

    def test_minimize_no_openmm(self, tmp_path: Path):
        from huginn.tools.sim.openmm_tool import OpenMMTool

        (tmp_path / "protein.pdb").write_text(
            "ATOM      1  N   ALA A   1      11.104  6.134  -6.504  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1      11.195  5.032  -5.622  1.00  0.00           C\n"
            "END\n"
        )
        with patch.dict("sys.modules", {"openmm": None, "openmm.app": None}):
            tool = OpenMMTool()
            result = tool.call(
                {
                    "action": "energy_minimize",
                    "working_dir": str(tmp_path),
                    "pdb_file": "protein.pdb",
                    "solvent": "vacuum",
                },
                CTX,
            )
        # Should return skipped status, not crash
        assert result.success or "openmm not installed" in (result.error or "")

    def test_md_run_no_pdb(self, tmp_path: Path):
        from huginn.tools.sim.openmm_tool import OpenMMTool

        tool = OpenMMTool()
        result = tool.call(
            {"action": "md_run", "working_dir": str(tmp_path), "pdb_file": "missing.pdb"},
            CTX,
        )
        assert not result.success
        assert "PDB file not found" in result.error

    def test_analyze_no_trajectory(self, tmp_path: Path):
        from huginn.tools.sim.openmm_tool import OpenMMTool

        tool = OpenMMTool()
        result = tool.call(
            {"action": "analyze", "working_dir": str(tmp_path), "trajectory_file": "missing.dcd"},
            CTX,
        )
        assert not result.success
        assert "Trajectory file not found" in result.error

    def test_parse_md_log(self, tmp_path: Path):
        from huginn.tools.sim.openmm_tool import OpenMMTool

        log = tmp_path / "md_log.csv"
        log.write_text(
            'step,potential_energy (kJ/mole),kinetic_energy (kJ/mole),temperature (K),volume (nm^3),speed (ns/day)\n'
            '0,-5000.0,2000.0,298.0,30.0,0.0\n'
            '500,-4800.0,2100.0,300.5,30.1,15.2\n'
        )
        series = OpenMMTool._parse_md_log(log)
        assert series["n_data_points"] == 2
        assert series["steps"] == [0, 500]
        assert series["temperatures"] == [298.0, 300.5]


# ── Hooks ──


class TestBioPharmaHooks:
    @pytest.mark.asyncio
    async def test_vina_docking_hook_no_poses(self):
        from huginn.hooks.science_hooks import vina_docking_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="vina_tool",
            args={},
            result={"result": {"action": "dock", "poses": [], "best_affinity": None}},
        )
        out = await vina_docking_hook(ctx)
        assert out is not None
        assert ctx.metadata.get("blocked_by_hook") is True

    @pytest.mark.asyncio
    async def test_vina_docking_hook_positive_affinity(self):
        from huginn.hooks.science_hooks import vina_docking_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="vina_tool",
            args={},
            result={"result": {"action": "dock", "poses": [{"rank": 1}], "best_affinity": 2.5}},
        )
        out = await vina_docking_hook(ctx)
        assert out is not None
        assert ctx.metadata.get("blocked_by_hook") is True

    @pytest.mark.asyncio
    async def test_vina_docking_hook_weak_binding(self):
        from huginn.hooks.science_hooks import vina_docking_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="vina_tool",
            args={},
            result={"result": {"action": "dock", "poses": [{"rank": 1}], "best_affinity": -2.0}},
        )
        out = await vina_docking_hook(ctx)
        # Should warn but not block
        assert out is None
        assert ctx.metadata.get("warnings")
        assert "偏弱" in ctx.metadata["warnings"][0]

    @pytest.mark.asyncio
    async def test_vina_docking_hook_good(self):
        from huginn.hooks.science_hooks import vina_docking_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="vina_tool",
            args={},
            result={"result": {"action": "dock", "poses": [{"rank": 1}], "best_affinity": -8.5}},
        )
        out = await vina_docking_hook(ctx)
        assert out is None
        assert "blocked_by_hook" not in ctx.metadata

    @pytest.mark.asyncio
    async def test_openmm_stability_hook_nan(self):
        from huginn.hooks.science_hooks import openmm_stability_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="openmm_tool",
            args={},
            result={"result": {"action": "md_run", "final_energy_kj_mol": float("nan")}},
        )
        out = await openmm_stability_hook(ctx)
        assert out is not None
        assert ctx.metadata.get("blocked_by_hook") is True

    @pytest.mark.asyncio
    async def test_openmm_stability_hook_energy_increase(self):
        from huginn.hooks.science_hooks import openmm_stability_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="openmm_tool",
            args={},
            result={"result": {
                "action": "energy_minimize",
                "initial_energy_kj_mol": -1000.0,
                "final_energy_kj_mol": -500.0,
            }},
        )
        out = await openmm_stability_hook(ctx)
        assert out is None  # warns, doesn't block
        assert ctx.metadata.get("warnings")
        assert "升高" in ctx.metadata["warnings"][0]

    @pytest.mark.asyncio
    async def test_openmm_stability_hook_temp_drift(self):
        from huginn.hooks.science_hooks import openmm_stability_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="openmm_tool",
            args={},
            result={"result": {
                "action": "md_run",
                "temperature_k": 300.0,
                "time_series": {"temperatures": [400.0, 410.0, 395.0]},
            }},
        )
        out = await openmm_stability_hook(ctx)
        assert out is None
        assert ctx.metadata.get("warnings")
        assert "偏离" in ctx.metadata["warnings"][0]

    @pytest.mark.asyncio
    async def test_hooks_ignore_other_tools(self):
        from huginn.hooks.science_hooks import vina_docking_hook, openmm_stability_hook
        from huginn.hooks import HookContext

        ctx = HookContext(
            tool_name="vasp_tool",
            args={},
            result={"result": {"action": "dock", "poses": []}},
        )
        assert await vina_docking_hook(ctx) is None
        assert await openmm_stability_hook(ctx) is None


# ── Tool Registration ──


class TestToolRegistration:
    def test_rdkit_tool_registered(self):
        from huginn.tools.registry import ToolRegistry
        from huginn.tools import register_all_tools

        register_all_tools()
        tools = ToolRegistry.list_tools()
        assert "rdkit_tool" in tools

    def test_vina_tool_registered(self):
        from huginn.tools.registry import ToolRegistry
        from huginn.tools import register_all_tools

        register_all_tools()
        tools = ToolRegistry.list_tools()
        assert "vina_tool" in tools

    def test_openmm_tool_registered(self):
        from huginn.tools.registry import ToolRegistry
        from huginn.tools import register_all_tools

        register_all_tools()
        tools = ToolRegistry.list_tools()
        assert "openmm_tool" in tools
