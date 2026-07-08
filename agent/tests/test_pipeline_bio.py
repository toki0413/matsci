"""Tests for the expanded pipeline with bio-pharma and cross-pollination tools."""

from __future__ import annotations

import pytest

from huginn.provenance.pipeline import (
    PipelineRule,
    PipelineStage,
    PIPELINE_RULES,
    SimulationPipeline,
    _infer_stage,
    _KNOWN_TOOLS,
    _STAGE_ORDER,
)


class TestPipelineStages:
    def test_new_stages_exist(self):
        assert hasattr(PipelineStage, "CHEMINFO")
        assert hasattr(PipelineStage, "DOCKING")
        assert hasattr(PipelineStage, "BIOMD")
        assert hasattr(PipelineStage, "FREE_ENERGY")
        assert hasattr(PipelineStage, "ENHANCED_SAMPLING")
        assert hasattr(PipelineStage, "KINETICS")
        assert hasattr(PipelineStage, "MOTIF_ANALYSIS")
        assert hasattr(PipelineStage, "INVERSE_DESIGN")
        assert hasattr(PipelineStage, "CONSENSUS")

    def test_stage_ordering(self):
        # Materials line
        assert _STAGE_ORDER[PipelineStage.STRUCTURE] < _STAGE_ORDER[PipelineStage.RELAX]
        assert _STAGE_ORDER[PipelineStage.RELAX] < _STAGE_ORDER[PipelineStage.STATIC]
        # Bio line
        assert _STAGE_ORDER[PipelineStage.CHEMINFO] < _STAGE_ORDER[PipelineStage.DOCKING]
        assert _STAGE_ORDER[PipelineStage.DOCKING] < _STAGE_ORDER[PipelineStage.FREE_ENERGY]
        # Cross-pollination
        assert _STAGE_ORDER[PipelineStage.FREE_ENERGY] < _STAGE_ORDER[PipelineStage.CONSENSUS]
        assert _STAGE_ORDER[PipelineStage.CONSENSUS] < _STAGE_ORDER[PipelineStage.ANALYSIS]

    def test_known_tools_updated(self):
        for tool in ["rdkit_tool", "vina_tool", "openmm_tool",
                     "fep_tool", "enhanced_sampling_tool", "msm_tool",
                     "inverse_design_tool", "motif_mining_tool", "consensus_scoring_tool"]:
            assert tool in _KNOWN_TOOLS, f"{tool} not in _KNOWN_TOOLS"


class TestInferStage:
    def test_rdkit(self):
        assert _infer_stage("rdkit_tool", {"action": "descriptors"}) == PipelineStage.CHEMINFO
        assert _infer_stage("rdkit_tool", {"action": "smiles_to_mol"}) == PipelineStage.CHEMINFO

    def test_vina(self):
        assert _infer_stage("vina_tool", {"action": "dock"}) == PipelineStage.DOCKING
        assert _infer_stage("vina_tool", {"action": "score_only"}) == PipelineStage.DOCKING

    def test_openmm(self):
        assert _infer_stage("openmm_tool", {"action": "energy_minimize"}) == PipelineStage.BIOMD
        assert _infer_stage("openmm_tool", {"action": "md_run"}) == PipelineStage.BIOMD
        assert _infer_stage("openmm_tool", {"action": "analyze"}) == PipelineStage.ANALYSIS

    def test_cross_pollination(self):
        assert _infer_stage("fep_tool", {"action": "ti"}) == PipelineStage.FREE_ENERGY
        assert _infer_stage("enhanced_sampling_tool", {"action": "wham"}) == PipelineStage.ENHANCED_SAMPLING
        assert _infer_stage("msm_tool", {"action": "build_msm"}) == PipelineStage.KINETICS
        assert _infer_stage("motif_mining_tool", {"action": "ring_analysis"}) == PipelineStage.MOTIF_ANALYSIS
        assert _infer_stage("inverse_design_tool", {"action": "pareto_frontier"}) == PipelineStage.INVERSE_DESIGN
        assert _infer_stage("consensus_scoring_tool", {"action": "borda"}) == PipelineStage.CONSENSUS


class TestPipelineRules:
    def test_drug_design_rules_exist(self):
        # rdkit → vina
        rdkit_rules = [r for r in PIPELINE_RULES if r.tool_name == "rdkit_tool"]
        assert len(rdkit_rules) >= 4
        # vina → fep/consensus
        vina_rules = [r for r in PIPELINE_RULES if r.tool_name == "vina_tool"]
        assert any(PipelineStage.FREE_ENERGY in r.next_stages for r in vina_rules)
        assert any(PipelineStage.CONSENSUS in r.next_stages for r in vina_rules)

    def test_biomd_rules_exist(self):
        openmm_rules = [r for r in PIPELINE_RULES if r.tool_name == "openmm_tool"]
        # energy_minimize → md_run
        assert any(r.action_matcher == "energy_minimize" for r in openmm_rules)
        # md_run → msm/enhanced_sampling/fep
        md_rules = [r for r in openmm_rules if r.action_matcher == "md_run"]
        assert len(md_rules) == 1
        assert PipelineStage.KINETICS in md_rules[0].next_stages
        assert PipelineStage.ENHANCED_SAMPLING in md_rules[0].next_stages

    def test_cross_pollination_bridge_rules(self):
        # LAMMPS → MSM/Enhanced Sampling (cross-pollination bridge)
        lammps_rules = [r for r in PIPELINE_RULES if r.tool_name == "lammps_tool"]
        has_bridge = any(
            PipelineStage.KINETICS in r.next_stages or PipelineStage.ENHANCED_SAMPLING in r.next_stages
            for r in lammps_rules
        )
        assert has_bridge, "LAMMPS should bridge to MSM/Enhanced Sampling"

        # VASP static → motif_mining (cross-pollination bridge)
        vasp_static_rules = [
            r for r in PIPELINE_RULES
            if r.tool_name == "vasp_tool" and r.action_matcher == "static"
        ]
        has_motif_bridge = any(
            PipelineStage.MOTIF_ANALYSIS in r.next_stages
            for r in vasp_static_rules
        )
        assert has_motif_bridge, "VASP static should bridge to Motif Mining"

    def test_inverse_design_loops_back(self):
        # inverse_design → relax (DFT verification) or docking
        inv_rules = [r for r in PIPELINE_RULES if r.tool_name == "inverse_design_tool"]
        assert len(inv_rules) == 1
        assert PipelineStage.RELAX in inv_rules[0].next_stages
        assert PipelineStage.DOCKING in inv_rules[0].next_stages

    def test_consensus_is_near_end(self):
        consensus_rules = [r for r in PIPELINE_RULES if r.tool_name == "consensus_scoring_tool"]
        assert len(consensus_rules) == 1
        assert PipelineStage.ANALYSIS in consensus_rules[0].next_stages


class TestSimulationPipelineSuggestions:
    """Test that suggest_next returns correct next steps for new tools."""

    def test_rdkit_suggests_docking_and_biomd(self):
        from huginn.provenance.registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        pipeline = SimulationPipeline(registry)

        suggestions = pipeline.suggest_next(
            "rdkit_tool",
            {"action": "smiles_to_mol"},
            {"canonical_smiles": "CCO"},
        )
        tool_hints = [s.tool_hint for s in suggestions]
        assert any("vina_tool" in h for h in tool_hints)
        assert any("openmm_tool" in h for h in tool_hints)

    def test_vina_dock_suggests_fep_and_consensus(self):
        from huginn.provenance.registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        pipeline = SimulationPipeline(registry)

        suggestions = pipeline.suggest_next(
            "vina_tool",
            {"action": "dock"},
            {"best_affinity": -8.5},
        )
        tool_hints = [s.tool_hint for s in suggestions]
        assert any("fep_tool" in h for h in tool_hints)
        assert any("consensus" in h for h in tool_hints)

    def test_openmm_md_suggests_msm_and_enhanced_sampling(self):
        from huginn.provenance.registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        pipeline = SimulationPipeline(registry)

        suggestions = pipeline.suggest_next(
            "openmm_tool",
            {"action": "md_run"},
            {"n_steps": 5000},
        )
        tool_hints = [s.tool_hint for s in suggestions]
        assert any("msm_tool" in h for h in tool_hints)
        assert any("enhanced_sampling" in h for h in tool_hints)

    def test_lammps_bridges_to_msm(self):
        from huginn.provenance.registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        pipeline = SimulationPipeline(registry)

        suggestions = pipeline.suggest_next(
            "lammps_tool",
            {"action": "md"},
            {"n_steps": 10000},
        )
        tool_hints = [s.tool_hint for s in suggestions]
        # Should suggest MSM (cross-pollination bridge)
        assert any("msm_tool" in h for h in tool_hints)
        assert any("enhanced_sampling" in h for h in tool_hints)

    def test_vasp_static_bridges_to_motif(self):
        from huginn.provenance.registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        pipeline = SimulationPipeline(registry)

        suggestions = pipeline.suggest_next(
            "vasp_tool",
            {"action": "static"},
            {"energy": -100.0},
        )
        tool_hints = [s.tool_hint for s in suggestions]
        assert any("motif_mining" in h for h in tool_hints)

    def test_fep_suggests_consensus(self):
        from huginn.provenance.registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        pipeline = SimulationPipeline(registry)

        suggestions = pipeline.suggest_next(
            "fep_tool",
            {"action": "ti"},
            {"delta_F_eV": -0.5},
        )
        tool_hints = [s.tool_hint for s in suggestions]
        assert any("consensus" in h for h in tool_hints)

    def test_inverse_design_loops_to_vasp(self):
        from huginn.provenance.registry import ProvenanceRegistry
        registry = ProvenanceRegistry()
        pipeline = SimulationPipeline(registry)

        suggestions = pipeline.suggest_next(
            "inverse_design_tool",
            {"action": "pareto_frontier"},
            {"n_pareto_optimal": 3},
        )
        tool_hints = [s.tool_hint for s in suggestions]
        assert any("vasp_tool" in h for h in tool_hints)
        assert any("vina_tool" in h for h in tool_hints)
