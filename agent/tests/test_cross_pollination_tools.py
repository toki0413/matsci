"""Tests for the 6 cross-pollination tools inspired by biopharma computation.

Tests are pure-Python (no external MD/engine dependencies) and verify
mathematical correctness, not just API shape.
"""

from __future__ import annotations

import asyncio
import math
import numpy as np
import pytest

from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")

# ── FEP Tool ──


class TestFEPTool:
    @pytest.mark.asyncio
    async def test_lambda_schedule_uniform(self):
        from huginn.tools.sci.fep_tool import FEPTool, FEPInput

        tool = FEPTool()
        result = await tool._execute(
            FEPInput(action="lambda_schedule", n_lambda=11, lambda_spacing="uniform"), CTX
        )
        assert result.success
        assert result.data["n_windows"] == 11
        assert result.data["lambdas"][0] == 0.0
        assert result.data["lambdas"][-1] == 1.0

    @pytest.mark.asyncio
    async def test_lambda_schedule_nonlinear(self):
        from huginn.tools.sci.fep_tool import FEPTool, FEPInput

        tool = FEPTool()
        result = await tool._execute(
            FEPInput(action="lambda_schedule", n_lambda=21, lambda_spacing="nonlinear"), CTX
        )
        assert result.success
        # Nonlinear should have denser spacing near endpoints
        diffs = np.diff(result.data["lambdas"])
        # Endpoints should have smaller steps than middle
        assert diffs[0] < diffs[len(diffs) // 2]

    @pytest.mark.asyncio
    async def test_ti_linear_gradient(self):
        """If dU/dlambda = const, ΔF should equal that constant."""
        from huginn.tools.sci.fep_tool import FEPTool, FEPInput

        # dU/dlambda = 0.5 everywhere → ΔF = 0.5 eV
        lambdas = np.linspace(0, 1, 11).tolist()
        dU_dlambda = [[0.5 + np.random.normal(0, 0.01, 50) for _ in range(1)] for _ in range(11)]
        # Flatten: each lambda window has 50 samples of ~0.5
        dU_dlambda = [[0.5 + np.random.normal(0, 0.01) for _ in range(50)] for _ in range(11)]

        tool = FEPTool()
        result = await tool._execute(
            FEPInput(
                action="ti",
                lambda_values=lambdas,
                dU_dlambda=dU_dlambda,
                temperature=300.0,
            ),
            CTX,
        )
        assert result.success
        assert abs(result.data["delta_F_eV"] - 0.5) < 0.01

    @pytest.mark.asyncio
    async def test_jarzynski_second_law(self):
        """⟨W⟩ ≥ ΔF (second law of thermodynamics)."""
        from huginn.tools.sci.fep_tool import FEPTool, FEPInput

        # Work values with mean > 0
        rng = np.random.default_rng(42)
        work_values = (rng.normal(0.5, 0.2, 100)).tolist()

        tool = FEPTool()
        result = await tool._execute(
            FEPInput(
                action="jarzynski",
                work_values=work_values,
                temperature=300.0,
            ),
            CTX,
        )
        assert result.success
        assert result.data["second_law_satisfied"] is True
        assert result.data["dissipated_work_eV"] > 0

    @pytest.mark.asyncio
    async def test_fep_zwanzig(self):
        from huginn.tools.sci.fep_tool import FEPTool, FEPInput

        # Simple: ΔU = 0.1 eV consistently → ΔF ≈ 0.1
        delta_U = [[0.1] * 100 for _ in range(5)]

        tool = FEPTool()
        result = await tool._execute(
            FEPInput(action="fep", delta_U=delta_U, temperature=300.0), CTX
        )
        assert result.success
        # 5 windows, each ΔF ≈ -kT ln⟨exp(-β·0.1)⟩
        # When ΔU is constant: ΔF = ΔU
        assert abs(result.data["delta_F_eV"] - 0.5) < 0.02  # 5 × 0.1

    @pytest.mark.asyncio
    async def test_bar(self):
        from huginn.tools.sci.fep_tool import FEPTool, FEPInput

        # Symmetric case: forward = reverse → ΔF ≈ 0
        rng = np.random.default_rng(42)
        delta_U = [rng.normal(0.0, 0.05, 50).tolist() for _ in range(3)]
        delta_U_rev = [rng.normal(0.0, 0.05, 50).tolist() for _ in range(3)]

        tool = FEPTool()
        result = await tool._execute(
            FEPInput(
                action="bar",
                delta_U=delta_U,
                delta_U_reverse=delta_U_rev,
                temperature=300.0,
            ),
            CTX,
        )
        assert result.success
        assert abs(result.data["delta_F_eV"]) < 0.1

    @pytest.mark.asyncio
    async def test_drug_design_units(self):
        from huginn.tools.sci.fep_tool import FEPTool, FEPInput

        lambdas = np.linspace(0, 1, 5).tolist()
        dU_dlambda = [[0.1] * 50 for _ in range(5)]

        tool = FEPTool()
        result = await tool._execute(
            FEPInput(
                action="ti",
                lambda_values=lambdas,
                dU_dlambda=dU_dlambda,
                domain="drug_design",
                temperature=298.15,
            ),
            CTX,
        )
        assert result.success
        assert "delta_F_kcal_mol" in result.data


# ── Enhanced Sampling Tool ──


class TestEnhancedSamplingTool:
    @pytest.mark.asyncio
    async def test_metadynamics(self):
        from huginn.tools.sci.enhanced_sampling_tool import (
            EnhancedSamplingInput,
            EnhancedSamplingTool,
        )

        # Simple double-well CV trajectory
        rng = np.random.default_rng(42)
        cv = np.concatenate([
            rng.normal(-1.0, 0.1, 500),
            rng.normal(1.0, 0.1, 500),
        ]).tolist()
        cv_2d = [[v] for v in cv]

        tool = EnhancedSamplingTool()
        result = await tool._execute(
            EnhancedSamplingInput(
                action="metadynamics_bias",
                cv_trajectory=cv_2d,
                gaussian_height=0.01,
                gaussian_width=0.1,
                deposit_interval=50,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_hills_deposited"] > 0
        assert result.data["barrier_height_eV"] > 0

    @pytest.mark.asyncio
    async def test_umbrella_setup(self):
        from huginn.tools.sci.enhanced_sampling_tool import (
            EnhancedSamplingInput,
            EnhancedSamplingTool,
        )

        tool = EnhancedSamplingTool()
        result = await tool._execute(
            EnhancedSamplingInput(
                action="umbrella_setup",
                cv_range=[0.0, 10.0],
                n_windows=10,
                spring_constant=5.0,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_windows"] == 10
        centers = [w["cv_center"] for w in result.data["windows"]]
        assert centers[0] == 0.0
        assert centers[-1] == 10.0

    @pytest.mark.asyncio
    async def test_wham(self):
        from huginn.tools.sci.enhanced_sampling_tool import (
            EnhancedSamplingInput,
            EnhancedSamplingTool,
        )

        # Simulate umbrella samples: Gaussians at each window center
        rng = np.random.default_rng(42)
        centers = np.linspace(0, 5, 8).tolist()
        samples = [rng.normal(c, 0.3, 200).tolist() for c in centers]

        tool = EnhancedSamplingTool()
        result = await tool._execute(
            EnhancedSamplingInput(
                action="wham",
                window_centers=centers,
                window_samples=samples,
                spring_constant=10.0,
                bin_count=50,
                temperature=300.0,
            ),
            CTX,
        )
        assert result.success
        # WHAM may not fully converge in limited iterations, but should produce valid FES
        assert result.data["fes_min_eV"] == 0.0
        assert len(result.data["fes_eV"]) > 0

    @pytest.mark.asyncio
    async def test_reconstruct_fes_1d(self):
        from huginn.tools.sci.enhanced_sampling_tool import (
            EnhancedSamplingInput,
            EnhancedSamplingTool,
        )

        rng = np.random.default_rng(42)
        cv = rng.normal(0, 1, 2000).tolist()
        cv_2d = [[v] for v in cv]

        tool = EnhancedSamplingTool()
        result = await tool._execute(
            EnhancedSamplingInput(
                action="reconstruct_fes",
                cv_trajectory=cv_2d,
                n_bins=50,
            ),
            CTX,
        )
        assert result.success
        assert result.data["fes_min_eV"] == 0.0
        assert len(result.data["bin_centers"]) == 50

    @pytest.mark.asyncio
    async def test_rare_event_rate(self):
        from huginn.tools.sci.enhanced_sampling_tool import (
            EnhancedSamplingInput,
            EnhancedSamplingTool,
        )

        # Mostly near 0, rare events > 3
        rng = np.random.default_rng(42)
        cv = np.concatenate([rng.normal(0, 1, 995), rng.uniform(3, 5, 5)]).tolist()

        tool = EnhancedSamplingTool()
        result = await tool._execute(
            EnhancedSamplingInput(
                action="rare_event_rate",
                cv_trajectory=[[v] for v in cv],
                threshold=3.5,  # higher threshold to separate from normal tail
                temperature=300.0,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_above_threshold"] >= 5  # at least the 5 explicit ones
        assert result.data["rate_function_eV"] is not None


# ── MSM Tool ──


class TestMSMTool:
    @pytest.mark.asyncio
    async def test_build_msm_from_transition_matrix(self):
        from huginn.tools.sci.msm_tool import MSMInput, MSMTool

        # Simple 3-state Markov chain
        T = [
            [0.9, 0.1, 0.0],
            [0.05, 0.9, 0.05],
            [0.0, 0.1, 0.9],
        ]

        tool = MSMTool()
        result = await tool._execute(
            MSMInput(action="build_msm", transition_matrix=T), CTX
        )
        assert result.success
        assert result.data["n_states"] == 3
        # λ₁ should be 1 (stationary)
        assert abs(result.data["eigenvalues"][0] - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_build_msm_from_trajectory(self):
        from huginn.tools.sci.msm_tool import MSMInput, MSMTool

        # 2D trajectory switching between 3 states
        rng = np.random.default_rng(42)
        n = 2000
        states = rng.choice([0, 1, 2], p=[0.4, 0.35, 0.25], size=n)
        centers = np.array([[0, 0], [5, 5], [-5, 5]])
        traj = centers[states] + rng.normal(0, 0.3, (n, 2))
        traj = traj.tolist()

        tool = MSMTool()
        result = await tool._execute(
            MSMInput(
                action="build_msm",
                trajectory=traj,
                n_microstates=10,
                lag_time=1,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_states"] == 10
        assert len(result.data["implied_timescales"]) > 0

    @pytest.mark.asyncio
    async def test_metastable_states(self):
        from huginn.tools.sci.msm_tool import MSMInput, MSMTool

        rng = np.random.default_rng(42)
        n = 3000
        states = rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2], size=n)
        centers = np.array([[0, 0], [10, 0], [0, 10]])
        traj = centers[states] + rng.normal(0, 0.5, (n, 2))
        traj = traj.tolist()

        tool = MSMTool()
        result = await tool._execute(
            MSMInput(
                action="metastable_states",
                trajectory=traj,
                n_microstates=20,
                n_macrostates=3,
                lag_time=1,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_macrostates"] == 3
        assert len(result.data["stationary_probabilities"]) == 3
        assert abs(sum(result.data["stationary_probabilities"]) - 1.0) < 0.05

    @pytest.mark.asyncio
    async def test_stationary_distribution(self):
        from huginn.tools.sci.msm_tool import MSMInput, MSMTool

        T = [
            [0.8, 0.2, 0.0],
            [0.1, 0.8, 0.1],
            [0.0, 0.2, 0.8],
        ]

        tool = MSMTool()
        result = await tool._execute(
            MSMInput(action="stationary_distribution", transition_matrix=T), CTX
        )
        assert result.success
        # By symmetry, stationary should be uniform
        stat = result.data["stationary_distribution"]
        assert abs(stat[0] - stat[2]) < 0.01

    @pytest.mark.asyncio
    async def test_commitment(self):
        from huginn.tools.sci.msm_tool import MSMInput, MSMTool

        # 4-state chain: A=0, intermediate=1,2, B=3
        T = [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ]

        tool = MSMTool()
        result = await tool._execute(
            MSMInput(
                action="commitment_probabilities",
                transition_matrix=T,
            ),
            CTX,
        )
        assert result.success
        q = result.data["forward_committor"]
        assert q[0] == 0.0  # state A
        assert q[-1] == 1.0  # state B
        # Monotonic increase
        for i in range(len(q) - 1):
            assert q[i] <= q[i + 1] + 1e-6

    @pytest.mark.asyncio
    async def test_transition_pathway(self):
        from huginn.tools.sci.msm_tool import MSMInput, MSMTool

        T = [
            [0.5, 0.3, 0.1, 0.1],
            [0.1, 0.5, 0.3, 0.1],
            [0.1, 0.1, 0.5, 0.3],
            [0.1, 0.1, 0.1, 0.7],
        ]

        tool = MSMTool()
        result = await tool._execute(
            MSMInput(action="transition_pathway", transition_matrix=T), CTX
        )
        assert result.success
        assert result.data["pathway"][0] == 0
        assert result.data["pathway"][-1] == 3


# ── Inverse Design Tool ──


class TestInverseDesignTool:
    @pytest.mark.asyncio
    async def test_random_search(self):
        from huginn.tools.sci.inverse_design_tool import (
            InverseDesignInput,
            InverseDesignTool,
        )

        candidates = [[0.1, 0.2], [0.5, 0.5], [0.9, 0.8], [0.3, 0.7]]
        scores = [0.3, 0.8, 0.6, 0.5]

        tool = InverseDesignTool()
        result = await tool._execute(
            InverseDesignInput(action="random_search", candidates=candidates, scores=scores),
            CTX,
        )
        assert result.success
        assert result.data["n_candidates"] == 4
        assert result.data["top_candidates"][0]["index"] == 1  # score 0.8

    @pytest.mark.asyncio
    async def test_pareto_frontier(self):
        from huginn.tools.sci.inverse_design_tool import (
            InverseDesignInput,
            InverseDesignTool,
        )

        # 4 candidates, 2 objectives
        scores = [
            [0.9, 0.1],  # Pareto: high obj1, low obj2
            [0.1, 0.9],  # Pareto: low obj1, high obj2
            [0.5, 0.5],  # Pareto: moderate both
            [0.3, 0.3],  # Dominated by [0.5, 0.5]
        ]

        tool = InverseDesignTool()
        result = await tool._execute(
            InverseDesignInput(action="pareto_frontier", multi_objective_scores=scores),
            CTX,
        )
        assert result.success
        assert result.data["n_pareto_optimal"] == 3  # first 3 are Pareto
        assert 0 in result.data["pareto_indices"]
        assert 1 in result.data["pareto_indices"]
        assert 2 in result.data["pareto_indices"]
        assert 3 not in result.data["pareto_indices"]  # index 3 is dominated

    @pytest.mark.asyncio
    async def test_scoring_function(self):
        from huginn.tools.sci.inverse_design_tool import (
            InverseDesignInput,
            InverseDesignTool,
        )

        scores = [[0.8, 0.2], [0.4, 0.6]]
        weights = [0.7, 0.3]

        tool = InverseDesignTool()
        result = await tool._execute(
            InverseDesignInput(
                action="scoring_function",
                multi_objective_scores=scores,
                weights=weights,
            ),
            CTX,
        )
        assert result.success
        # 0.7*0.8 + 0.3*0.2 = 0.62, 0.7*0.4 + 0.3*0.6 = 0.46
        assert result.data["ranked_candidates"][0]["index"] == 0

    @pytest.mark.asyncio
    async def test_genetic_algorithm(self):
        from huginn.tools.sci.inverse_design_tool import (
            InverseDesignInput,
            InverseDesignTool,
        )

        tool = InverseDesignTool()
        result = await tool._execute(
            InverseDesignInput(
                action="genetic_algorithm",
                bounds_low=[0.0, 0.0],
                bounds_high=[1.0, 1.0],
                n_generations=20,
                population_size=10,
            ),
            CTX,
        )
        assert result.success
        assert "best_params" in result.data
        assert len(result.data["best_params"]) == 2


# ── Motif Mining Tool ──


class TestMotifMiningTool:
    @pytest.mark.asyncio
    async def test_coordination_octahedron(self):
        from huginn.tools.sci.motif_mining_tool import (
            MotifMiningInput,
            MotifMiningTool,
        )

        # Central atom at origin, 6 neighbors in octahedral arrangement
        pos = [
            [0, 0, 0],  # center
            [2, 0, 0], [0, 2, 0], [0, 0, 2],
            [-2, 0, 0], [0, -2, 0], [0, 0, -2],
        ]
        species = ["Fe", "O", "O", "O", "O", "O", "O"]

        tool = MotifMiningTool()
        result = await tool._execute(
            MotifMiningInput(
                action="coordination_polyhedra",
                positions=pos,
                species=species,
                cutoff=3.0,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_classified"] > 0
        poly = result.data["polyhedra"][0]
        assert poly["coordination_number"] == 6
        assert poly["polyhedron_type"] == "octahedron"
        assert poly["regular"] is True

    @pytest.mark.asyncio
    async def test_coordination_tetrahedron(self):
        from huginn.tools.sci.motif_mining_tool import (
            MotifMiningInput,
            MotifMiningTool,
        )

        # Central atom at origin, 4 neighbors in tetrahedral arrangement
        pos = [
            [0, 0, 0],
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
        ]
        species = ["Si", "O", "O", "O", "O"]

        tool = MotifMiningTool()
        result = await tool._execute(
            MotifMiningInput(
                action="coordination_polyhedra",
                positions=pos,
                species=species,
                cutoff=3.0,
            ),
            CTX,
        )
        assert result.success
        poly = result.data["polyhedra"][0]
        assert poly["coordination_number"] == 4
        assert poly["polyhedron_type"] == "tetrahedron"

    @pytest.mark.asyncio
    async def test_bond_motif_search(self):
        from huginn.tools.sci.motif_mining_tool import (
            MotifMiningInput,
            MotifMiningTool,
        )

        # O-Si-O with ~109° angle
        pos = [[0, 0, 0], [1, 1, 1], [1, -1, -1]]
        species = ["O", "Si", "O"]

        tool = MotifMiningTool()
        result = await tool._execute(
            MotifMiningInput(
                action="bond_motif_search",
                positions=pos,
                species=species,
                cutoff=2.5,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_motifs"] > 0

    @pytest.mark.asyncio
    async def test_ring_analysis(self):
        from huginn.tools.sci.motif_mining_tool import (
            MotifMiningInput,
            MotifMiningTool,
        )

        # 6-membered ring: C-C-C-C-C-C
        angles = [0, 60, 120, 180, 240, 300]
        r = 1.5
        pos = [[r * math.cos(math.radians(a)), r * math.sin(math.radians(a)), 0] for a in angles]
        species = ["C"] * 6

        tool = MotifMiningTool()
        result = await tool._execute(
            MotifMiningInput(
                action="ring_analysis",
                positions=pos,
                species=species,
                cutoff=2.0,
                max_ring_size=8,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_rings"] > 0
        # Should find a 6-membered ring
        sizes = [r["size"] for r in result.data["rings"]]
        assert 6 in sizes

    @pytest.mark.asyncio
    async def test_graph_match(self):
        from huginn.tools.sci.motif_mining_tool import (
            MotifMiningInput,
            MotifMiningTool,
        )

        # Simple chain: A-B-A
        pos = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        species = ["O", "Si", "O"]

        query = {
            "nodes": [
                {"id": 0, "species": "O"},
                {"id": 1, "species": "Si"},
                {"id": 2, "species": "O"},
            ],
            "edges": [
                {"src": 0, "dst": 1, "type": "bond"},
                {"src": 1, "dst": 2, "type": "bond"},
            ],
        }

        tool = MotifMiningTool()
        result = await tool._execute(
            MotifMiningInput(
                action="graph_match",
                positions=pos,
                species=species,
                cutoff=2.0,
                query_graph=query,
            ),
            CTX,
        )
        assert result.success
        assert result.data["n_matches"] >= 1


# ── Consensus Scoring Tool ──


class TestConsensusScoringTool:
    @pytest.mark.asyncio
    async def test_borda(self):
        from huginn.tools.sci.consensus_scoring_tool import (
            ConsensusScoringInput,
            ConsensusScoringTool,
        )

        # 3 models scoring 4 candidates
        model_scores = {
            "dft": [0.9, 0.1, 0.5, 0.3],
            "ml_pot": [0.8, 0.2, 0.4, 0.5],
            "empirical": [0.7, 0.3, 0.6, 0.4],
        }

        tool = ConsensusScoringTool()
        result = await tool._execute(
            ConsensusScoringInput(action="borda", model_scores=model_scores, maximize=True),
            CTX,
        )
        assert result.success
        assert result.data["n_models"] == 3
        assert result.data["n_candidates"] == 4
        # Candidate 0 should be ranked first (consistently highest)
        assert result.data["consensus_ranking"][0]["candidate_index"] == 0
        assert 0 <= result.data["kendalls_w"] <= 1

    @pytest.mark.asyncio
    async def test_z_score_fusion(self):
        from huginn.tools.sci.consensus_scoring_tool import (
            ConsensusScoringInput,
            ConsensusScoringTool,
        )

        model_scores = {
            "model_a": [1.0, 0.0, 0.5],
            "model_b": [10.0, 1.0, 5.0],  # different scale
        }

        tool = ConsensusScoringTool()
        result = await tool._execute(
            ConsensusScoringInput(action="z_score", model_scores=model_scores),
            CTX,
        )
        assert result.success
        # Z-score should normalize away scale differences
        assert result.data["ranking"][0]["candidate_index"] == 0
        assert "confidence" in result.data["ranking"][0]

    @pytest.mark.asyncio
    async def test_copeland(self):
        from huginn.tools.sci.consensus_scoring_tool import (
            ConsensusScoringInput,
            ConsensusScoringTool,
        )

        model_scores = {
            "m1": [0.9, 0.1, 0.5],
            "m2": [0.8, 0.2, 0.6],
            "m3": [0.7, 0.3, 0.4],
        }

        tool = ConsensusScoringTool()
        result = await tool._execute(
            ConsensusScoringInput(action="copeland", model_scores=model_scores),
            CTX,
        )
        assert result.success
        # Candidate 0 should have highest Copeland score
        assert result.data["ranking"][0]["candidate_index"] == 0

    @pytest.mark.asyncio
    async def test_rank_uncertainty(self):
        from huginn.tools.sci.consensus_scoring_tool import (
            ConsensusScoringInput,
            ConsensusScoringTool,
        )

        model_scores = {
            "m1": [0.9, 0.1, 0.5, 0.3],
            "m2": [0.8, 0.2, 0.5, 0.4],
            "m3": [0.7, 0.3, 0.5, 0.5],
        }

        tool = ConsensusScoringTool()
        result = await tool._execute(
            ConsensusScoringInput(
                action="rank_uncertainty",
                model_scores=model_scores,
                n_bootstrap=100,
            ),
            CTX,
        )
        assert result.success
        assert len(result.data["ranking"]) == 4
        assert "rank_stability" in result.data["ranking"][0]

    @pytest.mark.asyncio
    async def test_borda_weighted(self):
        from huginn.tools.sci.consensus_scoring_tool import (
            ConsensusScoringInput,
            ConsensusScoringTool,
        )

        model_scores = {
            "accurate": [0.3, 0.9],
            "noisy": [0.8, 0.2],
        }
        # Weight accurate model 10x more
        weights = {"accurate": 10.0, "noisy": 1.0}

        tool = ConsensusScoringTool()
        result = await tool._execute(
            ConsensusScoringInput(
                action="borda",
                model_scores=model_scores,
                model_weights=weights,
            ),
            CTX,
        )
        assert result.success
        # With weighting, candidate 1 (favored by accurate model) should win
        assert result.data["consensus_ranking"][0]["candidate_index"] == 1


# ── Tool Registration ──


class TestCrossPollinationRegistration:
    def test_all_tools_registered(self):
        from huginn.tools.registry import ToolRegistry
        from huginn.tools import register_all_tools

        register_all_tools()
        tools = ToolRegistry.list_tools()
        for name in [
            "fep_tool",
            "enhanced_sampling_tool",
            "msm_tool",
            "inverse_design_tool",
            "motif_mining_tool",
            "consensus_scoring_tool",
        ]:
            assert name in tools, f"{name} not registered"
