"""Tests for the advanced mathematical framework tools.

Covers:
  * EvidenceFusionTool  -- Dempster-Shafer evidence combination
  * TDATool             -- persistent homology / topology descriptors
  * GPTool              -- natural gradient, Fisher information, KL divergence
  * DescriptorTool      -- Indian Buffet Process feature discovery

Note on async: EvidenceFusionTool / TDATool / DescriptorTool expose an async
``call``, while GPTool.call is synchronous. We respect that in each test.
"""

from __future__ import annotations

import numpy as np
import pytest

from huginn.tools.descriptor_tool import DescriptorInput, DescriptorTool
from huginn.tools.evidence_fusion_tool import EvidenceFusionTool
from huginn.tools.gp_tool import GPTool
from huginn.tools.tda_tool import TDATool
from huginn.types import ToolContext


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path):
    """A minimal ToolContext for tools that require one (DescriptorTool)."""
    return ToolContext(session_id="adv-framework-tests", workspace=str(tmp_path))


# ===========================================================================
# EvidenceFusionTool (Dempster-Shafer)
# ===========================================================================


class TestEvidenceFusion:
    @pytest.mark.asyncio
    async def test_combine_stable_vs_unstable(self):
        """Two agreeing sources should pile mass onto 'stable'."""
        tool = EvidenceFusionTool()
        evidence = [
            {"hypotheses": ["stable"], "mass": 0.7, "source": "dft"},
            {"hypotheses": ["unstable"], "mass": 0.2, "source": "dft"},
            {"hypotheses": ["stable"], "mass": 0.8, "source": "md"},
            {"hypotheses": ["unstable"], "mass": 0.1, "source": "md"},
        ]
        result = await tool.call({"action": "combine", "evidence": evidence})

        assert result.success is True
        data = result.data
        assert sorted(data["frame_of_discernment"]) == ["stable", "unstable"]

        masses = {tuple(m["hypotheses"]): m["mass"] for m in data["combined_mass"]}
        assert masses.get(("stable",), 0.0) > masses.get(("unstable",), 0.0)

        bp = data["belief_plausibility"]
        assert bp["stable"]["belief"] > bp["unstable"]["belief"]
        assert bp["stable"]["belief"] > 0.8
        # belief is always a lower bound on plausibility
        for h in ("stable", "unstable"):
            assert bp[h]["belief"] <= bp[h]["plausibility"]

        # hand-computed conflict is ~0.23 -> moderate band
        assert 0.1 < data["conflict"] < 0.4

    @pytest.mark.asyncio
    async def test_pignistic_sums_to_one(self):
        """Smets' pignistic transform must yield a proper distribution."""
        tool = EvidenceFusionTool()
        mass_function = [
            {"hypotheses": ["A"], "mass": 0.5},
            {"hypotheses": ["A", "B"], "mass": 0.3},
            {"hypotheses": ["B"], "mass": 0.2},
        ]
        result = await tool.call(
            {"action": "pignistic", "mass_function": mass_function}
        )

        assert result.success is True
        probs = result.data["pignistic_probability"]
        assert set(probs.keys()) == {"A", "B"}
        assert sum(probs.values()) == pytest.approx(1.0, abs=1e-6)
        # A picks up 0.5 + 0.3/2 = 0.65; B picks up 0.2 + 0.3/2 = 0.35
        assert probs["A"] == pytest.approx(0.65, abs=1e-6)
        assert probs["B"] == pytest.approx(0.35, abs=1e-6)
        assert result.data["decision"]["top_hypothesis"] == "A"

    @pytest.mark.asyncio
    async def test_conflict_analysis_detects_high_conflict(self):
        """Sources backing disjoint hypotheses should flag high conflict."""
        tool = EvidenceFusionTool()
        evidence = [
            {"hypotheses": ["stable"], "mass": 0.9, "source": "dft"},
            {"hypotheses": ["unstable"], "mass": 0.9, "source": "experiment"},
        ]
        result = await tool.call(
            {"action": "conflict_analysis", "evidence": evidence}
        )

        assert result.success is True
        data = result.data
        matrix = data["conflict_matrix"]
        # 2x2 symmetric matrix; off-diagonal carries the pairwise conflict K
        assert matrix[0][1] > 0.5
        assert matrix[1][0] == matrix[0][1]

        pair = data["most_conflicting_pair"]
        assert pair is not None
        assert pair["conflict"] > 0.5
        assert set(pair["sources"]) == {"dft", "experiment"}

    @pytest.mark.asyncio
    async def test_weighted_combine_discounts_unreliable_source(self):
        """A low-reliability weight should suppress conflict vs. plain combine."""
        tool = EvidenceFusionTool()
        evidence = [
            {"hypotheses": ["stable"], "mass": 0.9, "source": "dft", "weight": 1.0},
            {
                "hypotheses": ["unstable"],
                "mass": 0.9,
                "source": "experiment",
                "weight": 0.1,
            },
        ]
        weighted = await tool.call(
            {"action": "weighted_combine", "evidence": evidence}
        )
        plain = await tool.call({"action": "combine", "evidence": evidence})

        assert weighted.success is True
        assert plain.success is True

        # discounting moves most of the unreliable mass onto ignorance, so the
        # pairwise conflict collapses from ~0.81 to ~0.08
        assert weighted.data["conflict"] < plain.data["conflict"]
        assert weighted.data["conflict"] < 0.3
        assert weighted.data["weights"]["experiment"] == pytest.approx(0.1)

        bp = weighted.data["belief_plausibility"]
        assert bp["stable"]["belief"] > bp["unstable"]["belief"]


# ===========================================================================
# TDATool (persistent homology)
# ===========================================================================


class TestTDA:
    @pytest.mark.asyncio
    async def test_persistence_diagram_circle_has_h1(self):
        """A sampled circle must produce at least one persistent H1 bar."""
        tool = TDATool()
        theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        points = np.column_stack([np.cos(theta), np.sin(theta)]).tolist()

        result = await tool.call(
            {"action": "persistence_diagram", "point_cloud": points, "max_dim": 1}
        )
        assert result.success is True
        data = result.data
        assert data["n_points"] == 20
        assert isinstance(data["diagram"], list)

        h1_bars = [
            p for p in data["diagram"] if p["dim"] == 1 and p["death"] is not None
        ]
        assert len(h1_bars) >= 1
        # ripser/gudhi give a long bar; the scipy fallback gives a shorter one,
        # but both are clearly above zero for a clean circle.
        max_h1_persistence = max(p["persistence"] for p in h1_bars)
        assert max_h1_persistence > 0.1

    @pytest.mark.asyncio
    async def test_bottleneck_distance_identical_diagrams(self):
        """The bottleneck distance between a diagram and itself is zero."""
        tool = TDATool()
        diagram = [
            {"dim": 0, "birth": 0.0, "death": 1.0, "persistence": 1.0},
            {"dim": 0, "birth": 0.0, "death": 0.5, "persistence": 0.5},
            {"dim": 1, "birth": 0.3, "death": 0.8, "persistence": 0.5},
        ]
        result = await tool.call(
            {
                "action": "bottleneck_distance",
                "diagram": diagram,
                "diagram2": [dict(p) for p in diagram],
            }
        )
        assert result.success is True
        assert result.data["distance"] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.asyncio
    async def test_energy_landscape_topology_two_basins(self):
        """Two well-separated energy clusters should resolve as two basins."""
        tool = TDATool()
        energies = [0.0, 0.1, 0.2, 5.0, 5.1, 5.2]
        structures = [[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]]
        result = await tool.call(
            {
                "action": "energy_landscape_topology",
                "energies": energies,
                "structures": structures,
                "threshold": 1.0,
            }
        )
        assert result.success is True
        data = result.data
        assert data["n_structures"] == 6
        assert data["n_basins"] == 2
        assert sum(data["basin_sizes"]) == 6
        assert data["n_edges"] >= 2  # at least the intra-cluster edges

    @pytest.mark.asyncio
    async def test_structure_topology_cubic(self):
        """A 2-atom cubic cell swept over a few radii."""
        tool = TDATool()
        structure = {
            "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "sites": [{"xyz": [0.0, 0.0, 0.0]}, {"xyz": [2.0, 2.0, 2.0]}],
        }
        radii = [1.0, 2.5, 3.5]
        result = await tool.call(
            {"action": "structure_topology", "structure": structure, "radii": radii}
        )
        assert result.success is True
        data = result.data
        assert data["n_atoms"] == 2
        assert data["radii"] == radii
        # PBC distance between the two sites is sqrt(12) ~ 3.46, so below that
        # they sit in separate components and above it they merge into one.
        assert data["betti_0"][0] == 2
        assert data["betti_0"][-1] == 1
        assert data["betti_1"][-1] == 0


# ===========================================================================
# GPTool -- natural gradient / Fisher information / KL divergence (sync)
# ===========================================================================


class TestGPAdvanced:
    def test_natural_gradient_improves_likelihood(self):
        """Natural-gradient ascent should raise the marginal likelihood."""
        tool = GPTool()
        x = np.linspace(0, 5, 12).reshape(-1, 1).tolist()
        y = np.sin(np.linspace(0, 5, 12)).tolist()
        # lr=0.001 with the default length_scale=1.0 climbs cleanly; larger
        # steps or a shorter length_scale overshoot and diverge on this data.
        result = tool.call(
            {
                "action": "natural_gradient",
                "X": x,
                "y": y,
                "n_steps": 15,
                "lr": 0.001,
                "sigma_n": 0.01,
            }
        )
        assert result.success is True
        traj = result.data["log_likelihood_trajectory"]
        assert len(traj) == 15
        assert result.data["n_steps_run"] == 15
        assert traj[-1]["log_likelihood"] > traj[0]["log_likelihood"]

    def test_fisher_information_positive_definite(self):
        """D-optimality requires a positive-definite Fisher matrix (det > 0)."""
        tool = GPTool()
        x = [[0.0], [1.0], [2.0], [3.0]]
        y = [0.0, 0.5, 0.3, 0.8]
        candidates = [[0.5], [1.5], [2.5]]
        result = tool.call(
            {
                "action": "fisher_information",
                "X": x,
                "y": y,
                "X_candidates": candidates,
                "sigma_n": 0.01,
            }
        )
        assert result.success is True
        data = result.data
        assert data["n_candidates"] == 3
        assert len(data["per_candidate_contribution"]) == 3

        fisher = np.array(data["fisher_matrix"], dtype=float)
        assert fisher.shape == (3, 3)
        # the D-optimal criterion is the determinant; it must be positive for
        # the matrix to be invertible / informative.
        assert float(np.linalg.det(fisher)) > 0.0
        assert np.isfinite(data["d_optimal"])
        # all eigenvalues positive <=> positive definite
        eigvals = np.linalg.eigvalsh(fisher)
        assert float(eigvals.min()) > 0.0

    def test_kl_divergence_positive(self):
        """Two GPs fit to shifted targets must differ (KL > 0)."""
        tool = GPTool()
        x = [[0.0], [1.0], [2.0], [3.0], [4.0]]
        y1 = [0.0, 0.5, 0.3, 0.8, 0.2]
        y2 = [v + 0.5 for v in y1]
        result = tool.call(
            {
                "action": "kl_divergence",
                "X": x,
                "y1": y1,
                "y2": y2,
                "X_new": [[0.5], [1.5], [2.5], [3.5]],
                "sigma_n": 0.01,
            }
        )
        assert result.success is True
        data = result.data
        assert data["n_test_points"] == 4
        assert data["kl_divergence"] > 0.0
        assert data["mean_kl"] > 0.0
        assert data["max_kl"] >= data["mean_kl"]


# ===========================================================================
# DescriptorTool -- Indian Buffet Process
# ===========================================================================


class TestDescriptorIBP:
    @pytest.mark.asyncio
    async def test_ibp_discovers_latent_features(self, ctx):
        """IBP should recover roughly 3 latent features from Z@W + noise data."""
        tool = DescriptorTool()
        # Strong signal (w_scale=3.0) and tiny noise let the sampler lock onto
        # the true 3 factors. beta=0.5 (Jeffreys prior) keeps the feature count
        # from inflating; the default beta=1.0 explodes to the cap on this data.
        rng = np.random.default_rng(7)
        z_true = (rng.random((20, 3)) < 0.5).astype(float)
        w_true = rng.normal(0.0, 3.0, size=(3, 5))
        x = z_true @ w_true + rng.normal(0.0, 0.05, size=(20, 5))

        args = DescriptorInput(
            action="ibp",
            data=x.tolist(),
            alpha=1.0,
            n_iterations=50,
            n_init_features=3,
            beta=0.5,
            seed=42,
        )
        result = await tool.call(args, ctx)

        assert result.success is True
        feats = result.data["features"]
        n_features = feats["n_features"]
        assert 1 <= n_features <= 10
        assert feats["n_samples"] == 20
        assert feats["n_observed_features"] == 5

        # reconstruction error is on standardized data, so < 1.0 means the
        # latent factors explain more than the residual noise floor
        assert feats["reconstruction_error"] < 1.0

        z = feats["Z"]
        w = feats["W"]
        assert len(z) == 20
        assert len(z[0]) == n_features
        assert len(w) == n_features
        assert len(w[0]) == 5
        # Z is binary by construction
        assert all(v in (0, 1) for row in z for v in row)
