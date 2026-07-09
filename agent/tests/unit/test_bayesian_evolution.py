"""P0 tests for Bayesian skill evolution — Beta updates, trajectory parsing, filtering."""

from __future__ import annotations

import json

from huginn.skills.evolution import (
    SkillEvolutionLayer,
    ToolBelief,
    _MIN_SAMPLES,
)


# ── Beta distribution update ────────────────────────────────────────


class TestBetaUpdate:
    def test_success_increments_successes(self):
        b = ToolBelief("vasp_tool", "encut", "520")
        b.update(True)
        assert b.successes == 1
        assert b.failures == 0

    def test_failure_increments_failures(self):
        b = ToolBelief("vasp_tool", "encut", "520")
        b.update(False)
        assert b.successes == 0
        assert b.failures == 1

    def test_mixed_updates(self):
        b = ToolBelief("vasp_tool", "encut", "520")
        for ok in [True, True, False, True, False, False]:
            b.update(ok)
        assert b.successes == 3
        assert b.failures == 3
        assert b.total == 6

    def test_total_property(self):
        b = ToolBelief("vasp_tool", "encut", "520")
        b.update(True)
        b.update(True)
        b.update(False)
        assert b.total == 3

    def test_last_updated_set(self):
        b = ToolBelief("vasp_tool", "encut", "520")
        assert b.last_updated == 0.0
        b.update(True)
        assert b.last_updated > 0


# ── posterior_mean ──────────────────────────────────────────────────


class TestPosteriorMean:
    def test_uniform_prior_at_start(self):
        """Beta(1,1) prior → mean 0.5 before any observations."""
        b = ToolBelief("t", "k", "v")
        assert b.posterior_mean == 0.5

    def test_after_one_success(self):
        b = ToolBelief("t", "k", "v")
        b.update(True)
        # Beta(2,1) → 2/3
        assert abs(b.posterior_mean - 2 / 3) < 1e-9

    def test_after_one_failure(self):
        b = ToolBelief("t", "k", "v")
        b.update(False)
        # Beta(1,2) → 1/3
        assert abs(b.posterior_mean - 1 / 3) < 1e-9

    def test_after_3_success_2_failure(self):
        b = ToolBelief("t", "k", "v")
        for _ in range(3):
            b.update(True)
        for _ in range(2):
            b.update(False)
        # Beta(4,3) → 4/7
        assert abs(b.posterior_mean - 4 / 7) < 1e-9

    def test_all_successes_approaches_one(self):
        b = ToolBelief("t", "k", "v")
        for _ in range(20):
            b.update(True)
        assert b.posterior_mean > 0.95

    def test_all_failures_approaches_zero(self):
        b = ToolBelief("t", "k", "v")
        for _ in range(20):
            b.update(False)
        assert b.posterior_mean < 0.05

    def test_confidence_saturates_at_ten(self):
        b = ToolBelief("t", "k", "v")
        for _ in range(5):
            b.update(True)
        assert abs(b.confidence - 0.5) < 1e-9
        for _ in range(5):
            b.update(True)
        assert b.confidence == 1.0
        for _ in range(10):
            b.update(True)
        assert b.confidence == 1.0  # capped


# ── trajectory parsing and belief injection ────────────────────────


class TestTrajectoryParsing:
    def test_update_from_trajectory_records_beliefs(self, tmp_path):
        traj = tmp_path / "run.json"
        traj.write_text(json.dumps({
            "tool_calls": [
                {"tool": "vasp_tool", "args": {"action": "relax", "encut": 520}, "success": True},
                {"tool": "vasp_tool", "args": {"action": "relax", "encut": 520}, "success": False},
                {"tool": "vasp_tool", "args": {"action": "scf", "encut": 520}, "success": True},
            ],
        }), encoding="utf-8")

        layer = SkillEvolutionLayer()
        count = layer.update_from_trajectory(traj)

        assert count == 3
        # action=relax: 1 success + 1 failure
        b = layer.get_belief("vasp_tool", "action", "relax")
        assert b is not None
        assert b.successes == 1
        assert b.failures == 1
        # action=scf: 1 success
        b = layer.get_belief("vasp_tool", "action", "scf")
        assert b is not None
        assert b.successes == 1
        assert b.failures == 0
        # encut=520: 2 successes + 1 failure across both calls
        b = layer.get_belief("vasp_tool", "encut", "520")
        assert b is not None
        assert b.successes == 2
        assert b.failures == 1

    def test_update_from_trajectory_skips_empty_tool(self, tmp_path):
        traj = tmp_path / "run.json"
        traj.write_text(json.dumps({
            "tool_calls": [
                {"tool": "", "args": {"encut": 400}, "success": True},
                {"tool": "vasp_tool", "args": {"encut": 400}, "success": True},
            ],
        }), encoding="utf-8")

        layer = SkillEvolutionLayer()
        count = layer.update_from_trajectory(traj)
        assert count == 1  # only the non-empty tool counted

    def test_update_from_trajectory_defaults_success_true(self, tmp_path):
        traj = tmp_path / "run.json"
        traj.write_text(json.dumps({
            "tool_calls": [
                {"tool": "vasp_tool", "args": {"encut": 400}},
            ],
        }), encoding="utf-8")

        layer = SkillEvolutionLayer()
        layer.update_from_trajectory(traj)
        b = layer.get_belief("vasp_tool", "encut", "400")
        assert b is not None
        assert b.successes == 1

    def test_skill_context_injects_beliefs(self, tmp_path):
        traj = tmp_path / "run.json"
        traj.write_text(json.dumps({
            "tool_calls": [
                {"tool": "vasp_tool", "args": {"encut": 520}, "success": True},
                {"tool": "vasp_tool", "args": {"encut": 520}, "success": True},
            ],
        }), encoding="utf-8")

        layer = SkillEvolutionLayer()
        layer.update_from_trajectory(traj)
        ctx = layer.get_skill_context()
        assert "Skill Evolution" in ctx
        assert "vasp_tool" in ctx
        assert "encut=520" in ctx

    def test_skill_context_empty_when_no_beliefs(self):
        layer = SkillEvolutionLayer()
        assert layer.get_skill_context() == ""

    def test_update_from_directory_scans_all(self, tmp_path):
        for i in range(3):
            (tmp_path / f"run_{i}.json").write_text(json.dumps({
                "tool_calls": [
                    {"tool": "lammps_tool", "args": {"timestep": 1.0}, "success": True},
                ],
            }), encoding="utf-8")

        layer = SkillEvolutionLayer()
        total = layer.update_from_directory(tmp_path)
        assert total == 3
        b = layer.get_belief("lammps_tool", "timestep", "1.0")
        assert b is not None
        assert b.successes == 3


# ── _MIN_SAMPLES filtering ──────────────────────────────────────────


class TestMinSamplesFiltering:
    def test_below_threshold_excluded_from_context(self):
        layer = SkillEvolutionLayer()
        # One call → total=1, below _MIN_SAMPLES (2)
        layer.record_tool_call("vasp_tool", {"encut": 400}, True)
        ctx = layer.get_skill_context()
        assert "encut=400" not in ctx

    def test_at_threshold_included_in_context(self):
        layer = SkillEvolutionLayer()
        for _ in range(_MIN_SAMPLES):
            layer.record_tool_call("vasp_tool", {"encut": 520}, True)
        ctx = layer.get_skill_context()
        assert "encut=520" in ctx

    def test_mixed_above_below_threshold(self):
        layer = SkillEvolutionLayer()
        # Below threshold
        layer.record_tool_call("vasp_tool", {"encut": 400}, True)
        # At threshold
        for _ in range(_MIN_SAMPLES):
            layer.record_tool_call("vasp_tool", {"encut": 520}, True)

        ctx = layer.get_skill_context()
        assert "encut=400" not in ctx
        assert "encut=520" in ctx

    def test_filter_only_applies_to_context_not_belief_storage(self):
        """Belief is stored even below _MIN_SAMPLES, just not surfaced in context."""
        layer = SkillEvolutionLayer()
        layer.record_tool_call("vasp_tool", {"encut": 400}, True)
        b = layer.get_belief("vasp_tool", "encut", "400")
        assert b is not None
        assert b.total == 1
        assert b.total < _MIN_SAMPLES
