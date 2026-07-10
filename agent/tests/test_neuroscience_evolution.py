"""Tests for neuroscience-inspired SkillEvolutionLayer extensions.

Covers three changes motivated by ANCCR + "Dopamine takes a hit":
  1. Time-weighted Bayesian updates (ANCCR's inter-reward interval effect)
  2. Multi-dimensional feedback (duration + info_gain tracking)
  3. Forgetting factor + UCB exploration (anti path lock-in)
"""

from __future__ import annotations

import json
import time

from huginn.skills.evolution import SkillEvolutionLayer, ToolBelief


# ── 1. Time-weighted updates (ANCCR) ──────────────────────────────


class TestTimeWeightedUpdate:
    def test_rapid_succession_weight_is_one(self):
        """Calls within _IRI_BASELINE (10s) get weight=1.0."""
        b = ToolBelief("t", "k", "v")
        t0 = 1000.0
        b.update(True, timestamp=t0)
        b.update(True, timestamp=t0 + 1.0)  # 1s gap < 10s baseline
        # weighted_success should be exactly 2.0 (both weight=1.0)
        assert b.weighted_success == 2.0
        assert b.successes == 2

    def test_long_gap_increases_weight(self):
        """Observations after a long gap get weight > 1.0."""
        b = ToolBelief("t", "k", "v")
        t0 = 1000.0
        b.update(True, timestamp=t0)
        # 600s gap — ANCCR says this should learn faster
        b.update(True, timestamp=t0 + 600.0)
        assert b.weighted_success > 2.0  # second obs had weight > 1.0

    def test_weight_caps_at_three(self):
        """Even very long gaps cap at weight=3.0 (1.0 + _IRI_CAP=2.0)."""
        b = ToolBelief("t", "k", "v")
        t0 = 1000.0
        b.update(True, timestamp=t0)
        b.update(True, timestamp=t0 + 999999.0)
        # max weight = 1.0 + 2.0 = 3.0
        assert b.weighted_success <= 4.0  # 1.0 (first) + 3.0 (second, capped)

    def test_posterior_uses_weighted_values(self):
        """Posterior mean reflects time-weighted updates, not raw counts."""
        b = ToolBelief("t", "k", "v")
        t0 = 1000.0
        b.update(True, timestamp=t0)
        b.update(False, timestamp=t0 + 600.0)  # long gap → failure weighted more
        # weighted: 1 success (w=1.0) + 1 failure (w>1.0)
        # posterior should be less than 0.5 (failure weighted more heavily)
        assert b.posterior_mean < 0.5
        # but raw counts show 1 success, 1 failure
        assert b.successes == 1
        assert b.failures == 1

    def test_backward_compat_update_without_timestamp(self):
        """update(True) still works — uses time.time() internally."""
        b = ToolBelief("t", "k", "v")
        b.update(True)
        assert b.successes == 1
        assert b.weighted_success == 1.0  # first obs always weight=1.0
        assert b.posterior_mean == 2 / 3  # Beta(2, 1)


# ── 2. Multi-dimensional feedback ────────────────────────────────


class TestMultiDimensionalFeedback:
    def test_duration_tracked_as_ema(self):
        b = ToolBelief("t", "k", "v")
        b.update(True, timestamp=1000.0, duration=10.0, info_gain=5)
        assert b.avg_duration == 10.0  # first obs
        assert b.avg_info_gain == 5.0

    def test_ema_converges_to_recent(self):
        b = ToolBelief("t", "k", "v")
        for i in range(20):
            b.update(True, timestamp=1000.0 + i, duration=100.0, info_gain=10)
        # after 20 obs, EMA should be close to 100.0
        assert abs(b.avg_duration - 100.0) < 1.0
        assert abs(b.avg_info_gain - 10.0) < 0.5

    def test_record_tool_call_passes_metrics(self):
        layer = SkillEvolutionLayer()
        layer.record_tool_call(
            "vasp_tool",
            {"encut": 520},
            True,
            duration=5.5,
            info_gain=8,
        )
        b = layer.get_belief("vasp_tool", "encut", "520")
        assert b is not None
        assert b.avg_duration == 5.5
        assert b.avg_info_gain == 8.0

    def test_skill_context_shows_metrics(self, tmp_path):
        traj = tmp_path / "run.json"
        traj.write_text(json.dumps({
            "tool_calls": [
                {"tool": "vasp_tool", "args": {"encut": 520}, "success": True,
                 "duration": 3.2, "info_gain": 7},
                {"tool": "vasp_tool", "args": {"encut": 520}, "success": True,
                 "duration": 2.8, "info_gain": 6},
            ],
        }), encoding="utf-8")
        layer = SkillEvolutionLayer()
        layer.update_from_trajectory(traj)
        ctx = layer.get_skill_context()
        assert "vasp_tool" in ctx
        assert "[" in ctx  # duration shown as [X.Xs, Yk]


# ── 3. Forgetting factor + UCB exploration ───────────────────────


class TestDecayAndExploration:
    def test_decay_pulls_toward_prior(self):
        b = ToolBelief("t", "k", "v")
        for _ in range(10):
            b.update(True, timestamp=1000.0)
        original = b.posterior_mean
        assert original > 0.9
        b.decay(0.5)
        # after halving, posterior should move toward 0.5
        assert b.posterior_mean < original

    def test_decay_all(self):
        layer = SkillEvolutionLayer()
        layer.record_tool_call("vasp_tool", {"encut": 520}, True, duration=1.0, info_gain=1)
        layer.record_tool_call("vasp_tool", {"encut": 520}, True, duration=1.0, info_gain=1)
        b = layer.get_belief("vasp_tool", "encut", "520")
        original = b.posterior_mean
        layer.decay_all(0.5)
        b2 = layer.get_belief("vasp_tool", "encut", "520")
        assert b2.posterior_mean < original

    def test_decay_preserves_raw_counts(self):
        b = ToolBelief("t", "k", "v")
        b.update(True, timestamp=1000.0)
        b.update(True, timestamp=1001.0)
        b.update(False, timestamp=1002.0)
        b.decay(0.5)
        # raw counts unchanged
        assert b.successes == 2
        assert b.failures == 1
        assert b.total == 3

    def test_ucb_exploration_boosts_underexplored(self):
        """With enough exploration, under-sampled value overtakes well-explored one."""
        layer = SkillEvolutionLayer()
        # well-explored: 10 successes, posterior ≈ 0.917
        for _ in range(10):
            layer.record_tool_call("t", {"encut": 400}, True, timestamp=1000.0)
        # under-explored: 1 success, posterior ≈ 0.667
        layer.record_tool_call("t", {"encut": 500}, True, timestamp=1001.0)

        # with exploration=0, sort purely by posterior → 400 wins (0.917 > 0.667)
        recs_no_explore = layer.recommend_params("t", "encut", exploration=0.0)
        assert recs_no_explore[0][0] == "400"

        # with high exploration, under-sampled 500 gets a big UCB bonus and overtakes
        recs_explore = layer.recommend_params("t", "encut", exploration=0.5)
        assert recs_explore[0][0] == "500"
        assert len(recs_explore) == 2

    def test_recommend_params_empty_when_no_data(self):
        layer = SkillEvolutionLayer()
        assert layer.recommend_params("nonexistent", "key") == []


# ── 4. Persistence backward compat ───────────────────────────────


class TestPersistenceCompat:
    def test_load_v1_format(self, tmp_path):
        """Old v1 files (no weighted_success/avg_duration) load correctly."""
        old_data = {
            "version": "1.0",
            "saved_at": 1234567890.0,
            "beliefs": [
                {
                    "tool_name": "vasp_tool",
                    "param_key": "encut",
                    "param_value": "520",
                    "successes": 5,
                    "failures": 1,
                    "last_updated": 1234567890.0,
                },
            ],
        }
        path = tmp_path / "beliefs.json"
        path.write_text(json.dumps(old_data), encoding="utf-8")

        layer = SkillEvolutionLayer(persist_path=path)
        b = layer.get_belief("vasp_tool", "encut", "520")
        assert b is not None
        assert b.successes == 5
        assert b.failures == 1
        # weighted values should match raw counts (fallback)
        assert b.weighted_success == 5.0
        assert b.weighted_failure == 1.0
        # posterior should match old formula
        assert abs(b.posterior_mean - 6 / 8) < 1e-9

    def test_save_load_roundtrip_preserves_new_fields(self, tmp_path):
        path = tmp_path / "beliefs.json"
        layer = SkillEvolutionLayer(persist_path=path)
        layer.record_tool_call(
            "vasp_tool", {"encut": 520}, True,
            duration=3.5, info_gain=7,
        )
        layer._save()

        layer2 = SkillEvolutionLayer(persist_path=path)
        b = layer2.get_belief("vasp_tool", "encut", "520")
        assert b is not None
        assert b.avg_duration == 3.5
        assert b.avg_info_gain == 7.0
        assert b.n_obs == 1


# ── 5. Existing API backward compat ──────────────────────────────


class TestBackwardCompat:
    def test_posterior_mean_one_success(self):
        b = ToolBelief("t", "k", "v")
        b.update(True)
        assert abs(b.posterior_mean - 2 / 3) < 1e-9

    def test_posterior_mean_3s_2f(self):
        b = ToolBelief("t", "k", "v")
        for _ in range(3):
            b.update(True)
        for _ in range(2):
            b.update(False)
        # all rapid (weight=1.0), so weighted = raw
        assert abs(b.posterior_mean - 4 / 7) < 1e-9

    def test_confidence_unchanged(self):
        b = ToolBelief("t", "k", "v")
        for _ in range(5):
            b.update(True)
        assert abs(b.confidence - 0.5) < 1e-9
        for _ in range(5):
            b.update(True)
        assert b.confidence == 1.0

    def test_record_tool_call_without_optional_params(self):
        """Old-style record_tool_call(tool, args, success) still works."""
        layer = SkillEvolutionLayer()
        layer.record_tool_call("vasp_tool", {"encut": 400}, True)
        b = layer.get_belief("vasp_tool", "encut", "400")
        assert b is not None
        assert b.successes == 1
        assert b.weighted_success == 1.0
