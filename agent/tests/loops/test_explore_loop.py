"""P0 integration tests for the exploration loop (exploration/orchestrator.py).

Drives the real ExplorationOrchestrator.explore() with a stub branch
executor so we can control objective values and verify Pareto pruning,
best-branch selection, and convergence behaviour.
"""

from __future__ import annotations

import pytest

from huginn.exploration.core import Branch, BranchStatus
from huginn.exploration.orchestrator import ExplorationOrchestrator
from huginn.exploration.strategies import (
    Action,
    ExplorationStrategy,
    ParetoPruningStrategy,
)


# ── helpers ────────────────────────────────────────────────────────


def _executor(scores: dict[str, dict[str, float]]):
    """Build an async branch executor that returns fixed objective values.

    Looks up the branch name in *scores* and returns those objectives.
    Unknown branches get a default low score.
    """
    async def _exec(branch: Branch) -> dict:
        objs = scores.get(branch.name, {"score": 0.01})
        return {"success": True, "results": {"computed": True}, "objectives": objs}
    return _exec


def _make_branches(names: list[str]) -> list[dict]:
    return [{"name": n, "hypothesis": f"Hypothesis for {n}"} for n in names]


# ── 1. single objective: 3 branches → converges to best ───────────


class TestExploreSingleObjective:
    @pytest.mark.asyncio
    async def test_best_branch_selected(self):
        """Three branches with different scores; best one wins."""
        scores = {
            "low": {"score": 0.3},
            "mid": {"score": 0.6},
            "high": {"score": 0.9},
        }
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=5),
            branch_executor=_executor(scores),
            max_parallel=3,
        )
        result = await orch.explore(
            objective="maximize score",
            initial_branches=_make_branches(["low", "mid", "high"]),
            objectives_config={"score": "maximize"},
            max_iterations=5,
        )

        assert result.best_branch is not None
        assert result.best_branch["name"] == "high"
        assert result.best_branch["objectives"]["score"] == 0.9
        # All 3 branches were created and executed; some may be pruned by
        # the Pareto strategy if dominated, so check total throughput.
        assert result.n_branches_explored + result.n_branches_pruned == 3

    @pytest.mark.asyncio
    async def test_pareto_front_has_best(self):
        """Pareto front contains the non-dominated branch."""
        scores = {"a": {"score": 1.0}, "b": {"score": 0.5}}
        orch = ExplorationOrchestrator(
            branch_executor=_executor(scores),
            max_parallel=2,
        )
        result = await orch.explore(
            objective="maximize score",
            initial_branches=_make_branches(["a", "b"]),
            objectives_config={"score": "maximize"},
            max_iterations=3,
        )
        # 'a' dominates 'b' on score, so 'a' should be in the Pareto front
        front_names = {b["name"] for b in result.pareto_front}
        assert "a" in front_names


# ── 2. Pareto pruning: dominated branches are pruned ──────────────


class TestExploreParetoPruning:
    @pytest.mark.asyncio
    async def test_dominated_branch_pruned(self):
        """Branch dominated in all objectives gets pruned."""
        scores = {
            "best": {"score": 10.0, "speed": 10.0},
            "mid": {"score": 5.0, "speed": 5.0},
            "worst": {"score": 1.0, "speed": 1.0},
        }
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=5, min_objective_improvement=0.01),
            branch_executor=_executor(scores),
            max_parallel=3,
        )
        result = await orch.explore(
            objective="maximize score and speed",
            initial_branches=_make_branches(["best", "mid", "worst"]),
            objectives_config={"score": "maximize", "speed": "maximize"},
            max_iterations=5,
        )

        # 'best' dominates both 'mid' and 'worst' with a big margin → pruned
        assert result.n_branches_pruned >= 2
        assert result.best_branch["name"] == "best"

        # Verify pruned branches are actually marked as pruned
        space = result.space
        for name in ("mid", "worst"):
            pruned = [
                b for b in space.branches.values()
                if b.name == name and b.status == BranchStatus.PRUNED
            ]
            assert len(pruned) == 1, f"{name} should be pruned"


# ── 3. convergence: max_iterations reached → stops ─────────────────


class _AlwaysExpand(ExplorationStrategy):
    """Strategy that always proposes one new branch — never converges."""

    def name(self) -> str:
        return "always_expand"

    def evaluate(self, space) -> list[Action]:
        return [
            Action(
                action_type="expand",
                new_branches=[{"name": "new", "hypothesis": "keep exploring"}],
                reason="need more data",
            )
        ]


class TestExploreConvergence:
    @pytest.mark.asyncio
    async def test_max_iterations_stops(self):
        """Loop hits max_iterations and stops with the right reason."""
        orch = ExplorationOrchestrator(
            strategy=_AlwaysExpand(),
            branch_executor=_executor({"baseline": {"score": 0.5}}),
            max_parallel=2,
        )
        result = await orch.explore(
            objective="explore forever",
            initial_branches=_make_branches(["baseline"]),
            objectives_config={"score": "maximize"},
            max_iterations=3,
        )

        assert result.convergence_reason == "max_iterations reached"
        # Each iteration creates 1 new branch via expand, so we should have
        # the original + at least a few more
        assert result.n_branches_explored >= 1

    @pytest.mark.asyncio
    async def test_all_resolved_stops(self):
        """When all branches resolve and no new actions, loop stops early."""
        scores = {"solo": {"score": 0.8}}
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=5),
            branch_executor=_executor(scores),
            max_parallel=1,
        )
        result = await orch.explore(
            objective="single branch",
            initial_branches=_make_branches(["solo"]),
            objectives_config={"score": "maximize"},
            max_iterations=10,
        )

        # Single branch completes, no new actions proposed → early stop
        assert "resolved" in result.convergence_reason.lower()
        assert result.best_branch["name"] == "solo"
