"""Tests for exploration engine."""

import pytest

from huginn.exploration.core import Branch, BranchStatus, Decision, ExplorationSpace
from huginn.exploration.lifecycle import BranchLifecycleManager
from huginn.exploration.orchestrator import ExplorationOrchestrator, ExplorationResult
from huginn.exploration.strategies import (
    AdaptiveGridStrategy,
    BayesianExplorationStrategy,
    ParetoPruningStrategy,
)


class TestExplorationSpace:
    def test_add_branch_and_pareto(self):
        space = ExplorationSpace(
            id="test",
            name="Test",
            objective="Test objective",
            objectives_config={"energy": "minimize", "stability": "maximize"},
        )
        b1 = Branch(
            id="b1",
            name="A",
            hypothesis="h1",
            objectives={"energy": 10, "stability": 5},
        )
        b2 = Branch(
            id="b2", name="B", hypothesis="h2", objectives={"energy": 8, "stability": 6}
        )
        b3 = Branch(
            id="b3",
            name="C",
            hypothesis="h3",
            objectives={"energy": 12, "stability": 3},
        )
        for b in [b1, b2, b3]:
            b.status = BranchStatus.COMPLETED
            space.add_branch(b)

        front = space.update_pareto_front()
        assert "b2" in front  # b2 dominates b1 and b3
        assert "b1" not in front
        assert "b3" not in front

    def test_why_pruned(self):
        space = ExplorationSpace(id="test", name="Test", objective="test")
        space.mark_pruned("missing", "test reason")
        assert "not found" in space.why_pruned("missing")

    def test_export_knowledge_graph(self):
        space = ExplorationSpace(id="test", name="Test", objective="test")
        json_str = space.export_knowledge_graph("json")
        assert "nodes" in json_str


class TestStrategies:
    def test_pareto_strategy_evaluate(self):
        space = ExplorationSpace(
            id="test",
            name="Test",
            objective="test",
            objectives_config={"score": "maximize"},
        )
        b1 = Branch(id="b1", name="A", hypothesis="h1", objectives={"score": 1.0})
        b2 = Branch(id="b2", name="B", hypothesis="h2", objectives={"score": 2.0})
        for b in [b1, b2]:
            b.status = BranchStatus.COMPLETED
            space.add_branch(b)

        strategy = ParetoPruningStrategy(max_active=5)
        actions = strategy.evaluate(space)
        # b1 should be pruned (dominated by b2)
        prune_actions = [a for a in actions if a.action_type == "prune"]
        assert any(a.target_branch == "b1" for a in prune_actions)

    def test_bayesian_strategy_insufficient_samples(self):
        space = ExplorationSpace(id="test", name="Test", objective="test")
        strategy = BayesianExplorationStrategy(n_initial=5)
        actions = strategy.evaluate(space)
        assert any(a.action_type == "expand" for a in actions)

    def test_adaptive_grid_strategy(self):
        space = ExplorationSpace(id="test", name="Test", objective="test")
        strategy = AdaptiveGridStrategy()
        actions = strategy.evaluate(space)
        assert any(a.action_type == "expand" for a in actions)


class TestLifecycle:
    async def _mock_execute(self, branch: Branch) -> dict:
        return {
            "success": True,
            "results": {"test": True},
            "objectives": {"score": 0.8},
        }

    @pytest.mark.asyncio
    async def test_create_and_execute_branch(self):
        space = ExplorationSpace(id="test", name="Test", objective="test")
        mgr = BranchLifecycleManager(execute_fn=self._mock_execute)
        branch = await mgr.create_branch(space, "test_branch", "test hypothesis")
        assert branch.id in space.branches
        assert branch.status == BranchStatus.PENDING

        result = await mgr.execute_branch(space, branch.id)
        assert result["success"]
        assert space.branches[branch.id].status == BranchStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_prune_cascade(self):
        space = ExplorationSpace(id="test", name="Test", objective="test")
        mgr = BranchLifecycleManager()
        parent = await mgr.create_branch(space, "parent", "parent hyp")
        child = await mgr.create_branch(space, "child", "child hyp", parent=parent.id)

        await mgr.prune_branch(space, parent.id, "test prune", cascade=True)
        assert space.branches[parent.id].status == BranchStatus.PRUNED
        assert space.branches[child.id].status == BranchStatus.PRUNED

    @pytest.mark.asyncio
    async def test_backtrack(self):
        space = ExplorationSpace(id="test", name="Test", objective="test")
        mgr = BranchLifecycleManager()
        branch = await mgr.create_branch(space, "orig", "original")
        branch.decisions.append(
            Decision(
                id="d1",
                description="pick x",
                decision_type="categorical",
                chosen_option="A",
                available_options=["A", "B"],
            )
        )

        new_branch = await mgr.backtrack(space, branch.id, 0, "B")
        assert new_branch.parent_branch == branch.id
        assert new_branch.decisions[0].chosen_option == "B"


class TestOrchestrator:
    async def _mock_execute(self, branch: Branch) -> dict:
        import random

        return {
            "success": True,
            "results": {},
            "objectives": {"score": random.random()},
        }

    @pytest.mark.asyncio
    async def test_explore_basic(self):
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=3),
            branch_executor=self._mock_execute,
            max_parallel=2,
        )
        result = await orch.explore(
            objective="Test exploration",
            initial_branches=[
                {"name": "b1", "hypothesis": "h1"},
                {"name": "b2", "hypothesis": "h2"},
            ],
            objectives_config={"score": "maximize"},
            max_iterations=5,
        )
        assert isinstance(result, ExplorationResult)
        assert result.n_branches_explored >= 1
        assert result.convergence_reason != ""

    @pytest.mark.asyncio
    async def test_explore_stream(self):
        orch = ExplorationOrchestrator(branch_executor=self._mock_execute)
        updates = []
        async for update in orch.explore_stream(
            objective="Stream test",
            initial_branches=[{"name": "b1", "hypothesis": "h1"}],
            max_iterations=2,
        ):
            updates.append(update)
        assert len(updates) == 1
        assert updates[0]["type"] == "exploration_complete"
