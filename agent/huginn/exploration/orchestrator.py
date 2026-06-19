"""Exploration Orchestrator — top-level coordinator for design-space search.

Transforms a user objective into a systematic multi-branch exploration,
executing branches, applying strategies, and converging to a Pareto front.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from huginn.exploration.core import Branch, BranchStatus, Decision, ExplorationSpace
from huginn.exploration.lifecycle import BranchLifecycleManager
from huginn.exploration.strategies import (
    ExplorationStrategy,
    ParetoPruningStrategy,
)


@dataclass
class ExplorationResult:
    """Final result of an exploration run."""

    space: ExplorationSpace
    pareto_front: list[dict[str, Any]]
    best_branch: dict[str, Any] | None
    n_branches_explored: int
    n_branches_pruned: int
    convergence_reason: str
    knowledge_graph_json: str


class ExplorationOrchestrator:
    """Orchestrates the full exploration lifecycle.

    Usage:
        orch = ExplorationOrchestrator(strategy=ParetoPruningStrategy())
        result = await orch.explore(
            objective="Find highest energy density cathode",
            initial_branches=[{"name": "LiCoO2", "hypothesis": "Baseline layered"}, ...],
            max_iterations=20,
        )
    """

    def __init__(
        self,
        strategy: ExplorationStrategy | None = None,
        branch_executor: Callable[[Branch], Awaitable[dict[str, Any]]] | None = None,
        max_parallel: int = 5,
    ):
        self.strategy = strategy or ParetoPruningStrategy()
        self.lifecycle = BranchLifecycleManager(execute_fn=branch_executor)
        self.max_parallel = max_parallel
        self._should_stop = False

    async def explore(
        self,
        objective: str,
        initial_branches: list[dict[str, Any]],
        objectives_config: dict[str, str] | None = None,
        constraints: list[str] | None = None,
        max_iterations: int = 20,
        budget: dict[str, float] | None = None,
    ) -> ExplorationResult:
        """Run a complete exploration.

        Args:
            objective: Natural language exploration goal.
            initial_branches: List of dicts with keys: name, hypothesis, decisions (optional).
            objectives_config: Mapping of objective name → "minimize" or "maximize".
            constraints: List of constraint strings.
            max_iterations: Maximum exploration iterations.
            budget: Optional budget dict (e.g., {"max_cpu_hours": 100}).
        """
        space = ExplorationSpace(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            name=objective[:50],
            objective=objective,
            constraints=constraints or [],
            objectives_config=objectives_config or {},
        )

        # Create initial branches
        for init in initial_branches:
            await self.lifecycle.create_branch(
                space=space,
                name=init["name"],
                hypothesis=init["hypothesis"],
                decisions=[
                    Decision(
                        id=f"dec_{i}",
                        description=d.get("description", f"decision_{i}"),
                        decision_type=d.get("type", "categorical"),
                        chosen_option=d.get("chosen"),
                        available_options=d.get("options", []),
                        rationale=d.get("rationale", ""),
                    )
                    for i, d in enumerate(init.get("decisions", []))
                ],
            )

        iteration = 0
        convergence_reason = "max_iterations reached"

        while iteration < max_iterations and not self._should_stop:
            iteration += 1

            # 1. Execute all pending branches (up to max_parallel)
            pending = [
                bid
                for bid, b in space.branches.items()
                if b.status == BranchStatus.PENDING
            ]
            if pending:
                batch = pending[: self.max_parallel]
                await asyncio.gather(
                    *[self.lifecycle.execute_branch(space, bid) for bid in batch]
                )

            # 2. Evaluate strategy
            actions = self.strategy.evaluate(space)

            # 3. Apply actions
            terminate = False
            for action in actions:
                if action.action_type == "terminate":
                    terminate = True
                    convergence_reason = action.reason
                    break
                elif action.action_type == "prune" and action.target_branch:
                    await self.lifecycle.prune_branch(
                        space, action.target_branch, action.reason, cascade=True
                    )
                elif action.action_type == "expand" and action.new_branches:
                    for nb in action.new_branches:
                        await self.lifecycle.create_branch(
                            space=space,
                            name=nb.get("name", "unnamed"),
                            hypothesis=nb.get("hypothesis", "No hypothesis"),
                            parent=action.target_branch,
                        )
                elif action.action_type == "refine" and action.new_branches:
                    for nb in action.new_branches:
                        await self.lifecycle.create_branch(
                            space=space,
                            name=nb.get("name", "refinement"),
                            hypothesis=nb.get("hypothesis", "Refinement"),
                            parent=action.target_branch,
                        )

            if terminate:
                break

            # 4. Check if all branches are resolved
            unresolved = [
                b
                for b in space.branches.values()
                if b.status in {BranchStatus.PENDING, BranchStatus.RUNNING}
            ]
            if not unresolved and not any(
                a.action_type in {"expand", "refine"} for a in actions
            ):
                convergence_reason = "All branches resolved with no new actions"
                break

        # Finalize
        front = space.update_pareto_front()
        best = None
        if front:
            best_branch = space.branches[front[0]]
            best = {
                "id": best_branch.id,
                "name": best_branch.name,
                "hypothesis": best_branch.hypothesis,
                "objectives": best_branch.objectives,
                "results": best_branch.results,
            }

        return ExplorationResult(
            space=space,
            pareto_front=[
                {
                    "id": bid,
                    "name": space.branches[bid].name,
                    "objectives": space.branches[bid].objectives,
                }
                for bid in front
            ],
            best_branch=best,
            n_branches_explored=len(
                [
                    b
                    for b in space.branches.values()
                    if b.status == BranchStatus.COMPLETED
                ]
            ),
            n_branches_pruned=len(space.pruned_branches),
            convergence_reason=convergence_reason,
            knowledge_graph_json=space.export_knowledge_graph("json"),
        )

    async def explore_stream(
        self,
        objective: str,
        initial_branches: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream exploration progress in real-time.

        Yields status updates after each iteration.
        """
        result = await self.explore(objective, initial_branches, **kwargs)
        yield {
            "type": "exploration_complete",
            "pareto_front": result.pareto_front,
            "best_branch": result.best_branch,
            "n_explored": result.n_branches_explored,
            "n_pruned": result.n_branches_pruned,
            "convergence_reason": result.convergence_reason,
            "knowledge_graph": result.knowledge_graph_json,
        }

    def stop(self) -> None:
        """Signal the exploration to stop at the next safe point."""
        self._should_stop = True
