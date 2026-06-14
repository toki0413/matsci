"""Branch lifecycle manager — creation, execution, pruning, backtracking.

Handles the full lifecycle of an exploration branch from hypothesis
through computation to result or failure.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable, Awaitable

from huginn.exploration.core import Branch, Decision, BranchStatus, ExplorationSpace


class BranchLifecycleManager:
    """Manages the full lifecycle of exploration branches."""

    def __init__(
        self,
        execute_fn: Callable[[Branch], Awaitable[dict[str, Any]]] | None = None,
    ):
        """Args:
            execute_fn: Async function that executes a branch's computational tasks.
                       Should return a dict with at least {"success": bool, "objectives": dict}.
        """
        self.execute_fn = execute_fn or self._default_execute

    async def create_branch(
        self,
        space: ExplorationSpace,
        name: str,
        hypothesis: str,
        parent: str | None = None,
        decisions: list[Decision] | None = None,
    ) -> Branch:
        """Create a new branch in the exploration space."""
        branch_id = f"branch_{len(space.branches):04d}_{uuid.uuid4().hex[:6]}"
        branch = Branch(
            id=branch_id,
            name=name,
            hypothesis=hypothesis,
            decisions=decisions or [],
            parent_branch=parent,
        )
        space.add_branch(branch)
        space.active_branches.add(branch_id)
        return branch

    async def execute_branch(self, space: ExplorationSpace, branch_id: str) -> dict[str, Any]:
        """Execute a branch's computational tasks."""
        if branch_id not in space.branches:
            return {"success": False, "error": f"Branch {branch_id} not found"}

        branch = space.branches[branch_id]
        branch.status = BranchStatus.RUNNING

        try:
            result = await self.execute_fn(branch)

            if result.get("success"):
                branch.status = BranchStatus.COMPLETED
                branch.results = result.get("results", {})
                branch.objectives = result.get("objectives", {})
                # Update knowledge graph
                space.knowledge_graph.nodes[branch_id]["status"] = "completed"
                for key, val in branch.objectives.items():
                    space.knowledge_graph.nodes[branch_id][f"obj_{key}"] = val
            else:
                branch.status = BranchStatus.FAILED
                space.knowledge_graph.nodes[branch_id]["status"] = "failed"
                space.knowledge_graph.nodes[branch_id]["error"] = result.get("error", "Unknown")

            space.active_branches.discard(branch_id)
            return result

        except Exception as e:
            branch.status = BranchStatus.FAILED
            space.active_branches.discard(branch_id)
            space.knowledge_graph.nodes[branch_id]["status"] = "failed"
            space.knowledge_graph.nodes[branch_id]["error"] = str(e)
            return {"success": False, "error": str(e)}

    async def prune_branch(
        self,
        space: ExplorationSpace,
        branch_id: str,
        reason: str,
        cascade: bool = True,
    ) -> None:
        """Prune a branch and optionally all its descendants."""
        space.mark_pruned(branch_id, reason)

        if cascade:
            # Find and prune children
            children = [
                bid for bid, b in space.branches.items()
                if b.parent_branch == branch_id and b.status != BranchStatus.PRUNED
            ]
            for child_id in children:
                await self.prune_branch(space, child_id, f"Parent {branch_id} pruned: {reason}", cascade=True)

    async def backtrack(
        self,
        space: ExplorationSpace,
        branch_id: str,
        to_decision_index: int,
        new_option: Any,
    ) -> Branch:
        """Backtrack to a previous decision and create a new branch with a different choice."""
        if branch_id not in space.branches:
            raise ValueError(f"Branch {branch_id} not found")

        old_branch = space.branches[branch_id]
        if to_decision_index >= len(old_branch.decisions):
            raise ValueError(f"Decision index {to_decision_index} out of range")

        # Copy decisions up to backtrack point
        kept_decisions = old_branch.decisions[:to_decision_index]
        # Modify the decision at backtrack point
        target_decision = old_branch.decisions[to_decision_index]
        new_decision = Decision(
            id=f"{target_decision.id}_bt_{uuid.uuid4().hex[:4]}",
            description=target_decision.description,
            decision_type=target_decision.decision_type,
            chosen_option=new_option,
            available_options=target_decision.available_options,
            rationale=f"Backtrack from {branch_id} with new option",
            parent_decision=target_decision.parent_decision,
        )
        kept_decisions.append(new_decision)

        new_branch = await self.create_branch(
            space=space,
            name=f"{old_branch.name}_bt",
            hypothesis=f"Backtrack: {old_branch.hypothesis} → option {new_option}",
            parent=branch_id,
            decisions=kept_decisions,
        )
        return new_branch

    async def _default_execute(self, branch: Branch) -> dict[str, Any]:
        """Default no-op execution (for testing/mocking)."""
        import random
        return {
            "success": True,
            "results": {"mock": True, "hypothesis": branch.hypothesis},
            "objectives": {"score": random.random()},
        }
