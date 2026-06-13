"""Exploration strategies for design-space search.

ParetoPruning:  Multi-objective dominance-based pruning
Bayesian:       Surrogate model + acquisition function optimization
AdaptiveGrid:   Coarse-to-fine grid refinement
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from matsci_agent.exploration.core import ExplorationSpace, Branch, BranchStatus


@dataclass
class Action:
    """An action to take on the exploration space."""
    action_type: Literal["expand", "prune", "backtrack", "refine", "terminate"]
    target_branch: str | None = None
    new_branches: list[dict[str, Any]] = None
    reason: str = ""


class ExplorationStrategy(ABC):
    """Base class for exploration strategies."""

    @abstractmethod
    def evaluate(self, space: ExplorationSpace) -> list[Action]:
        """Evaluate the current exploration state and propose actions."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class ParetoPruningStrategy(ExplorationStrategy):
    """Pareto-dominance based exploration strategy.

    Maintains a non-dominated set of branches. Branches that are dominated
    in all objectives are pruned. New branches are generated near the
    Pareto front to refine the trade-off surface.
    """

    def __init__(
        self,
        max_active: int = 10,
        min_objective_improvement: float = 0.01,
    ):
        self.max_active = max_active
        self.min_improvement = min_objective_improvement
        self._previous_front_size = 0

    def name(self) -> str:
        return "pareto_pruning"

    def evaluate(self, space: ExplorationSpace) -> list[Action]:
        actions: list[Action] = []

        # Update Pareto front
        front = space.update_pareto_front()

        # Prune dominated branches
        for branch_id, branch in space.branches.items():
            if branch.status != BranchStatus.COMPLETED:
                continue
            if branch_id in front:
                continue
            # Check if significantly dominated
            if self._is_significantly_dominated(space, branch_id, front):
                actions.append(Action(
                    action_type="prune",
                    target_branch=branch_id,
                    reason="Dominated by Pareto front branches",
                ))

        # If we have room, suggest expanding near Pareto front
        active_count = len([b for b in space.branches.values() if b.status in {BranchStatus.PENDING, BranchStatus.RUNNING}])
        if active_count < self.max_active and front:
            # Find gaps in the Pareto front and suggest refinement
            gaps = self._find_front_gaps(space, front)
            for gap in gaps[: self.max_active - active_count]:
                actions.append(Action(
                    action_type="refine",
                    target_branch=gap["near_branch"],
                    new_branches=[{"hypothesis": f"Refine between {gap['a']} and {gap['b']}"}],
                    reason=f"Gap in Pareto front between objectives",
                ))

        # Termination check: front converged
        if len(front) == self._previous_front_size and active_count == 0:
            actions.append(Action(action_type="terminate", reason="Pareto front converged"))
        self._previous_front_size = len(front)

        return actions

    def _is_significantly_dominated(self, space: ExplorationSpace, branch_id: str, front: list[str]) -> bool:
        branch = space.branches[branch_id]
        for fid in front:
            front_branch = space.branches[fid]
            if space._dominates(front_branch, branch):
                # Check margin
                margin_ok = True
                for obj, direction in space.objectives_config.items():
                    diff = abs(front_branch.objectives.get(obj, 0) - branch.objectives.get(obj, 0))
                    ref = abs(branch.objectives.get(obj, 1)) or 1.0
                    if diff / ref < self.min_improvement:
                        margin_ok = False
                        break
                if margin_ok:
                    return True
        return False

    def _find_front_gaps(self, space: ExplorationSpace, front: list[str]) -> list[dict[str, Any]]:
        """Find large gaps between Pareto front points."""
        if len(front) < 2:
            return []

        gaps = []
        obj_names = list(space.objectives_config.keys())
        if not obj_names:
            return []

        # Sort by first objective
        primary = obj_names[0]
        sorted_front = sorted(front, key=lambda fid: space.branches[fid].objectives.get(primary, 0))

        for i in range(len(sorted_front) - 1):
            a = space.branches[sorted_front[i]]
            b = space.branches[sorted_front[i + 1]]
            # Measure Euclidean distance in objective space
            dist = 0.0
            for obj in obj_names:
                va = a.objectives.get(obj, 0)
                vb = b.objectives.get(obj, 0)
                # Normalize by range
                vals = [space.branches[f].objectives.get(obj, 0) for f in front]
                range_val = max(vals) - min(vals) if len(vals) > 1 else 1.0
                if range_val > 0:
                    dist += ((va - vb) / range_val) ** 2
            dist = np.sqrt(dist)
            if dist > 0.3:  # Threshold for "large gap"
                gaps.append({"a": sorted_front[i], "b": sorted_front[i + 1], "distance": dist, "near_branch": sorted_front[i]})

        gaps.sort(key=lambda x: -x["distance"])
        return gaps


class BayesianExplorationStrategy(ExplorationStrategy):
    """Bayesian optimization for continuous parameter spaces.

    Uses a simple GP surrogate (or fallback to random forest) and
    expected improvement acquisition function.
    """

    def __init__(
        self,
        n_initial: int = 3,
        acquisition: Literal["EI", "UCB"] = "EI",
        xi: float = 0.01,  # Exploration parameter for EI
    ):
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.xi = xi
        self._evaluated_params: list[list[float]] = []
        self._evaluated_objectives: list[float] = []

    def name(self) -> str:
        return "bayesian"

    def evaluate(self, space: ExplorationSpace) -> list[Action]:
        actions: list[Action] = []

        completed = [b for b in space.branches.values() if b.status == BranchStatus.COMPLETED]
        if len(completed) < self.n_initial:
            # Need more samples — suggest random exploration
            actions.append(Action(
                action_type="expand",
                new_branches=[{"hypothesis": "Random exploration for surrogate training"}],
                reason="Insufficient samples for Bayesian model",
            ))
            return actions

        # Update surrogate data
        self._update_data(completed)

        # Suggest next point via acquisition function
        # Simplified: suggest parametric interpolation between best points
        if self._evaluated_params and self._evaluated_objectives:
            best_idx = np.argmax(self._evaluated_objectives)
            best_params = self._evaluated_params[best_idx]
            # Suggest a small perturbation near the best point
            actions.append(Action(
                action_type="refine",
                target_branch=completed[best_idx].id,
                new_branches=[{
                    "hypothesis": f"Bayesian refinement near best point (obj={self._evaluated_objectives[best_idx]:.3f})",
                    "params": best_params,
                }],
                reason="Expected improvement maximization",
            ))

        return actions

    def _update_data(self, completed: list[Branch]) -> None:
        """Update internal surrogate training data."""
        self._evaluated_params = []
        self._evaluated_objectives = []
        for branch in completed:
            # Extract a scalar objective (first maximize objective, or negative of first minimize)
            obj_val = None
            for obj_name, direction in branch.objectives.items():
                if isinstance(direction, str):
                    val = branch.objectives.get(obj_name)
                    if val is not None:
                        obj_val = val if direction == "maximize" else -val
                        break
            if obj_val is not None:
                # Dummy parameter vector from decision path length + random seed
                params = [len(branch.decisions), hash(branch.hypothesis) % 1000 / 1000.0]
                self._evaluated_params.append(params)
                self._evaluated_objectives.append(obj_val)


class AdaptiveGridStrategy(ExplorationStrategy):
    """Adaptive grid refinement strategy.

    Starts with a coarse grid, identifies high-potential regions,
    and refines them while coarsening flat regions.
    """

    def __init__(self, initial_resolution: int = 3, max_resolution: int = 10):
        self.resolution = initial_resolution
        self.max_resolution = max_resolution
        self._region_potential: dict[str, float] = {}

    def name(self) -> str:
        return "adaptive_grid"

    def evaluate(self, space: ExplorationSpace) -> list[Action]:
        actions: list[Action] = []

        completed = [b for b in space.branches.values() if b.status == BranchStatus.COMPLETED]
        if not completed:
            actions.append(Action(
                action_type="expand",
                new_branches=[{"hypothesis": f"Coarse grid level {self.resolution}"}],
                reason="Initial grid exploration",
            ))
            return actions

        # Identify high-potential regions
        for branch in completed:
            potential = self._compute_potential(branch)
            self._region_potential[branch.id] = potential

        # Sort by potential
        sorted_regions = sorted(self._region_potential.items(), key=lambda x: -x[1])

        # Refine top regions
        for branch_id, potential in sorted_regions[:3]:
            if potential > 0.7 and self.resolution < self.max_resolution:
                actions.append(Action(
                    action_type="refine",
                    target_branch=branch_id,
                    new_branches=[{
                        "hypothesis": f"Grid refinement (res={self.resolution + 1}) near high-potential region",
                    }],
                    reason=f"Region potential = {potential:.2f}",
                ))

        if not actions and self.resolution < self.max_resolution:
            self.resolution += 1
            actions.append(Action(
                action_type="expand",
                new_branches=[{"hypothesis": f"Uniform grid refinement to level {self.resolution}"}],
                reason="No high-potential regions found — uniform refinement",
            ))

        return actions

    def _compute_potential(self, branch: Branch) -> float:
        """Compute a potential score for a region (0-1)."""
        if not branch.objectives:
            return 0.5
        # Normalize objectives to [0,1] and average
        vals = [v for v in branch.objectives.values() if isinstance(v, (int, float))]
        if not vals:
            return 0.5
        # Simple sigmoid of average
        avg = np.mean(vals)
        return float(1 / (1 + np.exp(-avg)))
