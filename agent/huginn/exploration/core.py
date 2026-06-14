"""Exploration Engine core — multi-branch, async, traceable exploration.

The soul of Huginn: transforms single-task execution into
systematic design-space exploration with Pareto pruning and knowledge graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from datetime import datetime
from enum import Enum

import networkx as nx


class BranchStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class Decision:
    """A decision node in the exploration tree."""
    id: str
    description: str
    decision_type: Literal["categorical", "continuous", "structural", "methodological"]
    chosen_option: Any
    available_options: list[Any]
    rationale: str = ""
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    parent_decision: str | None = None


@dataclass
class Branch:
    """An exploration branch = a path from root through decisions to a computational leaf."""
    id: str
    name: str
    hypothesis: str
    decisions: list[Decision] = field(default_factory=list)
    status: BranchStatus = BranchStatus.PENDING
    
    # Computational state
    jobs: list[str] = field(default_factory=list)
    results: dict[str, Any] = field(default_factory=dict)
    objectives: dict[str, float] = field(default_factory=dict)  # For Pareto evaluation
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    parent_branch: str | None = None
    children_branches: list[str] = field(default_factory=list)
    prune_reason: str | None = None
    
    @property
    def decision_path(self) -> list[str]:
        return [d.description for d in self.decisions]


@dataclass
class ExplorationSpace:
    """Complete design space under exploration."""
    id: str
    name: str
    objective: str  # Natural language objective
    constraints: list[str] = field(default_factory=list)
    
    # Branch management
    branches: dict[str, Branch] = field(default_factory=dict)
    active_branches: set[str] = field(default_factory=set)
    pruned_branches: set[str] = field(default_factory=set)
    
    # Pareto state
    pareto_front: list[str] = field(default_factory=list)
    objectives_config: dict[str, Literal["minimize", "maximize"]] = field(default_factory=dict)
    
    # Knowledge graph
    knowledge_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    def __post_init__(self):
        if not self.knowledge_graph.nodes():
            self.knowledge_graph.add_node("root", type="root", label=self.objective)
    
    def add_branch(self, branch: Branch) -> None:
        self.branches[branch.id] = branch
        
        # Add to knowledge graph
        parent = branch.parent_branch or "root"
        self.knowledge_graph.add_node(
            branch.id,
            type="branch",
            label=branch.name,
            hypothesis=branch.hypothesis,
            status=branch.status.value
        )
        self.knowledge_graph.add_edge(parent, branch.id, type="fork")
        
        # Record decisions as intermediate nodes
        prev_node = branch.id
        for decision in branch.decisions:
            self.knowledge_graph.add_node(
                decision.id,
                type="decision",
                label=decision.description,
                chosen=str(decision.chosen_option),
                confidence=decision.confidence
            )
            self.knowledge_graph.add_edge(prev_node, decision.id, type="decision")
            prev_node = decision.id
    
    def mark_pruned(self, branch_id: str, reason: str) -> None:
        if branch_id in self.branches:
            branch = self.branches[branch_id]
            branch.status = BranchStatus.PRUNED
            branch.prune_reason = reason
            self.pruned_branches.add(branch_id)
            self.active_branches.discard(branch_id)
            
            # Update knowledge graph
            self.knowledge_graph.nodes[branch_id]["status"] = "pruned"
            self.knowledge_graph.nodes[branch_id]["prune_reason"] = reason
    
    def update_pareto_front(self) -> list[str]:
        """Compute Pareto front from all completed branches."""
        completed = [
            b for b in self.branches.values()
            if b.status == BranchStatus.COMPLETED and b.objectives
        ]
        
        if not completed or not self.objectives_config:
            self.pareto_front = []
            return []
        
        # Simple Pareto dominance check
        pareto = []
        for branch in completed:
            dominated = False
            for other in completed:
                if other.id == branch.id:
                    continue
                if self._dominates(other, branch):
                    dominated = True
                    break
            if not dominated:
                pareto.append(branch.id)
        
        self.pareto_front = pareto
        return pareto
    
    def _dominates(self, a: Branch, b: Branch) -> bool:
        """Check if branch a dominates branch b (Pareto sense)."""
        better_in_at_least_one = False
        for obj, direction in self.objectives_config.items():
            val_a = a.objectives.get(obj)
            val_b = b.objectives.get(obj)
            if val_a is None or val_b is None:
                continue
            
            if direction == "maximize":
                if val_a < val_b:
                    return False
                if val_a > val_b:
                    better_in_at_least_one = True
            else:  # minimize
                if val_a > val_b:
                    return False
                if val_a < val_b:
                    better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def get_decision_path(self, branch_id: str) -> list[Decision]:
        """Get the full decision path for a branch (traceability)."""
        if branch_id not in self.branches:
            return []
        return self.branches[branch_id].decisions
    
    def why_pruned(self, branch_id: str) -> str:
        """Explain why a branch was pruned."""
        if branch_id not in self.branches:
            return f"Branch {branch_id} not found"
        branch = self.branches[branch_id]
        if branch.status != BranchStatus.PRUNED:
            return f"Branch {branch_id} was not pruned (status: {branch.status.value})"
        return branch.prune_reason or "No reason recorded"
    
    def export_knowledge_graph(self, format: Literal["json", "gml"] = "json") -> str:
        """Export knowledge graph for visualization."""
        if format == "json":
            from networkx.readwrite import json_graph
            data = json_graph.node_link_data(self.knowledge_graph)
            import json
            return json.dumps(data, indent=2, default=str)
        elif format == "gml":
            import io
            buffer = io.StringIO()
            nx.write_gml(self.knowledge_graph, buffer)
            return buffer.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def query(self, question: str) -> dict[str, Any]:
        """Simple structured query interface.
        
        Examples:
        - {"type": "pareto_front"} → list of branch IDs on Pareto front
        - {"type": "pruned", "reason_contains": "Mn"} → pruned branches matching
        - {"type": "path", "branch_id": "xxx"} → decision path
        """
        # TODO: integrate LLM for natural language → structured query
        return {"status": "not_implemented", "question": question}

    async def run_exploration(
        self,
        strategy: Any | None = None,
        branch_executor: Any | None = None,
        max_iterations: int = 20,
        max_parallel: int = 5,
    ) -> dict[str, Any]:
        """Run a complete exploration on this space using the orchestrator."""
        from huginn.exploration.orchestrator import ExplorationOrchestrator

        orch = ExplorationOrchestrator(
            strategy=strategy,
            branch_executor=branch_executor,
            max_parallel=max_parallel,
        )

        initial = [
            {"name": b.name, "hypothesis": b.hypothesis, "decisions": [
                {
                    "description": d.description,
                    "type": d.decision_type,
                    "chosen": d.chosen_option,
                    "options": d.available_options,
                    "rationale": d.rationale,
                }
                for d in b.decisions
            ]}
            for b in self.branches.values()
        ]

        result = await orch.explore(
            objective=self.objective,
            initial_branches=initial,
            objectives_config=self.objectives_config,
            constraints=self.constraints,
            max_iterations=max_iterations,
        )

        # Merge results back into this space
        self.pareto_front = result.pareto_front
        return {
            "pareto_front": result.pareto_front,
            "best_branch": result.best_branch,
            "n_explored": result.n_branches_explored,
            "n_pruned": result.n_branches_pruned,
            "convergence_reason": result.convergence_reason,
        }
