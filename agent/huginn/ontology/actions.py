"""Formal action types for the materials science agent.

Each action is a first-class object in the ontology — not just a tool call
but a typed, constrained, verifiable, traceable operation.

Action predictability (inspired by PNAS 2535161123):
  Global predictability = product of local contributions
  P(action) = P(precondition_met) * P(constraint_satisfied) * P(verifiable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ActionCategory(str, Enum):
    """Coarse classification of agent actions."""
    QUERY = "query"           # Read-only lookup, no side effects
    ANALYZE = "analyze"        # Computation on existing data
    SIMULATE = "simulate"     # Submit/monitor a heavy calculation
    FILE_OPS = "file_ops"     # Create/modify/delete files
    CODE = "code"             # Execute arbitrary code
    NETWORK = "network"       # External API / web / database calls
    LEARN = "learn"            # Update memory / knowledge / preferences
    COMMUNICATE = "communicate"  # Send info to user or external system


class ActionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFIED = "verified"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    DENIED = "denied"


class RiskLevel(str, Enum):
    NONE = "none"       # query/lookup
    LOW = "low"         # analyze existing data
    MEDIUM = "medium"   # file creation, code execution
    HIGH = "high"       # file modification/deletion, heavy simulation
    CRITICAL = "critical"  # git operations, external submissions


@dataclass
class Precondition:
    """A condition that must hold before an action executes."""
    name: str
    check: Callable[[dict[str, Any]], bool]
    description: str = ""
    # when False, action is blocked (not just warned)
    blocking: bool = True

    def evaluate(self, context: dict[str, Any]) -> bool:
        return self.check(context)


@dataclass
class Effect:
    """What changes after an action executes."""
    name: str
    # positive = creates/updates knowledge, negative = removes/overwrites
    positive: bool = True
    description: str = ""
    # which KG entity types are affected
    affects_entities: list[str] = field(default_factory=list)
    # which relations are created/modified/broken
    affects_relations: list[str] = field(default_factory=list)


@dataclass
class Constraint:
    """A bound that must hold during and after action execution.

    Inspired by learning mechanics: constraints are the "boundary conditions"
    of the action — they define the regime of applicability.
    """
    name: str
    # returns (satisfied, message) tuple
    check: Callable[[dict[str, Any]], tuple[bool, str]]
    description: str = ""
    # if violated, the action is rolled back
    rollback_on_violation: bool = True

    def evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        return self.check(context)


@dataclass
class Verifiability:
    """How to check that an action succeeded."""
    method: str  # "checksum", "physics_check", "file_exists", "diff", "custom"
    check: Callable[[dict[str, Any]], tuple[bool, str]] | None = None
    description: str = ""
    # external validation tools to run
    validators: list[str] = field(default_factory=list)

    def verify(self, context: dict[str, Any]) -> tuple[bool, str]:
        if self.check is None:
            return True, "no verification configured"
        return self.check(context)


@dataclass
class ActionType:
    """A formal action type in the materials science ontology.

    This is the "class" — instances are created per-execution.
    """
    name: str
    category: ActionCategory
    risk: RiskLevel
    description: str = ""

    preconditions: list[Precondition] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    verifiability: Verifiability | None = None

    # audit: what to log
    audit_fields: list[str] = field(default_factory=lambda: [
        "timestamp", "user", "tool_name", "args", "result", "status"
    ])

    # rollback: how to undo this action
    rollback_handler: Callable[[dict[str, Any]], bool] | None = None

    def predictability(self, context: dict[str, Any]) -> float:
        """Estimate action predictability (0..1).

        PNAS insight: predictability decomposes into local contributions.
        Each failed precondition halves the score; each violated constraint
        reduces it proportionally. This is a heuristic — the real bound comes
        from the network structure of dependencies.
        """
        score = 1.0
        for pre in self.preconditions:
            if not pre.evaluate(context):
                score *= 0.5 if pre.blocking else 0.9
        for con in self.constraints:
            ok, _ = con.evaluate(context)
            if not ok:
                score *= 0.3
        if self.verifiability:
            ok, _ = self.verifiability.verify(context)
            if not ok:
                score *= 0.5
        return score

    def can_execute(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check all preconditions. Returns (allowed, reasons)."""
        reasons: list[str] = []
        for pre in self.preconditions:
            if not pre.evaluate(context):
                msg = f"Precondition '{pre.name}' not met"
                if pre.description:
                    msg += f": {pre.description}"
                reasons.append(msg)
                if pre.blocking:
                    return False, reasons
        return True, reasons


# ── Registry of known action types ──────────────────────────────

_REGISTRY: dict[str, ActionType] = {}


def register_action_type(at: ActionType) -> None:
    _REGISTRY[at.name] = at


def get_action_type(name: str) -> ActionType | None:
    return _REGISTRY.get(name)


def list_action_types() -> dict[str, ActionType]:
    return dict(_REGISTRY)


def _init_builtin_types() -> None:
    """Register built-in action types for common materials science operations."""

    register_action_type(ActionType(
        name="query_property",
        category=ActionCategory.QUERY,
        risk=RiskLevel.NONE,
        description="Look up a materials property (band gap, lattice constant, etc.)",
        preconditions=[
            Precondition(
                name="material_specified",
                check=lambda ctx: bool(ctx.get("material") or ctx.get("compound")),
                description="A material or compound must be identified",
            ),
        ],
        effects=[
            Effect(name="property_retrieved", positive=True, affects_entities=["Property"]),
        ],
        verifiability=Verifiability(
            method="value_check",
            description="Returned value should be a number with units",
            validators=["physics_validator"],
        ),
    ))

    register_action_type(ActionType(
        name="run_dft",
        category=ActionCategory.SIMULATE,
        risk=RiskLevel.HIGH,
        description="Submit a DFT calculation (VASP/QE/CP2K)",
        preconditions=[
            Precondition(
                name="structure_available",
                check=lambda ctx: bool(ctx.get("structure") or ctx.get("poscar")),
                description="A crystal structure must be provided",
            ),
            Precondition(
                name="params_set",
                check=lambda ctx: bool(ctx.get("encut") and ctx.get("kpoints")),
                description="ENCUT and K-points must be set",
                blocking=False,
            ),
        ],
        constraints=[
            Constraint(
                name="energy_negative",
                check=lambda ctx: (
                    ctx.get("energy", 0) < 0,
                    f"Energy {ctx.get('energy')} should be negative for a bound system"
                ),
                description="Total energy must be negative",
            ),
            Constraint(
                name="forces_converged",
                check=lambda ctx: (
                    ctx.get("max_force", 999) < 0.01,
                    f"Max force {ctx.get('max_force')} exceeds 0.01 eV/A"
                ),
                description="Forces must be below 0.01 eV/A for convergence",
                rollback_on_violation=False,  # don't rollback, just warn
            ),
        ],
        effects=[
            Effect(name="calculation_result", positive=True,
                   affects_entities=["Property", "Fact"],
                   affects_relations=["computed_with", "has_property"]),
            Effect(name="cpu_hours_consumed", positive=False),
        ],
        verifiability=Verifiability(
            method="physics_check",
            description="Run PhysicsValidator on output",
            validators=["physics_validator", "dimensional_validator"],
        ),
        audit_fields=[
            "timestamp", "user", "tool_name", "args",
            "result", "status", "cpu_hours", "job_id"
        ],
    ))

    register_action_type(ActionType(
        name="file_edit",
        category=ActionCategory.FILE_OPS,
        risk=RiskLevel.MEDIUM,
        description="Edit or overwrite a file in the workspace",
        preconditions=[
            Precondition(
                name="file_exists",
                check=lambda ctx: bool(ctx.get("file_path")),
                description="A file path must be provided",
            ),
            Precondition(
                name="not_user_original",
                check=lambda ctx: not ctx.get("is_user_original", False),
                description="Cannot edit user's original input files",
            ),
        ],
        effects=[
            Effect(name="file_modified", positive=True, affects_entities=["Resource"]),
            Effect(name="previous_version_lost", positive=False),
        ],
        verifiability=Verifiability(
            method="checksum",
            description="Verify file content hash matches expected",
        ),
        rollback_handler=lambda ctx: bool(ctx.get("backup_path")),
        audit_fields=[
            "timestamp", "user", "file_path", "old_hash",
            "new_hash", "status", "backup_path"
        ],
    ))

    register_action_type(ActionType(
        name="code_execute",
        category=ActionCategory.CODE,
        risk=RiskLevel.MEDIUM,
        description="Execute arbitrary Python code",
        preconditions=[
            Precondition(
                name="not_empty",
                check=lambda ctx: bool(ctx.get("code", "").strip()),
                description="Code must not be empty",
            ),
        ],
        constraints=[
            Constraint(
                name="no_infinite_loop",
                check=lambda ctx: (
                    ctx.get("executed", False),
                    "Code did not complete execution"
                ),
                description="Code must terminate within timeout",
            ),
        ],
        effects=[
            Effect(name="code_executed", positive=True, affects_entities=["Fact"]),
        ],
        verifiability=Verifiability(
            method="output_check",
            description="Verify execution produced expected output",
        ),
        audit_fields=["timestamp", "user", "code_hash", "output", "status"],
    ))

    register_action_type(ActionType(
        name="knowledge_update",
        category=ActionCategory.LEARN,
        risk=RiskLevel.LOW,
        description="Update the knowledge graph with new information",
        effects=[
            Effect(name="kg_extended", positive=True,
                   affects_entities=["Topic", "Material", "Fact"],
                   affects_relations=["mentions", "has_property", "related_to"]),
        ],
        verifiability=Verifiability(
            method="graph_check",
            description="Verify new nodes/edges exist in the graph",
            validators=["dimensional_validator"],
        ),
        audit_fields=["timestamp", "user", "entity_type", "relation", "status"],
    ))


_init_builtin_types()
