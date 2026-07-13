"""Validation → Knowledge Graph feedback bridge.

When physics/dimensional validators find constraint violations, this module
writes them back to the project KG as new entities and relations. This closes
the loop: validation results become first-class knowledge assets that the
agent can reference in future queries.

Example:
    PhysicsValidator finds "energy must be negative for Si"
    → KG gets: Constraint("energy_negative") -[VALIDATES]-> Compound("Si")
    → Next time agent queries Si, the constraint is available
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def write_validation_to_kg(
    grader_results: list[dict[str, Any]],
    material: str | None = None,
    tool_name: str | None = None,
) -> int:
    """Write validation findings to the project knowledge graph.

    Returns the number of new KG entries created.
    """
    try:
        from huginn.kg.graph import ProjectKnowledgeGraph
        from huginn.kg.entities import EntityType, Relation
    except ImportError:
        logger.debug("[kg_feedback] KG modules not available")
        return 0

    created = 0
    try:
        # Get the project KG singleton — it may not exist in all contexts
        kg = _get_project_kg()
        if kg is None:
            return 0

        for result in grader_results:
            checks = result.get("checks", [])
            passed = result.get("passed", True)
            name = result.get("name", "unknown")

            for check in checks:
                severity = check.get("severity", "info")
                message = check.get("message", "")
                if severity == "error" or (not passed and severity == "warning"):
                    # Create a constraint entity
                    constraint_label = f"{name}:{_slugify(message[:40])}"
                    node_id = f"{EntityType.FACT}:{constraint_label}"
                    if not kg.has_node(node_id):
                        kg.add_node(
                            node_id,
                            entity_type=EntityType.FACT,
                            label=constraint_label,
                            properties={
                                "type": "validation_constraint",
                                "source": name,
                                "message": message,
                                "severity": severity,
                                "material": material or "",
                                "tool": tool_name or "",
                            },
                        )
                        created += 1

                        # Link to material if known
                        if material:
                            mat_id = f"{EntityType.COMPOUND}:{material}"
                            if kg.has_node(mat_id):
                                kg.add_edge(
                                    node_id, mat_id,
                                    relation=Relation.VALIDATES,
                                    properties={"source": "auto_validation"},
                                )

                        # Link to tool if known
                        if tool_name:
                            tool_id = f"{EntityType.TOOL}:{tool_name}"
                            if kg.has_node(tool_id):
                                kg.add_edge(
                                    node_id, tool_id,
                                    relation=Relation.RELATED_TO,
                                    properties={"source": "auto_validation"},
                                )

        if created > 0:
            kg.save()
            logger.info(f"[kg_feedback] wrote {created} constraint(s) to KG")
    except Exception as e:
        logger.debug(f"[kg_feedback] failed: {e}")

    return created


def _get_project_kg() -> Any | None:
    """Get the project knowledge graph from the current context."""
    try:
        from huginn.server_core import get_context
        ctx = get_context()
        kg = getattr(ctx, "knowledge_graph", None)
        if kg is not None:
            return kg
    except Exception:
        pass
    return None


def _slugify(text: str) -> str:
    """Quick slug for entity labels — not URL-safe, just KG-safe."""
    return text.replace(" ", "_").replace(":", "").replace("/", "_")[:40]
