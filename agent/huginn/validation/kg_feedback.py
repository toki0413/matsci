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
        from huginn.kg.entities import EntityType, Relation
    except ImportError:
        logger.debug("[kg_feedback] KG modules not available")
        return 0

    kg = _get_project_kg()
    if kg is None:
        return 0

    created = 0
    try:
        for result in grader_results:
            created += _write_result_constraints(
                kg, result, material, tool_name, EntityType, Relation
            )

        if created > 0:
            kg.save()
            logger.info(f"[kg_feedback] wrote {created} constraint(s) to KG")
    except Exception as e:
        logger.debug(f"[kg_feedback] failed: {e}")

    return created


def _write_result_constraints(
    kg: Any,
    result: dict[str, Any],
    material: str | None,
    tool_name: str | None,
    entity_type: Any,
    relation: Any,
) -> int:
    """把一个 grader 结果里的违规 check 落成 KG 约束节点, 返回新增节点数."""
    checks = result.get("checks", [])
    passed = result.get("passed", True)
    name = result.get("name", "unknown")

    created = 0
    for check in checks:
        severity = check.get("severity", "info")
        if severity != "error" and not (not passed and severity == "warning"):
            continue

        message = check.get("message", "")
        constraint_label = f"{name}:{_slugify(message[:40])}"
        node_id = f"{entity_type.FACT}:{constraint_label}"
        if kg.has_node(node_id):
            continue

        kg.add_node(
            node_id,
            entity_type=entity_type.FACT,
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

        if material:
            _link_if_present(
                kg,
                node_id,
                f"{entity_type.COMPOUND}:{material}",
                relation.VALIDATES,
            )
        if tool_name:
            _link_if_present(
                kg,
                node_id,
                f"{entity_type.TOOL}:{tool_name}",
                relation.RELATED_TO,
            )
    return created


def _link_if_present(kg: Any, src: str, dst: str, relation: Any) -> None:
    """目标节点存在时连边, 不存在就跳过 (不凭空造节点)."""
    if kg.has_node(dst):
        kg.add_edge(
            src,
            dst,
            relation=relation,
            properties={"source": "auto_validation"},
        )


def _get_project_kg() -> Any | None:
    """Get the project knowledge graph from the current context."""
    try:
        from huginn.server_core import get_context
        ctx = get_context()
        kg = getattr(ctx, "knowledge_graph", None)
        if kg is not None:
            return kg
    except Exception:  # — 项目知识图谱不可得 → 回退 None(不抛)
        logger.debug("project knowledge graph lookup failed", exc_info=True)
    return None


def _slugify(text: str) -> str:
    """Quick slug for entity labels — not URL-safe, just KG-safe."""
    return text.replace(" ", "_").replace(":", "").replace("/", "_")[:40]
