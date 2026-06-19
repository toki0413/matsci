"""Builders that populate a ProjectKnowledgeGraph from various sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huginn.kg.entities import EntityType, Relation
from huginn.kg.extractor import extract_entities, extract_error_pattern
from huginn.kg.graph import ProjectKnowledgeGraph


def build_from_memory(
    kg: ProjectKnowledgeGraph,
    longterm: Any,
    limit: int = 1000,
) -> dict[str, int]:
    """Import memories as Fact nodes and link mentioned entities."""
    stats = {"facts": 0, "links": 0}
    entries = (
        longterm.retrieve("", top_k=limit, semantic=False)
        if hasattr(longterm, "retrieve")
        else []
    )
    for entry in entries:
        content = entry.get("content", "")
        if not content:
            continue
        fact_id = kg.add_entity(
            _short_label(content, 80),
            EntityType.FACT,
            source="memory",
            confidence=entry.get("importance", 0.5),
            content=content,
            tags=entry.get("tags", []),
        )
        stats["facts"] += 1
        stats["links"] += _link_text_entities(kg, fact_id, content, source="memory")
    return stats


def build_from_logs(
    kg: ProjectKnowledgeGraph,
    logger: Any,
) -> dict[str, int]:
    """Import tool calls and conversations from ExecutionLogger."""
    stats = {"sessions": 0, "tools": 0, "errors": 0, "links": 0}

    tool_calls = getattr(logger, "_tool_calls", [])
    for record in tool_calls:
        session_id = record.session_id or "unknown"
        session_id = kg.add_entity(session_id, EntityType.SESSION, source="tool_call")
        stats["sessions"] += 1

        tool_name = record.tool_name
        tool_id = kg.add_entity(tool_name, EntityType.TOOL, source="tool_call")
        kg.add_relation(session_id, Relation.USED, tool_id, source="tool_call")
        stats["tools"] += 1

        if not record.success and record.error_message:
            pattern = extract_error_pattern(record.error_message)
            if pattern:
                err_id = kg.add_entity(
                    pattern, EntityType.ERROR_PATTERN, source="tool_call"
                )
                kg.add_relation(
                    session_id, Relation.FAILED_WITH, err_id, source="tool_call"
                )
                # Heuristic: link error pattern back to the tool.
                kg.add_relation(err_id, Relation.SOLVED_BY, tool_id, source="auto")
                stats["errors"] += 1

        # Extract entities from input/error text.
        text = str(record.tool_input) + " " + (record.error_message or "")
        stats["links"] += _link_text_entities(kg, tool_id, text, source="tool_call")

    for conv in getattr(logger, "_conversations", []):
        text = f"{conv.user_message} {conv.agent_response}"
        session_id = kg.add_entity(
            conv.session_id or "unknown", EntityType.SESSION, source="conversation"
        )
        stats["links"] += _link_text_entities(
            kg, session_id, text, source="conversation"
        )

    return stats


def build_from_seeds(
    kg: ProjectKnowledgeGraph,
    seed_dir: Path | str | None = None,
) -> dict[str, int]:
    """Import built-in seed knowledge documents as Topic nodes."""
    stats = {"topics": 0, "links": 0}
    if seed_dir is None:
        seed_dir = Path(__file__).parent.parent / "knowledge" / "seed"
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        return stats

    for path in sorted(seed_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        title = path.stem
        topic_id = kg.add_entity(
            title, EntityType.TOPIC, source="seed", content_path=str(path)
        )
        stats["topics"] += 1
        stats["links"] += _link_text_entities(kg, topic_id, content, source="seed")
    return stats


def build_from_session_text(
    kg: ProjectKnowledgeGraph,
    session_id: str,
    text: str,
) -> dict[str, int]:
    """Import a single session/user message as a transient Topic node."""
    session_id_node = kg.add_entity(session_id, EntityType.SESSION, source="session")
    links = _link_text_entities(kg, session_id_node, text, source="session")
    return {"links": links}


def _link_text_entities(
    kg: ProjectKnowledgeGraph,
    source_id: str,
    text: str,
    *,
    source: str = "auto",
) -> int:
    """Extract entities from text and link them to source_id."""
    entities = extract_entities(text)
    count = 0
    for tool in entities["tools"]:
        tid = kg.add_entity(tool, EntityType.TOOL, source=source)
        kg.add_relation(source_id, Relation.MENTIONS, tid, source=source)
        count += 1
    for method in entities["methods"]:
        mid = kg.add_entity(method, EntityType.METHOD, source=source)
        kg.add_relation(source_id, Relation.MENTIONS, mid, source=source)
        count += 1
    for material in entities["materials"]:
        matid = kg.add_entity(material, EntityType.MATERIAL, source=source)
        kg.add_relation(source_id, Relation.MENTIONS, matid, source=source)
        count += 1
    return count


def _short_label(text: str, max_len: int = 80) -> str:
    """Create a short label for a Fact node."""
    clean = text.replace("\n", " ").strip()
    if len(clean) <= max_len:
        return clean
    return clean[:max_len].rsplit(" ", 1)[0] + "..."
