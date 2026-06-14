"""Unified memory manager — orchestrates session and long-term memory.

Provides a single interface for all memory operations with automatic
promotion of important session data to long-term storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from huginn.memory.session import SessionContext, ToolCallRecord
from huginn.memory.longterm import LongTermMemory, MemoryEntry
from huginn.types import AgentMessage, ToolResult


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    auto_promote_to_longterm: bool = True
    promotion_importance_threshold: float = 0.6
    max_session_age_hours: float = 24.0
    enable_semantic_search: bool = True
    memory_md_path: Path | None = None


class MemoryManager:
    """Central memory coordinator for Huginn."""

    def __init__(
        self,
        session: SessionContext | None = None,
        longterm: LongTermMemory | None = None,
        config: MemoryConfig | None = None,
    ):
        self.session = session or SessionContext()
        self.longterm = longterm or LongTermMemory()
        self.config = config or MemoryConfig()

    # --- Session memory operations ---

    def add_message(self, role: str, content: str | dict[str, Any]) -> None:
        msg = AgentMessage(role=role, content=content)
        self.session.add_message(msg)

    def add_tool_call(
        self,
        tool_name: str,
        input_args: dict[str, Any],
        result: Any = None,
        duration_ms: float = 0.0,
    ) -> None:
        from huginn.types import ToolResult
        record = ToolCallRecord(
            tool_name=tool_name,
            input_args=input_args,
            result=result if isinstance(result, ToolResult) else None,
            duration_ms=duration_ms,
        )
        self.session.add_tool_call(record)

        # Auto-promote important tool results to long-term memory
        if self.config.auto_promote_to_longterm and result:
            if hasattr(result, "success") and result.success:
                if tool_name in {"vasp_tool", "lammps_tool", "structure_tool"}:
                    self._promote_tool_result(record)

    def add_reasoning(self, text: str) -> None:
        self.session.add_reasoning(text)

    def set_context(self, key: str, value: Any) -> None:
        self.session.set_working_memory(key, value)

    def get_context(self, key: str, default: Any = None) -> Any:
        return self.session.get_working_memory(key, default)

    # --- Long-term memory operations ---

    def remember(
        self,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        importance: float = 0.5,
        tier: str = "mid",
    ) -> str:
        """Explicitly store a fact in long-term memory."""
        return self.longterm.store(
            content=content,
            category=category,
            tags=tags,
            source=f"session:{self.session.session_id}",
            importance=importance,
            tier=tier,
        )

    def recall(
        self,
        query: str,
        category: str | None = None,
        tier: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search long-term memory."""
        return self.longterm.retrieve(
            query=query,
            category=category,
            tier=tier,
            top_k=top_k,
            semantic=self.config.enable_semantic_search,
        )

    def recall_for_prompt(self, query: str, max_entries: int = 3) -> str:
        """Format recalled memories for injection into LLM prompt."""
        results = self.recall(query, top_k=max_entries)
        if not results:
            return ""
        lines = ["## Relevant past knowledge:"]
        for r in results:
            provenance = f" ({r.get('source', '')})" if r.get("source") else ""
            lines.append(f"- [{r.get('category', 'fact')}] {r.get('content', '')}{provenance}")
        return "\n".join(lines)

    # --- Session promotion ---

    def promote_tool_result(self, name: str, result: dict[str, Any]) -> None:
        """Manually promote a tool result to long-term memory."""
        record = ToolCallRecord(
            tool_name=name,
            input_args={},
            result=ToolResult(data=result, success=True),
        )
        self._promote_tool_result(record)

    def _promote_tool_result(self, record: ToolCallRecord) -> None:
        """Promote a successful computational result to long-term memory."""
        if not record.result or not record.result.data:
            return
        content = f"{record.tool_name}: {json.dumps(record.result.data, default=str)[:500]}"
        self.longterm.store(
            content=content,
            category="calculation",
            tags=[record.tool_name, "auto_promoted"],
            source=f"session:{self.session.session_id}/call:{record.call_id}",
            importance=self.config.promotion_importance_threshold,
            tier="mid",
        )

    def promote_session_summary(self, tier: str = "mid") -> str:
        """Summarize current session and store in long-term memory."""
        summary = (
            f"Session {self.session.session_id}: "
            f"{len(self.session.messages)} messages, "
            f"{len(self.session.tool_calls)} tool calls. "
            f"Topics: {self._extract_topics()}"
        )
        return self.longterm.store(
            content=summary,
            category="conversation",
            tags=["session_summary"],
            source=f"session:{self.session.session_id}",
            importance=0.5,
            tier=tier,
        )

    def log_episode(self, content: str, importance: float = 0.5) -> str:
        """Log a concise daily episodic memory (session-level event)."""
        return self.longterm.store(
            content=content,
            category="episode",
            tags=["episodic", "daily_log"],
            source=f"session:{self.session.session_id}",
            importance=importance,
            tier="short",
        )

    def add_curated_memory(
        self,
        content: str,
        category: str = "insight",
        tags: list[str] | None = None,
        importance: float = 0.8,
    ) -> str:
        """Store a human- or agent-curated long-term memory to be synced to MEMORY.md."""
        return self.longterm.store(
            content=content,
            category=category,
            tags=tags or ["curated"],
            source=f"session:{self.session.session_id}",
            importance=importance,
            tier="long",
        )

    def sync_memory_md(self) -> Path | None:
        """Write curated long-tier memories to MEMORY.md in the project root."""
        path = self.config.memory_md_path
        if not path:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        long_entries = self.longterm.list_all(limit=9999, alive_only=True)
        long_entries = [e for e in long_entries if e.get("tier") == "long"]
        long_entries.sort(key=lambda e: e.get("importance", 0.0), reverse=True)
        lines = ["# MEMORY.md — Curated long-term memory", ""]
        for e in long_entries:
            tag_str = ", ".join(json.loads(e.get("tags", "[]")) or [])
            lines.append(f"## [{e.get('category', 'insight')}] {tag_str}")
            lines.append(f"- {e.get('content', '')}")
            lines.append(f"- source: {e.get('source', '')}")
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def load_memory_md(self) -> list[dict[str, Any]]:
        """Load curated memories from MEMORY.md if present."""
        path = self.config.memory_md_path
        if not path or not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        text = path.read_text(encoding="utf-8")
        current: dict[str, Any] | None = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("## ["):
                if current:
                    entries.append(current)
                parts = line[4:].split("]", 1)
                category = parts[0] if parts else "insight"
                tag_str = parts[1].strip() if len(parts) > 1 else ""
                current = {
                    "category": category,
                    "tags": [t.strip() for t in tag_str.split(",") if t.strip()],
                    "content": "",
                    "source": "MEMORY.md",
                }
            elif current and line.startswith("- "):
                body = line[2:]
                if body.startswith("source: "):
                    current["source"] = body[8:]
                elif not current["content"]:
                    current["content"] = body
                else:
                    current["content"] += "\n" + body
        if current:
            entries.append(current)
        return entries

    def _extract_topics(self) -> str:
        """Simple topic extraction from messages."""
        topics = set()
        for msg in self.session.messages:
            if isinstance(msg.content, str):
                text = msg.content.lower()
                for keyword in ["vasp", "lammps", "dft", "md", "band", "phonon", "defect", "surface"]:
                    if keyword in text:
                        topics.add(keyword)
        return ", ".join(sorted(topics)) if topics else "general"

    # --- Utility ---

    def get_session_summary(self) -> dict[str, Any]:
        return self.session.to_dict()

    def clear_session(self) -> None:
        old_id = self.session.session_id
        self.session = SessionContext()
        # Preserve link to old session in long-term memory
        self.longterm.store(
            content=f"New session started. Previous session: {old_id}",
            category="conversation",
            tags=["session_transition"],
            importance=0.3,
            tier="short",
        )

    def stats(self) -> dict[str, Any]:
        all_entries = self.longterm.list_all(limit=9999, alive_only=True)
        return {
            "session_id": self.session.session_id,
            "session_messages": len(self.session.messages),
            "session_tool_calls": len(self.session.tool_calls),
            "longterm_entries": len(all_entries),
            "tier_counts": {
                "short": sum(1 for e in all_entries if e.get("tier") == "short"),
                "mid": sum(1 for e in all_entries if e.get("tier") == "mid"),
                "long": sum(1 for e in all_entries if e.get("tier") == "long"),
            },
        }
