"""Unified memory manager — orchestrates session and long-term memory.

Provides a single interface for all memory operations with automatic
promotion of important session data to long-term storage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huginn.memory.index import build_memory_index, get_topic_file_path
from huginn.memory.longterm import LongTermMemory
from huginn.memory.session import SessionContext, ToolCallRecord
from huginn.memory.truncation import truncate_entrypoint
from huginn.memory.types import MemoryType
from huginn.types import AgentMessage, ToolResult


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    auto_promote_to_longterm: bool = True
    promotion_importance_threshold: float = 0.6
    max_session_age_hours: float = 24.0
    enable_semantic_search: bool = True
    memory_md_path: Path | None = None
    # 主题记忆目录：未设置时回退到 memory_md_path.parent/memory 或 ~/.huginn/memory
    memory_dir: Path | None = None


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
        if (
            self.config.auto_promote_to_longterm
            and result
            and hasattr(result, "success")
            and result.success
            and tool_name in {"vasp_tool", "lammps_tool", "structure_tool"}
        ):
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
            lines.append(
                f"- [{r.get('category', 'fact')}] {r.get('content', '')}{provenance}"
            )
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
        content = (
            f"{record.tool_name}: {json.dumps(record.result.data, default=str)[:500]}"
        )
        importance = self._score_importance(record)
        self.longterm.store(
            content=content,
            category="calculation",
            tags=[record.tool_name, "auto_promoted"],
            source=f"session:{self.session.session_id}/call:{record.call_id}",
            importance=importance,
            tier="mid",
        )

    def _score_importance(self, record: ToolCallRecord) -> float:
        """Heuristic importance score for a tool result (0.0 - 1.0).

        Successful calculations with physical quantities (energy, gap, etc.)
        and failures with actionable errors are scored higher than generic
        outputs.
        """
        base = self.config.promotion_importance_threshold
        score = base
        text = ""
        if record.result:
            text = json.dumps(record.result.data, default=str).lower()
            if record.result.success:
                score += 0.1
            else:
                score += 0.05

        high_value_markers = [
            "energy",
            "band_gap",
            "converged",
            "lattice",
            "formation",
            "diffusivity",
            "conductivity",
            "elastic",
            "phonon",
            "magnetic",
        ]
        for marker in high_value_markers:
            if marker in text:
                score += 0.05

        if record.tool_name in {"vasp_tool", "lammps_tool", "structure_tool"}:
            score += 0.05

        return max(0.0, min(1.0, score))

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

    def maintenance(
        self,
        decay_per_day: float = 0.97,
        prune_threshold: float = 0.15,
        deduplicate: bool = True,
    ) -> dict[str, int]:
        """Run long-term memory maintenance (decay, prune, dedupe, expire)."""
        return self.longterm.maintenance(
            decay_per_day=decay_per_day,
            prune_threshold=prune_threshold,
            deduplicate=deduplicate,
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
        raw = "\n".join(lines)
        # 双重截断保护，避免 MEMORY.md 超出 entrypoint 限制
        truncated, line_cut, byte_cut = truncate_entrypoint(raw)
        if line_cut or byte_cut:
            truncated = truncated.rstrip() + "\n\n<!-- truncated to fit entrypoint limits -->\n"
        path.write_text(truncated, encoding="utf-8")
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
                for keyword in [
                    "vasp",
                    "lammps",
                    "dft",
                    "md",
                    "band",
                    "phonon",
                    "defect",
                    "surface",
                ]:
                    if keyword in text:
                        topics.add(keyword)
        return ", ".join(sorted(topics)) if topics else "general"

    # --- Typed topic-file memory (Claude Code memdir-style) ---

    def _get_memory_dir(self) -> Path:
        """获取主题记忆目录，按 config 优先级回退。

        优先使用 ``config.memory_dir``；否则用 ``memory_md_path`` 同级的 ``memory``
        子目录；都没有设置时回退到 ``~/.huginn/memory``。目录会按需创建。
        """
        if self.config.memory_dir:
            path = Path(self.config.memory_dir)
        elif self.config.memory_md_path:
            path = self.config.memory_md_path.parent / "memory"
        else:
            path = Path.home() / ".huginn" / "memory"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def store_typed_memory(
        self, memory_type: MemoryType, topic: str, content: str
    ) -> Path:
        """按类型和主题把记忆追加到主题文件。

        文件不存在时新建并写入标题；已存在则在末尾追加内容，重复内容会被
        自动跳过。返回写入的主题文件路径。
        """
        memory_dir = self._get_memory_dir()
        topic_file = get_topic_file_path(memory_type, topic, memory_dir)
        header = f"# [{memory_type.value}] {topic}\n\n"
        if not topic_file.exists():
            topic_file.write_text(header + content + "\n", encoding="utf-8")
            return topic_file
        existing = topic_file.read_text(encoding="utf-8")
        # 跳过重复内容，避免主题文件无限膨胀
        if content in existing:
            return topic_file
        topic_file.write_text(
            existing.rstrip() + "\n\n" + content + "\n", encoding="utf-8"
        )
        return topic_file

    def recall_typed(
        self, memory_type: MemoryType, topic: str | None = None
    ) -> list[dict[str, str]]:
        """按类型（可选指定主题）读取主题文件内容。

        ``topic`` 为空时返回该类型下所有主题文件；每个条目包含 ``topic``、
        ``path``、``content`` 三个字段。找不到任何文件时返回空列表。
        """
        memory_dir = self._get_memory_dir()
        type_dir = memory_dir / memory_type.value
        if not type_dir.exists():
            return []
        results: list[dict[str, str]] = []
        if topic:
            topic_file = get_topic_file_path(memory_type, topic, memory_dir)
            if topic_file.exists():
                results.append(
                    {
                        "topic": topic,
                        "path": str(topic_file),
                        "content": topic_file.read_text(encoding="utf-8"),
                    }
                )
        else:
            for f in sorted(type_dir.glob("*.md")):
                results.append(
                    {
                        "topic": f.stem,
                        "path": str(f),
                        "content": f.read_text(encoding="utf-8"),
                    }
                )
        return results

    def get_memory_index(self) -> str:
        """构建所有主题文件的索引文本。

        扫描 ``memory_dir`` 下所有 ``*.md`` 文件，调用 :func:`build_memory_index`
        生成带行/字节截断保护的索引内容，方便注入 entrypoint 或打印给用户。
        """
        memory_dir = self._get_memory_dir()
        topic_files = sorted(memory_dir.rglob("*.md"))
        return build_memory_index(topic_files, memory_dir)

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
