"""Unified memory manager — orchestrates session and long-term memory.

Provides a single interface for all memory operations with automatic
promotion of important session data to long-term storage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huginn.memory.index import build_memory_index, get_topic_file_path
from huginn.memory.longterm import LongTermMemory
from huginn.memory.session import SessionContext, ToolCallRecord
from huginn.memory.truncation import truncate_entrypoint
from huginn.memory.types import MemoryType
from huginn.types import AgentMessage, ToolResult

logger = logging.getLogger(__name__)


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
        llm: Any = None,
    ):
        self.session = session or SessionContext()
        self.longterm = longterm or LongTermMemory()
        self.config = config or MemoryConfig()
        self._llm = llm  # optional LLM for insight extraction

    def set_llm(self, llm: Any) -> None:
        """Attach an LLM instance for insight extraction."""
        self._llm = llm

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

        # Auto-promote important tool results to long-term memory.
        # 以前只允许 vasp/lammps/structure 三个工具, 现在用模式匹配:
        # 任何 *_tool 或 *_runner 的成功结果都走重要性评分, 由 _score_importance
        # 决定是否真正晋升. 白名单是多余的守门——评分函数本身就是过滤器.
        if (
            self.config.auto_promote_to_longterm
            and result
            and hasattr(result, "success")
            and result.success
            and (tool_name.endswith("_tool") or tool_name.endswith("_runner"))
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
        material_filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search long-term memory. material_filter 可按 formula/category 过滤材料记忆."""
        formula = None
        cat = category
        if material_filter:
            formula = material_filter.get("formula")
            mcat = material_filter.get("category")
            if mcat:
                cat = f"material_{mcat}"
        return self.longterm.retrieve(
            query=query,
            category=cat,
            tier=tier,
            top_k=top_k,
            semantic=self.config.enable_semantic_search,
            formula=formula,
        )

    def recall_for_prompt(
        self,
        query: str,
        max_entries: int = 3,
        material_filter: dict[str, str] | None = None,
    ) -> str:
        """Format recalled memories for injection into LLM prompt."""
        results = self.recall(query, top_k=max_entries, material_filter=material_filter)
        if not results:
            return ""
        lines = ["## Relevant past knowledge:"]
        for r in results:
            provenance = f" ({r.get('source', '')})" if r.get("source") else ""
            lines.append(
                f"- [{r.get('category', 'fact')}] {r.get('content', '')}{provenance}"
            )
        return "\n".join(lines)

    # --- Cross-session continuity ---

    def load_last_session_context(self) -> dict[str, Any]:
        """Load the most recent session summary for cross-session continuity.

        Called at the start of a new session to restore context from the
        previous conversation. Returns a dict with 'summary', 'session_id',
        and 'l1_coordinates' if available.
        """
        try:
            entries = self.longterm.retrieve(
                query="session summary",
                category="conversation",
                top_k=1,
            )
            if entries:
                entry = entries[0]
                content = entry.get("content", "") if isinstance(entry, dict) else str(entry)
                # try to extract l1_coordinates from tags or content
                l1_coords = ""
                if isinstance(entry, dict):
                    tags = entry.get("tags", [])
                    if isinstance(tags, list):
                        for tag in tags:
                            if tag.startswith("l1:"):
                                l1_coords = tag[3:]
                                break
                return {
                    "summary": content,
                    "session_id": entry.get("source", "") if isinstance(entry, dict) else "",
                    "l1_coordinates": l1_coords,
                }
        except Exception:
            logger.debug("load_last_session_context 失败", exc_info=True)
        return {"summary": "", "session_id": "", "l1_coordinates": ""}

    def store_plan_progress(
        self,
        plan_id: str,
        objective: str,
        step_index: int,
        status: str,
        l1_coordinates: str = "",
    ) -> str:
        """Record plan progress in long-term memory.

        Lets the next session pick up where we left off — the agent can
        recall 'you were on step 2 of 3 for the GaN band structure calc.'
        """
        content = (
            f"Plan: {objective} | "
            f"Step: {step_index} | Status: {status}"
        )
        if l1_coordinates:
            content += f" | Position: {l1_coordinates}"
        return self.longterm.store(
            content=content,
            category="plan",
            tags=["plan_progress", plan_id],
            source=f"plan:{plan_id}",
            importance=0.7,
            tier="mid",
        )

    def load_active_plan(self) -> dict[str, Any] | None:
        """Load the most recent active plan from long-term memory.

        Called at session start to check whether an unfinished plan from
        a previous session should be resumed.
        """
        try:
            entries = self.longterm.retrieve(
                query="plan progress",
                category="plan",
                top_k=1,
            )
            if entries:
                entry = entries[0] if isinstance(entries, list) else entries
                if isinstance(entry, dict):
                    content = entry.get("content", "")
                    # Stored format: "Plan: X | Step: N | Status: S | Position: P"
                    parts = {}
                    for segment in content.split("|"):
                        segment = segment.strip()
                        if ":" in segment:
                            key, _, val = segment.partition(":")
                            parts[key.strip().lower()] = val.strip()
                    # plan_id is stored in the source field as "plan:{plan_id}"
                    plan_id = ""
                    source = entry.get("source", "") if isinstance(entry, dict) else ""
                    if source.startswith("plan:"):
                        plan_id = source[5:]
                    return {
                        "plan_id": plan_id,
                        "objective": parts.get("plan", ""),
                        "step_index": int(parts.get("step", "0")) if parts.get("step", "").isdigit() else 0,
                        "status": parts.get("status", ""),
                        "l1_coordinates": parts.get("position", ""),
                        "content": content,
                    }
        except Exception:
            logger.debug("load_active_plan 失败", exc_info=True)
        return None

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

        # ── Distilled knowledge verification loop ──────────────────
        # When a tool succeeds, check if any distilled knowledge
        # (error_lesson, success_pattern, tool_tip) is relevant to
        # this tool. If so, upgrade its verification_status to
        # "confirmed" — the knowledge was validated by real use.
        if record.result.success:
            self._verify_distilled_for_tool(record.tool_name, content)

    def _verify_distilled_for_tool(
        self, tool_name: str, result_content: str
    ) -> None:
        """Upgrade verification_status of distilled knowledge related to a
        successful tool call.

        This implements the self-correction loop: knowledge that gets
        used and leads to successful outcomes is promoted to "confirmed".
        """
        try:
            # Recall distilled knowledge related to this tool
            entries = self.longterm.retrieve(
                query=tool_name,
                category="distilled_knowledge",
                top_k=5,
            )
            for entry in entries:
                # Check if this distilled knowledge mentions the tool
                entry_content = (entry.get("content") or "").lower()
                if tool_name.lower() not in entry_content:
                    continue
                # Upgrade: touch the entry to rejuvenate TTL and
                # increment access_count. The actual verification_status
                # upgrade is handled by KnowledgeDistiller.verify_knowledge()
                # when the distiller is available.
                entry_id = entry.get("id")
                if entry_id:
                    self.longterm.touch(entry_id)
        except Exception:
            # Verification loop failure should never block tool promotion
            pass

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
        entry_id = self.longterm.store(
            content=summary,
            category="conversation",
            tags=["session_summary"],
            source=f"session:{self.session.session_id}",
            importance=0.5,
            tier=tier,
        )

        # Distill knowledge from tool calls — best-effort, never blocks the summary
        try:
            distilled = self._run_distillation()
            if distilled > 0:
                self.longterm.store(
                    content=f"Distilled {distilled} knowledge items from session {self.session.session_id}",
                    category="insight",
                    tags=["distillation", "auto"],
                    source=f"session:{self.session.session_id}",
                    importance=0.7,
                    tier="long",
                )
        except Exception:
            logger.debug("session 摘要蒸馏失败", exc_info=True)

        # LLM-based insight extraction — only runs if an LLM was wired in
        try:
            insight = self._extract_llm_insight()
            if insight:
                self.longterm.store(
                    content=insight,
                    category="insight",
                    tags=["llm_extracted", "session_insight"],
                    source=f"session:{self.session.session_id}",
                    importance=0.8,
                    tier="long",
                )
        except Exception:
            logger.debug("LLM 洞察提取失败", exc_info=True)

        return entry_id

    def _run_distillation(self) -> int:
        """Run knowledge distillation on session tool calls.

        Converts session tool-call records into the log dict format that
        KnowledgeDistiller expects, then feeds failures / successes / all
        calls into the three distill methods.  Newly distilled items are
        ingested into long-term memory so RAG can retrieve them later.

        Returns the count of newly distilled knowledge items.
        """
        try:
            from huginn.evolution.knowledge_distiller import KnowledgeDistiller

            distiller = KnowledgeDistiller()
        except Exception:
            return 0

        # Convert session tool calls to the flat log format the distiller wants
        logs: list[dict[str, Any]] = []
        for tc in self.session.tool_calls:
            success = tc.result.success if tc.result else False
            log: dict[str, Any] = {
                "tool_name": tc.tool_name,
                "success": success,
                "session_id": self.session.session_id,
                "tool_input": tc.input_args if isinstance(tc.input_args, dict) else {},
                "error_message": "",
                "software": tc.tool_name.replace("_tool", "") if tc.tool_name else "general",
                "calculation_type": "general",
            }
            if not success and tc.result and tc.result.error:
                log["error_message"] = str(tc.result.error)
            logs.append(log)

        if not logs:
            return 0

        failure_logs = [l for l in logs if not l["success"]]
        success_logs = [l for l in logs if l["success"]]

        total = 0
        total += len(distiller.distill_error_lessons(failure_logs))
        total += len(distiller.distill_success_patterns(success_logs))
        total += len(distiller.distill_tool_tips(logs))

        # Ingest distilled knowledge into long-term memory for RAG retrieval
        for dk in distiller.knowledge_base:
            if dk.verification_status == "unverified":
                self.longterm.store(
                    content=dk.content,
                    category="distilled_knowledge",
                    tags=dk.tags + [dk.source_type],
                    source=f"distiller:{dk.knowledge_id}",
                    importance=dk.confidence,
                    tier="long",
                )

        # ── Auto-ingest distilled knowledge into KnowledgeBase ────
        # This bridges Memory→KB: distilled experience (error lessons,
        # success patterns, tool tips) becomes RAG-retrievable via
        # the same ChromaDB collection that user-uploaded docs use.
        # Only confirmed or high-confidence knowledge is ingested.
        try:
            from huginn.server_core import get_context

            ctx = get_context()
            if ctx.kb is not None:
                ingested = distiller.auto_ingest_to_kb(kb=ctx.kb)
                if ingested > 0:
                    logger.info(
                        "Distilled knowledge → KB: %d chunks ingested", ingested
                    )
        except Exception:
            logger.debug("distilled→KB auto-ingest skipped", exc_info=True)

        return total

    def _extract_llm_insight(self) -> str | None:
        """Use the LLM to extract a concise insight from the session.

        Returns None when no LLM is available or the response is too short
        to be useful.
        """
        if self._llm is None:
            return None

        # Build a condensed conversation summary — cap at last 20 messages
        # to keep the prompt token count manageable.
        messages_text: list[str] = []
        for msg in self.session.messages[-20:]:
            role = msg.role if hasattr(msg, "role") else "unknown"
            content = msg.content if hasattr(msg, "content") else str(msg)
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)[:200]
            messages_text.append(f"{role}: {content}")

        if not messages_text:
            return None

        conversation = "\n".join(messages_text)
        prompt = (
            "Based on this conversation, extract 1-3 concise insights worth remembering "
            "for future sessions. Focus on: key findings, effective strategies, failed approaches, "
            "and parameter recommendations. Be specific and actionable.\n\n"
            f"Conversation:\n{conversation[:3000]}\n\n"
            "Insights (one per line, max 3 lines):"
        )

        try:
            response = self._llm.invoke(prompt)
            text = ""
            if hasattr(response, "content"):
                text = str(response.content)
            elif isinstance(response, str):
                text = response
            text = text.strip()
            if text and len(text) > 10:
                return text
        except Exception:
            logger.debug("LLM 调用失败, 跳过洞察提取", exc_info=True)
        return None

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
        long_entries = self.longterm.list_long_tier(limit=200)
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
        tier_counts = self.longterm.count_alive_by_tier()
        return {
            "session_id": self.session.session_id,
            "session_messages": len(self.session.messages),
            "session_tool_calls": len(self.session.tool_calls),
            "longterm_entries": tier_counts["total"],
            "tier_counts": {
                "short": tier_counts["short"],
                "mid": tier_counts["mid"],
                "long": tier_counts["long"],
            },
        }
