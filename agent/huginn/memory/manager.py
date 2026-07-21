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
        # LongTermMemory 优先用 config.memory_dir, 避免 ~/.huginn 在沙箱环境不可写
        # 导致 sqlite3.OperationalError. ponytail: 之前 MemoryConfig.memory_dir 被无视.
        if longterm:
            self.longterm = longterm
        elif config and config.memory_dir:
            self.longterm = LongTermMemory(db_path=config.memory_dir / "memory.db")
        else:
            self.longterm = LongTermMemory()
        self.config = config or MemoryConfig()
        self._llm = llm  # optional LLM for insight extraction

    def set_llm(self, llm: Any) -> None:
        """Attach an LLM instance for insight extraction."""
        self._llm = llm

    # --- Session memory operations ---

    def add_message(self, role: str, content: str | dict[str, Any]) -> None:
        # ARGUS: 用户输入标 source_class=user_input, 下游 PhaseGate 可降级.
        # 不动 assistant/system/tool, 那些由 tool/RAG hook 自己标.
        metadata: dict[str, Any] = {}
        if role == "user":
            metadata["source_class"] = "user_input"
        msg = AgentMessage(role=role, content=content, metadata=metadata)
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
        path: str | None = None,
    ) -> str:
        """Explicitly store a fact in long-term memory.

        path: optional hierarchical path (e.g. "materials/GaN/synthesis")
            for path-ranked recall. See LongTermMemory.store.
        """
        return self.longterm.store(
            content=content,
            category=category,
            tags=tags,
            source=f"session:{self.session.session_id}",
            importance=importance,
            tier=tier,
            path=path,
        )

    def recall(
        self,
        query: str,
        category: str | None = None,
        tier: str | None = None,
        top_k: int = 5,
        material_filter: dict[str, str] | None = None,
        path: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search long-term memory. material_filter 可按 formula/category 过滤材料记忆.

        path: optional lookup path — when set, results are re-ranked by
            _path_rank so memories at or near this path win over equally
            scoring global memories. See LongTermMemory.retrieve.
        """
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
            path=path,
        )

    def recall_for_prompt(
        self,
        query: str,
        max_entries: int = 3,
        material_filter: dict[str, str] | None = None,
    ) -> str:
        """Format recalled memories for injection into LLM prompt."""
        results = self.recall(query, top_k=max_entries, material_filter=material_filter)

        # M: typed memory 叠加在 FTS5 之上. 按 memory_type 优先级拉结构化记录,
        # 跟 FTS5 结果按 content 去重, typed 优先保留. ponytail: 不替换 FTS5,
        # 在其结果上叠加. 升级路径: FTS5 + typed 合并到同一 SQL.
        typed_results = self._recall_typed_for_prompt(max_entries)
        if typed_results:
            seen = {r.get("content", "") for r in results if r.get("content")}
            for tr in typed_results:
                if tr.get("content") and tr["content"] not in seen:
                    results.append(tr)
                    seen.add(tr["content"])

        blocks: list[str] = []
        if results:
            lines = ["## Relevant past knowledge:"]
            for r in results:
                provenance = f" ({r.get('source', '')})" if r.get("source") else ""
                # 消费 lint() 写入的 contradicts: tag, 标注矛盾条目
                conflict_warn = ""
                raw_tags = r.get("tags", "[]")
                if isinstance(raw_tags, str):
                    try:
                        tag_list = json.loads(raw_tags)
                    except (ValueError, TypeError):
                        tag_list = []
                elif isinstance(raw_tags, list):
                    tag_list = raw_tags
                else:
                    tag_list = []
                contradicts = [t for t in tag_list if isinstance(t, str) and t.startswith("contradicts:")]
                if contradicts:
                    ids = ", ".join(t.split(":", 1)[1] for t in contradicts[:3])
                    conflict_warn = f" [WARNING: conflicts with {ids}]"
                lines.append(
                    f"- [{r.get('category', 'fact')}] {r.get('content', '')}{provenance}{conflict_warn}"
                )
            blocks.append("\n".join(lines))

        # 同 session 内最近几条成功 tool call — 不带上 LLM 看不到上轮工具结果,
        # 多轮 tool 调用时会重复算/丢上下文. ponytail: 直接读 session.tool_calls,
        # 不另开存储; 失败的不召回 (对上下文是噪声).
        tool_block = self._recent_tool_results_block()
        if tool_block:
            blocks.append(tool_block)

        return "\n\n".join(blocks) if blocks else ""

    def _recent_tool_results_block(
        self, limit: int = 3, max_chars: int = 200
    ) -> str:
        """格式化 session 内最近成功的 tool_calls 成 prompt 块.

        只取最近 ``limit`` 条 tool_call, 再过滤 success=True; result 序列化后
        截断到 ``max_chars`` 避免撑爆 prompt. 没有符合条件的就返回空串.
        """
        recent = self.session.get_recent_tool_calls(limit)
        succ = [tc for tc in recent if tc.result and tc.result.success]
        if not succ:
            return ""
        # 反转让最新的排前面
        succ = list(reversed(succ))
        lines = ["## Recent tool results:"]
        for tc in succ:
            summary = ""
            if tc.result is not None and tc.result.data is not None:
                try:
                    summary = json.dumps(
                        tc.result.data, default=str, ensure_ascii=False
                    )
                except Exception:
                    summary = str(tc.result.data)
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "…"
            lines.append(f"- [{tc.tool_name}] {summary}")
        lines.append("## End Recent tool results")
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

    # --- Session snapshot (mode / csm / phase / plan 状态恢复) ---

    def save_session_snapshot(self, snapshot: dict[str, Any]) -> str:
        """保存 session 状态快照到 longterm, 下次会话可恢复 _mode / _csm / _phase.

        之前 session resume 只恢复消息历史, _mode / _csm / _phase_manager / _session_state
        全部丢, 用户感觉 agent "失忆" (mode 回 chat, plan 状态丢失).
        ponytail: 复用 longterm.store, category='session_snapshot', JSON 序列化.
        升级: 独立 sqlite store + 增量 diff, 避免每 N turn 全量存.
        """
        sid = snapshot.get("session_id", "") or "default"
        return self.longterm.store(
            content=json.dumps(snapshot, ensure_ascii=False, default=str),
            category="session_snapshot",
            tags=["session_snapshot", sid],
            source=f"session:{sid}",
            importance=0.8,
            tier="mid",
        )

    def load_session_snapshot(self, session_id: str = "") -> dict[str, Any] | None:
        """读最近一条 session_snapshot. session_id 为空则读任意最新一条."""
        try:
            entries = self.longterm.retrieve(
                query="session snapshot",
                category="session_snapshot",
                top_k=1,
            )
            if not entries:
                return None
            entry = entries[0] if isinstance(entries, list) else entries
            content = entry.get("content", "") if isinstance(entry, dict) else str(entry)
            return json.loads(content)
        except Exception:
            logger.debug("load_session_snapshot failed", exc_info=True)
            return None

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
        cluster: bool = False,
        llm_chat_fn: Any = None,
    ) -> dict[str, int]:
        """Run long-term memory maintenance (decay, prune, dedupe, expire, optional cluster)."""
        return self.longterm.maintenance(
            decay_per_day=decay_per_day,
            prune_threshold=prune_threshold,
            deduplicate=deduplicate,
            cluster=cluster,
            llm_chat_fn=llm_chat_fn,
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
        # 原子写: MEMORY.md 是 entrypoint, 半截写会让 agent 启动读到残缺上下文.
        from huginn.utils.concurrency import atomic_write_text
        atomic_write_text(path, truncated)
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
        # 原子写: 主题文件被半截写会让 recall_typed 读到残缺内容.
        # ponytail: read-modify-write 本身的并发一致性需要文件锁, 不在这次修复范围.
        from huginn.utils.concurrency import atomic_write_text
        if not topic_file.exists():
            atomic_write_text(topic_file, header + content + "\n")
            return topic_file
        existing = topic_file.read_text(encoding="utf-8")
        # 跳过重复内容，避免主题文件无限膨胀
        if content in existing:
            return topic_file
        atomic_write_text(
            topic_file, existing.rstrip() + "\n\n" + content + "\n"
        )
        return topic_file

    def recall_typed(
        self,
        memory_type: "MemoryType | str",
        topic: str | None = None,
        *,
        persona_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """按类型读取记忆. 两条路径, 按 memory_type 参数类型分发:

        - ``MemoryType`` enum (来自 types.py): 走文件 topic 路径, 配 ``topic``
          参数. 旧 API, 给 routes/memory.py 用.
        - ``str`` (如 "persona_history"): 走 SQLite typed 路径 (P12), 配
          ``persona_id`` / ``run_id`` / ``status`` / ``limit``. 新 API.

        ponytail: 同名方法靠参数类型分发, 不另起新名. 升级路径: 把文件 topic
        路径也吞到 SQLite, enum/str 分发逻辑就可以删.
        """
        if isinstance(memory_type, MemoryType):
            return self._recall_typed_file(memory_type, topic=topic)
        # P12 typed SQLite 路径
        return self._recall_typed(
            memory_type=str(memory_type),
            persona_id=persona_id,
            run_id=run_id,
            status=status,
            limit=limit,
        )

    def _recall_typed_file(
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

    # ── P12 Typed Memory (SQLite memory_type/run_id/persona_id/status) ──
    # 透传到 huginn.memory.typing, 内部方法直接走 longterm SQLite.
    # 现有 remember(content, category=...) 保持不变 (向后兼容).

    def remember_typed(
        self,
        content: str,
        memory_type: str,
        *,
        run_id: str | None = None,
        persona_id: str | None = None,
        status: str | None = None,
        importance: float = 0.5,
        tier: str = "mid",
        tags: list[str] | None = None,
        source: str = "",
        **extra: Any,
    ) -> str:
        """写 typed memory. 透传到 typing.remember_typed."""
        from huginn.memory.typing import remember_typed as _remember_typed
        return _remember_typed(
            self,
            content=content,
            memory_type=memory_type,
            run_id=run_id,
            persona_id=persona_id,
            status=status,
            importance=importance,
            tier=tier,
            tags=tags,
            source=source,
            **extra,
        )

    def record_failed_direction(
        self,
        hypothesis_text: str,
        reason: str,
        run_id: str,
        persona_id: str | None = None,
        math_concept: str = "",
    ) -> str:
        """记录失败方向. 透传到 typing.record_failed_direction."""
        from huginn.memory.typing import record_failed_direction as _rfd
        return _rfd(
            self,
            hypothesis_text=hypothesis_text,
            reason=reason,
            run_id=run_id,
            persona_id=persona_id,
            math_concept=math_concept,
        )

    def recall_failed_directions(
        self,
        limit: int = 5,
        persona_id: str | None = None,
    ) -> list[tuple[str, str, str]]:
        """查最近失败方向. 透传到 typing.recall_failed_directions."""
        from huginn.memory.typing import recall_failed_directions as _rfd
        return _rfd(self, limit=limit, persona_id=persona_id)

    def _update_typed_fields(
        self,
        entry_id: str,
        *,
        memory_type: str | None = None,
        run_id: str | None = None,
        persona_id: str | None = None,
        status: str | None = None,
    ) -> bool:
        """UPDATE memories SET memory_type/run_id/persona_id/status WHERE id=?.

        内部方法, 给 typing.remember_typed 用. 只 UPDATE 非 None 字段, 不
        覆盖已存在的值.
        """
        from datetime import datetime
        with self.longterm._connect() as conn:
            sets: list[str] = []
            params: list[Any] = []
            if memory_type is not None:
                sets.append("memory_type = ?")
                params.append(memory_type)
            if run_id is not None:
                sets.append("run_id = ?")
                params.append(run_id)
            if persona_id is not None:
                sets.append("persona_id = ?")
                params.append(persona_id)
            if status is not None:
                sets.append("status = ?")
                params.append(status)
            if not sets:
                return False
            params.append(entry_id)
            cur = conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
                tuple(params),
            )
            conn.commit()
            return cur.rowcount > 0

    # M: typed memory 在 prompt 注入里的优先级. 数值小 = 优先.
    # failed_direction 最优先 (负信号, 避免重蹈覆辙), iteration_result 次之
    # (正信号, 复用成功), cross_domain_transfer 弱正 (类比迁移). persona_history
    # 和 stable_principle 已有专门路径 (JSONL + grep), 这里冗余兜底.
    _TYPE_PRIORITY: dict[str, int] = {
        "failed_direction": 0,
        "iteration_result": 1,
        "cross_domain_transfer": 2,
        "persona_history": 3,
        "stable_principle": 4,
    }

    def _recall_typed_for_prompt(self, max_entries: int) -> list[dict]:
        """按 _TYPE_PRIORITY 拉结构化 memory, 每个 type 最多 1 条.

        给 recall_for_prompt 用, 跟 FTS5 结果叠加. 不走 _recall_typed 的
        lazy-migrate 路径 (那条返空时扫全表 NULL 行反推, 性能差).
        ponytail: 直接 SQL WHERE memory_type = ?, 不扫全表. 升级路径:
        跟 FTS5 合并到同一 SQL JOIN.
        """
        from datetime import datetime
        results: list[dict] = []
        now = datetime.now().isoformat()
        with self.longterm._connect() as conn:
            for mtype in sorted(
                self._TYPE_PRIORITY.keys(), key=lambda k: self._TYPE_PRIORITY[k]
            ):
                if len(results) >= max_entries:
                    break
                row = conn.execute(
                    "SELECT * FROM memories WHERE memory_type = ?"
                    " AND (expires_at IS NULL OR expires_at > ?)"
                    " AND archived = 0"
                    " ORDER BY last_accessed DESC LIMIT 1",
                    (mtype, now),
                ).fetchone()
                if row:
                    results.append(dict(row))
        return results

    def _recall_typed(
        self,
        memory_type: str,
        *,
        persona_id: str | None = None,
        run_id: str | None = None,
        status: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """SELECT * FROM memories WHERE memory_type=? (严格匹配 + lazy migrate).

        内部方法, 给 typing.recall_typed 用.
        1. 先严格匹配 memory_type, 拿 typed 行
        2. 如果结果为空, 扫 NULL 行用 _infer_memory_type_from_tags 反推,
           命中后 UPDATE 回填 (write-on-read), 重新查
        ponytail: 直接拼 SQL, 不走 retrieve (retrieve 走 FTS5 + vector search,
        不适合按字段精确过滤).
        """
        import json
        from datetime import datetime
        from huginn.memory.typing import _infer_memory_type_from_tags

        sql = "SELECT * FROM memories AS m WHERE memory_type = ?"
        params: list[Any] = [memory_type]
        if persona_id is not None:
            sql += " AND persona_id = ?"
            params.append(persona_id)
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        if status is not None:
            sql += " AND status = ?"
            params.append(status)
        # alive 过滤: 排除 expired + archived
        sql += (
            " AND (m.expires_at IS NULL OR m.expires_at > ?)"
            " AND m.archived = 0"
        )
        params.append(datetime.now().isoformat())
        sql += " ORDER BY m.last_accessed DESC LIMIT ?"
        params.append(limit)
        now = datetime.now().isoformat()
        with self.longterm._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            if rows:
                return [dict(r) for r in rows]
            # lazy migrate: strict 返空, 扫 NULL 行反推
            null_rows = conn.execute(
                "SELECT * FROM memories WHERE memory_type IS NULL"
                " AND (expires_at IS NULL OR expires_at > ?)"
                " AND archived = 0",
                (now,),
            ).fetchall()
            migrated = 0
            for r in null_rows:
                tags = json.loads(r["tags"] or "[]")
                inferred = _infer_memory_type_from_tags(
                    tags, r["source"] or "", r["category"] or ""
                )
                if inferred == memory_type:
                    conn.execute(
                        "UPDATE memories SET memory_type = ? WHERE id = ?",
                        (inferred, r["id"]),
                    )
                    migrated += 1
            if migrated:
                conn.commit()
                rows = conn.execute(sql, tuple(params)).fetchall()
            return [dict(r) for r in rows]

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

    # ── 模糊意图捕捉 (见 huginn/memory/intuition.py) ──────────────
    # 轻量代理, 实际逻辑在 intuition 模块. 这里暴露便捷方法让 chat() 入口
    # 直接调, 不用 import intuition 模块.

    def capture_intuition(self, message: str) -> str | None:
        """检测用户消息里的模糊意图/跨领域类比, 命中则存为 long tier 永久记忆.

        不打分, 不过滤, 不改写 — 原样保留让用户后续自己取舍.
        path=sessions/{session_id}/intuitions, hypothesis 阶段可拉回做 hint.
        """
        from huginn.memory.intuition import capture
        return capture(self, message, self.session.session_id)

    def recall_intuitions(self, top_k: int = 20) -> list[dict[str, Any]]:
        """拉回本会话的直觉信号, 给 hypothesis 阶段做 hint."""
        from huginn.memory.intuition import recall_intuitions
        return recall_intuitions(self, self.session.session_id, top_k=top_k)

    # ── Prospective memory (第 5 类: 记未来要做的事) ───────────────
    # 轻量代理到 huginn.memory.prospective.ProspectiveMemory. lazy import
    # 避免启动期循环依赖. 失败一律返回空, 不阻塞主流程.

    def _prospective_workspace(self) -> Path:
        """从 _get_memory_dir 反推 workspace 根.

        _get_memory_dir 典型返回 ws/.huginn/memory, 但 ProspectiveMemory
        自己拼 workspace/.huginn/prospective.jsonl, 直接传 .parent 会得到
        ws/.huginn → 写到 ws/.huginn/.huginn/prospective.jsonl (嵌套 .huginn).
        检测到 .huginn/memory 尾巴就上溯两级. ponytail: 启发式判断, 升级
        路径是 MemoryConfig 直接存 workspace 字段.
        """
        mem_dir = self._get_memory_dir()
        if mem_dir.parent.name == ".huginn":
            return mem_dir.parent.parent
        return mem_dir.parent

    def recall_prospective(self, current_state: dict) -> list:
        """召回已触发的 Prospective Intentions.

        current_state 含 current_step / events / variables. 返回
        list[ProspectiveIntention], 空列表表示无触发或失败.
        """
        from huginn.memory.prospective import ProspectiveMemory
        try:
            _pm = ProspectiveMemory(workspace=self._prospective_workspace())
            return _pm.scan_and_fire(current_state)
        except Exception:
            logger.debug("recall_prospective 失败", exc_info=True)
            return []

    def remember_prospective(self, intention) -> str:
        """存储 Prospective Intention.

        intention 是 ProspectiveIntention 或 dict. 返回 intention_id,
        失败返回空串.
        """
        from huginn.memory.prospective import ProspectiveMemory, ProspectiveIntention
        try:
            _pm = ProspectiveMemory(workspace=self._prospective_workspace())
            if isinstance(intention, dict):
                intention = ProspectiveIntention(**intention)
            return _pm.store(intention)
        except Exception:
            logger.debug("remember_prospective 失败", exc_info=True)
            return ""

    # ── Procedural memory (Episodic → Procedural 蒸馏, G65 / Task 19) ──
    # 同一 skill 连续 3 次成功 → 模板化蒸馏为 stable_principle.
    # ponytail: 前 20 字符归一化 + 模板 principle. 升级路径: 语义聚类 + LLM 改写.

    def distill_episodic_to_procedural(
        self, step_evaluations: list, workspace
    ) -> str | None:
        """检测连续 3 次同 skill 成功 → 蒸馏为 procedural memory.

        触发条件 (全部满足才蒸馏):
            1. step_evaluations >= 3 条
            2. 最后 3 条 attempted 归一化后完全相同 (前 20 字符, strip+lower)
            3. 最后 3 条 on_track == "true"

        返回: 蒸馏后的 principle 文本 (load_stable_principles 用文本本身做去重 key,
        所以文本即可当 id), 未触发或写入失败返回 None.
        ponytail: 关键词归一化不做语义聚类; principle 模板化, LLM 升级路径.
        """
        if len(step_evaluations) < 3:
            return None

        recent = list(step_evaluations[-3:])

        def _norm_skill(ev) -> str | None:
            a = getattr(ev, "attempted", "") or ""
            a = a.strip().lower()
            return a[:20] if a else None

        keys = [_norm_skill(ev) for ev in recent]
        if any(k is None for k in keys):
            return None
        if len(set(keys)) != 1:
            return None
        if not all(getattr(ev, "on_track", "") == "true" for ev in recent):
            return None

        attempted = (getattr(recent[-1], "attempted", "") or "").strip()
        found = (getattr(recent[-1], "found", "") or "").strip()
        if not attempted or not found:
            return None

        principle = f"当遇到 {attempted} 时, 有效方法是 {found}"
        try:
            from huginn.memory.longterm import store_stable_principle

            # source 标个来源, 方便 load_stable_principles 那边排查
            store_stable_principle(
                principle,
                source=f"episodic_distill:{str(workspace)[:64]}",
            )
        except Exception:
            logger.debug("distill_episodic_to_procedural 写入失败", exc_info=True)
            return None

        # P14: EvolutionManager.distill — flag on 时同步把 failed_direction
        # 蒸馏成 STABLE_PRINCIPLE (avoid persona X for math concept Y).
        # flag off (默认) 不调, 走原 episodic 蒸馏路径.
        import os as _os
        if _os.environ.get("HUGINN_USE_EVOLUTION_MANAGER", "0") == "1":
            try:
                from huginn.evolution.manager import EvolutionManager

                em = EvolutionManager.shared(self)
                em.distill()
            except Exception:
                logger.warning(
                    "EvolutionManager.distill failed", exc_info=True
                )
        return principle

    def recall_procedural(self, query: str, top_k: int = 3) -> list[str]:
        """召回相关 procedural memory (stable_principles).

        ponytail: 关键词包含匹配, 不分词不去停用词. 升级路径: embedding 召回.
        """
        try:
            from huginn.memory.longterm import load_stable_principles

            principles = load_stable_principles()
        except Exception:
            return []
        if not principles or not query:
            return []
        # ponytail: query 子串整体命中权重高, 再叠加词级命中数. 够用.
        query_lc = query.lower()
        query_words = [w for w in query_lc.split() if len(w) > 1]
        scored: list[tuple[int, str]] = []
        for p in principles:
            p_lc = p.lower()
            hits = 2 if query_lc in p_lc else 0
            for w in query_words:
                if w in p_lc:
                    hits += 1
            if hits > 0:
                scored.append((hits, p))
        scored.sort(key=lambda x: -x[0])
        return [p for _, p in scored[:top_k]]


# === 自检 ===
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    # 把 agent/ 加到 sys.path, 让 huginn.* 可导入
    _AGENT_ROOT = Path(__file__).resolve().parents[2]
    if str(_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(_AGENT_ROOT))

    import huginn.memory.longterm as _ltmod

    # Mock store_stable_principle, 自检不写盘
    _stored: list[tuple[str, str]] = []
    _orig_store = _ltmod.store_stable_principle

    def _mock_store(principle, source="S7_self_modify"):
        _stored.append((principle, source))

    _ltmod.store_stable_principle = _mock_store
    try:
        mgr = MemoryManager()

        def _mk_ev(attempted, found, on_track="true"):
            # 不依赖 metacog.step_evaluator, SimpleNamespace 够用
            return SimpleNamespace(
                step_id=0,
                attempted=attempted,
                found=found,
                target_chain_ref=None,
                on_track=on_track,
                structure_check="not_applicable",
                evidence_quality="medium",
                deviation="",
                pmk_feedback="",
            )

        # 1) 3 次同 skill 成功 → 触发蒸馏
        _stored.clear()
        evs = [
            _mk_ev("compute formation energy", "formation energy = -3.5 eV")
            for _ in range(3)
        ]
        pid = mgr.distill_episodic_to_procedural(evs, workspace="ws")
        assert pid is not None, "3x same skill success should trigger distillation"
        assert len(_stored) == 1, f"expected 1 store, got {len(_stored)}"
        assert "compute formation energy" in _stored[0][0]
        assert "formation energy = -3.5 eV" in _stored[0][0]
        assert pid.startswith("当遇到") and "有效方法是" in pid

        # 2) 2 成功 1 失败 → 不触发
        _stored.clear()
        evs = [
            _mk_ev("compute band gap", "gap = 1.5 eV"),
            _mk_ev("compute band gap", "gap = 1.6 eV"),
            _mk_ev("compute band gap", "no result", on_track="false"),
        ]
        pid = mgr.distill_episodic_to_procedural(evs, workspace="ws")
        assert pid is None, "2 success + 1 fail should not trigger"
        assert len(_stored) == 0, "no store on mixed success/fail"

        # 3) 3 次不同 skill → 不触发
        _stored.clear()
        evs = [
            _mk_ev("compute formation energy", "ok"),
            _mk_ev("compute band gap", "ok"),
            _mk_ev("run md simulation", "ok"),
        ]
        pid = mgr.distill_episodic_to_procedural(evs, workspace="ws")
        assert pid is None, "3 different skills should not trigger"
        assert len(_stored) == 0

        # 4) 不足 3 条 → 不触发
        _stored.clear()
        evs = [_mk_ev("compute formation energy", "ok") for _ in range(2)]
        pid = mgr.distill_episodic_to_procedural(evs, workspace="ws")
        assert pid is None, "<3 evals should not trigger"

        # 5) attempted 为空 → 归一化失败, 不触发
        _stored.clear()
        evs = [_mk_ev("", "ok") for _ in range(3)]
        pid = mgr.distill_episodic_to_procedural(evs, workspace="ws")
        assert pid is None, "empty attempted should not trigger"

        print("distill self-checks passed")
    finally:
        _ltmod.store_stable_principle = _orig_store

    # === M 块 selfcheck: typed memory 叠加到 recall_for_prompt ===
    # 用临时 DB 建 typed 记录, 验证 4 件事:
    #   M1. recall_for_prompt 返回结果含 typed 记录 (memory_type 非空)
    #   M2. typed 按 _TYPE_PRIORITY 排序 (failed_direction 在 iteration_result 前)
    #   M3. FTS5 路径保留 (普通 remember 写入的内容仍能召回)
    #   M4. _recall_typed_for_prompt 不走 lazy-migrate (SQL WHERE memory_type = ?)
    import tempfile
    from huginn.memory.longterm import LongTermMemory
    from huginn.memory.typing import remember_typed, MemoryType

    tmpdir_m = tempfile.mkdtemp(prefix="huginn_m_selfcheck_")
    db_path_m = Path(tmpdir_m) / "memory.db"
    ltm_m = LongTermMemory(db_path=str(db_path_m), enable_semantic=False)
    mm_m = MemoryManager(longterm=ltm_m)

    # 写两条 typed: failed_direction + iteration_result
    # failed_direction priority=0 应该排在 iteration_result priority=1 前面
    fd_id = remember_typed(
        mm_m,
        content="[Failed Direction] hypothesis: LDA gives GaN gap > 4 eV",
        memory_type=MemoryType.FAILED_DIRECTION.value,
        run_id="run_m_fd",
        importance=0.7,
        tier="long",
    )
    ir_id = remember_typed(
        mm_m,
        content="iteration_result: PBE sol gives gap 3.4 eV (run_m_ir)",
        memory_type=MemoryType.ITERATION_RESULT.value,
        run_id="run_m_ir",
        importance=0.6,
        tier="mid",
    )
    assert fd_id and ir_id, "remember_typed should return entry_id"

    # 写一条普通 FTS5 记录 (走 mm.remember, memory_type 留 NULL)
    legacy_id = mm_m.remember(
        content="GaN formation energy from VASP = -3.5 eV",
        category="calculation",
        tags=["vasp", "GaN"],
        importance=0.5,
        tier="mid",
    )
    assert legacy_id

    # M1 + M2: recall_for_prompt 返回 typed 记录, 且按 priority 排序
    # 用一个匹配度低的 query 让 FTS5 不抢光名额, typed 才能冒上来
    out = mm_m.recall_for_prompt("anything", max_entries=5)
    assert out, "recall_for_prompt should return non-empty string"

    # typed 记录都应该出现在结果里
    assert "Failed Direction" in out, (
        f"failed_direction should appear in recall_for_prompt, got: {out[:200]}"
    )
    assert "iteration_result" in out or "run_m_ir" in out, (
        f"iteration_result should appear in recall_for_prompt, got: {out[:200]}"
    )

    # M2: priority 排序 — failed_direction 在 iteration_result 前面
    fd_pos = out.find("Failed Direction")
    ir_pos = out.find("run_m_ir")
    if ir_pos == -1:
        ir_pos = out.find("iteration_result")
    assert fd_pos != -1 and ir_pos != -1, (
        f"both typed records should be in output, fd_pos={fd_pos}, ir_pos={ir_pos}"
    )
    assert fd_pos < ir_pos, (
        f"failed_direction (priority 0) should come before iteration_result (priority 1), "
        f"fd_pos={fd_pos}, ir_pos={ir_pos}"
    )
    print("M1+M2: recall_for_prompt 含 typed 记录 + priority 排序 OK")

    # M3: FTS5 路径保留 — 普通记录仍能召回
    out_fts = mm_m.recall_for_prompt("GaN formation energy", max_entries=5)
    assert "GaN formation energy" in out_fts and "VASP" in out_fts, (
        f"FTS5 path broken: legacy record missing, got: {out_fts[:200]}"
    )
    print("M3: FTS5 path preserved OK")

    # M4: _recall_typed_for_prompt 不走 lazy-migrate (直接 SQL WHERE memory_type=?)
    # 验证方法: typed 列表里没有 NULL 行 (legacy_id 的 memory_type 是 NULL)
    typed_only = mm_m._recall_typed_for_prompt(max_entries=10)
    assert len(typed_only) >= 2, (
        f"expected >= 2 typed records, got {len(typed_only)}"
    )
    for tr in typed_only:
        mt = tr.get("memory_type")
        assert mt is not None and mt != "", (
            f"_recall_typed_for_prompt returned non-typed row, memory_type={mt!r}"
        )
        assert tr["id"] != legacy_id, (
            "_recall_typed_for_prompt returned legacy NULL row — not strict WHERE"
        )
    # priority 排序在 _recall_typed_for_prompt 里也成立
    types_in_order = [tr["memory_type"] for tr in typed_only]
    if "failed_direction" in types_in_order and "iteration_result" in types_in_order:
        assert types_in_order.index("failed_direction") < types_in_order.index("iteration_result"), (
            f"_recall_typed_for_prompt priority broken: {types_in_order}"
        )
    print("M4: _recall_typed_for_prompt strict WHERE + priority OK")

    # 清理临时 DB
    import shutil
    shutil.rmtree(tmpdir_m, ignore_errors=True)
    print("all self-checks passed (distill + M block)")
