"""Context builder — extracts prompt assembly logic from HuginnAgent.

This module isolates the responsibility of building the dynamic context
that gets injected into each LLM call:

* Long-term memory recall (with research-log conjectures)
* Project knowledge graph queries
* Domain knowledge base (first-principles reference) retrieval
* Conversation tree history reconstruction
* Persona emotion tracking

By extracting these into a standalone class we reduce the ``HuginnAgent``
god-class footprint and make the context-building pipeline independently
testable.

The agent delegates to a ``ContextBuilder`` instance — all public methods
on the agent that previously built context now forward here.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# v14 Task 3: semantic overlap 用 TF-IDF + cosine. sklearn 已装就用, 没装 stdlib 兜.
# ponytail: 不引新依赖. sklearn 在 pyproject 是 optional (rag/all), env 装了就用.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine
    _SKLEARN_OK = True
except ImportError:  # pragma: no cover
    _SKLEARN_OK = False


def _flatten_replay_content(content: Any) -> str:
    """历史回放内容扁平化: block list → 纯文本.

    file/image block (base64 PDF 等) 换文本占位 — OpenAI 兼容 API 不收
    file variant (messages[i] 400), 且每轮重发几 MB base64 是纯 token 浪费.
    ponytail: 树节点里的原始 base64 仍在内存 (天花板: 内存膨胀), 升级路径
    是 ConversationTree.add_message 落节点时就 sanitize.
    """
    if not isinstance(content, list):
        return content if isinstance(content, str) else str(content or "")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            btype = block.get("type", "")
            if btype == "text":
                parts.append(str(block.get("text", "")))
            else:
                fname = block.get("filename") or block.get("name") or "unnamed"
                parts.append(f"[{btype or 'attachment'} omitted: {fname}]")
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p)


def _compute_semantic_overlap(text_a: str, text_b: str) -> float:
    """TF-IDF + cosine similarity between two free-form texts.

    v14 Task 3 supported_ratio 的核心算子: 算当前 entry.attempted 跟历史
    entry.evidence 的语义重叠. >0.7 视为支持 (spec v14 supported_ratio 阈值).

    实现选择 (按 spec SubTask 3.1):
    - sklearn 已装 → TfidfVectorizer + cosine_similarity (默认 word-level, 2+ char token)
    - sklearn 缺 → stdlib Counter (TF) + math.sqrt (cosine), 无 IDF

    空字符串 / 无有效 token → 0.0. 返回 float ∈ [0, 1].

    ponytail: 短文本 TF-IDF cosine 天然偏低 — 两段文本只共享 1-2 个 token 时
      cosine 常 < 0.3, >0.7 阈值要求文本共享大部分 content token. 升级路径:
      加 stemmer (Porter / Snowball) 或 char n-gram 提升短文本召回, 但需新依赖
      或自定义 tokenizer, 当前先按 spec 严格走 word-level TF-IDF.
    """
    if not text_a or not text_b:
        return 0.0
    a = text_a.strip()
    b = text_b.strip()
    if not a or not b:
        return 0.0

    if _SKLEARN_OK:
        try:
            vec = TfidfVectorizer()
            X = vec.fit_transform([a, b])
            # 矩阵只有 2 行, [0,1] 是 a vs b 的 cosine.
            return float(_sklearn_cosine(X)[0, 1])
        except ValueError:
            # TfidfVectorizer 在空 vocab (全 stop word / 全 1-char) 时抛 ValueError,
            # 走 stdlib 兜底.
            pass
        except Exception:
            logger.debug("sklearn tfidf cosine failed, fallback to stdlib", exc_info=True)

    # stdlib fallback: Counter (TF, 无 IDF) + math.sqrt 算 cosine.
    # token_pattern 跟 sklearn 默认一致: 2+ char word token.
    ta = re.findall(r"\b\w\w+\b", a.lower())
    tb = re.findall(r"\b\w\w+\b", b.lower())
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    if dot == 0:
        return 0.0
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def format_kb_chunks(
    chunks: list[dict[str, Any]],
    *,
    memory_recall_fn: Any = None,
    with_image_ref: bool = True,
    max_chars: int = 800,
    cross_ref_top_k: int = 2,
) -> str:
    """KB chunks → prompt body 文本. engine 和 ContextBuilder 共享, 消除双路径漂移.

    参数:
        chunks: KB query 返回的 chunk 列表, 每项含 text/metadata.
        memory_recall_fn: 可选, 签名 (query: str, max_entries: int) -> str.
            传入则对 top N chunk 做 KB→memory cross-ref (二次召回).
        with_image_ref: True 时 metadata.image_ref 视觉压缩页引用拼进去.
        max_chars: 单 chunk 文本截断长度.
        cross_ref_top_k: 对前 N 个 chunk 做 memory cross-ref.

    返回 body 文本 (无外壳), 调用方自己包 "### Domain Knowledge Context" 之类.
    空 chunks / 全空文本 → 返回 "".
    ponytail: 不包外壳 — engine 和 ContextBuilder 的外壳文案不同 (engine 是
      "ground your hypothesis and plan", chat 是 "ground your answer"), 强行
      统一会丢语义. 升级路径: 加外壳参数, 但要确认两边都接受.
    """
    if not chunks:
        return ""
    lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"[{i}] {text}")

        # 视觉压缩页引用: metadata 带 image_ref 的 chunk 是整页视觉压缩,
        # 多模态 agent 可以直接读图. 文本里只加引用路径, 不加载图像.
        if with_image_ref:
            meta = c.get("metadata") or {}
            img_ref = meta.get("image_ref") if isinstance(meta, dict) else None
            if img_ref:
                lines.append(f"    ↳ [视觉压缩页 图像]: {img_ref}")

        # KB chunk → memory cross-ref: 用 chunk 文本做 query 召回相关长期记忆,
        # 建立 memory↔KB 双向链接. 只对前 N 个 chunk 做, 避免每 chunk 都查一次.
        if memory_recall_fn is not None and i <= cross_ref_top_k:
            try:
                related = memory_recall_fn(text[:200], max_entries=1)
                if related:
                    lines.append(f"    ↳ Memory: {related[:200]}")
            except Exception:
                logger.debug("recall for prompt failed", exc_info=True)

    if not lines:
        return ""
    return "\n".join(lines)


def load_meta_trace_text(workspace: str | Path, last_n: int = 5) -> str:
    """读 .huginn/meta_trace.jsonl, 取最近 last_n 条拼成结构化摘要文本.

    engine (autoloop) 和 ContextBuilder 共享 — 之前 engine 只写不读, 导致
    autoloop 长轨迹里 agent 看不到自己上轮蒸馏的结构化历史. 文件不存在/
    空/读失败都返回 "".

    ponytail: 每字段截 200 字符, 总量约 2K tokens. 不做 schema 校验, 旧 entry
      补默认值. 升级路径: pydantic model + version tag + 按 darwin_score top-K.
    """
    trace_path = Path(workspace) / ".huginn" / "meta_trace.jsonl"
    if not trace_path.exists():
        return ""
    try:
        with trace_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""
    if not lines:
        return ""

    # 取最后 last_n 条, 倒序展示 (最新在前)
    recent = lines[-last_n:][::-1]
    entries: list[dict] = []
    for line in recent:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 旧 entry 缺 simplicial complex 字段, 补默认值不阻塞读取.
        if not {"simplex_id", "cochain_type", "domain", "task_id"}.issubset(e.keys()):
            e.setdefault("darwin_score", 0.0)
            e.setdefault("supported_ratio", 0.0)
            e.setdefault("simplex_id", None)
            e.setdefault("cochain_type", "legacy")
            e.setdefault("domain", "unknown")
            e.setdefault("task_id", "legacy")
        entries.append(e)

    if not entries:
        return ""

    def _short(s: Any, n: int = 200) -> str:
        s = str(s or "").strip().replace("\n", " ")
        return s[:n] + ("…" if len(s) > n else "")

    out = ["### Research Trace (recent iterations, newest first)"]
    for e in entries:
        it = e.get("iteration", "?")
        ds = e.get("darwin_score", "?")
        sr = e.get("supported_ratio", "?")
        out.append(f"[iter {it}] darwin={ds} supported={sr}")
        att = _short(e.get("attempted", ""))
        if att:
            out.append(f"  attempted: {att}")
        fnd = _short(e.get("found", ""))
        if fnd:
            out.append(f"  found: {fnd}")
        ev = e.get("evidence") or []
        if isinstance(ev, list) and ev:
            ev_text = "; ".join(_short(x, 100) for x in ev[:3])
            out.append(f"  evidence: {ev_text}")
        lim = e.get("limitations") or []
        if isinstance(lim, list) and lim:
            lim_text = "; ".join(_short(x, 150) for x in lim[:2])
            out.append(f"  limitations: {lim_text}")
        art = e.get("artifacts") or []
        if isinstance(art, list) and art:
            art_text = ", ".join(_short(x, 80) for x in art[:5])
            out.append(f"  artifacts: {art_text}")
        hint = _short(e.get("next_hint", ""), 250)
        if hint:
            out.append(f"  next_hint: {hint}")
    out.append("### End Research Trace")
    return "\n".join(out)


class ContextBuilder:
    """Builds the dynamic context (memory, KG, KB, emotion) for each turn.

    Parameters
    ----------
    memory_manager
        The agent's :class:`MemoryManager` for long-term recall.
    workspace
        Path to the workspace directory (for KG/KB initialization).
    kg_enabled / kb_enabled
        Whether the knowledge graph / knowledge base are active.
    kg_depth / kg_top_k
        Knowledge graph query parameters.
    emotion_tracker
        Optional persona emotion tracker (may be ``None``).
    checkpointer
        The LangGraph checkpointer (if active — affects history inclusion).
    conversation_tree
        The agent's conversation branch tree (for history reconstruction).
    cache_builder
        The :class:`PromptCacheBuilder` for assembling final message lists.
    """

    def __init__(
        self,
        memory_manager: Any,
        workspace: str | Path,
        *,
        kg_enabled: bool = False,
        kb_enabled: bool = False,
        kg_depth: int = 1,
        kg_top_k: int = 5,
        emotion_tracker: Any | None = None,
        checkpointer: Any | None = None,
        conversation_tree: Any | None = None,
        cache_builder: Any | None = None,
    ) -> None:
        self.memory = memory_manager
        self.workspace = str(workspace)
        self.kg_enabled = kg_enabled
        self.kb_enabled = kb_enabled
        self.kg_depth = kg_depth
        self.kg_top_k = kg_top_k
        self.emotion_tracker = emotion_tracker
        self.checkpointer = checkpointer
        self._conversation_tree = conversation_tree
        self._cache_builder = cache_builder

        # Lazy-init caches
        self._kg: Any | None = None
        self._kb: Any | None = None

    # ── Memory text ────────────────────────────────────────────────

    def build_memory_text(self, query: str | None = None) -> str:
        """Recall relevant long-term memory + research-log conjectures.

        Returns a formatted string suitable for the prompt tail.
        """
        if not query:
            # 没传 query 时, 优先用当前请求的 user message (contextvars 隔离,
            # 并发安全). 取不到才回退到领域默认串 — 保留旧行为, 测试和无 ctx 场景
            # 不会挂. ponytail: 不在 ContextBuilder 上塞 _current_query 属性,
            # 那会和 core.py 的 _current_user_message 一样被并发覆盖.
            try:
                from huginn.utils.session_context import get_user_message
                query = get_user_message() or "materials science computation"
            except Exception:
                query = "materials science computation"
        parts: list[str] = []
        try:
            mem = self.memory.recall_for_prompt(query, max_entries=3)
            if mem:
                parts.append(mem)
        except Exception:
            logger.warning("memory.recall failed in context injection", exc_info=True)

        # Inject verified/in-progress conjectures from the research log
        try:
            from huginn.research_log import get_research_log
            log = get_research_log()
            verified = log.list_by_status("verified", limit=3)
            in_progress = log.list_by_status("in_progress", limit=2)
            if verified or in_progress:
                lines = ["### Research Log (recent conjectures)"]
                for r in verified:
                    lines.append(f"- [verified] {r.title}")
                for r in in_progress:
                    lines.append(f"- [in_progress] {r.title}")
                lines.append("### End Research Log")
                parts.append("\n".join(lines))
        except Exception:
            logger.warning("research_log read failed", exc_info=True)

        return "\n\n".join(parts) if parts else ""

    # ── Knowledge graph ────────────────────────────────────────────

    def build_kg_text(self, query: str) -> str:
        """Query the project knowledge graph and format results."""
        if not self.kg_enabled:
            return ""
        try:
            from huginn.kg.graph import ProjectKnowledgeGraph

            if self._kg is None:
                self._kg = ProjectKnowledgeGraph(Path(self.workspace) / ".huginn")
            result = self._kg.query(query, depth=self.kg_depth, top_k=self.kg_top_k)
            nodes = {n["id"] for n in result.get("nodes", [])}
            if not nodes:
                return ""
            text = self._kg.to_text(nodes)
            if not text:
                return ""
            return (
                "### Project Knowledge Context\n"
                "The following project-specific facts and relationships may help:\n"
                f"{text}\n"
                "### End Project Knowledge Context"
            )
        except Exception:
            return ""

    def build_episode_history_text(
        self, kg: Any, current_step: int, look_back: int = 3
    ) -> str:
        """格式化当前 step 的前驱 episode 路径为 context 文本.

        kg 是 ProjectKnowledgeGraph 实例 (外部传入, 不走 self._kg — caller
        可能想用不同 workspace 的 kg 或 mock). kg 为 None / 方法缺失 / 路径
        空都返回 "".
        ponytail: 简单切片取最近 look_back 条, 不做相关性排序 — 升级路径
        是按 darwin_score / embedding 相似度排序后取 top-K.
        """
        if kg is None:
            return ""
        try:
            fn = getattr(kg, "query_episode_path", None)
            if not callable(fn):
                return ""
            episodes = fn(current_step, direction="backward")
            if not episodes:
                return ""
            # ponytail: 简单切片 (天花板: 路径已按 step_id 升序, 切片只能拿到
            # 最近 N 步, 不是最相关的 N 步). 升级路径见 docstring.
            recent = episodes[-look_back:] if look_back and look_back > 0 else episodes
            lines = ["历史相似步骤:"]
            for ep in recent:
                sid = ep.get("step_id", "?")
                att = ep.get("attempted", "") or ""
                fnd = ep.get("found", "") or ""
                res = ep.get("result", "") or ""
                lines.append(
                    f"- step {sid}: attempted={att}, found={fnd}, result={res}"
                )
            return "\n".join(lines)
        except Exception:
            logger.debug("build_episode_history_text failed", exc_info=True)
            return ""

    def build_failure_cause_text(self, kg: Any, failed_step_id: int) -> str:
        """格式化失败 episode 的因果链为 context 文本.

        kg 为 None / 方法缺失 / 因果链空都返回 "". query_failure_cause 只在
        目标 episode result=='failed' 时返回非空, 所以这里不用再判结果.
        ponytail: 纯文本拼接, 不做 LLM 摘要 — 升级路径是 LLM 把因果链改写
        成可读的根因分析建议.
        """
        if kg is None:
            return ""
        try:
            fn = getattr(kg, "query_failure_cause", None)
            if not callable(fn):
                return ""
            causes = fn(failed_step_id)
            if not causes:
                return ""
            lines = ["失败原因链:"]
            for ep in causes:
                sid = ep.get("step_id", "?")
                att = ep.get("attempted", "") or ""
                fnd = ep.get("found", "") or ""
                lines.append(f"- step {sid}: attempted={att}, found={fnd}")
            return "\n".join(lines)
        except Exception:
            logger.debug("build_failure_cause_text failed", exc_info=True)
            return ""

    # ── Domain knowledge base ──────────────────────────────────────

    def build_kb_text(self, query: str) -> str:
        """Query the domain knowledge base (vector retrieval).

        Also performs cross-reference: when KB chunks are found, their
        text is used as a secondary query to recall related memories,
        creating a memory↔KB cross-reference loop.
        """
        if not self.kb_enabled:
            return ""
        try:
            if self._kb is None:
                from huginn.knowledge.store import get_knowledge_base
                self._kb = get_knowledge_base(self.workspace)
            if self._kb.count() == 0:
                return ""
            chunks = self._kb.query(query, top_k=5)
            if not chunks:
                return ""
            # C1: 共享 format_kb_chunks, 跟 engine 走同一条格式化路径 — 消除双路径漂移.
            recall_fn = (
                self.memory.recall_for_prompt if self.memory is not None else None
            )
            body = format_kb_chunks(
                chunks,
                memory_recall_fn=recall_fn,
                with_image_ref=True,
                cross_ref_top_k=2,
            )
            if not body:
                return ""
            return (
                "### Domain Knowledge Context\n"
                "The following first-principles reference chunks may ground your answer. "
                "Cite the source numbers when relevant.\n"
                f"{body}\n"
                "### End Domain Knowledge Context"
            )
        except Exception:
            return ""

    # ── Emotion ────────────────────────────────────────────────────

    def build_emotion_text(self, message: str) -> str | None:
        """Update persona emotional trajectory and return mood context."""
        if self.emotion_tracker is None:
            return None
        self.emotion_tracker.update_from_message(message, source="user")
        return self.emotion_tracker.context_prompt()

    # ── Conversation history ────────────────────────────────────────

    def conversation_tree_to_messages(self) -> list[Any]:
        """Convert the active conversation path to LC messages.

        Excludes the last node (the current user message being handled).
        """
        if self._conversation_tree is None:
            return []

        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        messages: list[Any] = []
        path = self._conversation_tree.active_path()
        # 先收集每个 node 的 (role, tool_calls, tool_call_id), 用于 dangling 检测.
        # dangling = AIMessage(tool_calls) 后续没有对应的 ToolMessage (timeout/异常
        # 中断导致工具结果没回写). DeepSeek 严格校验 dangling tool_calls → 400,
        # Step 3 用同 thread 调 agent.chat 直接挂. 这里剥掉 dangling tool_calls,
        # 保留 content 作为纯 AIMessage. ponytail: lookahead 不改树, 只改重建结果.
        # 升级路径: 修 streaming.py 写树时保证 tool_calls + ToolMessage 原子提交.
        node_meta_list: list[dict] = []
        for node_id in path[:-1]:
            node = self._conversation_tree.get_node(node_id)
            if node is None:
                node_meta_list.append({})
                continue
            meta = node.metadata or {}
            node_meta_list.append({
                "role": node.role,
                "content": _flatten_replay_content(node.content),
                "tool_calls": meta.get("tool_calls"),
                "tool_call_id": meta.get("tool_call_id", ""),
                "name": meta.get("name"),
                "node_id": node_id,
            })

        # 收集所有 ToolMessage 的 tool_call_id, 用于判断 AIMessage 的 tool_calls 是否 dangling
        answered_call_ids: set[str] = set()
        for m in node_meta_list:
            if m.get("role") == "tool":
                tc_id = m.get("tool_call_id")
                if tc_id:
                    answered_call_ids.add(tc_id)

        for m in node_meta_list:
            if not m:
                continue
            role = m.get("role")
            msg_id = f"ct_{m['node_id']}"
            flat = m.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=flat, id=msg_id))
            elif role == "assistant":
                tool_calls = m.get("tool_calls")
                if tool_calls:
                    # 剥掉 dangling: 没有对应 ToolMessage 的 tool_call 不发给 LLM
                    kept = [tc for tc in tool_calls if tc.get("id") in answered_call_ids]
                    if kept:
                        messages.append(
                            AIMessage(content=flat, tool_calls=kept, id=msg_id)
                        )
                    else:
                        # 全部 dangling → 退化为纯 AIMessage (content 可能含已展示给用户的工具调用说明)
                        messages.append(AIMessage(content=flat, id=msg_id))
                else:
                    messages.append(AIMessage(content=flat, id=msg_id))
            elif role == "system":
                messages.append(SystemMessage(content=flat, id=msg_id))
            elif role == "tool":
                messages.append(
                    ToolMessage(
                        content=flat,
                        tool_call_id=m.get("tool_call_id", ""),
                        name=m.get("name"),
                        id=msg_id,
                    )
                )
        return messages

    # ── Plan & session continuity ──────────────────────────────────

    def build_plan_text(self, session_state=None) -> str:
        """Inject active plan context so the LLM knows where we are in the plan.

        Also injects L1 coordinates even without an active plan — they survive
        context compression and tell the model where we are structurally.
        """
        parts = []

        # L1 coordinates always injected (they're the structural breadcrumb)
        l1 = getattr(session_state, "l1_coordinates", "") if session_state else ""
        if l1:
            parts.append("### Structural Coordinates (L1)")
            parts.append(l1)
            parts.append("### End Structural Coordinates")

        if session_state and getattr(session_state, "active_plan_id", None):
            parts.append("### Current Plan")
            parts.append(f"Objective: {session_state.active_plan_objective}")
            parts.append(f"Step: {session_state.active_plan_step_index + 1}")
            parts.append(f"Cognitive mode: {session_state.cognitive_mode.value}")
            parts.append("### End Current Plan")

        return "\n".join(parts) if parts else ""

    def build_subgoal_text(self, session_state=None) -> str:
        """Inject active sub-goal constraints from /subgoal command."""
        if session_state is None:
            return ""
        sub_goals = getattr(session_state, "_sub_goals", [])
        if not sub_goals:
            return ""
        lines = ["### Active Sub-goal Constraints"]
        for i, sg in enumerate(sub_goals, 1):
            lines.append(f"{i}. {sg}")
        lines.append("### End Sub-goal Constraints")
        return "\n".join(lines)

    def build_goal_text(self, session_state=None) -> str:
        """Inject active persistent goal from GoalStore."""
        try:
            from huginn.autoloop.goal_store import get_goal_store
            store = get_goal_store()
            active = store.get_active()
            if not active:
                return ""
            lines = [
                f"### Persistent Goal (iter {active.iteration})",
                f"Goal: {active.text}",
            ]
            if active.sub_goals:
                lines.append("Sub-goal constraints:")
                for i, sg in enumerate(active.sub_goals, 1):
                    lines.append(f"  {i}. {sg}")
            lines.append("### End Persistent Goal")
            return "\n".join(lines)
        except Exception:
            return ""

    def build_session_continuity(self, session_state=None) -> str:
        """Inject previous session summary for cross-session continuity.

        Each new session would otherwise start blank — this surfaces
        what was discussed last time and the user's recent goals so the
        LLM can reference prior work.
        """
        if session_state is None:
            return ""

        parts = []
        if getattr(session_state, "last_session_summary", ""):
            parts.append("### Previous Session")
            parts.append(session_state.last_session_summary)
            parts.append("### End Previous Session")
        if getattr(session_state, "user_goals_history", []):
            recent_goals = session_state.user_goals_history[-5:]
            parts.append("### Your Recent Goals")
            for i, goal in enumerate(recent_goals, 1):
                parts.append(f"{i}. {goal}")
            parts.append("### End Recent Goals")
        return "\n\n".join(parts) if parts else ""

    def build_cognitive_prompt(self, session_state=None) -> str:
        """Inject the dual-mode attention prompt based on cognitive state.

        This is the practical implementation of 'singularity condensation' vs
        'axiom focus' — different system prompt additions for discovery vs
        construction modes.
        """
        if session_state is None:
            return ""
        # The cognitive prompt is stored on session_state by the agent
        # (which owns the CognitiveStateMachine). We just read it.
        prompt = getattr(session_state, "_cognitive_prompt", "")
        return prompt if prompt else ""

    def build_tool_preference_hint(self, session_state=None) -> str:
        """Inject tool preference hints based on cognitive state."""
        if session_state is None:
            return ""
        prefs = getattr(session_state, "_tool_preferences", {})
        if not prefs or (not prefs.get("prefer") and not prefs.get("deprioritize")):
            return ""
        parts = ["### Tool Preference Hint (cognitive state driven)"]
        if prefs.get("prefer"):
            parts.append(f"Prefer: {', '.join(prefs['prefer'])}")
        if prefs.get("deprioritize"):
            parts.append(f"Deprioritize: {', '.join(prefs['deprioritize'])}")
        parts.append("### End Tool Preference Hint")
        return "\n".join(parts)

    # ── Meta-Trace (Oxelra 启发) ───────────────────────────────────
    # autoloop engine 每轮把 attempted/found/evidence/limitations 蒸馏成
    # 一条结构化 entry 写到 .huginn/meta_trace.jsonl. 这里读回来拼成 prompt
    # 段, 让 agent 在长轨迹里看到结构化历史, 不靠 raw messages 携带.
    # ponytail: 文件不存在/空/读失败都返回空串, 不影响主流程.
    #   升级路径: 按 role 分文件 + 按 darwin_score 排序取 top-K.
    _meta_trace_cache: str | None = None
    _meta_trace_mtime: float = 0.0
    _meta_trace_count: int = 0

    def build_meta_trace_text(self, last_n: int = 5) -> str:
        """读 .huginn/meta_trace.jsonl, 取最近 last_n 条拼成结构化摘要.

        ponytail: mtime cache, 文件没变直接返回上次结果. 单次最多 last_n 条,
        每字段截 200 字符, 总量上限约 2K tokens. 升级: 流式 read + role 过滤.
        """
        trace_path = Path(self.workspace) / ".huginn" / "meta_trace.jsonl"
        if not trace_path.exists():
            return ""

        try:
            mtime = trace_path.stat().st_mtime
            size = trace_path.stat().st_size
        except OSError:
            return ""

        # mtime + size 双重缓存键 — mtime 秒级粒度, 加 size 防同秒多次写漏读
        cache_key = (mtime, size)
        if (
            self._meta_trace_cache is not None
            and (self._meta_trace_mtime, self._meta_trace_count) == cache_key
        ):
            return self._meta_trace_cache
        self._meta_trace_mtime = mtime
        self._meta_trace_count = size

        # C4: 走共享 load_meta_trace_text, engine 也调同一个 — 消除双路径漂移.
        text = load_meta_trace_text(self.workspace, last_n=last_n)
        self._meta_trace_cache = text
        return text

    def meta_trace_available(self) -> bool:
        """快速检查 trace 文件是否存在 — streaming.py 用它决定 compaction 强度."""
        from pathlib import Path
        return (Path(self.workspace) / ".huginn" / "meta_trace.jsonl").exists()

    _evolution_rules_cache: str | None = None
    _evolution_rules_mtime: float = 0.0

    def build_evolution_rules(self) -> str:
        """Inject learned evolution rules into context.

        Reads from the EvolutionEngine's persisted rules file and formats
        the most relevant ones as context for the LLM. This closes the
        evolution feedback loop: tool fails → rule learned → next call
        benefits from the lesson.
        """
        try:
            from pathlib import Path
            import os
            import json

            base = os.environ.get("HUGINN_CACHE_DIR", ".huginn")
            rules_path = Path(base) / "evolution_rules.json"
            if not rules_path.exists():
                return ""

            # Cache by mtime — file rarely changes during a session
            mtime = rules_path.stat().st_mtime
            if self._evolution_rules_cache is not None and mtime == self._evolution_rules_mtime:
                return self._evolution_rules_cache
            self._evolution_rules_mtime = mtime

            with rules_path.open("r", encoding="utf-8") as f:
                rules = json.load(f)
            if not rules:
                self._evolution_rules_cache = ""
                return ""

            # Only inject high-confidence rules, max 5 to keep prompt small
            relevant = [
                r for r in rules
                if r.get("confidence", 0) >= 0.5
            ][:5]
            if not relevant:
                self._evolution_rules_cache = ""
                return ""

            lines = ["### Learned Lessons (from past executions)"]
            for r in relevant:
                lines.append(
                    f"- When {r.get('trigger', '?')}: {r.get('action', '?')} "
                    f"(confidence: {r.get('confidence', 0):.0%})"
                )
            lines.append("### End Learned Lessons")
            self._evolution_rules_cache = "\n".join(lines)
            return self._evolution_rules_cache
        except Exception:
            return ""

    # ── Prospective / target chain / step eval ─────────────────────
    # 三个新 ctx 源: 前瞻意图 (第 5 类记忆), 目标链 (TargetChain), 上一步评估.
    # 都用 lazy import — metacog 模块还在并行开发, 启动期不能硬依赖.
    # 空输入一律返回空串, build_input_messages 自然跳过.

    def build_prospective_text(self, fired_intentions: list) -> str:
        """格式化已触发的 Prospective Intentions 为 context 文本.

        fired_intentions 是 list[ProspectiveIntention]. 空列表返回 "".
        """
        if not fired_intentions:
            return ""
        lines = []
        for it in fired_intentions:
            desc = getattr(it, "description", str(it))
            step = getattr(it, "source_step", "?")
            lines.append(f"- 你之前计划了 {desc}（创建于 step {step}），现在是执行的时候")
        return "## 待执行的前瞻意图\n" + "\n".join(lines)

    def build_target_chain_text(self, target_chains: list, current_step: int) -> str:
        """格式化目标链为 context 文本.

        target_chains 是 list[TargetChain]. 复用
        huginn.metacog.target_chain.format_target_chain_text.
        """
        if not target_chains:
            return ""
        from huginn.metacog.target_chain import format_target_chain_text as _fmt
        return _fmt(target_chains, current_step)

    def build_step_eval_text(self, last_evaluation) -> str:
        """格式化上一步评估反馈为 context 文本.

        last_evaluation 是 StepEvaluation 或 None. 复用
        huginn.metacog.step_evaluator.format_step_eval_text.
        """
        if last_evaluation is None:
            return ""
        from huginn.metacog.step_evaluator import format_step_eval_text as _fmt
        return _fmt(last_evaluation)

    def build_meta_agent_text(
        self,
        target_chains: list | None = None,
        last_step_evaluation: Any = None,
        tool_call_health: Any = None,
        drift_info: tuple | None = None,
    ) -> str:
        """元 Agent 视角重组 — Planner / Adviser / Reflector 三段.

        不是新组件, 只是把 TargetChain / StepEvaluation / ToolCallHealth /
        detect_drift 的输出按 PentAGI 三视角重新编排, 给 LLM 一个统一的
        "我在哪 / 偏没偏 / 工具坏没坏" 视图. 全部输入为 None/空时返回空串,
        不污染 context.

        ponytail: 视角重组不是新组件, 不新增依赖、不 import 新模块.
        升级路径: 真要独立 Reflector 组件时, 把第三段抽成
        ReflectorComponent.observe().
        """
        sections: list[str] = []

        # ── Planner: 目标链 → "拆解 N 步 / 当前位置 / 下一步该做什么" ──
        if target_chains:
            lines: list[str] = ["[Planner]"]
            total = 0
            done = 0
            next_step = ""
            for tc in target_chains:
                results = getattr(tc, "required_results", []) or []
                completed = getattr(tc, "completed_results", set()) or set()
                tid = getattr(tc, "target_id", "?")
                tgt = getattr(tc, "target", "")
                prog = getattr(tc, "progress", 0.0)
                total += len(results)
                done += sum(1 for r in results if r in completed)
                missing = [r for r in results if r not in completed]
                if missing and not next_step:
                    next_step = missing[0]
                lines.append(
                    f"- {tid}: {tgt} [{int(prog * 100)}%] missing={missing}"
                )
            if total > 0:
                lines.insert(
                    1,
                    f"拆解 {total} 步, 当前 {done}/{total}, "
                    f"下一步: {next_step or '(全部完成)'}",
                )
                sections.append("## 元 Agent 视角\n" + "\n".join(lines))

        # ── Adviser: on_track=false 或 drift_info 非 None → 漂移/策略告警 ──
        adviser: list[str] = []
        if last_step_evaluation is not None:
            on_track = getattr(last_step_evaluation, "on_track", "true")
            if on_track == "false":
                dev = getattr(last_step_evaluation, "deviation", "") or ""
                adviser.append(
                    "[Adviser] 上一步偏离目标链"
                    + (f": {dev}" if dev else "")
                    + " — 建议重审方法选择 / 补数据"
                )
        if drift_info is not None:
            is_drift, drift_msg = drift_info
            if is_drift:
                adviser.append(f"[Adviser] 漂移告警: {drift_msg}")
        if adviser:
            sections.append("## 元 Agent 视角\n" + "\n".join(adviser))

        # ── Reflector: tool_call_health.is_anomalous() → 介入建议 ──
        if tool_call_health is not None:
            is_anom = getattr(tool_call_health, "is_anomalous", None)
            if callable(is_anom) and is_anom():
                sr = getattr(tool_call_health, "success_rate", 1.0)
                rc = getattr(tool_call_health, "retry_count", 0)
                to = getattr(tool_call_health, "timeout_count", 0)
                pe = getattr(tool_call_health, "param_error_count", 0)
                sections.append(
                    "## 元 Agent 视角\n"
                    f"[Reflector] 工具调用异常: success_rate={sr:.2f}, "
                    f"retry={rc}, timeout={to}, param_err={pe}\n"
                    "介入建议: 检查工具参数 / 切换备选工具 / 暂停调用并人工介入"
                )

        return "\n\n".join(sections) if sections else ""

    def build_pmk_text(
        self,
        persona: Any = None,
        memory: Any = None,
        kb: Any = None,
        last_step_evaluation: Any = None,
    ) -> str:
        """PMK 三路立场显式呈现 — 给 LLM 看 persona/memory/knowledge 各自什么立场.

        解决问题: 之前 PMK 是隐式拼接, LLM 看不到三路各自立场, 无法判断是否一致.
        现在显式列出三路立场 + 一致性标签, 让 LLM 感知 "PMK 是否对齐".

        高阶网络视角: 三路立场是三个局部模型, Čech H¹ 检查能否粘合成全局.
        一致性标签用 _check_pmk_consistency (规则版 H¹ proxy), 不调 LLM.

        ponytail: 不新建 PMK 组件, 只加 build 方法. 三路文本任一非空就输出,
        全空返回空串. 升级路径: 一致性判定换 LLM 语义判断.
        """
        # 抽三路立场文本
        persona_text = ""
        if persona is not None:
            persona_text = str(
                getattr(persona, "description", None)
                or (persona.get("description") if isinstance(persona, dict) else "")
                or ""
            )
        memory_text = ""
        if last_step_evaluation is not None:
            pmk_fb = getattr(last_step_evaluation, "pmk_feedback", "") or ""
            # pmk_feedback 格式 "Persona: ...; Memory: ...; KB: ..." — 抓 Memory 段
            for seg in pmk_fb.split(";"):
                seg = seg.strip()
                if seg.lower().startswith("memory:"):
                    memory_text = seg[len("memory:"):].strip()
                    break
        kb_text = ""
        if kb is not None:
            kb_text = "(available)"  # kb 召回内容走 build_kb_text, 这里只标记可用性
        # 但如果 last_step_eval.pmk_feedback 有 KB 段, 优先用那个 (显式立场)
        if last_step_evaluation is not None:
            pmk_fb = getattr(last_step_evaluation, "pmk_feedback", "") or ""
            for seg in pmk_fb.split(";"):
                seg = seg.strip()
                if seg.lower().startswith("kb:"):
                    kb_text = seg[len("kb:"):].strip()
                    break

        if not (persona_text or memory_text or kb_text):
            return ""

        # 一致性标签 — 复用 task_lifecycle._check_pmk_consistency (H¹ proxy)
        consistency_label = "consistent"
        try:
            from huginn.runtime.task_lifecycle import _check_pmk_consistency
            pmk_state = {
                "persona": persona_text,
                "memory": memory_text,
                "kb": kb_text,
            }
            inconsistent, reason = _check_pmk_consistency(pmk_state)
            consistency_label = "INCONSISTENT" if inconsistent else "consistent"
        except Exception:
            pass  # import 失败或检查异常, 标 consistent 不阻塞

        lines = [
            "## PMK 循环状态",
            f"一致性: {consistency_label}",
            f"- Persona: {persona_text or '(无立场)'}",
            f"- Memory: {memory_text or '(无立场)'}",
            f"- Knowledge: {kb_text or '(无立场)'}",
        ]
        if consistency_label == "INCONSISTENT":
            lines.append(
                "- ⚠ 三路立场冲突 — 局部模型无法粘合成全局, "
                "请显式选择遵从哪一路"
            )
        return "\n".join(lines)

    # ── Full input messages ────────────────────────────────────────

    def build_input_messages(
        self,
        message: str,
        *,
        memory_text: str | None = None,
        kg_text: str | None = None,
        kb_text: str | None = None,
        include_history: bool | None = None,
        session_state: Any = None,
    ) -> list[Any]:
        """Assemble the full input message list for an LLM call.

        Combines: system prompt (via cache builder) + conversation history
        + memory + KG + KB + emotion + plan status + session continuity
        + current user message.
        """
        if memory_text is None:
            memory_text = self.build_memory_text(query=message)
        if kg_text is None:
            kg_text = self.build_kg_text(query=message)
        if kb_text is None:
            kb_text = self.build_kb_text(query=message)

        # G34: 之前 checkpointer 在时 include_history=False, 历史 langgraph 自己加,
        # compact_messages 只能修本轮新消息 → checkpoint 持久化历史从不被修剪,
        # 膨胀到 1.30 GB (报告 17 维度 3 差距 3). 改成 True 让 inputs["messages"]
        # 带完整历史, 配合 conversation_tree_to_messages 的 stable ID 防重复.
        if include_history is None:
            include_history = True

        history_messages: list[Any] | None = None
        if include_history:
            history_messages = self.conversation_tree_to_messages()

        messages = self._cache_builder.build_input_messages(
            memory_text,
            message,
            kg_text=kg_text,
            history_messages=history_messages,
            kb_text=kb_text,
        )

        # Merge all context injections into one SystemMessage to reduce
        # message overhead (each role tag costs tokens in the prompt).
        # Order: meta_trace → emotion → plan → cognitive → tool_hint → evolution → continuity
        # Meta-Trace 放最前: 长轨迹里它是 token 密度最高的历史信息, agent
        # 读 ctx_parts 顺序就是它看到上下文的顺序, trace 必须先看到.
        ctx_parts: list[str] = []

        # P3: Context Router — 根据 phase + task 语义稀疏化 context 段
        # 参考 "Diversity of information pathways drives sparsity in real-world
        # networks" (Nature Physics 2023). 信息路径多样性 D_proxy 监控.
        # 默认关, HUGINN_CONTEXT_ROUTER=1 开启 (零 LLM 成本, 纯规则).
        _router_enabled = os.environ.get("HUGINN_CONTEXT_ROUTER", "0") == "1"
        _selected: set[str] | None = None
        if _router_enabled:
            try:
                from huginn.runtime.context_router import (
                    route_context_segments, should_skip_segment,
                    log_routing_decision,
                )
                _phase = ""
                if session_state is not None:
                    _phase = getattr(session_state, "current_phase", "") or \
                             (session_state.get("current_phase", "")
                              if isinstance(session_state, dict) else "")
                _routing = route_context_segments(
                    phase=_phase, task_message=message,
                )
                _selected = set(_routing.selected)
                log_routing_decision(_routing, _phase, message)
            except Exception:
                logger.debug("ContextRouter disabled or failed, full ctx", exc_info=True)
                _selected = None

        def _keep(seg: str) -> bool:
            """P3: 未启用 router → 全塞; 启用时只塞 selected."""
            if _selected is None:
                return True
            return seg in _selected

        meta_trace_text = self.build_meta_trace_text()
        if meta_trace_text and _keep("meta_trace"):
            ctx_parts.append(meta_trace_text)

        emotion_text = self.build_emotion_text(message)
        if emotion_text and _keep("emotion"):
            ctx_parts.append(emotion_text)

        plan_text = self.build_plan_text(session_state)
        if plan_text and _keep("plan"):
            ctx_parts.append(plan_text)

        cognitive_text = self.build_cognitive_prompt(session_state)
        if cognitive_text and _keep("cognitive"):
            ctx_parts.append(cognitive_text)

        tool_hint = self.build_tool_preference_hint(session_state)
        if tool_hint and _keep("tool_hint"):
            ctx_parts.append(tool_hint)

        evolution_text = self.build_evolution_rules()
        if evolution_text and _keep("evolution"):
            ctx_parts.append(evolution_text)

        continuity_text = self.build_session_continuity(session_state)
        if continuity_text and _keep("continuity"):
            ctx_parts.append(continuity_text)

        subgoal_text = self.build_subgoal_text(session_state)
        if subgoal_text and _keep("subgoal"):
            ctx_parts.append(subgoal_text)

        goal_text = self.build_goal_text(session_state)
        if goal_text and _keep("goal"):
            ctx_parts.append(goal_text)

        if ctx_parts:
            from langchain_core.messages import SystemMessage
            messages.insert(-1, SystemMessage(
                content="\n\n".join(ctx_parts),
                id="ctx_block",
            ))

        return messages


def _selfcheck() -> None:
    """build_pmk_text selfcheck — 验证 PMK 三路立场显式呈现 + 一致性标签.

    覆盖: consistent / INCONSISTENT / 全空 / dict 兼容 / pmk_feedback 解析.
    ponytail: 用 __new__ 绕过 __init__ (避免依赖 workspace/model), 只测 build_pmk_text.
    """
    from types import SimpleNamespace

    ctx = ContextBuilder.__new__(ContextBuilder)

    # 1. 全空 → 空串
    out = ctx.build_pmk_text()
    assert out == "", f"empty pmk should return '', got: {out!r}"
    print("1. empty PMK → '' OK")

    # 2. 只有 persona → consistent
    persona = SimpleNamespace(description="recommend DFT calculation")
    out = ctx.build_pmk_text(persona=persona)
    assert "consistent" in out, f"single persona should be consistent: {out}"
    assert "Persona: recommend DFT calculation" in out, f"missing persona text: {out}"
    print("2. single persona → consistent OK")

    # 3. persona vs memory 冲突 → INCONSISTENT
    persona = SimpleNamespace(description="recommend DFT calculation")
    last_eval = SimpleNamespace(pmk_feedback="Memory: oppose DFT calculation; KB: support DFT")
    out = ctx.build_pmk_text(persona=persona, last_step_evaluation=last_eval)
    assert "INCONSISTENT" in out, f"conflict should be INCONSISTENT: {out}"
    assert "⚠" in out, f"missing warning: {out}"
    print("3. persona/memory conflict → INCONSISTENT + ⚠ OK")

    # 4. dict 兼容 (persona 是 dict)
    persona_dict = {"description": "use GNN model"}
    out = ctx.build_pmk_text(persona=persona_dict)
    assert "Persona: use GNN model" in out, f"dict persona failed: {out}"
    print("4. dict persona OK")

    # 5. pmk_feedback Memory 段解析
    last_eval = SimpleNamespace(pmk_feedback="Persona: try symbolic; Memory: avoid symbolic; KB: neutral")
    out = ctx.build_pmk_text(persona=SimpleNamespace(description="try symbolic"),
                             last_step_evaluation=last_eval)
    assert "avoid symbolic" in out, f"memory parse failed: {out}"
    print("5. pmk_feedback parse OK")

    print("context_builder.build_pmk_text selfcheck OK (5/5)")


if __name__ == "__main__":
    _selfcheck()
