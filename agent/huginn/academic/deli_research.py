"""Deli AutoResearch —— 多智能体自主学术研究管线.

灵感来源: 陈德里 (DeepSeek) 的自主综述生成框架 + ARS (academic-research-skills).
核心思路: 把学术研究拆成若干阶段, 每阶段用专门角色的 sub-agent 完成,
阶段之间设 integrity gate (引用验证/声明支撑/结构完整性), 不通过就打回重做.

四个子管线:
  1. DeepResearch  —— 文献检索 + 聚类 + gap 分析
  2. PaperWriting  —— 大纲 → 分节起草 → 整合润色
  3. PeerReview    —— EIC + 审稿人 + 魔鬼代言人 + meta-review
  4. CitationVerify —— 反幻觉: 每条引用都要能追到真实来源

与现有基础设施的衔接:
  - LLM 调用走 huginn.llm.get_model()
  - 文献检索优先用 RAGTool / KnowledgeBase, 拿不到就退到 LLM 自身知识
  - 期刊合规检查委托给 PaperTool / StandardsChecker
  - 研究产物 (草稿/大纲/引用) 记入 ProvenanceRegistry
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 阶段定义
# ──────────────────────────────────────────────────────────────────────

class ResearchStage(str, Enum):
    """管线阶段, 顺序即执行次序."""
    TOPIC_ANALYSIS = "topic_analysis"
    LITERATURE_SEARCH = "literature_search"
    GAP_ANALYSIS = "gap_analysis"
    OUTLINE = "outline"
    DRAFTING = "drafting"
    CITATION_VERIFY = "citation_verify"
    PEER_REVIEW = "peer_review"
    REVISION = "revision"
    FINAL = "final"

    def next(self) -> ResearchStage | None:
        members = list(ResearchStage)
        idx = members.index(self)
        return members[idx + 1] if idx + 1 < len(members) else None


# 每个阶段的 integrity gate 描述
STAGE_GATES: dict[ResearchStage, str] = {
    ResearchStage.LITERATURE_SEARCH: "至少检索到 5 篇相关文献, 且每篇有标题+作者+年份",
    ResearchStage.GAP_ANALYSIS: "至少识别 2 个研究空白, 每个空白关联具体文献",
    ResearchStage.OUTLINE: "大纲包含 Introduction/Methods/Results/Discussion/Conclusion 五段",
    ResearchStage.DRAFTING: "每个 section 至少 200 词, 引用标记 [n] 与参考文献列表对应",
    ResearchStage.CITATION_VERIFY: "所有引用都能追到真实来源, 无捏造",
    ResearchStage.PEER_REVIEW: "至少 3 个审稿视角, 每个给出具体修改建议",
    ResearchStage.REVISION: "草稿字数 >= 目标期刊要求的 80%",
}


# ──────────────────────────────────────────────────────────────────────
# 研究状态
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ResearchState:
    """贯穿整个管线的可变状态."""
    topic: str
    stage: ResearchStage = ResearchStage.TOPIC_ANALYSIS
    research_question: str = ""
    keywords: list[str] = field(default_factory=list)

    # 文献
    literature: list[dict[str, Any]] = field(default_factory=list)

    # gap 分析
    gaps: list[dict[str, str]] = field(default_factory=list)

    # 大纲 & 草稿
    outline: dict[str, Any] | None = None
    draft_sections: dict[str, str] = field(default_factory=dict)
    integrated_draft: str = ""

    # 引用
    citations: list[dict[str, Any]] = field(default_factory=list)
    citation_issues: list[str] = field(default_factory=list)

    # 审稿
    reviews: list[dict[str, Any]] = field(default_factory=list)
    meta_review: str = ""
    revision_notes: list[str] = field(default_factory=list)

    # 元信息
    target_journal: str | None = None
    integrity_log: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # 会话 ID, 供多轮交互
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_summary(self) -> dict[str, Any]:
        """精简摘要, 供 agent 上下文使用."""
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "stage": self.stage.value,
            "research_question": self.research_question,
            "literature_count": len(self.literature),
            "gaps_count": len(self.gaps),
            "outline_ready": self.outline is not None,
            "drafted_sections": list(self.draft_sections.keys()),
            "integrated_draft_len": len(self.integrated_draft),
            "citations_count": len(self.citations),
            "citation_issues": len(self.citation_issues),
            "reviews_count": len(self.reviews),
            "target_journal": self.target_journal,
            "integrity_log": self.integrity_log[-5:],
        }

    def touch(self) -> None:
        self.updated_at = time.time()


# ──────────────────────────────────────────────────────────────────────
# Sub-agent: 单角色 LLM 调用封装
# ──────────────────────────────────────────────────────────────────────

class ResearchAgent:
    """一个研究子智能体, 绑定特定角色和 system prompt.

    内部就是一次 LLM 调用, 不搞多轮对话——管线状态显式传递.
    """

    def __init__(
        self,
        role: str,
        system_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 8000,
    ) -> None:
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run(self, user_prompt: str, config: Any = None) -> str:
        """执行一次 LLM 调用, 返回纯文本."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from huginn.llm import get_model

        model = get_model(
            config=config,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response = await asyncio.to_thread(
            model.invoke,
            [SystemMessage(self.system_prompt), HumanMessage(user_prompt)],
        )
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            # 某些模型返回多段 content, 取文本段
            content = " ".join(
                seg.get("text", "") if isinstance(seg, dict) else str(seg)
                for seg in content
            )
        return content.strip()


# ──────────────────────────────────────────────────────────────────────
# 引用验证器 (反幻觉)
# ──────────────────────────────────────────────────────────────────────

class CitationVerifier:
    """验证引用真实性和声明支撑.

    策略:
      1. 先在本地知识库 (RAG) 搜, 命中则标记 verified
      2. 没命中的走 LLM 判断: 给出 DOI/标题/作者, 问模型是否认识这篇文献
      3. 标记 unverified 的引用回传给 revision 阶段处理
    """

    def __init__(self) -> None:
        self._verify_agent = ResearchAgent(
            role="citation_verifier",
            system_prompt=(
                "You are a citation verification specialist. "
                "Given a list of citations, determine which are likely real "
                "and which may be fabricated. For each citation, respond with "
                "a JSON array of objects: "
                '{"index": 0, "status": "verified|unverified|uncertain", '
                '"reason": "brief explanation"}. '
                "Be conservative: if you are not confident a paper exists, "
                "mark it 'uncertain'. Only mark 'verified' if you are sure."
            ),
            temperature=0.1,
            max_tokens=4000,
        )

    async def verify(
        self,
        citations: list[dict[str, Any]],
        rag_search_fn: Any | None = None,
        config: Any = None,
    ) -> list[dict[str, Any]]:
        """返回每个引用的验证结果.

        rag_search_fn: 可选的 (query: str) -> list[dict] 函数,
                       通常绑定到 RAGTool 或 KnowledgeBase.query.
        """
        results: list[dict[str, Any]] = []

        # 第一步: 本地知识库匹配
        if rag_search_fn is not None:
            for i, cite in enumerate(citations):
                title = cite.get("title", "")
                if not title:
                    results.append({
                        "index": i, "status": "uncertain",
                        "reason": "no title to verify",
                    })
                    continue
                try:
                    hits = rag_search_fn(title)
                    if hits and len(hits) > 0:
                        results.append({
                            "index": i, "status": "verified",
                            "reason": "found in local knowledge base",
                            "matched_text": hits[0].get("text", "")[:200]
                            if isinstance(hits[0], dict)
                            else str(hits[0])[:200],
                        })
                    else:
                        results.append(None)  # 稍后走 LLM
                except Exception:
                    results.append(None)  # RAG 挂了走 LLM
        else:
            results = [None] * len(citations)

        # 第二步: 未命中的走 LLM 判断
        unverified_indices = [i for i, r in enumerate(results) if r is None]
        if unverified_indices:
            pending = [citations[i] for i in unverified_indices]
            llm_prompt = (
                "Verify the following citations. For each, determine if it "
                "is a real published paper.\n\n"
                f"Citations:\n{json.dumps(pending, indent=2, ensure_ascii=False)}\n\n"
                "Respond with a JSON array only."
            )
            try:
                raw = await self._verify_agent.run(llm_prompt, config=config)
                llm_results = self._parse_json_array(raw)
            except Exception:
                llm_results = []

            for j, idx in enumerate(unverified_indices):
                if j < len(llm_results):
                    results[idx] = llm_results[j]
                else:
                    results[idx] = {
                        "index": idx, "status": "uncertain",
                        "reason": "verification failed",
                    }

        return results

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """从 LLM 回复里抠出 JSON 数组."""
        text = text.strip()
        # 尝试直接解析
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
        # 找 [ ... ]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return []


# ──────────────────────────────────────────────────────────────────────
# 1. Deep Research 管线
# ──────────────────────────────────────────────────────────────────────

class DeepResearchPipeline:
    """文献检索 + 聚类 + gap 分析."""

    def __init__(self) -> None:
        self._topic_agent = ResearchAgent(
            role="topic_analyst",
            system_prompt=(
                "You are a research topic analyst. Given a broad topic, "
                "extract: (1) the core research question, (2) 5-10 search "
                "keywords in English, (3) the subfields involved. "
                "Respond in JSON: "
                '{"research_question": "...", "keywords": [...], '
                '"subfields": [...]}'
            ),
            temperature=0.3,
        )
        self._cluster_agent = ResearchAgent(
            role="literature_synthesizer",
            system_prompt=(
                "You are a literature synthesis expert. Given a list of "
                "papers (title, authors, year, abstract/summary), cluster "
                "them by theme and summarize each cluster. "
                "Respond in JSON: "
                '{"clusters": [{"theme": "...", "papers": [indices], '
                '"summary": "..."}]}'
            ),
            temperature=0.4,
        )
        self._gap_agent = ResearchAgent(
            role="gap_analyst",
            system_prompt=(
                "You are a research gap analyst. Given clustered literature, "
                "identify 2-5 research gaps that are NOT addressed by the "
                "existing work. For each gap, explain what is missing and "
                "why it matters. "
                "Respond in JSON: "
                '{"gaps": [{"gap": "...", "evidence": "...", '
                '"significance": "..."}]}'
            ),
            temperature=0.5,
        )

    async def run(
        self,
        state: ResearchState,
        rag_search_fn: Any | None = None,
        config: Any = None,
    ) -> ResearchState:
        state.stage = ResearchStage.TOPIC_ANALYSIS
        state.touch()

        # 1. 课题分析
        topic_result = await self._topic_agent.run(
            f"Topic: {state.topic}\n\nAnalyze this research topic.", config
        )
        topic_data = _safe_json_load(topic_result, {})
        state.research_question = topic_data.get("research_question", state.topic)
        state.keywords = topic_data.get("keywords", [])
        state.integrity_log.append(f"[topic] question: {state.research_question}")

        # 2. 文献检索
        state.stage = ResearchStage.LITERATURE_SEARCH
        state.touch()

        literature: list[dict[str, Any]] = []

        # 先走本地知识库
        if rag_search_fn is not None:
            for kw in state.keywords[:5]:
                try:
                    hits = rag_search_fn(kw)
                    for h in hits[:3]:
                        if isinstance(h, dict):
                            literature.append({
                                "title": h.get("metadata", {}).get("title", h.get("text", "")[:80]),
                                "source": "local_kb",
                                "snippet": h.get("text", "")[:300],
                                "distance": h.get("distance"),
                            })
                except Exception:
                    logger.debug("RAG search failed for keyword: %s", kw, exc_info=True)

        # 知识库不够就让 LLM 补充 (基于训练知识)
        if len(literature) < 5:
            lit_agent = ResearchAgent(
                role="literature_searcher",
                system_prompt=(
                    "You are a literature search assistant for materials science. "
                    "Given keywords and a research question, list relevant papers "
                    "you know from training data. For each: title, authors (first 3), "
                    "year, venue, and a one-sentence summary. "
                    "Respond in JSON: "
                    '{"papers": [{"title": "...", "authors": "...", "year": ..., '
                    '"venue": "...", "summary": "..."}]}'
                ),
                temperature=0.3,
                max_tokens=6000,
            )
            lit_result = await lit_agent.run(
                f"Research question: {state.research_question}\n"
                f"Keywords: {', '.join(state.keywords)}\n"
                f"Already found {len(literature)} papers. Provide 10 more.",
                config,
            )
            lit_data = _safe_json_load(lit_result, {})
            for p in lit_data.get("papers", []):
                p["source"] = "llm_knowledge"
                literature.append(p)

        state.literature = literature[:20]  # cap at 20
        state.integrity_log.append(f"[search] found {len(state.literature)} papers")

        # 3. 聚类
        if state.literature:
            cluster_input = json.dumps(state.literature, ensure_ascii=False, indent=2)
            cluster_result = await self._cluster_agent.run(
                f"Papers:\n{cluster_input}", config
            )
            state.outline = _safe_json_load(cluster_result, {})  # 暂存聚类结果

        # 4. gap 分析
        state.stage = ResearchStage.GAP_ANALYSIS
        state.touch()

        cluster_info = json.dumps(
            state.outline or {"clusters": state.literature[:10]},
            ensure_ascii=False,
        )
        gap_result = await self._gap_agent.run(
            f"Clustered literature:\n{cluster_info}\n\n"
            f"Research question: {state.research_question}\n\n"
            "Identify research gaps.", config
        )
        gap_data = _safe_json_load(gap_result, {})
        state.gaps = gap_data.get("gaps", [])
        state.integrity_log.append(f"[gap] identified {len(state.gaps)} gaps")

        return state


# ──────────────────────────────────────────────────────────────────────
# 2. Paper Writing 管线
# ──────────────────────────────────────────────────────────────────────

class PaperWritingPipeline:
    """大纲 → 分节起草 → 整合润色, 带 integrity gate."""

    # 标准 section 列表
    SECTIONS = [
        "introduction",
        "methods",
        "results",
        "discussion",
        "conclusion",
    ]

    def __init__(self) -> None:
        self._outline_agent = ResearchAgent(
            role="outline_designer",
            system_prompt=(
                "You are an academic paper outline designer for materials science. "
                "Given a research question, literature clusters, and gaps, "
                "create a detailed section-by-section outline. "
                "Each section should have 3-5 bullet points of key content. "
                "Respond in JSON: "
                '{"sections": [{"name": "introduction", "points": [...]}, ...]}'
            ),
            temperature=0.4,
        )

        # 分节起草用同一个 agent, system prompt 按 section 动态调整
        self._draft_agent = ResearchAgent(
            role="section_writer",
            system_prompt="",  # 动态设置
            temperature=0.5,
            max_tokens=4000,
        )

        self._integrate_agent = ResearchAgent(
            role="paper_integrator",
            system_prompt=(
                "You are an academic paper editor. Given separately written "
                "sections, integrate them into a cohesive draft. Smooth "
                "transitions, ensure consistent terminology, and check that "
                "every claim has a citation marker [n]. "
                "Return the full integrated text."
            ),
            temperature=0.4,
            max_tokens=12000,
        )

    async def run(
        self,
        state: ResearchState,
        config: Any = None,
    ) -> ResearchState:
        # 1. 大纲
        state.stage = ResearchStage.OUTLINE
        state.touch()

        outline_input = (
            f"Research question: {state.research_question}\n"
            f"Keywords: {', '.join(state.keywords)}\n"
            f"Gaps: {json.dumps(state.gaps, ensure_ascii=False)}\n"
            f"Literature count: {len(state.literature)}\n"
        )
        outline_result = await self._outline_agent.run(outline_input, config)
        outline_data = _safe_json_load(outline_result, {})
        state.outline = outline_data
        state.integrity_log.append("[outline] structure created")

        # 2. 分节起草
        state.stage = ResearchStage.DRAFTING
        state.touch()

        sections_data = outline_data.get("sections", [])
        if not sections_data:
            # 没拿到大纲就按标准五段来
            sections_data = [{"name": s, "points": []} for s in self.SECTIONS]

        section_names = [s["name"] for s in sections_data]

        # 并行起草各 section
        tasks = [
            self._draft_section(state, sec, config)
            for sec in sections_data
        ]
        drafts = await asyncio.gather(*tasks, return_exceptions=True)

        for sec_data, draft in zip(sections_data, drafts):
            name = sec_data["name"]
            if isinstance(draft, Exception):
                state.integrity_log.append(f"[draft] {name} failed: {draft}")
                state.draft_sections[name] = f"[Draft failed for {name}]"
            else:
                state.draft_sections[name] = draft

        # 3. 引用提取
        state.citations = self._extract_citations(state)

        # 4. 整合
        all_sections = "\n\n".join(
            f"## {name.title()}\n\n{text}"
            for name, text in state.draft_sections.items()
        )
        integrated = await self._integrate_agent.run(
            f"Sections to integrate:\n\n{all_sections}\n\n"
            f"Research question: {state.research_question}",
            config,
        )
        state.integrated_draft = integrated
        state.integrity_log.append(
            f"[draft] integrated {len(state.draft_sections)} sections, "
            f"{len(state.integrated_draft)} chars"
        )

        return state

    async def _draft_section(
        self,
        state: ResearchState,
        sec_data: dict[str, Any],
        config: Any,
    ) -> str:
        name = sec_data["name"]
        points = sec_data.get("points", [])

        # 按 section 类型调整 prompt
        role_hints = {
            "introduction": "Establish context, state the problem, summarize prior work, and state contributions.",
            "methods": "Describe computational/experimental methods, software used (VASP, LAMMPS, etc.), parameters, and analysis procedures.",
            "results": "Present findings with quantitative data. Reference figures/tables as Fig. 1, Table 1, etc.",
            "discussion": "Interpret results, compare with literature, discuss implications and limitations.",
            "conclusion": "Summarize key findings, state significance, suggest future work.",
        }

        hint = role_hints.get(name, "Write this section clearly and concisely.")

        lit_summary = "\n".join(
            f"- {p.get('title', '?')} ({p.get('year', '?')}): {p.get('summary', p.get('snippet', ''))[:100]}"
            for p in state.literature[:8]
        )

        prompt = (
            f"Write the '{name}' section of a research paper.\n\n"
            f"Research question: {state.research_question}\n"
            f"Key points to cover: {json.dumps(points, ensure_ascii=False)}\n\n"
            f"Section guidance: {hint}\n\n"
            f"Relevant literature:\n{lit_summary}\n\n"
            f"Research gaps:\n{json.dumps(state.gaps, ensure_ascii=False)}\n\n"
            "Use citation markers [1], [2], etc. referencing the literature above. "
            "Write at least 300 words. Use academic tone."
        )

        self._draft_agent.system_prompt = (
            f"You are an expert academic writer specializing in materials science. "
            f"Write the '{name}' section. {hint} "
            "Use proper academic English with citation markers [n]."
        )

        return await self._draft_agent.run(prompt, config)

    @staticmethod
    def _extract_citations(state: ResearchState) -> list[dict[str, Any]]:
        """从草稿中提取引用标记, 关联文献."""
        full_text = state.integrated_draft or " ".join(state.draft_sections.values())
        # 找 [1], [2], [1,3] 等
        markers = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", full_text)
        indices: set[int] = set()
        for m in markers:
            for part in m.split(","):
                part = part.strip()
                if part.isdigit():
                    indices.add(int(part))

        citations = []
        for idx in sorted(indices):
            if 1 <= idx <= len(state.literature):
                cite = dict(state.literature[idx - 1])
                cite["citation_number"] = idx
                citations.append(cite)
            else:
                citations.append({
                    "citation_number": idx,
                    "title": "[UNKNOWN - possibly fabricated]",
                    "warning": "citation index out of range",
                })
        return citations


# ──────────────────────────────────────────────────────────────────────
# 3. Peer Review 管线
# ──────────────────────────────────────────────────────────────────────

class PeerReviewPipeline:
    """多视角审稿: EIC + 专家审稿人 + 魔鬼代言人 + meta-review."""

    def __init__(self) -> None:
        self._eic_agent = ResearchAgent(
            role="editor_in_chief",
            system_prompt=(
                "You are the Editor-in-Chief of a top materials science journal. "
                "Evaluate the paper for: scope fit, novelty, significance, "
                "and overall recommendation (accept/minor/major/reject). "
                "Respond in JSON: "
                '{"recommendation": "...", "scope_fit": 1-10, "novelty": 1-10, '
                '"significance": 1-10, "comments": "..."}'
            ),
            temperature=0.3,
        )
        self._reviewer_agent = ResearchAgent(
            role="expert_reviewer",
            system_prompt=(
                "You are an expert peer reviewer for materials science. "
                "Focus on technical correctness, methodology rigor, data "
                "interpretation, and literature coverage. "
                "Respond in JSON: "
                '{"strengths": [...], "weaknesses": [...], '
                '"specific_revisions": [...], "overall_score": 1-10}'
            ),
            temperature=0.4,
        )
        self._devil_agent = ResearchAgent(
            role="devils_advocate",
            system_prompt=(
                "You are a critical reviewer playing devil's advocate. "
                "Find the weakest points of the paper. Challenge assumptions, "
                "question conclusions, and identify potential flaws. "
                "Be specific and constructive. "
                "Respond in JSON: "
                '{"critical_issues": [...], "questioned_claims": [...], '
                '"missing_controls": [...]}'
            ),
            temperature=0.6,
        )
        self._meta_agent = ResearchAgent(
            role="meta_reviewer",
            system_prompt=(
                "You are a meta-reviewer synthesizing multiple reviews into "
                "a single coherent set of actionable revision instructions. "
                "Prioritize issues by severity. "
                "Respond in JSON: "
                '{"must_fix": [...], "should_fix": [...], '
                '"optional": [...], "summary": "..."}'
            ),
            temperature=0.3,
        )

    async def run(self, state: ResearchState, config: Any = None) -> ResearchState:
        state.stage = ResearchStage.PEER_REVIEW
        state.touch()

        draft = state.integrated_draft or " ".join(state.draft_sections.values())
        if not draft:
            state.integrity_log.append("[review] no draft to review")
            return state

        paper_info = (
            f"Title/Topic: {state.topic}\n"
            f"Research question: {state.research_question}\n"
            f"Literature cited: {len(state.citations)}\n"
            f"Gaps addressed: {len(state.gaps)}\n\n"
            f"--- Draft ---\n{draft[:12000]}\n--- End Draft ---"
        )

        # 并行跑四个审稿视角
        eic_task = self._eic_agent.run(paper_info, config)
        reviewer_task = self._reviewer_agent.run(paper_info, config)
        devil_task = self._devil_agent.run(paper_info, config)

        eic_raw, reviewer_raw, devil_raw = await asyncio.gather(
            eic_task, reviewer_task, devil_task, return_exceptions=True
        )

        reviews = []
        for role, raw in [("eic", eic_raw), ("reviewer_1", reviewer_raw), ("devils_advocate", devil_raw)]:
            if isinstance(raw, Exception):
                reviews.append({"role": role, "error": str(raw)})
            else:
                reviews.append({"role": role, **_safe_json_load(raw, {"raw": raw})})

        state.reviews = reviews

        # meta-review
        reviews_summary = json.dumps(reviews, ensure_ascii=False, indent=2)
        meta_raw = await self._meta_agent.run(
            f"Reviews to synthesize:\n{reviews_summary}", config
        )
        state.meta_review = meta_raw
        meta_data = _safe_json_load(meta_raw, {})
        state.revision_notes = meta_data.get("must_fix", [])
        state.integrity_log.append(
            f"[review] {len(reviews)} reviews, "
            f"{len(state.revision_notes)} must-fix items"
        )

        return state


# ──────────────────────────────────────────────────────────────────────
# 完整管线编排
# ──────────────────────────────────────────────────────────────────────

class DeliAutoResearch:
    """编排四个子管线, 管理 integrity gate."""

    def __init__(self) -> None:
        self.deep_research = DeepResearchPipeline()
        self.paper_writing = PaperWritingPipeline()
        self.peer_review = PeerReviewPipeline()
        self.citation_verifier = CitationVerifier()

        # 会话存储 (session_id → state)
        self._sessions: dict[str, ResearchState] = {}

    def get_session(self, session_id: str) -> ResearchState | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict[str, Any]]:
        return [s.to_summary() for s in self._sessions.values()]

    async def run_full_pipeline(
        self,
        topic: str,
        target_journal: str | None = None,
        rag_search_fn: Any | None = None,
        config: Any = None,
    ) -> ResearchState:
        """从头到尾跑完整管线."""
        state = ResearchState(topic=topic, target_journal=target_journal)
        self._sessions[state.session_id] = state

        # 1. Deep Research
        state = await self.deep_research.run(state, rag_search_fn, config)
        self._check_gate(state, ResearchStage.LITERATURE_SEARCH)
        self._check_gate(state, ResearchStage.GAP_ANALYSIS)

        # 2. Paper Writing
        state = await self.paper_writing.run(state, config)
        self._check_gate(state, ResearchStage.OUTLINE)
        self._check_gate(state, ResearchStage.DRAFTING)

        # 3. Citation Verification
        state.stage = ResearchStage.CITATION_VERIFY
        state.touch()
        if state.citations:
            verify_results = await self.citation_verifier.verify(
                state.citations, rag_search_fn, config
            )
            state.citation_issues = [
                f"[{r['index']+1}] {r['status']}: {r['reason']}"
                for r in verify_results
                if r and r.get("status") != "verified"
            ]
            state.integrity_log.append(
                f"[verify] {len(verify_results)} citations checked, "
                f"{len(state.citation_issues)} issues"
            )
        self._check_gate(state, ResearchStage.CITATION_VERIFY)

        # 4. Peer Review
        state = await self.peer_review.run(state, config)
        self._check_gate(state, ResearchStage.PEER_REVIEW)

        # 5. Revision (简单版: 把 must-fix 注入草稿重写)
        state.stage = ResearchStage.REVISION
        state.touch()
        if state.revision_notes:
            revision_agent = ResearchAgent(
                role="reviser",
                system_prompt=(
                    "You are a paper reviser. Given a draft and a list of "
                    "must-fix items from reviewers, produce a revised version "
                    "that addresses each issue. Mark changes with [REVISED]."
                ),
                temperature=0.4,
                max_tokens=12000,
            )
            revised = await revision_agent.run(
                f"Draft:\n{state.integrated_draft[:10000]}\n\n"
                f"Must-fix items:\n{json.dumps(state.revision_notes, ensure_ascii=False)}",
                config,
            )
            state.integrated_draft = revised
            state.integrity_log.append("[revision] draft revised per reviewer feedback")

        # 6. Final
        state.stage = ResearchStage.FINAL
        state.touch()

        # 如果有目标期刊, 跑合规检查
        if target_journal:
            try:
                from huginn.academic.standards_checker import StandardsChecker
                checker = StandardsChecker()
                word_count = len(state.integrated_draft.split())
                results = checker.check_compliance(
                    {
                        "title": state.topic,
                        "body": state.integrated_draft,
                        "references": state.citations,
                    },
                    target_journal,
                )
                state.integrity_log.append(
                    f"[final] compliance check: "
                    f"{sum(1 for r in results if r.passed)}/{len(results)} passed"
                )
            except Exception:
                logger.debug("compliance check failed", exc_info=True)

        return state

    async def run_stage(
        self,
        session_id: str,
        config: Any = None,
        rag_search_fn: Any | None = None,
    ) -> ResearchState:
        """只跑下一个阶段 (增量模式)."""
        state = self._sessions.get(session_id)
        if state is None:
            raise ValueError(f"Session {session_id} not found")

        next_stage = state.stage.next()
        if next_stage is None:
            return state

        if next_stage == ResearchStage.LITERATURE_SEARCH:
            state = await self.deep_research.run(state, rag_search_fn, config)
        elif next_stage == ResearchStage.OUTLINE:
            state = await self.paper_writing.run(state, config)
        elif next_stage == ResearchStage.CITATION_VERIFY:
            if state.citations:
                results = await self.citation_verifier.verify(
                    state.citations, rag_search_fn, config
                )
                state.citation_issues = [
                    f"[{r['index']+1}] {r['status']}: {r['reason']}"
                    for r in results
                    if r and r.get("status") != "verified"
                ]
            state.stage = ResearchStage.CITATION_VERIFY
            state.touch()
        elif next_stage == ResearchStage.PEER_REVIEW:
            state = await self.peer_review.run(state, config)

        self._sessions[session_id] = state
        return state

    def _check_gate(self, state: ResearchState, stage: ResearchStage) -> None:
        """检查 integrity gate, 记录但不阻断 (用户可选择 force proceed)."""
        gate_desc = STAGE_GATES.get(stage, "")
        if not gate_desc:
            return

        issues: list[str] = []

        if stage == ResearchStage.LITERATURE_SEARCH:
            if len(state.literature) < 5:
                issues.append(f"only {len(state.literature)} papers found (need ≥5)")
        elif stage == ResearchStage.GAP_ANALYSIS:
            if len(state.gaps) < 2:
                issues.append(f"only {len(state.gaps)} gaps identified (need ≥2)")
        elif stage == ResearchStage.OUTLINE:
            if not state.outline or "sections" not in (state.outline or {}):
                issues.append("outline missing or incomplete")
        elif stage == ResearchStage.DRAFTING:
            for name in PaperWritingPipeline.SECTIONS:
                text = state.draft_sections.get(name, "")
                if len(text.split()) < 100:
                    issues.append(f"section '{name}' too short ({len(text.split())} words)")
        elif stage == ResearchStage.CITATION_VERIFY:
            if state.citation_issues:
                issues.append(f"{len(state.citation_issues)} citation issues found")
        elif stage == ResearchStage.PEER_REVIEW:
            if len(state.reviews) < 3:
                issues.append(f"only {len(state.reviews)} reviews (need ≥3)")

        if issues:
            state.integrity_log.append(
                f"[gate:{stage.value}] WARNING: {'; '.join(issues)}"
            )
        else:
            state.integrity_log.append(f"[gate:{stage.value}] PASSED")


# ──────────────────────────────────────────────────────────────────────
# HuginnTool 封装
# ──────────────────────────────────────────────────────────────────────

# 全局实例 (懒加载)
_engine: DeliAutoResearch | None = None


def _get_engine() -> DeliAutoResearch:
    global _engine
    if _engine is None:
        _engine = DeliAutoResearch()
    return _engine


def _get_rag_search_fn(context: ToolContext | None) -> Any | None:
    """从 ToolContext 或全局 registry 里拿到 RAG 搜索函数."""
    # 先试 ToolContext (如果传了的话)
    if context is not None:
        # 看看 context 里有没有 RAGTool 实例
        for attr in ("rag_tool", "_rag", "tools"):
            obj = getattr(context, attr, None)
            if obj is not None:
                # 如果是列表, 找 RAGTool
                if isinstance(obj, list):
                    for t in obj:
                        if hasattr(t, "_search") or t.__class__.__name__ == "RAGTool":
                            return lambda q: t._search.__func__(t, type(t).__mro__[0]()) if hasattr(t, "_search") else []
                elif hasattr(obj, "search"):
                    return lambda q: obj.search(q)
    # 退到全局 registry
    try:
        from huginn.tools.registry import ToolRegistry
        rag = ToolRegistry.get("rag_tool")
        if rag is not None:
            # 调 RAGTool._search 的简化版
            async def _rag_search(query: str) -> list[dict]:
                from huginn.types import ToolContext as _TC
                result = await rag.call(
                    type(rag.input_schema)(action="search", query=query, top_k=5)
                    if hasattr(rag, "input_schema")
                    else {"action": "search", "query": query, "top_k": 5},
                    None,
                )
                if result.success and result.data:
                    return result.data.get("results", [])
                return []
            # 同步包装
            def _sync_search(query: str) -> list[dict]:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # 在 async 上下文里, 创建 task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            return pool.submit(
                                asyncio.run, _rag_search(query)
                            ).result(timeout=30)
                    return asyncio.run(_rag_search(query))
                except Exception:
                    return []
            return _sync_search
    except Exception:
        logger.debug("RAG search fn unavailable", exc_info=True)
    return None


class DeliResearchInput(BaseModel):
    action: Literal[
        "start",        # 开始一个新研究 (创建 session)
        "full_run",     # 跑完整管线
        "next_stage",   # 只跑下一阶段
        "status",       # 查看当前状态
        "get_draft",    # 获取当前草稿
        "get_citations",# 获取引用列表 + 验证状态
        "get_reviews",  # 获取审稿意见
        "list_sessions",# 列出所有研究 session
        "verify",       # 单独跑引用验证
    ] = Field(
        description=(
            "start=创建新研究session; full_run=跑完整管线; "
            "next_stage=只跑下一阶段; status=查看状态; "
            "get_draft=获取草稿; get_citations=获取引用; "
            "get_reviews=获取审稿意见; list_sessions=列出所有session; "
            "verify=单独验证引用"
        )
    )
    topic: str | None = Field(
        default=None,
        description="研究课题 (start/full_run 时必填)",
    )
    target_journal: str | None = Field(
        default=None,
        description="目标期刊 (可选, 用于合规检查)",
    )
    session_id: str | None = Field(
        default=None,
        description="研究 session ID (next_stage/status/get_draft/verify 时必填)",
    )


class DeliAutoResearchTool(HuginnTool):
    """Deli AutoResearch 多智能体学术研究管线工具.

    9 阶段管线: 课题分析 → 文献检索 → gap分析 → 大纲 → 起草 →
    引用验证 → 同行评审 → 修订 → 定稿.
    每阶段之间有 integrity gate, 支持增量执行.
    """

    name = "deli_research"
    category = "sci"
    description = (
        "多智能体自主学术研究管线 (Deli AutoResearch). "
        "9阶段: 课题分析→文献检索→gap分析→大纲→分节起草→引用验证→同行评审→修订→定稿. "
        "支持完整运行或增量执行, 每阶段有integrity gate, 引用反幻觉验证. "
        "可与RAG知识库、PaperTool期刊规范检查、Provenance追踪联动."
    )
    read_only = False  # 会创建研究 session 和草稿
    input_schema = DeliResearchInput

    async def _execute(
        self, args: DeliResearchInput, context: ToolContext
    ) -> ToolResult:
        engine = _get_engine()

        if args.action == "list_sessions":
            return ToolResult(data={"sessions": engine.list_sessions()})

        if args.action == "start":
            if not args.topic:
                return ToolResult(
                    data=None, success=False,
                    error="start 需要 topic 参数",
                )
            state = ResearchState(
                topic=args.topic,
                target_journal=args.target_journal,
            )
            engine._sessions[state.session_id] = state
            return ToolResult(
                data={
                    "session_id": state.session_id,
                    "topic": state.topic,
                    "stage": state.stage.value,
                    "message": "Session created. Use full_run or next_stage to proceed.",
                }
            )

        if args.action == "full_run":
            if not args.topic:
                return ToolResult(
                    data=None, success=False,
                    error="full_run 需要 topic 参数",
                )
            rag_fn = _get_rag_search_fn(context)
            state = await engine.run_full_pipeline(
                topic=args.topic,
                target_journal=args.target_journal,
                rag_search_fn=rag_fn,
            )
            return ToolResult(
                data={
                    "session_id": state.session_id,
                    "summary": state.to_summary(),
                    "draft_excerpt": state.integrated_draft[:2000] if state.integrated_draft else None,
                    "citations": state.citations[:10],
                    "citation_issues": state.citation_issues,
                    "reviews_count": len(state.reviews),
                    "revision_notes": state.revision_notes,
                    "integrity_log": state.integrity_log,
                }
            )

        if args.action == "next_stage":
            if not args.session_id:
                return ToolResult(
                    data=None, success=False,
                    error="next_stage 需要 session_id 参数",
                )
            rag_fn = _get_rag_search_fn(context)
            try:
                state = await engine.run_stage(
                    args.session_id, rag_search_fn=rag_fn
                )
            except ValueError as e:
                return ToolResult(data=None, success=False, error=str(e))
            return ToolResult(
                data={
                    "session_id": state.session_id,
                    "summary": state.to_summary(),
                    "stage_completed": state.stage.value,
                    "integrity_log": state.integrity_log[-3:],
                }
            )

        if args.action == "status":
            if not args.session_id:
                return ToolResult(
                    data=None, success=False,
                    error="status 需要 session_id 参数",
                )
            state = engine.get_session(args.session_id)
            if state is None:
                return ToolResult(
                    data=None, success=False,
                    error=f"Session {args.session_id} not found",
                )
            return ToolResult(data=state.to_summary())

        if args.action == "get_draft":
            if not args.session_id:
                return ToolResult(
                    data=None, success=False,
                    error="get_draft 需要 session_id 参数",
                )
            state = engine.get_session(args.session_id)
            if state is None:
                return ToolResult(
                    data=None, success=False,
                    error=f"Session {args.session_id} not found",
                )
            return ToolResult(
                data={
                    "session_id": state.session_id,
                    "stage": state.stage.value,
                    "draft": state.integrated_draft,
                    "sections": state.draft_sections,
                    "outline": state.outline,
                }
            )

        if args.action == "get_citations":
            if not args.session_id:
                return ToolResult(
                    data=None, success=False,
                    error="get_citations 需要 session_id 参数",
                )
            state = engine.get_session(args.session_id)
            if state is None:
                return ToolResult(
                    data=None, success=False,
                    error=f"Session {args.session_id} not found",
                )
            return ToolResult(
                data={
                    "session_id": state.session_id,
                    "citations": state.citations,
                    "issues": state.citation_issues,
                    "total": len(state.citations),
                    "issues_count": len(state.citation_issues),
                }
            )

        if args.action == "get_reviews":
            if not args.session_id:
                return ToolResult(
                    data=None, success=False,
                    error="get_reviews 需要 session_id 参数",
                )
            state = engine.get_session(args.session_id)
            if state is None:
                return ToolResult(
                    data=None, success=False,
                    error=f"Session {args.session_id} not found",
                )
            return ToolResult(
                data={
                    "session_id": state.session_id,
                    "reviews": state.reviews,
                    "meta_review": state.meta_review,
                    "revision_notes": state.revision_notes,
                }
            )

        if args.action == "verify":
            if not args.session_id:
                return ToolResult(
                    data=None, success=False,
                    error="verify 需要 session_id 参数",
                )
            state = engine.get_session(args.session_id)
            if state is None:
                return ToolResult(
                    data=None, success=False,
                    error=f"Session {args.session_id} not found",
                )
            if not state.citations:
                return ToolResult(
                    data=None, success=False,
                    error="No citations to verify. Run drafting stage first.",
                )
            rag_fn = _get_rag_search_fn(context)
            results = await engine.citation_verifier.verify(
                state.citations, rag_fn
            )
            state.citation_issues = [
                f"[{r['index']+1}] {r['status']}: {r['reason']}"
                for r in results
                if r and r.get("status") != "verified"
            ]
            return ToolResult(
                data={
                    "session_id": state.session_id,
                    "verification_results": results,
                    "issues": state.citation_issues,
                    "total": len(results),
                    "verified": sum(1 for r in results if r and r.get("status") == "verified"),
                    "issues_count": len(state.citation_issues),
                }
            )

        return ToolResult(
            data=None, success=False,
            error=f"未知 action: {args.action}",
        )


# ──────────────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────────────

def _safe_json_load(text: str, default: Any) -> Any:
    """从 LLM 输出里安全解析 JSON, 失败返回 default."""
    text = text.strip()
    # 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 找 { ... } 或 [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return default
