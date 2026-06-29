"""Hypothesis Generator Tool —— 文献综述 → 科学假设 → 可执行 workflow.

把 gap_analysis + LLM 推理串成一条链:
  1. 检索文献 (web_search_tool 优先, 退回 rag_tool, 走 ToolRegistry 互调)
  2. 识别研究空白 (gap_analysis_tool)
  3. LLM 生成可测试假设 (statement / rationale / testable_prediction / required_data)
  4. LLM 把假设映射到 workflow 模板
     (template_name / args / expected_observable / falsification_criterion)

假设生成和模板映射都需要 LLM 推理, SkillDefinition 的声明式步骤搞不定, 所以
独立成一个 tool. presets.py 里的 hypothesis_generator skill 只是声明式壳, 真活
在这里干. LLM 调用方式对齐 review_committee_tool (get_model + langchain messages).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)

# 已知 workflow 模板, 给 LLM 做映射时只能从里选. 名字对齐
# workflows/templates.py 和 skills/presets.py 里的 skill name.
_WORKFLOW_TEMPLATES: list[str] = [
    "standard_dft",
    "aimd",
    "defect_calculation",
    "surface_calculation",
    "ml_potential_training",
    "phonon_calculation",
    "lammps_melt_quench",
    "elastic_constants",
    "band_gap_analysis",
    "ht_screening",
]

_HYPOTHESIS_SYSTEM_PROMPT = """你是材料科学假设生成专家. 基于给定的研究主题、文献摘要和研究空白, 生成可测试的科学假设.

每个假设必须包含:
- statement: 假设的清晰陈述 (一句话)
- rationale: 提出该假设的依据, 要引用具体的研究空白
- testable_prediction: 可被计算/实验验证的预测, 要具体到可观测量
- required_data: 验证该假设需要的数据/结构/参数

要求:
1. 假设要具体、可证伪, 不要泛泛而谈.
2. 优先针对研究空白里的矛盾结论 / 未覆盖组合 / 少被研究的方法.
3. 数量不超过指定的 max_hypotheses.

输出严格 JSON, 不要 markdown 代码块标记, 不要任何解释文字. 格式:
{
  "hypotheses": [
    {"statement": "...", "rationale": "...", "testable_prediction": "...", "required_data": "..."}
  ]
}"""

_WORKFLOW_MAPPING_SYSTEM_PROMPT = (
    "你是计算材料科学 workflow 设计专家. 把科学假设映射到合适的 workflow 模板, "
    "让假设能被计算验证.\n\n"
    f"可选的 workflow 模板 (只能从里选): {_WORKFLOW_TEMPLATES}\n\n"
    "每个映射结果必须包含:\n"
    "- template_name: 从上面列表里选的模板名\n"
    "- args: 调用该模板需要的关键参数 (dict), 例如 "
    '{"structure_file": "...", "functional": "PBE"}\n'
    "- expected_observable: 该 workflow 能给出的可观测结果, 用来检验假设\n"
    "- falsification_criterion: 什么结果能证伪该假设 (具体判据)\n\n"
    "如果用户指定了 target_workflow, 全部映射到那个模板 (args 仍按假设调整).\n"
    "输出严格 JSON, 不要 markdown 代码块. 格式:\n"
    "{\n"
    '  "workflow_proposals": [\n'
    '    {"template_name": "...", "args": {...}, '
    '"expected_observable": "...", "falsification_criterion": "..."}\n'
    "  ]\n"
    "}"
)


class HypothesisGeneratorInput(BaseModel):
    research_topic: str = Field(
        ..., description="研究主题, 如 'GaN p-type doping efficiency'"
    )
    literature_query: str | None = Field(
        default=None, description="文献检索 query, 留空则用 research_topic"
    )
    max_hypotheses: int = Field(
        default=3, ge=1, le=10, description="最多生成几个假设"
    )
    target_workflow: str | None = Field(
        default=None,
        description="指定 workflow 模板名, 不指定则由 LLM 自动选",
    )


class HypothesisGeneratorTool(HuginnTool):
    """文献综述 → 科学假设 → 可执行 workflow 的编排工具.

    内部调 web_search_tool / rag_tool / gap_analysis_tool (走 ToolRegistry),
    再用 LLM 做假设生成和模板映射. 只读, 不写文件不提交作业.
    """

    name = "hypothesis_generator_tool"
    category = "search"
    description = (
        "From literature review to executable workflow: search literature, "
        "identify research gaps, generate testable scientific hypotheses via LLM, "
        "and map each hypothesis to a workflow template. Returns hypotheses + "
        "workflow_proposals + literature_summary."
    )
    input_schema = HypothesisGeneratorInput
    read_only = True

    async def call(
        self, args: HypothesisGeneratorInput, context: ToolContext
    ) -> ToolResult:
        query = args.literature_query or args.research_topic

        # 1. 检索文献
        papers, literature_summary = await self._search_literature(query, context)

        # 2. 识别研究空白
        research_gaps = await self._identify_gaps(
            args.research_topic, papers, context
        ) or {}

        # 3. 拿 LLM 客户端
        try:
            model = self._get_model(context)
        except Exception as exc:
            return ToolResult(
                data=None,
                success=False,
                error=f"初始化 LLM 客户端失败: {exc}",
            )

        # 4. LLM 生成假设
        hypotheses = await self._generate_hypotheses(
            args.research_topic,
            literature_summary,
            research_gaps,
            args.max_hypotheses,
            model,
        )

        # 5. LLM 映射到 workflow 模板
        workflow_proposals = await self._map_to_workflow(
            hypotheses, args.target_workflow, model
        )

        data: dict[str, Any] = {
            "research_topic": args.research_topic,
            "literature_summary": literature_summary,
            "research_gaps": research_gaps,
            "hypotheses": hypotheses,
            "workflow_proposals": workflow_proposals,
            "n_hypotheses": len(hypotheses),
        }
        return ToolResult(data=data, success=True)

    # ------------------------------------------------------------------ helpers

    def _get_model(self, context: ToolContext) -> Any:
        """拿 LangChain chat model, 优先用 context.config. 对齐 review_committee_tool."""
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.4, max_tokens=6000)

    async def _search_literature(
        self, query: str, context: ToolContext
    ) -> tuple[list[dict[str, Any]], str]:
        """走 ToolRegistry 调 literature_tool (arXiv/S2/CrossRef), 拿不到再退回
        web_search_tool / rag_tool. 返回 (papers, summary).

        literature_tool 给的是结构化论文元数据 (title/authors/year/venue/doi/
        abstract/url/citations), 比 web_search 的 snippet 质量高得多, 能撑起
        真正的文献综述. web_search/rag 只作 fallback.
        """
        from huginn.tools.registry import ToolRegistry

        ctx = self._fallback_context(context)
        papers: list[dict[str, Any]] = []
        summary_parts: list[str] = []

        # 1. 优先 literature_tool: 学术三路并发, 拿结构化论文
        lit = ToolRegistry.get("literature_tool")
        if lit is not None:
            try:
                res = await self._invoke_tool(
                    lit,
                    {"action": "search", "query": query, "max_results": 8},
                    ctx,
                )
                if res and res.success and isinstance(res.data, dict):
                    hits = res.data.get("papers", []) or []
                    for h in hits:
                        if isinstance(h, dict) and h.get("title"):
                            papers.append({
                                "title": h.get("title", ""),
                                "authors": h.get("authors", []),
                                "year": h.get("year"),
                                "venue": h.get("venue", ""),
                                "doi": h.get("doi"),
                                "abstract": h.get("abstract", "") or "",
                                "url": h.get("url", ""),
                                "citations": h.get("citations"),
                            })
                    src_status = res.data.get("source_status", {}) or {}
                    ok_sources = [
                        s for s, st in src_status.items()
                        if isinstance(st, dict) and st.get("ok")
                    ]
                    summary_parts.append(
                        f"literature_tool: {len(papers)} 篇 "
                        f"(sources: {','.join(ok_sources) or 'none'})"
                    )
                else:
                    # 调了但失败 (disabled/网络/空), 记一笔方便排查 fallback 链路
                    err = (res.error if res else "no result") or ""
                    summary_parts.append(
                        f"literature_tool: 无结果 ({err[:80]})"
                    )
            except Exception as exc:
                logger.warning("literature_tool 调用失败: %s", exc)
                summary_parts.append(f"literature_tool 失败: {exc}")

        # 2. literature_tool 没拿到, 退回 web_search_tool (snippet 当 abstract)
        if not papers:
            web = ToolRegistry.get("web_search_tool")
            if web is not None:
                try:
                    res = await self._invoke_tool(
                        web, {"query": query, "max_results": 8}, ctx
                    )
                    if res and res.success and isinstance(res.data, dict):
                        hits = res.data.get("results", []) or []
                        for h in hits:
                            if isinstance(h, dict):
                                papers.append({
                                    "title": h.get("title", ""),
                                    "authors": [],
                                    "year": None,
                                    "venue": "",
                                    "doi": None,
                                    "abstract": h.get("snippet", h.get("content", "")),
                                    "url": h.get("url", ""),
                                    "citations": None,
                                })
                        summary_parts.append(
                            f"web_search fallback: {len(papers)} 条 (query='{query}')"
                        )
                    else:
                        err = (res.error if res else "no result") or ""
                        summary_parts.append(
                            f"web_search fallback: 无结果 ({err[:80]})"
                        )
                except Exception as exc:
                    logger.warning("web_search_tool 调用失败: %s", exc)
                    summary_parts.append(f"web_search 失败: {exc}")

        # 3. 还没拿到, 退回本地 RAG
        if not papers:
            rag = ToolRegistry.get("rag_tool")
            if rag is not None:
                try:
                    res = await self._invoke_tool(
                        rag,
                        {"action": "search", "query": query, "top_k": 8},
                        ctx,
                    )
                    if res and res.success and isinstance(res.data, dict):
                        # rag_tool 返回 results 字段, 兼容 documents
                        docs = res.data.get("results") or res.data.get("documents") or []
                        for d in docs:
                            if isinstance(d, dict):
                                papers.append({
                                    "title": d.get("title", d.get("doc_id", "")),
                                    "authors": [],
                                    "year": None,
                                    "venue": "",
                                    "doi": None,
                                    "abstract": str(
                                        d.get("content", d.get("text", ""))
                                    )[:500],
                                    "url": "",
                                    "citations": None,
                                })
                        summary_parts.append(f"rag fallback: {len(papers)} 条本地文档")
                    else:
                        err = (res.error if res else "no result") or ""
                        summary_parts.append(f"rag fallback: 无结果 ({err[:80]})")
                except Exception as exc:
                    logger.warning("rag_tool 调用失败: %s", exc)
                    summary_parts.append(f"rag 失败: {exc}")

        if not papers:
            summary_parts.append(
                "未检索到文献 (literature_tool / web_search / rag 都没结果), "
                "仅基于研究主题生成假设"
            )

        return papers, "; ".join(summary_parts)

    async def _identify_gaps(
        self,
        topic: str,
        papers: list[dict[str, Any]],
        context: ToolContext,
    ) -> dict[str, Any] | None:
        from huginn.tools.registry import ToolRegistry

        tool = ToolRegistry.get("gap_analysis_tool")
        if tool is None:
            return None
        ctx = self._fallback_context(context)
        try:
            res = await self._invoke_tool(
                tool,
                {"action": "analyze_gaps", "topic": topic, "papers": papers},
                ctx,
            )
            if res and res.success and isinstance(res.data, dict):
                return res.data
        except Exception as exc:
            logger.warning("gap_analysis_tool 调用失败: %s", exc)
        return None

    async def _generate_hypotheses(
        self,
        topic: str,
        literature_summary: str,
        research_gaps: dict[str, Any],
        max_hypotheses: int,
        model: Any,
    ) -> list[dict[str, Any]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        gaps = research_gaps.get("gaps", []) if research_gaps else []
        user_prompt = (
            f"研究主题: {topic}\n\n"
            f"文献检索情况: {literature_summary}\n\n"
            f"识别出的研究空白 (JSON):\n"
            f"{json.dumps(gaps, ensure_ascii=False, indent=2)}\n\n"
            f"请生成不超过 {max_hypotheses} 条可测试的科学假设."
        )
        messages = [
            SystemMessage(content=_HYPOTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        content = await self._llm_invoke(model, messages)
        parsed = self._parse_json(content)
        if parsed and isinstance(parsed.get("hypotheses"), list):
            return [
                self._normalize_hypothesis(h)
                for h in parsed["hypotheses"][:max_hypotheses]
                if isinstance(h, dict)
            ]
        # LLM 没给合法 JSON, 兜底用 gap_analysis 的空白拼几条规则型假设
        logger.warning("假设生成 JSON 解析失败, 退回规则兜底")
        return self._fallback_hypotheses(topic, gaps, max_hypotheses)

    async def _map_to_workflow(
        self,
        hypotheses: list[dict[str, Any]],
        target_workflow: str | None,
        model: Any,
    ) -> list[dict[str, Any]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        if not hypotheses:
            return []
        user_prompt = (
            "科学假设 (JSON):\n"
            f"{json.dumps(hypotheses, ensure_ascii=False, indent=2)}\n\n"
        )
        if target_workflow:
            user_prompt += (
                f"用户指定 target_workflow='{target_workflow}', "
                "全部映射到该模板.\n"
            )
        user_prompt += "请输出每个假设对应的 workflow 映射."
        messages = [
            SystemMessage(content=_WORKFLOW_MAPPING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
        content = await self._llm_invoke(model, messages)
        parsed = self._parse_json(content)
        if parsed and isinstance(parsed.get("workflow_proposals"), list):
            proposals: list[dict[str, Any]] = []
            for p in parsed["workflow_proposals"]:
                if not isinstance(p, dict):
                    continue
                # 用户指定了模板就强制覆盖, 不信 LLM 自己选的
                if target_workflow:
                    p["template_name"] = target_workflow
                proposals.append(self._normalize_proposal(p))
            return proposals
        logger.warning("workflow 映射 JSON 解析失败, 返回空列表")
        return []

    # ---- 小工具 ----

    async def _invoke_tool(
        self, tool: Any, args: dict[str, Any], ctx: ToolContext
    ) -> Any:
        """统一调 ToolRegistry 里的工具. web_search_tool / gap_analysis_tool
        的 call 都吃 dict, rag_tool.call 吃 pydantic 模型但内部也兼容 dict."""
        if hasattr(tool, "call"):
            return await tool.call(args, ctx)
        if hasattr(tool, "execute"):
            return await asyncio.to_thread(tool.execute, args, ctx)
        return None

    async def _llm_invoke(self, model: Any, messages: list[Any]) -> str:
        """对齐 review_committee_tool: 优先 ainvoke, 退回同步 invoke + to_thread."""
        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)
        content = response.content if hasattr(response, "content") else str(response)
        # 个别 provider 返回 list[ContentBlock], 拼成纯文本
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return content

    @staticmethod
    def _fallback_context(context: ToolContext) -> ToolContext:
        """构造给子工具用的 ctx, 复用原 context 的 config/workspace."""
        return ToolContext(
            session_id=getattr(context, "session_id", "hypothesis") or "hypothesis",
            workspace=getattr(context, "workspace", ".") or ".",
            config=getattr(context, "config", None),
        )

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any] | None:
        """从 LLM 回复里抠 JSON. 容忍前后多余文字和 ```json 代码块."""
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _normalize_hypothesis(h: dict[str, Any]) -> dict[str, Any]:
        return {
            "statement": str(h.get("statement", "")),
            "rationale": str(h.get("rationale", "")),
            "testable_prediction": str(h.get("testable_prediction", "")),
            "required_data": str(h.get("required_data", "")),
        }

    @staticmethod
    def _normalize_proposal(p: dict[str, Any]) -> dict[str, Any]:
        args = p.get("args", {})
        if not isinstance(args, dict):
            args = {}
        template = str(p.get("template_name", "standard_dft"))
        return {
            "template_name": template,
            "args": args,
            "expected_observable": str(p.get("expected_observable", "")),
            "falsification_criterion": str(p.get("falsification_criterion", "")),
        }

    @staticmethod
    def _fallback_hypotheses(
        topic: str, gaps: list[dict[str, Any]], max_h: int
    ) -> list[dict[str, Any]]:
        """LLM 不可用时, 从 gap_analysis 的空白里拼几条规则型假设兜底."""
        out: list[dict[str, Any]] = []
        for g in gaps[:max_h]:
            desc = g.get("description", "")
            out.append(
                {
                    "statement": f"针对 '{topic}' 的研究空白待验证: {desc}",
                    "rationale": desc,
                    "testable_prediction": "",
                    "required_data": "",
                }
            )
        return out

    def estimate_cost(
        self, args: HypothesisGeneratorInput
    ) -> dict[str, float] | None:
        # 1 次检索 + 1 次 gap 分析 + 2 次 LLM 调用, 都是轻量
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.03}
