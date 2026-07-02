"""Gap analysis tool — 研究空白识别、方法对比、假设生成与新颖性评估.

纯规则分析，不调 LLM。喂进来一批论文，识别研究空白、对比方法、生成假设、评估新颖性。
所有计算都是基于关键词频率和重叠度的启发式规则，不做语义理解。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

# ── 情感关键词，用来扫 results 文本判断结论倾向 ──────────────────
_POSITIVE_WORDS = {
    "improve", "improved", "improves", "improving", "improvement",
    "enhance", "enhanced", "enhances", "enhancement",
    "increase", "increased", "increases", "increasing",
    "better", "superior", "excellent", "outstanding",
    "boost", "boosted", "promote", "promoted",
    "achieve", "achieved", "gain", "gained",
    "strengthen", "strengthened", "high", "higher", "highest",
    "optimal", "optimize", "optimized", "effective", "efficient",
    "success", "successful", "successfully",
}
_NEGATIVE_WORDS = {
    "decrease", "decreased", "decreases", "decreasing",
    "reduce", "reduced", "reduces", "reduction",
    "worse", "inferior", "poor", "limited",
    "decline", "declined", "fail", "failed", "failure",
    "drop", "dropped", "weaken", "weakened",
    "low", "lower", "lowest", "suppress", "suppressed",
    "ineffective", "inefficient", "unstable", "degrade", "degraded",
}

# 算关键词重叠时用的停用词表，够用就行
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its",
    "we", "our", "their", "they", "i", "you", "he", "she",
    "study", "studies", "paper", "research", "work", "result", "results",
    "show", "shows", "showed", "shown", "found", "find",
    "use", "used", "uses", "using", "based", "via", "through",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "can", "could", "may", "might", "should", "would", "will", "shall",
    "not", "no", "yes", "do", "does", "did", "has", "have", "had",
    "more", "most", "less", "least", "than", "then", "so", "such",
    "about", "into", "over", "under", "between", "within", "without",
}


class GapAnalysisInput(BaseModel):
    action: Literal[
        "analyze_gaps", "compare_methods", "generate_hypothesis", "assess_novelty"
    ] = Field(..., description="要执行的分析动作")
    topic: str = Field(..., description="研究主题")
    papers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="论文列表，每篇含 title/abstract/methods/results/tags",
    )
    existing_findings: list[str] = Field(
        default_factory=list,
        description="已有发现（可选），用于辅助 gap 分析",
    )
    hypotheses: list[str] = Field(
        default_factory=list,
        description="待评估的假设列表，assess_novelty 用",
    )


class GapAnalysisTool(HuginnTool):
    """识别研究空白、对比方法、生成假设、评估新颖性。纯规则，不调 LLM。"""

    name = "gap_analysis_tool"
    category = "design"
    description = (
        "分析已有文献/数据/发现，识别研究空白、对比方法效果、生成假设并评估新颖性。"
        "纯规则分析，不调用 LLM。"
    )
    input_schema = GapAnalysisInput
    read_only = True

    def is_read_only(self, args: GapAnalysisInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        try:
            input_data = GapAnalysisInput(**args)
            if input_data.action == "analyze_gaps":
                return self._analyze_gaps(input_data, context)
            if input_data.action == "compare_methods":
                return self._compare_methods(input_data)
            if input_data.action == "generate_hypothesis":
                return self._generate_hypothesis(input_data)
            if input_data.action == "assess_novelty":
                return self._assess_novelty(input_data)
            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {input_data.action}",
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    @staticmethod
    def _query_kb_coverage(gap_description: str, context: ToolContext | None) -> list[dict]:
        """查领域 KB 看这个 gap 是否已被已知结论覆盖. 命中返回 [{text, source}],
        KB 不可用/空/查询失败都返回 [], 不阻断 gap 分析."""
        if context is None:
            return []
        try:
            workspace = getattr(context, "workspace", None) or "."
            from huginn.knowledge.store import get_knowledge_base

            kb = get_knowledge_base(str(workspace))
            if kb.count() == 0:
                return []
            chunks = kb.query(gap_description, top_k=2)
            return [
                {
                    "text": (c.get("text") or "")[:200],
                    "source": c.get("source", ""),
                }
                for c in chunks
                if c.get("text")
            ]
        except Exception:
            return []

    # ── 字段抽取的小工具 ────────────────────────────────────────────

    @staticmethod
    def _normalize_methods(paper: dict[str, Any]) -> list[str]:
        """从 paper 里拿方法列表，兼容字符串和列表两种写法。"""
        raw = paper.get("methods", [])
        if isinstance(raw, str):
            parts = re.split(r"[,;/]+", raw)
            return [p.strip().lower() for p in parts if p.strip()]
        if isinstance(raw, list):
            return [str(m).strip().lower() for m in raw if str(m).strip()]
        return []

    @staticmethod
    def _normalize_tags(paper: dict[str, Any]) -> list[str]:
        raw = paper.get("tags", [])
        if isinstance(raw, str):
            parts = re.split(r"[,;/]+", raw)
            return [p.strip().lower() for p in parts if p.strip()]
        if isinstance(raw, list):
            return [str(t).strip().lower() for t in raw if str(t).strip()]
        return []

    @staticmethod
    def _extract_materials(paper: dict[str, Any]) -> list[str]:
        """优先用 materials 字段，没有就回退到 tags。"""
        if "materials" in paper:
            raw = paper["materials"]
            if isinstance(raw, str):
                parts = re.split(r"[,;/]+", raw)
                return [p.strip().lower() for p in parts if p.strip()]
            if isinstance(raw, list):
                return [str(m).strip().lower() for m in raw if str(m).strip()]
        return GapAnalysisTool._normalize_tags(paper)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """简单的英文分词，去停用词和过短的 token。"""
        if not text:
            return set()
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text.lower())
        return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    @staticmethod
    def _sentiment(text: str) -> str:
        """扫一遍 results 文本，给出 positive/negative/neutral 倾向。"""
        if not text:
            return "neutral"
        tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        pos = len(tokens & _POSITIVE_WORDS)
        neg = len(tokens & _NEGATIVE_WORDS)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    # ── action: analyze_gaps ───────────────────────────────────────

    def _analyze_gaps(
        self, args: GapAnalysisInput, context: ToolContext | None = None
    ) -> ToolResult:
        papers = args.papers or []
        n_total = len(papers)
        if n_total == 0:
            return ToolResult(
                data={
                    "action": "analyze_gaps",
                    "topic": args.topic,
                    "gaps": [],
                    "summary": "没有论文可分析，无法识别研究空白。",
                },
                success=True,
            )

        method_counter: Counter[str] = Counter()
        tag_counter: Counter[str] = Counter()
        # 每个 method 对应的论文列表，用来找矛盾
        method_papers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # method × material 已经出现过的组合
        seen_combos: set[tuple[str, str]] = set()
        all_methods: set[str] = set()
        all_materials: set[str] = set()

        for p in papers:
            methods = self._normalize_methods(p)
            tags = self._normalize_tags(p)
            materials = self._extract_materials(p)
            for m in methods:
                method_counter[m] += 1
                method_papers[m].append(p)
                all_methods.add(m)
                for mat in materials:
                    seen_combos.add((m, mat))
                    all_materials.add(mat)
            for t in tags:
                tag_counter[t] += 1

        gaps: list[dict[str, Any]] = []

        # 1) 少被研究的方法（频率 ≤ 总数的 20%）
        threshold = max(1, int(n_total * 0.2))
        for m, cnt in method_counter.items():
            if cnt <= threshold:
                gaps.append(
                    {
                        "description": f"方法 '{m}' 仅在 {cnt}/{n_total} 篇论文中出现，研究较少",
                        "evidence": [
                            p.get("title", f"paper#{i}")
                            for i, p in enumerate(papers)
                            if m in self._normalize_methods(p)
                        ],
                        "confidence": round(1.0 - cnt / n_total, 3),
                        "type": "under_studied_method",
                        "method": m,
                    }
                )

        # 2) 未出现的 method × material 组合
        for m in sorted(all_methods):
            for mat in sorted(all_materials):
                if (m, mat) not in seen_combos:
                    gaps.append(
                        {
                            "description": f"方法 '{m}' × 材料 '{mat}' 的组合在已有论文中未出现",
                            "evidence": [],
                            "confidence": 0.5,
                            "type": "missing_combination",
                            "method": m,
                            "material": mat,
                        }
                    )

        # 3) 矛盾结果：同一方法在不同论文中给出相反倾向
        for m, plist in method_papers.items():
            if len(plist) < 2:
                continue
            sentiments: list[tuple[dict[str, Any], str]] = []
            for p in plist:
                s = self._sentiment(str(p.get("results", "")))
                if s != "neutral":
                    sentiments.append((p, s))
            pos = [x for x in sentiments if x[1] == "positive"]
            neg = [x for x in sentiments if x[1] == "negative"]
            if pos and neg:
                gaps.append(
                    {
                        "description": (
                            f"方法 '{m}' 在不同论文中给出相反结论"
                            f"（{len(pos)} 篇正向 / {len(neg)} 篇负向）"
                        ),
                        "evidence": [
                            {
                                "title": p.get("title", ""),
                                "sentiment": s,
                                "results_snippet": str(p.get("results", ""))[:200],
                            }
                            for p, s in sentiments
                        ],
                        "confidence": 0.7,
                        "type": "contradiction",
                        "method": m,
                    }
                )

        # A5: KB 覆盖检查 — 每个 gap 查 KB, 命中的标注 kb_covered=True,
        # 防止 agent 把已知结论当新发现. KB 不可用时跳过.
        n_kb_covered = 0
        for g in gaps:
            kb_hits = self._query_kb_coverage(g.get("description", ""), context)
            if kb_hits:
                g["kb_covered"] = True
                g["kb_references"] = kb_hits
                n_kb_covered += 1
            else:
                g["kb_covered"] = False

        n_under = sum(1 for g in gaps if g["type"] == "under_studied_method")
        n_missing = sum(1 for g in gaps if g["type"] == "missing_combination")
        n_contra = sum(1 for g in gaps if g["type"] == "contradiction")
        summary = (
            f"在主题 '{args.topic}' 下分析了 {n_total} 篇论文，"
            f"识别出 {len(gaps)} 处潜在研究空白"
            f"（少被研究方法 {n_under}，未出现组合 {n_missing}，矛盾结果 {n_contra}）。"
            f"其中 {n_kb_covered} 处已被领域 KB 覆盖，建议验证而非重新假设。"
        )

        data = {
            "action": "analyze_gaps",
            "topic": args.topic,
            "n_papers": n_total,
            "method_frequency": dict(method_counter.most_common()),
            "tag_frequency": dict(tag_counter.most_common()),
            "gaps": gaps,
            "n_kb_covered": n_kb_covered,
            "summary": summary,
        }
        return ToolResult(data=data, success=True)

    # ── action: compare_methods ────────────────────────────────────

    def _compare_methods(self, args: GapAnalysisInput) -> ToolResult:
        papers = args.papers or []
        if not papers:
            return ToolResult(
                data={
                    "action": "compare_methods",
                    "topic": args.topic,
                    "comparison": [],
                    "summary": "没有论文可对比。",
                },
                success=True,
            )

        # 按 method 分组
        method_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in papers:
            for m in self._normalize_methods(p):
                method_groups[m].append(p)

        comparison: list[dict[str, Any]] = []
        for m, plist in method_groups.items():
            # 每篇论文的"条件"——优先 conditions 字段，没有就回退到 tags
            conditions_per_paper: list[list[str]] = []
            for p in plist:
                conds = p.get("conditions")
                if isinstance(conds, str) and conds.strip():
                    conditions_per_paper.append([conds.strip().lower()])
                elif isinstance(conds, list) and conds:
                    conditions_per_paper.append(
                        [str(c).strip().lower() for c in conds if str(c).strip()]
                    )
                else:
                    conditions_per_paper.append(self._normalize_tags(p))

            # 每个条件下的表现分布
            cond_perf: dict[str, list[str]] = defaultdict(list)
            for p, conds in zip(plist, conditions_per_paper):
                s = self._sentiment(str(p.get("results", "")))
                for c in conds:
                    cond_perf[c].append(s)

            # 每个条件的整体倾向
            cond_summary: dict[str, str] = {}
            for c, sentiments in cond_perf.items():
                pos = sentiments.count("positive")
                neg = sentiments.count("negative")
                if pos > neg:
                    cond_summary[c] = "positive"
                elif neg > pos:
                    cond_summary[c] = "negative"
                else:
                    cond_summary[c] = "neutral"

            # 边界条件：同一方法在不同条件下表现相反
            pos_conds = [c for c, s in cond_summary.items() if s == "positive"]
            neg_conds = [c for c, s in cond_summary.items() if s == "negative"]
            if pos_conds and neg_conds:
                gap_desc = (
                    f"方法 '{m}' 在条件 {pos_conds} 下表现正向，"
                    f"但在条件 {neg_conds} 下表现负向"
                )
            else:
                gap_desc = "未发现明显的条件边界"

            boundaries: list[dict[str, Any]] = []
            for c, sentiments in cond_perf.items():
                has_pos = "positive" in sentiments
                has_neg = "negative" in sentiments
                if has_pos and has_neg:
                    boundaries.append(
                        {
                            "condition": c,
                            "mixed_performance": True,
                            "sentiments": sentiments,
                        }
                    )
                else:
                    boundaries.append(
                        {
                            "condition": c,
                            "performance": sentiments[0] if sentiments else "neutral",
                            "n_papers": len(sentiments),
                        }
                    )

            comparison.append(
                {
                    "method": m,
                    "n_papers": len(plist),
                    "conditions": list(cond_perf.keys()),
                    "performance": cond_summary,
                    "boundaries": boundaries,
                    "gap_description": gap_desc,
                }
            )

        # 按论文数从多到少排
        comparison.sort(key=lambda x: -x["n_papers"])
        n_boundary = sum(
            1 for c in comparison if c["gap_description"] != "未发现明显的条件边界"
        )
        summary = (
            f"对比了 {len(method_groups)} 种方法，发现 {n_boundary} 处条件边界。"
        )
        data = {
            "action": "compare_methods",
            "topic": args.topic,
            "n_methods": len(method_groups),
            "comparison": comparison,
            "summary": summary,
        }
        return ToolResult(data=data, success=True)

    # ── action: generate_hypothesis ────────────────────────────────

    def _generate_hypothesis(self, args: GapAnalysisInput) -> ToolResult:
        # 复用 analyze_gaps 的结果作为假设生成的基础
        gap_result = self._analyze_gaps(args)
        if not gap_result.success or not gap_result.data:
            return gap_result
        gaps = gap_result.data.get("gaps", [])
        method_counter = gap_result.data.get("method_frequency", {})

        hypotheses: list[dict[str, Any]] = []
        for gap in gaps:
            gtype = gap.get("type")
            method = gap.get("method", "")
            material = gap.get("material")
            freq = method_counter.get(method, 0)

            if gtype == "missing_combination" and method and material:
                statement = (
                    f"如果将 {method} 应用到 {material}，"
                    f"可能获得与现有方法不同的性能表现"
                )
                rationale = (
                    f"已有文献中未见 {method} × {material} 的组合，"
                    f"而 {method} 在其他材料上已被验证有效。"
                )
                # 方法越成熟（出现次数多）越容易落地测试
                testability = min(0.9, 0.4 + 0.1 * freq)
            elif gtype == "under_studied_method" and method:
                statement = (
                    f"如果在 {args.topic} 中进一步应用 {method}，"
                    f"可能揭示被现有主流方法忽略的现象"
                )
                rationale = (
                    f"{method} 仅在少量论文中出现，研究尚不充分，存在拓展空间。"
                )
                testability = min(0.85, 0.3 + 0.1 * freq)
            elif gtype == "contradiction" and method:
                statement = (
                    f"对 {method} 在 {args.topic} 中的矛盾结论进行机理分析，"
                    f"可能识别出决定性边界条件"
                )
                rationale = (
                    "不同论文对 " + method + " 给出相反结论，"
                    "暗示存在未控制的关键变量。"
                )
                testability = 0.8
            else:
                continue

            hypotheses.append(
                {
                    "statement": statement,
                    "rationale": rationale,
                    "testability": round(testability, 3),
                    "source_gap": gap.get("description", ""),
                }
            )

        # 按 statement 去重
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for h in hypotheses:
            if h["statement"] in seen:
                continue
            seen.add(h["statement"])
            unique.append(h)

        if unique:
            avg_test = sum(h["testability"] for h in unique) / len(unique)
            summary = (
                f"基于 {len(gaps)} 处研究空白生成了 {len(unique)} 条假设，"
                f"平均可测试性 {avg_test:.2f}。"
            )
        else:
            summary = "未找到可生成假设的研究空白。"

        data = {
            "action": "generate_hypothesis",
            "topic": args.topic,
            "hypotheses": unique,
            "summary": summary,
        }
        return ToolResult(data=data, success=True)

    # ── action: assess_novelty ─────────────────────────────────────

    def _assess_novelty(self, args: GapAnalysisInput) -> ToolResult:
        papers = args.papers or []
        hypotheses = args.hypotheses or []
        if not hypotheses:
            return ToolResult(
                data={
                    "action": "assess_novelty",
                    "topic": args.topic,
                    "assessments": [],
                    "summary": "没有待评估的假设。",
                },
                success=True,
            )

        # 预处理每篇论文的关键词集合（title + abstract）
        paper_keywords: list[tuple[str, set[str]]] = []
        for p in papers:
            text = " ".join(
                [str(p.get("title", "")), str(p.get("abstract", ""))]
            )
            paper_keywords.append((p.get("title", ""), self._tokenize(text)))

        assessments: list[dict[str, Any]] = []
        for h in hypotheses:
            h_tokens = self._tokenize(h)
            if not h_tokens or not paper_keywords:
                # 假设里没拿到有效关键词，给个中庸的分数
                assessments.append(
                    {
                        "hypothesis": h,
                        "novelty_score": 1.0 if not paper_keywords else 0.5,
                        "max_similarity": 0.0,
                        "similar_works": [],
                    }
                )
                continue

            # 和每篇论文算 Jaccard 相似度
            scored = [
                (title, self._jaccard(h_tokens, kws))
                for title, kws in paper_keywords
            ]
            scored.sort(key=lambda x: -x[1])
            max_sim = scored[0][1] if scored else 0.0
            novelty = round(1.0 - max_sim, 3)
            # 相似度 ≥ 0.1 的算"类似工作"
            similar = [
                {"title": t, "similarity": round(s, 3)}
                for t, s in scored
                if s >= 0.1
            ][:5]
            assessments.append(
                {
                    "hypothesis": h,
                    "novelty_score": novelty,
                    "max_similarity": round(max_sim, 3),
                    "similar_works": similar,
                }
            )

        avg_novelty = (
            sum(a["novelty_score"] for a in assessments) / len(assessments)
            if assessments
            else 0.0
        )
        summary = (
            f"评估了 {len(assessments)} 条假设的新颖性，"
            f"平均 novelty_score = {avg_novelty:.3f}。"
        )
        data = {
            "action": "assess_novelty",
            "topic": args.topic,
            "n_papers_compared": len(papers),
            "assessments": assessments,
            "summary": summary,
        }
        return ToolResult(data=data, success=True)
