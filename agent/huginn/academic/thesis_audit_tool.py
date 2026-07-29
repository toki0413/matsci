"""论文审计工具 —— denominator-first 审稿, 4 阶段闭环.

把 tashan thesis-audit-reviewer 的核心工作流炼化成单文件 HuginnTool:
  1. 对象账本: 正则抽章节/图/表/公式/参考/强结论/建议
  2. 审阅矩阵: 对每个分母条目判 9 种覆盖状态
  3. 问题库:   issue 必须带 location/excerpt/basis/priority/certainty
  4. 完成门禁: _validate_audit_report 作 post-condition

不依赖 MinerU/docx, 输入已是 markdown 文本.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


# ── 9 种覆盖状态 (审阅矩阵) ──────────────────────────────────────────
STATUS_PASS = "pass"
STATUS_ISSUE = "issue"
STATUS_NEEDS_FACTCHECK = "needs_factcheck"
STATUS_NEEDS_INTERNAL_CROSSCHECK = "needs_internal_crosscheck"
STATUS_NEEDS_METHOD_PREMISE_CHECK = "needs_method_premise_check"
STATUS_NEEDS_RECALCULATION = "needs_recalculation"
STATUS_NEEDS_AUTHOR_SOURCE = "needs_author_source"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_BLOCKED = "blocked"

COVERAGE_STATUSES: tuple[str, ...] = (
    STATUS_PASS,
    STATUS_ISSUE,
    STATUS_NEEDS_FACTCHECK,
    STATUS_NEEDS_INTERNAL_CROSSCHECK,
    STATUS_NEEDS_METHOD_PREMISE_CHECK,
    STATUS_NEEDS_RECALCULATION,
    STATUS_NEEDS_AUTHOR_SOURCE,
    STATUS_NOT_APPLICABLE,
    STATUS_BLOCKED,
)

# 强结论 / 政策建议触发词, 抽分母用
_STRONG_CLAIM_KEYWORDS = ("首次", "证明", "发现", "提出")
_POLICY_KEYWORDS = ("建议", "应该", "应当")


class AuditValidationError(Exception):
    """完成门禁校验失败."""


# ── 阶段 1 产物: 对象账本 ────────────────────────────────────────────
@dataclass
class AuditDenominator:
    sections: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    references_count: int = 0
    strong_claims: list[str] = field(default_factory=list)
    policy_recommendations: list[str] = field(default_factory=list)

    def total(self) -> int:
        return (
            len(self.sections)
            + len(self.figures)
            + len(self.tables)
            + len(self.equations)
            + (1 if self.references_count > 0 else 0)
            + len(self.strong_claims)
            + len(self.policy_recommendations)
        )


# ── input / output schemas ──────────────────────────────────────────
class ThesisAuditInput(BaseModel):
    content: str = Field(..., description="待审稿论文 markdown 文本")
    mode: Literal["quick", "full"] = Field(
        default="full",
        description="quick=只跑分母+门禁, 不调 LLM; full=全流程含 LLM 语义判断",
    )
    focus_areas: list[str] | None = Field(
        default=None,
        description=(
            "聚焦检查项, 可选值: sections / figures / tables / equations / "
            "references / strong_claims / policy_recommendations; None=全量"
        ),
    )


class ThesisAuditOutput(BaseModel):
    denominator: dict[str, Any]
    coverage_matrix: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    work_notes: list[str]
    overall_verdict: Literal["pass", "minor_issues", "major_issues", "blocked"]
    completion_gate_passed: bool


# ── LLM system prompt: 审阅矩阵 + 问题库一次出 ─────────────────────
_AUDIT_SYSTEM_PROMPT = """你是论文审计员。给定论文 markdown 和分母条目清单, 你要:
1. 对每个条目判定覆盖状态, 只能从这 9 个里选:
   pass / issue / needs_factcheck / needs_internal_crosscheck / needs_method_premise_check / needs_recalculation / needs_author_source / not_applicable / blocked
   - pass 也要写检查范围 (scope)
   - blocked 必须显式列出原因 (reason), 禁止状态留空
2. 挑出有问题的条目, 每条 issue 必须含: id, location (章节号+段落), excerpt (原文摘录), basis (判断依据), priority (critical/major/minor), certainty (high/medium/low)
3. 无证据的怀疑进 work_notes, 不进正式 issues

输出严格 JSON, 不要 markdown 代码块:
{
  "coverage": [
    {"id": "section:1", "status": "pass", "scope": "...", "reason": null},
    {"id": "strong_claim:3", "status": "needs_factcheck", "scope": "...", "reason": null}
  ],
  "issues": [
    {"id": "strong_claim:3", "location": "3.2 第1段", "excerpt": "...", "basis": "...", "priority": "major", "certainty": "medium"}
  ],
  "work_notes": ["..."],
  "overall_verdict": "pass | minor_issues | major_issues | blocked"
}"""


class ThesisAuditTool(HuginnTool):
    """论文审计: 分母抽取 + 9 态覆盖矩阵 + 完成门禁."""

    name = "thesis_audit_tool"
    category = "analysis"
    description = (
        "论文审计工具: 从 markdown 文本抽取分母 (章节/图/表/公式/参考/强结论/建议), "
        "对每个条目判定 9 种覆盖状态, 输出带证据的问题库和总体结论. "
        "完成门禁强制校验覆盖完整性和 issue 字段. mode=quick 跳过 LLM."
    )
    input_schema = ThesisAuditInput
    output_schema = ThesisAuditOutput
    read_only = True

    async def call(
        self, args: ThesisAuditInput, context: ToolContext
    ) -> ToolResult:
        # 阶段 1: 对象账本 (纯正则, 不调 LLM)
        denom = self._extract_denominator(args.content)

        # 阶段 2/3: 审阅矩阵 + 问题库
        if args.mode == "quick":
            coverage, issues, work_notes, verdict = self._quick_audit(
                denom, args.focus_areas
            )
        else:
            try:
                model = self._get_model(context)
                coverage, issues, work_notes, verdict = await self._llm_audit(
                    args.content, denom, model, args.focus_areas
                )
            except Exception as exc:
                # 失败静默: LLM 挂了就全标 blocked, 不让审计本身崩
                logger.warning("LLM 审计失败, 退回 blocked: %s", exc)
                coverage, issues, work_notes, verdict = self._quick_audit(
                    denom, args.focus_areas
                )

        result: dict[str, Any] = {
            "denominator": self._denominator_to_dict(denom),
            "coverage_matrix": coverage,
            "issues": issues,
            "work_notes": work_notes,
            "overall_verdict": verdict,
            "completion_gate_passed": False,
        }

        # 阶段 4: 完成门禁 (post-condition)
        try:
            self._validate_audit_report(result)
            result["completion_gate_passed"] = True
        except AuditValidationError as exc:
            result["completion_gate_passed"] = False
            result["work_notes"].append(f"完成门禁失败: {exc}")
            return ToolResult(data=result, success=False, error=str(exc))

        return ToolResult(data=result, success=True)

    # ── 阶段 1: 分母抽取 ─────────────────────────────────────────
    def _extract_denominator(self, content: str) -> AuditDenominator:
        return AuditDenominator(
            sections=self._extract_sections(content),
            figures=self._extract_figures(content),
            tables=self._extract_tables(content),
            equations=self._extract_equations(content),
            references_count=self._count_references(content),
            strong_claims=self._extract_strong_claims(content),
            policy_recommendations=self._extract_policy_recs(content),
        )

    def _extract_sections(self, content: str) -> list[str]:
        # 1-3 级 markdown 标题, 带 level 前缀方便后续定位
        matches = re.findall(r"^(#{1,3})\s+(.+?)\s*$", content, re.MULTILINE)
        return [f"{len(h)}-{t}" for h, t in matches]

    def _extract_figures(self, content: str) -> list[str]:
        # Fig. 1 / Figure 1 / 图1, 去重保序
        matches = re.findall(
            r"(?:Fig\.|Figure)\s*(\d+)|图\s*(\d+)", content, re.IGNORECASE
        )
        return self._dedup_keys(f"fig:{a or b}" for a, b in matches)

    def _extract_tables(self, content: str) -> list[str]:
        matches = re.findall(r"Table\s*(\d+)|表\s*(\d+)", content, re.IGNORECASE)
        return self._dedup_keys(f"table:{a or b}" for a, b in matches)

    def _extract_equations(self, content: str) -> list[str]:
        # ponytail: $$ 配对计数 + equation 环境计数; 相邻 inline $ 不影响
        # 升级路径: 用 AST 级 latex 解析器区分 inline/display
        block_count = len(re.findall(r"\$\$", content)) // 2
        env_count = len(re.findall(r"\\begin\{equation\}", content))
        return [f"eq:{i+1}" for i in range(block_count + env_count)]

    def _count_references(self, content: str) -> int:
        # 优先找 References / 参考文献 段后的 [N] 最大编号
        ref_section = re.search(
            r"(?:^##?\s*(?:References|参考文献|Bibliography))\s*\n(.*)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if ref_section:
            nums = re.findall(r"\[(\d+)\]", ref_section.group(1))
            if nums:
                return max(int(n) for n in nums)
        # 兜底: 全文 [N] 最大值
        nums = re.findall(r"\[(\d+)\]", content)
        return max((int(n) for n in nums), default=0)

    def _extract_strong_claims(self, content: str) -> list[str]:
        sentences = self._split_sentences(content)
        return [
            f"strong_claim:{i}"
            for i, s in enumerate(sentences)
            if any(kw in s for kw in _STRONG_CLAIM_KEYWORDS)
        ]

    def _extract_policy_recs(self, content: str) -> list[str]:
        sentences = self._split_sentences(content)
        return [
            f"policy:{i}"
            for i, s in enumerate(sentences)
            if any(kw in s for kw in _POLICY_KEYWORDS)
        ]

    @staticmethod
    def _split_sentences(content: str) -> list[str]:
        # ponytail: 按中英文句号/问号/感叹号切, 不引 jieba/nltk
        parts = re.split(r"(?<=[。！？!?\.])\s+", content)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _dedup_keys(keys: Any) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    # ── 阶段 2/3: 审阅矩阵 + 问题库 ──────────────────────────────
    def _quick_audit(
        self, denom: AuditDenominator, focus_areas: list[str] | None
    ) -> tuple[list[dict], list[dict], list[str], str]:
        """quick 模式 / LLM 兜底: 全标 blocked, 等人工或 full 模式."""
        items = self._denom_items(denom, focus_areas)
        coverage = [
            {
                "id": iid,
                "status": STATUS_BLOCKED,
                "scope": None,
                "reason": "quick 模式或 LLM 不可用, 未做语义判断",
            }
            for iid in items
        ]
        return (
            coverage,
            [],
            ["所有条目标 blocked, 待 full 模式或人工审计"],
            "blocked",
        )

    async def _llm_audit(
        self,
        content: str,
        denom: AuditDenominator,
        model: Any,
        focus_areas: list[str] | None,
    ) -> tuple[list[dict], list[dict], list[str], str]:
        from langchain_core.messages import HumanMessage, SystemMessage

        items = self._denom_items(denom, focus_areas)
        user_prompt = self._build_audit_prompt(content, items, focus_areas)
        messages = [
            SystemMessage(content=_AUDIT_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        if hasattr(model, "ainvoke"):
            response = await model.ainvoke(messages)
        else:
            response = await asyncio.to_thread(model.invoke, messages)
        text = response.content if hasattr(response, "content") else str(response)
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        return self._parse_audit_json(text, items)

    def _denom_items(
        self, denom: AuditDenominator, focus_areas: list[str] | None
    ) -> list[str]:
        """把分母按 focus_areas 过滤, 没给就全量."""
        buckets: dict[str, list[str]] = {
            "sections": denom.sections,
            "figures": denom.figures,
            "tables": denom.tables,
            "equations": denom.equations,
            "references": (
                [f"ref:{i+1}" for i in range(denom.references_count)]
                if denom.references_count
                else []
            ),
            "strong_claims": denom.strong_claims,
            "policy_recommendations": denom.policy_recommendations,
        }
        if focus_areas:
            out: list[str] = []
            for area in focus_areas:
                out.extend(buckets.get(area, []))
            return out
        # 全量展开
        all_items: list[str] = []
        for v in buckets.values():
            all_items.extend(v)
        return all_items

    def _build_audit_prompt(
        self,
        content: str,
        items: list[str],
        focus_areas: list[str] | None,
    ) -> str:
        lines = ["请对以下论文进行审计。", ""]
        if focus_areas:
            lines.append(f"聚焦检查项: {focus_areas}")
        lines.append(f"分母条目清单 ({len(items)} 条):")
        lines.extend(f"  - {iid}" for iid in items)
        lines.append("")
        lines.append("=" * 60)
        lines.append("【论文内容】")
        lines.append("=" * 60)
        lines.append(content)
        return "\n".join(lines)

    def _parse_audit_json(
        self, text: str, items: list[str]
    ) -> tuple[list[dict], list[dict], list[str], str]:
        """从 LLM 回复抠 JSON, 补齐缺失条目, 字段不全的 issue 丢弃."""
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```\s*$", "", raw)
        data: dict[str, Any] = {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    data = {}

        # coverage: 以 items 为基准, LLM 没回或状态非法的补 blocked
        cov_by_id = {
            c.get("id"): c
            for c in (data.get("coverage") or [])
            if isinstance(c, dict)
        }
        coverage: list[dict] = []
        for iid in items:
            c = cov_by_id.get(iid)
            status = c.get("status") if c else None
            if status not in COVERAGE_STATUSES:
                status = STATUS_BLOCKED
            coverage.append(
                {
                    "id": iid,
                    "status": status,
                    "scope": c.get("scope") if c else None,
                    "reason": (
                        c.get("reason")
                        if c
                        else "LLM 未返回该条目"
                    ),
                }
            )

        # issues: 字段不全的丢进 work_notes, 不进正式列表
        required = ("location", "excerpt", "basis", "priority", "certainty")
        issues: list[dict] = []
        work_notes: list[str] = list(data.get("work_notes") or [])
        for it in data.get("issues") or []:
            if not isinstance(it, dict):
                continue
            if all(it.get(k) for k in required):
                issues.append(
                    {
                        "id": it.get("id"),
                        "location": it["location"],
                        "excerpt": it["excerpt"],
                        "basis": it["basis"],
                        "priority": it["priority"],
                        "certainty": it["certainty"],
                    }
                )
            else:
                work_notes.append(
                    f"issue {it.get('id')} 字段不全, 已降级为 work_note"
                )

        verdict = data.get("overall_verdict", "blocked")
        if verdict not in ("pass", "minor_issues", "major_issues", "blocked"):
            verdict = "blocked"
        return coverage, issues, work_notes, verdict

    # ── 阶段 4: 完成门禁 ─────────────────────────────────────────
    def _validate_audit_report(self, result: dict) -> None:
        """post-condition: 校验报告完整性, 失败抛 AuditValidationError."""
        coverage = result.get("coverage_matrix", [])
        issues = result.get("issues", [])

        # 1. 所有分母条目都有覆盖状态 (无空状态); blocked 必须给 reason
        for c in coverage:
            status = c.get("status")
            if not status or status not in COVERAGE_STATUSES:
                raise AuditValidationError(
                    f"条目 {c.get('id')} 覆盖状态为空或非法: {status!r}"
                )
            if status == STATUS_BLOCKED and not c.get("reason"):
                raise AuditValidationError(
                    f"条目 {c.get('id')} 状态为 blocked 但未给 reason"
                )

        # 2. 所有 issue 都有 5 个必填字段
        required = ("location", "excerpt", "basis", "priority", "certainty")
        for it in issues:
            for k in required:
                if not it.get(k):
                    raise AuditValidationError(
                        f"issue {it.get('id')} 缺字段: {k}"
                    )

        # 3. critical issue > 0 时 overall_verdict 不能是 pass
        critical = sum(1 for it in issues if it.get("priority") == "critical")
        if critical > 0 and result.get("overall_verdict") == "pass":
            raise AuditValidationError(
                f"存在 {critical} 条 critical issue, overall_verdict 不能是 pass"
            )

    # ── helpers ──────────────────────────────────────────────────
    def _get_model(self, context: ToolContext) -> Any:
        from huginn.llm import get_model

        config = getattr(context, "config", None)
        return get_model(config=config, temperature=0.2, max_tokens=8000)

    @staticmethod
    def _denominator_to_dict(denom: AuditDenominator) -> dict[str, Any]:
        return {
            "sections": denom.sections,
            "figures": denom.figures,
            "tables": denom.tables,
            "equations": denom.equations,
            "references_count": denom.references_count,
            "strong_claims": denom.strong_claims,
            "policy_recommendations": denom.policy_recommendations,
            "total": denom.total(),
        }

    def estimate_cost(self, args: ThesisAuditInput) -> dict[str, float] | None:
        # full 模式 1 次 LLM 调用; quick 模式 0 次
        walltime = 0.02 if args.mode == "full" else 0.0
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": walltime}
