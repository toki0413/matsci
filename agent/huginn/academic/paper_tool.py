"""论文写作辅助工具 —— 封装期刊查询、规范检查、参考文献格式化为 HuginnTool.

Agent 通过这个工具可以:
  - 查询期刊投稿规范 (get_spec)
  - 检查稿件是否符合期刊规范 (check)
  - 按期刊格式生成参考文献 (format_ref)
  - 生成投稿检查清单 (checklist)
  - 搜索/列出期刊 (search / list)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.academic.journal_db import (
    JournalSpec,
    get_journal,
    get_reference_format,
    list_journals,
    search_journals,
)
from huginn.academic.standards_checker import StandardsChecker
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class PaperToolInput(BaseModel):
    action: Literal[
        "get_spec",       # 查询单本期刊规范
        "check",          # 检查稿件合规性
        "format_ref",     # 格式化参考文献
        "checklist",      # 生成投稿检查清单
        "search",         # 搜索期刊
        "list",           # 列出期刊
        "ref_format",     # 获取参考文献格式说明
    ] = Field(
        description=(
            "get_spec=查询单本期刊规范; check=检查稿件合规性; "
            "format_ref=格式化参考文献; checklist=生成投稿检查清单; "
            "search=搜索期刊; list=列出期刊; ref_format=获取参考文献格式说明"
        )
    )
    journal: str | None = Field(
        default=None,
        description="期刊名称 (中英文均可), get_spec/check/format_ref/checklist/ref_format 时必填",
    )
    manuscript: dict[str, Any] | None = Field(
        default=None,
        description=(
            "稿件信息, check 时必填. 支持的 key: "
            "title, abstract, abstract_lang, body, methods, "
            "references(list), figure_count(int), keywords(list), "
            "data_availability, code_availability, reporting_summary, "
            "orcid, bilingual, copyright_agreement, classification_number, "
            "unit_proof, originality, ai_declaration"
        ),
    )
    ref_data: dict[str, Any] | None = Field(
        default=None,
        description=(
            "参考文献元数据, format_ref 时必填. 支持的 key: "
            "authors(list[str]), title, journal, year, volume, pages, doi"
        ),
    )
    query: str | None = Field(
        default=None,
        description="搜索关键词, search 时必填"
    )
    field: str | None = Field(
        default=None,
        description="学科领域过滤, list 时可选"
    )


class PaperTool(HuginnTool):
    """论文写作辅助: 期刊规范查询 + 合规检查 + 参考文献格式化."""

    name = "paper_tool"
    category = "sci"
    description = (
        "学术期刊规范查询与稿件合规检查工具. 支持查询14种国内外期刊的投稿规范, "
        "检查稿件标题/摘要/正文/参考文献/图表/关键词是否符合期刊要求, "
        "按期刊格式生成参考文献, 生成投稿检查清单."
    )
    read_only = True
    input_schema = PaperToolInput

    def __init__(self) -> None:
        super().__init__()
        self._checker = StandardsChecker()

    async def _execute(
        self, args: PaperToolInput, context: ToolContext
    ) -> ToolResult:
        action = args.action

        if action == "get_spec":
            return self._handle_get_spec(args)
        elif action == "check":
            return self._handle_check(args)
        elif action == "format_ref":
            return self._handle_format_ref(args)
        elif action == "checklist":
            return self._handle_checklist(args)
        elif action == "search":
            return self._handle_search(args)
        elif action == "list":
            return self._handle_list(args)
        elif action == "ref_format":
            return self._handle_ref_format(args)
        else:
            return ToolResult(
                data=None, success=False,
                error=f"未知 action: {action}",
            )

    # ── 各 action 处理 ────────────────────────────────────────

    def _handle_get_spec(self, args: PaperToolInput) -> ToolResult:
        if not args.journal:
            return ToolResult(
                data=None, success=False,
                error="get_spec 需要提供 journal 参数",
            )
        spec = get_journal(args.journal)
        if spec is None:
            return ToolResult(
                data=None, success=False,
                error=f"未找到期刊 '{args.journal}'",
            )
        return ToolResult(data=self._spec_to_dict(spec), success=True)

    def _handle_check(self, args: PaperToolInput) -> ToolResult:
        if not args.journal:
            return ToolResult(
                data=None, success=False,
                error="check 需要提供 journal 参数",
            )
        if not args.manuscript:
            return ToolResult(
                data=None, success=False,
                error="check 需要提供 manuscript 参数",
            )

        results = self._checker.check_compliance(args.manuscript, args.journal)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        warnings = sum(
            1 for r in results if not r.passed and r.severity == "warning"
        )

        return ToolResult(
            data={
                "journal": args.journal,
                "total_checks": len(results),
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "all_passed": failed == 0,
                "results": [asdict(r) for r in results],
            },
            success=True,
        )

    def _handle_format_ref(self, args: PaperToolInput) -> ToolResult:
        if not args.journal:
            return ToolResult(
                data=None, success=False,
                error="format_ref 需要提供 journal 参数",
            )
        if not args.ref_data:
            return ToolResult(
                data=None, success=False,
                error="format_ref 需要提供 ref_data 参数",
            )

        formatted = self._checker.format_reference(args.ref_data, args.journal)
        return ToolResult(
            data={
                "journal": args.journal,
                "formatted": formatted,
            },
            success=True,
        )

    def _handle_checklist(self, args: PaperToolInput) -> ToolResult:
        if not args.journal:
            return ToolResult(
                data=None, success=False,
                error="checklist 需要提供 journal 参数",
            )
        checklist = self._checker.generate_submission_checklist(args.journal)
        return ToolResult(
            data={
                "journal": args.journal,
                "checklist": checklist,
            },
            success=True,
        )

    def _handle_search(self, args: PaperToolInput) -> ToolResult:
        if not args.query:
            return ToolResult(
                data=None, success=False,
                error="search 需要提供 query 参数",
            )
        specs = search_journals(args.query)
        return ToolResult(
            data={
                "query": args.query,
                "count": len(specs),
                "journals": [self._spec_to_dict(s) for s in specs],
            },
            success=True,
        )

    def _handle_list(self, args: PaperToolInput) -> ToolResult:
        specs = list_journals(args.field)
        return ToolResult(
            data={
                "field": args.field,
                "count": len(specs),
                "journals": [self._spec_to_dict(s) for s in specs],
            },
            success=True,
        )

    def _handle_ref_format(self, args: PaperToolInput) -> ToolResult:
        if not args.journal:
            return ToolResult(
                data=None, success=False,
                error="ref_format 需要提供 journal 参数",
            )
        fmt_desc = get_reference_format(args.journal)
        spec = get_journal(args.journal)
        return ToolResult(
            data={
                "journal": args.journal,
                "format_key": spec.reference_format if spec else None,
                "description": fmt_desc,
            },
            success=True,
        )

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _spec_to_dict(spec: JournalSpec) -> dict[str, Any]:
        """JournalSpec -> dict, 去掉 None 值精简输出."""
        d = asdict(spec)
        return {k: v for k, v in d.items() if v is not None and v != []}
