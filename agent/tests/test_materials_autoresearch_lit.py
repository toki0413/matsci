"""Unit tests for materials_autoresearch_tool.py — 文献接入部分 (不跑 ratchet 循环).

覆盖: 文献检索 query 构造、literature_tool 未注册降级、findings 抽取、
LLM 报告失败时兜底报告含文献背景.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.materials_autoresearch_tool import (
    MaterialsAutoResearchInput,
    MaterialsAutoResearchTool,
)


def _make_args(**kw) -> MaterialsAutoResearchInput:
    base = {
        "research_goal": "minimize formation energy of Li7La3Zr2O12",
        "ratchet_metric": "formation_energy",
    }
    base.update(kw)
    return MaterialsAutoResearchInput(**base)


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


class TestLiteratureQuery:
    def test_goal_with_action_word_stripped(self):
        args = _make_args()
        # 'minimize ' 被剥离; 目标已含 'formation energy', 不重复追加
        assert MaterialsAutoResearchTool._literature_query(args) == (
            "formation energy of Li7La3Zr2O12"
        )

    def test_goal_metric_already_present_no_duplication(self):
        args = _make_args(
            research_goal="maximize conductivity of Li3PS4",
            ratchet_metric="conductivity",
        )
        assert (
            MaterialsAutoResearchTool._literature_query(args) == "conductivity of Li3PS4"
        )

    def test_metric_appended_when_absent(self):
        args = _make_args(
            research_goal="study the band gap of TiO2",
            ratchet_metric="formation_energy",
        )
        assert MaterialsAutoResearchTool._literature_query(args) == (
            "study the band gap of TiO2 formation energy"
        )


class TestGatherLiterature:
    @pytest.mark.asyncio
    async def test_tool_not_registered_degrades(self):
        tool = MaterialsAutoResearchTool()
        with patch(
            "huginn.tools.materials_autoresearch_tool.ToolRegistry.get",
            return_value=None,
        ):
            findings, note = await tool._gather_literature(_make_args(), _ctx())
        assert findings == []
        assert "未注册" in note

    @pytest.mark.asyncio
    async def test_findings_extracted_from_papers(self):
        fake = AsyncMock()
        fake.call.return_value = ToolResult(
            data={
                "papers": [
                    {
                        "title": "LLZO formation energy",
                        "abstract": "We compute the formation energy...",
                        "year": 2021,
                        "doi": "10.1000/xyz",
                        "source": "arxiv",
                    },
                    {"title": "Unrelated, no abstract"},  # 无 abstract 也要收
                    {"title": ""},  # 无 title 丢弃
                ]
            },
            success=True,
        )
        tool = MaterialsAutoResearchTool()
        with patch(
            "huginn.tools.materials_autoresearch_tool.ToolRegistry.get",
            return_value=fake,
        ):
            findings, note = await tool._gather_literature(_make_args(), _ctx())

        assert len(findings) == 2
        assert findings[0]["title"] == "LLZO formation energy"
        assert findings[0]["doi"] == "10.1000/xyz"
        assert findings[0]["year"] == 2021
        assert findings[0]["abstract"].startswith("We compute")
        assert "2 篇" in note

    @pytest.mark.asyncio
    async def test_failure_degrades_gracefully(self):
        fake = AsyncMock()
        fake.call.side_effect = RuntimeError("offline")
        tool = MaterialsAutoResearchTool()
        with patch(
            "huginn.tools.materials_autoresearch_tool.ToolRegistry.get",
            return_value=fake,
        ):
            findings, note = await tool._gather_literature(_make_args(), _ctx())
        assert findings == []
        assert "失败" in note

    @pytest.mark.asyncio
    async def test_unsuccessful_result_degrades(self):
        fake = AsyncMock()
        fake.call.return_value = ToolResult(data=None, success=False, error="boom")
        tool = MaterialsAutoResearchTool()
        with patch(
            "huginn.tools.materials_autoresearch_tool.ToolRegistry.get",
            return_value=fake,
        ):
            findings, note = await tool._gather_literature(_make_args(), _ctx())
        assert findings == []
        assert "未成功" in note


class TestReportFallback:
    @pytest.mark.asyncio
    async def test_report_fallback_includes_literature(self):
        """LLM 失败时兜底报告含研究目标 + 文献背景."""
        tool = MaterialsAutoResearchTool()
        ratchet_state = {
            "best_metric": -5.2,
            "best_params": {"encut": 500},
            "best_iteration": 3,
            "history": [{"i": 1}, {"i": 2}],
        }
        with patch.object(tool, "_get_model", side_effect=RuntimeError("no llm")):
            report = await tool._generate_report(
                _make_args(),
                ratchet_state,
                converged=True,
                lit_findings=[
                    {"title": "LLZO formation energy", "year": 2021, "source": "arxiv"}
                ],
                lit_note="检索到 1 篇相关文献",
            )
        assert "研究目标: minimize formation energy of Li7La3Zr2O12" in report
        assert "LLZO formation energy" in report
        assert "(LLM 报告生成失败" in report
