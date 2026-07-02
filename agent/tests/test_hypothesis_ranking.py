"""Phase 4a 测试: 假设排序 (novelty + feasibility + kb_relevance)."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from huginn.tools.hypothesis_generator_tool import (
    HypothesisGeneratorInput,
    HypothesisGeneratorTool,
)
from huginn.types import ToolContext


def _ctx(workspace: Path) -> ToolContext:
    return ToolContext(
        session_id="test_ranking",
        workspace=str(workspace),
        config=None,
    )


def _make_hyp(statement: str, required_data: str = "") -> dict:
    return {
        "statement": statement,
        "rationale": "test rationale",
        "testable_prediction": "pred",
        "required_data": required_data,
    }


# ── novelty ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_novelty_high_when_no_history(tmp_path):
    """长期记忆里没存过类似假设 → novelty=1.0."""
    tool = HypothesisGeneratorTool()
    ctx = _ctx(tmp_path)
    h = _make_hyp("GaN p-type doping with Mg acceptors")
    score = await tool._score_novelty(h, ctx)
    assert score == 1.0


@pytest.mark.asyncio
async def test_novelty_low_when_history_hit(tmp_path):
    """历史里塞 3 条相似假设 → novelty 跌到 0.2."""
    from huginn.memory.longterm import LongTermMemory

    mem = LongTermMemory(db_path=str(tmp_path / "memory.db"))
    # 用跟查询语句共享 token 的内容, FTS5 是 implicit AND, 得让查询词都命中
    query_statement = "GaN p-type doping with Mg acceptors"
    for i in range(3):
        mem.store(
            content=f"{query_statement} variant {i}",
            category="hypothesis",
            tags=["test"],
            importance=0.5,
            tier="mid",
        )
    tool = HypothesisGeneratorTool()
    ctx = _ctx(tmp_path)
    h = _make_hyp(query_statement)
    score = await tool._score_novelty(h, ctx)
    assert score <= 0.5  # 至少有 hit, 分数应该掉下来


# ── feasibility ──────────────────────────────────────────────────────


def test_feasibility_all_tools_available():
    """required_data 提到的工具都在 registry → feasibility=1.0."""
    tool = HypothesisGeneratorTool()
    h = _make_hyp("test", required_data="run VASP DFT calculation")
    with patch(
        "huginn.tools.registry.ToolRegistry.list_tools",
        return_value=["vasp_tool", "lammps_tool", "database_tool"],
    ):
        score = tool._score_feasibility(h)
    assert score == 1.0


def test_feasibility_missing_one_tool():
    """缺一个工具 → feasibility=0.5."""
    tool = HypothesisGeneratorTool()
    h = _make_hyp("test", required_data="run VASP DFT and LAMMPS MD")
    with patch(
        "huginn.tools.registry.ToolRegistry.list_tools",
        return_value=["vasp_tool"],  # 缺 lammps_tool
    ):
        score = tool._score_feasibility(h)
    assert score == 0.5


def test_feasibility_no_tool_keywords():
    """required_data 没抠到工具关键词 → 中性分 0.8."""
    tool = HypothesisGeneratorTool()
    h = _make_hyp("test", required_data="just some abstract reasoning")
    with patch(
        "huginn.tools.registry.ToolRegistry.list_tools",
        return_value=["vasp_tool"],
    ):
        score = tool._score_feasibility(h)
    assert score == 0.8


# ── kb_relevance ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kb_relevance_unavailable_returns_neutral(tmp_path):
    """KB 不可用 (没装 chromadb) → 中性分 0.5, 不报错."""
    tool = HypothesisGeneratorTool()
    ctx = _ctx(tmp_path)
    h = _make_hyp("test statement")
    # get_knowledge_base 会抛 ImportError 或其他异常, 应被吞掉
    with patch(
        "huginn.knowledge.store.get_knowledge_base",
        side_effect=RuntimeError("chromadb not installed"),
    ):
        score = await tool._score_kb_relevance(h, ctx)
    assert score == 0.5


# ── 整体排序 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_sorts_by_score_descending(tmp_path):
    """三个假设打分不同, 排序后 score 应该递减."""
    tool = HypothesisGeneratorTool()
    ctx = _ctx(tmp_path)
    candidates = [
        _make_hyp("low score", required_data=""),
        _make_hyp("high score vasp", required_data="run VASP DFT"),
        _make_hyp("mid score", required_data="run VASP"),
    ]
    # mock 三个子项打分, 跳过实际依赖
    async def fake_novelty(h, ctx):
        return 0.8

    def fake_feasibility(h):
        # 让 "high score vasp" 拿最高分
        if "high" in h["statement"]:
            return 1.0
        if "mid" in h["statement"]:
            return 0.6
        return 0.2

    async def fake_kb(h, ctx):
        return 0.5

    with patch.object(tool, "_score_novelty", side_effect=fake_novelty), \
         patch.object(tool, "_score_feasibility", side_effect=fake_feasibility), \
         patch.object(tool, "_score_kb_relevance", side_effect=fake_kb):
        ranked = await tool._rank_hypotheses(candidates, ctx)

    assert len(ranked) == 3
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True), f"scores not descending: {scores}"
    assert ranked[0]["statement"] == "high score vasp"
    assert ranked[-1]["statement"] == "low score"
    # 每个假设都带分数子字段
    for r in ranked:
        assert "novelty" in r
        assert "feasibility" in r
        assert "kb_relevance" in r
