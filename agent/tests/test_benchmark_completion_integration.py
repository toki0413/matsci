"""缺度追问与 benchmark_lookup 的对接集成测试.

覆盖用户要求的两点:
  (1) 缺度追问接进 `_do_benchmark_lookup`:
      reported 非空时返回 compat (local/global/completion), 含缺失自由度 +
      按 urgency 分级的补全查询 (followup).
  (2) 接入实际检索执行 (`_do_complement_retrieval`):
      首轮有关键缺度时, 把 target_query 喂给 _do_search → 再抽一轮报道值 →
      合并重跑局部-整体 → 补后校验 (missing_dims/verdict 补前补后对比).

全部 mock 掉 LLM 与 _do_search, 卡死确定性输入, 不碰真实网络.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.literature.tool import LiteratureInput, LiteratureTool


def _paper(title, doi, abstract):
    return {
        "title": title,
        "doi": doi,
        "year": 2022,
        "venue": "J. Fake Mater.",
        "abstract": abstract,
    }


_PAPERS_MAIN = [
    _paper(
        "PBE band gap of Li2O",
        "10.9/bb.pbe",
        "GGA gives 4.1 eV for the band gap of Li2O.",
    ),
    _paper(
        "Experimental Li2O gap",
        "10.9/bb.exp",
        "The measured Li2O band gap is 5.81 eV at room temperature.",
    ),
    _paper(
        "Hybrid functional Li2O",
        "10.9/bb.hse",
        "HSE06 predicts 5.98 eV for the Li2O band gap.",
    ),
]

# 首轮: PBE / experiment(room temp) / HSE06 → 只缺 temperature
_MAIN_LLM = json.dumps({
    "values": [
        {"value": 4.10, "unit": "eV", "method": "DFT-PBE", "paper_idx": 1, "note": ""},
        {"value": 5.81, "unit": "eV", "method": "experiment", "paper_idx": 2, "note": "room temperature"},
        {"value": 5.98, "unit": "eV", "method": "HSE06", "paper_idx": 3, "note": ""},
    ]
})

# 补全轮: 补充一个带温度的报道值 (走 _do_complement_retrieval 的二次抽取)
_COMP_LLM = json.dumps({
    "values": [
        {"value": 5.80, "unit": "eV", "method": "experiment", "paper_idx": 1, "note": "room temperature"},
    ]
})

_PAPERS_COMP = [_paper("Li2O optical gap measured", "10.9/bb.comp", "5.80 eV at RT.")]


class _FakeKB:
    def __init__(self):
        self.added = 0

    def add_text(self, text, filename="auto", metadata=None):
        self.added += 1
        return {"doc_id": "1", "chunks": 1}


def _build_tool(monkeypatch, *, comp_papers=None):
    """构造已 mock 好 LLM 与 _do_search 的 LiteratureTool."""
    tool = LiteratureTool()
    monkeypatch.setattr(tool, "_get_model", lambda ctx: MagicMock())
    monkeypatch.setattr(
        tool,
        "_llm_invoke",
        AsyncMock(side_effect=[_MAIN_LLM, _COMP_LLM]),
    )
    search_res = ToolResult(
        data={"papers": comp_papers or []}, success=True
    )
    monkeypatch.setattr(tool, "_do_search", AsyncMock(return_value=search_res))
    monkeypatch.setattr(
        "huginn.knowledge.store.get_knowledge_base", lambda *a, **k: _FakeKB()
    )
    return tool


def _run(tool, system="Li2O", property="band gap", papers=_PAPERS_MAIN):
    args = LiteratureInput(
        action="benchmark_lookup",
        system=system,
        property=property,
        papers=papers,
    )
    return asyncio.run(
        tool.call(args, ToolContext(session_id="t", workspace="/tmp"))
    )


class TestBenchmarkFetchesCompletion:
    def test_missing_dim_is_flagged_and_queried(self, tmp_path, monkeypatch):
        tool = _build_tool(monkeypatch, comp_papers=_PAPERS_COMP)
        result = _run(tool)

        assert result.success is True
        # 对接1: compat 与 completion 字段都在
        assert result.data["compat"] is not None
        completion = result.data["completion"]
        assert "temperature" in completion["missing_dims"]
        assert completion["missing_summaries"]["temperature"]
        # followup: 温度缺失 (urgency 3) → 生成了对应补全查询
        fills = completion["followup"]["fills"]
        assert fills and fills[0]["target_dim"] == "temperature"
        assert fills[0]["query"] and "Li2O" in fills[0]["query"]

        # 对接2: 实际补全检索执行了, 并带补后校验
        comp = completion["complement_retrieval"]
        assert comp["executed"] is True
        assert len(comp["new_reported"]) == 1
        assert comp["n_total_sources"] == 4
        # 补前/补后缺度与判定都记录
        assert "missing_dims_before" in comp
        assert "overall_verdict_before" in comp
        assert "revised" in comp

    def test_no_missing_dims_skips_complement(self, tmp_path, monkeypatch):
        # 三条值都声明了温度 → 不再缺 temperature, 不应发真实补全检索
        clean_llm = json.dumps({
            "values": [
                {"value": 5.80, "unit": "eV", "method": "experiment", "paper_idx": 1, "note": "room temperature"},
                {"value": 5.81, "unit": "eV", "method": "experiment", "paper_idx": 2, "note": "room temperature"},
            ]
        })
        tool = LiteratureTool()
        monkeypatch.setattr(tool, "_get_model", lambda ctx: MagicMock())
        monkeypatch.setattr(
            tool, "_llm_invoke", AsyncMock(return_value=clean_llm)
        )
        search = AsyncMock()
        monkeypatch.setattr(tool, "_do_search", search)
        monkeypatch.setattr(
            "huginn.knowledge.store.get_knowledge_base", lambda *a, **k: _FakeKB()
        )
        papers = _PAPERS_MAIN[:2]

        result = _run(tool, papers=papers)

        assert result.success is True
        completion = result.data["completion"]
        # functional/method_family 已知, 温度已知 → 无应发起的补全
        assert completion["missing_dims"] == []
        assert completion["followup"]["fills"] == []
        assert "complement_retrieval" not in completion
        # 绝不碰网络补全检索
        search.assert_not_called()