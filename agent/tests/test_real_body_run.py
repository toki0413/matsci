"""真实身体接入变体测试.

与纯 `analyze()` 不同, 这里走真实生产代码路径:
  _get_model (注入假身体) → _llm_invoke → ainvoke → _parse_json → paper 映射
  → _annotate_value_consistency → consensus 组装

验证:
  R1 真实路径端到端能跑通, 产出结构化字段 (consensus/consistency/reported)
  R2 换不同"身体"(seed 变) 跑真实路径, 稳健中心仍落在真值邻域
  R3 同 seed 两次运行, 产物逐字节一致 (可复现)

全部离线, 不碰网络。
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.experimental.body_swap_discrimination import _synthetic_papers
from huginn.experimental.real_body_run import run_real_body


@pytest.fixture(autouse=True)
def _offline_literature(monkeypatch: pytest.MonkeyPatch) -> None:
    """强制本模块离线: 与 web_search_tool 共用 HUGINN_DISABLE_WEB_SEARCH 开关.

    本文件宣称"不碰网络", 但 benchmark_lookup 的缺度补全检索
    (_do_complement_retrieval) 缺省会对 15 个学术源发起真实网络请求 —— 既违背
    文件自身的离线契约, 也会在 CI 上因单个源挂起而超时. 全局禁用后该步降级为
    "未执行", 主生产路径 (抽取/取证/一致性/共识) 仍完整跑通.
    """
    monkeypatch.setenv("HUGINN_DISABLE_WEB_SEARCH", "1")


def _subset(system: str = "Li2O", property_: str = "band_gap", seed: int = 5):
    return [
        p
        for p in _synthetic_papers(seed)
        if p["system"] == system and p["property"] == property_
    ]


class TestRealBodyEndToEnd:
    def test_production_path_produces_structure(self):
        papers = _subset()
        data = asyncio.run(
            run_real_body(
                papers=papers,
                system="Li2O",
                property_="band_gap",
                seed=1,
                noise_sd=0.04,
                drop=0.0,
            )
        )
        assert data["consensus"] is not None
        assert "median" in data["consensus"]
        assert "consistency" in data
        assert "overall" in data["consistency"]
        assert data["reported_values"]  # 至少抽到一些值

    def test_no_papers_errors_gracefully(self):
        data = asyncio.run(
            run_real_body(
                papers=[],
                system="Li2O",
                property_="band_gap",
                seed=1,
            )
        )
        # 空输入不应崩溃; 走的是 not papers 分支
        assert data or data == {}


class TestRealBodyCrossBody:
    def test_stable_across_bodies(self):
        papers = _subset()
        truth_median = papers[0]["value"]  # 合成真值围绕它
        medians = []
        for seed in (1, 2, 3, 4, 5):
            data = asyncio.run(
                run_real_body(
                    papers=papers,
                    system="Li2O",
                    property_="band_gap",
                    seed=seed,
                    noise_sd=0.06,
                    drop=0.15,
                )
            )
            if data.get("consensus"):
                medians.append(data["consensus"]["median"])
        assert len(medians) >= 3  # 至少几个身体交出中心
        for m in medians:
            tol = max(abs(truth_median) * 0.18, 0.05)
            assert abs(m - truth_median) <= tol


class TestRealBodyReproducible:
    def test_same_seed_identical(self):
        papers = _subset()
        a = asyncio.run(
            run_real_body(
                papers=papers,
                system="Li2O",
                property_="band_gap",
                seed=99,
                noise_sd=0.04,
                drop=0.1,
            )
        )
        b = asyncio.run(
            run_real_body(
                papers=papers,
                system="Li2O",
                property_="band_gap",
                seed=99,
                noise_sd=0.04,
                drop=0.1,
            )
        )
        assert a == b
