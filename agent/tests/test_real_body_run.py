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

from huginn.experimental.body_swap_discrimination import _synthetic_papers
from huginn.experimental.real_body_run import run_real_body


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
