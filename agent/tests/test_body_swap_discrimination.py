"""判别实验测试: Agent 是否只是"可替换的 LLM 身体"?

锁定三条不变量 (body_swap_discrimination):
  I1 结构可复现 — 确定性组合层同输入两次运行逐字节一致
  I2 跨身体稳健 — 换不同"身体"抽值, 稳健中位数与 verdict 不随身体漂移
  I3 结构独有性 — 裸 LLM 自由文本产不出 agent 的结构化产物 (来源/verdict/fid)

全部离线纯函数, 不碰网络、不调真实 LLM。
"""

from __future__ import annotations

from huginn.experimental.body_swap_discrimination import (
    _synthetic_papers,
    analyze,
    invariant_cross_body,
    invariant_reproducible,
    invariant_structure_only_in_agent,
    run_all,
)

TRUE_VERDICTS = None  # 占位保证 module import 干净


def _clean_reported():
    papers = _synthetic_papers(seed=42)
    return papers


# ── I1: 结构可复现 ───────────────────────────────────────────────────────


class TestInvariantReproducible:
    def test_same_input_same_output(self):
        reported = [
            {"value": 5.2, "unit": "eV"},
            {"value": 5.25, "unit": "eV"},
            {"value": 5.15, "unit": "eV"},
        ]
        assert invariant_reproducible(reported) is True

    def test_analyze_is_deterministic(self):
        reported = [
            {"value": 5.2, "unit": "eV"},
            {"value": 5.25, "unit": "eV"},
            {"value": 5.15, "unit": "eV"},
        ]
        assert analyze(reported) == analyze(reported)


# ── I2: 跨身体稳健 ───────────────────────────────────────────────────────


class TestInvariantCrossBody:
    def test_many_bodies_stay_stable(self):
        papers = _synthetic_papers(seed=7)
        # 只保留单个体系 Li2O band_gap (eV) — 单一 unit, 跨身体对比才语义正确
        papers = [
            p for p in papers if p["system"] == "Li2O" and p["property"] == "band_gap"
        ]
        assert len(papers) >= 3
        assert invariant_cross_body(papers, bodies=8) is True

    def test_no_truth_never_passes(self):
        # 只有两个极端身体 -> 中位数组离散过大, 应返回 False (稳健性拒绝)
        papers = [
            {
                "system": "X",
                "property": "p",
                "unit": "eV",
                "value": 1.0,
                "source_paper": "a",
                "doi": "1",
                "year": 2020,
            },
            {
                "system": "X",
                "property": "p",
                "unit": "eV",
                "value": 9.0,
                "source_paper": "b",
                "doi": "2",
                "year": 2020,
            },
        ]
        # 单 body 抽到两个极端, 中位数 5, 但只是 1 个 body, 无法跨 body 验证
        assert invariant_cross_body(papers, bodies=1) is False


# ── I3: 结构独有性 ───────────────────────────────────────────────────────


class TestInvariantStructureOnlyInAgent:
    def test_structure_keys_present_in_agent_out(self):
        reported = [{"value": 5.2, "unit": "eV", "source_paper": "p1", "doi": "10.1/1"}]
        assert invariant_structure_only_in_agent(reported) is True


# ── 端到端: run_all ──────────────────────────────────────────────────────


class TestRunAll:
    def test_all_invariants_pass(self):
        report = run_all(bodies=6, seed=3)
        assert report["all_pass"] is True


def test_seeded_run_is_reproducible():
    a = run_all(bodies=5, seed=11)
    b = run_all(bodies=5, seed=11)
    assert a == b
