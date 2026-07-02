"""W4 集成测试 — W2+W3+W4 全链路端到端.

验证: 假设图 → campaign 编排 → 多保真 UQ 传播 → red-team 审查 →
      FAIR provenance 记录 + ROCrate 导出.

不调真 LLM / 求解器, 全用 mock runner 模拟实验结果, 重点验证模块间
接线正确: R5 graph → R4 campaign → M3 UQ → R3 red-team → M4 provenance.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from huginn.autoloop.campaign import CampaignManager
from huginn.autoloop.hypothesis_loop import HypothesisGraph
from huginn.autoloop.red_team import RedTeamReviewer
from huginn.autoloop.uq_propagation import UQPipeline, UQStage
from huginn.provenance import (
    ProvenanceLogger,
    ProvenanceRecord,
    capture,
    capture_run_inputs,
    export_crate,
)


# ── Test 1: 假设→实验→UQ→provenance 全链路 ──────────────────────────────────


class TestFullChainSingleExperiment:
    """单个实验的完整链路: 假设 → 实验 → UQ 传播 → provenance 记录."""

    def test_single_supported_experiment(self, tmp_path):
        """实验支持假设 → UQ 传播误差棒 → provenance 记录 + crate 导出."""
        # 1. 建 campaign + 假设
        mgr = CampaignManager()
        cid = mgr.create_campaign("带隙优化", "找最小带隙的掺杂浓度")
        graph = mgr.campaign(cid).graph
        h1 = graph.add_hypothesis(
            "如果掺杂浓度增加, 则带隙单调减小, 在所有温度范围内成立",
            rationale="文献表明掺杂会引入带隙窄化效应",
            testable_prediction="带隙随掺杂浓度从 0% 到 10% 单调减小",
        )
        mgr.add_experiment(cid, hypothesis_id=h1, objective="DFT 计算带隙")

        # 2. mock runner: 模拟 DFT 结果 + UQ 传播
        async def dft_runner(exp):
            # 模拟 UQ pipeline: encut 误差 → bandgap 误差
            pipe = UQPipeline()
            pipe.add_stage(UQStage(name="encut", value=520.0, sigma=5.0))
            pipe.add_stage(UQStage(name="kpoint_err", value=0.02, sigma=0.005))
            pipe.add_stage(UQStage(
                name="bandgap",
                expression="0.5 * encut / 1000 + 10 * kpoint_err",
                dependencies=["encut", "kpoint_err"],
                method="monte_carlo",
                n_samples=500,
            ))
            uq_results = pipe.run(seed=42)
            return {
                "success": True,
                "hypothesis_supported": True,
                "evidence": {
                    "bandgap": uq_results["bandgap"].value,
                    "bandgap_sigma": uq_results["bandgap"].sigma,
                    "uq_method": "monte_carlo",
                },
            }

        result = asyncio.run(mgr.run_campaign(cid, runner=dft_runner, mode="serial"))

        # 3. 验证假设状态
        assert result["n_completed"] == 1
        assert result["hypotheses"]["supported"] == 1
        assert graph.get(h1).status == "supported"
        assert "bandgap" in graph.get(h1).evidence
        assert graph.get(h1).evidence["bandgap_sigma"] > 0

        # 4. 记 provenance
        logger = ProvenanceLogger(path=str(tmp_path / "prov.jsonl"))
        record = ProvenanceRecord(
            run_id=result["campaign_id"],
            objective="带隙优化",
            inputs=capture_run_inputs(params={"encut": 520, "kpoints": "2x2x2"}),
            outputs={"bandgap": graph.get(h1).evidence["bandgap"]},
            timestamps={"start": "2026-07-01T00:00:00Z", "end": "2026-07-01T01:00:00Z"},
            dois=["10.1234/bandgap_study"],
            tags=["dft", "bandgap"],
        )
        record.add_snapshot(capture("vasp_tool", {"encut": 520}))
        record.add_snapshot(capture("uq_tool", {"method": "monte_carlo"}))
        logger.log(record)

        # 5. 读回 + 导 crate
        loaded = logger.read_run(result["campaign_id"])
        assert len(loaded) == 1
        crate = export_crate(loaded[0])
        assert crate["@graph"][0]["name"] == "带隙优化"
        tools = [e for e in crate["@graph"] if e.get("@type") == "SoftwareApplication"]
        assert any(t["name"] == "vasp_tool" for t in tools)
        assert any(t["name"] == "uq_tool" for t in tools)


# ── Test 2: 反驳→修正闭环 ───────────────────────────────────────────────────


class TestRefuteRefineLoop:
    """反驳 → red-team 审查 → 修正假设 → 新实验 → 支持."""

    def test_refute_then_refine_then_support(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("掺杂研究", "验证掺杂对带隙的影响")
        graph = mgr.campaign(cid).graph
        h1 = graph.add_hypothesis(
            "如果掺杂浓度增加, 则带隙单调减小, 在所有温度范围内都成立, 不受温度影响",
            testable_prediction="带隙随掺杂浓度单调减小",
        )
        mgr.add_experiment(cid, hypothesis_id=h1, objective="DFT 计算")

        call_count = {"n": 0}

        async def mixed_runner(exp):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # 第一次: 反驳
                return {
                    "success": True,
                    "hypothesis_supported": False,
                    "evidence": {"result": "高掺杂下带隙反而增加", "doping": "10%"},
                }
            # 第二次: 支持
            return {
                "success": True,
                "hypothesis_supported": True,
                "evidence": {"result": "低掺杂下带隙确实减小", "p": 0.02},
            }

        result = asyncio.run(mgr.run_campaign(cid, runner=mixed_runner, mode="serial"))

        # 验证: 2 个实验, h1 被 superseded, 新假设 supported
        assert result["n_experiments"] == 2
        assert result["n_completed"] == 2
        assert graph.get(h1).status == "superseded"
        supported = graph.supported()
        assert len(supported) == 1
        # 修正假设的 rationale 应该提到 h1
        assert h1 in supported[0].rationale
        # refinement_basis 应该有 red-team findings
        assert len(supported[0].refinement_basis) > 0

    def test_refine_chain_with_provenance(self, tmp_path):
        """连续修正 + 每步记 provenance."""
        mgr = CampaignManager()
        cid = mgr.create_campaign("多轮修正", "迭代优化假设")
        graph = mgr.campaign(cid).graph
        h1 = graph.add_hypothesis(
            "假设 v1, 在所有条件下都成立, 不受任何参数影响",
            testable_prediction="预测 A",
        )
        mgr.add_experiment(cid, hypothesis_id=h1, objective="验证 v1")

        call_count = {"n": 0}
        logger = ProvenanceLogger(path=str(tmp_path / "prov.jsonl"))

        async def refining_runner(exp):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # 前两次反驳
                result = {
                    "success": True,
                    "hypothesis_supported": False,
                    "evidence": {"round": call_count["n"]},
                }
            else:
                # 第三次支持
                result = {
                    "success": True,
                    "hypothesis_supported": True,
                    "evidence": {"round": call_count["n"], "p": 0.01},
                }
            # 每步记 provenance
            record = ProvenanceRecord(
                run_id=f"{cid}_step{call_count['n']}",
                objective=f"实验 {exp.id}",
                tool_chain=[],
            )
            record.add_snapshot(capture("mock_tool", {"round": call_count["n"]}))
            logger.log(record)
            return result

        result = asyncio.run(mgr.run_campaign(cid, runner=refining_runner, mode="serial"))

        # 3 个实验, 2 个 superseded + 1 个 supported
        assert result["n_experiments"] == 3
        chain = graph.derivation_chain(graph.supported()[0].id)
        assert len(chain) == 3  # h1 → h2 → h3

        # provenance 记了 3 条
        all_records = logger.read_all()
        assert len(all_records) == 3


# ── Test 3: 多保真 + UQ 传播 + red-team 联动 ──────────────────────────────────


class TestMultiFidelityUQRedTeam:
    """多保真数据 → UQ 传播 → red-team 审查假设."""

    def test_uq_pipeline_feeds_red_team(self):
        """UQ 传播的 sigma 太大 → red-team 应该标记假设审查问题."""
        # 1. UQ pipeline: 模拟大不确定度的带隙预测
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="encut", value=400.0, sigma=20.0))  # 大误差
        pipe.add_stage(UQStage(name="bandgap", expression="encut / 100", dependencies=["encut"]))
        uq_results = pipe.run(seed=1)

        # 2. red-team 审查假设 (带大不确定度证据)
        reviewer = RedTeamReviewer()
        hypothesis = "如果掺杂增加, 则带隙减小, 在所有条件下成立"
        evidence = {
            "hypothesis": hypothesis,
            "bandgap": uq_results["bandgap"].value,
            "bandgap_sigma": uq_results["bandgap"].sigma,
        }
        report = reviewer.review("hypothesize", "plan", evidence)

        # 3. 假设较长没提边界条件 → red-team 应该发现 hidden_assumption
        finding_categories = {f.category for f in report.findings}
        assert "hidden_assumption" in finding_categories or len(report.findings) > 0

    def test_uq_linear_vs_mc_consistency(self):
        """同一模型 linear 和 MC 传播应该接近 (线性函数)."""
        pipe_linear = UQPipeline()
        pipe_linear.add_stage(UQStage(name="x", value=5.0, sigma=1.0))
        pipe_linear.add_stage(UQStage(
            name="y", expression="3 * x + 2", dependencies=["x"], method="linear",
        ))
        res_lin = pipe_linear.run()

        pipe_mc = UQPipeline()
        pipe_mc.add_stage(UQStage(name="x", value=5.0, sigma=1.0))
        pipe_mc.add_stage(UQStage(
            name="y", expression="3 * x + 2", dependencies=["x"],
            method="monte_carlo", n_samples=20000,
        ))
        res_mc = pipe_mc.run(seed=42)

        # 线性函数: linear sigma = 3*1 = 3, MC 应该接近
        assert res_lin["y"].sigma == pytest.approx(3.0, abs=1e-6)
        assert res_mc["y"].sigma == pytest.approx(3.0, abs=0.2)
        assert res_lin["y"].value == pytest.approx(res_mc["y"].value, abs=0.1)


# ── Test 4: 并行 campaign + provenance 批量记录 ──────────────────────────────


class TestParallelCampaignWithProvenance:
    """并行跑 3 个实验 + 批量记 provenance + 导 crate."""

    def test_parallel_campaign_provenance(self, tmp_path):
        mgr = CampaignManager()
        cid = mgr.create_campaign("高通量筛选", "并行筛选 3 个掺杂体系")
        graph = mgr.campaign(cid).graph
        h1 = graph.add_hypothesis("体系 A 的带隙 > 1eV, 在标准条件下成立")
        h2 = graph.add_hypothesis("体系 B 的带隙 > 1eV, 在标准条件下成立")
        h3 = graph.add_hypothesis("体系 C 的带隙 > 1eV, 在标准条件下成立")
        mgr.add_experiment(cid, h1, "DFT A")
        mgr.add_experiment(cid, h2, "DFT B")
        mgr.add_experiment(cid, h3, "DFT C")

        async def parallel_runner(exp):
            return {
                "success": True,
                "hypothesis_supported": True,
                "evidence": {"bandgap": 1.5, "method": "DFT"},
            }

        result = asyncio.run(
            mgr.run_campaign(cid, runner=parallel_runner, mode="parallel")
        )
        assert result["n_completed"] == 3
        assert result["hypotheses"]["supported"] == 3

        # 批量记 provenance
        logger = ProvenanceLogger(path=str(tmp_path / "prov.jsonl"))
        for exp in mgr.campaign(cid).experiments.values():
            record = ProvenanceRecord(
                run_id=exp.id,
                objective=exp.objective,
                inputs=capture_run_inputs(params={"system": exp.hypothesis_id}),
                outputs=exp.result.get("evidence", {}),
            )
            record.add_snapshot(capture("vasp_tool", {"system": exp.hypothesis_id}))
            logger.log(record)

        # 读回 + 每个导 crate
        all_records = logger.read_all()
        assert len(all_records) == 3
        for r in all_records:
            crate = export_crate(r)
            assert "@context" in crate
            assert len(crate["@graph"]) >= 2  # root + 至少一个 tool


# ── Test 5: DAG campaign 依赖图 + 假设衍生 ───────────────────────────────────


class TestDagCampaignWithDerivation:
    """DAG 模式: e2 依赖 e1, e1 结果衍生新假设给 e2 验证."""

    def test_dag_with_dependency(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("分层验证", "先粗算再精算")
        graph = mgr.campaign(cid).graph
        h1 = graph.add_hypothesis("粗算假设, 在标准条件下成立")
        h2 = graph.add_hypothesis("精算假设, 在粗算基础上成立")

        e1 = mgr.add_experiment(cid, h1, "粗算")
        e2 = mgr.add_experiment(cid, h2, "精算", depends_on=[e1])

        execution_order = []

        async def dag_runner(exp):
            execution_order.append(exp.id)
            await asyncio.sleep(0.01)
            return {"success": True, "hypothesis_supported": True, "evidence": {}}

        result = asyncio.run(mgr.run_campaign(cid, runner=dag_runner, mode="dag"))

        assert result["n_completed"] == 2
        # e1 必须在 e2 之前
        assert execution_order.index(e1) < execution_order.index(e2)
        assert result["hypotheses"]["supported"] == 2


# ── Test 6: 序列化往返 — campaign 持久化 ─────────────────────────────────────


class TestCampaignSerialization:
    """campaign.to_dict() → 重建 → 验证假设图完整."""

    def test_campaign_dict_has_graph(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        graph = mgr.campaign(cid).graph
        h1 = graph.add_hypothesis("假设 A, 在所有条件下成立")
        h2 = graph.add_hypothesis("衍生假设", parent_id=h1)
        mgr.add_experiment(cid, h1, "跑1")

        c = mgr.campaign(cid)
        d = c.to_dict()

        # 验证 graph 被序列化
        assert "graph" in d
        assert len(d["graph"]["nodes"]) == 2
        # 验证 derive 边
        derive_edges = [e for e in d["graph"]["edges"] if e["edge_type"] == "derive"]
        assert len(derive_edges) == 1

        # 重建 graph
        graph2 = HypothesisGraph.from_dict(d["graph"])
        assert len(graph2.all_nodes()) == 2
        assert len(graph2.children(h1)) == 1
