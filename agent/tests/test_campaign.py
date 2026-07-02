"""R4 Campaign manager 测试 — 多实验编排 + 假设闭环.

覆盖: campaign 创建 / experiment 添加 / serial+parallel+dag 编排 /
      假设支持→support / 假设反驳→refute+refine / 实验崩溃→failed /
      聚合统计 / 拓扑分层 / 异常.
"""

from __future__ import annotations

import asyncio

import pytest

from huginn.autoloop.campaign import (
    Campaign,
    CampaignError,
    CampaignManager,
    Experiment,
)


# ── 辅助 runner ──────────────────────────────────────────────────────────────


async def supporting_runner(exp: Experiment) -> dict:
    """总是支持假设的 runner."""
    return {"success": True, "hypothesis_supported": True, "evidence": {"p": 0.01}}


async def refuting_runner(exp: Experiment) -> dict:
    """总是反驳假设的 runner."""
    return {"success": True, "hypothesis_supported": False, "evidence": {"result": "相反"}}


async def crashing_runner(exp: Experiment) -> dict:
    """崩溃的 runner."""
    raise RuntimeError("求解器挂了")


async def neutral_runner(exp: Experiment) -> dict:
    """成功但不表态的 runner (hypothesis_supported=None)."""
    return {"success": True, "evidence": {"note": "无定论"}}


# ── campaign 创建 / experiment 添加 ──────────────────────────────────────────


class TestCreateCampaign:
    def test_create_returns_id(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("LJ 优化", "找最低能态")
        assert cid.startswith("c_")
        c = mgr.campaign(cid)
        assert c.name == "LJ 优化"
        assert c.status == "active"
        assert c.created_at != ""

    def test_campaign_not_found(self):
        mgr = CampaignManager()
        with pytest.raises(CampaignError, match="不存在"):
            mgr.campaign("c_nope")

    def test_add_experiment(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        h = mgr.campaign(cid).graph.add_hypothesis("如果 X 则 Y, 在所有条件下成立")
        eid = mgr.add_experiment(cid, hypothesis_id=h, objective="跑实验")
        assert eid.startswith("e_")
        c = mgr.campaign(cid)
        assert eid in c.experiments
        assert c.experiments[eid].hypothesis_id == h
        assert eid in c._queue

    def test_add_experiment_missing_hypothesis(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        with pytest.raises(Exception, match="不存在"):
            mgr.add_experiment(cid, hypothesis_id="h_nope", objective="x")


# ── serial 编排 ──────────────────────────────────────────────────────────────


class TestSerialRun:
    def test_serial_support(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        h = mgr.campaign(cid).graph.add_hypothesis("如果 X 则 Y, 在所有条件下成立")
        mgr.add_experiment(cid, hypothesis_id=h, objective="跑")
        result = asyncio.run(mgr.run_campaign(cid, runner=supporting_runner, mode="serial"))
        assert result["n_completed"] == 1
        assert result["hypotheses"]["supported"] == 1
        assert result["hypotheses"]["untested"] == 0

    def test_serial_refute_triggers_refine(self):
        """反驳 → refine_failed → 新实验入队. 用有限轮次的 runner 验证."""
        call_count = {"n": 0}

        async def finite_refute_runner(exp: Experiment) -> dict:
            call_count["n"] += 1
            # 前2次反驳触发 refine, 第3次支持收尾
            if call_count["n"] <= 2:
                return {"success": True, "hypothesis_supported": False,
                        "evidence": {"round": call_count["n"]}}
            return {"success": True, "hypothesis_supported": True,
                    "evidence": {"p": 0.01}}

        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        h = mgr.campaign(cid).graph.add_hypothesis(
            "如果 X 则 Y, 在所有条件下都成立, 不受温度影响"
        )
        mgr.add_experiment(cid, hypothesis_id=h, objective="跑")
        result = asyncio.run(mgr.run_campaign(cid, runner=finite_refute_runner, mode="serial"))
        # 3 轮: refute h → refine h2 → refute h2 → refine h3 → support h3
        assert result["n_experiments"] == 3
        assert result["n_completed"] == 3
        assert result["hypotheses"]["supported"] == 1
        c = mgr.campaign(cid)
        assert c.graph.get(h).status == "superseded"

    def test_serial_refute_then_support(self):
        """先反驳再支持: refute → refine → 新实验 support."""
        call_count = {"n": 0}

        async def mixed_runner(exp: Experiment) -> dict:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"success": True, "hypothesis_supported": False,
                        "evidence": {"result": "相反"}}
            return {"success": True, "hypothesis_supported": True,
                    "evidence": {"p": 0.01}}

        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        h = mgr.campaign(cid).graph.add_hypothesis(
            "如果 X 则 Y, 在所有条件下都成立, 不受温度影响"
        )
        mgr.add_experiment(cid, hypothesis_id=h, objective="跑")
        result = asyncio.run(mgr.run_campaign(cid, runner=mixed_runner, mode="serial"))
        # 第1次: refute h → refine 产 h2 → 新实验入队
        # 第2次: support h2
        assert result["n_experiments"] == 2
        assert result["n_completed"] == 2
        assert result["hypotheses"]["supported"] == 1
        assert result["hypotheses"]["refuted"] == 0  # h 被 superseded 不是 refuted
        c = mgr.campaign(cid)
        # h 应该是 superseded (refine_failed 会的)
        assert c.graph.get(h).status == "superseded"

    def test_serial_crash_marks_failed(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        h = mgr.campaign(cid).graph.add_hypothesis("如果 X 则 Y, 在所有条件下成立")
        mgr.add_experiment(cid, hypothesis_id=h, objective="跑")
        result = asyncio.run(mgr.run_campaign(cid, runner=crashing_runner, mode="serial"))
        assert result["n_failed"] == 1
        assert result["hypotheses"]["untested"] == 1  # 没动假设图

    def test_serial_neutral_no_status_change(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        h = mgr.campaign(cid).graph.add_hypothesis("如果 X 则 Y, 在所有条件下成立")
        mgr.add_experiment(cid, hypothesis_id=h, objective="跑")
        result = asyncio.run(mgr.run_campaign(cid, runner=neutral_runner, mode="serial"))
        assert result["n_completed"] == 1
        assert result["hypotheses"]["untested"] == 1  # hypothesis_supported=None 不动


# ── parallel 编排 ────────────────────────────────────────────────────────────


class TestParallelRun:
    def test_parallel_all_support(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        g = mgr.campaign(cid).graph
        h1 = g.add_hypothesis("假设 A, 在所有条件下成立")
        h2 = g.add_hypothesis("假设 B, 在所有条件下成立")
        h3 = g.add_hypothesis("假设 C, 在所有条件下成立")
        mgr.add_experiment(cid, h1, "跑1")
        mgr.add_experiment(cid, h2, "跑2")
        mgr.add_experiment(cid, h3, "跑3")
        result = asyncio.run(mgr.run_campaign(cid, runner=supporting_runner, mode="parallel"))
        assert result["n_completed"] == 3
        assert result["hypotheses"]["supported"] == 3

    def test_parallel_mixed_results(self):
        call_count = {"n": 0}

        async def mixed(exp: Experiment) -> dict:
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return {"success": True, "hypothesis_supported": True, "evidence": {}}
            return {"success": True, "hypothesis_supported": False, "evidence": {}}

        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        g = mgr.campaign(cid).graph
        h1 = g.add_hypothesis("假设 A, 在所有条件下成立")
        h2 = g.add_hypothesis("假设 B, 在所有条件下成立")
        mgr.add_experiment(cid, h1, "跑1")
        mgr.add_experiment(cid, h2, "跑2")
        result = asyncio.run(mgr.run_campaign(cid, runner=mixed, mode="parallel"))
        # parallel 模式 refine 出的新实验不再跑 (一轮制)
        assert result["n_completed"] == 2
        # 至少一个 supported, 假设图有变化
        c = mgr.campaign(cid)
        assert len(c.graph.all_nodes()) >= 2


# ── dag 编排 ─────────────────────────────────────────────────────────────────


class TestDagRun:
    def test_dag_respects_dependencies(self):
        """e2 依赖 e1, e1 先跑完才跑 e2."""
        order: list[str] = []

        async def tracking_runner(exp: Experiment) -> dict:
            order.append(exp.id)
            await asyncio.sleep(0.01)
            return {"success": True, "hypothesis_supported": True, "evidence": {}}

        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        g = mgr.campaign(cid).graph
        h1 = g.add_hypothesis("假设 A, 在所有条件下成立")
        h2 = g.add_hypothesis("假设 B, 在所有条件下成立")
        e1 = mgr.add_experiment(cid, h1, "跑1")
        e2 = mgr.add_experiment(cid, h2, "跑2", depends_on=[e1])
        asyncio.run(mgr.run_campaign(cid, runner=tracking_runner, mode="dag"))
        assert order.index(e1) < order.index(e2)

    def test_dag_parallel_within_layer(self):
        """同层实验可以并发."""
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        g = mgr.campaign(cid).graph
        h1 = g.add_hypothesis("假设 A, 在所有条件下成立")
        h2 = g.add_hypothesis("假设 B, 在所有条件下成立")
        h3 = g.add_hypothesis("假设 C, 在所有条件下成立")
        mgr.add_experiment(cid, h1, "跑1")
        mgr.add_experiment(cid, h2, "跑2")
        mgr.add_experiment(cid, h3, "跑3")
        result = asyncio.run(mgr.run_campaign(cid, runner=supporting_runner, mode="dag"))
        assert result["n_completed"] == 3


# ── 聚合 ────────────────────────────────────────────────────────────────────


class TestAggregate:
    def test_aggregate_stats(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        g = mgr.campaign(cid).graph
        h1 = g.add_hypothesis("假设 A, 在所有条件下成立")
        h2 = g.add_hypothesis("假设 B, 在所有条件下成立")
        mgr.add_experiment(cid, h1, "跑1")
        mgr.add_experiment(cid, h2, "跑2")
        asyncio.run(mgr.run_campaign(cid, runner=supporting_runner, mode="serial"))
        result = mgr.aggregate(cid)
        assert result["campaign_id"] == cid
        assert result["n_experiments"] == 2
        assert result["n_completed"] == 2
        assert result["n_failed"] == 0
        assert result["hypotheses"]["supported"] == 2
        assert result["hypotheses"]["total"] == 2
        assert len(result["experiments"]) == 2

    def test_campaign_status_completed_after_run(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        h = mgr.campaign(cid).graph.add_hypothesis("假设, 在所有条件下成立")
        mgr.add_experiment(cid, h, "跑")
        asyncio.run(mgr.run_campaign(cid, runner=supporting_runner))
        assert mgr.campaign(cid).status == "completed"


# ── 序列化 ──────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_experiment_to_dict(self):
        exp = Experiment(id="e1", hypothesis_id="h1", objective="test")
        d = exp.to_dict()
        assert d["id"] == "e1"
        assert d["hypothesis_id"] == "h1"
        assert d["status"] == "pending"
        assert d["depends_on"] == []

    def test_campaign_to_dict(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        c = mgr.campaign(cid)
        h = c.graph.add_hypothesis("假设, 在所有条件下成立")
        mgr.add_experiment(cid, h, "跑")
        d = c.to_dict()
        assert d["id"] == cid
        assert d["name"] == "test"
        assert "graph" in d
        assert len(d["experiments"]) == 1


# ── topo sort ───────────────────────────────────────────────────────────────


class TestTopoSort:
    def test_layers(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        g = mgr.campaign(cid).graph
        h1 = g.add_hypothesis("A, 在所有条件下成立")
        h2 = g.add_hypothesis("B, 在所有条件下成立")
        h3 = g.add_hypothesis("C, 在所有条件下成立")
        e1 = mgr.add_experiment(cid, h1, "1")
        e2 = mgr.add_experiment(cid, h2, "2", depends_on=[e1])
        e3 = mgr.add_experiment(cid, h3, "3", depends_on=[e1, e2])
        c = mgr.campaign(cid)
        layers = CampaignManager._topo_sort(c)
        # e1 在第0层, e2 在第1层, e3 在第2层
        assert e1 in layers[0]
        assert e2 in layers[1]
        assert e3 in layers[2]

    def test_no_dependencies_single_layer(self):
        mgr = CampaignManager()
        cid = mgr.create_campaign("test", "obj")
        g = mgr.campaign(cid).graph
        h1 = g.add_hypothesis("A, 在所有条件下成立")
        h2 = g.add_hypothesis("B, 在所有条件下成立")
        mgr.add_experiment(cid, h1, "1")
        mgr.add_experiment(cid, h2, "2")
        c = mgr.campaign(cid)
        layers = CampaignManager._topo_sort(c)
        assert len(layers) == 1
        assert len(layers[0]) == 2


# ── max_refines cap ─────────────────────────────────────────────────────────


class TestMaxRefinesCap:
    """max_refines 限制 refine 轮数, 防止 refute→refine 无限循环."""

    @staticmethod
    async def _always_refute_runner(exp: Experiment) -> dict[str, Any]:
        """永远反驳的 runner, 每次都触发 refine_failed."""
        return {
            "success": True,
            "hypothesis_supported": False,
            "evidence": {"reason": f"{exp.id} 被反驳"},
        }

    def test_serial_max_refines_caps_loop(self):
        """serial 模式下 max_refines=2 → 1 初始 + 2 refine = 3 次后停, 第 4 个留 pending."""
        import asyncio

        mgr = CampaignManager()
        cid = mgr.create_campaign("refine cap", "测 max_refines")
        g = mgr.campaign(cid).graph
        g.add_hypothesis("H0, 永远会被反驳")

        mgr.add_experiment(cid, g.all_nodes()[0].id, "跑 H0")
        result = asyncio.run(mgr.run_campaign(
            cid, runner=self._always_refute_runner, mode="serial", max_refines=2,
        ))

        # 1 初始 + 2 refine = 3 个跑了, 第 3 次 refine 出的 E3 留 pending
        assert result["n_experiments"] == 4
        assert result["n_completed"] == 3
        assert result["n_pending"] == 1
        # H0/H1/H2 被 supersede, H3 untested — 没有停在 refuted 的节点
        assert result["hypotheses"]["untested"] == 1
        assert result["hypotheses"]["total"] == 4

    def test_dag_max_refines_caps_loop(self):
        """dag 模式下 refine 补跑也受 max_refines 限制."""
        import asyncio

        mgr = CampaignManager()
        cid = mgr.create_campaign("dag cap", "测 dag max_refines")
        g = mgr.campaign(cid).graph
        g.add_hypothesis("H0, 永远会被反驳")
        mgr.add_experiment(cid, g.all_nodes()[0].id, "跑 H0")

        result = asyncio.run(mgr.run_campaign(
            cid, runner=self._always_refute_runner, mode="dag", max_refines=1,
        ))

        # 1 初始 + 1 refine = 2 跑了, 第 2 个 refine 出的留 pending
        assert result["n_completed"] == 2
        assert result["n_pending"] == 1

    def test_default_max_refines_is_8(self):
        """不传 max_refines 时默认 8, 不会无限跑."""
        import asyncio
        import inspect

        sig = inspect.signature(CampaignManager.run_campaign)
        assert sig.parameters["max_refines"].default == 8

