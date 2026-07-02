"""UQPipeline 测试 — 多阶段不确定度传播.

覆盖: 直接测量 / linear GUM / monte_carlo / 拓扑序 / 环检测 / 贡献度分解.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from huginn.autoloop.uq_propagation import (
    UQPipeline,
    UQPipelineError,
    UQResult,
    UQStage,
)


# ── 直接测量 stage ──────────────────────────────────────────────────────────


class TestDirectStage:
    def test_direct_value_sigma(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="temp", value=300.0, sigma=2.0))
        res = pipe.run()
        assert res["temp"].value == 300.0
        assert res["temp"].sigma == 2.0
        assert res["temp"].method == "direct"
        assert res["temp"].contribution == {}

    def test_direct_zero_sigma(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="const", value=42.0))
        res = pipe.run()
        assert res["const"].sigma == 0.0

    def test_direct_missing_value_raises(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="bad"))
        with pytest.raises(UQPipelineError, match="既没 expression 也没 value"):
            pipe.run()


# ── linear 传播 ─────────────────────────────────────────────────────────────


class TestLinearPropagation:
    def test_single_dependency_linear(self):
        # bandgap = 2 * temp, temp sigma=2 → bandgap sigma=4
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="temp", value=300.0, sigma=2.0))
        pipe.add_stage(UQStage(
            name="bandgap",
            expression="2 * temp",
            dependencies=["temp"],
            method="linear",
        ))
        res = pipe.run()
        assert res["bandgap"].value == 600.0
        assert res["bandgap"].sigma == pytest.approx(4.0, abs=1e-6)
        assert res["bandgap"].method == "linear"

    def test_multi_dependency_quadrature(self):
        # z = x + y, x sigma=3, y sigma=4 → z sigma=5 (正交)
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="x", value=10.0, sigma=3.0))
        pipe.add_stage(UQStage(name="y", value=20.0, sigma=4.0))
        pipe.add_stage(UQStage(
            name="z",
            expression="x + y",
            dependencies=["x", "y"],
            method="linear",
        ))
        res = pipe.run()
        assert res["z"].value == 30.0
        assert res["z"].sigma == pytest.approx(5.0, abs=1e-6)

    def test_correlation_cross_term(self):
        # z = x + y, r=1, sigma_x=3, sigma_y=4 → sigma_z = 3+4 = 7
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="x", value=10.0, sigma=3.0))
        pipe.add_stage(UQStage(name="y", value=20.0, sigma=4.0))
        pipe.add_stage(UQStage(
            name="z",
            expression="x + y",
            dependencies=["x", "y"],
            method="linear",
            correlations={"x_y": 1.0},
        ))
        res = pipe.run()
        assert res["z"].sigma == pytest.approx(7.0, abs=1e-6)

    def test_contribution_breakout(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="x", value=10.0, sigma=3.0))
        pipe.add_stage(UQStage(name="y", value=20.0, sigma=4.0))
        pipe.add_stage(UQStage(
            name="z",
            expression="x + y",
            dependencies=["x", "y"],
            method="linear",
        ))
        res = pipe.run()
        contrib = res["z"].contribution
        # x 贡献 9/(9+16)=36%, y 贡献 64%
        assert contrib["x"] == pytest.approx(36.0, abs=0.1)
        assert contrib["y"] == pytest.approx(64.0, abs=0.1)


# ── monte_carlo 传播 ────────────────────────────────────────────────────────


class TestMonteCarloPropagation:
    def test_mc_linear_function_matches_gum(self):
        # z = 2*x + 3*y, MC 应该跟 GUM 接近
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="x", value=5.0, sigma=1.0))
        pipe.add_stage(UQStage(name="y", value=2.0, sigma=0.5))
        pipe.add_stage(UQStage(
            name="z",
            expression="2*x + 3*y",
            dependencies=["x", "y"],
            method="monte_carlo",
            n_samples=20000,
        ))
        res = pipe.run(seed=42)
        # GUM: sigma = sqrt((2*1)^2 + (3*0.5)^2) = sqrt(4+2.25)=2.5
        assert res["z"].value == pytest.approx(16.0, abs=0.1)
        assert res["z"].sigma == pytest.approx(2.5, abs=0.15)
        assert res["z"].method == "monte_carlo"

    def test_mc_nonlinear_function(self):
        # z = x**2, x=5 sigma=1 → mean≈26, sigma≈10 (delta method)
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="x", value=5.0, sigma=1.0))
        pipe.add_stage(UQStage(
            name="z",
            expression="x**2",
            dependencies=["x"],
            method="monte_carlo",
            n_samples=50000,
        ))
        res = pipe.run(seed=7)
        # E[x^2] = mu^2 + sigma^2 = 26, std ≈ sqrt(4*mu^2*sigma^2 + 2*sigma^4) ≈ 10.1
        assert res["z"].value == pytest.approx(26.0, abs=0.3)
        assert res["z"].sigma == pytest.approx(10.1, abs=0.5)

    def test_mc_reproducible_with_seed(self):
        def run_once(seed):
            pipe = UQPipeline()
            pipe.add_stage(UQStage(name="x", value=1.0, sigma=0.1))
            pipe.add_stage(UQStage(
                name="y",
                expression="3 * x",
                dependencies=["x"],
                method="monte_carlo",
                n_samples=1000,
            ))
            return pipe.run(seed=seed)["y"].sigma

        assert run_once(99) == pytest.approx(run_once(99), abs=1e-12)

    def test_mc_contribution_keys(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="a", value=1.0, sigma=0.2))
        pipe.add_stage(UQStage(name="b", value=2.0, sigma=0.1))
        pipe.add_stage(UQStage(
            name="c",
            expression="a + b",
            dependencies=["a", "b"],
            method="monte_carlo",
            n_samples=5000,
        ))
        res = pipe.run(seed=3)
        assert set(res["c"].contribution.keys()) == {"a", "b"}


# ── 拓扑序 ──────────────────────────────────────────────────────────────────


class TestTopoSort:
    def test_chain_order(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="c", value=1.0))
        pipe.add_stage(UQStage(name="a", value=2.0))
        pipe.add_stage(UQStage(
            name="b",
            expression="a + 1",
            dependencies=["a"],
        ))
        pipe.add_stage(UQStage(
            name="d",
            expression="b + c",
            dependencies=["b", "c"],
        ))
        res = pipe.run()
        # a, c 先算, b 依赖 a, d 依赖 b+c
        assert res["a"].method == "direct"
        assert res["b"].value == 3.0
        assert res["d"].value == 4.0

    def test_diamond_dependency(self):
        #      base
        #      /  \
        #    left  right
        #      \  /
        #      merge
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="base", value=10.0, sigma=1.0))
        pipe.add_stage(UQStage(
            name="left", expression="base * 2", dependencies=["base"],
        ))
        pipe.add_stage(UQStage(
            name="right", expression="base + 5", dependencies=["base"],
        ))
        pipe.add_stage(UQStage(
            name="merge",
            expression="left + right",
            dependencies=["left", "right"],
        ))
        res = pipe.run()
        assert res["left"].value == 20.0
        assert res["right"].value == 15.0
        assert res["merge"].value == 35.0

    def test_cycle_detection(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(
            name="a", expression="b + 1", dependencies=["b"],
        ))
        pipe.add_stage(UQStage(
            name="b", expression="a + 1", dependencies=["a"],
        ))
        with pytest.raises(UQPipelineError, match="环依赖"):
            pipe.run()

    def test_missing_dependency(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(
            name="z", expression="x + 1", dependencies=["x"],
        ))
        with pytest.raises(UQPipelineError, match="依赖未注册的 stage 'x'"):
            pipe.run()


# ── 错误处理 ────────────────────────────────────────────────────────────────


class TestErrors:
    def test_duplicate_stage_name(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="x", value=1.0))
        with pytest.raises(UQPipelineError, match="已存在"):
            pipe.add_stage(UQStage(name="x", value=2.0))

    def test_empty_stage_name(self):
        pipe = UQPipeline()
        with pytest.raises(UQPipelineError, match="stage 名不能为空"):
            pipe.add_stage(UQStage(name="", value=1.0))

    def test_expression_without_dependencies(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(
            name="z", expression="1 + 2", dependencies=[],
        ))
        with pytest.raises(UQPipelineError, match="有 expression 但没 dependencies"):
            pipe.run()


# ── to_dict ─────────────────────────────────────────────────────────────────


class TestSerialization:
    def test_result_to_dict(self):
        r = UQResult(
            name="z", value=10.0, sigma=2.0,
            method="linear", contribution={"x": 100.0},
        )
        d = r.to_dict()
        assert d["name"] == "z"
        assert d["value"] == 10.0
        assert d["sigma"] == 2.0
        assert d["method"] == "linear"
        assert d["contribution"] == {"x": 100.0}

    def test_pipeline_full_run_to_dict_chain(self):
        pipe = UQPipeline()
        pipe.add_stage(UQStage(name="raw", value=100.0, sigma=5.0))
        pipe.add_stage(UQStage(
            name="processed",
            expression="raw * 0.5",
            dependencies=["raw"],
            method="linear",
        ))
        res = pipe.run()
        d = res["processed"].to_dict()
        assert d["value"] == 50.0
        assert d["sigma"] == pytest.approx(2.5, abs=1e-6)
        assert d["contribution"]["raw"] == pytest.approx(100.0, abs=1e-6)
