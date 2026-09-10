"""运行时判官 = 溯源 AND 科学契约(量纲/有效域) 的融合测试.

把 `scientific_contract`(机器可读, 域数据)并进 `claim_reward.grounding_source_reward`
后, 判官从"纯溯源"升级为"溯源 AND 契约"。核心属性(架构红线):
  - 契约判定与溯源**正交**——一个数值就算溯源到了轨迹(grounding 高), 只要越出声明的
    有效域, `trusted` 仍是 False (判官能分歧, 自洽 bug 的治愈剂);
  - 默认不传 objectives/quantities 时行为完全不变(零回归, grounding_score 保持纯溯源).

诚实边界: 判定全部非学习; 契约来自 compile_domain_guards 透出的域数据工件, 不硬编码.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from huginn.research.coldstart_guards import compile_domain_guards

_CR = Path(__file__).resolve().parents[1] / "huginn" / "validation" / "claim_reward.py"


def _load_cr():
    spec = importlib.util.spec_from_file_location("claim_reward_contract", str(_CR))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cr = _load_cr()
_QUANT = compile_domain_guards("ecology_dynamics")["scientific_contract"]["quantities"]


def test_runtime_judge_backward_compat_no_contract():
    """不传契约时: 纯溯源, 零回归."""
    r = cr.grounding_source_reward("η=0.5056", ["run: 0.5056"])
    assert r["grounding_score"] == 1.0
    assert r["contract_verdict"] is None
    assert r["trusted"] is True


def test_runtime_judge_ok_when_trace_and_contract_pass():
    """溯源过 AND 契约(有效域)过 → trusted."""
    obj = {"x_star": 0.6, "y_star": 1.5, "period_est": 17.15,
           "final_prey": 0.1, "trace": 0.0}
    r = cr.grounding_source_reward("period_est=17.15", ["run: 17.15"],
                                   objectives=obj, quantities=_QUANT)
    assert r["contract_verdict"] == "ok"
    assert r["contract_score"] == 1.0
    assert r["domain_violations"] == []
    assert r["trusted"] is True


def test_runtime_judge_contract_gates_even_traced_value():
    """核心: 数值溯源到了轨迹(grounding_score=1)但越出有效域 → 判官仍不信任.

    这就是"检查器能与产出分歧"在运行时判官的落地 —— 契约层不靠 harness 代偿.
    """
    obj = {"period_est": -1.0, "x_star": 0.6}   # period_est 越出 [0, +∞)
    r = cr.grounding_source_reward("period_est=-1.0", ["run: -1.0"],
                                   objectives=obj, quantities=_QUANT)
    assert r["grounding_score"] >= 1.0 - 1e-9   # 溯源确实过了
    assert r["contract_verdict"] == "needs_grounding"
    assert r["trusted"] is False                # 判官 = 溯源 AND 契约 → 否
    assert any("period_est" in v and "下界" in v for v in r["domain_violations"])


def test_runtime_judge_coverage_gap_partial_not_fatal():
    """未声明量(coverage gap): 不硬判, 但"有效域未知" → 不全信(partial/不 trusted)."""
    obj = {"mystery_coef": 5.0}
    r = cr.grounding_source_reward("mystery_coef=5.0", ["run: 5.0"],
                                   objectives=obj, quantities=_QUANT)
    assert r["domain_violations"] == []
    assert any("未声明" in g for g in r["coverage_gaps"])
    assert r["contract_verdict"] == "partial"
    assert r["trusted"] is False