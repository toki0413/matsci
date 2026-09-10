"""A·运行时判官接线 e2e: `ExecutableExperience.replay` 把科学契约并入 true 运行时奖励路径.

本轮 B 之前, `grounding_source_reward` 支持 objectives+quantities, 但唯一部署调用方
`experience_archive.replay` 没传 —— 判官"溯源 AND 契约"只是"实现了", 不是"运行了".
本测试证明接上 `quantities` 后, replay 产出的 reward 里真的带契约判定: 即便数值溯源
到了轨迹, 越出有效域照样 `contract_verdict=needs_grounding`, domain_violations 露给消费方.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from huginn.research.coldstart_guards import compile_domain_guards

_MOD = Path(__file__).resolve().parents[1] / "huginn" / "research" / "experience_archive.py"
_QUANT = compile_domain_guards("ecology_dynamics")["scientific_contract"]["quantities"]


def _load_ea():
    spec = importlib.util.spec_from_file_location("experience_archive_fused", str(_MOD))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ea = _load_ea()


def _runner_with(obj) -> object:
    def runner(cfg=None):
        return {"success": True, "objectives": obj, "target": {}}
    return runner


def test_replay_fused_judge_passes_quantities_and_flags_domain_violation():
    """replay 接上 quantities 后, reward 里带契约判定: 越有效域 → needs_grounding, 且露 violations.

    关键: objectives 里 period_est=-1.0 **溯源到了轨迹**(grounded==1), 但越出 min=0
    有效域 —— "检查器能与产出分歧", 且这发生在真正部署的 replay 奖励路径上, 非测试 hack.
    """
    exp = ea.ExecutableExperience(
        goal="eco", dimension="period", config={},
        objectives={"period_est": 17.15, "x_star": 0.6},
        ground_truth={"period_est": 17.15, "x_star": 0.6},
        quantities=_QUANT,   # 域科学契约作为经验自带工件
    )
    res = exp.replay(_runner_with({"period_est": -1.0, "x_star": 0.6}))
    rw = res["reward"]
    assert rw is not None
    assert rw["grounded"] >= 1.0 - 1e-9          # 溯源确实过了
    assert rw["contract_verdict"] == "needs_grounding"   # 但契约不过
    assert any("period_est" in v and "下界" in v for v in rw["domain_violations"])


def test_replay_fused_judge_ok_when_objectives_in_domain():
    """契约内 objectives → contract_verdict ok, 无 violations."""
    exp = ea.ExecutableExperience(
        goal="eco", dimension="period", config={},
        objectives={"period_est": 17.15, "x_star": 0.6},
        ground_truth={"period_est": 17.15, "x_star": 0.6},
        quantities=_QUANT,
    )
    res = exp.replay(_runner_with({"period_est": 17.15, "x_star": 0.6}))
    rw = res["reward"]
    assert rw["contract_verdict"] == "ok"
    assert rw["domain_violations"] == []