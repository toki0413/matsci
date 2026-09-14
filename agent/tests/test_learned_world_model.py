"""世界模型数据驱动升级 (LearnedWorldModel) 验收.

从 agent 真实执行数据 (state_before, action, state_after) 学习前向效应,
替代/补强硬编码 FORWARD_EFFECTS — dual-axis RSI 的"第二根轴"雏形.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from huginn.security.world_model import (
    LearnedWorldModel, PhysicalAction, learned_world_model,
)


def _epoch(state: dict, delta: float) -> dict:
    return {"reagent_vol": float(state["reagent_vol"]) - delta,
            "sample_vol": float(state["sample_vol"]) + delta}


def test_learned_model_builds_from_executions(tmp_path: Path) -> None:
    """连续喂 3+ 次真实转移 → 学到 aspirate 效应 (需 >=3 样本才有模型)."""
    wm = LearnedWorldModel(path=tmp_path / "learned.json")
    assert wm.has_model("aspirate") is False
    for v in (1.0, 2.0, 3.0):
        before = {"reagent_vol": 10.0, "sample_vol": 0.0}
        wm.learn(before, PhysicalAction("aspirate", {"vol": v}), _epoch(before, v))
    assert wm.has_model("aspirate") is True
    snap = wm.snapshot()
    assert snap["aspirate"]["count"] == 3
    # 平均 delta 收敛到每次转移量
    assert abs(snap["aspirate"]["deltas"]["reagent_vol"] - (-2.0)) < 1e-6
    assert abs(snap["aspirate"]["deltas"]["sample_vol"] - 2.0) < 1e-6


def test_learned_predict_applies_effect(tmp_path: Path) -> None:
    """学到效应后 predict 能前向预测; 未学动作则保守原样拷贝."""
    wm = LearnedWorldModel(path=tmp_path / "learned.json")
    for i in range(3):
        before = {"reagent_vol": 10.0, "sample_vol": 0.0}
        wm.learn(before, PhysicalAction("aspirate", {"vol": 2.0}), _epoch(before, 2.0))
    out = wm.predict({"reagent_vol": 8.0, "sample_vol": 1.0},
                     PhysicalAction("aspirate", {"vol": 2.0}))
    assert abs(out["reagent_vol"] - 6.0) < 1e-6   # 8 - 2 (学到的 delta)
    assert abs(out["sample_vol"] - 3.0) < 1e-6    # 1 + 2
    # 未学动作 → 原样拷贝 (保守)
    out2 = wm.predict({"x": 1.0}, PhysicalAction("teleport", {}))
    assert out2 == {"x": 1.0}


def test_learned_handles_set_keys(tmp_path: Path) -> None:
    """新增键 (如 mixed flag) 学成 sets."""
    wm = LearnedWorldModel(path=tmp_path / "learned.json")
    for _ in range(4):
        wm.learn({"reagent_vol": 1.0}, PhysicalAction("mix", {}), {"mixed": True})
    assert "mixed" in wm.snapshot()["mix"]["sets"]
    out = wm.predict({"reagent_vol": 1.0}, PhysicalAction("mix", {}))
    assert out.get("mixed") is True


def test_learned_persistence(tmp_path: Path) -> None:
    """learned.json 落盘 + reload 保留."""
    path = tmp_path / "learned.json"
    wm = LearnedWorldModel(path=path)
    for i in range(3):
        before = {"reagent_vol": 10.0, "sample_vol": 0.0}
        wm.learn(before, PhysicalAction("dispense", {"vol": 1.0}), _epoch(before, 1.0))
    assert path.exists()
    wm2 = LearnedWorldModel(path=path)
    assert wm2.has_model("dispense") is True


def test_verifiable_gate_feeds_learned_world_model(tmp_path: Path) -> None:
    """VerifiableGate.record_execution 的真实观测喂给 LearnedWorldModel (dual-axis)."""
    from huginn.harness.meta_improver import VerifiableGate
    from huginn.security.world_model import learned_world_model as lwm_get

    # 用独立路径隔离, 避免污染共享单例
    shared = lwm_get(path=tmp_path / "lw.json")
    vg = VerifiableGate()
    # 打开事件记录: 直接构造再调 record_execution
    for v in (1.0, 1.0, 1.0):
        before = {"reagent_vol": 5.0, "sample_vol": 0.0}
        vg.record_execution(action_type="aspirate", params={"vol": v},
                            state_before=before,
                            observed={"reagent_vol": before["reagent_vol"] - v,
                                      "sample_vol": before["sample_vol"] + v})
    # VerifiableGate 默认路径学了一份; 直接断言共享实例至少学了 3 次 aspirate 效应
    assert shared.has_model("aspirate") is True