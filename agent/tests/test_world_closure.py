"""世界闭环: agent 在真实 (仿真) 世界里跑, 协调净评价器产出 r_phys, 全链对真实产物闭合.

SimWorld 是真目标域 (带协调净评价器):
  - agent 每次执行物理动作, 世界返回**真实观测**;
  - 评价器按真实状态给出 r_phys;
  - 真实观测经 note_generation(real_action=...) 流入:
      * VerifiableGate 真实实验账本;
      * LearnedWorldModel 从执行学"世界如何变换" (dual-axis 第二根轴);
      * RPhysTrack 真实 r_phys 归因;
      * RandomControl ablation 采到真实双臂样本.
验证点: 数据驱动世界模型收敛到真实仿真动力学; r_phys 真实归因; ablation 读真实样本.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import huginn.harness.meta_improver as mi
import huginn.harness.prompt_patch as pp

from huginn.harness.meta_improver import MetaImprover
from huginn.harness.source_patch import SourcePatchStore
from huginn.harness.meta_improver import VerifiableGate
from huginn.security.world_model import (
    PhysicalAction, SimWorld, learned_world_model,
)


def _flags(key: str, default: bool = False) -> bool:
    return key in (
        "harness_meta_improver", "harness_prompt_patch", "harness_rphys_gate",
        "harness_verifiable_gate", "harness_source_patch",
    )


async def _fake_llm(prompt: str, task: str = "summarize") -> str:
    return '{"block_name": "mem", "op": "append", "new_text": "x"}'


def test_world_closure_sim(tmp_path: Path) -> None:
    os.environ["HUGINN_CACHE_DIR"] = str(tmp_path)
    for cls in (MetaImprover, SourcePatchStore):
        cls._instance = None
    mi._harness_enabled = _flags
    pp._harness_enabled = lambda key, default=False: key == "harness_prompt_patch"

    # 隔离 LearnedWorldModel: 注入 tmp 路径的共享实例, 让 VerifiableGate 学到该实例里
    lw = learned_world_model(path=tmp_path / "lw.json")

    world = SimWorld(target_reagent=2.0, noise=0.05, seed=7)
    meta = MetaImprover.get_instance()

    # 设一个 active improver champion, 让 champion 臂也真实归因 r_phys
    from huginn.harness.meta_improver import ImproverConfig
    champ = ImproverConfig(config_id="world_champ", improver_prompt="P", r_phys_gate=0.7)
    champ.active = True
    meta._candidates["world_champ"] = champ
    meta._active_id = "world_champ"

    vols = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    r_physes: list[float] = []
    for v in vols:
        before = world.snapshot()
        observed = world.step(PhysicalAction("aspirate", {"vol": v}))
        r = world.r_phys()
        r_physes.append(r)
        asyncio.run(meta.note_generation(
            "hypothesize", [("body", "b"), ("mem", "m")], r, "world hint", _fake_llm,
            real_action={"action_type": "aspirate", "params": {"vol": v},
                         "state_before": before, "observed": observed},
        ))

    # 1. 数据驱动世界模型从真实执行学到 aspirate 效应 (第二根轴): reagent 增量为 -v 均值
    assert lw.has_model("aspirate") is True, "真实执行应学到模型"
    snap = lw.snapshot()["aspirate"]
    # vols = [1,2,1,2,1,2] → mean = 1.5; reagent 减量 = -1.5 (dec 无噪声)
    assert abs(snap["deltas"]["reagent_vol"] - (-1.5)) < 0.05, snap

    # 2. VerifiableGate 真实账本非空且含合法项
    vg = VerifiableGate()
    assert vg._executions, "真实实验应入库"
    gave = vg.verify_recent_executions(k=10)
    assert gave["n"] == len(vols)

    # 3. RPhysTrack 真实 r_phys 归因到 champion
    assert meta._rphys.series("world_champ"), "应归因真实 r_phys 到 champion"
    assert abs(meta._rphys.series("world_champ")[0] - r_physes[0]) < 1e-6

    # 4. 协调净评价器: reagent 逼近 target → r_phys 单调上升 (10→减量→接近2)
    assert r_physes[-1] > r_physes[0], "协评应反映出真实接近目标 (reagent 逼近 2)"

    # 5. RandomControl ablation 真读真实双臂样本 (champion 臂=world_champ)
    champ_r, base_r = meta._ablation_samples()
    assert champ_r and champ_r, "ablation 应读到 champion 真实 r_phys 样本"
    assert abs(champ_r[0] - r_physes[0]) < 1e-6

    # 6. LearnedWorldModel.predict 与真实世界动力学一致 (用后面相同动作预测)
    out = lw.predict({"reagent_vol": 5.0, "sample_vol": 0.0},
                     PhysicalAction("aspirate", {"vol": 1.5}))
    assert abs(out["reagent_vol"] - 3.5) < 0.05, out  # 5 - 1.5 (学到的均值)