#!/usr/bin/env python3
"""世界模型 / VLA 式科学团队演示 —— 数学语言为规划与验证的核心.

对标 世界模型(forward model) / VLA(视觉-语言-动作) / 机器人分层规划:
  - 感知   : 读入系外行星观测(轨道周期) → 落下初始世界状态 (WorldState).
  - 定律预告: FirstPrinciplesWorldModel 用数学定律(开普勒/日晒/Stefan-Boltzmann)对
            每个候选动作(a_scale 半长轴缩放)预告后继状态 → 模型基规划选出动作.
  - 执行   : 科学家对选定动作做独立第一性原理真值重算(真相检验).
  - 对账   : 批判者把 predict 与 actual 数值对账 —— 相符=定律证实, 偏差=定律证伪
            (如实标注, 不悄悄覆盖). 数学, 而非散文, 是最终裁判.
  - 综合   : 仅把被证实的证据送入报告, grounding 门禁兜底.

演示还将一个**假设错误**的世界模型(α=0.6)跑同一批观测 → 证明验证环节能抓住定律
被破坏 (可证伪性是科学/数学语言真正的价值, 而非华丽辞藻).

用法: python examples/ai4s_worldmodel_demo.py   (离线确定性, 零网络/零 LLM)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from huginn.research.science_team import ModelBasedScienceTeam  # noqa: E402
from huginn.research.world_model import (  # noqa: E402
    Action, FirstPrinciplesWorldModel, WorldState,
)

# 真实执行的独立第一性原理(不借用世界模型 predict, 保证对账有意义)
_AU, _SL, _SB = 1.49597870700e11, 3.828e26, 5.670374419e-8


def real_executor(init: WorldState, act: Action) -> dict:
    a_au = init.get("a_AU", 1.0) * act.config.get("a_scale", 1.0)
    S = _SL / (4 * math.pi * (a_au * _AU) ** 2)
    T_eq = ((1 - 0.1) * S / (4 * _SB)) ** 0.25
    return {"S_Wm2": round(S, 1), "T_eq_K": round(T_eq, 1)}


def _run_model(wm, label: str) -> None:
    records = json.loads((_HERE / "out" / "real_exoplanet.json")
                         .read_text(encoding="utf-8"))[:3]
    obs = [{"name": r["name"], "orbper_d": r["orbper_d"]} for r in records]
    actions = [Action({"a_scale": s}, label=f"a_scale={s}") for s in (0.9, 1.0, 1.1)]
    team = ModelBasedScienceTeam(wm, objective="T_eq_K", sense="maximize",
                                 n_scientists=2)
    out = team.run(f"{label}: 系外行星平衡温度(定律预告-执行-对账)", obs, actions,
                   real_executor)
    print(f"\n===== {label} =====")
    print(f"[定律 law] {out.law}")
    print(f"[预告 plan] {[(p['id'], p['action']['label']) for p in out.plan]}")
    print(f"[对账] survivors={out.survivors} pruned={out.pruned}")
    for rc in out.reconciliations:
        print(f"   {rc['id']}: borne_out={rc['borne_out']} err={rc.get('errors')}")
    print(f"[报告门禁] {out.verdict}")
    return out


def main() -> int:
    # 正确世界模型 (匹配真实执行反照率 α=0.1) → 定律被证实
    _run_model(FirstPrinciplesWorldModel(albedo=0.1), "正确世界模型 α=0.1")

    # 错误世界模型 (α=0.6, 反照率假设错) → 定律被证伪
    _run_model(FirstPrinciplesWorldModel(albedo=0.6), "错误世界模型 α=0.6 (应被证伪)")
    print("\n结论: 数学对账如实区分『定律被证实』vs『定律被证伪』—— 世界模型是假说,"
          " 执行是真相, 偏差不被抹平而是成为发现。")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    raise SystemExit(main())