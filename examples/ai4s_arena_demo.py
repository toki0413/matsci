#!/usr/bin/env python3
"""Huginn × 书生 — 热木星 backend 演示『产品级自主深研统一管线』.

这是对产品能力 (huginn.research.run_research_program) 的一次域插件调用, 不是孤岛 demo:
  - 热木星(非线性/线性/缩放 × WASP/HD) 只是其中一个 Experiment backend;
  - 深研闭环(假说生成 → 真实实验 → Pareto 演化 → 门禁综合 → 兜底)全在 `huginn/research/program.py`.
  - 产品能力域无关: 换成材质/化学/系外 backend 即可复用同一管线.

用法: python examples/ai4s_arena_demo.py --dry     (确定性, 不调模型)
      python examples/ai4s_arena_demo.py           (真机: 存活假说证据交由书生成文 + 门禁)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import hotjupiter_sw as hj  # noqa: E402
import hotjupiter_sw2 as hj2  # noqa: E402
from huginn.research import Experiment, run_research_program  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

NX, NY, T_END = 48, 24, 8.0e5
NX_NL = 64
_GOAL = "为潮汐锁定热木星昼夜环流寻找多证据方向的稳健研究假说"
_OBJECTIVES = {"eastward_offset": "maximize", "daynight_contrast": "maximize", "conservation": "maximize"}


def _run(system: str, kind: str) -> dict:
    if kind == "linear":
        r = hj.run_scenario(system, nx=NX, ny=NY, t_end_s=T_END)
        d = hj.analyze(r)
        return {"objectives": {"eastward_offset": float(d["hot_spot_offset_deg"]),
                               "daynight_contrast": float(d["day_night_delta_T_K"]),
                               "conservation": -abs(float(r["rel_energy_drift"]))},
                "summary": {"model": "linear", "system": system,
                            "deltaT_K": d["day_night_delta_T_K"], "offset_deg": d["hot_spot_offset_deg"],
                            "jet_ms": d["equatorial_jet_ms"], "energy_drift": round(r["rel_energy_drift"], 5)},
                "success": True}
    if kind == "nonlinear":
        r = hj2.run_scenario2(system, nx=NX_NL, ny=NY, t_end_s=2.0e6)
        return {"objectives": {"eastward_offset": float(r["hot_spot_offset_deg"]),
                               "daynight_contrast": float(r["day_night_delta_T_K"]),
                               "conservation": -abs(float(r["rel_energy_drift"]))},
                "summary": {"model": "nonlinear", "system": system,
                            "deltaT_K": r["day_night_delta_T_K"], "offset_deg": r["hot_spot_offset_deg"],
                            "jet_ms": r["equatorial_jet_max_ms"], "energy_drift": round(r["rel_energy_drift"], 8)},
                "success": True}
    # sweep
    s = hj.sweep_daynight(system, nx=NX, ny=NY)
    dTs = [p["delta_T_K"] for p in s["sweep"]]
    mono = bool(all(dTs[i] >= dTs[i + 1] for i in range(len(dTs) - 1)))
    return {"objectives": {"eastward_offset": float(s["sweep"][-1]["offset_deg"]),
                           "daynight_contrast": float(dTs[-1]),
                           "conservation": (1.0 if (s["stable_range"] and mono) else 0.0)},
            "summary": {"model": "sweep", "system": system, "deltaT_K": float(dTs[-1]),
                        "deltaT_lim_K": s["radiative_limit_delta_T_K"],
                        "norm_span": [p["norm_delta_T"] for p in s["sweep"]],
                        "monotone": mono, "radiative_limit_ok": s["stable_range"]},
            "success": True}


def _hotjupiter_backend() -> list[Experiment]:
    specs = [
        ("lin_was_daynight", "WASP-43b", "linear", "线性浅水能定量复现潮汐锁定热木星的昼夜温差与东向热点。"),
        ("lin_hd_daynight", "HD 209458b", "linear", "线性浅水对较远/较暗热木星仍给出朝向超自转的响应。"),
        ("nonlin_was_jet", "WASP-43b", "nonlinear", "加入非线性后超自转喷射显著增强于线性 → 非线性是其本质来源。"),
        ("nonlin_hd_jet", "HD 209458b", "nonlinear", "非线性超自转增强在另一系统上稳健（不依赖个别参数）。"),
        ("sweep_was_scaling", "WASP-43b", "sweep", "昼夜温差随热再分配时间定量缩放并逼近纯辐射极限（可证伪校验）。"),
        ("sweep_hd_scaling", "HD 209458b", "sweep", "缩放规律与极限校验在另一系统上同样成立（可迁移）。"),
    ]
    return [Experiment(name=n, hypothesis=h, run=lambda s=sys_, k=kind: _run(s, k))
            for (n, sys_, kind, h) in specs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="确定性运行(不调模型), 展示管线")
    ap.add_argument("--model", default="intern-s2-preview")
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    client = None
    if not args.dry:
        key = os.environ.get("INTERNLM_API_KEY")
        if not key:
            print("error: INTERNLM_API_KEY not set (或 --dry)", file=sys.stderr); return 2
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=args.base_url or
                        os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1"))

    out = run_research_program(
        goal=_GOAL,
        experiments=_hotjupiter_backend(),
        objectives_config=_OBJECTIVES,
        client=client, model=args.model, base_url=args.base_url,
        out_md=OUT / "ai4s_arena_report.md",
    )

    print(f"[program] explored={out.explored} pruned={out.pruned} "
          f"pareto_front={len(out.pareto_front)} convergence={out.converred}")
    for b in out.pareto_front:
        print(f"  surv → {b['name']}")
    print(f"[gate] {out.verdict} {out.ungrounded} | report_source={out.report_source}")
    print("报告:", OUT / "ai4s_arena_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())