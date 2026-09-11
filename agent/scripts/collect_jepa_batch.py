"""批量真实 plan→actual 配对采集 (扩采 JEPA 语料).

每个目标: 真实 InternLM (书生) 从 hypothesis 生成 plan+expected_prediction (真实预测),
           真实 python/numpy 计算得 actual (真实执行), 存入 {runtime_home}/corpus/jepa_pairs.jsonl.

诚实边界: 这是数据采集脚本, 不训练不碰权重; 只追加真实配对.
用法:  PYTHONPATH=. python scripts/collect_jepa_batch.py [--limit N] [--index i]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from huginn.autoloop.engine import AutoloopEngine
from huginn.memory.manager import MemoryManager

from tests.test_autoloop_e2e import _noop_maybe_clarify
from tests.test_autoloop_e2e import _DummyTracker


def _run_python(src: str) -> str:
    import subprocess
    import sys
    import tempfile
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as d:
        f = P(d) / "probe.py"
        f.write_text(src, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, str(f)], capture_output=True, text=True, timeout=60
        )
        return (r.stdout or r.stderr).strip()


# (name, hypothesis, python 计算脚本[打印 actual])
OBJECTIVES = [
    ("pendulum_L1",
     "An ideal pendulum of length L=1.0 m in g=9.81 m/s² has period T=2π√(L/g). Predict with numpy the numeric period.",
     "import math\nL=1.0; g=9.81\nprint(f'T={2*math.pi*math.sqrt(L/g):.6f} s')\n"),
    ("spring_period",
     "A mass-spring system with m=0.5 kg and k=8.0 N/m oscillates with period T=2π√(m/k). Predict the numeric period.",
     "import math\nm=0.5; k=8.0\nprint(f'T={2*math.pi*math.sqrt(m/k):.6f} s')\n"),
    ("freefall_distance",
     "An object dropped from rest in g=9.81 m/s² for t=2.0 s falls a distance d=0.5 g t². Predict the numeric distance.",
     "g=9.81; t=2.0\nprint(f'd={0.5*g*t*t:.4f} m')\n"),
    ("projectile_range45",
     "A projectile launched at v0=10 m/s at 45° in g=9.81 m/s² has horizontal range R=v0² sin(2θ)/g. Predict the numeric range.",
     "import math\nv0=10.0; g=9.81; th=math.radians(45)\nprint(f'R={v0*v0*math.sin(2*th)/g:.4f} m')\n"),
    ("kinetic_energy",
     "A body with mass m=2.0 kg moving at v=3.0 m/s has kinetic energy E=0.5 m v². Predict the numeric energy in joules.",
     "m=2.0; v=3.0\nprint(f'E={0.5*m*v*v:.3f} J')\n"),
    ("spring_potential",
     "A spring with k=100 N/m stretched x=0.05 m stores potential energy U=0.5 k x². Predict the numeric energy in joules.",
     "k=100.0; x=0.05\nprint(f'U={0.5*k*x*x:.4f} J')\n"),
    ("max_height_throw",
     "An object thrown upward at v0=20 m/s in g=9.81 m/s² reaches max height h=v0²/(2g). Predict the numeric height.",
     "v0=20.0; g=9.81\nprint(f'h={v0*v0/(2*g):.3f} m')\n"),
    ("centripetal_force",
     "A mass m=0.5 kg moving in a circle of radius r=2.0 m at angular speed ω=3.0 rad/s needs centripetal force F=m r ω². Predict the numeric force in N.",
     "m=0.5; r=2.0; w=3.0\nprint(f'F={m*r*w*w:.3f} N')\n"),
    ("wave_speed_string",
     "A string under tension T=60 N with linear density μ=0.04 kg/m supports transverse waves at speed v=√(T/μ). Predict the numeric speed in m/s.",
     "import math\nT=60.0; mu=0.04\nprint(f'v={math.sqrt(T/mu):.3f} m/s')\n"),
    ("rc_time_constant",
     "An RC circuit with R=1000 Ω and C=0.001 F has time constant τ=R C. Predict the numeric time constant in seconds.",
     "R=1000.0; C=0.001\nprint(f'tau={R*C:.3f} s')\n"),
    ("electric_power",
     "A resistor across V=12 V carrying I=2 A dissipates power P=V I. Predict the numeric power in watts.",
     "V=12.0; I=2.0\nprint(f'P={V*I:.1f} W')\n"),
    ("inductor_energy",
     "An inductor with L=0.01 H carrying I=5.0 A stores energy U=0.5 L I². Predict the numeric energy in joules.",
     "L=0.01; I=5.0\nprint(f'U={0.5*L*I*I:.4f} J')\n"),
    ("ideal_gas_pressure",
     "One mole of ideal gas at T=300 K in V=0.025 m³ with R=8.314 J/(mol K) has pressure P=nRT/V. Predict the numeric pressure in Pa.",
     "n=1.0; R=8.314; T=300.0; V=0.025\nprint(f'P={n*R*T/V:.1f} Pa')\n"),
    ("pendulum_L0p25",
     "An ideal pendulum of length L=0.25 m in g=9.81 m/s² has period T=2π√(L/g). Predict the numeric period.",
     "import math\nL=0.25; g=9.81\nprint(f'T={2*math.pi*math.sqrt(L/g):.6f} s')\n"),
    ("momentum",
     "A body with mass m=0.5 kg at velocity v=4.0 m/s has momentum p=m v. Predict the numeric momentum in kg·m/s.",
     "m=0.5; v=4.0\nprint(f'p={m*v:.3f} kg·m/s')\n"),
    ("thermal_flow_rate",
     "Heat conduction through a slab: Q/t = k A ΔT / d with k=0.8, A=2.0, ΔT=20, d=0.1. Predict the numeric heat flow rate in W.",
     "k=0.8; A=2.0; dT=20.0; d=0.1\nprint(f'Qdot={k*A*dT/d:.3f} W')\n"),
    ("gravitational_potential_energy",
     "A mass m=5.0 kg at height h=2.0 m in g=9.81 m/s² has potential energy U=m g h. Predict the numeric energy in joules.",
     "m=5.0; g=9.81; h=2.0\nprint(f'U={m*g*h:.3f} J')\n"),
    ("work_force_distance",
     "A constant force F=10 N moves an object d=3.0 m along its direction, doing work W=F d. Predict the numeric work in joules.",
     "F=10.0; d=3.0\nprint(f'W={F*d:.2f} J')\n"),
    ("mechanical_power_Fv",
     "A force F=20 N moves a body at constant speed v=5.0 m/s, so mechanical power P=F v. Predict the numeric power in watts.",
     "F=20.0; v=5.0\nprint(f'P={F*v:.1f} W')\n"),
    ("ohm_resistance",
     "A resistor carrying I=0.3 A across V=9.0 V obeys Ohm's law V=I R. Predict the numeric resistance in ohms.",
     "V=9.0; I=0.3\nprint(f'R={V/I:.2f} ohm')\n"),
    ("capacitor_energy",
     "A capacitor C=0.02 F charged to V=10 V stores energy U=0.5 C V². Predict the numeric energy in joules.",
     "C=0.02; V=10.0\nprint(f'U={0.5*C*V*V:.4f} J')\n"),
    ("hydrostatic_pressure",
     "Hydrostatic pressure at depth h=10 m in water (ρ=1000 kg/m³, g=9.81) is P=ρ g h. Predict the numeric gauge pressure in Pa.",
     "rho=1000.0; g=9.81; h=10.0\nprint(f'P={rho*g*h:.1f} Pa')\n"),
    ("thermal_expansion",
     "Thermal linear expansion ΔL=α L ΔT with α=1.2e-5, L=2.0, ΔT=50. Predict the numeric length change in m.",
     "a=1.2e-5; L=2.0; dT=50.0\nprint(f'dL={a*L*dT:.6f} m')\n"),
    ("sound_wavelength",
     "Sound at speed v=343 m/s and frequency f=440 Hz has wavelength λ=v/f. Predict the numeric wavelength in m.",
     "v=343.0; f=440.0\nprint(f'l={v/f:.4f} m')\n"),
    ("projectile_time_aloft",
     "A projectile launched upward at v0=15 m/s at angle 60° in g=9.81 m/s² returns to ground after t=2 v0 sinθ/g. Predict the numeric time in s.",
     "import math\nv0=15.0; g=9.81; th=math.radians(60)\nprint(f't={2*v0*math.sin(th)/g:.4f} s')\n"),
    ("pendulum_frequency",
     "A pendulum of length L=0.5 m in g=9.81 m/s² has frequency f=1/(2π√(L/g)). Predict the numeric frequency in Hz.",
     "import math\nL=0.5; g=9.81\nprint(f'f={1/(2*math.pi*math.sqrt(L/g)):.4f} Hz')\n"),
    ("electric_field_parallel_plates",
     "Electric field between plates with V=120 V separation d=0.02 m is E=V/d. Predict the numeric field in V/m.",
     "V=120.0; d=0.02\nprint(f'E={V/d:.1f} V/m')\n"),
    ("magnetic_force_qvB",
     "A charge q=2 C at speed v=5 m/s perpendicular to B=0.3 T feels magnetic force F=q v B. Predict the numeric force in N.",
     "q=2.0; v=5.0; B=0.3\nprint(f'F={q*v*B:.2f} N')\n"),
    ("ideal_gas_volume",
     "n=0.5 mol ideal gas at T=310 K and P=101325 Pa (R=8.314) has volume V=nRT/P. Predict the numeric volume in m³.",
     "n=0.5; R=8.314; T=310.0; P=101325.0\nprint(f'V={n*R*T/P:.6f} m^3')\n"),
    ("parallel_plate_capacitance",
     "A parallel-plate capacitor with ε0=8.854e-12, A=0.01 m², d=0.001 m has C=ε0 A/d. Predict the numeric capacitance in F.",
     "e0=8.854e-12; A=0.01; d=0.001\nprint(f'C={e0*A/d:.4e} F')\n"),
    ("stress",
     "A tensile force F=500 N over area A=0.01 m² produces normal stress σ=F/A. Predict the numeric stress in Pa.",
     "F=500.0; A=0.01\nprint(f'sig={F/A:.1f} Pa')\n"),
    ("gravitational_force",
     "Two 1.0 kg masses 1.0 m apart attract with F=G m1 m2 /r², G=6.674e-11. Predict the numeric force in N.",
     "G=6.674e-11; m1=1.0; m2=1.0; r=1.0\nprint(f'F={G*m1*m2/(r*r):.4e} N')\n"),
]


async def _collect_one(eng, name, hyp, src):
    import os
    from pathlib import Path as P
    context = {
        "changed_files": ["probe.py"],
        "git_diff": "",
        "timestamp": "2026-09-11T00:00:00Z",
        "goal": "compute a physical quantity and verify a numeric prediction",
    }
    print(f"\n===== {name} =====", flush=True)
    try:
        plan = await eng._plan(hyp, context)
    except Exception as exc:
        print(f"PLAN_ERROR {name}: {exc!r}", flush=True)
        return
    pred = getattr(eng, "_current_prediction", "").strip()
    actual = _run_python(src)
    if not pred or not actual:
        print(f"SKIP {name} (pred={bool(pred)} actual={bool(actual)})", flush=True)
        return
    try:
        robust = eng._compute_surprise_robust(pred, actual)
        surprise = robust["worst"]
    except Exception:
        surprise = 1.0
    eng._record_jepa_pair(pred, actual, surprise)
    print(f"RECORDED {name} surprise={surprise:.3f} actual={actual!r}", flush=True)


async def _main(args):
    from huginn.autoloop.conjecture import get_kg
    import huginn.autoloop.conjecture as cj
    import huginn.autoloop.engine as _eng
    cj.get_kg = lambda *a, **kw: None

    eng = AutoloopEngine(workspace=str(Path("/tmp/jepa_batch_ws")), memory_manager=MemoryManager())
    eng._use_llm_decider = False
    eng.model_router = None
    eng.progress_tracker = _DummyTracker()
    eng._get_kb = lambda: None
    eng._get_plan_store = lambda: None
    eng._maybe_clarify = _noop_maybe_clarify
    _eng.AutoloopEngine._get_kb = lambda self: None  # noqa

    objs = OBJECTIVES
    if args.index is not None:
        objs = [objs[args.index]]
    elif args.start is not None:
        objs = objs[args.start :]
    elif args.limit:
        objs = objs[: args.limit]

    for i, (name, hyp, src) in enumerate(objs):
        await _collect_one(eng, name, hyp, src)

    out = Path(os.environ.get("HUGINN_JEPA_CORPUS")) if (os.environ.get("HUGINN_JEPA_CORPUS")) else (Path.home() / ".huginn" / "corpus")
    f = out / "jepa_pairs.jsonl"
    n = len([l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]) if f.exists() else 0
    print(f"\n===== CORPUS now: {n} pairs @ {f} =====", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--start", type=int, default=None)
    a = ap.parse_args()
    asyncio.run(_main(a))