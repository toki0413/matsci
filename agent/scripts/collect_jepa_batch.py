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
    # ── 第三批: 结构 / 方法级目标 (非纯数值量, 拓宽 predictor 潜空间) ──
    ("energy_conservation_pendulum",
     "Simulate a small-angle pendulum numerically and verify mechanical energy is conserved: predict the relative energy drift over ~5 periods and give a PASS/FAIL verdict if drift<1%.",
     "import math\nL=1.0;g=9.81;dt=1e-3;th0=0.05;w0=0.0\nth=th0;w=w0\nn=int(5*2*math.pi*math.sqrt(L/g)/dt)\nE0=0.5*L*L*w0*w0+g*L*(1-math.cos(th0))\nfor i in range(n):\n    w += (-(g/L)*math.sin(th))*dt\n    th += w*dt\nE=0.5*L*L*w*w+g*L*(1-math.cos(th))\ndrift=abs(E-E0)/E0\nprint(f'energy_drift={drift:.2e}')\nprint('VERDICT: PASS' if drift<0.01 else 'VERDICT: FAIL')\n"),
    ("small_angle_approx_error",
     "Compare small-angle period T0=2π√(L/g) to the finite-angle correction T=T0(1+θ0²/16) at θ0=30°, and predict the correction factor and the relative error of neglecting it.",
     "import math\nL=1.0;g=9.81\nth0=math.radians(30)\nT0=2*math.pi*math.sqrt(L/g)\ncorr=1/(1+th0*th0/16)\nprint(f'T0={T0:.6f} s')\nprint(f'correction_factor={1+th0*th0/16:.5f}')\nprint(f'neglect_error={th0*th0/16*100:.2f}%')\n"),
    ("spring_series_parallel",
     "For two springs k1=100 and k2=200 N/m, compute effective stiffness in series and in parallel, and predict both numeric values.",
     "k1=100.0; k2=200.0\nks=k1*k2/(k1+k2)\nkp=k1+k2\nprint(f'k_series={ks:.2f} N/m')\nprint(f'k_parallel={kp:.1f} N/m')\nprint('series < parallel: True' if ks<kp else 'series < parallel: False')\n"),
    ("rc_discharge_half_time",
     "An RC circuit with R=1000 Ω and C=0.002 F has time constant τ=RC. Predict the time for the capacitor to discharge to half its charge, which is τ·ln2.",
     "import math\nR=1000.0; C=0.002\ntau=R*C\nprint(f'tau={tau:.3f} s')\nprint(f't_half={tau*math.log(2):.4f} s')\n"),
    ("damped_oscillation_decay",
     "An underdamped mass-spring with m=1 kg, k=100 N/m, c=2 N·s/m oscillates. Predict the damping ratio ζ and the energy e-folding decay time.",
     "import math\nm=1.0; k=100.0; c=2.0\nzeta=c/(2*math.sqrt(m*k))\nomega_n=math.sqrt(k/m)\nt_decay=1/(zeta*omega_n)\nprint(f'zeta={zeta:.4f}')\nprint(f't_decay={t_decay:.4f} s')\nprint('underdamped: True' if zeta<1 else 'underdamped: False')\n"),
    ("beats_frequency",
     "Two sound waves at f1=440 Hz and f2=443 Hz superpose. Predict the beat frequency (their difference) and the beat period.",
     "f1=440.0; f2=443.0\nfb=abs(f2-f1)\nprint(f'f_beat={fb:.1f} Hz')\nprint(f'T_beat={1/fb:.4f} s')\n"),
    ("integral_numeric_vs_analytic",
     "Numerically integrate ∫₀¹ e^x dx with Simpson's rule and compare to the analytic = e−1. Predict both values and the absolute error.",
     "import math\na=0;b=1;n=1000\ndx=(b-a)/n\nx=a\ns=math.exp(a)+math.exp(b)\nfor i in range(1,n):\n    x=a+i*dx\n    s+= (4 if i%2==1 else 2)*math.exp(x)\nnum=s*dx/3\nana=math.e-1\nprint(f'numeric={num:.7f}')\nprint(f'analytic={ana:.7f}')\nprint(f'abs_err={abs(num-ana):.1e}')\n"),
    ("heat_linear_profile_mid",
     "Steady 1D heat conduction across a slab with T=20 °C and 80 °C faces gives a linear temperature profile. Predict the mid-slab temperature.",
     "T1=20.0; T2=80.0\nTmid=(T1+T2)/2\nprint(f'T_mid={Tmid:.1f} C')\nprint('profile: linear')\n"),
    ("torque_balance_seesaw",
     "A 3 kg mass is 2 m left of a seesaw pivot. Predict the distance to the right where a 2 kg mass balances it (m1·l1 = m2·l2), and output a balance verdict.",
     "m1=3.0; l1=2.0; m2=2.0\nl2=m1*l1/m2\nprint(f'l2={l2:.3f} m')\nprint(f'check: {m1*l1:.1f} vs {m2*l2:.1f} Nm')\nprint('VERDICT: balanced')\n"),
    ("wave_constructive_interference",
     "Two coherent waves with λ=0.5 m arrive with path difference d=1.0 m. Predict whether interference is constructive or destructive (d/λ integer) and give a verdict.",
     "lam=0.5; d=1.0\nm=d/lam\nprint(f'path_diff_over_lambda={m:.2f}')\nprint('VERDICT: constructive' if abs(m-round(m))<1e-9 and m>=0 else 'VERDICT: destructive')\n"),
    ("projectile_multi_quantity",
     "A projectile at v0=10 m/s, 45° in g=9.81 m/s². Predict and report range, max height, AND time aloft together (multi-output).",
     "import math\nv0=10.0;g=9.81;th=math.radians(45)\nR=v0*v0*math.sin(2*th)/g\nH=(v0*math.sin(th))**2/(2*g)\nT=2*v0*math.sin(th)/g\nprint(f'range={R:.4f} m')\nprint(f'max_height={H:.4f} m')\nprint(f'time_aloft={T:.4f} s')\n"),
    ("stress_safety_check",
     "A rod (σ=F/A) under F=500 N, A=0.01 m² has a yield stress of 250 MPa. Predict the applied stress and a PASS/FAIL safety verdict.",
     "F=500.0; A=0.01; y=250e6\nsig=F/A\nprint(f'sigma={sig:.0f} Pa')\nprint(f'utilization={sig/y*100:.2f}%')\nprint('VERDICT: PASS' if sig<y else 'VERDICT: FAIL')\n"),
    ("param_sweep_pendulum_period",
     "Predict the pendulum period T=2π√(L/g) for a set of lengths L=0.25,0.5,1.0,2.0 m, and report all four periods as a sweep.",
     "import math\ng=9.81\nfor L in [0.25,0.5,1.0,2.0]:\n    print(f'L={L}: T={2*math.pi*math.sqrt(L/g):.5f} s')\n"),
    ("mach_regime_check",
     "An object moves at v=500 m/s in air where sound speed a=343 m/s. Predict the Mach number and judge whether the incompressible/low-speed regime holds.",
     "v=500.0; a=343.0\nM=v/a\nprint(f'Mach={M:.3f}')\nprint('VERDICT: supersonic, low-speed regime invalid' if M>1 else 'VERDICT: subsonic, regime valid')\n"),
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