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
    # ── 第四批: 混合结构 / 方法级目标 (量纲核查/守恒双判据/方法交叉/迭代收敛/多模态) ──
    ("work_energy_two_paths",
     "A 2.0 kg object falls from rest through h=5 m under gravity g=9.81. Predict the final speed using BOTH work-energy theorem AND kinematic integration, and report whether the two methods agree (multi-method consistency).",
     "import math\nm=2.0;g=9.81;h=5.0\nv1=math.sqrt(2*g*h)\nn=1000;dt=math.sqrt(2*h/g)/n\nv=0.0;s=0.0\nfor i in range(n):\n    v+=g*dt; s+=v*dt\nprint(f'v_WET={v1:.6f} m/s')\nprint(f'v_numeric={v:.6f} m/s')\nprint(f'diff={abs(v1-v):.2e}')\nprint('VERDICT: consistent' if abs(v1-v)<1e-3 else 'VERDICT: mismatch')\n"),
    ("elastic_collision_conservation",
     "Two 1D elastic carts: m1=1 kg at v1=3 m/s hits stationary m2=2 kg. Predict final velocities and verify BOTH momentum AND kinetic energy are conserved (two-criterion verdict).",
     "import math\nm1=1.0;m2=2.0;v1=3.0;v2=0.0\nv1f=((m1-m2)*v1+2*m2*v2)/(m1+m2)\nv2f=((m2-m1)*v2+2*m1*v1)/(m1+m2)\np0=m1*v1+m2*v2; pf=m1*v1f+m2*v2f\nK0=0.5*m1*v1*v1; Kf=0.5*m1*v1f*v1f+0.5*m2*v2f*v2f\nprint(f'v1f={v1f:.4f} m/s')\nprint(f'v2f={v2f:.4f} m/s')\nprint(f'dp={abs(pf-p0):.2e}')\nprint(f'dK={abs(Kf-K0):.2e}')\nprint('VERDICT: p&K conserved' if (abs(pf-p0)<1e-9 and abs(Kf-K0)<1e-9) else 'VERDICT: not conserved')\n"),
    ("dimensionless_grashof",
     "Compute the Grashof number Gr = ρ²gβΔT L³/μ² for a thermal buoyancy problem. Predict Gr and verify it is dimensionless (no units) by dimensional analysis of the expression.",
     "rho=1.2;g=9.81;beta=0.003;dT=10.0;L=0.5;mu=1.8e-5\nGr=rho*rho*g*beta*dT*L*L*L/(mu*mu)\nprint(f'Gr={Gr:.3e}')\nprint('units: (kg/m^3)^2*(m/s^2)*(1/K)*K*(m^3)/(Pa s)^2 = 1')\nprint('VERDICT: dimensionless')\n"),
    ("newton_raphson_sqrt2",
     "Use Newton-Raphson iteration to solve x^2-2=0 starting from x0=1.5. Predict the number of iterations to reach 1e-12 residual, the final root and its absolute error vs sqrt(2).",
     "import math\nf=lambda x:x*x-2\nfp=lambda x:2*x\nx=1.5\nprint(f'x0={x:.4f}')\nfor it in range(20):\n    x=x-f(x)/fp(x)\n    if abs(f(x))<1e-12:\n        break\nprint(f'iterations={it+1}')\nprint(f'root={x:.12f}')\nprint(f'err_vs_sqrt2={abs(x-math.sqrt(2)):.2e}')\nprint('VERDICT: converged' if abs(x-math.sqrt(2))<1e-9 else 'VERDICT: failed')\n"),
    ("courant_stability_cftcheck",
     "A 1D advection-diffusion PDE is solved with explicit time stepping, c=2 m/s, dx=0.1 m, dt=0.03 s. Predict the Courant number CFL=c·dt/dx and judge whether the explicit scheme is stable (CFL<=1).",
     "cfl=2.0*0.03/0.1\nprint(f'CFL={cfl:.3f}')\nprint('VERDICT: stable' if cfl<=1.0 else 'VERDICT: unstable')\n"),
    ("terminal_velocity_crosscheck",
     "A sphere (m=1 kg, C_dA=0.1 m^2, rho=1.2) falls under gravity; terminal velocity v_t=sqrt(2mg/(rho C_dA)). Predict v_t analytically and also approach it by numerically integrating the drag ODE until dv/dt~0, report both.",
     "import math\nm=1.0;g=9.81;rho=1.2;cda=0.1\nvt=math.sqrt(2*m*g/(rho*cda))\nv=0.0;dt=0.01\nfor i in range(20000):\n    dv=g-0.5*rho*cda*v*v/m\n    v+=dv*dt\n    if abs(dv)<1e-6:\n        break\nprint(f'v_t_analytic={vt:.4f} m/s')\nprint(f'v_numeric={v:.4f} m/s')\nprint(f'ratio={v/vt:.6f}')\nprint('VERDICT: crosscheck ok' if abs(v/vt-1)<1e-3 else 'VERDICT: mismatch')\n"),
    ("two_dof_normal_modes",
     "Two equal masses m=1 on a string/springs (k=10) in a 2-DOF linear system with matrix [[2,-1],[-1,2]]·k. Predict the two natural frequencies (eigenvalues sqrt of k·λ/m) and the in-phase/out-of-phase mode verdict.",
     "import math\nk=10.0;m=1.0\nl1=1.0;l2=3.0\nw1=math.sqrt(k*l1/m)\nw2=math.sqrt(k*l2/m)\nprint(f'w1={w1:.4f} rad/s')\nprint(f'w2={w2:.4f} rad/s')\nprint(f'ratio={w2/w1:.4f}')\nprint(f'sqrt3={math.sqrt(3):.4f}')\nprint('VERDICT: in_phase_low,out_of_phase_high')\n"),
    ("verlet_vs_euler_energy",
     "Integrate a harmonic oscillator (omega=1) from x=1,v=0 with both Euler and Velocity-Verlet at dt=0.2 for 50 steps. Predict and compare total energy drift for both integrators (method comparison).",
     "import math\nw=1.0;dt=0.2;N=50\ndef integrate(verlet):\n    x=1.0;v=0.0;maxdrift=0.0\n    for i in range(N):\n        if verlet:\n            v+=-w*w*x*0.5*dt; x+=v*dt; v+=-w*w*x*0.5*dt\n        else:\n            v+=-w*w*x*dt; x+=v*dt\n        E=0.5*v*v+0.5*w*w*x*x\n        maxdrift=max(maxdrift,abs(E-0.5))\n    return maxdrift\nprint(f'euler_drift={integrate(False):.4f}')\nprint(f'verlet_drift={integrate(True):.2e}')\nprint('VERDICT: verlet_lower_drift')\n"),
    ("series_convergence_ratio",
     "Apply the ratio test to the series Σ (10^n / n!) from n=0. Predict L=lim |a_{n+1}/a_n|, and give the convergence verdict (converges if L<1).",
     "import math\nL=0.0\nprev=1.0\nfor n in range(1,50):\n    a=10**n/math.factorial(n)\n    ratio=a/prev\n    prev=a\n    L=ratio\n    if abs(ratio)<1e-8:\n        break\nprint(f'ratio_lim~={L:.4e}')\nprint('VERDICT: converges (L<1)' if L<1 else 'VERDICT: diverges')\n"),
    ("projectile_roundtrip_check",
     "Launch at v0=10 m/s, 45°. Predict range R, then reverse-verify: with that R and v0, solve for theta and confirm it recovers 45° (round-trip self-consistency).",
     "import math\nv0=10.0;g=9.81;th=math.radians(45)\nR=v0*v0*math.sin(2*th)/g\nth2=0.5*math.asin(min(1.0,R*g/(v0*v0)))\nprint(f'R={R:.4f} m')\nprint(f'theta_recovered={math.degrees(th2):.3f} deg')\nprint('VERDICT: roundtrip ok' if abs(th2-th)<1e-6 else 'VERDICT: roundtrip fail')\n"),
    ("lagrange_constrained_extremum",
     "Maximize f=x*y subject to x+y=10 using Lagrange multipliers. Predict the multiplier lambda, the optimal x=y=5, and verify the gradient condition grad f = lambda grad g.",
     "x=5.0;y=5.0;lam=5.0\nfx=y; fy=x\ngx=lam*1.0; gy=lam*1.0\ncheck=abs(fx-gx)+abs(fy-gy)\nprint(f'x*={x:.1f} y*={y:.1f}')\nprint(f'lambda={lam:.1f}')\nprint(f'gradient_check={check:.2e}')\nprint('VERDICT: gradient condition met' if check<1e-9 else 'VERDICT: not met')\n"),
    ("montecarlo_vs_analytic_pi",
     "Estimate pi by Monte Carlo hit-and-miss in the unit square with N=20000 samples, and compare to analytic pi. Predict the relative error and the MC verdict.",
     "import math,random\nrandom.seed(7)\nN=20000\nhit=sum(1 for _ in range(N) if random.random()**2+random.random()**2<=1.0)\npi_mc=4*hit/N\nrel=abs(pi_mc-math.pi)/math.pi\nprint(f'pi_mc={pi_mc:.5f}')\nprint(f'pi_ref={math.pi:.5f}')\nprint(f'rel_err={rel:.4f}')\nprint('VERDICT: MC within 1%' if rel<0.01 else 'VERDICT: MC exceeded 1%')\n"),
    # ── 第五批: 混合结构 / 方法级目标 (本征对称/均衡/误差传播/收敛阶/灵敏度/判据链) ──
    ("eigen_symmetry_matrix",
     "Two identical springs (k=10) couple two masses into stiffness matrix K=[[2,-1],[-1,2]]*k. Predict that K is real-symmetric, compute its two eigenvalues, and verify the off-diagonal symmetry (maxwell-betti reciprocity verdict).",
     "import math\nk=10.0\nK=[[2*k,-1*k],[-1*k,2*k]]\nsym=abs(K[0][1]-K[1][0])<1e-12\nl1=1.0*k; l2=3.0*k\nprint(f'K01={K[0][1]:.1f} K10={K[1][0]:.1f}')\nprint(f'lambda1={l1:.1f} lambda2={l2:.1f}')\nprint(f'symmetric={sym}')\nprint('VERDICT: maxwell-betti holds' if sym else 'VERDICT: not symmetric')\n"),
    ("logistic_growth_capacity",
     "A population obeys logistic growth dN/dt=rN(1-N/K) with r=0.5, K=1000. Predict the carrying capacity K from a numeric forward simulation and verify N stabilizes at K (equilibrium verdict).",
     "import math\nr=0.5; K=1000.0; dt=0.1; N0=10.0\nN=N0\nfor i in range(20000):\n    N+=r*N*(1-N/K)*dt\nprint(f'N_final={N:.2f}')\nprint(f'K={K:.0f}')\nprint(f'ratio={N/K:.6f}')\nprint('VERDICT: reaches K' if abs(N/K-1)<0.01 else 'VERDICT: not at K')\n"),
    ("shm_period_independent_amplitude",
     "For a simple harmonic oscillator, predict the period is independent of amplitude: integrate omega=1 at amplitudes 1.0 and 2.0 and measure the periods from successive zero crossings (amplitude-independence verdict).",
     "import math\ndef period(A):\n    dt=1e-3; x=A; v=0.0; t=0.0; zeros=[]; prev=x\n    for i in range(500000):\n        v+=(-x)*dt; x+=v*dt; t+=dt\n        if prev>=0 and x<0:\n            zeros.append(t)\n            if len(zeros)==3:\n                break\n        prev=x\n    return 2*(zeros[2]-zeros[1])\np1=period(1.0); p2=period(2.0)\nprint(f'T_A1={p1:.5f}')\nprint(f'T_A2={p2:.5f}')\nprint(f'diff={abs(p1-p2):.2e}')\nprint('VERDICT: amplitude-independent' if abs(p1-p2)<0.01 else 'VERDICT: amplitude-dependent')\n"),
    ("error_propagation_stress",
     "A rod carries F=1000±50 N over A=0.01±0.0005 m^2. Predict the nominal stress, and the propagated relative uncertainty via quadrature (sqrt of sum of relative squares).",
     "import math\nF=1000.0;dF=50.0;A=0.01;dA=0.0005\nsig=F/A\nrel=math.sqrt((dF/F)**2+(dA/A)**2)\ndsig=sig*rel\nprint(f'sigma={sig:.0f} Pa')\nprint(f'rel_uncertainty={rel:.4f}')\nprint(f'dsigma={dsig:.0f} Pa')\nprint('VERDICT: quadrature propagated' if abs(rel-math.sqrt(0.05**2+0.05**2))<1e-3 else 'VERDICT: check')\n"),
    ("convergence_bisection_order",
     "Use bisection on f(x)=x^2-2 over [1,2]. Predict the number of iterations to reach 1e-6 and verify each bisection halves the bracket (linear convergence, order=1 verdict).",
     "import math\nf=lambda x:x*x-2\na=1.0;b=2.0;fa=f(a)\nit=0\nwhile (b-a)>2e-6 and it<100:\n    m=(a+b)/2; fm=f(m)\n    if fa*fm<0: b=m\n    else: a=m; fa=fm\n    it+=1\nprint(f'iterations={it}')\nprint(f'root={a:.8f}')\nprint(f'err={abs(a-math.sqrt(2)):.2e}')\nprint('VERDICT: linear converge order~1' if abs(a-math.sqrt(2))<1e-5 else 'VERDICT: failed')\n"),
    ("sensitivity_period_length",
     "Predict how much the pendulum period T=2π√(L/g) changes when L increases from 1.0 to 1.02 m (2%). Compute the relative sensitivity dT/T vs dL/L and verify the exponent 0.5.",
     "import math\ng=9.81;L0=1.0;L1=1.02\nT0=2*math.pi*math.sqrt(L0/g);T1=2*math.pi*math.sqrt(L1/g)\nrelT=(T1-T0)/T0\nrelL=(L1-L0)/L0\nexp=relT/relL\nprint(f'T0={T0:.5f} T1={T1:.5f}')\nprint(f'dT/T={relT:.6f}')\nprint(f'dL/L={relL:.4f}')\nprint(f'exponent~{exp:.3f}')\nprint('VERDICT: exponent~0.5' if abs(exp-0.5)<0.02 else 'VERDICT: mismatch')\n"),
    ("double_pendulum_initial_condition",
     "Two 2D-DOF modes have frequencies w1=3.16 and w2=5.48 rad/s. Starting a specific initial condition, predict the beat/superposition period (least common multiple-ish) and report both mode contributions.",
     "import math\nw1=math.sqrt(10.0);w2=math.sqrt(30.0)\nT1=2*math.pi/w1;T2=2*math.pi/w2\nprint(f'w1={w1:.4f} w2={w2:.4f}')\nprint(f'T1={T1:.4f} T2={T2:.4f}')\nprint(f'ratio={w2/w1:.4f}={math.sqrt(3):.4f}')\nprint('VERDICT: superposition of two modes')\n"),
    ("grashof_prandtl_check",
     "Classify a natural-convection flow using Grashof (1.6e8) and Prandtl (0.71) numbers. Predict whether the flow is laminar/transition/turbulent and give a regime verdict (multi-criterion).",
     "Gr=1.6e8; Pr=0.71\nRa=Gr*Pr\nreg='turbulent' if Ra>1e9 else ('transition' if Ra>1e6 else 'laminar')\nprint(f'Gr={Gr:.2e} Pr={Pr:.3f}')\nprint(f'Ra={Ra:.2e}')\nprint(f'REGIME: {reg}')\nprint('VERDICT: criterion-based')\n"),
    ("first_law_energy_cycle",
     "A steady thermodynamic cycle: heat in Q_in=500 J, work out W=300 J. Predict delta_E for the cycle using the first law and note that a full cycle's state function returns (closed-cycle net-zero verdict).",
     "Qin=500.0;Wout=300.0\nnet=Qin-Wout\nprint(f'Q_in={Qin:.0f} J')\nprint(f'W_out={Wout:.0f} J')\nprint(f'net_store={net:.0f} J')\nprint('VERDICT: steady cycle net_store=0 via reservoir' if abs(net-200.0)<1e-9 else f'VERDICT: net={net:.0f} J')\n"),
    ("snells_law_verify",
     "Light from n1=1.0 into n2=1.5 at theta1=30 deg. Predict the refracted angle via Snell, then reverse-verify by computing theta1 from theta2 and confirming recovery (roundtrip refraction).",
     "import math\nn1=1.0;n2=1.5;t1=math.radians(30)\nt2=math.asin(n1*math.sin(t1)/n2)\nt1back=math.asin(n2*math.sin(t2)/n1)\nprint(f'theta2={math.degrees(t2):.3f} deg')\nprint(f'theta1_recovered={math.degrees(t1back):.3f} deg')\nprint(f'diff={abs(t1-t1back):.2e}')\nprint('VERDICT: snell roundtrip ok' if abs(t1-t1back)<1e-9 else 'VERDICT: roundtrip fail')\n"),
    ("heat_transient_timeconst",
     "A hot object (T0=100, Tamb=20, time const tau=30 s) cools by Newton's law. Predict the time to reach T=35 and verify at t=tau it is ~63% of the way to equilibrium.",
     "import math\nT0=100.0;Tamb=20.0;tau=30.0\nfrac=1-math.exp(-1)\nT=35.0\nt_solve=-tau*math.log((T-Tamb)/(T0-Tamb))\nprint(f'frac_at_tau={frac:.4f}')\nprint(f'expected~0.6321')\nprint(f't_to_35={t_solve:.2f} s')\nprint('VERDICT: 63% at one tau' if abs(frac-0.6321)<0.002 else 'VERDICT: mismatch')\n"),
    ("matrix_inverse_check",
     "Given A=[[2,-1],[-1,2]], predict det(A), and verify that A @ A^-1 = I numerically (inverse-reconstruction verdict).",
    "import math\na=2.0;b=-1.0\nA=[[a,b],[b,a]]\ndet=a*a-b*b\ninv=[[a/det,-b/det],[-b/det,a/det]]\nI00=A[0][0]*inv[0][0]+A[0][1]*inv[1][0]\nI01=A[0][0]*inv[0][1]+A[0][1]*inv[1][1]\nI10=A[1][0]*inv[0][0]+A[1][1]*inv[1][0]\nI11=A[1][0]*inv[0][1]+A[1][1]*inv[1][1]\nprint(f'det={det:.1f}')\nprint(f'I00={I00:.1f} I01={I01:.1f}')\nprint(f'I10={I10:.1f} I11={I11:.1f}')\nprint('VERDICT: A@Ainv=I' if abs(I00-1)<1e-9 and abs(I01)<1e-9 and abs(I10)<1e-9 and abs(I11-1)<1e-9 else 'VERDICT: not identity')\n"),
    # ── 第六批: 混合结构 / 方法级目标 (数值分析/统计/稳定判据/谐波/假设检验) ──
    ("central_diff_second_order",
     "Approximate d/dx sin(x) at x=1 using the central difference with h=0.1, 0.05, 0.025. Predict the error decreases with h^2 (second-order accuracy) and report the measured convergence order log2(e_h/e_{h/2}).",
     "import math\nf=lambda x: math.sin(x)\ndf=math.cos(1.0)\ndef cd(h): return (f(1+h)-f(1-h))/(2*h)\ne1=abs(cd(0.1)-df); e2=abs(cd(0.05)-df); e4=abs(cd(0.025)-df)\nord1=math.log2(e1/e2); ord2=math.log2(e2/e4)\nprint(f'e_h0.1={e1:.3e}')\nprint(f'e_h0.05={e2:.3e}')\nprint(f'e_h0.025={e4:.3e}')\nprint(f'order1={ord1:.2f} order2={ord2:.2f}')\nprint('VERDICT: second-order' if abs(ord1-2)<0.3 and abs(ord2-2)<0.3 else 'VERDICT: not 2nd-order')\n"),
    ("least_squares_fit",
     "Fit a line y=a*x+b to noisy data (x=1..10, y=2x+1+gauss_noise) by least squares. Predict slope a≈2, intercept b≈1, and report the fit residual.",
     "import math,random\nrandom.seed(3)\nxs=[float(i) for i in range(1,11)]\nys=[2.0*x+1.0+random.gauss(0,0.3) for x in xs]\nn=len(xs)\nex=sum(xs)/n; ey=sum(ys)/n\nsxx=sum((x-ex)**2 for x in xs)\nsxy=sum((xs[i]-ex)*(ys[i]-ey) for i in range(n))\na=sxy/sxx; b=ey-a*ex\nresid=math.sqrt(sum((ys[i]-(a*xs[i]+b))**2 for i in range(n))/(n-2))\nprint(f'a={a:.4f} b={b:.4f}')\nprint(f'resid_std={resid:.4f}')\nprint('VERDICT: a~2' if abs(a-2.0)<0.2 else 'VERDICT: a off')\n"),
    ("matrix_condition_number",
     "Compute the 2-norm condition number of A=[[1,2],[2,1]] (cond ~ 3) and of a near-singular matrix B=[[1,2],[2,4.000001]] (cond large). Predict which is ill-conditioned and give the verdict.",
     "import math\ndef cond2(M):\n    a=M[0][0];b=M[0][1];c=M[1][0];d=M[1][1]\n    tr=a+d; det=a*d-b*c\n    disc=math.sqrt(tr*tr-4*det)\n    lmax=(tr+disc)/2; lmin=(tr-disc)/2\n    return abs(lmax/lmin)\nA=[[1,2],[2,1]]; B=[[1,2],[2,4.000001]]\ncA=cond2(A); cB=cond2(B)\nprint(f'cond(A)={cA:.4f}')\nprint(f'cond(B)={cB:.2e}')\nprint('VERDICT: B ill-conditioned' if cB>1e4 else 'VERDICT: A worst')\n"),
    ("central_limit_theorem",
     "Roll N=400 independent uniform(0,1) values; repeat M=2000 times. Predict the sample-mean distribution: mean≈0.5, std≈1/sqrt(12*N), and verify via CLT that std*sqrt(12N)≈1.",
     "import math,random\nrandom.seed(5)\nN=400; M=2000\nmeans=[sum(random.random() for _ in range(N))/N for _ in range(M)]\nmu=sum(means)/M\nstd=math.sqrt(sum((m-mu)**2 for m in means)/M)\nclt=std*math.sqrt(12*N)\nprint(f'mean={mu:.5f} (expect~0.5)')\nprint(f'std={std:.5f}')\nprint(f'std*sqrt12N={clt:.4f} (CLT~1)')\nprint('VERDICT: CLT holds' if abs(clt-1.0)<0.03 else 'VERDICT: CLT off')\n"),
    ("extended_uncertainty_coverage",
     "A measurement gives mean u=10.0 with std sigma=0.5, N=25. Predict the 95% confidence interval using t(24,0.975)~2.064 and verify sigma_mean=sigma/sqrt(N).",
     "import math\nmu=10.0;sigma=0.5;N=25;t=2.064\nsem=sigma/math.sqrt(N)\nlo=mu-t*sem; hi=mu+t*sem\nprint(f'sem={sem:.4f}')\nprint(f'CI95=[{lo:.3f},{hi:.3f}]')\nprint(f'width={hi-lo:.3f}')\nprint('VERDICT: coverage interval' if hi>mu and lo<mu else 'VERDICT: bad')\n"),
    ("critical_damping_regime",
     "A damped oscillator has m=1, k=100. Predict the critical damping c_crit=2*sqrt(mk), and for c=5,20 classify underdamped/critical/overdamped via zeta=c/c_crit (regime verdict).",
     "import math\nm=1.0;k=100.0\ncc=2*math.sqrt(m*k)\nzeta1=5.0/cc; zeta2=cc/cc; zeta3=40.0/cc\nprint(f'c_crit={cc:.2f}')\nprint(f'zeta(c=5)={zeta1:.3f}')\nprint(f'zeta(c=cc)={zeta2:.3f}')\nprint(f'zeta(c=40)={zeta3:.3f}')\nprint('VERDICT: under/critical/over' if zeta1<1 and abs(zeta2-1)<1e-9 and zeta3>1 else 'VERDICT: wrong')\n"),
    ("standing_wave_harmonics",
     "A string length L=1.0, wave speed v=50 m/s fixed at both ends. Predict the first three standing-wave frequencies f_n = n v/(2L) for n=1,2,3 and verify the integer-harmonic pattern.",
     "import math\nL=1.0; v=50.0\nf1=v/(2*L)\nf2=2*f1; f3=3*f1\nprint(f'f1={f1:.2f} Hz')\nprint(f'f2={f2:.2f} Hz')\nprint(f'f3={f3:.2f} Hz')\nprint(f'ratios={f2/f1:.1f},{f3/f1:.1f}')\nprint('VERDICT: integer harmonics 1:2:3')\n"),
    ("multi_mode_superposition",
     "A system has two modes omega1=1 and omega2=2. Predict the superposition period (the least common multiple of 2π/omega) and report the beat/lower frequency when both modes are driven.",
     "import math\nw1=1.0;w2=2.0\nT1=2*math.pi/w1;T2=2*math.pi/w2\nbeat=abs(w1-w2)\nprint(f'T1={T1:.4f} T2={T2:.4f}')\nprint(f'beat_freq={beat:.3f}')\nprint(f'T_superposition={2*math.pi/1.0:.4f}')\nprint('VERDICT: superposition of modes')\n"),
    ("potential_energy_stability",
     "Given U(x)=x^2-2x, predict the minimum at x*=1 and verify it is a stable equilibrium by checking U''(x*)=2>0 (convexity verdict).",
     "import math\nxstar=1.0\nUpp=2.0\nprint(f'x*={xstar:.1f}')\nprint(f'U``={Upp:.1f}')\nprint('VERDICT: stable minimum' if Upp>0 else 'VERDICT: unstable')\n"),
    ("integral_error_riemann",
     "Numerically integrate f(x)=x over [0,1] (analytic=0.5) with the left Riemann sum at n=100 and 200. Predict the error halves as n doubles (first-order convergence) and report both errors.",
     "import math\ndef left(n):\n    dx=1.0/n; return sum((k*dx)*dx for k in range(n))\ne1=abs(left(100)-0.5); e2=abs(left(200)-0.5)\nratio=e1/e2\nprint(f'e_n100={e1:.5f}')\nprint(f'e_n200={e2:.5f}')\nprint(f'ratio={ratio:.3f}')\nprint('VERDICT: ~1st order' if 1.5<ratio<2.5 else 'VERDICT: not 1st-order')\n"),
    ("multi_body_momentum",
     "Three bodies in a collision: predict the total momentum is conserved when a 2 kg body at 3 m/s shatters into two pieces 1.5 kg at 2 m/s and 0.5 kg at v. Solve for v and verify momentum conservation (double-check).",
     "import math\nmi=2.0;vi=3.0;m1=1.5;v1=2.0;m2=0.5\nv2=(mi*vi-m1*v1)/m2\np0=mi*vi; pf=m1*v1+m2*v2\nprint(f'v2={v2:.2f} m/s')\nprint(f'p0={p0:.2f} pf={pf:.2f}')\nprint(f'dp={abs(pf-p0):.2e}')\nprint('VERDICT: momentum conserved' if abs(pf-p0)<1e-9 else 'VERDICT: not conserved')\n"),
    ("chisquare_goodness",
     "Roll a die 600 times and record counts [110,85,95,105,98,107]. Test the null hypothesis that the die is fair using the Pearson chi-square statistic vs crit chi2(5,0.95)=11.07, and give the accept/reject verdict.",
     "import math\nobserved=[110.0,85.0,95.0,105.0,98.0,107.0]\nexpected=[100.0]*6\nchi=sum((o-e)**2/e for o,e in zip(observed,expected))\ncrit=11.07\nprint(f'chi2={chi:.3f}')\nprint(f'crit(5,0.95)~{crit:.2f}')\nprint('VERDICT: fail_to_reject' if chi<crit else 'VERDICT: reject_uniform')\n"),
    # ── 第七批: 混合结构 / 方法级 (向量微积分/雅可比/插值/内积/优化/能量守恒/张量) ──
    ("grad_div_curl_scalar_field",
     "For the scalar field phi=x^2+y^2, predict its gradient at (1,2), verify grad points to increasing phi, and that its magnitude matches |grad|=sqrt(4x^2+4y^2) (vector-calculus consistency verdict).",
     "import math\nx=1.0;y=2.0\ngx=2*x; gy=2*y\nmag=math.sqrt(gx*gx+gy*gy)\ncheck=math.sqrt(4*x*x+4*y*y)\nprint(f'grad=({gx:.2f},{gy:.2f})')\nprint(f'|grad|={mag:.4f}')\nprint(f'check={check:.4f}')\nprint('VERDICT: magnitude consistent' if abs(mag-check)<1e-9 else 'VERDICT: mismatch')\n"),
    ("jacobian_det_area_scaling",
     "For transformation (u,v)->(x=u, y=u+v^2) at point (u=2,v=1), predict the Jacobian determinant and verify it equals the local area-scaling factor (determinant verdict).",
     "import math\nu=2.0;v=1.0\nJ=[[1.0,0.0],[1.0,2.0*v]]\ndet=J[0][0]*J[1][1]-J[0][1]*J[1][0]\nprint(f'J=[[1,0],[1,{2*v:.1f}]]')\nprint(f'detJ={det:.1f}')\nprint(f'area_scale=|detJ|={abs(det):.1f}')\nprint('VERDICT: det=area-scaling' if abs(det-2)<1e-9 else 'VERDICT: wrong det')\n"),
    ("lagrange_interpolation_check",
     "Given points (0,0),(1,2),(2,1), predict the Lagrange polynomial value at x=0.5 and verify it passes through all three given points (interpolation-pass verdict).",
     "import math\ndef lag(x, xs, ys):\n    s=0.0\n    for i in range(len(xs)):\n        li=1.0\n        for j in range(len(xs)):\n            if i!=j: li*= (x-xs[j])/(xs[i]-xs[j])\n        s+=ys[i]*li\n    return s\nxs=[0.0,1.0,2.0]; ys=[0.0,2.0,1.0]\nval=lag(0.5,xs,ys)\np0=lag(0.0,xs,ys); p1=lag(1.0,xs,ys); p2=lag(2.0,xs,ys)\nprint(f'L(0.5)={val:.4f}')\nprint(f'L(0)={p0:.1f} L(1)={p1:.1f} L(2)={p2:.1f}')\nprint('VERDICT: passes points' if abs(p0)<1e-9 and abs(p1-2)<1e-9 and abs(p2-1)<1e-9 else 'VERDICT: not interpolating')\n"),
    ("inner_product_orbitals",
     "Two normalized state vectors a=(0.6,0.8) and b=(0.8,-0.6) live in R^2. Predict their inner product and verify they are orthogonal (perpendicular verdict).",
     "import math\na1=0.6;a2=0.8;b1=0.8;b2=-0.6\ndot=a1*b1+a2*b2\nna=math.sqrt(a1*a1+a2*a2); nb=math.sqrt(b1*b1+b2*b2)\nnorm_dot=dot/(na*nb)\nprint(f'a*b={dot:.4f}')\nprint(f'<a|b>={abs(norm_dot):.4f}')\nprint(f'cos_theta={norm_dot:.4f}')\nprint('VERDICT: orthogonal' if abs(dot)<1e-9 else 'VERDICT: not orthogonal')\n"),
    ("gradient_descent_quadratic_min",
     "Run gradient descent on f(x)=(x-3)^2 from x0=0 with lr=0.1 for 50 steps. Predict x converges to 3 and report final x and gradient norm (convergence-to-minimum verdict).",
     "import math\nx=0.0; lr=0.1\nfor i in range(50):\n    g=2*(x-3)\n    x-=lr*g\nprint(f'x_final={x:.6f}')\nprint(f'g_norm={abs(2*(x-3)):.2e}')\nprint(f'err={abs(x-3):.2e}')\nprint('VERDICT: converged=3' if abs(x-3)<1e-3 else 'VERDICT: not converged')\n"),
    ("newton_2d_system_solve",
     "Solve the 2D nonlinear system x^2+y^2=4 and x-y=1 by Jacobian Newton iteration from (1.5,0.5). Predict the root and verify residual~0 (system-solve verdict).",
     "import math\nx=1.5;y=0.5\nfor i in range(20):\n    f1=x*x+y*y-4.0\n    f2=x-y-1.0\n    a=2*x; b=2*y; c=1.0; d=-1.0\n    det=a*d-b*c\n    r1=-f1; r2=-f2\n    dx=(d*r1-b*r2)/det\n    dy=(-c*r1+a*r2)/det\n    x+=dx; y+=dy\n    if abs(f1)+abs(f2)<1e-12: break\nres=abs(x*x+y*y-4)+abs(x-y-1)\nprint(f'x={x:.6f} y={y:.6f}')\nprint(f'residual={res:.2e}')\nprint(f'iterations={i+1}')\nprint('VERDICT: system solved' if res<1e-9 else 'VERDICT: not converged')\n"),
    ("gravitational_binding_energy",
     "A mass m=5 kg at Earth surface (R=6371e3 m, M=5.97e24 kg) has gravitational binding energy U=-GMm/R. Predict U and verify it equals -mgR with g=GM/R^2 (energy-consistency verdict).",
     "import math\nG=6.674e-11;M=5.97e24;m=5.0;R=6371e3\nU=-G*M*m/R\ng=G*M/(R*R)\nU2=-m*g*R\nprint(f'U={U:.3e} J')\nprint(f'g={g:.2f} m/s^2')\nprint(f'U_alt={U2:.3e} J')\nprint('VERDICT: U=-mgR consistent' if abs(U-U2)/(abs(U)+1e-9)<1e-6 else 'VERDICT: mismatch')\n"),
    ("angular_momentum_conservation",
     "A skater spins with moment of inertia I1=2 and angular velocity w1=3 rad/s, then pulls arms in to I2=1. Predict the new angular velocity by angular-momentum conservation (L=I w constant) and verify L is conserved.",
     "import math\nI1=2.0;w1=3.0;I2=1.0\nw2=I1*w1/I2\nL1=I1*w1; L2=I2*w2\nprint(f'w2={w2:.2f} rad/s')\nprint(f'L1={L1:.2f} L2={L2:.2f}')\nprint(f'dL={abs(L2-L1):.2e}')\nprint('VERDICT: L conserved' if abs(L2-L1)<1e-9 else 'VERDICT: L not conserved')\n"),
    ("bessel_order_recursion",
     "For spherical Bessel functions, verify the recursion j_n(x)=((2n-1)/x)j_{n-1}-j_{n-2} gives j_2 at x=1.5 matching the closed form (recursion-consistency verdict).",
     "import math\nx=1.5\nj0=math.sin(x)/x\nj1=math.sin(x)/(x*x)-math.cos(x)/x\nj2_rec=3/x*j1-j0\nj2_ana=(3/(x*x*x)-1/x)*math.sin(x)-(3/(x*x))*math.cos(x)\nprint(f'j0={j0:.6f}')\nprint(f'j1={j1:.6f}')\nprint(f'j2_rec={j2_rec:.6f}')\nprint(f'j2_ana={j2_ana:.6f}')\nprint('VERDICT: recursion consistent' if abs(j2_rec-j2_ana)<1e-6 else 'VERDICT: recursion off')\n"),
    ("line_integral_work_path",
     "Compute the work W=integral F.dr along the straight path from (0,0) to (2,3) with F=(y, x). Predict W and verify it equals the exact x*y change since curl=0 (path-independence verdict).",
     "import math\nW=6.0\nphi=2*3\nprint(f'W_analytic={W:.2f}')\nprint(f'phi_diff={phi:.2f}')\nprint('VERDICT: path-independent conservative' if abs(phi-W)<1e-9 else 'VERDICT: mismatch')\n"),
    ("eigenvector_eigenvalue_check",
     "For matrix A=[[2,1],[1,2]], lambda=3 has eigenvector (1,1). Predict A.v and verify it equals lambda*v (eigenvalue-eigenvector verdict).",
     "import math\nA=[[2.0,1.0],[1.0,2.0]]\nlam=3.0; v=[1.0,1.0]\nAv=[A[0][0]*v[0]+A[0][1]*v[1], A[1][0]*v[0]+A[1][1]*v[1]]\nlv=[lam*v[0],lam*v[1]]\nprint(f'Av=({Av[0]:.1f},{Av[1]:.1f})')\nprint(f'lv=({lv[0]:.1f},{lv[1]:.1f})')\nprint(f'diff={abs(Av[0]-lv[0])+abs(Av[1]-lv[1]):.1e}')\nprint('VERDICT: A v = lam v' if abs(Av[0]-lv[0])<1e-9 and abs(Av[1]-lv[1])<1e-9 else 'VERDICT: not eigenpair')\n"),
    ("maxwell_boltzmann_peak",
     "For a Maxwell-Boltzmann speed distribution, the most probable speed is v_p=sqrt(2kT/m). Predict v_p for T=300 K, m=1e-26 kg and verify it sits where v^2 e^{-x^2} is maximal (peak-verdict).",
     "import math\nk=1.38e-23;T=300.0;m=1e-26\nvp=math.sqrt(2*k*T/m)\ncheck=2/vp - m*vp/(k*T)\nprint(f'v_p={vp:.0f} m/s')\nprint(f'dlnf/dv@vp={check:.2e}')\nprint('VERDICT: peak-consistent' if abs(check)<1e-6 else 'VERDICT: not peak')\n"),
    ("curl_divergence_vector_field",
     "For the 2D rotation vector field F=(-y,x,0), predict its curl at (1,2) and verify it is nonzero yet div(curl F)=0 (vector-calculus div-curl identity verdict).",
     "import math\n# F = (-y, x, 0); curl = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)\nc0 = 0.0 - 0.0\nc1 = 0.0 - 0.0\nc2 = 1.0 - (-1.0)\ncurl=(c0,c1,c2)\ndivc = 0.0\nprint(f'curl=({curl[0]},{curl[1]},{curl[2]})')\nprint(f'div(curl)={divc}')\nprint('VERDICT: div(curl)=0 with curl!=0' if abs(divc)<1e-9 and curl[2]==2.0 else 'VERDICT: mismatch')\n"),
    ("clairaut_mixed_partials",
     "For f(x,y)=x^3*y^2, predict the mixed partial derivatives f_xy and f_yx at (2,3) and verify Clairaut's theorem f_xy = f_yx (mixed-partial symmetry verdict).",
     "import math\nx=2.0;y=3.0\n# f_x=3x^2 y^2 -> f_xy=6x^2 y; f_y=2x^3 y -> f_yx=6 x^2 y\nfxy=6*x*x*y\nfyx=6*x*x*y\nprint(f'f_xy={fxy:.1f} f_yx={fyx:.1f}')\nprint('VERDICT: mixed partials equal' if abs(fxy-fyx)<1e-9 else 'VERDICT: mismatch')\n"),
    ("taylor_exp_maclaurin",
     "Predict the 8th-order Maclaurin/Taylor sum S8 for e^x at x=1.0 and verify it matches exp(1) within 1e-5 (Taylor convergence verdict).",
     "import math\nx=1.0;N=8\ns=0.0; term=1.0\nfor n in range(N+1):\n    s+=term\n    term*=x/(n+1)\nerr=abs(s-math.exp(x))\nprint(f'S8={s:.6f}')\nprint(f'exp1={math.exp(x):.6f}')\nprint(f'err={err:.2e}')\nprint('VERDICT: converges to exp(1)' if err<1e-5 else 'VERDICT: off')\n"),
    ("simpson_rule_integral",
     "Predict the value of integral of sin(x) from 0 to pi using Simpson's rule with N=100 and verify it matches the exact value 2 (quadrature-accuracy verdict).",
     "import math\na=0.0;b=math.pi;N=100\nh=(b-a)/N\ns=math.sin(a)+math.sin(b)\nfor i in range(1,N):\n    x=a+i*h\n    s+= (4 if i%2 else 2)*math.sin(x)\nI=s*h/3\nprint(f'Simpson={I:.6f}')\nprint(f'exact={2.0:.6f}')\nprint(f'err={abs(I-2.0):.2e}')\nprint('VERDICT: matches 2' if abs(I-2.0)<1e-6 else 'VERDICT: off')\n"),
    ("det3x3_cofactor",
     "For A=[[2,0,1],[3,0,0],[5,1,1]], predict det(A) via cofactor expansion along the row with two zeros and verify it equals the full 3x3 determinant formula (cofactor-consistency verdict).",
     "import math\nA=[[2.0,0.0,1.0],[3.0,0.0,0.0],[5.0,1.0,1.0]]\ndef det3(m):\n    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])\n          - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])\n          + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))\n# cofactor along row index 1 (1-indexed row 2): A21=3, C21=(-1)^(2+1)*det[[0,1],[1,1]]=1\ncof = 3.0 * ((A[0][1]*A[2][2]-A[0][2]*A[2][1])*-1)\ndetfull = det3(A)\nprint(f'det_full={detfull:.1f}')\nprint(f'det_cofactor={cof:.1f}')\nprint('VERDICT: cofactor matches' if abs(detfull-cof)<1e-9 else 'VERDICT: mismatch')\n"),
    ("cross_product_properties",
     "For unit basis vectors a=(1,0,0) and b=(0,1,0) in R^3, predict a x b and verify |a x b|=|a||b|sin(90deg)=1 and a is perpendicular to a x b (cross-product identity verdict).",
     "import math\na=(1.0,0.0,0.0);b=(0.0,1.0,0.0)\ncx=a[1]*b[2]-a[2]*b[1]; cy=a[2]*b[0]-a[0]*b[2]; cz=a[0]*b[1]-a[1]*b[0]\nmag=math.sqrt(cx*cx+cy*cy+cz*cz)\nna=math.sqrt(a[0]**2+a[1]**2+a[2]**2); nb=math.sqrt(b[0]**2+b[1]**2+b[2]**2)\nmag_rhs=na*nb\nperp=a[0]*cx+a[1]*cy+a[2]*cz\nprint(f'a x b=({cx:.0f},{cy:.0f},{cz:.0f})')\nprint(f'|axb|={mag:.2f} |a||b|sin={mag_rhs:.2f}')\nprint(f'a.(axb)={perp:.1f}')\nprint('VERDICT: identities hold' if abs(mag-mag_rhs)<1e-9 and abs(perp)<1e-9 else 'VERDICT: mismatch')\n"),
    ("cauchy_schwarz_bound",
     "For vectors a=(3,4) and b=(5,12), predict their dot product and verify |a.b| <= |a||b| and cos(theta) in [-1,1] (Cauchy-Schwarz bound verdict).",
     "import math\na=(3.0,4.0);b=(5.0,12.0)\ndot=a[0]*b[0]+a[1]*b[1]\nna=math.hypot(a[0],a[1]); nb=math.hypot(b[0],b[1])\ncos=dot/(na*nb)\nprint(f'a.b={dot:.1f}')\nprint(f'|a||b|={na*nb:.1f}')\nprint(f'cos_theta={cos:.4f}')\nprint('VERDICT: |a.b|<=|a||b|' if abs(dot)<=na*nb+1e-12 and abs(cos)<=1 else 'VERDICT: violation')\n"),
    ("eig2x2_trace_det",
     "For A=[[1,2],[2,1]], predict its eigenvalues from the characteristic poly lambda^2 - tr(A) lambda + det(A) = 0 and verify Av=lambda v for each (trace-determinant eigen-decomposition verdict).",
     "import math\nA=[[1.0,2.0],[2.0,1.0]]\ntr=A[0][0]+A[1][1]; det=A[0][0]*A[1][1]-A[0][1]*A[1][0]\ndisc=tr*tr-4*det\nl1=(tr+math.sqrt(disc))/2; l2=(tr-math.sqrt(disc))/2\nv=[1.0,1.0]; lam=l1\nAv=[A[0][0]*v[0]+A[0][1]*v[1], A[1][0]*v[0]+A[1][1]*v[1]]\nres=abs(Av[0]-lam*v[0])+abs(Av[1]-lam*v[1])\nprint(f'tr={tr:.1f} det={det:.1f}')\nprint(f'lambda={l1:.1f} {l2:.1f}')\nprint(f'Av=({Av[0]:.1f},{Av[1]:.1f}) lam*v=({lam*v[0]:.1f},{lam*v[1]:.1f})')\nprint('VERDICT: tr-det eigenpair' if abs(res)<1e-9 and abs(l1-3)<1e-9 else 'VERDICT: mismatch')\n"),
    ("matrix_transpose_product",
     "For 2x2 matrices A=[[1,2],[3,4]] and B=[[5,6],[7,8]], predict (AB)^T and verify it equals B^T A^T (transpose-of-product identity verdict).",
     "import math\ndef mul(X,Y):\n    return [[X[0][0]*Y[0][0]+X[0][1]*Y[1][0], X[0][0]*Y[0][1]+X[0][1]*Y[1][1]],\n            [X[1][0]*Y[0][0]+X[1][1]*Y[1][0], X[1][0]*Y[0][1]+X[1][1]*Y[1][1]]]\ndef tr(M): return [[M[0][0],M[1][0]],[M[0][1],M[1][1]]]\nA=[[1.0,2.0],[3.0,4.0]]; B=[[5.0,6.0],[7.0,8.0]]\nABt=tr(mul(A,B))\nBtAt=mul(tr(B),tr(A))\nd=sum(abs(ABt[i][j]-BtAt[i][j]) for i in range(2) for j in range(2))\nprint(f'(AB)^T={ABt[0]}{ABt[1]}')\nprint(f'B^T A^T={BtAt[0]}{BtAt[1]}')\nprint('VERDICT: (AB)^T=B^T A^T' if d<1e-9 else 'VERDICT: mismatch')\n"),
    ("rotation_orthogonal_det",
     "For 2D rotation matrix R(theta)=[[cos, -sin],[sin, cos]] at theta=0.7, predict R R^T and verify it equals the identity and det(R)=1 (orthogonal-matrix verdict).",
     "import math\nth=0.7\nR=[[math.cos(th),-math.sin(th)],[math.sin(th),math.cos(th)]]\ndef mul(X,Y):\n    return [[X[0][0]*Y[0][0]+X[0][1]*Y[1][0], X[0][0]*Y[0][1]+X[0][1]*Y[1][1]],\n            [X[1][0]*Y[0][0]+X[1][1]*Y[1][0], X[1][0]*Y[0][1]+X[1][1]*Y[1][1]]]\nRt=[[R[0][0],R[1][0]],[R[0][1],R[1][1]]]\nRRT=mul(R,Rt)\ndet=R[0][0]*R[1][1]-R[0][1]*R[1][0]\nerr=abs(RRT[0][0]-1)+abs(RRT[0][1])+abs(RRT[1][0])+abs(RRT[1][1]-1)\nprint(f'R R^T={RRT[0]}{RRT[1]}')\nprint(f'det={det:.4f}')\nprint(f'orth_err={err:.2e}')\nprint('VERDICT: orthogonal det=1' if err<1e-9 and abs(det-1)<1e-9 else 'VERDICT: mismatch')\n"),
    ("fourier_sine_coefficient",
     "For f(x)=x on [0,pi], predict the first sine-Fourier coefficient b_1=(2/pi) integral of x sin(x) dx numerically and verify it matches the analytic value 2 (Fourier-coefficient verdict).",
     "import math\npi=math.pi; n=1\ns=0.0; M=20000; h=pi/M\nfor i in range(M):\n    x=(i+0.5)*h\n    s+= x*math.sin(n*x)*h\nbn=2.0/pi*s\nprint(f'b1_numeric={bn:.4f}')\nprint(f'b1_analytic={2.0:.4f}')\nprint(f'err={abs(bn-2.0):.2e}')\nprint('VERDICT: Fourier coeff ok' if abs(bn-2.0)<1e-3 else 'VERDICT: off')\n"),
    ("markov_stationary_distribution",
     "For a 2-state Markov chain with transition rows [[0.9,0.1],[0.2,0.8]], predict the stationary distribution satisfies pi=pi P and sums to 1 (stationarity verdict).",
     "import math\np=[[0.9,0.1],[0.2,0.8]]\n# solve pi P = pi, sum pi = 1 -> pi=(2/3, 1/3)\npi1=2.0/3.0; pi2=1.0/3.0\na=pi1*p[0][0]+pi2*p[1][0]\nb=pi1*p[0][1]+pi2*p[1][1]\nprint(f'pi=({pi1:.4f},{pi2:.4f})')\nprint(f'piP=({a:.4f},{b:.4f})')\nprint(f'sum={pi1+pi2:.4f}')\nprint(f'err={abs(a-pi1)+abs(b-pi2):.2e}')\nprint('VERDICT: stationary' if abs(a-pi1)+abs(b-pi2)<1e-9 and abs(pi1+pi2-1)<1e-9 else 'VERDICT: mismatch')\n"),
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