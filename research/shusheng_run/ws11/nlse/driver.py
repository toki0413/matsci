"""Refinement-study driver for the 1-D NLSE soliton pipeline.

Runs 10 joint (dx, dt) refinement levels for BOTH the SSFM and the Crank-Nicolson
schemes, records the L2 error at t = 0, 1, 5, 10, the energy and momentum drifts,
and performs the Richardson-extrapolation convergence-order analysis together with
the boundary-reflection plateau detection.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import analytic as A
from . import params as P
from .operators import CrankNicolson, SSFM

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _steps(dt: float, t_target: float) -> int:
    """Number of steps to reach t_target with step dt (round to nearest integer)."""
    return int(round(t_target / dt))


def run_level_ssfm(ell: int, t_check=P.T_CHECK):
    """Run one SSFM refinement level. Returns a dict of results."""
    N, dx, dt, x, k = A.level_params(ell)
    psi0 = A.exact(x, 0.0)

    integrator = SSFM(x, k, dx, dt)

    out = {"level": ell, "N": N, "dx": dx, "dt": dt, "scheme": "SSFM"}
    t_elapsed = 0.0
    for t_target in t_check:
        nsteps = _steps(dt, t_target)
        t0 = time.perf_counter()
        psi, _ = integrator.march(psi0, nsteps, record_every=0)
        t_elapsed += time.perf_counter() - t0
        psi_ex = A.exact(x, t_target)
        err = A.l2_relative(psi, psi_ex)
        E = A.energy_numerical(psi, dx)
        Pmom = A.momentum_numerical(psi, dx)
        out[f"t{t_target:g}"] = {
            "steps": nsteps,
            "l2_rel": err,
            "E": E,
            "dE_E0": (E - P.E0_EXACT) / P.E0_EXACT,
            "P": Pmom,
            "dP_P0": (Pmom - P.P0_EXACT) / P.P0_EXACT,
        }
    out["wall_s"] = t_elapsed
    return out


def run_level_cn(ell: int, t_check=P.T_CHECK):
    """Run one Crank-Nicolson refinement level."""
    N, dx, dt, x, _ = A.level_params(ell)
    psi0 = A.exact(x, 0.0)

    integrator = CrankNicolson(x, dx, dt)

    out = {"level": ell, "N": N, "dx": dx, "dt": dt, "scheme": "CN"}
    t_elapsed = 0.0
    for t_target in t_check:
        nsteps = _steps(dt, t_target)
        t0 = time.perf_counter()
        psi, _ = integrator.march(psi0, nsteps, record_every=0)
        t_elapsed += time.perf_counter() - t0
        psi_ex = A.exact(x, t_target)
        err = A.l2_relative(psi, psi_ex)
        E = A.energy_numerical(psi, dx)
        Pmom = A.momentum_numerical(psi, dx)
        out[f"t{t_target:g}"] = {
            "steps": nsteps,
            "l2_rel": err,
            "E": E,
            "dE_E0": (E - P.E0_EXACT) / P.E0_EXACT,
            "P": Pmom,
            "dP_P0": (Pmom - P.P0_EXACT) / P.P0_EXACT,
        }
    out["wall_s"] = t_elapsed
    return out


def richardson_analysis(results: list[dict], t_key: str = "t10"):
    """Observed convergence order and Richardson-extrapolated limit.

    With a refinement ratio of 2 and error model  e(h) = C h^p + e_plateau,
    the local order between levels ell and ell+1 is
        p_ell = log2(e_ell / e_{ell+1}).
    Once the plateau is reached, p_ell -> 0.
    """
    errs = np.array([r[t_key]["l2_rel"] for r in results], dtype=float)
    levels = np.array([r["level"] for r in results])
    orders = [np.nan]
    for i in range(1, len(errs)):
        orders.append(A.richardson_order(errs[i - 1], errs[i]))

    # Richardson extrapolation using the last two pre-plateau levels:
    #   e* = (2^p e_fine - e_coarse) / (2^p - 1)
    # We use the largest p that is still stable (i.e. before the plateau).
    rich = [np.nan] * len(errs)
    for i in range(1, len(errs)):
        p = orders[i]
        if not np.isfinite(p) or p < 0.5:
            continue
        rich[i] = (2.0 ** p * errs[i] - errs[i - 1]) / (2.0 ** p - 1.0)

    return {"levels": levels.tolist(),
            "l2": errs.tolist(),
            "order": orders,
            "richardson": rich}


def detect_plateau(orders: list[float], errs: list[float],
                   threshold: float = 0.5):
    """Detect the level at which the error stops decreasing (plateau onset).

    The plateau is declared at the first level where the observed order drops
    below `threshold` and stays below it thereafter.
    """
    for i in range(1, len(orders)):
        if all(o < threshold for o in orders[i:]):
            return {"onset_level": i,
                    "onset_dx": None,
                    "plateau_height": errs[i],
                    "orders_after": orders[i:]}
    return {"onset_level": None, "plateau_height": None, "orders_after": []}


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 78)
    print("1-D NLSE soliton structure-preserving verification pipeline")
    print("=" * 78)
    print(f"  Equation : i psi_t + psi_xx + 2 |psi|^2 psi = 0")
    print(f"  Exact    : psi = A sech[A(x-vt)] exp{{i[vx/2 + (A^2-v^2/4)t]}}")
    print(f"  A = {P.A},  v = {P.V},  L = {P.L},  T = {P.T_END}")
    print(f"  E0 = {P.E0_EXACT:.10f}   P0 = {P.P0_EXACT:.10f}")
    print(f"  Refinement levels: {P.N_REFINE}  (dx -> dx/2, dt -> dt/2 each level)")
    print(f"  Base: N0 = {P.N0}, dt0 = {P.DT0}")
    print()

    # ---- SSFM ----------------------------------------------------------- #
    print("-" * 78)
    print("[1/3] SSFM refinement study")
    print("-" * 78)
    ssfm = []
    for ell in range(P.N_REFINE):
        t0 = time.perf_counter()
        r = run_level_ssfm(ell)
        ssfm.append(r)
        e10 = r["t10"]["l2_rel"]
        de = r["t10"]["dE_E0"]
        dp = r["t10"]["dP_P0"]
        print(f"  level {ell:2d}: N={r['N']:7d} dx={r['dx']:.3e} dt={r['dt']:.2e}  "
              f"L2(t=10)={e10:.3e}  dE/E0={de:+.3e}  dP/P0={dp:+.3e}  "
              f"({time.perf_counter()-t0:.1f}s)")

    # ---- Crank-Nicolson ------------------------------------------------- #
    print()
    print("-" * 78)
    print("[2/3] Crank-Nicolson refinement study")
    print("-" * 78)
    cn = []
    for ell in range(P.N_REFINE):
        t0 = time.perf_counter()
        r = run_level_cn(ell)
        cn.append(r)
        e10 = r["t10"]["l2_rel"]
        de = r["t10"]["dE_E0"]
        dp = r["t10"]["dP_P0"]
        print(f"  level {ell:2d}: N={r['N']:7d} dx={r['dx']:.3e} dt={r['dt']:.2e}  "
              f"L2(t=10)={e10:.3e}  dE/E0={de:+.3e}  dP/P0={dp:+.3e}  "
              f"({time.perf_counter()-t0:.1f}s)")

    # ---- Analysis ------------------------------------------------------- #
    print()
    print("-" * 78)
    print("[3/3] Richardson analysis & boundary-reflection plateau")
    print("-" * 78)

    report = {}
    for name, res in (("SSFM", ssfm), ("CN", cn)):
        print(f"\n### {name}")
        print(f"  {'lvl':>3} {'N':>7} {'dx':>9} {'dt':>9} | "
              f"{'L2(t=1)':>10} {'L2(t=5)':>10} {'L2(t=10)':>10} | order(t=10)")
        for r in res:
            o = richardson_analysis(res, "t10")
            lvl = r["level"]
            ord_str = f"{o['order'][lvl]:.2f}" if lvl > 0 else "   -"
            print(f"  {lvl:3d} {r['N']:7d} {r['dx']:9.3e} {r['dt']:9.2e} | "
                  f"{r['t1']['l2_rel']:10.3e} {r['t5']['l2_rel']:10.3e} "
                  f"{r['t10']['l2_rel']:10.3e} | {ord_str}")

        for tk in ("t1", "t5", "t10"):
            o = richardson_analysis(res, tk)
            plat = detect_plateau(o["order"], o["l2"])
            print(f"\n  {tk}: observed orders = "
                  + " ".join(f"{v:5.2f}" if np.isfinite(v) else "  nan"
                             for v in o["order"][1:]))
            print(f"        plateau onset level = {plat['onset_level']}, "
                  f"height = {plat['plateau_height']:.3e}")
            report[f"{name.lower()}_{tk}"] = {"richardson": o, "plateau": plat}

        # Certification at the finest level
        finest = res[-1]
        l2_10 = finest["t10"]["l2_rel"]
        de_10 = abs(finest["t10"]["dE_E0"])
        dp_10 = abs(finest["t10"]["dP_P0"])
        print(f"\n  FINEST LEVEL (level {finest['level']}, N={finest['N']}, "
              f"dx={finest['dx']:.3e}, dt={finest['dt']:.2e}):")
        print(f"    L2(t=10)   = {l2_10:.3e}   target < {P.TOL_L2}  "
              f"{'PASS' if l2_10 < P.TOL_L2 else 'FAIL'}")
        print(f"    |dE/E0|    = {de_10:.3e}   target < {P.TOL_E}   "
              f"{'PASS' if de_10 < P.TOL_E else 'FAIL'}")
        print(f"    |dP/P0|    = {dp_10:.3e}   target < {P.TOL_P}  "
              f"{'PASS' if dp_10 < P.TOL_P else 'FAIL'}")
        report[f"{name.lower()}_finest"] = {
            "N": finest["N"], "dx": finest["dx"], "dt": finest["dt"],
            "l2_t10": l2_10, "dE_E0": finest["t10"]["dE_E0"],
            "dP_P0": finest["t10"]["dP_P0"],
            "pass_l2": l2_10 < P.TOL_L2,
            "pass_e": de_10 < P.TOL_E,
            "pass_p": dp_10 < P.TOL_P,
            "all_pass": (l2_10 < P.TOL_L2 and de_10 < P.TOL_E
                         and dp_10 < P.TOL_P),
        }

    # Save raw data
    with open(os.path.join(RESULTS_DIR, "ssfm.json"), "w") as f:
        json.dump(ssfm, f, indent=2, default=str)
    with open(os.path.join(RESULTS_DIR, "cn.json"), "w") as f:
        json.dump(cn, f, indent=2, default=str)
    with open(os.path.join(RESULTS_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nResults written to {RESULTS_DIR}/")
    return ssfm, cn, report


if __name__ == "__main__":
    run_all()