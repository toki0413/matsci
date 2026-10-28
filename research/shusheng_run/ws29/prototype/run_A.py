"""
Direction A (C2) -- 1D hydrogen-bed thermal model: main analysis script.

Produces:
  * charging curves (fill time, peak temperature) for a range of designs
  * grid-independence study at Nx=50 and Nx=100 (charging time, peak T, energy residual)
  * energy-conservation verification
  * a parametric sweep over (bed thickness, supply pressure) that maps the
    "safe operating envelope" -- the core design insight for C2.

All outputs are written to results/ and results.json.
"""

import json
import numpy as np
from bed_model import BedModel, energy_balance

SEED = 42
np.random.seed(SEED)

RESULTS = {}


# ---------------------------------------------------------------------------
# 0. Reference case
# ---------------------------------------------------------------------------
def reference_case():
    """Nominal design: L=20 mm, 2 bar supply, 573 K cooling."""
    m = BedModel(N=41, L=0.02, T0=573.0, T_cool=573.0, p_supply=2.0)
    # Charge until X reaches 0.95 at the cold face, or until a safety cutoff.
    times, Ts, Xs = m.simulate(t_end=5000.0, dt=5.0, record_every=10)
    X_cold = Xs[:, 0]
    i95 = int(np.argmax(X_cold >= 0.95))
    if i95 == 0 and X_cold[0] < 0.95:
        i95 = len(X_cold) - 1
    t_fill = times[i95]
    T_peak = float(np.max(Ts[:i95 + 1]))
    print(f"reference: t_fill={t_fill:.1f} s  T_peak={T_peak:.1f} K")
    return dict(t_fill_s=round(float(t_fill), 3), T_peak_K=round(T_peak, 3),
                safe=T_peak < 723.0)


# ---------------------------------------------------------------------------
# 1. Grid-independence study
# ---------------------------------------------------------------------------
def grid_study():
    """Compare Nx=50 vs Nx=100 on the reference case.

    Metrics: charging time (X_cold=0.95), peak temperature, max relative
    energy imbalance.  Richardson-style convergence is reported as the
    relative change between the two resolutions.
    """
    out = {}
    for N in (50, 100):
        m = BedModel(N=N, L=0.02, T0=573.0, T_cool=573.0, p_supply=2.0)
        times, Ts, Xs = m.simulate(t_end=5000.0, dt=2.0, record_every=5)
        eb = energy_balance(m, times, Ts, Xs, m.p_supply)
        X_cold = Xs[:, 0]
        i95 = int(np.argmax(X_cold >= 0.95))
        if i95 == 0 and X_cold[0] < 0.95:
            i95 = len(X_cold) - 1
        t_fill = float(times[i95])
        T_peak = float(np.max(Ts[:i95 + 1]))
        out[str(N)] = dict(t_fill_s=round(t_fill, 4), T_peak_K=round(T_peak, 4),
                           max_rel_imbalance=float(np.max(np.abs(eb["rel_error"]))))
    t50, t100 = out["50"]["t_fill_s"], out["100"]["t_fill_s"]
    T50, T100 = out["50"]["T_peak_K"], out["100"]["T_peak_K"]
    out["convergence"] = dict(
        dt_fill_rel=abs(t50 - t100) / t100,
        dT_peak_rel=abs(T50 - T100) / T100,
        verdict="grid independent" if abs(t50 - t100) / t100 < 0.01 else "refine further")
    RESULTS["grid_independence"] = out
    print("grid study:", out["convergence"])
    return out


# ---------------------------------------------------------------------------
# 2. Energy conservation
# ---------------------------------------------------------------------------
def energy_check():
    m = BedModel(N=101, L=0.02, T0=573.0, T_cool=573.0, p_supply=2.0)
    times, Ts, Xs = m.simulate(t_end=5000.0, dt=2.0, record_every=5)
    eb = energy_balance(m, times, Ts, Xs, m.p_supply)
    out = dict(max_rel_imbalance=float(np.max(np.abs(eb["rel_error"]))),
               final_rel_imbalance=float(np.abs(eb["rel_error"][-1])),
               n_nodes=m.N)
    RESULTS["energy_conservation"] = out
    print("energy max rel imbalance:", out["max_rel_imbalance"])
    return out


# ---------------------------------------------------------------------------
# 3. Safe envelope sweep
# ---------------------------------------------------------------------------
def envelope_sweep():
    """Map (L, p_supply) -> (fill time, peak temperature, safe/unsafe).

    A design is 'safe' if the peak temperature during charge stays below
    T_limit = 723 K (450 C), a typical MgH2 degradation/sintering threshold.
    """
    T_limit = 723.0
    Ls = [0.002, 0.005, 0.010, 0.020]
    ps = [1.05, 1.2, 1.5, 2.0, 5.0]
    table = []
    for L in Ls:
        for p in ps:
            m = BedModel(N=41, L=L, T0=573.0, T_cool=573.0, p_supply=p)
            times, Ts, Xs = m.simulate(t_end=20000.0, dt=10.0, record_every=50)
            T_max = float(np.max(Ts))
            X_cold = Xs[:, 0]
            i95 = int(np.argmax(X_cold >= 0.95))
            t_fill = float(times[i95]) if (i95 > 0 or X_cold[0] >= 0.95) else float("inf")
            safe = T_max < T_limit
            table.append(dict(L_m=L, p_bar=p, t_fill_s=round(t_fill, 2) if np.isfinite(t_fill) else None,
                              T_peak_K=round(T_max, 1), safe=safe))
    return dict(T_limit_K=T_limit, sweep=table)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Direction A (C2) 1D MgH2 thermal model ===")
    RESULTS["reference"] = reference_case()
    RESULTS["grid_independence"] = grid_study()
    RESULTS["energy_conservation"] = energy_check()
    env = envelope_sweep()
    RESULTS["safe_envelope"] = env
    for row in env["sweep"]:
        print(f"  L={row['L_m']:.3f} m p={row['p_bar']:4.2f} bar  "
              f"t_fill={str(row['t_fill_s']):>8s} s  T_peak={row['T_peak_K']:7.1f} K  "
              f"safe={row['safe']}")

    with open("results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nWrote results.json")