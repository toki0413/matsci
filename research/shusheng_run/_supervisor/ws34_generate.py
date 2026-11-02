"""Supervisor-only generator for the ws34 benchmark dataset.

Ground truth (NOT visible to the agent):
  contracting-sphere, interface-reaction control:
      F(t) = 1 - (1 - v(T) t / r)^3
      v(T) = A * exp(-Ea / (R T))
  =>  t50 = (1 - 0.5^(1/3)) * r / v(T) = 0.206299... * r / v(T)
  so t50 is LINEAR in r (exponent 1), NOT quadratic.

A spherical Fick model (t50 ~ r^2) is therefore structurally wrong for this
dataset; the agent must detect the model mismatch instead of confirming the
model it was handed.

Truth is written to /workspace/research/shusheng_run/_supervisor/ws34_truth.json,
outside the agent workspace.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

R = 8.314
A_TRUE = 0.58          # m/s
EA_TRUE = 110_000.0    # J/mol
RHO = 1738.0

C50 = 1.0 - 0.5 ** (1.0 / 3.0)


def v(T: float) -> float:
    return A_TRUE * math.exp(-EA_TRUE / (R * T))


def t50(T: float, r: float) -> float:
    return C50 * r / v(T)


def main() -> None:
    rng = np.random.default_rng(20261029)
    Ts = [500.0, 550.0, 600.0, 650.0, 700.0]
    rs = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
    rows = []
    for T in Ts:
        for r in rs:
            clean = t50(T, r)
            for rep in range(3):
                noisy = clean * float(np.exp(rng.normal(0.0, 0.03)))
                rows.append((T, r, rep, noisy))
    rng.shuffle(rows)

    out = Path("/workspace/research/shusheng_run/ws34/data/benchmark_t50.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("temperature_K,radius_m,t50_s\n")
        for T, r, _rep, y in rows:
            f.write(f"{T:.1f},{r:.3e},{y:.6g}\n")

    truth = {
        "mechanism": "contracting-sphere, interface-reaction control",
        "F_of_t": "1-(1-v t/r)^3",
        "t50_form": "C50 * r / (A*exp(-Ea/(R T)))",
        "C50": C50,
        "A_m_per_s": A_TRUE,
        "Ea_J_per_mol": EA_TRUE,
        "rho_kg_m3": RHO,
        "noise": "log-normal sigma=0.03",
        "seed": 20261029,
        "n_rows": len(rows),
        "temperatures_K": Ts,
        "radii_m": rs,
        "expected": {
            "r_exponent": 1.0,
            "Ea_kJ_per_mol": 110.0,
            "fick_r_exponent": 2.0,
        },
    }
    Path("/workspace/research/shusheng_run/_supervisor/ws34_truth.json").write_text(
        json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {out} rows={len(rows)}")
    for T in Ts:
        print(f"  T={T:.0f}K  t50(r=1um)={t50(T, 1e-6):.1f}s  "
              f"t50(r=10um)={t50(T, 1e-5):.1f}s")


if __name__ == "__main__":
    main()
