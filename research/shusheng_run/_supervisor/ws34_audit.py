"""Supervisor-side audit for ws34: compare agent output against held-out ground truth.

Ground truth is in ws34_truth.json (outside the agent workspace).
The agent never sees it; this script decides whether the inverse problem was
actually recovered, not merely reported.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

WS = Path("/workspace/research/shusheng_run/ws34")
TRUTH = json.loads(
    Path("/workspace/research/shusheng_run/_supervisor/ws34_truth.json").read_text()
)


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def truth_t50(T: float, r: float) -> float:
    R = 8.314
    c50 = 1.0 - 0.5 ** (1.0 / 3.0)
    v = TRUTH["A_m_per_s"] * math.exp(-TRUTH["Ea_J_per_mol"] / (R * T))
    return c50 * r / v


def main() -> None:
    print("=== ground truth ===")
    print("  r_exponent = 1.0 (contracting-sphere, reaction control)")
    print(f"  Ea = {TRUTH['Ea_J_per_mol'] / 1000:.1f} kJ/mol")
    print(f"  t50(650K, 2um) = {truth_t50(650.0, 2e-6):.1f} s")
    print(f"  sphere-Fick would predict r_exponent = 2.0")

    print("\n=== deliverables ===")
    for name in ("analysis.py", "results.json", "loop_log.jsonl", "report.md"):
        p = WS / name
        if p.is_file():
            print(f"  {name}: {p.stat().st_size} bytes")
        else:
            print(f"  {name}: MISSING")

    ll = WS / "loop_log.jsonl"
    if ll.is_file():
        rows = []
        for i, line in enumerate(ll.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                print(f"  loop_log.jsonl line {i}: NOT JSON")
        print(f"\n  loop_log lines parsed = {len(rows)} (need >= 50)")
        for key in ("r_exponent", "Ea_kJ_per_mol", "t50_pred_650K_2um_s"):
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                vals_sorted = sorted(vals)
                med = vals_sorted[len(vals_sorted) // 2]
                print(f"  {key}: n={len(vals)} median={_fmt(med)} "
                      f"min={_fmt(min(vals))} max={_fmt(max(vals))}")

    rj = WS / "results.json"
    if rj.is_file():
        try:
            data = json.loads(rj.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"\n  results.json INVALID JSON: {e}")
            return
        print("\n=== results.json top-level keys ===")
        print(" ", list(data)[:40])
        flat: dict[str, float] = {}

        def walk(o, prefix=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{prefix}[{i}]")
            elif isinstance(o, (int, float)) and not isinstance(o, bool):
                flat[prefix] = float(o)

        walk(data)
        interesting = {k: v for k, v in flat.items()
                       if any(s in k.lower() for s in
                              ("ea", "exp", "activat", "predict", "t50", "aic", "bic",
                               "exponent", "slope"))}
        print("\n=== key numeric fields (filtered) ===")
        for k, v in list(interesting.items())[:60]:
            print(f"  {k} = {_fmt(v)}")

    rep = WS / "report.md"
    if rep.is_file():
        text = rep.read_text(encoding="utf-8")
        print(f"\n=== report.md: {len(text)} chars ===")
        for token in ("Fick", "AIC", "置信", "bootstrap", "未公开", "预测"):
            print(f"  contains {token!r}: {token in text}")


if __name__ == "__main__":
    main()
