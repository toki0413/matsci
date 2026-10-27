"""Validate the three hard conditions against data/results.json.

Hard conditions:
  1. The 95% CI of the mean difference lies entirely below zero.
  2. |r_pb| >= 0.30.
  3. The direction is negative (r_pb < 0).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results.json"

REQUIRED_KEYS = [
    "welch.ci_95",
    "welch.mean_difference_disease_minus_no_disease",
    "point_biserial.r_pb",
]


def get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        cur = cur[part]
    return cur


def main() -> int:
    data = json.loads(RESULTS.read_text())

    ci = get(data, "welch.ci_95")
    diff = get(data, "welch.mean_difference_disease_minus_no_disease")
    r_pb = get(data, "point_biserial.r_pb")

    checks = {
        "ci_diff_entirely_below_zero": bool(ci[1] < 0),
        "abs_r_pb_ge_0.30": bool(abs(r_pb) >= 0.30),
        "direction_negative": bool(r_pb < 0),
    }

    print(f"[info] mean difference (disease - no disease) = {diff:+.4f}")
    print(f"[info] 95% CI of mean difference = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"[info] r_pb = {r_pb:.4f} (|r_pb| = {abs(r_pb):.4f})")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    all_pass = all(checks.values())
    print(f"[{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}]")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())