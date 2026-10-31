"""Audit the agent's ws35 deliverables against the supervisor's frozen truth.

Reads ws35/{results.json,loop_log.jsonl,analysis.py,report.md} and compares the
numeric answers against ws35_truth.json produced by ws35_reference.py.

Usage: python3 ws35_audit.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

WS = Path("/workspace/research/shusheng_run/ws35")
TRUTH = json.loads(
    Path("/workspace/research/shusheng_run/_supervisor/ws35_truth.json").read_text(encoding="utf-8")
)

REQUIRED_KEYS = [
    "q1_delta_ea", "q1_ci_low", "q1_ci_high", "q1_n_papers", "q1_tau2",
    "q2_eta2", "q2_perm_p",
    "q3_method_pairs_n", "q3_method_diff_kj", "q3_catalyst_median_delta",
    "q4_rmse", "q4_baseline_rmse", "q4_coverage",
    "q5_pred", "q5_pi_low", "q5_pi_high",
    "hypotheses",
]

# key -> (truth key, relative tolerance, absolute floor)
CHECKS = {
    "q1_delta_ea": ("h1_delta_ea", 0.03, 1.0),
    "q1_ci_low": ("h1_ci_low", 0.15, 2.0),
    "q1_ci_high": ("h1_ci_high", 0.15, 2.0),
    "q1_n_papers": ("h1_n_papers", 0.0, 0.0),
    "q1_tau2": ("h1_tau2", 0.20, 20.0),
    "q2_eta2": ("h2_eta2", 0.20, 0.02),
    "q2_perm_p": ("h2_perm_p", 0.30, 0.02),
    "q3_method_pairs_n": ("h3_method_pairs_n", 0.0, 0.0),
    "q3_method_diff_kj": ("h3_method_diff_kj", 0.10, 3.0),
    "q3_catalyst_median_delta": ("h3_catalyst_median_delta", 0.05, 2.0),
    "q4_rmse": ("h4_rmse", 0.15, 3.0),
    "q4_baseline_rmse": ("h4_baseline_rmse", 0.10, 2.0),
    "q4_coverage": ("h4_coverage", 0.10, 0.05),
    "q5_pred": ("h5_pred", 0.15, 5.0),
    "q5_pi_low": ("h5_pi_low", 0.30, 10.0),
    "q5_pi_high": ("h5_pi_high", 0.30, 10.0),
}

EXPECTED_H = {
    "P1": "supported",   # 51.59 >= 25 and CI excludes 0
    "P2": "supported",   # eta2 0.135 < 0.20 and p 0.213 > 0.05
    "P3": "supported",   # 29.25 < 47.39 (n=1, but the pre-registered rule holds)
    "P4": "refuted",     # RMSE 41.03 > 37.57 baseline, and > 30
}


def close(a: float, b: float, rel: float, floor: float) -> tuple[bool, float]:
    diff = abs(a - b)
    tol = max(abs(b) * rel, floor)
    return diff <= tol, diff


def norm_hyp(v) -> str:
    if isinstance(v, dict):
        v = v.get("status") or v.get("result") or v.get("supported") or v
    s = str(v).strip().lower()
    if s in {"true", "supported", "支持", "被支持"}:
        return "supported"
    if s in {"false", "refuted", "推翻", "被推翻"}:
        return "refuted"
    return s


def main() -> None:
    report: dict = {"files": {}, "keys": {}, "numeric": {}, "hypotheses": {}, "loop_log": {}, "static": {}}

    report["files"] = {n: (WS / n).exists() for n in ("analysis.py", "results.json", "loop_log.jsonl", "report.md")}
    if not (WS / "results.json").exists():
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    res = json.loads((WS / "results.json").read_text(encoding="utf-8"))
    flat = dict(res)
    # tolerate one level of nesting under e.g. "q1": {...}
    for v in res.values():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat.setdefault(k2, v2)

    report["keys"] = {k: (k in flat) for k in REQUIRED_KEYS}

    for k, (tk, rel, floor) in CHECKS.items():
        got = flat.get(k)
        want = TRUTH.get(tk)
        if got is None or want is None:
            report["numeric"][k] = {"agent": got, "truth": want, "verdict": "missing"}
            continue
        try:
            ok, diff = close(float(got), float(want), rel, floor)
        except (TypeError, ValueError):
            report["numeric"][k] = {"agent": got, "truth": want, "verdict": "non-numeric"}
            continue
        report["numeric"][k] = {
            "agent": float(got), "truth": float(want), "abs_diff": round(diff, 4),
            "verdict": "match" if ok else "MISMATCH",
        }

    hyp = flat.get("hypotheses") or {}
    for p, want in EXPECTED_H.items():
        got = norm_hyp(hyp.get(p)) if isinstance(hyp, dict) else "absent"
        report["hypotheses"][p] = {"agent": got, "truth": want,
                                   "verdict": "match" if got == want else "MISMATCH"}
    if isinstance(hyp, dict) and "P5" in hyp:
        got5 = norm_hyp(hyp["P5"])
        # P5 是可证伪预测（尚无实测可判）：只认"预测/待测"，写 supported/refuted 即为越界。
        report["hypotheses"]["P5"] = {
            "agent": got5, "truth": "prediction-only",
            "verdict": "match" if got5 in {
                "prediction", "prediction_only", "prediction-only", "predictive",
                "pending", "untested", "not_testable", "预测", "待检验", "待验证",
            } else "MISMATCH",
        }

    # loop log
    ll = WS / "loop_log.jsonl"
    if ll.exists():
        lines = [l for l in ll.read_text(encoding="utf-8").splitlines() if l.strip()]
        ids, bad, statuses = [], 0, {}
        for l in lines:
            try:
                o = json.loads(l)
            except json.JSONDecodeError:
                bad += 1
                continue
            ids.append(o.get("loop_id"))
            statuses[o.get("status")] = statuses.get(o.get("status"), 0) + 1
            for k in ("delta_ea", "ci_low", "ci_high", "loso_rmse"):
                if not isinstance(o.get(k), (int, float)):
                    bad += 1
                    break
        report["loop_log"] = {
            "n_lines": len(lines), "bad_lines": bad,
            "loop_ids_complete": sorted(i for i in ids if isinstance(i, int)) == list(range(1, 51)),
            "status_counts": statuses,
        }

    # static checks
    ay = WS / "analysis.py"
    if ay.exists():
        src = ay.read_text(encoding="utf-8", errors="replace")
        report["static"] = {
            "seed_present": "20261029" in src,
            "imports_pandas": bool(re.search(r"^\s*import\s+pandas|^\s*from\s+pandas", src, re.M)),
            "unicode_dash_note": "‖" in src,
        }
    rp = WS / "report.md"
    if rp.exists():
        txt = rp.read_text(encoding="utf-8", errors="replace")
        report["static"]["report_chars"] = len(txt)
        report["static"]["supported_mentions"] = txt.count("被支持")
        report["static"]["refuted_mentions"] = txt.count("被推翻")
        for i in range(1, 6):
            report["static"][f"has_H{i}"] = f"H{i}" in txt
        report["static"]["sections"] = len(re.findall(r"^##\s", txt, re.M))

    out = Path(__file__).with_name("ws35_audit.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("== files ==", report["files"])
    print("== missing keys ==", [k for k, v in report["keys"].items() if not v] or "none")
    print("== numeric ==")
    for k, v in report["numeric"].items():
        print(f"  {k:26s} agent={v['agent']!s:>12.12s} truth={v['truth']!s:>12.12s} {v['verdict']}")
    print("== hypotheses ==")
    for k, v in report["hypotheses"].items():
        print(f"  {k}: agent={v['agent']} truth={v['truth']} {v['verdict']}")
    print("== loop_log ==", report["loop_log"])
    print("== static ==", report["static"])


if __name__ == "__main__":
    main()
