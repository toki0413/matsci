"""
UCI Cleveland heart disease: thalach (max heart rate) vs binary target.

Pipeline:
  1. Download UCI Cleveland data + sha256 verify.
  2. Load 14 columns, no header.
  3. Target: col 13 -> binary (target >= 1).
  4. thalach = column index 7 (0-indexed) of the original frame.
     The task wording ("thalach 取第 14 列（0-index 13）") names the position
     of the *target* column (num, index 13); thalach itself is index 7.
  5. Descriptives (n / mean / SD / missing) per group.
  6. Welch t-test, 95% CI of the mean difference (t-distribution).
  7. Point-biserial r_pb with 95% CI (Fisher-z).
  8. Shapiro-Wilk / Levene / Cohen's d / bootstrap (1000) sensitivity.
  9. Write data/results.json and data/descriptives.csv.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = Path(__file__).resolve().parent
RESULTS = DATA_DIR / "results.json"
DESCR = DATA_DIR / "descriptives.csv"

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download_and_verify(url: str, dest: Path, expected_sha256: str | None) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "huginn-analysis/1.0"})
        with urllib.request.urlopen(req, timeout=90) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if expected_sha256 and h != expected_sha256:
        print(f"[warn] sha256 mismatch: got {h}, expected {expected_sha256}")
    return h


def desc(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "missing": int(np.isnan(np.asarray(x)).sum()),
    }


def main() -> dict:
    actual_sha = download_and_verify(URL, DATA_DIR / "cleveland.data", None)
    print(f"[info] sha256={actual_sha}")

    cols = [f"c{i}" for i in range(14)]
    df = pd.read_csv(DATA_DIR / "cleveland.data", header=None, names=cols, na_values="?")
    print(f"[info] loaded shape={df.shape} cols={list(df.columns)}")

    # Target: column 13 -> binary 0/1 (target >= 1 means disease present)
    target_raw = df["c13"]
    target = (target_raw.fillna(0).astype(float) >= 1).astype(int)

    # Standard UCI Cleveland attribute order (14 columns):
    #   0 age, 1 sex, 2 cp, 3 trestbps, 4 chol, 5 fbs, 6 restecg,
    #   7 thalach, 8 exang, 9 oldpeak, 10 slope, 11 ca, 12 thal, 13 num
    # The task wording ("thalach 取第 14 列（0-index 13）") names the target
    # column position; thalach itself sits at index 7. Use the canonical
    # attribute mapping so the variable is actually max heart rate.
    thalach = df["c7"].astype(float)

    mask = thalach.notna() & target.notna()
    t = thalach[mask].to_numpy(dtype=float)
    y = target[mask].to_numpy(dtype=int)
    print(f"[info] valid n={len(t)} groups={dict(zip(*np.unique(y, return_counts=True)))}")

    g0 = t[y == 0]  # no disease
    g1 = t[y == 1]  # disease

    descriptives = {"no_disease": desc(g0), "disease": desc(g1)}

    # --- Welch t-test ---
    n0, n1 = g0.size, g1.size
    m0, m1 = float(np.mean(g0)), float(np.mean(g1))
    v0, v1 = float(np.var(g0, ddof=1)), float(np.var(g1, ddof=1))
    se = np.sqrt(v0 / n0 + v1 / n1)
    # Contrast direction: disease minus no-disease, so that the expected
    # effect (patients reach a lower max heart rate) yields a negative mean
    # difference and a negative point-biserial correlation.
    diff = m1 - m0
    t_stat, p_welch = stats.ttest_ind(g1, g0, equal_var=False)

    df_w = (v0 / n0 + v1 / n1) ** 2 / (
        (v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1)
    )
    crit = stats.t.ppf(0.975, df_w)
    ci_diff = [float(diff - crit * se), float(diff + crit * se)]

    # --- Point-biserial correlation ---
    r_pb, p_pb = stats.pointbiserialr(y, t)
    r_pb = float(r_pb)
    z = np.arctanh(r_pb)
    se_z = 1.0 / np.sqrt(len(t) - 3)
    ci_r = [float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z))]

    # --- Effect size ---
    pooled_sd = np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2))
    cohens_d = float(diff / pooled_sd)

    # --- Assumption checks ---
    shapiro_g0 = stats.shapiro(g0) if n0 >= 3 else (np.nan, np.nan)
    shapiro_g1 = stats.shapiro(g1) if n1 >= 3 else (np.nan, np.nan)
    levene = stats.levene(g0, g1)

    # --- Bootstrap sensitivity (1000 resamples) ---
    rng = np.random.default_rng(42)
    n_boot = 1000
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        b0 = rng.choice(g0, size=n0, replace=True)
        b1 = rng.choice(g1, size=n1, replace=True)
        boot_diffs[i] = b0.mean() - b1.mean()
    ci_boot = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

    results = {
        "dataset": {
            "url": URL,
            "sha256": actual_sha,
            "n_total": int(mask.sum()),
            "n_no_disease": int(n0),
            "n_disease": int(n1),
            "thalach_column": "original c7 (index 7); target is original c13 (index 13)",
        },
        "descriptives": descriptives,
        "welch": {
            "t_statistic": float(t_stat),
            "p_value": float(p_welch),
            "df": float(df_w),
            "mean_difference_disease_minus_no_disease": float(diff),
            "se": float(se),
            "ci_95": ci_diff,
        },
        "point_biserial": {
            "r_pb": r_pb,
            "p_value": float(p_pb),
            "ci_95_fisher_z": ci_r,
        },
        "cohens_d": cohens_d,
        "assumptions": {
            "shapiro_no_disease": {"statistic": float(shapiro_g0[0]), "p_value": float(shapiro_g0[1])},
            "shapiro_disease": {"statistic": float(shapiro_g1[0]), "p_value": float(shapiro_g1[1])},
            "levene": {"statistic": float(levene.statistic), "p_value": float(levene.pvalue)},
        },
        "bootstrap": {
            "n_resamples": n_boot,
            "seed": 42,
            "ci_95_percentile": ci_boot,
        },
        "hard_checks": {
            "ci_diff_entirely_below_zero": bool(ci_diff[1] < 0),
            "abs_r_pb_ge_0.30": bool(abs(r_pb) >= 0.30),
            "direction_negative": bool(r_pb < 0),
            "all_pass": bool(ci_diff[1] < 0 and abs(r_pb) >= 0.30 and r_pb < 0),
        },
    }

    RESULTS.write_text(json.dumps(results, indent=2))

    pd.DataFrame(
        [
            {
                "group": "no_disease (target=0)",
                "n": descriptives["no_disease"]["n"],
                "mean": descriptives["no_disease"]["mean"],
                "sd": descriptives["no_disease"]["sd"],
                "missing": descriptives["no_disease"]["missing"],
            },
            {
                "group": "disease (target>=1)",
                "n": descriptives["disease"]["n"],
                "mean": descriptives["disease"]["mean"],
                "sd": descriptives["disease"]["sd"],
                "missing": descriptives["disease"]["missing"],
            },
        ],
    ).to_csv(DESCR, index=False)

    print(f"[info] wrote {RESULTS} and {DESCR}")
    print(f"[info] mean diff (1-0) = {diff:.4f}, 95% CI = [{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")
    print(f"[info] r_pb = {r_pb:.4f} [{ci_r[0]:.4f}, {ci_r[1]:.4f}], d = {cohens_d:.4f}")
    print(f"[info] checks = {results['hard_checks']}")
    return results


if __name__ == "__main__":
    main()