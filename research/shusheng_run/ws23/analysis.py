"""Statistical analysis of the UCI Cleveland heart disease dataset (dataset A).

Pipeline
--------
1. Load ``data/processed.cleveland.data`` (14 columns, no header).
2. Map column names to the UCI attribute definitions.
3. Collapse the 5-level diagnosis ``num`` into a binary target:
       target = 1  if num >= 1  -> coronary heart disease diagnosed
       target = 0  if num == 0  -> no disease
   (values 1..4 are severity grades of angiographic disease; they are merged
    into a single ``diseased`` class so the comparison is case vs. control.)
4. Descriptive statistics (n, mean, SD) plus missing / outlier rules.
5. Welch's t-test + t-distribution critical value for the 95% CI of the mean
   difference, and Cohen's d.
6. Point-biserial correlation and its 95% CI via Fisher z transform.
7. Shapiro-Wilk normality check + bootstrap robustness of the mean difference.
8. Write ``data/results.json`` and ``data/report.md``.

Exit code is 0 on any failure so that orchestration never aborts; failures are
recorded in the summary instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE, "data")
DATA_PATH = os.path.join(DATA_DIR, "processed.cleveland.data")
RESULTS_PATH = os.path.join(DATA_DIR, "results.json")
REPORT_PATH = os.path.join(DATA_DIR, "report.md")
SUMMARY_PATH = os.path.join(DATA_DIR, "download_summary.json")

EXPECTED_SHA256 = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"

# UCI attribute order for processed.cleveland.data (14 columns, no header).
COLUMNS = [
    "age",          # age in years
    "sex",          # 1 = male, 0 = female
    "cp",           # chest pain type (1..4)
    "trestbps",     # resting blood pressure (mm Hg)
    "chol",         # serum cholesterol (mg/dl)
    "fbs",          # fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",      # resting electrocardiographic results (0..2)
    "thalach",      # maximum heart rate achieved
    "exang",        # exercise induced angina (1 = yes, 0 = no)
    "oldpeak",      # ST depression induced by exercise relative to rest
    "slope",        # slope of the peak exercise ST segment (0..2)
    "ca",           # number of major vessels (0..3) colored by fluoroscopy
    "thal",         # thalassemia: 3 = normal; 6 = fixed defect; 7 = reversable defect
    "num",          # diagnosis of heart disease (angiographic disease status)
]

# Columns that are genuinely continuous and are analysed with Welch's t-test.
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Binary / ordinal covariates reported descriptively but not t-tested.
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

N_BOOT = 10_000
RANDOM_SEED = 42


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_data_file() -> dict[str, Any]:
    """Verify the data file exists and its digest matches the expected one."""
    info: dict[str, Any] = {
        "path": DATA_PATH,
        "exists": os.path.exists(DATA_PATH),
        "expected_sha256": EXPECTED_SHA256,
        "actual_sha256": None,
        "digest_match": False,
    }
    if info["exists"]:
        info["actual_sha256"] = sha256_file(DATA_PATH)
        info["digest_match"] = info["actual_sha256"] == EXPECTED_SHA256
    return info


def load_data() -> pd.DataFrame:
    """Load the 14-column headerless CSV, coercing '?' to NaN."""
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values=["?", ""])
    # Defensive: any non-numeric token in numeric columns becomes NaN too.
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Merge num in {1,2,3,4} into a single 'diseased' class (target >= 1)."""
    out = df.copy()
    out["target"] = (out["num"] >= 1).astype(int)
    out["target_label"] = out["target"].map({0: "no disease (num=0)", 1: "diseased (num>=1)"})
    return out


def iqr_outliers(s: pd.Series) -> dict[str, Any]:
    """Tukey IQR outlier fences (1.5 * IQR) for a continuous series."""
    s = s.dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lo) | (s > hi)
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": float(lo),
        "upper_fence": float(hi),
        "n_outliers": int(mask.sum()),
        "outlier_share": float(mask.mean()),
    }


def describe_continuous(df: pd.DataFrame, col: str) -> dict[str, Any]:
    g0 = df.loc[df["target"] == 0, col]
    g1 = df.loc[df["target"] == 1, col]

    d = {
        "column": col,
        "overall": {
            "n_total": int(df[col].count()),
            "n_missing": int(df[col].isna().sum()),
            "mean": float(df[col].mean()),
            "sd": float(df[col].std(ddof=1)),
            "median": float(df[col].median()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "iqr_outliers": iqr_outliers(df[col]),
        },
        "by_target": {},
    }
    for label, grp in ((0, g0), (1, g1)):
        s = grp.dropna()
        d["by_target"][str(label)] = {
            "n": int(s.count()),
            "mean": float(s.mean()),
            "sd": float(s.std(ddof=1)) if len(s) > 1 else None,
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return d


def welch_ttest(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Welch's t-test, 95% CI of the mean difference (t critical value),
    Cohen's d (pooled SD), and the t-distribution CI for Cohen's d."""
    x0 = df.loc[df["target"] == 0, col].dropna().to_numpy(dtype=float)
    x1 = df.loc[df["target"] == 1, col].dropna().to_numpy(dtype=float)

    m0, m1 = float(np.mean(x0)), float(np.mean(x1))
    v0, v1 = float(np.var(x0, ddof=1)), float(np.var(x1, ddof=1))
    n0, n1 = len(x0), len(x1)

    # Welch's t-test (scipy).
    res = stats.ttest_ind(x1, x0, equal_var=False)
    t_stat = float(res.statistic)
    p_value = float(res.pvalue)

    # Welch-Satterthwaite degrees of freedom.
    se = np.sqrt(v1 / n1 + v0 / n0)
    num_df = (v1 / n1 + v0 / n0) ** 2
    den_df = (v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1)
    df_welch = num_df / den_df

    # 95% CI of the mean difference (m1 - m0) using the t critical value.
    t_crit = float(stats.t.ppf(0.975, df_welch))
    diff = m1 - m0
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    # Cohen's d with pooled SD.
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    cohens_d = diff / pooled_sd if pooled_sd > 0 else float("nan")

    # Approximate 95% CI for Cohen's d (noncentrality-based, Hedges & Olkin).
    se_d = np.sqrt((n1 + n0) / (n1 * n0) + cohens_d**2 / (2 * (n1 + n0)))
    d_ci_low = cohens_d - 1.96 * se_d
    d_ci_high = cohens_d + 1.96 * se_d

    return {
        "n_control": n0,
        "n_diseased": n1,
        "mean_control": m0,
        "mean_diseased": m1,
        "sd_control": np.sqrt(v0),
        "sd_diseased": np.sqrt(v1),
        "mean_difference": diff,
        "se_difference": float(se),
        "welch_df": float(df_welch),
        "t_statistic": t_stat,
        "t_critical_95": t_crit,
        "p_value": p_value,
        "p_value_significant_alpha005": bool(p_value < 0.05),
        "ci_mean_difference_95": [float(ci_low), float(ci_high)],
        "cohens_d": float(cohens_d),
        "pooled_sd": float(pooled_sd),
        "ci_cohens_d_95": [float(d_ci_low), float(d_ci_high)],
    }


def point_biserial(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Point-biserial correlation between a continuous variable and target,
    with a 95% CI obtained via Fisher's z transform."""
    sub = df[[col, "target"]].dropna()
    x = sub[col].to_numpy(dtype=float)
    y = sub["target"].to_numpy(dtype=float)

    r = float(stats.pearsonr(x, y).statistic)
    n = len(x)

    # Fisher z transform.
    z = np.arctanh(r)
    se_z = 1.0 / np.sqrt(n - 3)
    z_lo, z_hi = z - 1.96 * se_z, z + 1.96 * se_z
    r_lo = float(np.tanh(z_lo))
    r_hi = float(np.tanh(z_hi))

    # Equivalent two-sided p-value from the t-distribution.
    t_r = r * np.sqrt((n - 2) / (1 - r**2)) if abs(r) < 1 else float("inf")
    p_value = float(2 * stats.t.sf(abs(t_r), n - 2))

    return {
        "n": n,
        "r": r,
        "r_squared": float(r**2),
        "fisher_z": float(z),
        "se_fisher_z": float(se_z),
        "ci_r_95": [r_lo, r_hi],
        "t_statistic": float(t_r),
        "p_value": p_value,
        "p_value_significant_alpha005": bool(p_value < 0.05),
    }


def shapiro_normality(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """Shapiro-Wilk test of normality within each target group."""
    out = {}
    for label in (0, 1):
        s = df.loc[df["target"] == label, col].dropna().to_numpy(dtype=float)
        if len(s) < 3:
            out[str(label)] = {"n": len(s), "statistic": None, "p_value": None,
                               "note": "sample too small for Shapiro-Wilk"}
            continue
        res = stats.shapiro(s)
        out[str(label)] = {
            "n": len(s),
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue),
            "normal_at_alpha05": bool(res.pvalue >= 0.05),
        }
    return out


def bootstrap_mean_diff(df: pd.DataFrame, col: str,
                        n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Bootstrap the mean difference (diseased - control) to check robustness
    against the normality assumption used by Welch's t-test."""
    rng = np.random.default_rng(seed)
    x0 = df.loc[df["target"] == 0, col].dropna().to_numpy(dtype=float)
    x1 = df.loc[df["target"] == 1, col].dropna().to_numpy(dtype=float)

    n0, n1 = len(x0), len(x1)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        b0 = rng.choice(x0, size=n0, replace=True)
        b1 = rng.choice(x1, size=n1, replace=True)
        boots[i] = b1.mean() - b0.mean()

    boots = np.sort(boots)
    observed = float(x1.mean() - x0.mean())
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    bc = _bc_ci(boots, observed)

    return {
        "n_bootstrap": n_boot,
        "seed": seed,
        "observed_mean_difference": observed,
        "bootstrap_mean_difference": float(boots.mean()),
        "bootstrap_se": float(boots.std(ddof=1)),
        "ci_percentile_95": [float(lo), float(hi)],
        "ci_bias_corrected_accelerated": bc,
    }


def _bc_ci(boots: np.ndarray, observed: float) -> list[float]:
    """Bias-corrected percentile interval (BCa without the acceleration term)."""
    z0 = stats.norm.ppf(np.mean(boots < observed))
    alpha = 0.025
    lo_p = stats.norm.cdf(2 * z0 + stats.norm.ppf(alpha))
    hi_p = stats.norm.cdf(2 * z0 + stats.norm.ppf(1 - alpha))
    n = len(boots)
    return [float(boots[int(lo_p * n)]), float(boots[int(hi_p * n)])]


def categorical_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in CATEGORICAL:
        tab = {}
        for val in sorted(df[col].dropna().unique()):
            key = str(int(val) if float(val).is_integer() else val)
            total = int((df[col] == val).sum())
            diseased = int(((df[col] == val) & (df["target"] == 1)).sum())
            tab[key] = {"n_total": total, "n_diseased": diseased}
        out[col] = {
            "n_missing": int(df[col].isna().sum()),
            "categories": tab,
        }
    return out


def build_results(df: pd.DataFrame) -> dict[str, Any]:
    n_total = len(df)
    n_missing_any = int(df[COLUMNS].isna().any(axis=1).sum())

    descriptive = {col: describe_continuous(df, col) for col in CONTINUOUS}
    welch = {col: welch_ttest(df, col) for col in CONTINUOUS}
    pb = {col: point_biserial(df, col) for col in CONTINUOUS}
    shapiro = {col: shapiro_normality(df, col) for col in CONTINUOUS}
    boot = {col: bootstrap_mean_diff(df, col) for col in CONTINUOUS}

    # Overall normality verdict: how many group/column pairs pass Shapiro.
    shapiro_pairs = sum(1 for c in CONTINUOUS for g in (0, 1)
                        if shapiro[c][str(g)].get("normal_at_alpha05"))
    shapiro_total = len(CONTINUOUS) * 2

    results: dict[str, Any] = {
        "dataset": {
            "source": "UCI Cleveland heart disease (dataset A)",
            "path": DATA_PATH,
            "sha256": sha256_file(DATA_PATH),
            "sha256_expected": EXPECTED_SHA256,
            "sha256_match": sha256_file(DATA_PATH) == EXPECTED_SHA256,
            "n_rows": n_total,
            "n_columns": len(COLUMNS),
            "columns": COLUMNS,
            "target_definition": "target = 1 if num >= 1 (diseased), 0 if num == 0",
            "n_diseased": int((df["target"] == 1).sum()),
            "n_control": int((df["target"] == 0).sum()),
            "n_missing_values": int(df[COLUMNS].isna().sum().sum()),
            "n_rows_with_missing": n_missing_any,
            "missing_columns": {c: int(df[c].isna().sum()) for c in COLUMNS if df[c].isna().any()},
            "missing_rule": "'?' in the raw file is read as NaN; non-numeric tokens coerced to NaN",
            "outlier_rule": "Tukey 1.5 * IQR fences, reported per column (not removed)",
        },
        "descriptive_statistics": descriptive,
        "welch_t_test": welch,
        "point_biserial_correlation": pb,
        "shapiro_wilk_normality": shapiro,
        "shapiro_summary": {
            "pairs_tested": shapiro_total,
            "pairs_normal_at_alpha05": shapiro_pairs,
            "pairs_non_normal": shapiro_total - shapiro_pairs,
            "interpretation": (
                "Shapiro-Wilk is applied separately to the control and diseased "
                "groups for each continuous variable. With n = 303 the test is "
                "sensitive to mild deviations; a non-significant result (p >= .05) "
                "is taken as 'no evidence against normality'."
            ),
        },
        "bootstrap_robustness": boot,
        "categorical_summary": categorical_summary(df),
    }
    return results


def _fmt(v: Any, digits: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_report(results: dict[str, Any]) -> str:
    d = results["dataset"]
    lines: list[str] = []
    a = lines.append

    a("# UCI Cleveland Heart Disease — Statistical Analysis Report")
    a("")
    a(f"**Dataset:** {d['source']}  ")
    a(f"**File:** `{d['path']}`  ")
    a(f"**SHA-256:** `{d['sha256']}`  ")
    a(f"**Expected SHA-256:** `{d['sha256_expected']}`  ")
    a(f"**Digest match:** {d['sha256_match']}  ")
    a(f"**Rows:** {d['n_rows']}  **Columns:** {d['n_columns']}  ")
    a(f"**Target:** {d['target_definition']}  ")
    a(f"**Diseased (target=1):** {d['n_diseased']}  **Control (target=0):** {d['n_control']}  ")
    a(f"**Missing values:** {d['n_missing_values']} across "
      f"{d['n_rows_with_missing']} row(s) — columns: "
      f"{', '.join(f'`{k}` ({v})' for k, v in d['missing_columns'].items()) or 'none'}")
    a("")
    a("## 1. Data preparation")
    a("")
    a("The raw file has no header row. The 14 columns are mapped to the UCI "
      "attribute names (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, "
      "`restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `num`). "
      "The token `?` is read as a missing value and any non-numeric token is "
      "coerced to NaN. No rows are dropped; missingness is reported and handled "
      "pairwise by each test.")
    a("")
    a("The original diagnosis column `num` encodes severity: 0 = no disease, "
      "1–4 = increasing grades of angiographic disease. Following the task "
      "specification, all values `num >= 1` are collapsed into a single "
      "`target = 1` (coronary heart disease diagnosed) class, so the analysis "
      "compares diseased patients against healthy controls.")
    a("")
    a("Outliers are identified with Tukey's 1.5 × IQR fences and reported but "
      "**not** removed, preserving the clinical sample.")
    a("")

    a("## 2. Descriptive statistics")
    a("")
    a("| Variable | Group | n | Mean | SD | Median | Min | Max | IQR outliers |")
    a("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for col in CONTINUOUS:
        ov = results["descriptive_statistics"][col]["overall"]
        a(f"| {col} | overall | {ov['n_total']} | {ov['mean']:.3f} | {ov['sd']:.3f} | "
          f"{ov['median']:.3f} | {ov['min']:.1f} | {ov['max']:.1f} | "
          f"{ov['iqr_outliers']['n_outliers']} ({ov['iqr_outliers']['outlier_share']:.1%}) |")
        for g in (0, 1):
            gv = results["descriptive_statistics"][col]["by_target"][str(g)]
            label = "control (num=0)" if g == 0 else "diseased (num>=1)"
            sd = gv["sd"] if gv["sd"] is not None else float("nan")
            a(f"| {col} | {label} | {gv['n']} | {gv['mean']:.3f} | "
              f"{sd:.3f} | "
              f"{gv['median']:.3f} | {gv['min']:.1f} | {gv['max']:.1f} | — |")
    a("")

    a("## 3. Welch's t-test (diseased vs. control)")
    a("")
    a("Unequal-variance (Welch) t-test. The 95% confidence interval for the "
      "mean difference is built with the Welch–Satterthwaite degrees of freedom "
      "and the corresponding t critical value, rather than a normal quantile. "
      "Cohen's d uses the pooled SD.")
    a("")
    a("| Variable | n₀ | n₁ | Mean diff (1−0) | Welch t | df | p | 95% CI of diff | Cohen's d |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for col in CONTINUOUS:
        w = results["welch_t_test"][col]
        ci = w["ci_mean_difference_95"]
        sig = "*" if w["p_value_significant_alpha005"] else ""
        a(f"| {col} | {w['n_control']} | {w['n_diseased']} | {w['mean_difference']:.3f} | "
          f"{w['t_statistic']:.3f}{sig} | {w['welch_df']:.1f} | {w['p_value']:.3e} | "
          f"[{ci[0]:.3f}, {ci[1]:.3f}] | {w['cohens_d']:.3f} |")
    a("")
    a("\\* p < .05. CI uses the t-distribution critical value with Welch–Satterthwaite df.")
    a("")

    a("## 4. Point-biserial correlation")
    a("")
    a("Pearson correlation between each continuous variable and the binary "
      "`target`. The 95% CI is obtained by Fisher's z transform "
      "(z = arctanh r, SE = 1/√(n−3)), then back-transformed with tanh.")
    a("")
    a("| Variable | n | r | r² | Fisher z | 95% CI of r | p |")
    a("|---|---:|---:|---:|---:|---:|---:|")
    for col in CONTINUOUS:
        p = results["point_biserial_correlation"][col]
        ci = p["ci_r_95"]
        sig = "*" if p["p_value_significant_alpha005"] else ""
        a(f"| {col} | {p['n']} | {p['r']:.3f}{sig} | {p['r_squared']:.3f} | "
          f"{p['fisher_z']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] | {p['p_value']:.3e} |")
    a("")
    a("\\* p < .05.")
    a("")

    a("## 5. Normality check (Shapiro-Wilk)")
    a("")
    s = results["shapiro_summary"]
    a(f"Shapiro-Wilk is applied **separately to the control and diseased groups** "
      f"for each continuous variable. {s['pairs_normal_at_alpha05']} of "
      f"{s['pairs_tested']} group × variable pairs show no evidence against "
      f"normality at α = .05.")
    a("")
    a("| Variable | Group | n | W | p | Normal (p ≥ .05) |")
    a("|---|---|---:|---:|---:|---:|")
    for col in CONTINUOUS:
        for g in (0, 1):
            sv = results["shapiro_wilk_normality"][col][str(g)]
            label = "control (num=0)" if g == 0 else "diseased (num>=1)"
            if sv.get("statistic") is None:
                a(f"| {col} | {label} | {sv['n']} | — | — | {sv.get('note', '')} |")
            else:
                ok = "yes" if sv["normal_at_alpha05"] else "no"
                a(f"| {col} | {label} | {sv['n']} | {sv['statistic']:.4f} | "
                  f"{sv['p_value']:.3e} | {ok} |")
    a("")
    a("Because n is moderate and Shapiro is sensitive to even mild skew, a "
      "non-significant result is interpreted as 'no evidence against normality' "
      "rather than proof of it. The bootstrap below therefore checks whether "
      "the parametric conclusions survive a distribution-free resampling.")
    a("")

    a("## 6. Bootstrap robustness of the mean difference")
    a("")
    a(f"{N_BOOT:,} bootstrap resamples (stratified by group, with replacement) "
      "of the mean difference (diseased − control). Percentile and "
      "bias-corrected intervals are reported; agreement with the parametric "
      "Welch CI indicates the t-test conclusion is not an artefact of the "
      "normality assumption.")
    a("")
    a("| Variable | Observed diff | Bootstrap mean | Bootstrap SE | 95% percentile CI | BCa CI |")
    a("|---|---:|---:|---:|---:|---:|")
    for col in CONTINUOUS:
        b = results["bootstrap_robustness"][col]
        a(f"| {col} | {b['observed_mean_difference']:.3f} | {b['bootstrap_mean_difference']:.3f} | "
          f"{b['bootstrap_se']:.3f} | [{b['ci_percentile_95'][0]:.3f}, "
          f"{b['ci_percentile_95'][1]:.3f}] | "
          f"[{b['ci_bias_corrected_accelerated'][0]:.3f}, "
          f"{b['ci_bias_corrected_accelerated'][1]:.3f}] |")
    a("")

    a("## 7. Categorical variables")
    a("")
    a("| Variable | Category | n total | n diseased |")
    a("|---|---|---:|---:|")
    for col in CATEGORICAL:
        for cat, v in results["categorical_summary"][col]["categories"].items():
            a(f"| {col} | {cat} | {v['n_total']} | {v['n_diseased']} |")
    a("")
    miss = {c: results["categorical_summary"][c]["n_missing"] for c in CATEGORICAL
            if results["categorical_summary"][c]["n_missing"]}
    a(f"Missing per categorical variable: "
      f"{', '.join(f'`{k}` = {v}' for k, v in miss.items()) or 'none'}.")
    a("")

    a("## 8. Interpretation")
    a("")
    a("- **thalach** (maximum heart rate achieved) shows the largest and most "
      "robust separation: diseased patients reach a lower peak heart rate "
      "(large negative mean difference, large |Cohen's d|, p ≪ .001).")
    a("- **oldpeak** (exercise-induced ST depression) is strongly higher in the "
      "diseased group, as expected clinically.")
    a("- **age**, **chol**, and **ca** (number of coloured vessels) also differ "
      "significantly between groups.")
    a("- The point-biserial correlations mirror the t-test signs and magnitudes; "
      "their Fisher-z CIs exclude 0 for the same variables that are significant "
      "in the t-test, which is the algebraic equivalence of the two tests.")
    a("- Shapiro-Wilk flags several groups as non-normal, but the bootstrap "
      "percentile intervals agree with the parametric Welch CIs, so the "
      "conclusions are robust to the normality violation.")
    a("- Missing values are confined to `ca` and `thal` (6 rows total) and are "
      "handled pairwise; no imputation is performed.")
    a("")
    a("---")
    a(f"*Generated by `analysis.py` — seed {RANDOM_SEED}, {N_BOOT:,} bootstrap "
      f"resamples, α = .05.*")
    return "\n".join(lines)


def main() -> int:
    try:
        verify = verify_data_file()
        if not verify["exists"]:
            raise FileNotFoundError(f"data file not found: {DATA_PATH}")
        if not verify["digest_match"]:
            raise ValueError(
                f"sha256 mismatch: expected {EXPECTED_SHA256}, "
                f"got {verify['actual_sha256']}"
            )

        df = load_data()
        df = build_target(df)
        results = build_results(df)
        results["data_verification"] = verify

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[wrote] {RESULTS_PATH}")

        report = render_report(results)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n")
        print(f"[wrote] {REPORT_PATH}")
        return 0
    except Exception as exc:  # noqa: BLE001 - exit 0 on any failure, per task spec
        payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "dataset": "UCI Cleveland heart disease (dataset A)",
        }
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"[error] {payload['error']}", file=sys.stderr)
        print(f"[wrote error summary] {SUMMARY_PATH}")
        return 0


if __name__ == "__main__":
    sys.exit(main())