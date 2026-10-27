"""Clean the UCI Cleveland heart disease dataset into data/clean_analysis.csv.

Missing-value handling
----------------------
The raw UCI file encodes missing values as the literal ``?`` in columns 1-13
(the 14th column, ``target``, is never missing). This script maps ``?`` -> NaN,
documents every imputation decision, and writes a sidecar JSON log
``data/cleaning_log.json`` so the cleaning is fully auditable.

Outlier handling
----------------
Continuous clinical variables are winsorised at the 1st/99th percentile
(computed on the non-missing values) rather than deleted: the Cleveland cohort
is small (n=303) and row deletion would discard ~10% of the sample. Winsorising
keeps every observation while capping implausible extremes. Categorical columns
(cp, restecg, slope, ca, thal, fbs, exang, sex, target) are left untouched.

The output is a single tidy CSV ready for analysis.py.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE, "data")
RAW_PATH = os.path.join(DATA_DIR, "processed.cleveland.data")
OUT_CSV = os.path.join(DATA_DIR, "clean_analysis.csv")
LOG_PATH = os.path.join(DATA_DIR, "cleaning_log.json")

# Column names from the UCI repository documentation.
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]

# Columns 1-13 (0-based 0..12) may contain "?"; target (index 13) never does.
MISSING_TOKEN_COLS = COLS[:-1]
TARGET_COL = "target"

# Continuous variables subject to winsorisation. Chosen because they are
# measured on a ratio scale and have clinically implausible extremes.
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

WINSOR_LO, WINSOR_HI = 0.01, 0.99


def main():
    if not os.path.exists(RAW_PATH):
        print(f"[error] raw data not found at {RAW_PATH}; run data/download.py "
              "first")
        return 1

    # ---- 1. Load, mapping "?" to NaN in columns 1-13 -----------------------
    df = pd.read_csv(RAW_PATH, header=None, names=COLS, na_values=["?"])
    n_rows_raw, n_cols_raw = df.shape

    missing_report = {}
    for c in MISSING_TOKEN_COLS:
        n_missing = int(df[c].isna().sum())
        if n_missing:
            missing_report[c] = {
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / len(df), 2),
                "missing_row_indices": df.index[df[c].isna()].tolist(),
            }

    # ---- 2. Imputation -----------------------------------------------------
    # Strategy: median imputation within target strata (0 vs >=1). Rationale:
    # the distribution of e.g. thalach differs strongly by disease status, so a
    # global median would bias each group toward the other. Using the group
    # median preserves the between-group contrast that the analysis tests.
    # For ca/thal the missingness is sparse (4 and 2 rows), so the choice has
    # negligible influence; it is logged regardless.
    imputation_log = {}
    df_clean = df.copy()
    for col, info in missing_report.items():
        group_medians = {}
        for g, label in ((0, "target==0"), (1, "target>=1")):
            group_medians[label] = float(
                df_clean.loc[df_clean[TARGET_COL] == g, col].median()
        )
        global_median = float(df_clean[col].median())

        for g in (0, 1):
            mask = (df_clean[TARGET_COL] == g) & df_clean[col].isna()
            fill = group_medians["target==0"] if g == 0 else group_medians[
                "target>=1"
            ]
            df_clean.loc[mask, col] = fill

        imputation_log[col] = {
            "method": "median imputation within target stratum (0 vs >=1)",
            "global_median": global_median,
            "group_medians": group_medians,
            "n_imputed": info["n_missing"],
        }

    # ---- 3. Outlier handling (winsorise) -----------------------------------
    outlier_log = {}
    for col in CONTINUOUS:
        lo = df_clean[col].quantile(WINSOR_LO)
        hi = df_clean[col].quantile(WINSOR_HI)
        n_below = int((df_clean[col] < lo).sum())
        n_above = int((df_clean[col] > hi).sum())
        before = (float(df_clean[col].min()), float(df_clean[col].max()))
        df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)
        after = (float(df_clean[col].min()), float(df_clean[col].max()))
        outlier_log[col] = {
            "method": "winsorise at 1st/99th percentile",
            "lower_bound": float(lo),
            "upper_bound": float(hi),
            "n_clipped_low": n_below,
            "n_clipped_high": n_above,
            "range_before": before,
            "range_after": after,
        }

    # ---- 4. Type / semantic fixes -----------------------------------------
    # target: 0 = no disease, 1-4 = disease severity. Collapse to binary
    # disease/no-disease for the Welch t-test (group 0 vs group >=1).
    df_clean["target_binary"] = (df_clean[TARGET_COL] >= 1).astype(int)
    # Keep the original ordinal target for reference.
    df_clean["target"] = df_clean[TARGET_COL].astype(int)

    # Categorical columns: ensure integer dtype (they arrived as float because
    # of NaN imputation above).
    for c in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
        df_clean[c] = df_clean[c].astype(int)

    df_clean = df_clean.drop(columns=[TARGET_COL])

    # ---- 5. Write ----------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    df_clean.to_csv(OUT_CSV, index=False)

    log = {
        "source": RAW_PATH,
        "raw_shape": [n_rows_raw, n_cols_raw],
        "missing_token": "?",
        "missing_token_columns": MISSING_TOKEN_COLS,
        "missing_report": missing_report,
        "imputation_log": imputation_log,
        "outlier_log": outlier_log,
        "final_shape": list(df_clean.shape),
        "final_columns": list(df_clean.columns),
        "output_csv": OUT_CSV,
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[wrote] {OUT_CSV}  shape={df_clean.shape}")
    print(f"[wrote] {LOG_PATH}")
    print(f"missing cells: {int(df_clean.isna().sum().sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())