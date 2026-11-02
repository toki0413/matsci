"""Reproduce clean.py main() body to find the NaN leak."""
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]
MISSING_TOKEN_COLS = COLS[:-1]
TARGET_COL = "target"
df = pd.read_csv("data/processed.cleveland.data", header=None, names=COLS,
                 na_values=["?"])
missing_report = {}
for c in MISSING_TOKEN_COLS:
    n_missing = int(df[c].isna().sum())
    if n_missing:
        missing_report[c] = {"n_missing": n_missing}
print("missing_report keys:", list(missing_report))
df_clean = df.copy()
for col, info in missing_report.items():
    group_medians = {}
    for g, label in ((0, "target==0"), (1, "target>=1")):
        group_medians[label] = float(
            df_clean.loc[df_clean[TARGET_COL] == g, col].median())
    for g in (0, 1):
        mask = (df_clean[TARGET_COL] == g) & df_clean[col].isna()
        fill = group_medians["target==0"] if g == 0 else group_medians[
            "target>=1"]
        df_clean.loc[mask, col] = fill
print("after imputation NaN:", int(df_clean.isna().sum().sum()))
for c in ["ca", "thal"]:
    print(c, "NaN", int(df_clean[c].isna().sum()))