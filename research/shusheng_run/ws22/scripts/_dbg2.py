"""Reproduce the imputation step in isolation."""
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv("data/processed.cleveland.data", header=None, names=COLS,
                 na_values=["?"])
TARGET_COL = "target"
df_clean = df.copy()
col = "ca"
group_medians = {}
for g, label in ((0, "target==0"), (1, "target>=1")):
    group_medians[label] = float(df_clean.loc[df_clean[TARGET_COL] == g, col]
                                 .median())
print("medians", group_medians)
for g in (0, 1):
    mask = (df_clean[TARGET_COL] == g) & df_clean[col].isna()
    fill = group_medians["target==0"] if g == 0 else group_medians["target>=1"]
    print("g", g, "mask sum", int(mask.sum()), "fill", fill)
    df_clean.loc[mask, col] = fill
print("remaining NaN", int(df_clean[col].isna().sum()))
print(df_clean.ca.value_counts(dropna=False).sort_index())