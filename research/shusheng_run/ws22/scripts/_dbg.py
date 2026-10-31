"""Debug imputation: are group medians NaN?"""
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = pd.read_csv("data/processed.cleveland.data", header=None, names=COLS,
                 na_values=["?"])
for c in ["ca", "thal"]:
    print(c, "missing", int(df[c].isna().sum()))
    for g in [0, 1]:
        sub = df.loc[df.target == g, c]
        print("  group", g, "n", len(sub), "median", sub.median())
print(df.target.value_counts().sort_index())
print("ca rows:", df[df.ca.isna()][["target", "ca"]].to_dict("records"))
print("thal rows:", df[df.thal.isna()][["target", "thal"]].to_dict("records"))