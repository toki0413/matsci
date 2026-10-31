"""Inspect the raw Cleveland dataset."""
import pandas as pd

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "target"]

df = pd.read_csv("data/processed.cleveland.data", header=None, names=COLS,
                 na_values=["?"])
print("shape:", df.shape)
print(df.dtypes)
print("\nmissing per col:"); print(df.isna().sum())
print("\ntarget counts:"); print(df["target"].value_counts().sort_index())
print("\ndescribe:"); print(df.describe().T)
print("\nrows with any NaN:"); print(df[df.isna().any(axis=1)])