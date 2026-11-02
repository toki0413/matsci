import pandas as pd

C = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "rest_ecg", "max_hr", "exercise_angina",
    "st_depression", "st_slope", "num",
]
df = pd.read_csv("data/heart.csv", header=None, names=C, na_values="?")
print(df.dtypes)
print("--- missing ---")
print(df.isna().sum())
print("--- target counts ---")
print(df["num"].value_counts(dropna=False).sort_index())
d2 = pd.read_csv("data/heart.csv", header=None, names=C)
print("--- ? counts raw ---")
print((d2 == "?").sum())