"""Patch clean.py: impute within all 5 target strata, not just 0/1."""
import io

P = "data/clean.py"
src = io.open(P, encoding="utf-8").read()

old = '''        group_medians = {}
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
        }'''

new = '''        strata = sorted(df_clean[TARGET_COL].unique())
        group_medians = {
            str(int(s)): float(df_clean.loc[df_clean[TARGET_COL] == s, col]
                               .median())
            for s in strata
        }
        global_median = float(df_clean[col].median())

        for s in strata:
            mask = (df_clean[TARGET_COL] == s) & df_clean[col].isna()
            fill = group_medians[str(int(s))]
            if int(mask.sum()):
                df_clean.loc[mask, col] = fill

        imputation_log[col] = {
            "method": "median imputation within target stratum (0..4)",
            "global_median": global_median,
            "stratum_medians": group_medians,
            "n_imputed": info["n_missing"],
        }'''

assert old in src, "anchor not found"
src = src.replace(old, new, 1)

cmt = '''    # Strategy: median imputation within target strata. The target variable is
    # ordinal (0 = no disease, 1-4 = increasing severity), and the distribution
    # of e.g. thalach differs strongly by disease status, so a global median
    # would bias each group toward the other. We therefore impute within each
    # of the five original strata (0..4); for the binary contrast used later
    # (0 vs >=1) this is equivalent to imputing within the binary group, while
    # also covering rows whose severity is 2-4 (which a 0/1-only loop would
    # miss -- this is what left one thal NaN un-imputed in an earlier version).'''

old_cmt = '''    # Strategy: median imputation within target strata (0 vs >=1). Rationale:
    # the distribution of e.g. thalach differs strongly by disease status, so a
    # global median would bias each group toward the other. Using the group
    # median preserves the between-group contrast that the analysis tests.
    # For ca/thal the missingness is sparse (4 and 2 rows), so the choice has
    # negligible influence; it is logged regardless.'''
assert old_cmt in src, "comment anchor not found"
src = src.replace(old_cmt, cmt, 1)

io.open(P, "w", encoding="utf-8").write(src)
print("patched")