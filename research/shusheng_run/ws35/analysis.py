#!/usr/bin/env python3
"""
Comprehensive hypothesis testing for MgH2 activation energy dataset.

Defines 5 explicit, testable hypotheses H1-H5 with thresholds and statistical criteria,
validates data structure/units, performs all five analyses with proper uncertainty
quantification, and outputs a report classifying each hypothesis as "supported" or "refuted".

Hypotheses:
  H1: Within-paper paired ΔEa shows systematic method-dependent differences.
      Test: DerSimonian-LairD random-effects meta-analysis of within-paper paired differences.
      Threshold: Random-effects mean difference ≠ 0 (95% CI excludes 0), p < 0.05.

  H2: Catalyzed samples have significantly lower Ea than pure MgH2.
      Test: Welch's two-sample t-test (catalyzed vs pure_MgH2, desorption only).
      Threshold: Mean Ea(catalyzed) - Mean Ea(pure) ≤ -20 kJ/mol, p < 0.05.

  H3: Method choice (Kissinger vs isothermal_Arrhenius) significantly affects Ea
      for the same material within the same study.
      Test: Paired t-test on within-study, within-material method pairs.
      Threshold: Mean |ΔEa| ≥ 15 kJ/mol, paired p < 0.05.

  H4: Ea prediction using material features achieves reasonable accuracy under LOSO CV.
      Test: Leave-One-Subgroup-Out CV (group by material_class + catalyst_family).
      Threshold: R² ≥ 0.30 and RMSE ≤ 25 kJ/mol.

  H5: OOD prediction interval (95%) for the Nb2C MXene sample covers its reported Ea.
      Test: Leave-one-sample-out prediction interval using bootstrap residuals.
      Threshold: 95% PI covers the reported Ea (~111.49 kJ/mol).
"""

import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
DATA_DIR = '/workspace/research/shusheng_run/ws35/data'
RESULTS_DIR = '/workspace/research/shusheng_run/ws35/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 1. DATA VALIDATION
# ============================================================
def validate_data():
    """Validate data structure, units, and integrity."""
    print("=" * 70)
    print("DATA VALIDATION")
    print("=" * 70)

    # Load datasets
    cleaned = pd.read_csv(f"{DATA_DIR}/cleaned_ea_dataset.csv")
    original = pd.read_csv(f"{DATA_DIR}/ea_dataset.csv")

    validation_results = {
        "shape": {},
        "columns": {},
        "units": {},
        "missing_values": {},
        "duplicates": {},
        "unit_consistency": {},
        "summary": {}
    }

    # Shape
    validation_results["shape"] = {
        "original_rows": len(original),
        "original_cols": len(original.columns),
        "cleaned_rows": len(cleaned),
        "cleaned_cols": len(cleaned.columns),
        "rows_removed": len(original) - len(cleaned)
    }
    print(f"\nOriginal dataset: {len(original)} rows, {len(original.columns)} columns")
    print(f"Cleaned dataset: {len(cleaned)} rows, {len(cleaned.columns)} columns")
    print(f"Rows removed: {len(original) - len(cleaned)}")

    # Columns
    expected_cols = ['material', 'material_class', 'catalyst_or_additive',
                     'Ea_kJ_per_mol', 'direction', 'T_des_C', 'method',
                     'delta_H_kJ_per_mol_H2', 'doi', 'year', 'locator',
                     'quote', 'url', 'notes', 'verify_status', 'verify_source',
                     'catalyst_family', 'Ea_parsed']
    validation_results["columns"] = {
        "expected": expected_cols,
        "present": list(cleaned.columns),
        "missing": [c for c in expected_cols if c not in cleaned.columns]
    }
    print(f"\nColumns present: {list(cleaned.columns)}")

    # Units validation for Ea
    # Ea should be in kJ/mol, positive values
    ea_valid = cleaned['Ea_parsed'].dropna()
    validation_results["units"]["Ea_kJ_per_mol"] = {
        "min": float(ea_valid.min()),
        "max": float(ea_valid.max()),
        "mean": float(ea_valid.mean()),
        "median": float(ea_valid.median()),
        "std": float(ea_valid.std()),
        "all_positive": bool((ea_valid > 0).all()),
        "unit": "kJ/mol"
    }
    print(f"\nEa (kJ/mol): min={ea_valid.min():.1f}, max={ea_valid.max():.1f}, "
          f"mean={ea_valid.mean():.1f}, median={ea_valid.median():.1f}")

    # Direction validation
    valid_directions = {'desorption', 'absorption'}
    validation_results["units"]["direction"] = {
        "unique": list(cleaned['direction'].unique()),
        "counts": cleaned['direction'].value_counts().to_dict()
    }
    print(f"Direction counts: {cleaned['direction'].value_counts().to_dict()}")

    # Missing values
    validation_results["missing_values"] = {
        "Ea_parsed": int(cleaned['Ea_parsed'].isna().sum()),
        "T_des_C": int(cleaned['T_des_C'].isna().sum()),
        "delta_H_kJ_per_mol_H2": int(cleaned['delta_H_kJ_per_mol_H2'].isna().sum()),
        "method": int(cleaned['method'].isna().sum())
    }
    print(f"\nMissing values: {validation_results['missing_values']}")

    # Duplicates by doi
    doi_duplicates = cleaned.duplicated(subset='doi', keep=False)
    validation_results["duplicates"] = {
        "total_doi_duplicates": int(doi_duplicates.sum()),
        "unique_doins": cleaned['doi'].nunique()
    }
    print(f"\nDOI duplicates: {doi_duplicates.sum()} rows share DOIs, "
          f"{cleaned['doi'].nunique()} unique DOIs")

    # Unit consistency check: Ea_parsed should match Ea_kJ_per_mol (numeric extraction)
    ea_parsed_valid = cleaned['Ea_parsed'].dropna()
    ea_kj_valid = cleaned['Ea_kJ_per_mol'].dropna()
    validation_results["unit_consistency"]["ea_parsed_vs_original"] = {
        "n_compared": min(len(ea_parsed_valid), len(ea_kj_valid)),
        "mean_difference": float(np.mean(ea_parsed_valid - ea_kj_valid)),
        "max_abs_diff": float(np.max(np.abs(ea_parsed_valid - ea_kj_valid)))
    }
    print(f"\nEa_parsed vs original Ea: mean diff={validation_results['unit_consistency']['ea_parsed_vs_original']['mean_difference']:.4f}, "
          f"max diff={validation_results['unit_consistency']['ea_parsed_vs_original']['max_abs_diff']:.4f}")

    # Summary of validation
    validation_results["summary"] = {
        "all_positive_ea": bool((ea_valid > 0).all()),
        "valid_directions": bool(set(cleaned['direction'].unique()).issubset(valid_directions)),
        "no_missing_ea": int(cleaned['Ea_parsed'].isna().sum()) == 0,
        "method_categories": list(cleaned['method'].unique()),
        "material_classes": list(cleaned['material_class'].unique()),
        "catalyst_families": list(cleaned['catalyst_family'].unique())
    }
    print(f"\nValidation summary: {validation_results['summary']}")

    # Save validation results
    validation_path = f"{RESULTS_DIR}/validation_results.json"
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    print(f"\nValidation results saved to {validation_path}")

    return cleaned, validation_results


# ============================================================
# H1: Within-paper paired ΔEa with DerSimonian-Laird random-effects meta-analysis
# ============================================================
def test_h1_within_paper_method_difference(cleaned):
    """
    H1: Within-paper paired ΔEa shows systematic method-dependent differences.
    
    For papers that report Ea for the same material using different methods,
    we compute the paired differences and apply DerSimonian-Laird random-effects
    meta-analysis to estimate the overall mean difference.
    
    Threshold: Random-effects mean difference ≠ 0 (95% CI excludes 0), p < 0.05.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS H1: Within-paper paired ΔEa (DerSimonian-Laird meta-analysis)")
    print("=" * 70)

    # Find rows with same doi and same material but different methods
    # We need to identify pairs within the same paper (same doi) that have
    # the same material (or very similar material names) but different methods
    # The key example is MgH2-10 wt% H-V2O5 in doi 10.1007/s40820-024-01375-8
    # which has both Kissinger (113.8) and isothermal_Arrhenius (84.55)

    # Group by doi and material, then find pairs with different methods
    grouped = cleaned.groupby(['doi', 'material'])
    pairs = []

    for (doi, material), group in grouped:
        if len(group) > 1:
            # Check if there are different methods within the same material
            methods = group['method'].unique()
            if len(methods) > 1:
                # Create all pairs of different methods
                for i, row1 in group.iterrows():
                    for j, row2 in group.iterrows():
                        if i < j and row1['method'] != row2['method']:
                            diff = row1['Ea_parsed'] - row2['Ea_parsed']
                            pairs.append({
                                'doi': doi,
                                'material': material,
                                'method1': row1['method'],
                                'method2': row2['method'],
                                'ea1': row1['Ea_parsed'],
                                'ea2': row2['Ea_parsed'],
                                'difference': diff,
                                'variance': 0  # Will estimate from typical measurement error
                            })

    pairs_df = pd.DataFrame(pairs)
    print(f"\nFound {len(pairs_df)} within-paper, within-material method pairs")
    print(pairs_df[['doi', 'material', 'method1', 'method2', 'ea1', 'ea2', 'difference']].to_string())

    if len(pairs_df) == 0:
        print("ERROR: No within-paper method pairs found. H1 cannot be tested.")
        return {"supported": False, "reason": "No within-paper method pairs found"}

    # DerSimonian-Laird random-effects meta-analysis
    # Step 1: Compute within-study variances (assume typical measurement error)
    # For Ea measurements, typical uncertainty is ~5-10 kJ/mol based on the data
    # We'll use a conservative estimate of 10 kJ²/mol² for each measurement
    # The variance of the difference is Var(ea1) + Var(ea2) = 2 * 100 = 200
    within_study_var = 200.0  # (10 kJ/mol)^2 * 2 for difference

    # Step 2: Compute weighted mean and between-study variance (Q statistic)
    n = len(pairs_df)
    diffs = pairs_df['difference'].values
    # Use inverse variance weights (fixed-effects first)
    weights = np.ones(n) / within_study_var

    # Fixed-effects weighted mean
    fe_mean = np.sum(weights * diffs) / np.sum(weights)
    fe_se = np.sqrt(1.0 / np.sum(weights))

    # Q statistic for heterogeneity
    Q = np.sum(weights * (diffs - fe_mean) ** 2)
    df_Q = n - 1

    # DerSimonian-Laird estimate of between-study variance (tau^2)
    if Q > df_Q:
        tau_sq = (Q - df_Q) / (np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))
        tau_sq = max(tau_sq, 0)  # Ensure non-negative
    else:
        tau_sq = 0.0

    # Random-effects weights
    re_weights = 1.0 / (within_study_var + tau_sq)
    re_mean = np.sum(re_weights * diffs) / np.sum(re_weights)
    re_se = np.sqrt(1.0 / np.sum(re_weights))

    # 95% CI for random-effects mean
    re_ci_lower = re_mean - 1.96 * re_se
    re_ci_upper = re_mean + 1.96 * re_se

    # p-value (two-sided)
    z_stat = re_mean / re_se
    p_value = 2 * (1 - __import__('scipy.stats').norm.cdf(abs(z_stat)))

    # I² statistic (proportion of total variance due to heterogeneity)
    total_var = within_study_var + tau_sq
    I2 = max(0, (tau_sq / total_var) * 100) if total_var > 0 else 0

    print(f"\nDerSimonian-Laird Random-Effects Meta-Analysis:")
    print(f"  Number of studies (pairs): {n}")
    print(f"  Fixed-effects mean difference: {fe_mean:.2f} kJ/mol (SE={fe_se:.2f})")
    print(f"  Random-effects mean difference: {re_mean:.2f} kJ/mol (SE={re_se:.2f})")
    print(f"  95% CI for RE mean: [{re_ci_lower:.2f}, {re_ci_upper:.2f}] kJ/mol")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Between-study variance (τ²): {tau_sq:.2f}")
    print(f"  I² statistic: {I2:.1f}%")

    # Threshold: 95% CI excludes 0 and p < 0.05
    h1_supported = (re_ci_lower > 0 or re_ci_upper < 0) and (p_value < 0.05)
    h1_direction = "positive" if re_mean > 0 else "negative"

    h1_result = {
        "hypothesis": "H1",
        "description": "Within-paper paired ΔEa shows systematic method-dependent differences",
        "test": "DerSimonian-Laird random-effects meta-analysis",
        "n_pairs": int(n),
        "mean_difference_kJ_per_mol": float(re_mean),
        "se_kJ_per_mol": float(re_se),
        "ci_95_lower_kJ_per_mol": float(re_ci_lower),
        "ci_95_upper_kJ_per_mol": float(re_ci_upper),
        "p_value": float(p_value),
        "tau_squared": float(tau_sq),
        "I2_percent": float(I2),
        "threshold": "95% CI excludes 0 and p < 0.05",
        "supported": bool(h1_supported),
        "direction": h1_direction,
        "reasoning": f"Mean difference = {re_mean:.2f} kJ/mol, 95% CI = [{re_ci_lower:.2f}, {re_ci_upper:.2f}], p = {p_value:.4f}. "
                     f"CI {'excludes' if h1_supported else 'includes'} zero, {'supports' if h1_supported else 'does not support'} systematic method-dependent differences."
    }

    # Save H1 results
    h1_path = f"{RESULTS_DIR}/h1_results.json"
    with open(h1_path, 'w', encoding='utf-8') as f:
        json.dump(h1_result, f, indent=2, ensure_ascii=False)
    print(f"\nH1 results saved to {h1_path}")

    return h1_result


# ============================================================
# H2: Catalyst effect on Ea (catalyzed vs pure MgH2)
# ============================================================
def test_h2_catalyst_effect(cleaned):
    """
    H2: Catalyzed samples have significantly lower Ea than pure MgH2.
    
    Test: Welch's two-sample t-test comparing catalyzed_MgH2 vs pure_MgH2
    (desorption only, to avoid absorption/desorption confounding).
    
    Threshold: Mean Ea(catalyzed) - Mean Ea(pure) ≤ -20 kJ/mol, p < 0.05.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS H2: Catalyst effect on Ea (Welch's t-test)")
    print("=" * 70)

    # Filter to desorption only (absorption rows must be excluded per README)
    desorption = cleaned[cleaned['direction'] == 'desorption'].copy()

    # Separate catalyzed vs pure
    catalyzed = desorption[desorption['material_class'] == 'catalyzed_MgH2']['Ea_parsed']
    pure = desorption[desorption['material_class'] == 'pure_MgH2']['Ea_parsed']

    print(f"\nCatalyzed samples (desorption): {len(catalyzed)} samples")
    print(f"Pure MgH2 samples (desorption): {len(pure)} samples")
    print(f"\nCatalyzed Ea: mean={catalyzed.mean():.2f} kJ/mol, std={catalyzed.std():.2f}, range=[{catalyzed.min():.1f}, {catalyzed.max():.1f}]")
    print(f"Pure Ea: mean={pure.mean():.2f} kJ/mol, std={pure.std():.2f}, range=[{pure.min():.1f}, {pure.max():.1f}]")

    # Welch's t-test (does not assume equal variances)
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(catalyzed, pure, equal_var=False)

    mean_diff = catalyzed.mean() - pure.mean()
    # Effect size (Cohen's d, using pooled std)
    pooled_std = np.sqrt(((len(catalyzed)-1)*catalyzed.std()**2 + (len(pure)-1)*pure.std()**2) / (len(catalyzed) + len(pure) - 2))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # 95% CI for the difference (Welch)
    se_diff = np.sqrt(catalyzed.var(ddof=1)/len(catalyzed) + pure.var(ddof=1)/len(pure))
    # Welch-Satterthwaite degrees of freedom
    num = (catalyzed.var(ddof=1)/len(catalyzed) + pure.var(ddof=1)/len(pure))**2
    denom = (catalyzed.var(ddof=1)/len(catalyzed))**2/(len(catalyzed)-1) + (pure.var(ddof=1)/len(pure))**2/(len(pure)-1)
    df_welch = num / denom if denom > 0 else len(catalyzed) + len(pure) - 2
    t_crit = stats.t.ppf(0.975, df_welch)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    print(f"\nWelch's t-test results:")
    print(f"  Mean difference (catalyzed - pure): {mean_diff:.2f} kJ/mol")
    print(f"  95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}] kJ/mol")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")

    # Threshold: mean_diff ≤ -20 AND p < 0.05
    h2_supported = (mean_diff <= -20) and (p_value < 0.05)

    h2_result = {
        "hypothesis": "H2",
        "description": "Catalyzed samples have significantly lower Ea than pure MgH2",
        "test": "Welch's two-sample t-test (desorption only)",
        "n_catalyzed": int(len(catalyzed)),
        "n_pure": int(len(pure)),
        "mean_catalyzed_kJ_per_mol": float(catalyzed.mean()),
        "mean_pure_kJ_per_mol": float(pure.mean()),
        "mean_difference_kJ_per_mol": float(mean_diff),
        "ci_95_lower_kJ_per_mol": float(ci_lower),
        "ci_95_upper_kJ_per_mol": float(ci_upper),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "threshold": "Mean difference ≤ -20 kJ/mol and p < 0.05",
        "supported": bool(h2_supported),
        "reasoning": f"Mean difference = {mean_diff:.2f} kJ/mol (catalyzed - pure), "
                     f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}], p = {p_value:.4f}. "
                     f"Mean difference {'is' if mean_diff <= -20 else 'is not'} ≤ -20 kJ/mol; "
                     f"p-value {'is' if p_value < 0.05 else 'is not'} < 0.05; "
                     f"{'SUPPORTS' if h2_supported else 'REFUTES'} H2."
    }

    h2_path = f"{RESULTS_DIR}/h2_results.json"
    with open(h2_path, 'w', encoding='utf-8') as f:
        json.dump(h2_result, f, indent=2, ensure_ascii=False)
    print(f"\nH2 results saved to {h2_path}")

    return h2_result


# ============================================================
# H3: Method dependency within same sample (paired comparison)
# ============================================================
def test_h3_method_dependency(cleaned):
    """
    H3: Method choice (Kissinger vs isothermal_Arrhenius) significantly affects Ea
    for the same material within the same study.
    
    Test: Paired t-test on within-study, within-material method pairs.
    Threshold: Mean |ΔEa| ≥ 15 kJ/mol, paired p < 0.05.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS H3: Method dependency within same sample (paired t-test)")
    print("=" * 70)

    # Find pairs where the same material in the same paper has both Kissinger
    # and isothermal_Arrhenius measurements
    # The key example: MgH2-10 wt% H-V2O5 (doi 10.1007/s40820-024-01375-8)
    # has Kissinger Ea=113.8 and isothermal_Arrhenius Ea=84.55

    # Group by doi and material, then check if both methods are present
    method_pairs = []
    for (doi, material), group in cleaned.groupby(['doi', 'material']):
        kissinger_vals = group[group['method'] == 'Kissinger']['Ea_parsed'].values
        iso_vals = group[group['method'] == 'isothermal_Arrhenius']['Ea_parsed'].values
        if len(kissinger_vals) > 0 and len(iso_vals) > 0:
            # Use the first of each (should be one each typically)
            for k in kissinger_vals:
                for i in iso_vals:
                    method_pairs.append({
                        'doi': doi,
                        'material': material,
                        'kissinger_ea': float(k),
                        'iso_ea': float(i),
                        'difference': abs(k - i),
                        'signed_difference': k - i
                    })

    pairs_df = pd.DataFrame(method_pairs)
    print(f"\nFound {len(pairs_df)} within-study, within-material method pairs (Kissinger vs isothermal_Arrhenius)")
    print(pairs_df[['doi', 'material', 'kissinger_ea', 'iso_ea', 'difference', 'signed_difference']].to_string())

    if len(pairs_df) == 0:
        print("ERROR: No method pairs found. H3 cannot be tested.")
        return {"supported": False, "reason": "No within-study method pairs found"}

    # Paired t-test on signed differences
    signed_diffs = pairs_df['signed_difference'].values
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(signed_diffs, 0)  # Test if mean difference ≠ 0

    mean_abs_diff = np.abs(signed_diffs).mean()
    mean_signed_diff = signed_diffs.mean()
    std_abs_diff = np.abs(signed_diffs).std(ddof=1)

    # 95% CI for mean absolute difference
    se_abs = std_abs_diff / np.sqrt(len(signed_diffs))
    t_crit = stats.t.ppf(0.975, len(signed_diffs) - 1)
    ci_abs_lower = mean_abs_diff - t_crit * se_abs
    ci_abs_upper = mean_abs_diff + t_crit * se_abs

    print(f"\nPaired t-test results:")
    print(f"  Mean signed difference (Kissinger - isothermal): {mean_signed_diff:.2f} kJ/mol")
    print(f"  Mean absolute difference: {mean_abs_diff:.2f} kJ/mol (95% CI: [{ci_abs_lower:.2f}, {ci_abs_upper:.2f}])")
    print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

    # Threshold: mean |ΔEa| ≥ 15 kJ/mol AND paired p < 0.05
    h3_supported = (mean_abs_diff >= 15) and (p_value < 0.05)

    h3_result = {
        "hypothesis": "H3",
        "description": "Method choice significantly affects Ea for the same material within the same study",
        "test": "Paired t-test (Kissinger vs isothermal_Arrhenius)",
        "n_pairs": int(len(pairs_df)),
        "mean_signed_difference_kJ_per_mol": float(mean_signed_diff),
        "mean_absolute_difference_kJ_per_mol": float(mean_abs_diff),
        "ci_95_lower_abs_kJ_per_mol": float(ci_abs_lower),
        "ci_95_upper_abs_kJ_per_mol": float(ci_abs_upper),
        "p_value": float(p_value),
        "threshold": "Mean |ΔEa| ≥ 15 kJ/mol and paired p < 0.05",
        "supported": bool(h3_supported),
        "reasoning": f"Mean |ΔEa| = {mean_abs_diff:.2f} kJ/mol, p = {p_value:.4f}. "
                     f"Mean |ΔEa| {'is' if mean_abs_diff >= 15 else 'is not'} ≥ 15 kJ/mol; "
                     f"p-value {'is' if p_value < 0.05 else 'is not'} < 0.05; "
                     f"{'SUPPORTS' if h3_supported else 'REFUTES'} H3."
    }

    h3_path = f"{RESULTS_DIR}/h3_results.json"
    with open(h3_path, 'w', encoding='utf-8') as f:
        json.dump(h3_result, f, indent=2, ensure_ascii=False)
    print(f"\nH3 results saved to {h3_path}")

    return h3_result


# ============================================================
# H4: LOSO cross-validation for Ea prediction
# ============================================================
def test_h4_loso_cv(cleaned):
    """
    H4: Ea prediction using material features achieves reasonable accuracy under LOSO CV.
    
    Test: Leave-One-Subgroup-Out CV where subgroups are defined by material_class
    combined with catalyst_family. Features: one-hot encoded material_class and
    catalyst_family. Target: Ea_parsed.
    
    Threshold: R² ≥ 0.30 and RMSE ≤ 25 kJ/mol.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS H4: LOSO Cross-Validation for Ea Prediction")
    print("=" * 70)

    # Prepare features and target
    # Use only desorption rows for consistency
    df = cleaned[cleaned['direction'] == 'desorption'].copy()
    df = df.dropna(subset=['Ea_parsed', 'material_class', 'catalyst_family'])

    # Create subgroup identifier (material_class + catalyst_family)
    df['subgroup'] = df['material_class'] + '_' + df['catalyst_family']

    # One-hot encode features
    feature_cols = ['material_class', 'catalyst_family']
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    y = df['Ea_parsed'].values

    print(f"\nDataset for CV: {len(y)} samples, {X.shape[1]} features")
    print(f"Subgroups: {df['subgroup'].nunique()} unique subgroups")
    print(f"Subgroups: {sorted(df['subgroup'].unique())}")

    # LOSO CV: leave out one subgroup at a time
    subgroups = sorted(df['subgroup'].unique())
    n_subgroups = len(subgroups)

    predictions = []
    true_values = []
    fold_info = []

    for i, test_subgroup in enumerate(subgroups):
        # Train on all other subgroups
        train_mask = ~df['subgroup'].isin([test_subgroup])
        test_mask = df['subgroup'] == test_subgroup

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        # Simple linear regression using normal equation (OLS)
        # Add intercept
        X_train_with_intercept = np.column_stack([np.ones(len(X_train)), X_train.values])
        X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test.values])

        # Solve OLS: β = (X'X)^(-1) X'y
        try:
            beta = np.linalg.lstsq(X_train_with_intercept, y_train, rcond=None)[0]
            y_pred = X_test_with_intercept @ beta
        except np.linalg.LinAlgError:
            # Fallback: use mean prediction
            y_pred = np.full(len(y_test), np.mean(y_train))

        predictions.extend(y_pred)
        true_values.extend(y_test)
        fold_info.append({
            'subgroup': test_subgroup,
            'n_test': len(y_test),
            'predictions': y_pred.tolist(),
            'true': y_test.tolist()
        })

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    # Calculate metrics
    ss_res = np.sum((true_values - predictions) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    rmse = np.sqrt(np.mean((true_values - predictions) ** 2))

    print(f"\nLOSO CV Results ({n_subgroups} subgroups):")
    print(f"  R² = {r_squared:.4f}")
    print(f"  RMSE = {rmse:.2f} kJ/mol")
    print(f"  Mean true Ea = {true_values.mean():.2f} kJ/mol")
    print(f"  Mean predicted Ea = {predictions.mean():.2f} kJ/mol")

    # Threshold: R² ≥ 0.30 AND RMSE ≤ 25 kJ/mol
    h4_supported = (r_squared >= 0.30) and (rmse <= 25.0)

    h4_result = {
        "hypothesis": "H4",
        "description": "Ea prediction using material features achieves reasonable accuracy under LOSO CV",
        "test": "Leave-One-Subgroup-Out CV (material_class + catalyst_family), OLS regression",
        "n_subgroups": int(n_subgroups),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "r_squared": float(r_squared),
        "rmse_kJ_per_mol": float(rmse),
        "threshold": "R² ≥ 0.30 and RMSE ≤ 25 kJ/mol",
        "supported": bool(h4_supported),
        "reasoning": f"R² = {r_squared:.4f}, RMSE = {rmse:.2f} kJ/mol. "
                     f"R² {'is' if r_squared >= 0.30 else 'is not'} ≥ 0.30; "
                     f"RMSE {'is' if rmse <= 25 else 'is not'} ≤ 25 kJ/mol; "
                     f"{'SUPPORTS' if h4_supported else 'REFUTES'} H4."
    }

    h4_path = f"{RESULTS_DIR}/h4_results.json"
    with open(h4_path, 'w', encoding='utf-8') as f:
        json.dump(h4_result, f, indent=2, ensure_ascii=False)
    print(f"\nH4 results saved to {h4_path}")

    # Also save fold-level predictions for potential visualization
    fold_path = f"{RESULTS_DIR}/h4_folds.json"
    with open(fold_path, 'w', encoding='utf-8') as f:
        json.dump(fold_info, f, indent=2, ensure_ascii=False)

    return h4_result


# ============================================================
# H5: OOD prediction interval for Nb2C sample
# ============================================================
def test_h5_ood_prediction_interval(cleaned):
    """
    H5: OOD prediction interval (95%) for the Nb2C MXene sample covers its reported Ea.
    
    The MgH2@Nb2C sample (Ea ≈ 111.49 kJ/mol, marked with ~ in original) is treated
    as out-of-distribution. We train on all other samples and compute a bootstrap-based
    prediction interval for this sample.
    
    Threshold: 95% PI covers the reported Ea (~111.49 kJ/mol).
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS H5: OOD Prediction Interval for Nb2C Sample")
    print("=" * 70)

    # Find the Nb2C sample
    nb2c_mask = cleaned['material'].str.contains('Nb2C', na=False)
    nb2c = cleaned[nb2c_mask]
    if len(nb2c) == 0:
        print("ERROR: Nb2C sample not found. H5 cannot be tested.")
        return {"supported": False, "reason": "Nb2C sample not found"}

    nb2c_ea = nb2c['Ea_parsed'].values[0]
    nb2c_material = nb2c['material'].values[0]
    print(f"\nNb2C sample: {nb2c_material}, Ea = {nb2c_ea:.2f} kJ/mol")

    # Prepare training data (all other samples, desorption only)
    df = cleaned[cleaned['direction'] == 'desorption'].copy()
    df = df.dropna(subset=['Ea_parsed', 'material_class', 'catalyst_family'])

    # Remove Nb2C from training
    train_df = df[~df['material'].str.contains('Nb2C', na=False)].copy()
    train_df = train_df.dropna(subset=['Ea_parsed'])

    # Features: one-hot encoded material_class and catalyst_family
    X_train = pd.get_dummies(train_df[['material_class', 'catalyst_family']], drop_first=True)
    y_train = train_df['Ea_parsed'].values

    # Feature vector for Nb2C (same encoding)
    nb2c_row = pd.DataFrame([{
        'material_class': nb2c['material_class'].values[0],
        'catalyst_family': nb2c['catalyst_family'].values[0]
    }])
    X_nb2c = pd.get_dummies(nb2c_row, drop_first=True)
    # Ensure same columns as training
    for col in X_train.columns:
        if col not in X_nb2c.columns:
            X_nb2c[col] = 0
    X_nb2c = X_nb2c[X_train.columns]

    print(f"\nTraining set: {len(train_df)} samples, {X_train.shape[1]} features")
    print(f"Nb2C feature vector: {X_nb2c.to_dict('records')[0]}")

    # Bootstrap-based prediction interval
    n_bootstrap = 10000
    np.random.seed(42)

    # Fit OLS on training data
    X_train_with_intercept = np.column_stack([np.ones(len(X_train)), X_train.values])
    beta = np.linalg.lstsq(X_train_with_intercept, y_train, rcond=None)[0]
    y_train_pred = X_train_with_intercept @ beta
    residuals = y_train - y_train_pred

    # Bootstrap predictions for Nb2C
    bootstrap_predictions = []
    for _ in range(n_bootstrap):
        # Resample residuals with replacement
        resampled_residuals = np.random.choice(residuals, size=len(residuals), replace=True)
        # Bootstrap training predictions
        y_train_boot = y_train_pred + resampled_residuals
        # Re-fit OLS with bootstrapped data
        X_boot_with_intercept = np.column_stack([np.ones(len(X_train)), X_train.values])
        beta_boot = np.linalg.lstsq(X_boot_with_intercept, y_train_boot, rcond=None)[0]
        # Predict for Nb2C
        nb2c_with_intercept = np.array([1, X_nb2c.values[0]])
        pred = nb2c_with_intercept @ beta_boot
        bootstrap_predictions.append(pred)

    bootstrap_predictions = np.array(bootstrap_predictions)

    # 95% prediction interval (2.5th and 97.5th percentiles)
    pi_lower = np.percentile(bootstrap_predictions, 2.5)
    pi_upper = np.percentile(bootstrap_predictions, 97.5)

    # Check if true value is covered
    pi_covers = pi_lower <= nb2c_ea <= pi_upper

    print(f"\nBootstrap Prediction Interval (95%):")
    print(f"  Lower bound: {pi_lower:.2f} kJ/mol")
    print(f"  Upper bound: {pi_upper:.2f} kJ/mol")
    print(f"  Nb2C reported Ea: {nb2c_ea:.2f} kJ/mol")
    print(f"  PI covers Nb2C Ea: {pi_covers}")

    # Threshold: 95% PI covers the reported Ea
    h5_supported = pi_covers

    h5_result = {
        "hypothesis": "H5",
        "description": "OOD prediction interval (95%) for Nb2C sample covers its reported Ea",
        "test": "Bootstrap-based prediction interval (10,000 iterations) with OLS regression",
        "n_bootstrap": int(n_bootstrap),
        "n_train_samples": int(len(train_df)),
        "reported_ea_kJ_per_mol": float(nb2c_ea),
        "pi_95_lower_kJ_per_mol": float(pi_lower),
        "pi_95_upper_kJ_per_mol": float(pi_upper),
        "pi_covers_ea": bool(pi_covers),
        "threshold": "95% PI covers reported Ea",
        "supported": bool(h5_supported),
        "reasoning": f"95% PI = [{pi_lower:.2f}, {pi_upper:.2f}] kJ/mol, "
                     f"reported Ea = {nb2c_ea:.2f} kJ/mol. "
                     f"PI {'covers' if pi_covers else 'does not cover'} reported Ea; "
                     f"{'SUPPORTS' if h5_supported else 'REFUTES'} H5."
    }

    h5_path = f"{RESULTS_DIR}/h5_results.json"
    with open(h5_path, 'w', encoding='utf-8') as f:
        json.dump(h5_result, f, indent=2, ensure_ascii=False)
    print(f"\nH5 results saved to {h5_path}")

    return h5_result


# ============================================================
# Main execution
# ============================================================
def main():
    print("\n" + "=" * 70)
    print("MgH2 ACTIVATION ENERGY DATASET - HYPOTHESIS TESTING FRAMEWORK")
    print("=" * 70)

    # 1. Data validation
    cleaned, validation_results = validate_data()

    # 2. Run all hypothesis tests
    print("\n" + "=" * 70)
    print("RUNNING HYPOTHESIS TESTS")
    print("=" * 70)

    h1_result = test_h1_within_paper_method_difference(cleaned)
    h2_result = test_h2_catalyst_effect(cleaned)
    h3_result = test_h3_method_dependency(cleaned)
    h4_result = test_h4_loso_cv(cleaned)
    h5_result = test_h5_ood_prediction_interval(cleaned)

    # 3. Generate summary report
    all_results = [h1_result, h2_result, h3_result, h4_result, h5_result]
    supported_count = sum(1 for r in all_results if r['supported'])
    total_count = len(all_results)

    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("=" * 70)
    print(f"\n{'Hypothesis':<10} {'Status':<12} {'Key Metric':<30} {'Conclusion'}")
    print("-" * 70)
    for r in all_results:
        status = "SUPPORTED" if r['supported'] else "REFUTED"
        if 'mean_absolute_difference_kJ_per_mol' in r:
            metric = f"|ΔEa| = {r['mean_absolute_difference_kJ_per_mol']:.2f} kJ/mol"
        elif 'mean_difference_kJ_per_mol' in r:
            metric = f"ΔEa = {r['mean_difference_kJ_per_mol']:.2f} kJ/mol"
        elif 'r_squared' in r:
            metric = f"R² = {r['r_squared']:.4f}, RMSE = {r['rmse_kJ_per_mol']:.2f} kJ/mol"
        elif 'reported_ea_kJ_per_mol' in r:
            metric = f"PI = [{r['pi_95_lower_kJ_per_mol']:.2f}, {r['pi_95_upper_kJ_per_mol']:.2f}]"
        else:
            metric = f"mean = {r.get('mean_difference_kJ_per_mol', 0):.2f}"
        print(f"{r['hypothesis']:<10} {status:<12} {metric:<30} {'✓' if r['supported'] else '✗'}")

    print("-" * 70)
    print(f"\nOverall: {supported_count}/{total_count} hypotheses supported")

    # 4. Save final report
    report = {
        "validation": validation_results,
        "hypotheses": {
            "H1": h1_result,
            "H2": h2_result,
            "H3": h3_result,
            "H4": h4_result,
            "H5": h5_result
        },
        "overall_summary": {
            "total_hypotheses": total_count,
            "supported_hypotheses": supported_count,
            "refuted_hypotheses": total_count - supported_count,
            "support_rate": supported_count / total_count
        }
    }

    report_path = f"{RESULTS_DIR}/report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to {report_path}")

    # 5. Generate markdown report
    md_report = "# Hypothesis Testing Report\n\n"
    md_report += "## Summary\n\n"
    md_report += f"Of {total_count} hypotheses tested, {supported_count} were **SUPPORTED** and "
    f"{total_count - supported_count} were **REFUTED**.\n\n"

    md_report += "## Individual Hypothesis Results\n\n"
    for r in all_results:
        status = "**SUPPORTED**" if r['supported'] else "**REFUTED**"
        md_report += f"### {r['hypothesis']}: {r['description']}\n\n"
        md_report += f"Status: {status}\n\n"
        md_report += f"**Test:** {r['test']}\n\n"
        md_report += f"**Threshold:** {r['threshold']}\n\n"
        # Add key metrics
        metrics = []
        for k, v in r.items():
            if k not in ['hypothesis', 'description', 'test', 'threshold', 'supported', 'reasoning', 'direction']:
                metrics.append(f"  - {k}: {v}")
        md_report += f"**Results:**\n" + "\n".join(metrics) + "\n\n"
        md_report += f"**Conclusion:** {r['reasoning']}\n\n"

    md_path = f"{RESULTS_DIR}/report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"Markdown report saved to {md_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()