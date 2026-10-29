#!/usr/bin/env python3
"""
Comprehensive Meta-Analysis of MgH2 Activation Energy Data

This script performs five hypothesis tests on the extracted activation energy dataset:
H1: Within-paper catalytic effect with random-effects meta-analysis and bootstrap CI
H2: Catalyst family explanatory power with permutation test
H3: Method noise vs catalytic signal comparison
H4: LOSO cross-validation with feature-based prediction
H5: Extrapolative prediction for missing cell

All inferences are clustered by DOI to account for non-independent rows.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

# Ensure results directory exists
os.makedirs('/workspace/research/shusheng_run/ws35/results', exist_ok=True)


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_and_preprocess_data(filepath):
    """Load CSV and apply cleaning rules from README."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {list(df.columns)}")

    # Rule 2: Remove row with questionable mapping (MgH2 + 5 wt% V)
    before = len(df)
    df = df[df['material'] != 'MgH2 + 5 wt% V'].copy()
    print(f"Removed questionable row (MgH2 + 5 wt% V): {before - len(df)} rows")

    # Rule 3: Remove duplicate rows for same material with different methods
    # Mg-5Ni: rows with 210 and 206 kJ/mol (same paper, same material, different Tm definition)
    # Keep the first occurrence (210 kJ/mol)
    before = len(df)
    mg5ni_mask = (df['material'] == 'Mg-5Ni (hydrogenated)')
    mg5ni_df = df[mg5ni_mask].copy()
    if len(mg5ni_df) > 1:
        keep_idx = mg5ni_df['Ea_kJ_per_mol'].idxmax()
        df = df.drop(mg5ni_df[~mg5ni_df.index.isin([keep_idx])].index)
        print(f"Removed duplicate Mg-5Ni row (kept 210 kJ/mol, removed 206 kJ/mol)")

    # Parse Ea_kJ_per_mol - extract first float from string like "109 ± 1" or "153.09"
    def parse_ea(value):
        if pd.isna(value):
            return np.nan
        val_str = str(value)
        import re
        match = re.search(r'[-+]?\d+\.?\d*', val_str)
        if match:
            return float(match.group())
        return np.nan

    df['Ea_parsed'] = df['Ea_kJ_per_mol'].apply(parse_ea)

    # Check for any remaining NaN Ea values
    nan_ea_count = df['Ea_parsed'].isna().sum()
    if nan_ea_count > 0:
        print(f"Warning: {nan_ea_count} rows have unparsable Ea values, will be excluded from Ea analyses")

    # Drop rows with NaN Ea for Ea-dependent analyses
    df_clean = df[df['Ea_parsed'].notna()].copy()
    print(f"After parsing Ea: {len(df_clean)} rows with valid Ea values")

    # Count unique DOIs
    unique_dois = df_clean['doi'].nunique()
    print(f"Unique DOIs in cleaned data: {unique_dois}")

    return df, df_clean


# ============================================================
# H1: WITHIN-PAPER CATALYTIC EFFECT
# ============================================================

def h1_within_paper_catalytic_effect(df_clean):
    """
    H1: Within-paper catalytic effect with random-effects meta-analysis and bootstrap CI.
    Only within-paper pairing, only desorption,
    DeltaEa = mean(Ea_pure) - mean(Ea_catalyzed) for each paper.
    """
    print("\n" + "="*70)
    print("H1: Within-Paper Catalytic Effect")
    print("="*70)

    # Filter to desorption only
    df_des = df_clean[df_clean['direction'] == 'desorption'].copy()
    print(f"Desorption rows: {len(df_des)}")

    # Identify papers with both pure and catalyzed samples
    papers_with_both = []
    for doi in df_des['doi'].unique():
        paper_data = df_des[df_des['doi'] == doi]
        pure_rows = paper_data[paper_data['material_class'] == 'pure_MgH2']
        catalyzed_rows = paper_data[paper_data['material_class'] == 'catalyzed_MgH2']
        if len(pure_rows) > 0 and len(catalyzed_rows) > 0:
            papers_with_both.append(doi)

    print(f"Papers with both pure and catalyzed samples: {len(papers_with_both)}")

    # Compute within-paper DeltaEa
    delta_ea_list = []
    for doi in papers_with_both:
        paper_data = df_des[df_des['doi'] == doi]
        pure_ea = paper_data[paper_data['material_class'] == 'pure_MgH2']['Ea_parsed'].mean()
        catalyzed_ea = paper_data[paper_data['material_class'] == 'catalyzed_MgH2']['Ea_parsed'].mean()
        delta_ea = pure_ea - catalyzed_ea
        delta_ea_list.append({
            'doi': doi,
            'pure_mean_ea': pure_ea,
            'catalyzed_mean_ea': catalyzed_ea,
            'delta_ea': delta_ea,
            'n_pure': len(paper_data[paper_data['material_class'] == 'pure_MgH2']),
            'n_catalyzed': len(paper_data[paper_data['material_class'] == 'catalyzed_MgH2'])
        })

    delta_ea_df = pd.DataFrame(delta_ea_list)
    print(f"\nWithin-paper DeltaEa values:")
    print(delta_ea_df[['doi', 'pure_mean_ea', 'catalyzed_mean_ea', 'delta_ea']].to_string())

    n = len(delta_ea_df)
    delta_ea_values = delta_ea_df['delta_ea'].values

    # Fixed-effects meta-analysis (equal weights since no within-study variances)
    fe_mean = float(np.mean(delta_ea_values))
    fe_se = float(np.std(delta_ea_values, ddof=1) / np.sqrt(n))

    # Random-effects (DerSimonian-Laird)
    q_stat = float(np.sum((delta_ea_values - fe_mean) ** 2))
    df_q = n - 1

    if q_stat > df_q:
        tau2 = (q_stat - df_q) / n  # Simplified estimator
        tau2 = max(0.0, tau2)
    else:
        tau2 = 0.0

    # Random-effects weights
    if tau2 > 0:
        group_var = float(np.var(delta_ea_values, ddof=1))
        weights = 1.0 / (tau2 + group_var)
        re_mean = float(weights * delta_ea_values.sum() / weights.sum())
        re_se = float(np.sqrt(1.0 / weights.sum()))
    else:
        re_mean = fe_mean
        re_se = fe_se

    # Confidence intervals
    re_ci_lower = re_mean - 1.96 * re_se
    re_ci_upper = re_mean + 1.96 * re_se

    print(f"\nRandom-effects meta-analysis (DerSimonian-Laird):")
    print(f"  Between-study variance (tau^2) = {tau2:.2f}")
    print(f"  Mean DeltaEa = {re_mean:.2f} kJ/mol (SE = {re_se:.2f})")
    print(f"  95% CI: [{re_ci_lower:.2f}, {re_ci_upper:.2f}]")

    # Bootstrap CI (clustered by DOI)
    n_bootstrap = 10000
    bootstrap_means = []

    for _ in range(n_bootstrap):
        boot_indices = np.random.choice(n, size=n, replace=True)
        boot_delta = delta_ea_values[boot_indices]
        bootstrap_means.append(np.mean(boot_delta))

    bootstrap_means = np.array(bootstrap_means)
    bootstrap_ci_lower = float(np.percentile(bootstrap_means, 2.5))
    bootstrap_ci_upper = float(np.percentile(bootstrap_means, 97.5))

    print(f"\nBootstrap 95% CI (n={n_bootstrap}): [{bootstrap_ci_lower:.2f}, {bootstrap_ci_upper:.2f}]")

    # Significance test
    p_value = float(2 * (1 - stats.norm.cdf(abs(re_mean) / re_se)))
    print(f"\nRandom-effects p-value: {p_value:.4f}")

    h1_results = {
        'n_papers': n,
        'delta_ea_values': delta_ea_values.tolist(),
        'fe_mean': fe_mean,
        'fe_se': fe_se,
        're_mean': re_mean,
        're_se': re_se,
        're_ci_lower': re_ci_lower,
        're_ci_upper': re_ci_upper,
        'tau2': tau2,
        'bootstrap_ci_lower': bootstrap_ci_lower,
        'bootstrap_ci_upper': bootstrap_ci_upper,
        'p_value': p_value,
        'delta_ea_df': delta_ea_df
    }

    return h1_results


# ============================================================
# H2: CATALYST FAMILY EXPLANATORY POWER
# ============================================================

def h2_catalyst_family_explanatory_power(df_clean):
    """
    H2: Catalyst family explanatory power with permutation test.
    Only catalyzed_MgH2 rows, permutation test with paper-level shuffling.
    """
    print("\n" + "="*70)
    print("H2: Catalyst Family Explanatory Power")
    print("="*70)

    # Filter to catalyzed_MgH2 and desorption only
    df_cat = df_clean[(df_clean['material_class'] == 'catalyzed_MgH2') &
                      (df_clean['direction'] == 'desorption')].copy()
    print(f"Catalyzed_MgH2 desorption rows: {len(df_cat)}")

    # Remove rows with missing catalyst_family
    df_cat = df_cat[df_cat['catalyst_family'].notna()].copy()
    print(f"After removing missing catalyst_family: {len(df_cat)} rows")

    unique_dois = df_cat['doi'].unique()
    print(f"Unique DOIs in catalyzed_MgH2 set: {len(unique_dois)}")

    families = sorted(df_cat['catalyst_family'].unique())
    print(f"Catalyst families: {families}")

    # Compute group means and overall mean
    group_means = df_cat.groupby('catalyst_family')['Ea_parsed'].mean()
    overall_mean = float(df_cat['Ea_parsed'].mean())

    # Between-group sum of squares
    ss_between = 0.0
    for fm in families:
        fm_data = df_cat[df_cat['catalyst_family'] == fm]
        ss_between += len(fm_data) * (group_means[fm] - overall_mean) ** 2

    # Within-group sum of squares
    ss_within = 0.0
    for fm in families:
        fm_data = df_cat[df_cat['catalyst_family'] == fm]
        ss_within += float((fm_data['Ea_parsed'] - group_means[fm]) ** 2).sum()

    k = len(families)
    n = len(df_cat)

    # F-statistic
    df_between = k - 1
    df_within = n - k
    f_stat = float((ss_between / df_between) / (ss_within / df_within))

    print(f"\nBaseline F-statistic: {f_stat:.4f}")
    print(f"  Between-group SS: {ss_between:.2f}, df: {df_between}")
    print(f"  Within-group SS: {ss_within:.2f}, df: {df_within}")

    # Permutation test with paper-level shuffling
    n_permutations = 10000
    perm_f_stats = []

    doi_groups = df_cat.groupby('doi')

    for _ in range(n_permutations):
        permuted_ea = []
        permuted_families = []

        for doi, group in doi_groups:
            if len(group) > 1:
                families_in_group = group['catalyst_family'].values
                permuted_families_in_group = np.random.permutation(families_in_group)
                permuted_ea.extend(group['Ea_parsed'].values)
                permuted_families.extend(permuted_families_in_group)
            else:
                permuted_ea.append(group['Ea_parsed'].values[0])
                permuted_families.append(group['catalyst_family'].values[0])

        perm_df = pd.DataFrame({'Ea_parsed': permuted_ea, 'catalyst_family': permuted_families})
        perm_group_means = perm_df.groupby('catalyst_family')['Ea_parsed'].mean()
        perm_overall_mean = perm_df['Ea_parsed'].mean()

        perm_ss_between = 0.0
        for fm in perm_group_means.index:
            fm_data = perm_df[perm_df['catalyst_family'] == fm]
            perm_ss_between += len(fm_data) * (perm_group_means[fm] - perm_overall_mean) ** 2

        perm_ss_within = 0.0
        for fm in perm_group_means.index:
            fm_data = perm_df[perm_df['catalyst_family'] == fm]
            perm_ss_within += float((fm_data['Ea_parsed'] - perm_group_means[fm]) ** 2).sum()

        if df_between > 0 and df_within > 0:
            perm_f = float((perm_ss_between / df_between) / (perm_ss_within / df_within))
        else:
            perm_f = 0.0
        perm_f_stats.append(perm_f)

    perm_f_stats = np.array(perm_f_stats)

    p_value = float(np.mean(perm_f_stats >= f_stat))
    print(f"\nPermutation test (n={n_permutations}):")
    print(f"  Observed F = {f_stat:.4f}")
    print(f"  Mean permuted F = {np.mean(perm_f_stats):.4f}")
    print(f"  P-value = {p_value:.4f}")

    # Effect size: eta-squared
    ss_total = ss_between + ss_within
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0
    print(f"\nEffect size (eta^2): {eta_squared:.4f}")

    h2_results = {
        'f_stat': f_stat,
        'perm_f_stats': perm_f_stats.tolist(),
        'p_value': p_value,
        'eta_squared': eta_squared,
        'group_means': group_means.to_dict(),
        'n_observations': n,
        'n_families': k
    }

    return h2_results


# ============================================================
# H3: METHOD NOISE VS CATALYTIC SIGNAL
# ============================================================

def h3_method_noise_vs_catalytic_signal(df_clean):
    """
    H3: Method noise vs catalytic signal comparison.
    Same doi, same material, different methods (Kissinger vs isothermal_Arrhenius).
    """
    print("\n" + "="*70)
    print("H3: Method Noise vs Catalytic Signal")
    print("="*70)

    df_des = df_clean[df_clean['direction'] == 'desorption'].copy()

    # Find rows with same doi, same material, different methods
    method_pairs = []

    for doi in df_des['doi'].unique():
        paper_data = df_des[df_des['doi'] == doi]
        for material in paper_data['material'].unique():
            material_data = paper_data[paper_data['material'] == material]
            methods = material_data['method'].unique()
            if 'Kissinger' in methods and 'isothermal_Arrhenius' in methods:
                kissinger_ea = material_data[material_data['method'] == 'Kissinger']['Ea_parsed'].values
                iso_ea = material_data[material_data['method'] == 'isothermal_Arrhenius']['Ea_parsed'].values
                if len(kissinger_ea) > 0 and len(iso_ea) > 0:
                    method_pairs.append({
                        'doi': doi,
                        'material': material,
                        'kissinger_ea': float(kissinger_ea[0]),
                        'iso_ea': float(iso_ea[0]),
                        'method_diff': abs(kissinger_ea[0] - iso_ea[0])
                    })

    print(f"Found {len(method_pairs)} same-material, same-paper method comparison pairs")
    print(f"\nMethod comparison pairs:")
    for mp in method_pairs:
        print(f"  {mp['doi']}: {mp['material']} - Kissinger: {mp['kissinger_ea']:.2f}, "
              f"isothermal_Arrhenius: {mp['iso_ea']:.2f}, diff: {mp['method_diff']:.2f}")

    # Get catalytic effects from H1 analysis
    h1_results = h1_within_paper_catalytic_effect(df_clean)
    delta_ea_values = h1_results['delta_ea_values']

    method_diffs = [mp['method_diff'] for mp in method_pairs]
    print(f"\nMethod-induced differences: {method_diffs}")
    print(f"Mean method difference: {np.mean(method_diffs):.2f} kJ/mol")
    print(f"Within-paper DeltaEa values: {delta_ea_values}")
    print(f"Mean DeltaEa: {np.mean(delta_ea_values):.2f} kJ/mol")

    # Compute ratios
    ratios = []
    for mp in method_pairs:
        paper_delta = delta_ea_values[h1_results['delta_ea_df']['doi'] == mp['doi']]
        if len(paper_delta) > 0 and paper_delta[0] > 0:
            ratio = mp['method_diff'] / paper_delta[0]
            ratios.append(ratio)

    if ratios:
        print(f"\nRatio (method_diff / catalytic_effect): {ratios}")
        print(f"Mean ratio: {np.mean(ratios):.2f}")
        print(f"Proportion of method diffs < catalytic effect: {np.mean(np.array(ratios) < 1):.2f}")

    # Permutation test
    n_perm = 10000
    perm_ratios = []
    shuffled_delta = np.random.permutation(delta_ea_values)

    for _ in range(n_perm):
        perm_ratios_for_this = []
        for i, mp in enumerate(method_pairs):
            delta_val = shuffled_delta[i % len(shuffled_delta)]
            if delta_val > 0:
                ratio = mp['method_diff'] / delta_val
                perm_ratios_for_this.append(ratio)
        if perm_ratios_for_this:
            perm_ratios.append(np.mean(perm_ratios_for_this))

    perm_ratios = np.array(perm_ratios)
    observed_mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    p_value = float(np.mean(perm_ratios >= observed_mean_ratio)) if observed_mean_ratio > 0 else 1.0

    print(f"\nPermutation test (n={n_perm}):")
    print(f"  Observed mean ratio: {observed_mean_ratio:.4f}")
    print(f"  Mean permuted ratio: {np.mean(perm_ratios):.4f}")
    print(f"  P-value: {p_value:.4f}")

    h3_results = {
        'method_pairs': method_pairs,
        'method_diffs': method_diffs,
        'delta_ea_values': delta_ea_values,
        'ratios': ratios,
        'perm_ratios': perm_ratios.tolist(),
        'p_value': p_value
    }

    return h3_results


# ============================================================
# H4: LOSO CROSS-VALIDATION
# ============================================================

def h4_loso_cross_validation(df_clean):
    """
    H4: LOSO cross-validation with feature-based prediction.
    Features: material_class + catalyst_family + method, desorption only.
    """
    print("\n" + "="*70)
    print("H4: Leave-One-Paper-Out Cross-Validation")
    print("="*70)

    df_des = df_clean[df_clean['direction'] == 'desorption'].copy()
    df_des = df_des.dropna(subset=['material_class', 'catalyst_family', 'method'])

    unique_dois = df_des['doi'].unique()
    print(f"Unique DOIs for LOSO CV: {len(unique_dois)}")

    df_des['feature'] = df_des['material_class'] + '_' + df_des['catalyst_family'] + '_' + df_des['method']

    all_preds = []
    all_true = []

    for test_doi in unique_dois:
        train_mask = df_des['doi'] != test_doi
        train_data = df_des[train_mask]
        test_data = df_des[df_des['doi'] == test_doi]

        if len(train_data) == 0 or len(test_data) == 0:
            continue

        feature_means = train_data.groupby('feature')['Ea_parsed'].mean()

        for _, row in test_data.iterrows():
            feat = row['feature']
            pred = feature_means.get(feat, train_data['Ea_parsed'].mean())
            all_preds.append(pred)
            all_true.append(row['Ea_parsed'])

    if all_preds and all_true:
        all_preds = np.array(all_preds)
        all_true = np.array(all_true)

        overall_rmse = float(np.sqrt(np.mean((all_preds - all_true) ** 2)))
        overall_mae = float(np.mean(np.abs(all_preds - all_true)))

        baseline_rmse = float(np.sqrt(np.mean((all_true - np.mean(all_true)) ** 2)))
        baseline_mae = float(np.mean(np.abs(all_true - np.mean(all_true))))

        print(f"\nLOSO CV Results:")
        print(f"  Number of folds (papers): {len(unique_dois)}")
        print(f"  RMSE: {overall_rmse:.2f} kJ/mol")
        print(f"  MAE: {overall_mae:.2f} kJ/mol")
        print(f"  Baseline (overall mean) RMSE: {baseline_rmse:.2f} kJ/mol")
        print(f"  Baseline (overall mean) MAE: {baseline_mae:.2f} kJ/mol")
        print(f"  Improvement over baseline RMSE: {((baseline_rmse - overall_rmse) / baseline_rmse * 100):.1f}%")

        feature_stats = df_des.groupby('feature')['Ea_parsed'].agg(['mean', 'std', 'count'])
        print(f"\nFeature combinations (mean Ea):")
        print(feature_stats.sort_values('mean').to_string())

        h4_results = {
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'baseline_rmse': baseline_rmse,
            'baseline_mae': baseline_mae,
            'feature_stats': feature_stats.to_dict(),
            'n_folds': len(unique_dois),
            'n_predictions': len(all_preds)
        }
    else:
        h4_results = {'error': 'No valid predictions for CV'}

    return h4_results


# ============================================================
# H5: EXTRAPOLATIVE PREDICTION
# ============================================================

def h5_extrapolative_prediction(df_clean):
    """
    H5: Extrapolative prediction for missing cell.
    Predict for material_class=catalyzed_MgH2, catalyst_family=other, method=isothermal_Arrhenius.
    """
    print("\n" + "="*70)
    print("H5: Extrapolative Prediction for Missing Cell")
    print("="*70)

    df_des = df_clean[df_clean['direction'] == 'desorption'].copy()

    target_class = 'catalyzed_MgH2'
    target_family = 'other'
    target_method = 'isothermal_Arrhenius'

    existing = df_des[
        (df_des['material_class'] == target_class) &
        (df_des['catalyst_family'] == target_family) &
        (df_des['method'] == target_method)
    ]

    if len(existing) > 0:
        print(f"Target cell already exists in data: {len(existing)} row(s)")
        print(f"  Ea values: {existing['Ea_parsed'].values}")
        pred_value = float(existing['Ea_parsed'].mean())
        print(f"  Predicted Ea (mean of existing): {pred_value:.2f} kJ/mol")

        h5_results = {
            'predicted_ea': pred_value,
            'existing_values': existing['Ea_parsed'].values.tolist(),
            'n_existing': len(existing),
            'std': float(existing['Ea_parsed'].std()) if len(existing) > 1 else 0.0
        }
    else:
        print(f"Target cell ({target_class}, {target_family}, {target_method}) does NOT exist in data.")
        print("Performing extrapolative prediction based on similar cells...")

        similar = df_des[df_des['material_class'] == target_class].copy()
        print(f"\nAll catalyzed_MgH2 desorption rows: {len(similar)}")
        print(similar.groupby(['catalyst_family', 'method'])['Ea_parsed'].agg(['mean', 'count']).to_string())

        family_effects = similar[similar['catalyst_family'] != 'other'].groupby('catalyst_family')['Ea_parsed'].mean()
        overall_catalyzed_mean = float(similar['Ea_parsed'].mean())
        method_effects = similar.groupby('method')['Ea_parsed'].mean()

        avg_family_effect = overall_catalyzed_mean - family_effects.mean() if len(family_effects) > 0 else 0.0

        if 'Kissinger' in method_effects.index and 'isothermal_Arrhenius' in method_effects.index:
            method_effect = float(method_effects['isothermal_Arrhenius'] - method_effects['Kissinger'])
        else:
            method_effect = 0.0

        pred_value = overall_catalyzed_mean + avg_family_effect + method_effect

        # Find nearest neighbor
        closest_dist = np.inf
        closest_combo = None
        closest_mean = None
        for combo in similar.groupby(['catalyst_family', 'method']).groups:
            if combo[0] != target_family and combo[1] != target_method:
                combo_mean = float(similar[
                    (similar['catalyst_family'] == combo[0]) &
                    (similar['method'] == combo[1])
                ]['Ea_parsed'].mean())
                dist = (abs(combo[0] != target_family) + abs(combo[1] != target_method))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_combo = combo
                    closest_mean = combo_mean

        if closest_combo is not None:
            print(f"Nearest neighbor: {closest_combo} with mean Ea = {closest_mean:.2f} kJ/mol")
            if 'isothermal_Arrhenius' in method_effects.index:
                pred_value = closest_mean + method_effect

        print(f"\nPredicted Ea for (catalyzed_MgH2, other, isothermal_Arrhenius): {pred_value:.2f} kJ/mol")

        similar_std = float(similar['Ea_parsed'].std())
        print(f"Similar cells std: {similar_std:.2f} kJ/mol")
        print(f"Approximate 95% CI: [{pred_value - 1.96*similar_std:.2f}, {pred_value + 1.96*similar_std:.2f}]")

        h5_results = {
            'predicted_ea': pred_value,
            'similar_cells_count': len(similar),
            'similar_std': similar_std,
            'overall_mean': overall_catalyzed_mean,
            'family_effects': family_effects.to_dict(),
            'method_effects': method_effects.to_dict(),
            'closest_combo': closest_combo,
            'closest_mean': closest_mean
        }

    return h5_results


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("Starting comprehensive meta-analysis...")
    print("="*70)

    # Load and preprocess data
    data_path = '/workspace/research/shusheng_run/ws35/data/ea_dataset.csv'
    df_raw, df_clean = load_and_preprocess_data(data_path)

    # Save cleaned data for reference
    df_clean.to_csv('/workspace/research/shusheng_run/ws35/data/cleaned_ea_dataset.csv', index=False)
    print(f"\nCleaned data saved to cleaned_ea_dataset.csv ({len(df_clean)} rows)")

    # Run all analyses
    h1_results = h1_within_paper_catalytic_effect(df_clean)
    h2_results = h2_catalyst_family_explanatory_power(df_clean)
    h3_results = h3_method_noise_vs_catalytic_signal(df_clean)
    h4_results = h4_loso_cross_validation(df_clean)
    h5_results = h5_extrapolative_prediction(df_clean)

    # Print summary
    summary = f"""Meta-Analysis Summary
======================
Data: {len(df_clean)} rows from {df_clean['doi'].nunique()} papers

H1: Within-Paper Catalytic Effect
  Mean DeltaEa (random-effects): {h1_results['re_mean']:.2f} kJ/mol
  95% CI: [{h1_results['re_ci_lower']:.2f}, {h1_results['re_ci_upper']:.2f}]
  p-value: {h1_results['p_value']:.4f}

H2: Catalyst Family Explanatory Power
  F-statistic: {h2_results['f_stat']:.4f}
  P-value (permutation): {h2_results['p_value']:.4f}
  Effect size (eta^2): {h2_results['eta_squared']:.4f}

H3: Method Noise vs Catalytic Signal
  Method-induced differences: {h3_results['method_diffs']}
  P-value (permutation): {h3_results['p_value']:.4f}

H4: LOSO Cross-Validation
  RMSE: {h4_results.get('overall_rmse', 'N/A'):.2f} kJ/mol
  MAE: {h4_results.get('overall_mae', 'N/A'):.2f} kJ/mol

H5: Extrapolative Prediction
  Predicted Ea (catalyzed_MgH2, other, isothermal_Arrhenius): {h5_results['predicted_ea']:.2f} kJ/mol
"""
    print(summary)

    # Save summary
    with open('/workspace/research/shusheng_run/ws35/results/meta_analysis_summary.txt', 'w') as f:
        f.write(summary)
    print("Results saved to results/meta_analysis_summary.txt")

    # Save detailed results
    results = {
        'h1': h1_results,
        'h2': h2_results,
        'h3': h3_results,
        'h4': h4_results,
        'h5': h5_results,
        'n_papers': df_clean['doi'].nunique(),
        'n_rows': len(df_clean)
    }

    with open('/workspace/research/shusheng_run/ws35/results/meta_analysis_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Detailed results saved to results/meta_analysis_results.pkl")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()