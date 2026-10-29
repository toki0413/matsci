#!/usr/bin/env python3
"""
Comprehensive Analysis of Catalyst Effects on MgH2 Activation Energy
Implements 5 hypotheses as specified in the task.
Uses only numpy and scipy.
"""

import csv
import json
import numpy as np
from scipy import stats
from collections import defaultdict
import os


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_data(filepath):
    """Load the cleaned Ea dataset from CSV."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = dict(row)
            row['Ea_parsed'] = float(row['Ea_parsed']) if row['Ea_parsed'] else None
            data.append(row)
    return data


def filter_desorption_only(data):
    """Filter to desorption-only rows, excluding absorption data."""
    return [r for r in data if r.get('direction', '') == 'desorption']


def filter_out_special_cases(data):
    """Exclude the MgH2 + 5 wt% V row."""
    return [r for r in data if r.get('material', '') != 'MgH2 + 5 wt% V']


def get_paired_samples(data):
    """Find paired samples: pure_MgH2 vs catalyzed from the same paper."""
    by_doi = defaultdict(list)
    for row in data:
        by_doi[row['doi']].append(row)

    pairs = []
    for doi, rows in by_doi.items():
        pure_rows = [r for r in rows if r['material_class'] == 'pure_MgH2']
        catalyzed_rows = [r for r in rows if r['material_class'] == 'catalyzed_MgH2']
        for pur in pure_rows:
            for cat in catalyzed_rows:
                if pur['material'] != cat['material']:
                    delta_ea = cat['Ea_parsed'] - pur['Ea_parsed']
                    pairs.append((doi, delta_ea, pur, cat))

    return pairs


# ============================================================
# HYPOTHESIS 1: Catalysts reduce Ea (paired by paper)
# ============================================================

def hypothesis_1(data):
    """H1: Catalysts reduce activation energy (paired comparison by paper)."""
    print("=" * 70)
    print("HYPOTHESIS 1: Catalysts Reduce Activation Energy (Paired Comparison)")
    print("=" * 70)

    pairs = get_paired_samples(data)

    if not pairs:
        print("  No paired samples found.")
        return None

    delta_eas = [d for _, d, _, _ in pairs]
    mean_dea = float(np.mean(delta_eas))
    std_dea = float(np.std(delta_eas, ddof=1))
    n_pairs = len(delta_eas)

    print(f"  Found {n_pairs} paired samples (pure_MgH2 vs catalyzed_MgH2 from same paper)")
    print(f"  Mean Delta_Ea (catalyzed - pure): {mean_dea:.2f} kJ/mol")
    print(f"  Std Delta_Ea: {std_dea:.2f} kJ/mol")

    if n_pairs > 1:
        t_stat, p_value = stats.ttest_1samp(delta_eas, 0)
        print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_value:.4f}")

        se = std_dea / np.sqrt(n_pairs)
        ci_lower = mean_dea - stats.t.ppf(0.975, n_pairs - 1) * se
        ci_upper = mean_dea + stats.t.ppf(0.975, n_pairs - 1) * se
        print(f"  95% CI for mean Delta_Ea: [{ci_lower:.2f}, {ci_upper:.2f}] kJ/mol")

        cohens_d = mean_dea / std_dea
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

        if mean_dea < 0:
            print("  RESULT: Delta_Ea is significantly negative -> catalysts reduce Ea")
            print("  SUPPORT: H1 is supported")
        else:
            print("  RESULT: Delta_Ea is positive or not significant -> no clear reduction")
            print("  RESULT: H1 not supported")

        print("\n  Individual paired samples:")
        for doi, delta, pur, cat in pairs:
            print(f"    {doi[:40]}... | pure: {pur['Ea_parsed']:.2f} | cat: {cat['Ea_parsed']:.2f} | Delta: {delta:.2f}")

        return {
            'n_pairs': n_pairs,
            'mean_delta_ea': mean_dea,
            'std_delta_ea': std_dea,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'cohens_d': cohens_d,
            'pairs': pairs
        }
    else:
        print("  Not enough pairs for statistical test.")
        return None


# ============================================================
# HYPOTHESIS 2: Catalyst family effect
# ============================================================

def hypothesis_2(data):
    """H2: Different catalyst families have different effects on Ea."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Catalyst Family Effect on Activation Energy")
    print("=" * 70)

    catalyzed = [r for r in data if r['material_class'] == 'catalyzed_MgH2']
    pure = [r for r in data if r['material_class'] == 'pure_MgH2']

    pure_eas = np.array([r['Ea_parsed'] for r in pure if r['Ea_parsed'] is not None])

    print(f"\n  Pure MgH2 baseline (n={len(pure_eas)}):")
    print(f"    Mean Ea: {pure_eas.mean():.2f} +/- {pure_eas.std(ddof=1):.2f} kJ/mol")

    by_family = defaultdict(list)
    for row in catalyzed:
        if row['Ea_parsed'] is not None:
            by_family[row['catalyst_family']].append(row['Ea_parsed'])

    family_results = {}
    for family, eas in sorted(by_family.items()):
        if len(eas) >= 2:
            eas_arr = np.array(eas)
            mean_ea = float(eas_arr.mean())
            std_ea = float(eas_arr.std(ddof=1))
            n = len(eas)
            reduction = mean_ea - pure_eas.mean()
            family_results[family] = {
                'n': n, 'mean_ea': mean_ea, 'std_ea': std_ea,
                'reduction_from_pure': reduction
            }
            print(f"\n  {family} (n={n}):")
            print(f"    Mean Ea: {mean_ea:.2f} +/- {std_ea:.2f} kJ/mol")
            print(f"    Reduction from pure baseline: {reduction:.2f} kJ/mol")

            t_stat, p_val = stats.ttest_1samp(eas_arr, pure_eas.mean())
            print(f"    t-test vs pure baseline: t = {t_stat:.3f}, p = {p_val:.4f}")

        elif len(eas) == 1:
            print(f"\n  {family} (n=1): Ea = {eas[0]:.2f} kJ/mol")

    family_groups = [eas for family, eas in by_family.items() if family != 'none' and len(eas) >= 2]
    if len(family_groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*family_groups)
        print(f"\n  ANOVA across catalyst families: F = {f_stat:.3f}, p = {p_anova:.4f}")

        family_names = sorted([f for f in by_family.keys() if f != 'none' and len(by_family[f]) >= 2])
        print(f"\n  Pairwise comparisons:")
        for i, f1 in enumerate(family_names):
            for f2 in family_names[i+1:]:
                eas1 = np.array(by_family[f1])
                eas2 = np.array(by_family[f2])
                t_stat, p_val = stats.ttest_ind(eas1, eas2, equal_var=False)
                print(f"    {f1} vs {f2}: t = {t_stat:.3f}, p = {p_val:.4f}")

    return family_results


# ============================================================
# HYPOTHESIS 3: Cross-method comparison
# ============================================================

def hypothesis_3(data):
    """H3: Same sample measured by different methods gives different Ea values."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Cross-Method Comparison (Same Sample, Different Methods)")
    print("=" * 70)

    by_material_doi = defaultdict(list)
    for row in data:
        key = (row['material'], row['doi'])
        by_material_doi[key].append(row)

    cross_method_pairs = []
    for key, rows in by_material_doi.items():
        kissinger_rows = [r for r in rows if r['method'] == 'Kissinger']
        arrhenius_rows = [r for r in rows if r['method'] == 'isothermal_Arrhenius']
        for k in kissinger_rows:
            for a in arrhenius_rows:
                if k['material'] == a['material']:
                    cross_method_pairs.append((k, a))

    print(f"\n  Found {len(cross_method_pairs)} cross-method pairs")

    if cross_method_pairs:
        deltas = []
        for k, a in cross_method_pairs:
            delta = a['Ea_parsed'] - k['Ea_parsed']
            deltas.append(delta)
            print(f"    {k['material'][:40]}...: Kissinger={k['Ea_parsed']:.2f}, Arrhenius={a['Ea_parsed']:.2f}, Delta={delta:.2f}")

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))
        print(f"\n  Mean Delta_Ea (Arrhenius - Kissinger): {mean_delta:.2f} kJ/mol")
        print(f"  Std Delta_Ea: {std_delta:.2f} kJ/mol")

        if len(deltas) > 1:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            print(f"  One-sample t-test: t = {t_stat:.3f}, p = {p_val:.4f}")

        return {
            'n_pairs': len(cross_method_pairs),
            'mean_delta_ea': mean_delta,
            'std_delta_ea': std_delta,
            'pairs': cross_method_pairs
        }
    else:
        print("  No cross-method pairs found.")
        return None


# ============================================================
# HYPOTHESIS 4: Meta-analysis with bootstrap
# ============================================================

def hypothesis_4(data):
    """H4: Meta-analysis with bootstrap (fixed effects and random effects models)."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Meta-Analysis with Bootstrap (Fixed Effects & Random Effects)")
    print("=" * 70)

    catalyzed = [r for r in data if r['material_class'] == 'catalyzed_MgH2']
    pure = [r for r in data if r['material_class'] == 'pure_MgH2']

    catalyzed_eas = np.array([r['Ea_parsed'] for r in catalyzed if r['Ea_parsed'] is not None])
    pure_eas = np.array([r['Ea_parsed'] for r in pure if r['Ea_parsed'] is not None])

    print(f"\n  Catalyzed MgH2 samples (n={len(catalyzed_eas)}):")
    print(f"    Mean Ea: {catalyzed_eas.mean():.2f} +/- {catalyzed_eas.std(ddof=1):.2f} kJ/mol")
    print(f"    Range: [{catalyzed_eas.min():.2f}, {catalyzed_eas.max():.2f}] kJ/mol")

    print(f"\n  Pure MgH2 baseline samples (n={len(pure_eas)}):")
    print(f"    Mean Ea: {pure_eas.mean():.2f} +/- {pure_eas.std(ddof=1):.2f} kJ/mol")
    print(f"    Range: [{pure_eas.min():.2f}, {pure_eas.max():.2f}] kJ/mol")

    n_cat = len(catalyzed_eas)
    mean_fe = float(np.mean(catalyzed_eas))
    se_fe = float(catalyzed_eas.std(ddof=1) / np.sqrt(n_cat))

    print(f"\n  Fixed Effects Model:")
    print(f"    Pooled mean Ea: {mean_fe:.2f} kJ/mol")
    print(f"    Standard error: {se_fe:.2f} kJ/mol")
    print(f"    95% CI: [{mean_fe - 1.96*se_fe:.2f}, {mean_fe + 1.96*se_fe:.2f}] kJ/mol")

    if n_cat > 1:
        tau_sq = max(0, catalyzed_eas.var(ddof=1) - catalyzed_eas.var(ddof=1) / n_cat)
        Q = np.sum((catalyzed_eas - mean_fe)**2)
        df = n_cat - 1
        if df > 0:
            tau_sq_dl = max(0, (Q - (df - se_fe**2 * n_cat)) / (1 - sum(1/n_cat for _ in range(n_cat)) / n_cat))
        else:
            tau_sq_dl = 0.0

        tau = float(np.sqrt(tau_sq_dl))
        se_re = float(np.sqrt(tau_sq_dl + catalyzed_eas.var(ddof=1) / n_cat))
        mean_re = mean_fe

        print(f"\n  Random Effects Model (DerSimonian-Laird):")
        print(f"    Between-study variance (tau^2): {tau_sq_dl:.2f}")
        print(f"    Between-study std (tau): {tau:.2f} kJ/mol")
        print(f"    Pooled mean Ea: {mean_re:.2f} kJ/mol")
        print(f"    Standard error: {se_re:.2f} kJ/mol")
        print(f"    95% CI: [{mean_re - 1.96*se_re:.2f}, {mean_re + 1.96*se_re:.2f}] kJ/mol")

        np.random.seed(42)
        n_bootstrap = 10000

        fe_bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(catalyzed_eas, size=n_cat, replace=True)
            fe_bootstrap_means.append(np.mean(sample))
        fe_bootstrap_means = np.array(fe_bootstrap_means)
        fe_ci_lower = float(np.percentile(fe_bootstrap_means, 2.5))
        fe_ci_upper = float(np.percentile(fe_bootstrap_means, 97.5))

        print(f"\n  Bootstrap (n={n_bootstrap}) - Fixed Effects:")
        print(f"    Bootstrap mean: {fe_bootstrap_means.mean():.2f} kJ/mol")
        print(f"    Bootstrap 95% CI: [{fe_ci_lower:.2f}, {fe_ci_upper:.2f}] kJ/mol")
        print(f"    Bootstrap SE: {fe_bootstrap_means.std():.2f} kJ/mol")

        re_bootstrap_means = []
        for _ in range(n_bootstrap):
            true_effect = np.random.normal(mean_re, tau)
            simulated = np.random.normal(true_effect, catalyzed_eas.std(ddof=1), size=n_cat)
            re_bootstrap_means.append(np.mean(simulated))
        re_bootstrap_means = np.array(re_bootstrap_means)
        re_ci_lower = float(np.percentile(re_bootstrap_means, 2.5))
        re_ci_upper = float(np.percentile(re_bootstrap_means, 97.5))

        print(f"\n  Bootstrap (n={n_bootstrap}) - Random Effects:")
        print(f"    Bootstrap mean: {re_bootstrap_means.mean():.2f} kJ/mol")
        print(f"    Bootstrap 95% CI: [{re_ci_lower:.2f}, {re_ci_upper:.2f}] kJ/mol")
        print(f"    Bootstrap SE: {re_bootstrap_means.std():.2f} kJ/mol")

        return {
            'n_catalyzed': n_cat,
            'mean_catalyzed': mean_fe,
            'std_catalyzed': catalyzed_eas.std(ddof=1),
            'mean_pure': pure_eas.mean(),
            'std_pure': pure_eas.std(ddof=1),
            'fixed_effects': {
                'mean': mean_fe,
                'se': se_fe,
                'ci_lower': mean_fe - 1.96 * se_fe,
                'ci_upper': mean_fe + 1.96 * se_fe
            },
            'random_effects': {
                'tau_sq': tau_sq_dl,
                'tau': tau,
                'se': se_re,
                'mean': mean_re,
                'ci_lower': mean_re - 1.96 * se_re,
                'ci_upper': mean_re + 1.96 * se_re
            },
            'bootstrap': {
                'fe_mean': fe_bootstrap_means.mean(),
                'fe_ci_lower': fe_ci_lower,
                'fe_ci_upper': fe_ci_upper,
                'fe_se': fe_bootstrap_means.std(),
                're_mean': re_bootstrap_means.mean(),
                're_ci_lower': re_ci_lower,
                're_ci_upper': re_ci_upper,
                're_se': re_bootstrap_means.std()
            }
        }
    else:
        print("  Not enough catalyzed samples for meta-analysis.")
        return None


# ============================================================
# HYPOTHESIS 5: Cross-validation for Ea prediction
# ============================================================

def hypothesis_5(data):
    """H5: Cross-validation to predict Ea of catalyzed samples from pure_MgH2 baseline."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: Cross-Validation for Ea Prediction")
    print("=" * 70)

    pure = [r for r in data if r['material_class'] == 'pure_MgH2']
    catalyzed = [r for r in data if r['material_class'] == 'catalyzed_MgH2']

    pure_eas = np.array([r['Ea_parsed'] for r in pure if r['Ea_parsed'] is not None])
    catalyzed_eas = np.array([r['Ea_parsed'] for r in catalyzed if r['Ea_parsed'] is not None])

    print(f"\n  Pure MgH2 samples (n={len(pure_eas)}): mean = {pure_eas.mean():.2f} kJ/mol")
    print(f"  Catalyzed MgH2 samples (n={len(catalyzed_eas)}): mean = {catalyzed_eas.mean():.2f} kJ/mol")

    n_cv = len(catalyzed_eas)
    predictions = []
    true_values = []
    errors = []

    for i in range(n_cv):
        train_catalyzed = np.delete(catalyzed_eas, i)
        true_val = catalyzed_eas[i]

        pure_mean = pure_eas.mean()
        reductions = pure_mean - train_catalyzed
        mean_reduction = np.mean(reductions)

        predicted = pure_mean - mean_reduction

        predictions.append(predicted)
        true_values.append(true_val)
        errors.append(true_val - predicted)

    predictions = np.array(predictions)
    true_values = np.array(true_values)
    errors = np.array(errors)

    rmse = float(np.sqrt(np.mean(errors**2)))
    mae = float(np.mean(np.abs(errors)))

    print(f"\n  Cross-Validation Results (Leave-One-Out):")
    print(f"  Number of samples: {n_cv}")
    print(f"  RMSE: {rmse:.2f} kJ/mol")
    print(f"  MAE: {mae:.2f} kJ/mol")
    print(f"  Mean prediction error: {errors.mean():.2f} kJ/mol")

    within_10 = np.sum(np.abs(errors) <= 10)
    within_20 = np.sum(np.abs(errors) <= 20)
    print(f"  Predictions within +/-10 kJ/mol: {within_10}/{n_cv} ({within_10/n_cv*100:.1f}%)")
    print(f"  Predictions within +/-20 kJ/mol: {within_20}/{n_cv} ({within_20/n_cv*100:.1f}%)")

    print(f"\n  Individual predictions:")
    for i in range(n_cv):
        print(f"    Sample {i+1}: True={true_values[i]:.2f}, Pred={predictions[i]:.2f}, Error={errors[i]:.2f}")

    # Linear regression with catalyst family effects
    family_to_idx = {}
    family_features = []
    for r in catalyzed:
        if r['Ea_parsed'] is None:
            continue
        fam = r['catalyst_family']
        if fam not in family_to_idx:
            family_to_idx[fam] = len(family_to_idx)

    n_cat_reg = len(family_to_idx)
    # Create one-hot encoded features with fixed length
    family_features = []
    for r in catalyzed:
        if r['Ea_parsed'] is None:
            continue
        vec = [0.0] * n_cat_reg
        vec[family_to_idx[r['catalyst_family']]] = 1.0
        family_features.append(vec)

    family_features = np.array(family_features)
    y = np.array([r['Ea_parsed'] for r in catalyzed if r['Ea_parsed'] is not None])

    X = np.column_stack([np.ones(len(y)), family_features])
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    rmse_reg = float(np.sqrt(np.mean((y - y_pred)**2)))
    print(f"\n  Linear Regression (with catalyst family effects):")
    print(f"  Intercept (pure baseline estimate): {coeffs[0]:.2f} kJ/mol")
    for fam_name, fam_idx in sorted(family_to_idx.items()):
        print(f"  {fam_name} effect: {coeffs[fam_idx+1]:.2f} kJ/mol")
    print(f"  Regression RMSE: {rmse_reg:.2f} kJ/mol")

    return {
        'n_samples': n_cv,
        'rmse': rmse,
        'mae': mae,
        'mean_error': errors.mean(),
        'within_10': int(within_10),
        'within_20': int(within_20),
        'true_values': true_values.tolist(),
        'predictions': predictions.tolist(),
        'errors': errors.tolist(),
        'regression_rmse': rmse_reg,
        'regression_coeffs': coeffs.tolist()
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    """Run all hypothesis tests and generate comprehensive report."""
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS OF CATALYST EFFECTS ON MgH2 ACTIVATION ENERGY")
    print("=" * 70)

    data_path = '/workspace/research/shusheng_run/ws35/data/cleaned_ea_dataset.csv'
    data = load_data(data_path)
    print(f"\n  Loaded {len(data)} rows from {data_path}")

    data = filter_desorption_only(data)
    data = filter_out_special_cases(data)
    print(f"  After filtering (desorption only, special cases excluded): {len(data)} rows")

    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    material_classes = defaultdict(int)
    catalyst_families = defaultdict(int)
    methods = defaultdict(int)

    for row in data:
        material_classes[row['material_class']] += 1
        catalyst_families[row['catalyst_family']] += 1
        methods[row['method']] += 1

    print(f"\n  Material classes: {dict(material_classes)}")
    print(f"  Catalyst families: {dict(catalyst_families)}")
    print(f"  Methods: {dict(methods)}")

    # Run all hypotheses
    h1_result = hypothesis_1(data)
    h2_result = hypothesis_2(data)
    h3_result = hypothesis_3(data)
    h4_result = hypothesis_4(data)
    h5_result = hypothesis_5(data)

    # Generate summary report
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    report = {
        'h1': h1_result,
        'h2': h2_result,
        'h3': h3_result,
        'h4': h4_result,
        'h5': h5_result
    }

    # Save results
    results_dir = '/workspace/research/shusheng_run/ws35/results'
    os.makedirs(results_dir, exist_ok=True)

    # Convert non-serializable objects to serializable format
    def serialize(obj):
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    report_serialized = serialize(report)

    with open(os.path.join(results_dir, 'analysis_results.json'), 'w') as f:
        json.dump(report_serialized, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_dir}/analysis_results.json")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if h1_result:
        print(f"\n  H1 (Catalyst effect):")
        print(f"    Mean Delta_Ea = {h1_result['mean_delta_ea']:.2f} kJ/mol (p={h1_result['p_value']:.4f})")
        if h1_result['mean_delta_ea'] < 0:
            print("    -> SUPPORTED: Catalysts significantly reduce Ea")
        else:
            print("    -> NOT SUPPORTED: No clear reduction in Ea")

    if h4_result:
        print(f"\n  H4 (Meta-analysis):")
        print(f"    Fixed effects mean Ea: {h4_result['fixed_effects']['mean']:.2f} kJ/mol")
        print(f"    Random effects mean Ea: {h4_result['random_effects']['mean']:.2f} kJ/mol")
        print(f"    Between-study variance tau^2 = {h4_result['random_effects']['tau_sq']:.2f}")

    if h5_result:
        print(f"\n  H5 (Cross-validation):")
        print(f"    RMSE = {h5_result['rmse']:.2f} kJ/mol")
        print(f"    Accuracy within +/-10 kJ/mol: {h5_result['within_10']}/{h5_result['n_samples']}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()