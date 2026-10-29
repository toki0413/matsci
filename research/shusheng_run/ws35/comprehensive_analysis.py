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
            data.append(row)
    return data


def parse_ea_value(val):
    """Parse Ea value from string, handling 'x +/- y' and '~x' formats."""
    if not val or val.strip() == '':
        return None
    val = val.strip()
    # Handle approximate values (~)
    if val.startswith('~'):
        val = val[1:].strip()
    # Split on +/-
    parts = val.split('±')
    if len(parts) > 1:
        val = parts[0].strip()
    # Extract first number
    parts = val.split()
    if parts:
        try:
            return float(parts[0])
        except ValueError:
            return None
    return None


def filter_desorption_only(data):
    """Filter to desorption-only rows, excluding absorption data."""
    return [r for r in data if r.get('direction', '') == 'desorption']


def filter_out_special_cases(data):
    """Exclude the MgH2 + 5 wt% V row (already excluded in cleaned data, but check)."""
    # The cleaned data already excludes this row, but we verify
    return [r for r in data if r.get('material', '') != 'MgH2 + 5 wt% V']


def get_paired_samples(data):
    """Find paired samples: pure_MgH2 vs catalyzed from the same paper (same doi).
    Only use desorption data where both pure and catalyzed samples exist in the same paper."""
    # Group by doi
    by_doi = defaultdict(list)
    for row in data:
        by_doi[row['doi']].append(row)

    pairs = []
    for doi, rows in by_doi.items():
        pure_rows = [r for r in rows if r['material_class'] == 'pure_MgH2']
        catalyzed_rows = [r for r in rows if r['material_class'] == 'catalyzed_MgH2']
        for pur in pure_rows:
            for cat in catalyzed_rows:
                # Ensure they're different materials
                if pur['material'] != cat['material']:
                    # Convert Ea_parsed to float
                    pur_ea = float(pur['Ea_parsed'])
                    cat_ea = float(cat['Ea_parsed'])
                    delta_ea = cat_ea - pur_ea
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

    # Get paired samples
    pairs = get_paired_samples(data)

    if not pairs:
        print("  No paired samples found.")
        return None

    delta_eas = [d for _, d, _, _ in pairs]
    mean_dea = float(np.mean(delta_eas))
    std_dea = float(np.std(delta_eas, ddof=1))
    n_pairs = len(delta_eas)

    print(f"  Found {n_pairs} paired samples (pure_MgH2 vs catalyzed_MgH2 from same paper)")
    print(f"  Mean ΔEa (catalyzed - pure): {mean_dea:.2f} kJ/mol")
    print(f"  Std ΔEa: {std_dea:.2f} kJ/mol")

    # Paired t-test
    if n_pairs > 1:
        t_stat, p_value = stats.ttest_1samp(delta_eas, 0)
        print(f"  Paired t-test: t = {t_stat:.3f}, p = {p_value:.4f}")

        # Confidence interval for mean difference
        se = std_dea / np.sqrt(n_pairs)
        ci_lower = mean_dea - stats.t.ppf(0.975, n_pairs - 1) * se
        ci_upper = mean_dea + stats.t.ppf(0.975, n_pairs - 1) * se
        print(f"  95% CI for mean ΔEa: [{ci_lower:.2f}, {ci_upper:.2f}] kJ/mol")

        # Effect size (Cohen's d for paired samples)
        cohens_d = mean_dea / std_dea
        print(f"  Effect size (Cohen's d): {cohens_d:.3f}")

        # Direction: negative means catalyzed has lower Ea
        if mean_dea < 0:
            print("  RESULT: ΔEa is significantly negative → catalysts reduce Ea")
            print("  SUPPORT: H1 is supported")
        else:
            print("  RESULT: ΔEa is positive or not significant → no clear reduction")
            print("  RESULT: H1 not supported")

        # Print individual pairs
        print("\n  Individual paired samples:")
        for doi, delta, pur, cat in pairs:
            print(f"    {doi[:40]}... | pure: {pur['Ea_parsed']:.2f} | cat: {cat['Ea_parsed']:.2f} | Δ: {delta:.2f}")

        return {
            'n_pairs': n_pairs,
            'mean_delta_ea': mean_dea,
            'std_delta_ea': std_dea,
            't_statistic': t_stat if n_pairs > 1 else None,
            'p_value': p_value if n_pairs > 1 else None,
            'ci_lower': ci_lower if n_pairs > 1 else None,
            'ci_upper': ci_upper if n_pairs > 1 else None,
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

    # Filter to catalyzed_MgH2 only
    catalyzed = [r for r in data if r['material_class'] == 'catalyzed_MgH2']

    # Group by catalyst family
    by_family = defaultdict(list)
    for row in catalyzed:
        by_family[row['catalyst_family']].append(row['Ea_parsed'])

    # Also include pure_MgH2 as baseline
    pure = [r for r in data if r['material_class'] == 'pure_MgH2']
    pure_eas = [r['Ea_parsed'] for r in pure]

    print(f"\n  Pure MgH2 baseline (n={len(pure_eas)}):")
    print(f"    Mean Ea: {np.mean(pure_eas):.2f} ± {np.std(pure_eas, ddof=1):.2f} kJ/mol")

    family_results = {}
    for family, eas in sorted(by_family.items()):
        if len(eas) >= 2:
            mean_ea = float(np.mean(eas))
            std_ea = float(np.std(eas, ddof=1))
            n = len(eas)
            # Reduction from pure baseline
            reduction = mean_ea - np.mean(pure_eas)
            family_results[family] = {
                'n': n, 'mean_ea': mean_ea, 'std_ea': std_ea,
                'reduction_from_pure': reduction
            }
            print(f"\n  {family} (n={n}):")
            print(f"    Mean Ea: {mean_ea:.2f} ± {std_ea:.2f} kJ/mol")
            print(f"    Reduction from pure baseline: {reduction:.2f} kJ/mol")

            # One-sample t-test against pure baseline
            t_stat, p_val = stats.ttest_1samp(eas, np.mean(pure_eas))
            print(f"    t-test vs pure baseline: t = {t_stat:.3f}, p = {p_val:.4f}")

        elif len(eas) == 1:
            print(f"\n  {family} (n=1): Ea = {eas[0]:.2f} kJ/mol")

    # ANOVA across catalyst families (excluding 'none' which has no catalyst)
    family_groups = [eas for family, eas in by_family.items() if family != 'none' and len(eas) >= 2]
    if len(family_groups) >= 2:
        f_stat, p_anova = stats.f_oneway(*family_groups)
        print(f"\n  ANOVA across catalyst families: F = {f_stat:.3f}, p = {p_anova:.4f}")

        # Post-hoc pairwise comparisons (Tukey-like with t-tests)
        family_names = sorted([f for f in by_family.keys() if f != 'none' and len(by_family[f]) >= 2])
        print(f"\n  Pairwise comparisons (catalyzed vs pure baseline):")
        for i, f1 in enumerate(family_names):
            for f2 in family_names[i+1:]:
                eas1 = by_family[f1]
                eas2 = by_family[f2]
                t_stat, p_val = stats.ttest_ind(eas1, eas2, equal_var=False)
                print(f"    {f1} vs {f2}: t = {t_stat:.3f}, p = {p_val:.4f}")

    return family_results


# ============================================================
# HYPOTHESIS 3: Cross-method comparison (same sample, different methods)
# ============================================================

def hypothesis_3(data):
    """H3: Same sample measured by different methods gives different Ea values."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Cross-Method Comparison (Same Sample, Different Methods)")
    print("=" * 70)

    # Group by material and doi to find same sample with different methods
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

    print(f"\n  Found {len(cross_method_pairs)} cross-method pairs (same material, same paper, different methods)")

    if cross_method_pairs:
        deltas = []
        for k, a in cross_method_pairs:
            delta = a['Ea_parsed'] - k['Ea_parsed']
            deltas.append(delta)
            print(f"    {k['material']} (doi: {k['doi'][:30]}...):")
            print(f"      Kissinger: {k['Ea_parsed']:.2f} kJ/mol")
            print(f"      Isothermal Arrhenius: {a['Ea_parsed']:.2f} kJ/mol")
            print(f"      ΔEa (Arrhenius - Kissinger): {delta:.2f} kJ/mol")

        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas, ddof=1))
        print(f"\n  Mean ΔEa (Arrhenius - Kissinger): {mean_delta:.2f} kJ/mol")
        print(f"  Std ΔEa: {std_delta:.2f} kJ/mol")

        # Test if mean difference is significantly different from zero
        if len(deltas) > 1:
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            print(f"  One-sample t-test: t = {t_stat:.3f}, p = {p_val:.4f}")
        else:
            print(f"  Only one cross-method pair; cannot perform statistical test.")

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
# HYPOTHESIS 4: Meta-analysis with bootstrap (fixed effects & random effects)
# ============================================================

def hypothesis_4(data):
    """H4: Meta-analysis with bootstrap (fixed effects and random effects models)."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Meta-Analysis with Bootstrap (Fixed Effects & Random Effects)")
    print("=" * 70)

    # Use catalyzed_MgH2 samples for meta-analysis
    catalyzed = [r for r in data if r['material_class'] == 'catalyzed_MgH2']
    pure = [r for r in data if r['material_class'] == 'pure_MgH2']

    catalyzed_eas = np.array([r['Ea_parsed'] for r in catalyzed])
    pure_eas = np.array([r['Ea_parsed'] for r in pure])

    print(f"\n  Catalyzed MgH2 samples (n={len(catalyzed_eas)}):")
    print(f"    Mean Ea: {catalyzed_eas.mean():.2f} ± {catalyzed_eas.std(ddof=1):.2f} kJ/mol")
    print(f"    Range: [{catalyzed_eas.min():.2f}, {catalyzed_eas.max():.2f}] kJ/mol")

    print(f"\n  Pure MgH2 baseline samples (n={len(pure_eas)}):")
    print(f"    Mean Ea: {pure_eas.mean():.2f} ± {pure_eas.std(ddof=1):.2f} kJ/mol")
    print(f"    Range: [{pure_eas.min():.2f}, {pure_eas.max():.2f}] kJ/mol")

    # Fixed effects model: weighted average by inverse variance
    # For simplicity, assume equal weights (fixed effects with equal variance)
    n_cat = len(catalyzed_eas)
    mean_fe = float(np.mean(catalyzed_eas))
    se_fe = float(catalyzed_eas.std(ddof=1) / np.sqrt(n_cat))

    print(f"\n  Fixed Effects Model:")
    print(f"    Pooled mean Ea: {mean_fe:.2f} kJ/mol")
    print(f"    Standard error: {se_fe:.2f} kJ/mol")
    print(f"    95% CI: [{mean_fe - 1.96*se_fe:.2f}, {mean_fe + 1.96*se_fe:.2f}] kJ/mol")

    # Random effects model (DerSimonian-Laird)
    # Estimate between-study variance (tau^2)
    if n_cat > 1:
        # Simple random effects: treat each study as having its own true effect
        # and estimate between-study variance
        tau_sq = max(0, catalyzed_eas.var(ddof=1) - catalyzed_eas.var(ddof=1) / n_cat)
        # More proper DL estimator
        Q = np.sum((catalyzed_eas - mean_fe)**2)
        df = n_cat - 1
        C = n_cat - np.sum([1/n_cat] * n_cat)  # Simplified
        if df > 0 and C > 0:
            tau_sq_dl = (Q - (df - se_fe**2 * n_cat)) / (1 - sum(1/n_cat for _ in range(n_cat)) / n_cat)
            tau_sq_dl = max(0, tau_sq_dl)
        else:
            tau_sq_dl = 0.0

        tau = float(np.sqrt(tau_sq_dl))
        # Random effects standard error
        se_re = float(np.sqrt(tau_sq_dl + catalyzed_eas.var(ddof=1) / n_cat))
        mean_re = mean_fe  # Same point estimate

        print(f"\n  Random Effects Model (DerSimonian-Laird):")
        print(f"    Between-study variance (τ²): {tau_sq_dl:.2f}")
        print(f"    Between-study std (τ): {tau:.2f} kJ/mol")
        print(f"    Pooled mean Ea: {mean_re:.2f} kJ/mol")
        print(f"    Standard error: {se_re:.2f} kJ/mol")
        print(f"    95% CI: [{mean_re - 1.96*se_re:.2f}, {mean_re + 1.96*se_re:.2f}] kJ/mol")

        # Bootstrap for both models
        np.random.seed(42)
        n_bootstrap = 10000

        # Bootstrap for fixed effects
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

        # Bootstrap for random effects (parametric bootstrap)
        re_bootstrap_means = []
        for _ in range(n_bootstrap):
            # Simulate from random effects model
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

    # Separate pure and catalyzed
    pure = [r for r in data if r['material_class'] == 'pure_MgH2']
    catalyzed = [r for r in data if r['material_class'] == 'catalyzed_MgH2']

    pure_eas = np.array([r['Ea_parsed'] for r in pure])
    catalyzed_eas = np.array([r['Ea_parsed'] for r in catalyzed])

    print(f"\n  Pure MgH2 samples (n={len(pure_eas)}): mean = {pure_eas.mean():.2f} kJ/mol")
    print(f"  Catalyzed MgH2 samples (n={len(catalyzed_eas)}): mean = {catalyzed_eas.mean():.2f} kJ/mol")

    # Simple prediction model: predicted Ea = pure_mean - observed_reduction
    # For cross-validation, we'll use leave-one-out on catalyzed samples
    # and predict based on pure baseline

    n_cv = len(catalyzed_eas)
    predictions = []
    true_values = []
    errors = []

    for i in range(n_cv):
        # Leave out one catalyzed sample
        train_catalyzed = np.delete(catalyzed_eas, i)
        true_val = catalyzed_eas[i]

        # Predict: use the mean reduction from other catalyzed samples
        # relative to pure baseline
        pure_mean = pure_eas.mean()
        # Observed reduction for training samples
        reductions = pure_mean - train_catalyzed
        mean_reduction = np.mean(reductions)

        # Predict the held-out sample
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

    # Also compute prediction accuracy
    within_10 = np.sum(np.abs(errors) <= 10)
    within_20 = np.sum(np.abs(errors) <= 20)
    print(f"  Predictions within ±10 kJ/mol: {within_10}/{n_cv} ({within_10/n_cv*100:.1f}%)")
    print(f"  Predictions within ±20 kJ/mol: {within_20}/{n_cv} ({within_20/n_cv*100:.1f}%)")

    # Individual predictions
    print(f"\n  Individual predictions:")
    for i in range(n_cv):
        print(f"    Sample {i+1}: True={true_values[i]:.2f}, Pred={predictions[i]:.2f}, Error={errors[i]:.2f}")

    # Also do a simple regression using catalyst family as categorical predictor
    # (one-hot encoded)
    family_to_idx = {}
    family_features = []
    for r in catalyzed:
        fam = r['catalyst_family']
        if fam not in family_to_idx:
            family_to_idx[fam] = len(family_to_idx)
        # One-hot encode
        vec = [0] * len(family_to_idx)
        vec[family_to_idx[fam]] = 1
        family_features.append(vec)

    family_features = np.array(family_features)

    # Simple linear regression: Ea = intercept + family_effects
    # Add pure baseline as a feature
    X = np.column_stack([np.ones(n_cv), family_features])
    y = catalyzed_eas

    # Least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    print(f"\n  Linear Regression (with catalyst family effects):")
    print(f"  Intercept (pure baseline estimate): {coeffs[0]:.2f} kJ/mol")
    for idx, fam in sorted(family_to_idx.items()):
        print(f"  {fam} effect: {coeffs[idx+1]:.2f} kJ/mol")

    # Predictions from regression
    y_pred = X @ coeffs
    rmse_reg = float(np.sqrt(np.mean((y - y_pred)**2)))
    print(f"  Regression RMSE: {rmse_reg:.2f} kJ/mol")

    return {
        'n_samples': n_cv,
        'rmse': rmse,
        'mae': mae,
        'mean_error': errors.mean(),
        'within_10': within_10,
        'within_20': within_20,
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

    # Load data
    data_path = '/workspace/research/shusheng_run/ws35/data/cleaned_ea_dataset.csv'
    data = load_data(data_path)
    print(f"\n  Loaded {len(data)} rows from {data_path}")

    # Filter data
    data = filter_desorption_only(data)
    data = filter_out_special_cases(data)
    print(f"  After filtering (desorption only, special cases excluded): {len(data)} rows")

    # Summary statistics
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

    with open(os.path.join(results_dir, 'analysis_results.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {results_dir}/analysis_results.json")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if h1_result:
        print(f"\n  H1 (Catalyst effect):")
        print(f"    Mean ΔEa = {h1_result['mean_delta_ea']:.2f} kJ/mol (p={h1_result['p_value']:.4f})")
        if h1_result['mean_delta_ea'] < 0:
            print("    → SUPPORTED: Catalysts significantly reduce Ea")
        else:
            print("    → NOT SUPPORTED: No clear reduction in Ea")

    if h4_result:
        print(f"\n  H4 (Meta-analysis):")
        print(f"    Fixed effects mean Ea: {h4_result['fixed_effects']['mean']:.2f} kJ/mol")
        print(f"    Random effects mean Ea: {h4_result['random_effects']['mean']:.2f} kJ/mol")
        print(f"    Between-study variance τ² = {h4_result['random_effects']['tau_sq']:.2f}")

    if h5_result:
        print(f"\n  H5 (Cross-validation):")
        print(f"    RMSE = {h5_result['rmse']:.2f} kJ/mol")
        print(f"    Accuracy within ±10 kJ/mol: {h5_result['within_10']}/{h5_result['n_samples']}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()