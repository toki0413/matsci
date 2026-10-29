#!/usr/bin/env python3
"""Meta-analysis on MgH2 Ea dataset.

Conducts 5 hypothesis tests on the MgH2 activation energy dataset.
"""

import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Set random seed
np.random.seed(20261029)

DATA_DIR = '/workspace/research/shusheng_run/ws35/data'
CSV_FILE = os.path.join(DATA_DIR, 'ea_dataset.csv')
RESULTS_FILE = 'results.json'
LOOP_LOG_FILE = 'loop_log.jsonl'


def load_data(filepath):
    """Load CSV data using csv module."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['Ea_kJ_per_mol'] = float(row['Ea_kJ_per_mol'])
            rows.append(row)
    return rows


def filter_desorption(rows):
    """Filter to only desorption rows."""
    return [r for r in rows if r.get('direction', '').strip().lower() == 'desorption']


def cluster_by_doi(rows):
    """Group rows by doi (paper)."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['doi']].append(row)
    return grouped


def compute_intra_paper_deltas(grouped):
    """Compute intra-paper DeltaEa (pure - catalyzed) for each paper.

    Returns: deltas array, paper_ids array (same length as deltas)
    """
    deltas = []
    paper_ids = []

    for doi, paper_rows in grouped.items():
        pure_rows = [r for r in paper_rows if r.get('material_class', '').startswith('pure')]
        catalyzed_rows = [r for r in paper_rows if r.get('material_class', '').startswith('catalyzed')]

        for pure_row in pure_rows:
            for cat_row in catalyzed_rows:
                delta = pure_row['Ea_kJ_per_mol'] - cat_row['Ea_kJ_per_mol']
                deltas.append(delta)
                paper_ids.append(doi)

    return np.array(deltas), np.array(paper_ids)


def dsl_random_effects(deltas):
    """DerSimonian-Laird random effects model for pooling deltas.

    With equal weights, the DL estimator simplifies to:
    - pooled mean = simple mean of deltas
    - tau_sq = between-study variance = max(0, (Q - df) / df)
    - SE = sqrt(tau_sq / n)
    - 95% CI = mean +/- 1.96 * SE

    Returns: pooled_mean, tau_sq, se, ci_lower, ci_upper
    """
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mean = np.mean(deltas)

    # Q statistic (sum of squared deviations)
    q_stat = np.sum((deltas - mean) ** 2)
    df = n - 1

    # DL estimator for between-study variance (equal weights)
    if df > 0:
        tau_sq = max(0.0, (q_stat - df) / df)
    else:
        tau_sq = 0.0

    # Standard error (using between-study variance only)
    se = np.sqrt(tau_sq / n) if n > 0 else 0.0

    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se

    return float(mean), float(tau_sq), float(se), float(ci_lower), float(ci_upper)


def cluster_bootstrap_ci(deltas, paper_ids, n_bootstrap=1000, alpha=0.05):
    """Compute 95% CI via cluster bootstrap (resampling papers).

    Resamples papers with replacement, computes mean of deltas from resampled papers,
    and uses the bootstrap distribution to get the CI.
    """
    unique_papers = np.unique(paper_ids)
    n_papers = len(unique_papers)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample papers with replacement
        sampled_papers = np.random.choice(unique_papers, size=n_papers, replace=True)

        # Collect deltas from sampled papers
        sampled_deltas = []
        for paper in sampled_papers:
            mask = paper_ids == paper
            sampled_deltas.extend(deltas[mask])

        if sampled_deltas:
            bootstrap_means.append(np.mean(sampled_deltas))

    bootstrap_means = np.array(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return float(ci_lower), float(ci_upper)


def permutation_test_catalyst_family(catalyzed_rows, n_permutations=10000):
    """Permutation test for catalyst family effect on catalyzed samples.

    Returns: p_value, eta_squared, observed_f
    """
    # Filter out rows with null/empty catalyst_family (none)
    valid_rows = [r for r in catalyzed_rows
                  if r.get('catalyst_family', '').strip() != 'none'
                  and r.get('catalyst_family', '').strip()]

    if len(valid_rows) < 2:
        return 1.0, 0.0, 0.0

    ea_values = np.array([r['Ea_kJ_per_mol'] for r in valid_rows])
    families = np.array([r['catalyst_family'] for r in valid_rows])
    unique_families = np.unique(families)
    n_families = len(unique_families)
    n = len(ea_values)

    # One-way ANOVA: compute F-statistic
    grand_mean = np.mean(ea_values)

    # Between-group sum of squares
    ss_between = 0.0
    for fam in unique_families:
        group_ea = ea_values[families == fam]
        group_mean = np.mean(group_ea)
        ss_between += len(group_ea) * (group_mean - grand_mean) ** 2

    # Within-group sum of squares
    ss_within = 0.0
    for fam in unique_families:
        group_ea = ea_values[families == fam]
        group_mean = np.mean(group_ea)
        ss_within += np.sum((group_ea - group_mean) ** 2)

    df_between = n_families - 1
    df_within = n - n_families

    if df_within > 0 and ss_between > 0:
        f_stat = (ss_between / df_between) / (ss_within / df_within)
    else:
        f_stat = 0.0

    # Eta-squared
    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    # Permutation test
    perm_f_stats = []
    ea_array = ea_values.copy()

    for _ in range(n_permutations):
        np.random.shuffle(ea_array)

        # Compute permuted F-stat
        perm_ss_between = 0.0
        for fam in unique_families:
            perm_group_ea = ea_array[families == fam]
            perm_group_mean = np.mean(perm_group_ea)
            perm_ss_between += len(perm_group_ea) * (perm_group_mean - grand_mean) ** 2

        perm_ss_within = 0.0
        for fam in unique_families:
            perm_group_ea = ea_array[families == fam]
            perm_group_mean = np.mean(perm_group_ea)
            perm_ss_within += np.sum((perm_group_ea - perm_group_mean) ** 2)

        if df_within > 0 and perm_ss_between > 0:
            perm_f = (perm_ss_between / df_between) / (perm_ss_within / df_within)
        else:
            perm_f = 0.0
        perm_f_stats.append(perm_f)

    perm_f_stats = np.array(perm_f_stats)
    p_value = float(np.mean(perm_f_stats >= f_stat))

    return p_value, eta_squared, float(f_stat)


def find_same_paper_method_pairs(grouped):
    """Find papers with both Kissinger and isothermal_Arrhenius methods.

    Returns: list of (doi, method1_ea, method2_ea, delta) pairs
    """
    pairs = []

    for doi, paper_rows in grouped.items():
        kissinger_rows = [r for r in paper_rows if r.get('method', '').strip() == 'Kissinger']
        isothermal_rows = [r for r in paper_rows if r.get('method', '').strip() == 'isothermal_Arrhenius']

        if kissinger_rows and isothermal_rows:
            for k_row in kissinger_rows:
                for i_row in isothermal_rows:
                    # Check if they refer to the same material
                    k_material = k_row['material']
                    i_material = i_row['material']

                    # Simple heuristic: match if materials share common substrings or are identical
                    if k_material.lower() in i_material.lower() or i_material.lower() in k_material.lower():
                        delta = abs(k_row['Ea_kJ_per_mol'] - i_row['Ea_kJ_per_mol'])
                        pairs.append((doi, k_row['Ea_kJ_per_mol'], i_row['Ea_kJ_per_mol'], delta))
                    elif k_material == i_material:
                        delta = abs(k_row['Ea_kJ_per_mol'] - i_row['Ea_kJ_per_mol'])
                        pairs.append((doi, k_row['Ea_kJ_per_mol'], i_row['Ea_kJ_per_mol'], delta))

    return pairs


def loso_cv_additive_model(rows):
    """Leave-one-subject-out CV with additive model (material_class + family + method).

    Returns: rmse, coverage, predictions, actuals
    """
    grouped = cluster_by_doi(rows)
    unique_papers = list(grouped.keys())

    predictions = []
    actuals = []
    residuals = []

    for leave_out_paper in unique_papers:
        train_rows = [r for r in rows if r['doi'] != leave_out_paper]
        test_rows = [r for r in rows if r['doi'] == leave_out_paper]

        if not train_rows or not test_rows:
            continue

        # Get unique categories from training data
        train_material_classes = sorted(set(r['material_class'] for r in train_rows))
        train_families = sorted(set(['none'] + [r['catalyst_family'] for r in train_rows if r['catalyst_family'] != 'none']))
        train_methods = sorted(set(r['method'] for r in train_rows))

        # Build design matrix and target for training
        X_rows = []
        y_rows = []

        for row in train_rows:
            mc_vec = [1 if mc == row['material_class'] else 0 for mc in train_material_classes]
            fam_vec = [1 if fam == row['catalyst_family'] else 0 for fam in train_families]
            method_vec = [1 if m == row['method'] else 0 for m in train_methods]

            X_rows.append([1] + mc_vec + fam_vec + method_vec)
            y_rows.append(row['Ea_kJ_per_mol'])

        X = np.array(X_rows)
        y = np.array(y_rows)

        # Solve for coefficients using least squares
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            coeffs = np.zeros(X.shape[1])

        # Predict on test rows
        for row in test_rows:
            mc_vec = [1 if mc == row['material_class'] else 0 for mc in train_material_classes]
            fam_vec = [1 if fam == row['catalyst_family'] else 0 for fam in train_families]
            method_vec = [1 if m == row['method'] else 0 for m in train_methods]

            x_test = np.array([1] + mc_vec + fam_vec + method_vec)
            pred = np.dot(coeffs, x_test)

            predictions.append(pred)
            actuals.append(row['Ea_kJ_per_mol'])
            residuals.append(row['Ea_kJ_per_mol'] - pred)

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    residuals = np.array(residuals)

    rmse = float(np.sqrt(np.mean((predictions - actuals) ** 2)))

    # Coverage: proportion of predictions within 95% CI of residuals
    if len(residuals) > 1:
        residual_std = np.std(residuals)
        ci_lower = actuals - 1.96 * residual_std
        ci_upper = actuals + 1.96 * residual_std
        coverage = float(np.mean((predictions >= ci_lower) & (predictions <= ci_upper)))
    else:
        coverage = 0.0

    return rmse, coverage, predictions, actuals


def predict_missing_combination(grouped, rows):
    """Predict Ea for missing combination (catalyzed, other, isothermal).

    Uses the additive model trained on all data.
    """
    all_material_classes = sorted(set(r['material_class'] for r in rows))
    all_families = sorted(set(['none'] + [r['catalyst_family'] for r in rows if r['catalyst_family'] != 'none']))
    all_methods = sorted(set(r['method'] for r in rows))

    # Build design matrix and target
    X_rows = []
    y_rows = []

    for row in rows:
        mc_vec = [1 if mc == row['material_class'] else 0 for mc in all_material_classes]
        fam_vec = [1 if fam == row['catalyst_family'] else 0 for fam in all_families]
        method_vec = [1 if m == row['method'] else 0 for m in all_methods]

        X_rows.append([1] + mc_vec + fam_vec + method_vec)
        y_rows.append(row['Ea_kJ_per_mol'])

    X = np.array(X_rows)
    y = np.array(y_rows)

    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    except:
        coeffs = np.zeros(X.shape[1])

    # Create prediction for (catalyzed, other, isothermal)
    mc_vec = [1 if mc == 'catalyzed_MgH2' else 0 for mc in all_material_classes]
    fam_vec = [1 if fam == 'other' else 0 for fam in all_families]
    method_vec = [1 if m == 'isothermal_Arrhenius' else 0 for m in all_methods]

    x_pred = np.array([1] + mc_vec + fam_vec + method_vec)
    pred = float(np.dot(coeffs, x_pred))

    # Estimate prediction interval using residual std
    y_pred_full = np.dot(X, coeffs)
    residuals = y - y_pred_full
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 1.0

    pi_lower = pred - 1.96 * residual_std
    pi_upper = pred + 1.96 * residual_std

    return pred, pi_lower, pi_upper


def run_cluster_bootstrap_iterations(n_iterations=50):
    """Run cluster bootstrap iterations and log results.

    Resamples papers, computes pooled DeltaEa and CI, and LOSO RMSE for each iteration.
    """
    # Load and filter data
    rows = load_data(CSV_FILE)
    des_rows = filter_desorption(rows)
    grouped = cluster_by_doi(des_rows)

    # Compute intra-paper deltas for full dataset
    deltas, paper_ids = compute_intra_paper_deltas(grouped)

    # Get unique papers for bootstrap
    unique_papers = np.unique(paper_ids)

    log_entries = []

    for loop_id in range(n_iterations):
        # Resample papers with replacement
        sampled_papers = np.random.choice(unique_papers, size=len(unique_papers), replace=True)

        # Collect deltas from sampled papers
        sampled_deltas = []
        sampled_paper_ids = []
        for paper in sampled_papers:
            mask = paper_ids == paper
            sampled_deltas.extend(deltas[mask])
            sampled_paper_ids.extend(paper_ids[mask])

        if sampled_deltas:
            sampled_deltas = np.array(sampled_deltas)
            sampled_paper_ids = np.array(sampled_paper_ids)

            # Compute DL pooled mean and CI
            mean, tau_sq, se, ci_lower, ci_upper = dsl_random_effects(sampled_deltas)

            # Compute LOSO CV on sampled data
            sampled_rows = []
            for paper in sampled_papers:
                sampled_rows.extend(grouped[paper])

            loso_rmse, _, _, _ = loso_cv_additive_model(sampled_rows)
        else:
            mean, ci_lower, ci_upper, loso_rmse = 0.0, 0.0, 0.0, 0.0

        entry = {
            'loop_id': int(loop_id),
            'delta_ea': float(mean),
            'ci_low': float(ci_lower),
            'ci_high': float(ci_upper),
            'loso_rmse': float(loso_rmse)
        }
        log_entries.append(entry)

        if (loop_id + 1) % 10 == 0:
            print(f"Completed {loop_id + 1}/{n_iterations} bootstrap iterations")

    # Write loop log
    with open(LOOP_LOG_FILE, 'w') as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"Loop log written to {LOOP_LOG_FILE}")
    return log_entries


def main():
    print("Loading data...")
    rows = load_data(CSV_FILE)
    print(f"Loaded {len(rows)} rows")

    print("Filtering to desorption rows...")
    des_rows = filter_desorption(rows)
    print(f"Filtered to {len(des_rows)} desorption rows")

    print("Grouping by doi...")
    grouped = cluster_by_doi(des_rows)
    print(f"Found {len(grouped)} unique papers")

    # H1: Intra-paper DeltaEa with random effects
    print("\n--- H1: Intra-paper DeltaEa (Random Effects) ---")
    deltas, paper_ids = compute_intra_paper_deltas(grouped)
    print(f"Computed {len(deltas)} intra-paper deltas")

    if len(deltas) > 0:
        mean, tau_sq, se, ci_lower, ci_upper = dsl_random_effects(deltas)
        print(f"Pooled DeltaEa (mean): {mean:.2f} kJ/mol")
        print(f"Between-study variance (tau^2): {tau_sq:.2f}")
        print(f"SE: {se:.2f}")
        print(f"95% CI (DL): [{ci_lower:.2f}, {ci_upper:.2f}]")

        # Cluster bootstrap CI
        ci_boot_lower, ci_boot_upper = cluster_bootstrap_ci(deltas, paper_ids, n_bootstrap=1000)
        print(f"Bootstrap 95% CI: [{ci_boot_lower:.2f}, {ci_boot_upper:.2f}]")
    else:
        print("No intra-paper deltas found")
        mean, ci_lower, ci_upper, ci_boot_lower, ci_boot_upper = 0.0, 0.0, 0.0, 0.0, 0.0

    # H2: Permutation test for catalyst family effect
    print("\n--- H2: Catalyst Family Effect (Permutation Test) ---")
    catalyzed_rows = [r for r in des_rows if r.get('material_class', '').startswith('catalyzed')]
    p_value, eta_squared, f_stat = permutation_test_catalyst_family(catalyzed_rows, n_permutations=10000)
    print(f"F-statistic: {f_stat:.4f}")
    print(f"eta-squared (eta^2): {eta_squared:.4f}")
    print(f"P-value (10,000 permutations): {p_value:.4f}")

    # H3: Same-paper method pairs
    print("\n--- H3: Same-Paper Method Comparison ---")
    method_pairs = find_same_paper_method_pairs(grouped)
    print(f"Found {len(method_pairs)} same-paper method pairs")

    if method_pairs:
        deltas_method = [p[3] for p in method_pairs]
        median_delta_method = float(np.median(deltas_method))
        print(f"Median |Delta_method|: {median_delta_method:.2f} kJ/mol")

        if len(deltas) > 0:
            median_delta_ea = float(np.median(deltas))
            print(f"Median DeltaEa: {median_delta_ea:.2f} kJ/mol")
            ratio = median_delta_method / median_delta_ea if median_delta_ea > 0 else 0.0
            print(f"Ratio |Delta_method|/DeltaEa: {ratio:.2f}")
        else:
            median_delta_ea = 0.0
            ratio = 0.0
    else:
        median_delta_method = 0.0
        median_delta_ea = 0.0
        ratio = 0.0

    # H4: LOSO CV with additive model
    print("\n--- H4: LOSO CV (Additive Model) ---")
    loso_rmse, coverage, predictions, actuals = loso_cv_additive_model(des_rows)
    print(f"LOSO RMSE: {loso_rmse:.2f} kJ/mol")
    print(f"Coverage (95% CI): {coverage:.4f}")

    # Baseline RMSE (predicting mean)
    baseline_rmse = float(np.sqrt(np.mean((np.mean(actuals) - actuals) ** 2)))
    print(f"Baseline RMSE (mean predictor): {baseline_rmse:.2f} kJ/mol")
    if baseline_rmse > 0:
        rmse_improvement = ((baseline_rmse - loso_rmse) / baseline_rmse * 100)
        print(f"RMSE improvement: {rmse_improvement:.1f}%")
    else:
        rmse_improvement = 0.0

    # H5: Predict missing combination
    print("\n--- H5: Predict Missing Combination (catalyzed, other, isothermal) ---")
    pred, pi_lower, pi_upper = predict_missing_combination(grouped, des_rows)
    print(f"Predicted Ea: {pred:.2f} kJ/mol")
    print(f"95% Prediction Interval: [{pi_lower:.2f}, {pi_upper:.2f}] kJ/mol")

    # Run cluster bootstrap iterations
    print("\n--- Running Cluster Bootstrap Iterations ---")
    log_entries = run_cluster_bootstrap_iterations(n_iterations=50)

    # Compute summary statistics from bootstrap
    delta_eas = np.array([e['delta_ea'] for e in log_entries])
    ci_lows = np.array([e['ci_low'] for e in log_entries])
    ci_highs = np.array([e['ci_high'] for e in log_entries])
    loso_rmses = np.array([e['loso_rmse'] for e in log_entries])

    print(f"Bootstrap DeltaEa mean: {np.mean(delta_eas):.2f} kJ/mol")
    print(f"Bootstrap DeltaEa std: {np.std(delta_eas):.2f} kJ/mol")
    print(f"Bootstrap LOSO RMSE mean: {np.mean(loso_rmses):.2f} kJ/mol")

    # Determine hypothesis statuses
    # H1: Reject null if CI does not include 0 (DeltaEa significantly different from 0)
    h1_significant = (ci_lower > 0) or (ci_upper < 0)

    # H2: Reject null if p < 0.05 (catalyst family has significant effect)
    h2_significant = p_value < 0.05

    # H3: Reject null if |Delta_method| > median DeltaEa (method differences larger than catalytic effect)
    h3_significant = (median_delta_method > median_delta_ea) and (median_delta_ea > 0)

    # H4: Reject null if model RMSE < baseline RMSE (model improves prediction)
    h4_significant = loso_rmse < baseline_rmse

    # Build results
    results = {
        'h1': {
            'description': 'Intra-paper DeltaEa (pure - catalyzed) with random effects',
            'n_pairs': int(len(deltas)),
            'pooled_delta_ea': mean,
            'tau_squared': tau_sq,
            'se': se,
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'ci_bootstrap_lower': ci_boot_lower,
            'ci_bootstrap_upper': ci_boot_upper,
            'significant': bool(h1_significant),
            'status': 'REJECTED' if h1_significant else 'NOT REJECTED'
        },
        'h2': {
            'description': 'Catalyst family effect on catalyzed samples (permutation test)',
            'n_catalyzed_samples': len(catalyzed_rows),
            'n_families': int(len(set(r['catalyst_family'] for r in catalyzed_rows if r['catalyst_family'] != 'none'))),
            'f_statistic': f_stat,
            'eta_squared': eta_squared,
            'p_value': p_value,
            'significant': bool(h2_significant),
            'status': 'REJECTED' if h2_significant else 'NOT REJECTED'
        },
        'h3': {
            'description': 'Same-paper method comparison (Kissinger vs isothermal)',
            'n_method_pairs': int(len(method_pairs)),
            'median_delta_method': median_delta_method,
            'median_delta_ea': median_delta_ea,
            'ratio': ratio,
            'significant': bool(h3_significant),
            'status': 'REJECTED' if h3_significant else 'NOT REJECTED'
        },
        'h4': {
            'description': 'LOSO CV with additive model (material_class + family + method)',
            'loso_rmse': loso_rmse,
            'baseline_rmse': baseline_rmse,
            'rmse_improvement': rmse_improvement,
            'coverage': coverage,
            'significant': bool(h4_significant),
            'status': 'REJECTED' if h4_significant else 'NOT REJECTED'
        },
        'h5': {
            'description': 'Predict Ea for missing combination (catalyzed, other, isothermal)',
            'predicted_ea': pred,
            'pi_95_lower': pi_lower,
            'pi_95_upper': pi_upper,
            'pi_width': pi_upper - pi_lower
        },
        'bootstrap_summary': {
            'n_iterations': 50,
            'mean_delta_ea': float(np.mean(delta_eas)),
            'std_delta_ea': float(np.std(delta_eas)),
            'mean_loso_rmse': float(np.mean(loso_rmses)),
            'std_loso_rmse': float(np.std(loso_rmses))
        },
        'metadata': {
            'seed': 20261029,
            'n_desorption_rows': len(des_rows),
            'n_papers': len(grouped),
            'n_catalyzed_rows': len(catalyzed_rows)
        }
    }

    # Write results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {RESULTS_FILE}")
    print(f"Loop log written to {LOOP_LOG_FILE}")

    # Print summary
    print("\n=== SUMMARY ===")
    print(f"H1 (DeltaEa significant): {'YES' if h1_significant else 'NO'} (CI: [{ci_lower:.2f}, {ci_upper:.2f}])")
    print(f"H2 (Catalyst family effect): {'YES' if h2_significant else 'NO'} (p={p_value:.4f})")
    print(f"H3 (Method effect > DeltaEa): {'YES' if h3_significant else 'NO'} (|Delta_method|={median_delta_method:.2f}, DeltaEa={median_delta_ea:.2f})")
    print(f"H4 (Model improves baseline): {'YES' if h4_significant else 'NO'} (RMSE={loso_rmse:.2f} vs {baseline_rmse:.2f})")
    print(f"H5 (Prediction): {pred:.2f} kJ/mol [95% PI: {pi_lower:.2f}, {pi_upper:.2f}]")


if __name__ == '__main__':
    main()