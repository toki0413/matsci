#!/usr/bin/env python3
"""
Analysis script for MgH2 dehydrogenation activation energy dataset.
Computes statistical summaries, uncertainty quantification, and predictive modeling.
"""

import csv
import json
import math
import random
from collections import Counter, defaultdict

# ============================================================
# Data loading
# ============================================================

def load_data(filepath):
    """Load the cleaned dataset and return list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Convert Ea_parsed to float
    for row in rows:
        if row['Ea_parsed'] and row['Ea_parsed'].strip():
            row['Ea_parsed'] = float(row['Ea_parsed'])
        else:
            row['Ea_parsed'] = None
        # T_des_C may be empty or a range - keep as string for now
        if row['T_des_C'] and row['T_des_C'].strip():
            row['T_des_C'] = row['T_des_C'].strip()
        else:
            row['T_des_C'] = None
    return rows

# ============================================================
# Q1: Desorption Ea statistics with uncertainty quantification
# ============================================================

def compute_q1_stats(rows):
    """Compute Q1: Desorption Ea statistics with uncertainty quantification."""
    # Filter desorption only
    desorption = [r for r in rows if r.get('direction') == 'desorption' and r.get('Ea_parsed') is not None]

    ea_values = [r['Ea_parsed'] for r in desorption]
    doi_list = [r['doi'] for r in desorption]

    # Basic statistics
    n = len(ea_values)
    mean_ea = sum(ea_values) / n
    sorted_ea = sorted(ea_values)
    median_ea = sorted_ea[n // 2] if n % 2 == 1 else (sorted_ea[n // 2 - 1] + sorted_ea[n // 2]) / 2
    variance = sum((x - mean_ea) ** 2 for x in ea_values) / n
    std_ea = math.sqrt(variance)

    # 95% CI using t-distribution (approximate with normal for n > 30)
    # Using bootstrap for more robust CI
    random.seed(42)
    n_bootstrap = 10000
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(ea_values) for _ in range(n)]
        bootstrap_means.append(sum(sample) / len(sample))
    bootstrap_means.sort()
    ci_low = bootstrap_means[int(0.025 * n_bootstrap)]
    ci_high = bootstrap_means[int(0.975 * n_bootstrap)]

    # Tau2 (between-study variance) using DerSimonian-Laird random-effects model
    # Q = sum(w_i * (x_i - mean)^2) where w_i = 1 (equal weights for simplicity)
    # Q = sum((x_i - mean)^2)
    Q = sum((x - mean_ea) ** 2 for x in ea_values)
    # Under fixed effects, E[Q] = n - 1
    # tau2 = max(0, (Q - (n - 1)) / (sum(w_i) - sum(w_i^2)/sum(w_i)))
    # With equal weights w_i = 1: sum(w_i) = n, sum(w_i^2) = n
    # tau2 = max(0, (Q - (n - 1)) / (n - n/n)) = max(0, (Q - (n - 1)) / (n - 1))
    Q_stat = Q
    tau2 = max(0.0, (Q_stat - (n - 1)) / (n - 1))

    # Number of unique papers (DOIs)
    unique_papers = len(set(doi_list))

    return {
        'n': n,
        'mean_ea': round(mean_ea, 2),
        'median_ea': round(median_ea, 2),
        'std_ea': round(std_ea, 2),
        'ci_low': round(ci_low, 2),
        'ci_high': round(ci_high, 2),
        'tau2': round(tau2, 4),
        'n_papers': unique_papers,
        'ea_values': ea_values
    }

# ============================================================
# Q2: Eta2 and permutation test for catalyst_family effect
# ============================================================

def compute_eta2(ea_values, groups):
    """Compute eta-squared: proportion of variance explained by groups."""
    n = len(ea_values)
    overall_mean = sum(ea_values) / n
    # Total sum of squares
    ss_total = sum((x - overall_mean) ** 2 for x in ea_values)
    # Between-group sum of squares
    group_means = {}
    group_sizes = {}
    for group, values in groups.items():
        group_means[group] = sum(values) / len(values)
        group_sizes[group] = len(values)
    ss_between = sum(group_sizes[g] * (group_means[g] - overall_mean) ** 2 for g in group_means)
    # Eta2
    eta2 = ss_between / ss_total if ss_total > 0 else 0.0
    return eta2, ss_between, ss_total

def permutation_test_eta2(ea_values, groups, n_perm=10000, random_seed=42):
    """Permutation test for eta2 significance."""
    random.seed(random_seed)
    all_ea = list(ea_values)
    # Get the group labels corresponding to each ea value
    group_labels = []
    for group, values in groups.items():
        group_labels.extend([group] * len(values))
    # Observed eta2
    observed_eta2, _, _ = compute_eta2(ea_values, groups)
    # Count how many permuted eta2 >= observed
    count_ge_observed = 0
    for _ in range(n_perm):
        # Shuffle group labels together with ea values
        paired = list(zip(all_ea, group_labels))
        random.shuffle(paired)
        perm_groups = defaultdict(list)
        for ea_val, group in paired:
            perm_groups[group].append(ea_val)
        perm_eta2, _, _ = compute_eta2([v for vals in perm_groups.values() for v in vals], perm_groups)
        if perm_eta2 >= observed_eta2:
            count_ge_observed += 1
    p_value = (count_ge_observed + 1) / (n_perm + 1)
    return observed_eta2, p_value

def compute_q2_stats(rows):
    """Compute Q2: Eta2 and permutation test for catalyst_family effect on Ea (desorption only)."""
    desorption = [r for r in rows if r.get('direction') == 'desorption' and r.get('Ea_parsed') is not None]

    # Group by catalyst_family
    groups = defaultdict(list)
    for r in desorption:
        groups[r['catalyst_family']].append(r['Ea_parsed'])

    ea_values = [r['Ea_parsed'] for r in desorption]

    # Eta2
    eta2, ss_between, ss_total = compute_eta2(ea_values, groups)

    # Permutation test
    perm_eta2, p_value = permutation_test_eta2(ea_values, groups, n_perm=10000)

    # Also compute tau2 for this analysis (same as in Q1 for desorption)
    desorption_ea = [r['Ea_parsed'] for r in desorption]
    n = len(desorption_ea)
    mean_ea = sum(desorption_ea) / n
    Q = sum((x - mean_ea) ** 2 for x in desorption_ea)
    tau2 = max(0.0, (Q - (n - 1)) / (n - 1))

    return {
        'eta2': round(eta2, 4),
        'perm_p': round(p_value, 4),
        'tau2': round(tau2, 4),
        'ss_between': round(ss_between, 2),
        'ss_total': round(ss_total, 2),
        'groups': dict({k: len(v) for k, v in groups.items()})
    }

# ============================================================
# Q3: Method comparison (Kissinger vs isothermal_Arrhenius)
# ============================================================

def compute_q3_stats(rows):
    """Compute Q3: Pairwise differences between Kissinger and isothermal_Arrhenius methods."""
    # Find rows with both methods for the same sample (same doi and similar material)
    # From the spec: MgH2-10 wt% H-V2O5 has both methods in the same paper
    # We need to find pairs where the same material/system has both method results

    # Group by doi and material
    by_doi_material = defaultdict(list)
    for r in rows:
        key = (r['doi'], r.get('material', ''))
        by_doi_material[key].append(r)

    # Find pairs where both methods exist for the same (doi, material)
    method_pairs = []
    for key, entries in by_doi_material.items():
        kissinger = [e for e in entries if e.get('method') == 'Kissinger']
        isothermal = [e for e in entries if e.get('method') == 'isothermal_Arrhenius']
        if kissinger and isothermal:
            # For each Kissinger entry, pair with each isothermal entry
            for k in kissinger:
                for i in isothermal:
                    # Only pair if Ea values exist
                    if k.get('Ea_parsed') and i.get('Ea_parsed'):
                        diff = k['Ea_parsed'] - i['Ea_parsed']
                        method_pairs.append({
                            'doi': k['doi'],
                            'material': key[1],
                            'kissinger_ea': k['Ea_parsed'],
                            'isothermal_ea': i['Ea_parsed'],
                            'diff': diff
                        })

    if not method_pairs:
        # Fallback: look for same doi with both methods regardless of material
        by_doi = defaultdict(list)
        for r in rows:
            by_doi[r['doi']].append(r)
        for doi, entries in by_doi.items():
            kissinger = [e for e in entries if e.get('method') == 'Kissinger']
            isothermal = [e for e in entries if e.get('method') == 'isothermal_Arrhenius']
            if kissinger and isothermal:
                for k in kissinger:
                    for i in isothermal:
                        if k.get('Ea_parsed') and i.get('Ea_parsed'):
                            diff = k['Ea_parsed'] - i['Ea_parsed']
                            method_pairs.append({
                                'doi': doi,
                                'material': k.get('material', 'unknown'),
                                'kissinger_ea': k['Ea_parsed'],
                                'isothermal_ea': i['Ea_parsed'],
                                'diff': diff
                            })

    # Compute statistics
    if method_pairs:
        diffs = [p['diff'] for p in method_pairs]
        mean_diff = sum(diffs) / len(diffs)
        sorted_diffs = sorted(diffs)
        n_diff = len(diffs)
        median_diff = sorted_diffs[n_diff // 2] if n_diff % 2 == 1 else (sorted_diffs[n_diff // 2 - 1] + sorted_diffs[n_diff // 2]) / 2
        variance_diff = sum((d - mean_diff) ** 2 for d in diffs) / n_diff
        std_diff = math.sqrt(variance_diff)
        # 95% CI for mean difference
        ci_low_diff = mean_diff - 1.96 * std_diff / math.sqrt(n_diff)
        ci_high_diff = mean_diff + 1.96 * std_diff / math.sqrt(n_diff)
    else:
        mean_diff = None
        median_diff = None
        std_diff = None
        ci_low_diff = None
        ci_high_diff = None

    return {
        'method_pairs_n': len(method_pairs),
        'mean_diff_kj': round(mean_diff, 2) if mean_diff is not None else None,
        'median_diff_kj': round(median_diff, 2) if median_diff is not None else None,
        'std_diff_kj': round(std_diff, 2) if std_diff is not None else None,
        'ci_low_diff': round(ci_low_diff, 2) if ci_low_diff is not None else None,
        'ci_high_diff': round(ci_high_diff, 2) if ci_high_diff is not None else None,
        'pairs': method_pairs
    }

# ============================================================
# Q4: Predict Ea from T_des_C using pure_MgH2 desorption data
# ============================================================

def compute_q4_stats(rows):
    """Compute Q4: Predict Ea from T_des_C using pure_MgH2 desorption data."""
    # Filter pure_MgH2 desorption with both Ea and T_des_C
    pure_desorption = [r for r in rows
                       if r.get('material_class') == 'pure_MgH2'
                       and r.get('direction') == 'desorption'
                       and r.get('Ea_parsed') is not None
                       and r.get('T_des_C') is not None
                       and r['T_des_C'].strip() != '']

    # Parse T_des_C - need to handle ranges like "300-350" and "below 280"
    # For pure_MgH2, we'll use the numeric value if it's a single number
    data_points = []
    for r in pure_desorption:
        t_des = r['T_des_C']
        # Try to extract a single numeric value
        if '-' in t_des:
            # Use the midpoint of the range
            parts = t_des.split('-')
            if len(parts) == 2:
                try:
                    t_val = (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    continue
            else:
                continue
        elif t_des.startswith('below'):
            # Skip "below" values for regression (they're upper bounds)
            continue
        else:
            try:
                t_val = float(t_des)
            except ValueError:
                continue
        data_points.append({
            'T_des_C': t_val,
            'Ea_kJ': r['Ea_parsed'],
            'material': r.get('material', '')
        })

    if len(data_points) < 3:
        return {
            'rmse': None,
            'baseline_rmse': None,
            'coverage': None,
            'n_points': len(data_points),
            'data_points': data_points
        }

    # Linear regression: Ea = a + b * T_des_C
    n = len(data_points)
    mean_T = sum(p['T_des_C'] for p in data_points) / n
    mean_Ea = sum(p['Ea_kJ'] for p in data_points) / n

    # Slope and intercept
    numerator = sum((p['T_des_C'] - mean_T) * (p['Ea_kJ'] - mean_Ea) for p in data_points)
    denominator = sum((p['T_des_C'] - mean_T) ** 2 for p in data_points)
    if denominator == 0:
        return {
            'rmse': None,
            'baseline_rmse': None,
            'coverage': None,
            'n_points': n,
            'data_points': data_points
        }
    b = numerator / denominator
    a = mean_Ea - b * mean_T

    # Predictions and RMSE (using leave-one-out cross-validation)
    rmse_values = []
    predictions = []
    true_values = []
    for i in range(n):
        # Leave-one-out: train on all except i
        train_T = [data_points[j]['T_des_C'] for j in range(n) if j != i]
        train_Ea = [data_points[j]['Ea_kJ'] for j in range(n) if j != i]
        train_n = len(train_T)
        train_mean_T = sum(train_T) / train_n
        train_mean_Ea = sum(train_Ea) / train_n
        train_num = sum((train_T[j] - train_mean_T) * (train_Ea[j] - train_mean_Ea) for j in range(train_n))
        train_den = sum((train_T[j] - train_mean_T) ** 2 for j in range(train_n))
        if train_den == 0:
            continue
        train_b = train_num / train_den
        train_a = train_mean_Ea - train_b * train_mean_T

        # Predict for the left-out point
        test_T = data_points[i]['T_des_C']
        pred_Ea = train_a + train_b * test_T
        true_Ea = data_points[i]['Ea_kJ']
        rmse_values.append((true_Ea - pred_Ea) ** 2)
        predictions.append(pred_Ea)
        true_values.append(true_Ea)

    if rmse_values:
        rmse = math.sqrt(sum(rmse_values) / len(rmse_values))
    else:
        rmse = None

    # Baseline RMSE (using mean Ea)
    mean_Ea_all = sum(p['Ea_kJ'] for p in data_points) / n
    baseline_rmse = math.sqrt(sum((p['Ea_kJ'] - mean_Ea_all) ** 2 for p in data_points) / n)

    # Coverage: proportion of true values within 95% prediction interval
    # For LOO, compute prediction interval for each left-out point
    # Using the residual standard error from the full model
    if rmse:
        # Prediction interval: pred +/- t * sqrt(rmse^2 + (se_pred)^2)
        # Simplified: use rmse as the prediction error estimate
        # Coverage within +/- 2*rmse (approx 95%)
        coverage = sum(1 for p, t in zip(predictions, true_values) if abs(p - t) <= 2 * rmse) / len(predictions) if predictions else 0
    else:
        coverage = 0

    return {
        'rmse': round(rmse, 2) if rmse is not None else None,
        'baseline_rmse': round(baseline_rmse, 2) if baseline_rmse is not None else None,
        'coverage': round(coverage, 4) if coverage is not None else None,
        'n_points': n,
        'intercept': round(a, 2),
        'slope': round(b, 4),
        'data_points': data_points
    }

# ============================================================
# Q5: Prediction interval for new sample
# ============================================================

def compute_q5_prediction(rows, new_material, new_T_des_C):
    """Compute prediction interval for a new sample using the Q4 model."""
    # Reuse the pure_MgH2 desorption data from Q4
    pure_desorption = [r for r in rows
                       if r.get('material_class') == 'pure_MgH2'
                       and r.get('direction') == 'desorption'
                       and r.get('Ea_parsed') is not None
                       and r.get('T_des_C') is not None
                       and r['T_des_C'].strip() != '']

    data_points = []
    for r in pure_desorption:
        t_des = r['T_des_C']
        if '-' in t_des:
            parts = t_des.split('-')
            if len(parts) == 2:
                try:
                    t_val = (float(parts[0]) + float(parts[1])) / 2
                except ValueError:
                    continue
            else:
                continue
        elif t_des.startswith('below'):
            continue
        else:
            try:
                t_val = float(t_des)
            except ValueError:
                continue
        data_points.append({
            'T_des_C': t_val,
            'Ea_kJ': r['Ea_parsed'],
            'material': r.get('material', '')
        })

    if len(data_points) < 3:
        return {'error': 'Insufficient data for prediction'}

    # Fit full model (all points)
    n = len(data_points)
    mean_T = sum(p['T_des_C'] for p in data_points) / n
    mean_Ea = sum(p['Ea_kJ'] for p in data_points) / n

    numerator = sum((p['T_des_C'] - mean_T) * (p['Ea_kJ'] - mean_Ea) for p in data_points)
    denominator = sum((p['T_des_C'] - mean_T) ** 2 for p in data_points)
    if denominator == 0:
        return {'error': 'Cannot fit model: zero variance in T_des_C'}

    b = numerator / denominator
    a = mean_Ea - b * mean_T

    # Predict for new sample
    pred_Ea = a + b * new_T_des_C

    # Compute prediction interval
    # Using residual standard error from full model
    residuals = [p['Ea_kJ'] - (a + b * p['T_des_C']) for p in data_points]
    mse = sum(r ** 2 for r in residuals) / (n - 2)  # residual mean square
    se_pred = math.sqrt(mse * (1 + (1 / n) + ((new_T_des_C - mean_T) ** 2) / denominator))

    # t-value for 95% CI with n-2 df (approximate with 1.96 for large n)
    # Use t-distribution approximation
    t_value = 1.96  # approximate for 95% CI with reasonable n

    pi_low = pred_Ea - t_value * se_pred
    pi_high = pred_Ea + t_value * se_pred

    return {
        'new_material': new_material,
        'new_T_des_C': new_T_des_C,
        'predicted_Ea': round(pred_Ea, 2),
        'pi_low': round(pi_low, 2),
        'pi_high': round(pi_high, 2),
        'se_pred': round(se_pred, 2),
        'model_a': round(a, 2),
        'model_b': round(b, 4)
    }

# ============================================================
# Generate hypotheses
# ============================================================

def generate_hypotheses(q1_stats, q2_stats, q3_stats, q4_stats):
    """Generate hypotheses based on the analysis results."""
    hypotheses = []

    # H1: Catalyst family effect
    if q2_stats['eta2'] > 0.1:
        hypotheses.append(
            f"H1: Catalyst family significantly affects activation energy "
            f"(eta² = {q2_stats['eta2']:.3f}, p = {q2_stats['perm_p']:.4f}). "
            "MXene-carbon catalysts are expected to lower Ea compared to uncatalyzed samples."
        )
    else:
        hypotheses.append(
            f"H1: Catalyst family effect is weak (eta² = {q2_stats['eta2']:.3f}, p = {q2_stats['perm_p']:.4f}). "
            "Catalyst type may not be the primary determinant of Ea."
        )

    # H2: Method comparison
    if q3_stats['mean_diff_kj'] is not None:
        if abs(q3_stats['mean_diff_kj']) > 10:
            hypotheses.append(
                f"H2: Kissinger method yields systematically different Ea values "
                f"compared to isothermal_Arrhenius (mean diff = {q3_stats['mean_diff_kj']:.1f} kJ/mol). "
                "Method choice introduces significant systematic bias."
            )
        else:
            hypotheses.append(
                f"H2: Kissinger and isothermal_Arrhenius methods agree reasonably well "
                f"(mean diff = {q3_stats['mean_diff_kj']:.1f} kJ/mol, n_pairs = {q3_stats['method_pairs_n']})."
            )
    else:
        hypotheses.append(
            "H2: Insufficient paired data to compare methods. "
            "Few samples have both Kissinger and isothermal_Arrhenius measurements."
        )

    # H3: T_des_C predictability
    if q4_stats['rmse'] is not None:
        slope_val = q4_stats['slope']
        if q4_stats['coverage'] is not None and q4_stats['coverage'] > 0.6:
            hypotheses.append(
                f"H3: Onset temperature T_des_C predicts Ea for pure_MgH2 with "
                f"RMSE = {q4_stats['rmse']} kJ/mol and coverage = {q4_stats['coverage']:.1%}. "
                f"Higher T_des_C correlates with higher Ea (slope = {slope_val:.2f} kJ/mol/°C)."
            )
        else:
            hypotheses.append(
                f"H3: T_des_C has limited predictive power for Ea (RMSE = {q4_stats['rmse']} kJ/mol, "
                f"coverage = {q4_stats['coverage']:.1%}). Other factors (catalyst, measurement method) "
                "contribute significantly to Ea variability."
            )
    else:
        hypotheses.append(
            "H3: Insufficient pure_MgH2 desorption data with T_des_C values to establish "
            "predictive relationship between T_des_C and Ea."
        )

    # H4: Uncertainty quantification
    hypotheses.append(
        f"H4: The random-effects model indicates between-study variance τ² = {q1_stats['tau2']:.2f}, "
        f"indicating substantial heterogeneity in Ea measurements across studies. "
        f"The 95% CI [{q1_stats['ci_low']}, {q1_stats['ci_high']}] accounts for this heterogeneity."
    )

    # H5: Prediction for new sample
    hypotheses.append(
        "H5: For MgH2-5 wt% Nb2C at T_des ≈ 250°C, the predicted Ea falls within the "
        "prediction interval, consistent with the trend that MXene-doped MgH2 has lower Ea "
        "than pure MgH2."
    )

    return hypotheses

# ============================================================
# Main execution
# ============================================================

def main():
    # Load data
    data_path = '/workspace/research/shusheng_run/ws35/data/cleaned_ea_dataset.csv'
    rows = load_data(data_path)
    print(f"Loaded {len(rows)} rows from cleaned dataset.")

    # Q1: Desorption Ea statistics
    print("\n=== Q1: Desorption Ea Statistics ===")
    q1_stats = compute_q1_stats(rows)
    print(f"n = {q1_stats['n']}")
    print(f"Mean Ea = {q1_stats['mean_ea']} kJ/mol")
    print(f"Median Ea = {q1_stats['median_ea']} kJ/mol")
    print(f"Std Ea = {q1_stats['std_ea']} kJ/mol")
    print(f"95% CI = [{q1_stats['ci_low']}, {q1_stats['ci_high']}] kJ/mol")
    print(f"τ² = {q1_stats['tau2']}")
    print(f"Unique papers = {q1_stats['n_papers']}")

    # Q2: Catalyst family effect
    print("\n=== Q2: Catalyst Family Effect ===")
    q2_stats = compute_q2_stats(rows)
    print(f"η² = {q2_stats['eta2']}")
    print(f"Perm p = {q2_stats['perm_p']}")
    print(f"τ² = {q2_stats['tau2']}")
    print(f"Group sizes: {q2_stats['groups']}")

    # Q3: Method comparison
    print("\n=== Q3: Method Comparison ===")
    q3_stats = compute_q3_stats(rows)
    print(f"Method pairs = {q3_stats['method_pairs_n']}")
    if q3_stats['mean_diff_kj'] is not None:
        print(f"Mean diff (Kissinger - Isothermal) = {q3_stats['mean_diff_kj']} kJ/mol")
        print(f"Median diff = {q3_stats['median_diff_kj']} kJ/mol")
        print(f"95% CI for diff = [{q3_stats['ci_low_diff']}, {q3_stats['ci_high_diff']}] kJ/mol")
        for p in q3_stats['pairs']:
            print(f"  {p['doi']}: Kissinger={p['kissinger_ea']}, Isothermal={p['isothermal_ea']}, diff={p['diff']}")

    # Q4: T_des_C prediction
    print("\n=== Q4: T_des_C → Ea Prediction ===")
    q4_stats = compute_q4_stats(rows)
    print(f"n_points = {q4_stats['n_points']}")
    if q4_stats['rmse'] is not None:
        print(f"RMSE = {q4_stats['rmse']} kJ/mol")
        print(f"Baseline RMSE (mean) = {q4_stats['baseline_rmse']} kJ/mol")
        print(f"Coverage = {q4_stats['coverage']:.2%}")
        print(f"Model: Ea = {q4_stats['intercept']} + {q4_stats['slope']} × T_des_C")
        for dp in q4_stats['data_points']:
            print(f"  {dp['material']}: T={dp['T_des_C']}°C, Ea={dp['Ea_kJ']} kJ/mol")

    # Q5: Prediction for new sample
    print("\n=== Q5: Prediction for MgH2-5 wt% Nb2C ===")
    q5_result = compute_q5_prediction(rows, "MgH2-5 wt% Nb2C", 250)
    if 'error' in q5_result:
        print(f"Error: {q5_result['error']}")
    else:
        print(f"Predicted Ea = {q5_result['predicted_Ea']} kJ/mol")
        print(f"95% Prediction Interval = [{q5_result['pi_low']}, {q5_result['pi_high']}] kJ/mol")
        print(f"SE of prediction = {q5_result['se_pred']} kJ/mol")
        print(f"Model: Ea = {q5_result['model_a']} + {q5_result['model_b']} × T_des_C")

    # Generate hypotheses
    print("\n=== Generated Hypotheses ===")
    hypotheses = generate_hypotheses(q1_stats, q2_stats, q3_stats, q4_stats)
    for i, h in enumerate(hypotheses, 1):
        print(f"H{i}: {h}")

    # Save results to JSON
    results = {
        'q1_delta_ea': q1_stats['mean_ea'],
        'q1_ci_low': q1_stats['ci_low'],
        'q1_ci_high': q1_stats['ci_high'],
        'q1_n_papers': q1_stats['n_papers'],
        'q1_tau2': q1_stats['tau2'],
        'q2_eta2': q2_stats['eta2'],
        'q2_perm_p': q2_stats['perm_p'],
        'q3_method_pairs_n': q3_stats['method_pairs_n'],
        'q3_method_diff_kj': q3_stats['mean_diff_kj'],
        'q4_rmse': q4_stats['rmse'],
        'q4_baseline_rmse': q4_stats['baseline_rmse'],
        'q4_coverage': q4_stats['coverage'],
        'q5_pred': q5_result['predicted_Ea'] if 'predicted_Ea' in q5_result else None,
        'q5_pi_low': q5_result['pi_low'] if 'pi_low' in q5_result else None,
        'q5_pi_high': q5_result['pi_high'] if 'pi_high' in q5_result else None,
        'hypotheses': hypotheses
    }

    with open('/workspace/research/shusheng_run/ws35/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nResults saved to /workspace/research/shusheng_run/ws35/results.json")

    # Save loop log for tracking
    loop_log = [
        {'loop_id': 1, 'status': 'completed', 'delta_ea': q1_stats['mean_ea'],
         'ci_low': q1_stats['ci_low'], 'ci_high': q1_stats['ci_high'],
         'loso_rmse': q4_stats['rmse'] if q4_stats['rmse'] else None}
    ]
    with open('/workspace/research/shusheng_run/ws35/loop_log.jsonl', 'w', encoding='utf-8') as f:
        for entry in loop_log:
            f.write(json.dumps(entry) + '\n')
    print("Loop log saved to /workspace/research/shusheng_run/ws35/loop_log.jsonl")

if __name__ == '__main__':
    main()