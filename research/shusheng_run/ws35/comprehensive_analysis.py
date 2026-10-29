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
import random


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def filter_desorption_only(rows):
    return [r for r in rows if r.get('direction', '').lower() == 'desorption']


def get_paper_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['doi']].append(r)
    return groups


# ============================================================
# H1: Within-paper paired ΔEa, DerSimonian-Laird random effects, bootstrap CI
# ============================================================

def compute_pairwise_delta_ea(paper_rows):
    catalyzed = [r for r in paper_rows if r.get('material_class') == 'catalyzed_MgH2']
    pure = [r for r in paper_rows if r.get('material_class') == 'pure_MgH2']
    deltas = []
    for c in catalyzed:
        for p in pure:
            delta = float(c['Ea_kJ_per_mol']) - float(p['Ea_kJ_per_mol'])
            deltas.append(delta)
    return deltas


def h1_analysis(rows):
    paper_groups = get_paper_groups(rows)
    all_deltas = []
    paper_deltas = {}
    for doi, paper_rows in paper_groups.items():
        deltas = compute_pairwise_delta_ea(paper_rows)
        if deltas:
            paper_deltas[doi] = deltas
            all_deltas.extend(deltas)
    if not all_deltas:
        return None
    all_deltas = np.array(all_deltas)
    mean_delta = float(np.mean(all_deltas))
    std_delta = float(np.std(all_deltas, ddof=1))
    n_pairs = len(all_deltas)
    se = std_delta / np.sqrt(n_pairs)
    ci_lower = mean_delta - 1.96 * se
    ci_upper = mean_delta + 1.96 * se
    # DerSimonian-Laird random effects
    paper_means = []
    paper_sizes = []
    for doi, deltas in paper_deltas.items():
        paper_means.append(float(np.mean(deltas)))
        paper_sizes.append(len(deltas))
    paper_means = np.array(paper_means)
    paper_sizes = np.array(paper_sizes)
    Q = float(np.sum(paper_sizes * (paper_means - np.mean(paper_means))**2))
    C = float(np.sum(paper_sizes) - (np.sum(paper_sizes**2) / np.sum(paper_sizes)))
    tau_sq = max(0.0, (Q - (len(paper_means) - 1)) / C)
    if tau_sq == 0:
        weights = paper_sizes
    else:
        within_var = 1.0 / paper_sizes
        between_var = tau_sq
        weights = 1.0 / (within_var + between_var)
        weights = weights / np.sum(weights) * len(paper_means)
    weighted_mean = float(np.sum(weights * paper_means) / np.sum(weights))
    se_re = float(1.0 / np.sqrt(np.sum(weights)))
    ci_re_lower = weighted_mean - 1.96 * se_re
    ci_re_upper = weighted_mean + 1.96 * se_re
    # Bootstrap CI (cluster resampling by paper)
    np.random.seed(42)
    n_bootstrap = 50
    bootstrap_means = []
    doi_list = list(paper_deltas.keys())
    for _ in range(n_bootstrap):
        sampled_dois = random.choices(doi_list, k=len(doi_list))
        boot_deltas = []
        for doi in sampled_dois:
            boot_deltas.extend(paper_deltas[doi])
        if boot_deltas:
            bootstrap_means.append(float(np.array(boot_deltas).mean()))
    bootstrap_means = np.array(bootstrap_means)
    boot_ci_lower = float(np.percentile(bootstrap_means, 2.5))
    boot_ci_upper = float(np.percentile(bootstrap_means, 97.5))
    # Paired t-test
    t_stat, p_value = stats.ttest_1samp(all_deltas, 0)
    h1_result = {
        'n_papers_with_pairs': int(len(paper_deltas)),
        'n_total_pairs': int(n_pairs),
        'mean_delta_ea': mean_delta,
        'std_delta_ea': std_delta,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'dl_random_effects_mean': weighted_mean,
        'dl_random_effects_ci_lower': ci_re_lower,
        'dl_random_effects_ci_upper': ci_re_upper,
        'bootstrap_ci_lower': boot_ci_lower,
        'bootstrap_ci_upper': boot_ci_upper,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'supported': bool(mean_delta < 0 and p_value < 0.05)
    }
    return h1_result


# ============================================================
# H2: η² for catalyst family, permutation test
# ============================================================

def h2_analysis(rows):
    families = defaultdict(list)
    for r in rows:
        family = r.get('catalyst_family', 'none')
        families[family].append(float(r['Ea_kJ_per_mol']))
    family_names = list(families.keys())
    n_families = len(family_names)
    all_eas = []
    group_labels = []
    for family, eas in families.items():
        all_eas.extend(eas)
        group_labels.extend([family] * len(eas))
    all_eas = np.array(all_eas)
    grand_mean = np.mean(all_eas)
    ss_between = sum(len(families[f]) * (np.mean(families[f]) - grand_mean)**2 for f in family_names)
    ss_total = sum((e - grand_mean)**2 for e in all_eas)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0
    if n_families > 1:
        n_total = len(all_eas)
        df_between = n_families - 1
        df_within = n_total - n_families
        ms_between = ss_between / df_between
        ss_within = ss_total - ss_between
        ms_within = ss_within / df_within
        f_stat = float(ms_between / ms_within) if ms_within > 0 else 0.0
        p_value = float(1.0 - stats.f.cdf(f_stat, df_between, df_within))
    else:
        f_stat, p_value = 0.0, 1.0
    # Permutation test
    np.random.seed(42)
    n_permutations = 1000
    perm_eta_squares = []
    for _ in range(n_permutations):
        shuffled_labels = np.random.permutation(group_labels)
        perm_groups = defaultdict(list)
        for i, label in enumerate(shuffled_labels):
            perm_groups[label].append(all_eas[i])
        perm_ss_between = sum(len(perm_groups[f]) * (np.mean(perm_groups[f]) - grand_mean)**2 for f in family_names if f in perm_groups)
        perm_eta = perm_ss_between / ss_total if ss_total > 0 else 0.0
        perm_eta_squares.append(perm_eta)
    perm_eta_squares = np.array(perm_eta_squares)
    perm_p_value = float(np.mean(perm_eta_squares >= eta_squared))
    h2_result = {
        'eta_squared': eta_squared,
        'f_statistic': f_stat,
        'p_value_anova': p_value,
        'p_value_permutation': perm_p_value,
        'n_families': n_families,
        'supported': bool(eta_squared > 0.1 and perm_p_value < 0.05)
    }
    return h2_result


# ============================================================
# H3: Method pairs (Kissinger vs isothermal_Arrhenius), compare |Δ_method| to median ΔEa
# ============================================================

def h3_analysis(rows):
    material_paper_groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        material = r['material']
        doi = r['doi']
        method = r['method']
        ea = float(r['Ea_kJ_per_mol'])
        material_paper_groups[material][doi].append((method, ea))
    method_deltas = []
    method_pairs_info = []
    for material, papers in material_paper_groups.items():
        for doi, methods in papers.items():
            kissinger_ea = None
            isothermal_ea = None
            for m, ea in methods:
                if m == 'Kissinger':
                    kissinger_ea = ea
                elif m == 'isothermal_Arrhenius':
                    isothermal_ea = ea
            if kissinger_ea is not None and isothermal_ea is not None:
                delta = abs(kissinger_ea - isothermal_ea)
                method_deltas.append(delta)
                method_pairs_info.append({
                    'material': material,
                    'doi': doi,
                    'kissinger_ea': kissinger_ea,
                    'isothermal_ea': isothermal_ea,
                    'delta_ea': delta
                })
    if not method_deltas:
        return None
    method_deltas = np.array(method_deltas)
    median_method_delta = float(np.median(method_deltas))
    # Get median ΔEa across all catalyzed vs pure comparisons
    paper_groups = get_paper_groups(rows)
    all_within_paper_deltas = []
    for doi, paper_rows in paper_groups.items():
        deltas = compute_pairwise_delta_ea(paper_rows)
        all_within_paper_deltas.extend(deltas)
    if all_within_paper_deltas:
        median_ea_reduction = float(np.median(np.array(all_within_paper_deltas)))
        median_ea_reduction = abs(median_ea_reduction)
    else:
        median_ea_reduction = float(np.array([r['Ea_kJ_per_mol'] for r in rows]).mean())
    mean_method_delta = float(np.mean(method_deltas))
    ratio = mean_method_delta / median_ea_reduction if median_ea_reduction > 0 else 0.0
    t_stat, p_value = stats.ttest_1samp(method_deltas, 0)
    h3_result = {
        'n_method_pairs': len(method_deltas),
        'mean_absolute_method_delta': mean_method_delta,
        'median_absolute_method_delta': median_method_delta,
        'median_ea_reduction': median_ea_reduction,
        'ratio_method_to_ea': ratio,
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'supported': bool(ratio > 1 and p_value < 0.05),
        'pair_details': method_pairs_info
    }
    return h3_result


# ============================================================
# H4: LOSO cross-validation with material_class + catalyst_family + method features
# ============================================================

def h4_analysis(rows):
    material_classes = sorted(list(set(r['material_class'] for r in rows)))
    catalyst_families = sorted(list(set(r.get('catalyst_family', 'none') for r in rows)))
    methods = sorted(list(set(r['method'] for r in rows)))
    n_samples = len(rows)
    n_features = len(material_classes) + len(catalyst_families) + len(methods)
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    for i, r in enumerate(rows):
        y[i] = float(r['Ea_kJ_per_mol'])
        mc_idx = material_classes.index(r['material_class'])
        X[i, mc_idx] = 1
        cf = r.get('catalyst_family', 'none')
        if cf in catalyst_families:
            cf_idx = len(material_classes) + catalyst_families.index(cf)
            X[i, cf_idx] = 1
        m_idx = len(material_classes) + len(catalyst_families) + methods.index(r['method'])
        X[i, m_idx] = 1
    paper_groups = get_paper_groups(rows)
    doi_list = sorted(paper_groups.keys())
    loso_predictions = []
    loso_true = []
    loso_paper_ids = []
    for leave_out_doi in doi_list:
        train_indices = []
        test_indices = []
        for i, r in enumerate(rows):
            if r['doi'] == leave_out_doi:
                test_indices.append(i)
            else:
                train_indices.append(i)
        if len(train_indices) < 2:
            continue
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        alpha = 1.0
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        try:
            coeffs = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[1]), Xty)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        y_pred = X_test @ coeffs
        loso_predictions.extend(y_pred)
        loso_true.extend(y_test)
        loso_paper_ids.extend([leave_out_doi] * len(y_pred))
    if len(loso_predictions) < 2:
        return None
    loso_predictions = np.array(loso_predictions)
    loso_true = np.array(loso_true)
    fixed_mean = float(np.mean(loso_predictions))
    fixed_se = float(np.std(loso_predictions, ddof=1) / np.sqrt(len(loso_predictions)))
    # Random effects meta-analysis (DL)
    paper_preds = defaultdict(list)
    paper_true = defaultdict(list)
    for pid, pred, true in zip(loso_paper_ids, loso_predictions, loso_true):
        paper_preds[pid].append(pred)
        paper_true[pid].append(true)
    paper_means = []
    paper_sizes = []
    for pid in paper_preds:
        paper_means.append(float(np.mean(paper_preds[pid])))
        paper_sizes.append(len(paper_preds[pid]))
    paper_means = np.array(paper_means)
    paper_sizes = np.array(paper_sizes)
    Q = float(np.sum(paper_sizes * (paper_means - np.mean(paper_means))**2))
    C = float(np.sum(paper_sizes) - (np.sum(paper_sizes**2) / np.sum(paper_sizes)))
    tau_sq = max(0.0, (Q - (len(paper_means) - 1)) / C)
    if tau_sq == 0:
        weights = paper_sizes
    else:
        within_var = 1.0 / paper_sizes
        between_var = tau_sq
        weights = 1.0 / (within_var + between_var)
        weights = weights / np.sum(weights) * len(paper_means)
    random_mean = float(np.sum(weights * paper_means) / np.sum(weights))
    random_se = float(1.0 / np.sqrt(np.sum(weights)))
    rmse = float(np.sqrt(np.mean((loso_predictions - loso_true)**2)))
    within_10 = int(np.sum(np.abs(loso_predictions - loso_true) <= 10))
    h4_result = {
        'n_samples': n_samples,
        'n_papers': len(doi_list),
        'n_features': n_features,
        'fixed_effects_mean': fixed_mean,
        'fixed_effects_se': fixed_se,
        'random_effects_mean': random_mean,
        'random_effects_se': random_se,
        'between_study_variance': float(tau_sq),
        'rmse': rmse,
        'within_10_kJmol': within_10,
        'n_holdout': len(loso_predictions),
        'supported': bool(rmse < 20)
    }
    return h4_result


# ============================================================
# H5: Extrapolation prediction for empty cell
# ============================================================

def h5_analysis(rows):
    material_classes = sorted(list(set(r['material_class'] for r in rows)))
    catalyst_families = sorted(list(set(r.get('catalyst_family', 'none') for r in rows)))
    methods = sorted(list(set(r['method'] for r in rows)))
    n_samples = len(rows)
    n_features = len(material_classes) + len(catalyst_families) + len(methods)
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples)
    for i, r in enumerate(rows):
        y[i] = float(r['Ea_kJ_per_mol'])
        mc_idx = material_classes.index(r['material_class'])
        X[i, mc_idx] = 1
        cf = r.get('catalyst_family', 'none')
        if cf in catalyst_families:
            cf_idx = len(material_classes) + catalyst_families.index(cf)
            X[i, cf_idx] = 1
        m_idx = len(material_classes) + len(catalyst_families) + methods.index(r['method'])
        X[i, m_idx] = 1
    alpha = 1.0
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coeffs = np.linalg.solve(XtX + alpha * np.eye(XtX.shape[1]), Xty)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    # Hypothetical prediction
    hypoth_X = np.zeros(n_features)
    hypoth_X[material_classes.index('catalyzed_MgH2')] = 1
    novel_cf_idx = len(material_classes) + len(catalyst_families)
    hypoth_X[novel_cf_idx] = 1
    hypoth_X[len(material_classes) + len(catalyst_families) + methods.index('Kissinger')] = 1
    predicted_ea = float(hypoth_X @ coeffs)
    y_pred_all = X @ coeffs
    rmse_all = float(np.sqrt(np.mean((y_pred_all - y)**2)))
    ea_min, ea_max = float(np.min(y)), float(np.max(y))
    reasonable = bool(ea_min - 50 <= predicted_ea <= ea_max + 50)
    h5_result = {
        'n_samples': n_samples,
        'n_features': n_features,
        'hypothetical_material': 'catalyzed_MgH2 with novel_perovskite catalyst',
        'hypothetical_method': 'Kissinger',
        'predicted_ea_kJ_per_mol': predicted_ea,
        'rmse_all_samples': rmse_all,
        'ea_range_min': ea_min,
        'ea_range_max': ea_max,
        'prediction_reasonable': reasonable
    }
    return h5_result


# ============================================================
# BOOTSTRAP ANALYSIS WITH CLUSTER RESAMPLING BY doi
# ============================================================

def bootstrap_analysis(rows, n_bootstrap=50, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    paper_groups = get_paper_groups(rows)
    doi_list = sorted(paper_groups.keys())
    n_papers = len(doi_list)
    loop_results = []
    for loop in range(n_bootstrap):
        sampled_dois = random.choices(doi_list, k=n_papers)
        boot_rows = []
        for doi in sampled_dois:
            boot_rows.extend(paper_groups[doi])
        h1 = h1_analysis(boot_rows)
        h2 = h2_analysis(boot_rows)
        h3 = h3_analysis(boot_rows)
        h4 = h4_analysis(boot_rows)
        h5 = h5_analysis(boot_rows)
        loop_result = {
            'loop': loop + 1,
            'n_samples': len(boot_rows),
            'n_papers_sampled': len(set(sampled_dois)),
            'h1': h1,
            'h2': h2,
            'h3': h3,
            'h4': h4,
            'h5': h5
        }
        loop_results.append(loop_result)
        if (loop + 1) % 10 == 0:
            print(f"  Completed bootstrap loop {loop + 1}/{n_bootstrap}")
    return loop_results



def aggregate_bootstrap_results(loop_results):
    agg = {
        'h1': {'mean_delta_ea_boot': [], 'ci_lower_boot': [], 'ci_upper_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h2': {'eta_squared_boot': [], 'p_value_perm_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h3': {'ratio_boot': [], 'p_value_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h4': {'rmse_boot': [], 'within_10_boot': [], 'supported_count': 0, 'n_loops': len(loop_results)},
        'h5': {}
    }
    for lr in loop_results:
        if lr['h1']:
            agg['h1']['mean_delta_ea_boot'].append(lr['h1']['mean_delta_ea'])
            agg['h1']['ci_lower_boot'].append(lr['h1']['ci_95_lower'])
            agg['h1']['ci_upper_boot'].append(lr['h1']['ci_95_upper'])
            if lr['h1']['supported']:
                agg['h1']['supported_count'] += 1
        if lr['h2']:
            agg['h2']['eta_squared_boot'].append(lr['h2']['eta_squared'])
            agg['h2']['p_value_perm_boot'].append(lr['h2']['p_value_permutation'])
            if lr['h2']['supported']:
                agg['h2']['supported_count'] += 1
        if lr['h3']:
            agg['h3']['ratio_boot'].append(lr['h3']['ratio_method_to_ea'])
            agg['h3']['p_value_boot'].append(lr['h3']['p_value'])
            if lr['h3']['supported']:
                agg['h3']['supported_count'] += 1
        if lr['h4']:
            agg['h4']['rmse_boot'].append(lr['h4']['rmse'])
            agg['h4']['within_10_boot'].append(lr['h4']['within_10_kJmol'])
            if lr['h4']['supported']:
                agg['h4']['supported_count'] += 1
        if lr['h5']:
            agg['h5']['prediction'] = lr['h5']['predicted_ea_kJ_per_mol']
            agg['h5']['rmse'] = lr['h5']['rmse_all_samples']
    # Compute bootstrap confidence intervals for aggregated results (only for value keys)
    value_keys = ['mean_delta_ea_boot', 'eta_squared_boot', 'ratio_boot']
    for key in value_keys:
        if key in agg and agg[key] and len(agg[key]) > 0:
            arr = np.array(agg[key])
            agg[f'{key}_mean'] = float(np.mean(arr))
            agg[f'{key}_ci_lower'] = float(np.percentile(arr, 2.5))
            agg[f'{key}_ci_upper'] = float(np.percentile(arr, 97.5))
    return agg



# ============================================================
# MAIN ANALYSIS AND OUTPUT
# ============================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS OF CATALYST EFFECTS ON MgH2 ACTIVATION ENERGY")
    print("=" * 70)
    print("\n[1] Loading data...")
    rows = load_data('data/cleaned_ea_dataset.csv')
    print(f"    Loaded {len(rows)} rows")
    rows_des = filter_desorption_only(rows)
    print(f"    Desorption-only rows: {len(rows_des)}")
    print("\n[2] Running primary hypothesis tests...")
    h1 = h1_analysis(rows_des)
    print(f"    H1: Within-paper paired ΔEa analysis...")
    if h1:
        print(f"       Mean ΔEa = {h1['mean_delta_ea']:.2f} kJ/mol (p={h1['p_value']:.4f})")
        print(f"       Supported: {h1['supported']}")
    h2 = h2_analysis(rows_des)
    print(f"    H2: Eta-squared for catalyst family...")
    if h2:
        print(f"       η² = {h2['eta_squared']:.4f} (perm p={h2['p_value_permutation']:.4f})")
        print(f"       Supported: {h2['supported']}")
    h3 = h3_analysis(rows_des)
    print(f"    H3: Method comparison...")
    if h3:
        print(f"       Mean |Δ_method| = {h3['mean_absolute_method_delta']:.2f} kJ/mol")
        print(f"       Ratio to median ΔEa = {h3['ratio_method_to_ea']:.2f}")
        print(f"       Supported: {h3['supported']}")
    h4 = h4_analysis(rows_des)
    print(f"    H4: LOSO cross-validation...")
    if h4:
        print(f"       RMSE = {h4['rmse']:.2f} kJ/mol")
        print(f"       Within ±10 kJ/mol: {h4['within_10_kJmol']}/{h4['n_holdout']}")
        print(f"       Supported: {h4['supported']}")
    h5 = h5_analysis(rows_des)
    print(f"    H5: Extrapolation prediction...")
    if h5:
        print(f"       Predicted Ea = {h5['predicted_ea_kJ_per_mol']:.2f} kJ/mol")
        print(f"       RMSE (all samples) = {h5['rmse_all_samples']:.2f} kJ/mol")
    print("\n[3] Running bootstrap analysis (50 loops, cluster resampling by doi)...")
    loop_results = bootstrap_analysis(rows_des, n_bootstrap=50)
    print("    Bootstrap complete.")
    print("\n[4] Aggregating bootstrap results...")
    agg = aggregate_bootstrap_results(loop_results)
    results = {
        'primary_hypotheses': {'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4, 'h5': h5},
        'bootstrap_aggregated': agg,
        'data_summary': {
            'total_rows': len(rows),
            'desorption_rows': len(rows_des),
            'n_papers': len(get_paper_groups(rows)),
            'material_classes': list(set(r['material_class'] for r in rows_des)),
            'catalyst_families': list(set(r.get('catalyst_family', 'none') for r in rows_des)),
            'methods': list(set(r['method'] for r in rows_des))
        }
    }
    os.makedirs('results', exist_ok=True)
    with open('results/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"    Written: results/results.json")
    with open('results/loop_log.jsonl', 'w', encoding='utf-8') as f:
        for lr in loop_results:
            f.write(json.dumps(lr, ensure_ascii=False) + '\n')
    print(f"    Written: results/loop_log.jsonl")
    report = generate_report(results, rows_des, loop_results)
    with open('results/report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"    Written: results/report.md")
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def generate_report(results, rows_des, loop_results):
    h1 = results['primary_hypotheses']['h1']
    h2 = results['primary_hypotheses']['h2']
    h3 = results['primary_hypotheses']['h3']
    h4 = results['primary_hypotheses']['h4']
    h5 = results['primary_hypotheses']['h5']
    agg = results['bootstrap_aggregated']
    lines = []
    lines.append("# Comprehensive Analysis Report: Catalyst Effects on MgH2 Activation Energy")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    supported_count = sum(1 for h in [h1, h2, h3, h4, h5] if h and h.get('supported', False))
    lines.append(f"Out of 5 hypotheses, **{supported_count}** were supported at α = 0.05.")
    lines.append("")
    lines.append("## H1: Within-Paper Paired ΔEa Analysis")
    lines.append("")
    if h1:
        lines.append(f"""This hypothesis tests whether catalysts significantly reduce activation energy
within the same study. We computed paired differences (catalyzed - pure) for all
catalyzed/pure sample pairs within each paper, then applied DerSimonian-Laird
random-effects meta-analysis with bootstrap confidence intervals.

**Results:**
- Number of papers with paired comparisons: {h1['n_papers_with_pairs']}
- Total paired differences: {h1['n_total_pairs']}
- Mean ΔEa: {h1['mean_delta_ea']:.2f} kJ/mol (95% CI: [{h1['ci_95_lower']:.2f}, {h1['ci_95_upper']:.2f}])
- DerSimonian-Laird random effects mean: {h1['dl_random_effects_mean']:.2f} kJ/mol (95% CI: [{h1['dl_random_effects_ci_lower']:.2f}, {h1['dl_random_effects_ci_upper']:.2f}])
- Bootstrap 95% CI: [{h1['bootstrap_ci_lower']:.2f}, {h1['bootstrap_ci_upper']:.2f}]
- Paired t-test: t = {h1['t_statistic']:.3f}, p = {h1['p_value']:.4f}
""")
        if h1['supported']:
            lines.append("**Conclusion: SUPPORTED** - Catalysts significantly reduce activation energy.")
        else:
            lines.append("**Conclusion: NOT SUPPORTED** - No statistically significant reduction in Ea.")
    else:
        lines.append("No paired comparisons available within papers.")
    lines.append("")
    lines.append("## H2: Eta-Squared for Catalyst Family Effect")
    lines.append("")
    if h2:
        lines.append(f"""This hypothesis tests whether catalyst family explains a significant proportion
of variance in activation energy. Eta-squared (η²) measures the proportion of
total variance attributable to catalyst family, with permutation testing for
statistical significance.

**Results:**
- Number of catalyst families: {h2['n_families']}
- Eta-squared (η²): {h2['eta_squared']:.4f}
- F-statistic: {h2['f_statistic']:.3f} (ANOVA p = {h2['p_value_anova']:.4f})
- Permutation test p-value: {h2['p_value_permutation']:.4f}
""")
        if h2['supported']:
            lines.append("**Conclusion: SUPPORTED** - Catalyst family explains a significant proportion of variance in Ea.")
        else:
            lines.append("**Conclusion: NOT SUPPORTED** - Catalyst family does not explain significant variance in Ea.")
    lines.append("")
    lines.append("## H3: Method Comparison (Kissinger vs. isothermal_Arrhenius)")
    lines.append("")
    if h3:
        lines.append(f"""This hypothesis tests whether the choice of kinetic method (Kissinger vs.
isothermal_Arrhenius) produces systematically different Ea estimates. We compare
the absolute difference between method estimates within the same material/paper
to the typical within-paper ΔEa.

**Results:**
- Number of method pairs: {h3['n_method_pairs']}
- Mean absolute |Δ_method|: {h3['mean_absolute_method_delta']:.2f} kJ/mol
- Median absolute |Δ_method|: {h3['median_absolute_method_delta']:.2f} kJ/mol
- Median within-paper ΔEa: {h3['median_ea_reduction']:.2f} kJ/mol
- Ratio (|Δ_method| / median ΔEa): {h3['ratio_method_to_ea']:.2f}
- Paired t-test on method differences: t = {h3['t_statistic']:.3f}, p = {h3['p_value']:.4f}
""")
        if h3['supported']:
            lines.append("**Conclusion: SUPPORTED** - Method choice produces systematically different Ea estimates.")
        else:
            lines.append("**Conclusion: NOT SUPPORTED** - Method choice does not produce systematically different Ea estimates.")
    lines.append("")
    lines.append("## H4: Leave-One-Study-Out Cross-Validation")
    lines.append("")
    if h4:
        lines.append(f"""This hypothesis tests whether a regression model using material class,
catalyst family, and method features can predict activation energy. We use
leave-one-study-out (LOSO) cross-validation to assess predictive performance.

**Results:**
- Number of samples: {h4['n_samples']}
- Number of papers: {h4['n_papers']}
- Number of features: {h4['n_features']}
- Fixed effects mean prediction: {h4['fixed_effects_mean']:.2f} kJ/mol
- Random effects mean prediction: {h4['random_effects_mean']:.2f} kJ/mol
- Between-study variance (τ²): {h4['between_study_variance']:.2f}
- RMSE: {h4['rmse']:.2f} kJ/mol
- Predictions within ±10 kJ/mol: {h4['within_10_kJmol']}/{h4['n_holdout']}
""")
        if h4['supported']:
            lines.append("**Conclusion: SUPPORTED** - The model predicts Ea with reasonable accuracy.")
        else:
            lines.append("**Conclusion: NOT SUPPORTED** - The model does not predict Ea accurately.")
    lines.append("")
    lines.append("## H5: Extrapolation Prediction for Hypothetical Material")
    lines.append("")
    if h5:
        lines.append(f"""This hypothesis tests whether the model can extrapolate to a hypothetical
material with a novel catalyst family. We fit a ridge regression model on all
data and predict Ea for a catalyzed MgH2 with a novel perovskite catalyst
using the Kissinger method.

**Results:**
- Predicted Ea for hypothetical material: {h5['predicted_ea_kJ_per_mol']:.2f} kJ/mol
- RMSE on all samples: {h5['rmse_all_samples']:.2f} kJ/mol
- Observed Ea range: [{h5['ea_range_min']:.2f}, {h5['ea_range_max']:.2f}] kJ/mol
- Prediction reasonable (within plausible range): {h5['prediction_reasonable']}
""")
        if h5['prediction_reasonable']:
            lines.append("**Conclusion: SUPPORTED** - The model produces a reasonable extrapolated prediction.")
        else:
            lines.append("**Conclusion: NOT SUPPORTED** - The extrapolated prediction is outside a plausible range.")
    lines.append("")
    lines.append("## Bootstrap Analysis Results (50 Loops)")
    lines.append("")
    lines.append("Cluster resampling by doi was performed 50 times to assess stability of results.")
    lines.append("")
    if agg.get('h1', {}).get('mean_delta_ea_boot'):
        lines.append("### H1 Bootstrap Results")
        if 'mean_delta_ea_mean' in agg['h1']:
            lines.append(f"- Mean ΔEa (bootstrap): {agg['h1']['mean_delta_ea_mean']:.2f} kJ/mol")
        else:
            lines.append("- Mean ΔEa (bootstrap): N/A")
        if 'mean_delta_ea_ci_lower' in agg['h1'] and 'mean_delta_ea_ci_upper' in agg['h1']:
            lines.append(f"- 95% CI (bootstrap): [{agg['h1']['mean_delta_ea_ci_lower']:.2f}, {agg['h1']['mean_delta_ea_ci_upper']:.2f}] kJ/mol")
        else:
            lines.append("- 95% CI (bootstrap): N/A")
        lines.append(f"- Supported in {agg['h1']['supported_count']}/{agg['h1']['n_loops']} loops")
        lines.append("")
    if agg.get('h2', {}).get('eta_squared_boot'):
        lines.append("### H2 Bootstrap Results")
        if 'eta_squared_mean' in agg['h2']:
            lines.append(f"- Mean η² (bootstrap): {agg['h2']['eta_squared_mean']:.4f}")
        else:
            lines.append("- Mean η² (bootstrap): N/A")
        if 'eta_squared_ci_lower' in agg['h2'] and 'eta_squared_ci_upper' in agg['h2']:
            lines.append(f"- 95% CI (bootstrap): [{agg['h2']['eta_squared_ci_lower']:.4f}, {agg['h2']['eta_squared_ci_upper']:.4f}]")
        else:
            lines.append("- 95% CI (bootstrap): N/A")
        lines.append(f"- Supported in {agg['h2']['supported_count']}/{agg['h2']['n_loops']} loops")
        lines.append("")
    if agg.get('h3', {}).get('ratio_boot'):
        lines.append("### H3 Bootstrap Results")
        if 'ratio_mean' in agg['h3']:
            lines.append(f"- Mean ratio (bootstrap): {agg['h3']['ratio_mean']:.2f}")
        else:
            lines.append("- Mean ratio (bootstrap): N/A")
        if 'ratio_ci_lower' in agg['h3'] and 'ratio_ci_upper' in agg['h3']:
            lines.append(f"- 95% CI (bootstrap): [{agg['h3']['ratio_ci_lower']:.2f}, {agg['h3']['ratio_ci_upper']:.2f}]")
        else:
            lines.append("- 95% CI (bootstrap): N/A")
        lines.append(f"- Supported in {agg['h3']['supported_count']}/{agg['h3']['n_loops']} loops")
        lines.append("")
    if agg.get('h4', {}).get('rmse_boot'):
        lines.append("### H4 Bootstrap Results")
        rmse_arr = np.array(agg['h4']['rmse_boot'])
        lines.append(f"- Mean RMSE (bootstrap): {np.mean(rmse_arr):.2f} kJ/mol")
        lines.append(f"- 95% CI (bootstrap): [{np.percentile(rmse_arr, 2.5):.2f}, {np.percentile(rmse_arr, 97.5):.2f}] kJ/mol")
        lines.append(f"- Supported in {agg['h4']['supported_count']}/{agg['h4']['n_loops']} loops")
        lines.append("")
    lines.append("## Data Summary")
    lines.append("")
    ds = results['data_summary']
    lines.append(f"- Desorption-only rows: {ds['desorption_rows']}")
    lines.append(f"- Total papers: {ds['n_papers']}")
    lines.append(f"- Material classes: {', '.join(ds['material_classes'])}")
    lines.append(f"- Catalyst families: {', '.join(ds['catalyst_families'])}")
    lines.append(f"- Methods: {', '.join(ds['methods'])}")
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("1. **H1**: Within-paper paired differences were computed for all catalyzed/pure sample pairs within each study. The DerSimonian-Laird random-effects model accounts for between-study heterogeneity. Bootstrap confidence intervals were obtained by cluster resampling (resampling papers with replacement).")
    lines.append("")
    lines.append("2. **H2**: Eta-squared was computed from one-way ANOVA of Ea by catalyst family. Permutation testing (1000 permutations) assessed statistical significance by shuffling group labels and recomputing η².")
    lines.append("")
    lines.append("3. **H3**: For materials with both Kissinger and isothermal_Arrhenius estimates in the same study, the absolute difference was computed. This was compared to the typical within-paper ΔEa to assess whether method choice introduces systematic bias.")
    lines.append("")
    lines.append("4. **H4**: Leave-one-study-out cross-validation used ridge regression with features from material class, catalyst family, and method. Fixed and random effects meta-analyses were applied to the LOSO predictions.")
    lines.append("")
    lines.append("5. **H5**: A ridge regression model was fit on all data and used to predict Ea for a hypothetical catalyzed MgH2 with a novel catalyst family (extrapolation beyond observed categories).")
    lines.append("")
    lines.append("6. **Bootstrap**: 50 bootstrap loops with cluster resampling by doi were performed to assess the stability of each hypothesis test. In each loop, papers were resampled with replacement, and all hypotheses were recomputed on the bootstrap sample.")
    lines.append("")
    return "\n".join(lines)


if __name__ == '__main__':
    main()