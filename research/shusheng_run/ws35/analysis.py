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
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_ea_value(val):
    if not val or val.strip() == '':
        return None
    val = val.strip()
    parts = val.split('±')
    if len(parts) > 1:
        val = parts[0].strip()
    try:
        return float(val.split()[0])
    except:
        return None


def filter_data(rows):
    filtered = []
    for row in rows:
        if row.get('direction', '').strip() == 'absorption':
            continue
        ea = parse_ea_value(row.get('Ea_kJ_per_mol', ''))
        if ea is None:
            continue
        row = row.copy()
        row['Ea_parsed'] = ea
        # Strip whitespace from categorical fields
        for key in ['material_class', 'catalyst_family', 'method']:
            if key in row:
                row[key] = row[key].strip()
        filtered.append(row)
    return filtered


def get_paired_rows(filtered_rows):
    by_doi = defaultdict(list)
    for row in filtered_rows:
        doi = row.get('doi', '').strip() if row.get('doi') else ''
        if doi:
            by_doi[doi].append(row)
    return by_doi


# ============================================================
# HYPOTHESIS 1: Within-paper paired ΔEa, random effects meta-analysis
# ============================================================

def h1_within_paper_paired(filtered_rows, by_doi, n_bootstrap=50):
    print("\n=== H1: Within-paper paired ΔEa analysis ===")

    # Build pairs directly from by_doi with normalized doi
    pairs = []  # list of (doi, delta_ea)
    for doi, rows in by_doi.items():
        doi_clean = doi.strip() if doi else ''
        pure_rows = [r for r in rows if r.get('material_class') == 'pure_MgH2']
        catalyzed_rows = [r for r in rows if r.get('material_class') == 'catalyzed_MgH2']
        for pur in pure_rows:
            for cat in catalyzed_rows:
                delta_ea = cat['Ea_parsed'] - pur['Ea_parsed']
                pairs.append((doi_clean, delta_ea))

    if not pairs:
        print("  No paired samples found.")
        return None

    delta_eas = [d for _, d in pairs]
    mean_dea = float(np.mean(delta_eas))
    std_dea = float(np.std(delta_eas, ddof=1))
    print(f"  Found {len(pairs)} paired samples")
    print(f"  Mean ΔEa (fixed effects): {mean_dea:.2f} kJ/mol")
    print(f"  Std ΔEa: {std_dea:.2f} kJ/mol")

    # Compute paper-level statistics directly from pairs
    # Group deltas by doi
    paper_deltas = defaultdict(list)
    for doi, delta in pairs:
        paper_deltas[doi].append(delta)

    paper_means = {}
    paper_counts = {}
    for doi, deltas in paper_deltas.items():
        paper_means[doi] = float(np.mean(deltas))
        paper_counts[doi] = len(deltas)

    print(f"  Papers with pairs: {len(paper_means)}")
    for doi, pm in paper_means.items():
        print(f"    {doi[:30]}: mean={pm:.2f}, n={paper_counts[doi]}")

    if not paper_means:
        return None

    overall_mean = float(np.mean(list(paper_means.values())))
    Q = float(sum((paper_means[doi] - overall_mean)**2 for doi in paper_means))
    df_Q = len(paper_means) - 1

    print(f"  Q={Q:.2f}, df_Q={df_Q}")

    if Q > df_Q and df_Q > 0 and paper_counts:
        weight_term = sum(1.0 / paper_counts[doi] for doi in paper_means)
        if weight_term > 0:
            tau_sq = (Q - df_Q) / weight_term
        else:
            tau_sq = 0.0
        tau_sq = max(0.0, tau_sq)
    else:
        tau_sq = 0.0

    print(f"  tau_sq={tau_sq:.2f}")

    # Random effects weights
    weights = {}
    for doi in paper_means:
        denom = paper_counts[doi] + tau_sq
        weights[doi] = 1.0 / denom if denom > 0 else 1.0
    total_weight = sum(weights[doi] for doi in paper_means)
    re_mean = float(sum(weights[doi] * paper_means[doi] for doi in paper_means) / total_weight)

    print(f"  Random effects mean ΔEa: {re_mean:.2f} kJ/mol")

    # Cluster bootstrap: resample papers (doi) with replacement
    bootstrap_means = []
    np.random.seed(42)
    paper_keys = list(paper_means.keys())
    for _ in range(n_bootstrap):
        sampled_dois = np.random.choice(paper_keys, size=len(paper_keys), replace=True)
        boot_deltas = []
        for doi in sampled_dois:
            # Include all deltas for this paper
            boot_deltas.extend(paper_deltas[doi])
        if boot_deltas:
            bootstrap_means.append(float(np.mean(boot_deltas)))

    if not bootstrap_means:
        print("  Bootstrap failed - no valid samples.")
        return None

    bootstrap_means = np.array(bootstrap_means)
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    print(f"  Cluster bootstrap CI (95%): [{ci_lower:.2f}, {ci_upper:.2f}] kJ/mol")

    return {
        'n_pairs': len(pairs),
        'n_papers': len(paper_means),
        'mean_delta_ea_fixed': mean_dea,
        'std_delta_ea': std_dea,
        'tau_sq': float(tau_sq),
        'mean_delta_ea_random': re_mean,
        'bootstrap_ci_lower': ci_lower,
        'bootstrap_ci_upper': ci_upper,
        'bootstrap_means': bootstrap_means.tolist()
    }


# ============================================================
# HYPOTHESIS 2: η² for catalyst family, permutation test
# ============================================================

def h2_eta2_permutation(filtered_rows, n_permutations=1000):
    print("\n=== H2: η² for catalyst family with permutation test ===")

    by_doi = defaultdict(list)
    for row in filtered_rows:
        by_doi[row['doi']].append(row)

    all_eas = np.array([r['Ea_parsed'] for r in filtered_rows])
    all_families = np.array([r.get('catalyst_family', 'none') for r in filtered_rows])
    unique_families = sorted(set(all_families))

    n_total = len(all_eas)
    grand_mean = np.mean(all_eas)
    ss_total = float(np.sum((all_eas - grand_mean)**2))

    ss_between = 0.0
    for fam in unique_families:
        mask = all_families == fam
        if np.sum(mask) > 0:
            fam_mean = np.mean(all_eas[mask])
            ss_between += float(np.sum(mask)) * (fam_mean - grand_mean)**2

    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    print(f"  Observed η²: {eta_squared:.4f}")

    # Permutation test: shuffle within each paper
    perm_eta2s = []
    np.random.seed(42)

    for perm in range(n_permutations):
        shuffled_families = np.empty(n_total, dtype=object)
        for doi, rows in by_doi.items():
            paper_indices = [idx for idx, fr in enumerate(filtered_rows) if fr in rows]
            paper_families = np.array([r.get('catalyst_family', 'none') for r in rows])
            np.random.shuffle(paper_families)
            for idx, fam in zip(paper_indices, paper_families):
                shuffled_families[idx] = fam

        perm_ss_between = 0.0
        for fam in set(shuffled_families):
            mask = shuffled_families == fam
            if np.sum(mask) > 0:
                fam_mean = np.mean(all_eas[mask])
                perm_ss_between += float(np.sum(mask)) * (fam_mean - grand_mean)**2
        perm_eta2 = perm_ss_between / ss_total if ss_total > 0 else 0.0
        perm_eta2s.append(perm_eta2)

    perm_eta2s = np.array(perm_eta2s)
    p_value = float(np.mean(perm_eta2s >= eta_squared))

    print(f"  Permutation p-value: {p_value:.4f}")

    return {
        'eta_squared': eta_squared,
        'ss_between': ss_between,
        'ss_total': ss_total,
        'n_permutations': n_permutations,
        'p_value': p_value,
        'perm_eta2s': perm_eta2s.tolist()
    }


# ============================================================
# HYPOTHESIS 3: Method comparison pairs
# ============================================================

def h3_method_comparison(filtered_rows):
    print("\n=== H3: Method comparison pairs ===")

    by_key = defaultdict(list)
    for row in filtered_rows:
        key = (row['doi'], row['material'])
        by_key[key].append(row)

    comparison_pairs = []
    for key, rows in by_key.items():
        kissinger_rows = [r for r in rows if r.get('method') == 'Kissinger']
        iso_rows = [r for r in rows if r.get('method') == 'isothermal_Arrhenius']
        for k in kissinger_rows:
            for i in iso_rows:
                diff = i['Ea_parsed'] - k['Ea_parsed']
                comparison_pairs.append({
                    'doi': key[0],
                    'material': key[1],
                    'Ea_Kissinger': k['Ea_parsed'],
                    'EIsothermal': i['Ea_parsed'],
                    'delta_Ea': diff
                })

    print(f"  Found {len(comparison_pairs)} method comparison pairs")
    if comparison_pairs:
        deltas = [p['delta_Ea'] for p in comparison_pairs]
        print(f"  Mean difference (Iso - Kissinger): {np.mean(deltas):.2f} kJ/mol")
        t_stat, p_val = stats.ttest_1samp(deltas, 0)
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")

    return {
        'n_pairs': len(comparison_pairs),
        'pairs': comparison_pairs
    }


# ============================================================
# HYPOTHESIS 4: LOSO cross-validation
# ============================================================

def h4_loso_crossval(filtered_rows, n_repeats=10):
    print("\n=== H4: LOSO cross-validation ===")

    by_doi = defaultdict(list)
    for row in filtered_rows:
        by_doi[row['doi']].append(row)

    unique_dois = sorted(by_doi.keys())
    n_dois = len(unique_dois)

    material_classes = sorted(set(r['material_class'] for r in filtered_rows))
    catalyst_families = sorted(set(r.get('catalyst_family', 'none') for r in filtered_rows))
    methods = sorted(set(r.get('method', '') for r in filtered_rows))

    print(f"  Material classes: {material_classes}")
    print(f"  Catalyst families: {catalyst_families}")
    print(f"  Methods: {methods}")
    print(f"  Total papers for LOSO: {n_dois}")

    def build_features(rows):
        X_rows = []
        y_rows = []
        for r in rows:
            vec = [
                1 if r['material_class'] == mc else 0 for mc in material_classes
            ] + [
                1 if r.get('catalyst_family', 'none') == cf else 0 for cf in catalyst_families
            ] + [
                1 if r.get('method', '') == m else 0 for m in methods
            ]
            X_rows.append(vec)
            y_rows.append(r['Ea_parsed'])
        return np.array(X_rows), np.array(y_rows)

    errors = []
    predictions = []
    true_values = []

    for i, holdout_doi in enumerate(unique_dois):
        if i % 10 == 0:
            print(f"  LOSO iteration {i+1}/{n_dois} (holding out {holdout_doi})")

        train_rows = []
        for doi, rows in by_doi.items():
            if doi != holdout_doi:
                train_rows.extend(rows)

        holdout_rows = by_doi[holdout_doi]

        if len(train_rows) < 2:
            continue

        X_train, y_train = build_features(train_rows)
        X_holdout, y_holdout = build_features(holdout_rows)

        try:
            beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            y_pred = X_holdout @ beta
            errors.extend(y_holdout - y_pred)
            predictions.extend(y_pred)
            true_values.extend(y_holdout)
        except Exception as e:
            print(f"  Error in iteration {i}: {e}")
            continue

    if errors:
        errors = np.array(errors)
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        ss_res = float(np.sum(errors**2))
        ss_tot = float(np.sum((np.array(true_values) - np.mean(true_values))**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"  LOSO RMSE: {rmse:.2f} kJ/mol")
        print(f"  LOSO MAE: {mae:.2f} kJ/mol")
        print(f"  LOSO R²: {r2:.4f}")

        return {
            'n_papers': n_dois,
            'n_features': len(beta),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'errors': errors.tolist(),
            'predictions': predictions,
            'true_values': true_values
        }
    else:
        return {
            'n_papers': n_dois,
            'n_features': 0,
            'rmse': None,
            'mae': None,
            'r2': None,
            'errors': [],
            'predictions': [],
            'true_values': []
        }


# ============================================================
# HYPOTHESIS 5: Out-of-sample prediction
# ============================================================

def h5_out_of_sample_prediction(filtered_rows):
    print("\n=== H5: Out-of-sample prediction ===")

    heldout_rows = [r for r in filtered_rows
                    if r.get('material_class') == 'catalyzed_MgH2'
                    and r.get('catalyst_family') == 'other'
                    and r.get('method') == 'isothermal_Arrhenius']

    train_rows = [r for r in filtered_rows if r not in heldout_rows]

    print(f"  Training rows: {len(train_rows)}")
    print(f"  Held-out rows: {len(heldout_rows)}")

    if len(train_rows) < 2:
        print("  Not enough training data.")
        return None

    if len(heldout_rows) == 0:
        print("  No rows match the held-out combination.")
        # Predict for hypothetical point
        material_classes = sorted(set(r['material_class'] for r in filtered_rows))
        catalyst_families = sorted(set(r.get('catalyst_family', 'none') for r in filtered_rows))
        methods = sorted(set(r.get('method', '') for r in filtered_rows))

        def build_features(rows):
            X_rows = []
            y_rows = []
            for r in rows:
                vec = [
                    1 if r['material_class'] == mc else 0 for mc in material_classes
                ] + [
                    1 if r.get('catalyst_family', 'none') == cf else 0 for cf in catalyst_families
                ] + [
                    1 if r.get('method', '') == m else 0 for m in methods
                ]
                X_rows.append(vec)
                y_rows.append(r['Ea_parsed'])
            return np.array(X_rows), np.array(y_rows)

        X_train, y_train = build_features(train_rows)
        beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

        vec = [
            1 if mc == 'catalyzed_MgH2' else 0 for mc in material_classes
        ] + [
            1 if cf == 'other' else 0 for cf in catalyst_families
        ] + [
            1 if m == 'isothermal_Arrhenius' else 0 for m in methods
        ]
        X_hyp = np.array([vec])
        y_pred = X_hyp @ beta

        print(f"  Hypothetical prediction: {y_pred[0]:.2f} kJ/mol")

        return {
            'n_train': len(train_rows),
            'n_holdout': 0,
            'true_values': [],
            'predicted_values': [float(y_pred[0])],
            'coefficients': beta.tolist(),
            'n_features': len(beta),
            'hypothetical_prediction': True
        }

    material_classes = sorted(set(r['material_class'] for r in filtered_rows))
    catalyst_families = sorted(set(r.get('catalyst_family', 'none') for r in filtered_rows))
    methods = sorted(set(r.get('method', '') for r in filtered_rows))

    def build_features(rows):
        X_rows = []
        y_rows = []
        for r in rows:
            vec = [
                1 if r['material_class'] == mc else 0 for mc in material_classes
            ] + [
                1 if r.get('catalyst_family', 'none') == cf else 0 for cf in catalyst_families
            ] + [
                1 if r.get('method', '') == m else 0 for m in methods
            ]
            X_rows.append(vec)
            y_rows.append(r['Ea_parsed'])
        return np.array(X_rows), np.array(y_rows)

    X_train, y_train = build_features(train_rows)
    X_holdout, y_holdout = build_features(heldout_rows)

    beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    y_pred = X_holdout @ beta

    print(f"  True Ea values: {y_holdout.tolist()}")
    print(f"  Predicted Ea values: {y_pred.tolist()}")

    errors = y_holdout - y_pred
    rmse = float(np.sqrt(np.mean(errors**2)))
    print(f"  Out-of-sample RMSE: {rmse:.2f} kJ/mol")

    return {
        'n_train': len(train_rows),
        'n_holdout': len(heldout_rows),
        'true_values': y_holdout.tolist(),
        'predicted_values': y_pred.tolist(),
        'coefficients': beta.tolist(),
        'n_features': len(beta),
        'rmse': rmse
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("Loading data...")
    raw_rows = load_data('/workspace/research/shusheng_run/ws35/data/cleaned_ea_dataset.csv')
    print(f"  Raw rows: {len(raw_rows)}")

    filtered_rows = filter_data(raw_rows)
    print(f"  After filtering (desorption only): {len(filtered_rows)}")

    by_doi = get_paired_rows(filtered_rows)
    print(f"  Papers (unique doi): {len(by_doi)}")

    results = {}

    print("\n--- Running H1 ---")
    results['h1'] = h1_within_paper_paired(filtered_rows, by_doi, n_bootstrap=50)

    print("\n--- Running H2 ---")
    results['h2'] = h2_eta2_permutation(filtered_rows, n_permutations=1000)

    print("\n--- Running H3 ---")
    results['h3'] = h3_method_comparison(filtered_rows)

    print("\n--- Running H4 ---")
    results['h4'] = h4_loso_crossval(filtered_rows, n_repeats=10)

    print("\n--- Running H5 ---")
    results['h5'] = h5_out_of_sample_prediction(filtered_rows)

    os.makedirs('/workspace/research/shusheng_run/ws35/results', exist_ok=True)

    print("\nWriting results.json...")
    with open('/workspace/research/shusheng_run/ws35/results/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Writing loop_log.jsonl...")
    with open('/workspace/research/shusheng_run/ws35/results/loop_log.jsonl', 'w', encoding='utf-8') as f:
        if results.get('h1') and results['h1'].get('bootstrap_means'):
            for i, val in enumerate(results['h1']['bootstrap_means']):
                f.write(json.dumps({'loop': i, 'type': 'h1_bootstrap', 'mean_delta_ea': val}) + '\n')
        if results.get('h2') and results['h2'].get('perm_eta2s'):
            for i, val in enumerate(results['h2']['perm_eta2s']):
                f.write(json.dumps({'loop': i, 'type': 'h2_permutation', 'eta_squared': val}) + '\n')

    print("Writing report.md...")
    report = generate_report(results)
    with open('/workspace/research/shusheng_run/ws35/results/report.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\nAnalysis complete!")


def generate_report(results):
    lines = []
    lines.append("# Catalyst Effect Analysis Report")
    lines.append("")
    lines.append("## Summary of Hypothesis Tests")
    lines.append("")

    h1 = results.get('h1', {})
    if h1:
        lines.append("### H1: Within-paper paired ΔEa Analysis")
        lines.append(f"- Number of paired samples: {h1['n_pairs']}")
        lines.append(f"- Number of papers: {h1['n_papers']}")
        lines.append(f"- Mean ΔEa (fixed effects): {h1['mean_delta_ea_fixed']:.2f} kJ/mol")
        lines.append(f"- Random effects mean ΔEa: {h1['mean_delta_ea_random']:.2f} kJ/mol")
        lines.append(f"- Between-study variance (τ²): {h1['tau_sq']:.2f}")
        lines.append(f"- 95% Cluster bootstrap CI: [{h1['bootstrap_ci_lower']:.2f}, {h1['bootstrap_ci_upper']:.2f}] kJ/mol")
        lines.append("")

    h2 = results.get('h2', {})
    if h2:
        lines.append("### H2: η² for Catalyst Family with Permutation Test")
        lines.append(f"- Observed η²: {h2['eta_squared']:.4f}")
        lines.append(f"- Permutation p-value: {h2['p_value']:.4f}")
        lines.append("")

    h3 = results.get('h3', {})
    if h3:
        lines.append("### H3: Method Comparison (Kissinger vs Isothermal Arrhenius)")
        lines.append(f"- Number of comparison pairs: {h3['n_pairs']}")
        if h3['pairs']:
            deltas = [p['delta_Ea'] for p in h3['pairs']]
            lines.append(f"- Mean difference (Iso - Kissinger): {np.mean(deltas):.2f} kJ/mol")
            t_stat, p_val = stats.ttest_1samp(deltas, 0)
            lines.append(f"- Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        lines.append("")

    h4 = results.get('h4', {})
    if h4 and h4.get('rmse') is not None:
        lines.append("### H4: Leave-One-Subject-Out Cross-Validation")
        lines.append(f"- Number of papers: {h4['n_papers']}")
        lines.append(f"- RMSE: {h4['rmse']:.2f} kJ/mol")
        lines.append(f"- MAE: {h4['mae']:.2f} kJ/mol")
        lines.append(f"- R²: {h4['r2']:.4f}")
        lines.append("")

    h5 = results.get('h5', {})
    if h5:
        lines.append("### H5: Out-of-Sample Prediction (catalyzed_MgH2, other, isothermal_Arrhenius)")
        lines.append(f"- Training samples: {h5['n_train']}")
        if h5.get('n_holdout', 0) > 0:
            lines.append(f"- Held-out samples: {h5['n_holdout']}")
            lines.append(f"- True Ea values: {h5['true_values']}")
            lines.append(f"- Predicted Ea values: {h5['predicted_values']}")
            lines.append(f"- RMSE: {h5['rmse']:.2f} kJ/mol")
        elif h5.get('hypothetical_prediction'):
            lines.append("- No matching rows found; predicting for hypothetical")
            lines.append(f"- Predicted Ea: {h5['predicted_values'][0]:.2f} kJ/mol")
        lines.append("")

    lines.append("## Data Summary")
    lines.append("- Desorption-only rows from multiple papers")
    lines.append("- Material classes: catalyzed_MgH2, pure_MgH2, alloy, complex_hydride")
    lines.append("- Catalyst families: mxene_carbon, oxide, metal_nanoparticle, other, none")
    lines.append("- Methods: Kissinger, isothermal_Arrhenius")

    return "\n".join(lines)


if __name__ == '__main__':
    main()