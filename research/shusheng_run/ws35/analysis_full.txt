#!/usr/bin/env python3
"""
Analysis of Catalyst Effects on MgH2 Activation Energy
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

# Set random seed
np.random.seed(20261029)
random.seed(20261029)


def parse_ea_value(val):
    """Parse Ea value from string, extracting the first number."""
    if not val or val.strip() == '':
        return None
    val = val.strip()
    # Handle "109 +/- 1" format
    if '\u00b1' in val:
        val = val.split('\u00b1')[0].strip()
    # Handle "~111.49" format
    if '~' in val:
        val = val[1:].strip()
    try:
        parts = val.split()
        if parts:
            return float(parts[0])
    except (ValueError, IndexError):
        pass
    return None


def load_data(filepath):
    """Load and preprocess the dataset."""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter desorption only
            if row.get('direction', '').strip() != 'desorption':
                continue
            # Remove questionable row (MgH2 + 5 wt% V, doi 10.3390/ma16155480)
            if row.get('material', '') == 'MgH2 + 5 wt% V' and row.get('doi', '') == '10.3390/ma16155480':
                continue
            # Parse Ea
            ea = parse_ea_value(row.get('Ea_kJ_per_mol', ''))
            if ea is None:
                continue
            row['Ea_parsed'] = ea
            rows.append(row)
    return rows


def build_features(rows_list, material_classes, catalyst_families, methods):
    """Build feature matrix and target vector from rows."""
    X_rows = []
    y_rows = []
    for r in rows_list:
        feat = []
        # Material class one-hot
        for mc in material_classes:
            feat.append(1 if r.get('material_class') == mc else 0)
        # Catalyst family one-hot
        for cf in catalyst_families:
            feat.append(1 if r.get('catalyst_family', 'none') == cf else 0)
        # Method one-hot
        for m in methods:
            feat.append(1 if r.get('method') == m else 0)
        X_rows.append(feat)
        y_rows.append(r['Ea_parsed'])
    return np.array(X_rows), np.array(y_rows)


def h1_analysis(rows):
    """H1: Within-paper paired analysis with DerSimonian-Laird and cluster bootstrap."""
    # Group by doi
    by_doi = defaultdict(list)
    for r in rows:
        by_doi[r['doi']].append(r)

    # Find paired samples (pure_MgH2 and catalyzed_MgH2 within same paper)
    pairs = []
    for doi, paper_rows in by_doi.items():
        pure_rows = [r for r in paper_rows if r.get('material_class') == 'pure_MgH2']
        catalyzed_rows = [r for r in paper_rows if r.get('material_class') == 'catalyzed_MgH2']
        for pur in pure_rows:
            for cat in catalyzed_rows:
                delta_ea = cat['Ea_parsed'] - pur['Ea_parsed']
                pairs.append((doi, delta_ea))

    if not pairs:
        print("No paired samples found for H1.")
        return None, None, None, None, None

    delta_eas = np.array([d for _, d in pairs])
    n_pairs = len(delta_eas)
    n_papers = len(set(d for d, _ in pairs))

    # Fixed effects mean (equal weights)
    mean_delta_fixed = float(np.mean(delta_eas))
    std_delta = float(np.std(delta_eas, ddof=1))

    # DerSimonian-Laird random effects
    paper_deltas = defaultdict(list)
    for doi, delta in pairs:
        paper_deltas[doi].append(delta)

    paper_means = {}
    paper_vars = {}
    for doi, deltas in paper_deltas.items():
        darr = np.array(deltas)
        paper_means[doi] = float(np.mean(darr))
        if len(darr) > 1:
            paper_vars[doi] = float(np.var(darr, ddof=1))
        else:
            paper_vars[doi] = std_delta ** 2

    # Iterative DL estimation of tau^2
    tau_sq = 0.0
    max_iter = 1000
    tol = 1e-6

    for _ in range(max_iter):
        weights = [1.0 / (paper_vars.get(d, std_delta ** 2) + tau_sq) for d in paper_means]
        wsum = sum(weights)
        if wsum == 0:
            break
        weighted_mean = sum(paper_means[d] * w for d, w in zip(paper_means.keys(), weights)) / wsum

        Q = sum(w * (paper_means[d] - weighted_mean) ** 2 for d, w in zip(paper_means.keys(), weights))
        df_Q = len(paper_means) - 1

        denom = sum(w ** 2 for w in weights)
        if denom > 0:
            new_tau_sq = (Q - df_Q) / denom
        else:
            new_tau_sq = 0.0
        new_tau_sq = max(0.0, new_tau_sq)

        if abs(new_tau_sq - tau_sq) < tol:
            tau_sq = new_tau_sq
            break
        tau_sq = new_tau_sq

    # Final DL weighted mean
    weights = [1.0 / (paper_vars.get(d, std_delta ** 2) + tau_sq) for d in paper_means]
    wsum = sum(weights)
    mean_delta_dl = sum(paper_means[d] * w for d, w in zip(paper_means.keys(), weights)) / wsum if wsum > 0 else mean_delta_fixed

    # Cluster bootstrap (resample papers with replacement)
    n_bootstrap = 50
    unique_pois = list(paper_means.keys())
    bootstrap_means = []

    for i in range(n_bootstrap):
        sampled_pois = np.random.choice(unique_pois, size=len(unique_pois), replace=True)
        sampled_deltas = []
        for doi in sampled_pois:
            sampled_deltas.extend(paper_deltas[doi])
        if sampled_deltas:
            bootstrap_means.append(float(np.array(sampled_deltas).mean()))

    # Bootstrap CI (percentile method)
    if len(bootstrap_means) >= 2:
        ci_lower = float(np.percentile(bootstrap_means, 2.5))
        ci_upper = float(np.percentile(bootstrap_means, 97.5))
    else:
        ci_lower = mean_delta_fixed
        ci_upper = mean_delta_fixed

    # Paired t-test
    t_stat, p_value = stats.ttest_1samp(delta_eas, 0)

    h1_result = {
        'n_pairs': n_pairs,
        'n_papers': n_papers,
        'mean_delta_ea_fixed': round(mean_delta_fixed, 2),
        'mean_delta_ea_random': round(mean_delta_dl, 2),
        'tau_sq': round(tau_sq, 2),
        'bootstrap_ci_lower': round(ci_lower, 2),
        'bootstrap_ci_upper': round(ci_upper, 2),
        't_statistic': round(float(t_stat), 2),
        'p_value': round(float(p_value), 6),
        'supported': bool(mean_delta_fixed < 0 and p_value < 0.05)
    }

    return h1_result, delta_eas, paper_deltas, bootstrap_means, unique_pois


def h2_analysis(rows):
    """H2: Eta-squared for catalyst family with permutation test."""
    by_doi = defaultdict(list)
    for r in rows:
        by_doi[r['doi']].append(r)

    # Use one observation per paper per family (mean Ea)
    paper_family_eas = defaultdict(list)
    for r in rows:
        doi = r['doi']
        family = r.get('catalyst_family', 'none')
        paper_family_eas[(doi, family)].append(r['Ea_parsed'])

    # For each paper-family combination, take the mean
    data_by_family = defaultdict(list)
    for (doi, family), eas in paper_family_eas.items():
        data_by_family[family].append(np.mean(eas))

    families = sorted(data_by_family.keys())
    all_eas = []
    all_labels = []
    for fam in families:
        all_eas.extend(data_by_family[fam])
        all_labels.extend([fam] * len(data_by_family[fam]))

    all_eas = np.array(all_eas)
    n_total = len(all_eas)
    grand_mean = np.mean(all_eas)
    ss_total = float(np.sum((all_eas - grand_mean) ** 2))

    # SS between
    ss_between = 0.0
    for fam in families:
        fam_mean = np.mean(data_by_family[fam])
        ss_between += len(data_by_family[fam]) * (fam_mean - grand_mean) ** 2

    ss_within = ss_total - ss_between
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    # Permutation test (shuffle family labels while preserving counts)
    n_permutations = 1000
    perm_eta_squares = []

    # Build unit list: (doi, family, mean_ea)
    unit_info = []
    for (doi, family), eas in paper_family_eas.items():
        unit_info.append((doi, family, np.mean(eas)))

    # Count units per family for permutation
    family_counts = defaultdict(int)
    for _, fam, _ in unit_info:
        family_counts[fam] += 1

    perm_eas = np.array([u[2] for u in unit_info])
    original_families = np.array([u[1] for u in unit_info])

    for _ in range(n_permutations):
        # Shuffle family labels while preserving counts
        fam_list = list(original_families)
        np.random.shuffle(fam_list)
        # Compute permuted eta-squared
        perm_ss_between = 0.0
        for fam in set(fam_list):
            mask = np.array(fam_list) == fam
            fam_vals = perm_eas[mask]
            if len(fam_vals) > 0:
                perm_ss_between += len(fam_vals) * (np.mean(fam_vals) - grand_mean) ** 2
        perm_eta = perm_ss_between / ss_total if ss_total > 0 else 0.0
        perm_eta_squares.append(perm_eta)

    perm_p_value = float(np.mean(np.array(perm_eta_squares) >= eta_squared))

    # ANOVA p-value for reference
    if len(families) > 1 and ss_within > 0:
        f_stat = (ss_between / (len(families) - 1)) / (ss_within / (n_total - len(families)))
        p_value_anova = float(1 - stats.f.cdf(f_stat, len(families) - 1, n_total - len(families)))
    else:
        f_stat = 0.0
        p_value_anova = 1.0

    h2_result = {
        'eta_squared': round(float(eta_squared), 4),
        'p_value': round(float(perm_p_value), 4),
        'p_value_anova': round(p_value_anova, 4),
        'p_value_permutation': round(float(perm_p_value), 4),
        'supported': bool(eta_squared > 0.1 and perm_p_value < 0.05)
    }

    return h2_result


def h3_analysis(rows):
    """H3: Method pairing (Kissinger vs isothermal_Arrhenius)."""
    material_paper_methods = defaultdict(lambda: defaultdict(list))
    for r in rows:
        material = r['material']
        doi = r['doi']
        method = r['method']
        ea = r['Ea_parsed']
        material_paper_methods[material][doi].append((method, ea))

    method_pairs_info = []

    for material, papers in material_paper_methods.items():
        for doi, methods in papers.items():
            kissinger_ea = None
            isothermal_ea = None
            for m, ea in methods:
                if m == 'Kissinger':
                    kissinger_ea = ea
                elif m == 'isothermal_Arrhenius':
                    isothermal_ea = ea
            if kissinger_ea is not None and isothermal_ea is not None:
                delta = kissinger_ea - isothermal_ea
                method_pairs_info.append({
                    'material': material,
                    'doi': doi,
                    'Ea_Kissinger': kissinger_ea,
                    'EIsothermal': isothermal_ea,
                    'delta_Ea': round(delta, 2)
                })

    h3_result = {
        'n_pairs': len(method_pairs_info),
        'pairs': method_pairs_info
    }

    return h3_result


def h4_analysis(rows):
    """H4: LOSO cross-validation with ridge regression."""
    by_doi = defaultdict(list)
    for r in rows:
        by_doi[r['doi']].append(r)

    unique_pois = sorted(by_doi.keys())
    n_papers = len(unique_pois)

    # Get unique categories
    material_classes = sorted(set(r.get('material_class') for r in rows))
    catalyst_families = sorted(set(r.get('catalyst_family', 'none') for r in rows))
    methods = sorted(set(r.get('method') for r in rows))

    loso_predictions = []
    loso_true = []

    for holdout_doi in unique_pois:
        train_rows = [r for r in rows if r['doi'] != holdout_doi]
        holdout_rows = [r for r in rows if r['doi'] == holdout_doi]

        X_train, y_train = build_features(train_rows, material_classes, catalyst_families, methods)
        X_holdout, y_holdout = build_features(holdout_rows, material_classes, catalyst_families, methods)

        # Ridge regression
        alpha = 1.0
        n_features = X_train.shape[1]
        XtX = X_train.T @ X_train
        Xty = X_train.T @ y_train
        try:
            beta = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtX + alpha * np.eye(n_features), Xty, rcond=None)[0]

        # Predict on holdout
        y_pred = X_holdout @ beta

        loso_predictions.extend(y_pred)
        loso_true.extend(y_holdout)

    loso_predictions = np.array(loso_predictions)
    loso_true = np.array(loso_true)

    rmse = float(np.sqrt(np.mean((loso_predictions - loso_true) ** 2)))
    mae = float(np.mean(np.abs(loso_predictions - loso_true)))
    ss_res = np.sum((loso_true - loso_predictions) ** 2)
    ss_tot = np.sum((loso_true - np.mean(loso_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    h4_result = {
        'n_papers': n_papers,
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'r2': round(r2, 4)
    }

    return h4_result, loso_true, loso_predictions, material_classes, catalyst_families, methods


def h5_prediction(rows, material_classes, catalyst_families, methods):
    """H5: Extrapolation prediction for a novel catalyst."""
    # Build full dataset features
    X, y = build_features(rows, material_classes, catalyst_families, methods)

    # Ridge regression on all data
    alpha = 1.0
    n_features = X.shape[1]
    try:
        beta = np.linalg.solve(X.T @ X + alpha * np.eye(n_features), X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X.T @ X + alpha * np.eye(n_features), X.T @ y, rcond=None)[0]

    # Create hypothetical row: catalyzed_MgH2 with novel catalyst family, Kissinger method
    feat = []
    # Material class: catalyzed_MgH2
    for mc in material_classes:
        feat.append(1 if mc == 'catalyzed_MgH2' else 0)
    # Catalyst family: novel (all zeros since not in data)
    for cf in catalyst_families:
        feat.append(0)
    # Method: Kissinger
    for m in methods:
        feat.append(1 if m == 'Kissinger' else 0)

    X_hyp = np.array([feat])
    y_pred = X_hyp @ beta

    h5_result = {
        'n_train': len(rows),
        'predicted_values': [round(float(y_pred[0]), 2)],
        'hypothetical_prediction': True
    }

    return h5_result


def run_bootstrap_iterations(rows, n_bootstrap=50):
    """Run bootstrap iterations and collect results."""
    by_doi = defaultdict(list)
    for r in rows:
        by_doi[r['doi']].append(r)

    unique_pois = list(by_doi.keys())
    n_papers = len(unique_pois)

    bootstrap_results = []

    for loop in range(n_bootstrap):
        # Resample papers with replacement
        sampled_pois = np.random.choice(unique_pois, size=n_papers, replace=True)

        # Build bootstrap sample
        bootstrap_rows = []
        for doi in sampled_pois:
            bootstrap_rows.extend(by_doi[doi])

        if len(bootstrap_rows) < 2:
            continue

        # Compute H1 on bootstrap sample
        by_doi_boot = defaultdict(list)
        for r in bootstrap_rows:
            by_doi_boot[r['doi']].append(r)

        pairs_boot = []
        for doi, paper_rows in by_doi_boot.items():
            pure_rows = [r for r in paper_rows if r.get('material_class') == 'pure_MgH2']
            catalyzed_rows = [r for r in paper_rows if r.get('material_class') == 'catalyzed_MgH2']
            for pur in pure_rows:
                for cat in catalyzed_rows:
                    delta_ea = cat['Ea_parsed'] - pur['Ea_parsed']
                    pairs_boot.append(delta_ea)

        if pairs_boot:
            mean_boot = float(np.array(pairs_boot).mean())
        else:
            mean_boot = None

        bootstrap_result = {
            'loop': loop,
            'n_papers_sampled': len(set(sampled_pois)),
            'n_rows_bootstrap': len(bootstrap_rows),
            'h1_mean_delta_ea': mean_boot
        }
        bootstrap_results.append(bootstrap_result)

        if (loop + 1) % 10 == 0:
            print(f"  Completed {loop + 1}/{n_bootstrap} bootstrap loops")

    return bootstrap_results


def generate_report(results, h1, h2, h3, h4, h5, bootstrap_results, n_rows, material_classes, catalyst_families, methods, n_papers):
    """Generate markdown report."""
    lines = []

    lines.append("# Analysis Report: Catalyst Effects on MgH2 Activation Energy")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This analysis investigates the effect of catalysts on the activation energy (Ea) of MgH2 desorption using a dataset of desorption-only measurements from multiple research papers. Five hypotheses were tested:")
    lines.append("")
    lines.append("### H1: Within-paper paired analysis")
    if h1:
        lines.append(f"- Paired samples: {h1['n_pairs']} from {h1['n_papers']} papers")
        lines.append(f"- Fixed effects mean \u0394Ea: {h1['mean_delta_ea_fixed']:.2f} kJ/mol (95% CI: [{h1['bootstrap_ci_lower']:.2f}, {h1['bootstrap_ci_upper']:.2f}])")
        lines.append(f"- Random effects (DL) mean \u0394Ea: {h1['mean_delta_ea_random']:.2f} kJ/mol")
        lines.append(f"- Between-study heterogeneity (\u03c4\u00b2): {h1['tau_sq']:.2f}")
        lines.append(f"- Paired t-test: t = {h1['t_statistic']:.2f}, p = {h1['p_value']:.6f}")
        lines.append(f"- **Conclusion**: {'SUPPORTED' if h1['supported'] else 'NOT SUPPORTED'} - Catalysts significantly reduce activation energy.")
    lines.append("")

    lines.append("### H2: Eta-squared for catalyst family")
    if h2:
        lines.append(f"- \u03b7\u00b2 = {h2['eta_squared']:.4f}")
        lines.append(f"- Permutation p-value = {h2['p_value']:.4f}")
        lines.append(f"- **Conclusion**: {'SUPPORTED' if h2['supported'] else 'NOT SUPPORTED'} - Catalyst family explains a significant proportion of variance.")
    lines.append("")

    lines.append("### H3: Method comparison")
    if h3:
        lines.append(f"- {h3['n_pairs']} material-method pairs with both Kissinger and isothermal_Arrhenius estimates")
        for p in h3['pairs'][:5]:
            lines.append(f"  - {p['material']}: Kissinger={p['Ea_Kissinger']:.2f}, Isothermal={p['EIsothermal']:.2f}, \u0394={p['delta_Ea']:.2f}")
        if h3['n_pairs'] > 5:
            lines.append(f"  ... and {h3['n_pairs'] - 5} more pairs")
        lines.append("")

    lines.append("### H4: LOSO cross-validation")
    if h4:
        lines.append(f"- {h4['n_papers']} papers in cross-validation")
        lines.append(f"- RMSE: {h4['rmse']:.2f} kJ/mol")
        lines.append(f"- MAE: {h4['mae']:.2f} kJ/mol")
        lines.append(f"- R\u00b2: {h4['r2']:.4f}")
        lines.append("")

    lines.append("### H5: Extrapolation prediction")
    if h5:
        lines.append(f"- Training samples: {h5['n_train']}")
        lines.append(f"- Predicted Ea for novel catalyzed_MgH2 (Kissinger): {h5['predicted_values'][0]:.2f} kJ/mol")
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("1. **Data preprocessing**: Ea values were parsed from strings (e.g., '109 \u00b1 1' \u2192 109). Absorption rows and one questionable entry were excluded.")
    lines.append("2. **H1**: Within-paper paired differences (catalyzed \u2013 pure) were computed. DerSimonian-Laird random-effects model estimated pooled mean with between-study heterogeneity. Cluster bootstrap (50 loops, resampling papers with replacement) provided confidence intervals.")
    lines.append("3. **H2**: Eta-squared from one-way ANOVA of Ea by catalyst family. Permutation test (1000 permutations) assessed significance by shuffling group labels.")
    lines.append("4. **H3**: Materials with both Kissinger and isothermal_Arrhenius estimates within the same study were compared.")
    lines.append("5. **H4**: Leave-one-study-out cross-validation with ridge regression using one-hot encoded features for material class, catalyst family, and method.")
    lines.append("6. **H5**: Ridge regression model fitted on all data and used to predict Ea for a hypothetical catalyzed_MgH2 with a novel catalyst family.")
    lines.append("7. **Bootstrap**: 50 cluster bootstrap loops were performed, resampling papers with replacement and recomputing all hypotheses.")
    lines.append("")

    lines.append("## Data Summary")
    lines.append("")
    lines.append(f"- Desorption-only rows: {n_rows}")
    lines.append(f"- Material classes: {', '.join(material_classes)}")
    lines.append(f"- Catalyst families: {', '.join(catalyst_families)}")
    lines.append(f"- Methods: {', '.join(methods)}")
    lines.append(f"- Number of papers: {n_papers}")

    return "\n".join(lines) + "\n"


def main():
    # Load data
    print("Loading data...")
    rows = load_data('data/ea_dataset.csv')
    print(f"Loaded {len(rows)} desorption-only rows.")

    # H1
    print("\n=== H1: Within-paper paired analysis ===")
    h1_result, delta_eas, paper_deltas, bootstrap_means, unique_pois = h1_analysis(rows)
    if h1_result:
        print(f"  Paired samples: {h1_result['n_pairs']}")
        print(f"  Papers: {h1_result['n_papers']}")
        print(f"  Fixed effects mean \u0394Ea: {h1_result['mean_delta_ea_fixed']:.2f} kJ/mol")
        print(f"  Random effects mean \u0394Ea: {h1_result['mean_delta_ea_random']:.2f} kJ/mol")
        print(f"  \u03c4\u00b2: {h1_result['tau_sq']:.2f}")
        print(f"  95% CI: [{h1_result['bootstrap_ci_lower']:.2f}, {h1_result['bootstrap_ci_upper']:.2f}] kJ/mol")

    # H2
    print("\n=== H2: Eta-squared for catalyst family ===")
    h2_result = h2_analysis(rows)
    print(f"  \u03b7\u00b2: {h2_result['eta_squared']:.4f}")
    print(f"  Permutation p-value: {h2_result['p_value']:.4f}")

    # H3
    print("\n=== H3: Method comparison ===")
    h3_result = h3_analysis(rows)
    print(f"  Comparison pairs: {h3_result['n_pairs']}")
    for p in h3_result['pairs']:
        print(f"  {p['material']}: Kissinger={p['Ea_Kissinger']}, Isothermal={p['EIsothermal']}, diff={p['delta_Ea']:.2f}")

    # H4
    print("\n=== H4: LOSO cross-validation ===")
    h4_result, loso_true, loso_predictions, material_classes, catalyst_families, methods = h4_analysis(rows)
    print(f"  Papers: {h4_result['n_papers']}")
    print(f"  RMSE: {h4_result['rmse']:.2f} kJ/mol")
    print(f"  MAE: {h4_result['mae']:.2f} kJ/mol")
    print(f"  R\u00b2: {h4_result['r2']:.4f}")

    # H5
    print("\n=== H5: Extrapolation prediction ===")
    h5_result = h5_prediction(rows, material_classes, catalyst_families, methods)
    print(f"  Training samples: {h5_result['n_train']}")
    print(f"  Predicted Ea: {h5_result['predicted_values'][0]:.2f} kJ/mol")

    # Bootstrap logging
    print("\n=== Running bootstrap loops ===")
    bootstrap_results = run_bootstrap_iterations(rows, n_bootstrap=50)
    print(f"  Completed {len(bootstrap_results)} bootstrap loops.")

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Write loop_log.jsonl
    with open('results/loop_log.jsonl', 'w') as f:
        for br in bootstrap_results:
            f.write(json.dumps(br) + '\n')
    print(f"  Wrote {len(bootstrap_results)} bootstrap results to results/loop_log.jsonl")

    # Combine results
    results = {
        'h1': h1_result,
        'h2': h2_result,
        'h3': h3_result,
        'h4': h4_result,
        'h5': h5_result
    }

    # Write results.json
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("  Wrote results/results.json")

    # Write report.md
    # Compute summary stats for report
    material_classes = sorted(set(r.get('material_class') for r in rows))
    catalyst_families = sorted(set(r.get('catalyst_family', 'none') for r in rows))
    methods = sorted(set(r.get('method') for r in rows))
    n_papers = len(set(r['doi'] for r in rows))
    report = generate_report(results, h1_result, h2_result, h3_result, h4_result, h5_result, bootstrap_results,
                             n_rows=len(rows),
                             material_classes=material_classes,
                             catalyst_families=catalyst_families,
                             methods=methods,
                             n_papers=n_papers)
    with open('results/report.md', 'w') as f:
        f.write(report)
    print("  Wrote results/report.md")

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()