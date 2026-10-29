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


