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


