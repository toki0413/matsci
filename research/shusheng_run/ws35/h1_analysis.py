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


