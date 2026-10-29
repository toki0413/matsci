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


