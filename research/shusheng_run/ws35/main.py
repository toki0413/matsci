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