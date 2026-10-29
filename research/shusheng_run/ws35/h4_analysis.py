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


