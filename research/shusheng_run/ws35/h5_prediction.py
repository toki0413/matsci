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


