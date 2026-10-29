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


