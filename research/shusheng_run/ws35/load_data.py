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


