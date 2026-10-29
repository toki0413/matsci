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


