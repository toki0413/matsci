import csv
import json

# Read the CSV file
file_path = '/workspace/research/shusheng_run/ws35/data/cleaned_ea_dataset.csv'
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")
print(f"Columns: {list(rows[0].keys())}")

# Filter desorption only
desorption_rows = [r for r in rows if r.get('direction', '') == 'desorption']
print(f"\nDesorption rows: {len(desorption_rows)}")

# Check material classes
material_classes = set(r['material_class'] for r in desorption_rows)
print(f"Material classes: {material_classes}")

# Check methods
methods = set(r['method'] for r in desorption_rows)
print(f"Methods: {methods}")

# Check catalyst families
catalyst_families = set(r.get('catalyst_family', 'none') for r in desorption_rows)
print(f"Catalyst families: {catalyst_families}")

# Check DOIs
dois = set(r['doi'] for r in desorption_rows)
print(f"Number of DOIs: {len(dois)}")

# Check for paired samples (same doi with both pure_MgH2 and catalyzed_MgH2)
by_doi = {}
for r in desorption_rows:
    doi = r['doi']
    if doi not in by_doi:
        by_doi[doi] = []
    by_doi[doi].append(r)

paired_papers = []
for doi, paper_rows in by_doi.items():
    pure = [r for r in paper_rows if r['material_class'] == 'pure_MgH2']
    cat = [r for r in paper_rows if r['material_class'] == 'catalyzed_MgH2']
    if pure and cat:
        paired_papers.append(doi)
        print(f"\nDOI {doi}: pure={len(pure)}, catalyzed={len(cat)}")
        for p in pure:
            ea_p = float(p.get('Ea_parsed', p.get('Ea_kJ_per_mol', 0)))
            print(f"  pure: Ea={ea_p}")
        for c in cat:
            ea_c = float(c.get('Ea_parsed', c.get('Ea_kJ_per_mol', 0)))
            print(f"  catalyzed: Ea={ea_c}")

print(f"\nTotal paired papers: {len(paired_papers)}")

# Method pairs (Kissinger vs isothermal_Arrhenius for same material)
material_paper_methods = {}
for r in desorption_rows:
    material = r['material']
    doi = r['doi']
    method = r['method']
    ea = float(r.get('Ea_parsed', r.get('Ea_kJ_per_mol', 0)))
    key = (material, doi)
    if key not in material_paper_methods:
        material_paper_methods[key] = []
    material_paper_methods[key].append((method, ea))

method_pairs = []
for (material, doi), methods_list in material_paper_methods.items():
    kissinger_ea = None
    isothermal_ea = None
    for m, ea in methods_list:
        if m == 'Kissinger':
            kissinger_ea = ea
        elif m == 'isothermal_Arrhenius':
            isothermal_ea = ea
    if kissinger_ea is not None and isothermal_ea is not None:
        method_pairs.append({
            'material': material,
            'doi': doi,
            'Ea_Kissinger': kissinger_ea,
            'EIsothermal': isothermal_ea,
            'delta_Ea': kissinger_ea - isothermal_ea
        })

print(f"\nMethod pairs (Kissinger vs isothermal): {len(method_pairs)}")
for p in method_pairs:
    print(f"  {p['material']}: Kissinger={p['Ea_Kissinger']:.2f}, Isothermal={p['EIsothermal']:.2f}, delta={p['delta_Ea']:.2f}")

# Save results for later use
summary = {
    'total_rows': len(rows),
    'desorption_rows': len(desorption_rows),
    'material_classes': sorted(list(material_classes)),
    'methods': sorted(list(methods)),
    'catalyst_families': sorted(list(catalyst_families)),
    'n_papers': len(dois),
    'paired_papers': paired_papers,
    'method_pairs': method_pairs
}

with open('/workspace/research/shusheng_run/ws35/data/summary.json', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\nSummary saved to data/summary.json")