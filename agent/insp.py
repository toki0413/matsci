import json

d = json.load(open("nlse_symbolic.json"))
print("=== SOLITON DERIVATION ===")
print(json.dumps(d["soliton_derivation"], indent=2))
print()
print("=== FIELD-THEORETIC NOETHER ===")
print(json.dumps(d["field_theoretic_noether"], indent=2))
print()
print("=== NOETHER TOOL ===")
print(json.dumps(d["noether_tool"], indent=2))
print()
print("=== CLASSIFY ===")
print(json.dumps(d["pde_classify"], indent=2))