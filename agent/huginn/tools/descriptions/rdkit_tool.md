# RDKit Cheminformatics Tool

`rdkit_tool` does molecular manipulation for drug discovery and small-molecule
chemistry via RDKit: SMILES parsing, descriptors, fingerprints, similarity,
substructure search, 2D depiction, and 3D conformer generation.

## Actions

| action | what it does | key inputs |
|---|---|---|
| `smiles_to_mol` | parse a SMILES, report canonical SMILES / formula / atom & ring counts | `smiles` |
| `descriptors` | compute physicochemical descriptors (MW, LogP, TPSA, HBD/HBA, rotatable bonds, ...) | `smiles` |
| `fingerprint` | generate a molecular fingerprint | `smiles`, `fingerprint_type`, `radius`, `n_bits` |
| `similarity` | Tanimoto similarity between two molecules | `query_smiles`, `reference_smiles` |
| `substructure_search` | does a query contain a substructure? | `smiles`, `substructure` |
| `draw` | render a 2D depiction (PNG) | `smiles`, `output_file`, `image_size` |
| `conformers` | generate 3D conformers (optional MMFF94 optimization) | `smiles`, `n_conformers`, `optimize` |
| `smiles_to_sdf` | write an SDF file (single or batch) | `smiles` / `smiles_list`, `output_file` |

## Typical use

- Quickly profile a candidate molecule: `smiles_to_mol` -> `descriptors`.
- Build a similarity matrix across a ligand set with `fingerprint` + `similarity`.
- Filter a library by pharmacophore/substructure with `substructure_search`.
- Generate 3D starting geometries for docking (`conformers`) or a 2D image for
  a report (`draw`).

## Notes

- Light cost tier; HYPOTHESIS and PLANNING phases.
- RDKit is imported lazily — the tool loads even if rdkit isn't installed, and
  only the requested action fails with an install hint.
- Read-only except `draw`, `conformers`, and `smiles_to_sdf`, which write files.
