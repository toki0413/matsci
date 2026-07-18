# JARVIS (NIST) Materials Databases

> Data source: JARVIS (jarvis.nist.gov), NIST public domain (US gov work).

JARVIS (Joint Automated Repository for Various Integrated Simulations) is a
NIST-led multi-database infrastructure covering DFT, force fields, tight-binding,
ML models, and ChemNLP. All NIST works are U.S. public domain — no license
friction for redistribution.

## Sub-databases

| Database | Size | Method | Use case |
|---|---|---|---|
| **JARVIS-DFT** | ~75k 3D + ~2k 2D materials | VASP, optB88vDW / PBE / SCAN | Formation energy, band gap, elastic, dielectric, magnetic moment, SLME |
| **JARVIS-FF** | ~4k snapshots | LAMMPS, ~30 FF types (EAM, Tersoff, ReaxFF, COMB, eFF, MTP, SNAP...) | FF benchmark + selection per material class |
| **JARVIS-QETB** | 2D tight-binding | PythTB / Wannier | Topological invariants (Z2, Chern), edge states |
| **JARVIS-ML** | ~50k trained models | CFID descriptors + classic ML (RF, XGBoost, GP) | Property prediction without DFT |
| **ChemNLP** | ~5M materials science abstracts | BERT/SciBERT fine-tuned | Materials text mining, relation extraction |

All five feed into the **JARVIS-Leaderboard** (open benchmark, ~50 tasks).

## OPTIMADE access

Base URL: `https://jarvis.nist.gov/optimade`
- `/structures` — JARVIS-DFT entries
- `/info` — schema
- No API key required for read queries

## Direct REST API

`https://jarvis.nist.gov/rest/`

Examples:
- `/materials/{jid}` — single material (e.g. `JVASP-1002`)
- `/materials/{jid}/cif` — CIF download
- `/ml/{model_id}/{jid}` — ML prediction
- `/ff/{formula}` — list applicable force fields

## Key JARVIS-only features (not in MP/OQMD)

- **SLME (Spectral Limiting Maximum Efficiency)**: solar-cell efficiency limit
  per material, more accurate than Shockley-Queisser for direct-gap absorbers.
- **CFID (Classical Force-field Inspired Descriptors)**: 1,557 descriptors
  combining structural, chemical, and electronic features. Use for
  interpretable property prediction; CFID + XGBoost often beats deep learning
  on tabular property data.
- **FF benchmark matrix**: every FF evaluated against DFT reference across
  multiple properties (lattice, elastic, surface, defect). Use to pick the
  right FF for a material class before running MD.
- **Topological invariants**: pre-computed Z2 and Chern numbers for 2D materials,
  enables direct screening of topological insulators.
- **ChemNLP**: SciBERT fine-tuned on materials abstracts. Use for relation
  extraction ("material X has band_gap Y"), entity recognition, abstract
  classification.

## When to use JARVIS vs MP

- **Use JARVIS when**: you need FF selection guidance, topological invariants,
  SLME, CFID descriptors, or 2D materials focus.
- **Use MP when**: you need largest 3D bulk coverage, phase diagrams, or the
  full MP REST suite (charge density, DOS, band structure images).
- **Use both**: when cross-validating formation energies or screening
  (agreement across independent DFT stacks is a strong correctness signal).

## JARVIS-Leaderboard

Open benchmark, ~50 tasks across DFT/FF/QETB/ML. Submission format:
predictions on a fixed test set; ranked by MAE/RMSE/Accuracy. Useful as
agent-internal reference for "what's the current SOTA on property X".

URL: https://jarvis.nist.gov/jarvisleaderboard

## Tools

- `jarvis-tools` (PyPI, NIST public domain) — Python client, structure I/O,
  CFID descriptors, analysis. `pip install jarvis-tools`.
- `matminer` (BSD-3) — composes with JARVIS for featurization pipelines.

## Sources

- Main: https://jarvis.nist.gov (NIST public domain)
- GitHub: https://github.com/usnistgov/jarvis (NIST public domain)
- Paper: Choudhary et al., npj Comput. Mater. 6, 173 (2020)
- Leaderboard: https://jarvis.nist.gov/jarvisleaderboard
