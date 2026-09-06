---
name: gromacs-solvation-environment
description: Inspect GROMACS TPR topology and compute fixed-radius first-shell solvation/coordination-environment compositions with MDAnalysis using atom-to-molecule-COG or molecule-COG-to-molecule-COG distances. Use when Codex receives a TPR, a user-supplied shell radius, an approximate center atom or molecule name, and an optional XTC/GRO, and needs per-frame environment records, type distributions, validation, and publication-style PNG figures.
---

# GROMACS Solvation Environment

## Workflow

1. Collect the TPR path, shell radius in nm, approximate center description, optional XTC/GRO, sampling, output directory, and plotting choice.
2. Read `references/methods_and_schema.md` before using the scripts.
3. Run only `scripts/gromacs_solvation_environment.py inspect --tpr <file.tpr>` before confirmation.
4. Infer the center type and explicit MDAnalysis selection from the reported `moltype`, `resname`, and `atomname` using Agent reasoning. Do not implement fuzzy-name rules or chemical alias dictionaries.
5. Present one complete settings summary: paths/mode, radius, original center description, inferred type and actual TPR names, selection, distance definition, sampling, output directory, and plotting.
6. Obtain explicit user confirmation. **Do not run `analyze`, generate results, or modify user data before confirmation.**
7. If any setting changes, present the entire updated summary and confirm again.
8. After confirmation, run `analyze` exactly with the confirmed parameters. Verify event count, fraction sum, output encoding, PNG DPI, and absence of unexpected PDFs.

## Method Rules

- Require MDAnalysis >= 2.10.0 and use TPR `molnum`/`moltype` as the only molecule/species identities.
- Use an atom coordinate for `center-mode=atom`; use the complete molecule COG for `center-mode=molecule`.
- Use complete-molecule COG for every neighbor.
- Reconstruct every multi-atom molecule across PBC from TPR bonds before computing COG. Fail if its bond graph is disconnected.
- Exclude the center's own `molnum`; retain other molecules of the same `moltype`.
- Use the current frame box and reject a cutoff larger than half the shortest lattice-vector length.
- With no XTC/GRO, use the TPR current frame. If unavailable, require GRO or XTC.
- With no sampling option, process all stored frames with `stride=1`.
- Keep CSV output RFC 4180 compliant with UTF-8 BOM and CRLF.
- Export PNG only at 600 DPI, white background, Arial 7 pt, and 0.75 pt axes.

## Scripts

- `scripts/gromacs_solvation_environment.py`: deterministic `inspect` and `analyze` CLI; self-contained plotting.
- `scripts/smoke_test_gromacs_solvation.py`: in-memory regression test for PBC, atom/molecule centers, sampling, outputs, and figures.

## Outputs

- `solvation_environment_records.csv`
- `solvation_environment_distribution.csv`
- `solvation_environment_summary.json`
- `fig_coordination_environment_polar.png` when plotting is confirmed
- `fig_coordination_environment_distribution.png` when plotting is confirmed

Read `README.md` for installation and end-to-end demonstrations. Read `references/methods_and_schema.md` for the confirmation template, CLI contract, algorithm, and output fields.
