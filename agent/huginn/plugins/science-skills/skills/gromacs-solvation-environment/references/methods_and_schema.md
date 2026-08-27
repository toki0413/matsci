# GROMACS Fixed-Radius Solvation Environment

## Scope and dependency boundary

Use this workflow when the user already provides a first-shell/RDF radius and wants the composition around one or more center atoms or molecules.

- Required: GROMACS TPR, radius in nm, approximate center description.
- Optional and mutually exclusive: XTC or GRO.
- Core runtime: Python standard library, NumPy, MDAnalysis >= 2.10.0.
- Plotting: uses the self-contained Matplotlib functions in `gromacs_solvation_environment.py` only when `--plot` is confirmed.
- Do not require ITP, NDX, MOL2, VMD, `System.xlsx`, or a species map.
- Treat TPR `molnum` as molecule identity and TPR `moltype` as species identity. Do not silently substitute residue identity.
- The user owns the supplied radius. Do not infer or validate its RDF provenance.

## Mandatory confirmation gate

`inspect` is read-only and may run before confirmation. `analyze` must not run before the user explicitly confirms the complete settings summary.

Use this template:

```text
请确认固定半径溶剂化环境统计设置：
- TPR: <absolute path>
- Coordinates/trajectory: <TPR current frame | GRO path | XTC path>
- Mode: <snapshot | trajectory>
- First-shell radius: <value> nm
- Original center description: <verbatim user text>
- Inferred center type: <atom | molecule>
- Resolved TPR names: moltype=<...>, resname=<...>, atomname=<...>
- MDAnalysis selection: <explicit selection>
- Distance definition: <atom–molecule COG | molecule COG–molecule COG>
- Sampling: <stride=1/all stored frames | stride=N | N uniformly sampled frames>
- Output directory: <absolute path>
- Plot: <yes | no>

Only after you explicitly confirm these settings will I run the analysis.
```

Infer the center with Agent reasoning after reading `inspect`; do not build fuzzy-name rules, alias dictionaries, or chemistry matching into the script. If more than one plausible selection remains, explain the ambiguity and ask the user to choose. If any setting changes, present the entire updated summary and confirm again.

## Read-only inspection

```bash
python \
  scripts/gromacs_solvation_environment.py inspect \
  --tpr /absolute/path/system.tpr
```

The JSON report includes:

- `moltypes`: molecule counts, atoms per molecule, associated residue and atom names;
- `resnames`: residue counts, associated molecule types and atom names;
- `atomnames`: counts and associated residue/molecule types;
- `capabilities`: availability of `molnum`, `moltype`, bonds, current coordinates, and box;
- topology/frame counts.

Inspection performs no coordination counting and writes no files.

## Confirmed analysis CLI

Trajectory example:

```bash
python \
  scripts/gromacs_solvation_environment.py analyze \
  --tpr /absolute/path/system.tpr \
  --xtc /absolute/path/traj.xtc \
  --rdf-radius-nm 0.35 \
  --center-mode atom \
  --center-selection 'name LI' \
  --stride 1 \
  --output-dir /absolute/path/solvation_environment \
  --plot
```

Snapshot example using a molecule center:

```bash
python \
  scripts/gromacs_solvation_environment.py analyze \
  --tpr /absolute/path/system.tpr \
  --gro /absolute/path/snapshot.gro \
  --rdf-radius-nm 0.45 \
  --center-mode molecule \
  --center-selection 'moltype EC' \
  --output-dir /absolute/path/solvation_environment
```

Interface rules:

- `--xtc` and `--gro` are optional and mutually exclusive. With neither, read the current frame stored in the TPR; if unavailable, stop and request GRO/XTC.
- `--stride` and `--n-frames` are optional and mutually exclusive.
- With neither sampling option, process every stored frame (`stride=1`).
- `--stride N` samples frame indices `0, N, 2N, ...`.
- `--n-frames N` selects N unique, uniformly spaced indices including the first and last when `N > 1`. It is an error if N exceeds stored frames.
- `--plot` is opt-in. Never add it unless plotting was explicitly included in the confirmed settings.

## Method

MDAnalysis uses Å internally; the script converts `--rdf-radius-nm` to Å before searching.

1. Group every atom by TPR `molnum` and require exactly one `moltype` per molecule.
2. Require every multi-atom molecule to be connected by TPR bonds.
3. For each frame, reconstruct each multi-atom molecule across PBC along a bond spanning tree. Apply the minimum-image displacement to every traversed bond, then compute the unweighted arithmetic mean of all molecule atom coordinates (COG).
4. For `center-mode=atom`, every atom selected by `center-selection` is an independent center and uses its atom coordinate.
5. For `center-mode=molecule`, use selected atoms only to identify unique `molnum` values; expand each to the complete molecule and use its COG.
6. Use every system molecule COG as a potential neighbor. Perform a periodic sparse cutoff search with the current frame box.
7. Exclude the center's own `molnum`; retain other molecules with the same `moltype`.
8. Count each neighbor `molnum` at most once per center-frame event.

The radius must be positive and no greater than half the shortest current-frame lattice-vector length. A missing/invalid box or a disconnected multi-atom molecule is a hard error because a reliable PBC COG cannot then be defined.

## Output schema

All CSV files use RFC 4180 quoting, comma delimiters, UTF-8 with BOM, and CRLF line endings. JSON is UTF-8.

### `solvation_environment_records.csv`

One row per sampled frame and center:

- identity/time: `event_id`, `sample_index`, `frame_index`, `time_ps`;
- center: `center_id`, `center_mode`, `center_atom_index`, `center_molnum`, `center_moltype`, `center_atomname`, `center_resname`;
- environment: `total_neighbor_molecules`, `environment_type_id`, `composition_json`;
- one integer `count::<moltype>` column per species in stable TPR molecule order.

### `solvation_environment_distribution.csv`

One row per unique full composition vector:

- `environment_type_id`, `count`, `fraction`, `total_neighbor_molecules`;
- `composition_vector`, `composition_json`;
- the same `count::<moltype>` columns as the records table.

Sort by descending `count`, then lexicographic composition vector. Environment ids start at 1.

### `solvation_environment_summary.json`

Contains the confirmed CLI settings, actual center selection resolution, distance method, `species_order`, sampled frame indices, environment distribution, output paths, and validation fields.

Required validation checks:

- `actual_total_events == center_count * sampled_frame_count`;
- environment fractions sum to 1 within absolute tolerance `1e-12`;
- molecule/species identities are `molnum`/`moltype`;
- PBC bond reconstruction is enabled.

### `fig_coordination_environment_polar.png`

Generated only with `--plot`: white background, Arial 7 pt, 0.75 pt frame conventions, PNG at 600 DPI. Do not generate PDF.

### `fig_coordination_environment_distribution.png`

Generated together with the polar plot when `--plot` is enabled. Show the 11 highest-frequency environments as readable composition labels and aggregate the remaining long tail as `Other`; keep percentages normalized by all center-frame events. The full, unaggregated distribution remains in CSV/JSON.

## Validation commands

```bash
python scripts/smoke_test_gromacs_solvation.py
```

The new smoke test uses an in-memory topology to verify atom and molecule centers, cross-PBC COG, self exclusion, same-species retention, molecule deduplication, sampling, CSV encoding, invariants, and PNG DPI. For release validation with real data, additionally compare a small TPR snapshot and TPR+XTC run against manual molecule counting.
