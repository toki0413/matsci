# VASP Quick Reference

Key tags, file conventions, and troubleshooting for VASP 5/6.

## Input files

- `INCAR`: control tags.
- `POSCAR`: structure (lattice, coordinates).
- `POTCAR`: pseudopotentials.
- `KPOINTS`: k-point grid.
- (Optional) `INCAR` can be split into `INCAR` + `SYSTEM` line.

## Essential INCAR tags

| Task | Tags |
|------|------|
| Single-point | `NSW = 0` |
| Ionic relaxation | `ISIF = 2`, `IBRION = 2`, `NSW = 100` |
| Cell shape/volume relax | `ISIF = 3` |
| Static DOS/band | `ICHARG = 11` after SCF relaxation |
| MD | `IBRION = 0`, `NSW`, `POTIM`, `TEBEG`, `TEEND` |
| Spin polarized | `ISPIN = 2` |
| Metallic | `ISMEAR = 1`, small `SIGMA` (0.1–0.2) |
| Insulating | `ISMEAR = 0`, `SIGMA = 0.05` |
| Accurate forces | `EDIFFG = -0.01` or tighter |

## Key tag meanings

- `ENCUT`: plane-wave cutoff (eV). Use ≥ max `ENMAX` of POTCARs, typically 1.3×.
- `ISMEAR`/`SIGMA`: smearing method and width.
- `ALGO`: algorithm. `Normal`, `Fast`, `All`, `Damped`, `Eigenval`.
- `EDIFF`: electronic convergence criterion (eV).
- `EDIFFG`: ionic relaxation convergence (negative = force, positive = energy change).
- `IBRION`: relaxation/MD algorithm.
- `ISIF`: what to relax (ions, cell shape, volume).
- `NELM`, `NELMIN`: max/min electronic steps.
- `NBANDS`: number of bands. Increase if "too many bands" or if empty states needed.

## KPOINTS advice

- Metals: dense grids, e.g., 11×11×11 Monkhorst-Pack.
- Semiconductors: 6×6×6 or 8×8×8.
- Large supercells/surfaces: Gamma-centered, reduce depth.

## Common errors

- `BRIONS internal error`: check `POSCAR` for atoms too close or unreasonable cell.
- `too many bands`: increase `NBANDS`.
- SCF oscillations: try `ALGO = Damped` or `NELMIN`.
- Wrong magnetism: set initial `MAGMOM` and use `ISPIN = 2`.
