# Quantum ESPRESSO Quick Reference

Essential settings for pw.x, relaxation, and band/DOS calculations.

## Input file structure

```fortran
&CONTROL
  calculation = 'scf'
  prefix = 'myrun'
  pseudo_dir = './pseudo'
  outdir = './tmp'
/
&SYSTEM
  ibrav = 0
  nat = 4
  ntyp = 2
  ecutwfc = 60
  ecutrho = 480
  occupations = 'smearing'
  smearing = 'mp'
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
 Si 28.085 Si.pbe-n-rrkjus_psl.1.0.0.UPF
 O  15.999 O.pbe-n-rrkjus_psl.1.0.0.UPF
ATOMIC_POSITIONS angstrom
 ...
K_POINTS automatic
 6 6 6 0 0 0
CELL_PARAMETERS angstrom
 ...
```

## Common calculations

- `calculation = 'scf'`: single-point energy.
- `calculation = 'relax'`: relax atomic positions.
- `calculation = 'vc-relax'`: relax positions and cell.
- `calculation = 'bands'`: band structure (needs `nbnd`).
- `calculation = 'nscf'`: non-self-consistent for DOS.
- `calculation = 'md'`: Born-Oppenheimer MD.

## Key parameters

- `ecutwfc`: wavefunction cutoff (Ry). Converge carefully.
- `ecutrho`: charge-density cutoff, typically 8–12× `ecutwfc`.
- `conv_thr`: SCF convergence threshold (Ry).
- `degauss`: smearing width (Ry). Metals: 0.01–0.02 Ry; insulators: smaller or no smearing.
- `mixing_beta`: 0.7 typical; reduce if SCF diverges.
- `diagonalization = 'david'`: robust for medium systems; `cg` for tough cases.

## K-points

- Use `K_POINTS automatic` with Monkhorst-Pack grid.
- `6 6 6 0 0 0` means 6×6×6 with no offset.
- For hexagonal cells, use offset `0 0 0` or `1 1 1` as recommended.

## Post-processing

- DOS: `pw.x` nscf then `dos.x`.
- Bands: `pw.x` nscf then `bands.x`.
- Phonons: `ph.x` with `q2r.x` and `matdyn.x`.

## Troubleshooting

- `charge sloshing`: reduce `mixing_beta`, increase `ecutrho`, or use `mixing_mode = 'local-TF'`.
- `negative or imaginary phonon frequencies`: may indicate an unstable structure or insufficient k-sampling.
