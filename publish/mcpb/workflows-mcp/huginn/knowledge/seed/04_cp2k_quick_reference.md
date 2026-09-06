# CP2K Quick Reference

Notes for DFT and MD with CP2K, which uses Gaussian basis sets and plane waves for the electron density.

## Input sections

A CP2K input is a nested tree:

```
&GLOBAL
  PROJECT myrun
  RUN_TYPE ENERGY_FORCE
&END GLOBAL
&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    &QS
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 30
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
  &END DFT
  &SUBSYS
    &CELL
      ABC 10.0 10.0 10.0
    &END CELL
    &COORD
      Si 0.0 0.0 0.0
      ...
    &END COORD
    &KIND Si
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE
    &END KIND
  &END SUBSYS
&END FORCE_EVAL
```

## Common run types

- `RUN_TYPE ENERGY`: single-point.
- `RUN_TYPE ENERGY_FORCE`: energy + forces.
- `RUN_TYPE GEO_OPT`: geometry optimization.
- `RUN_TYPE CELL_OPT`: cell + geometry optimization.
- `RUN_TYPE MD`: molecular dynamics.
- `RUN_TYPE BSSE`: counterpoise correction.

## Basis sets and potentials

- Use `BASIS_MOLOPT` or `BASIS_MOLOPT_UCL` for molecular/condensed systems.
- Common potentials: `GTH-PBE`, `GTH-BLYP`, `GTH-PADE`.
- For transition metals, consider `ALL` or `MOLOPT` basis sets.

## Key parameters

- `EPS_DEFAULT`: overall precision threshold. Lower is tighter.
- `CUTOFF`: density cutoff (Ry). Typical 300–600 Ry.
- `REL_CUTOFF`: relative cutoff for basis set-specific density. 40–60 Ry typical.
- `EPS_SCF`: SCF convergence. 1e-6 for geometry, 1e-7 for properties.
- `NGRIDS`: multi-grid integration. 4 or 5 typical.

## MD settings

- `ENSEMBLE NVT`, `NVE`, `NPT_F`.
- `TIMESTEP`: 0.5 fs for systems with H, 1.0–2.0 fs otherwise.
- `TEMPERATURE`: target temperature in K.
- Thermostat: `NOSE`, `CSVR`, `GLE`.

## Pitfalls

- Wrong basis set/potential combination.
- Insufficient `CUTOFF` causing energy drift in MD.
- Forgetting `PERIODIC XYZ` for bulk systems.
