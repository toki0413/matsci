# Defect Chemistry in Materials

## Point Defects
The formation energy of a point defect `D` in charge state `q` is:

`E_f(D^q) = E_tot(D^q) - E_tot(bulk) - Σ n_i μ_i + q (E_F + E_VBM) + E_corr`

- `E_F` is the Fermi level relative to the valence-band maximum.
- `E_corr` corrects for finite-size electrostatic effects (e.g., Freysoldt-Neugebauer-Van de Walle correction).

## Defect Transition Levels
- A transition level `ε(q/q')` is the Fermi-level position where charge states `q` and `q'` have equal formation energy.
- Plot `E_f` vs. `E_F` to identify which charge state is stable in different doping regimes.

## Doping Strategies
- p-type doping: introduce acceptors with low formation energy under Fermi levels near the VBM.
- n-type doping: introduce donors stable near the conduction-band minimum.
- Compensating native defects (e.g., oxygen vacancies in oxides) can pin the Fermi level and limit doping.

## Tools
- `pymatgen.analysis.defects`
- `doped`
- `ASE` for structure generation and supercell building

## Best Practices
- Use large enough supercells (> 100 atoms) and check convergence with cell size.
- Align electrostatic potentials between defect and bulk calculations.
- Sample relevant chemical potentials, including oxygen-rich and oxygen-poor limits.
