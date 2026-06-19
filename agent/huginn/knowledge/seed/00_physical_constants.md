# Physical Constants and Unit Conventions

Use these values and conversions when setting up computational materials simulations.

## Fundamental constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Planck constant | h | 6.62607015e-34 | J·s |
| Reduced Planck constant | ℏ | 1.054571817e-34 | J·s |
| Elementary charge | e | 1.602176634e-19 | C |
| Electron mass | m_e | 9.1093837015e-31 | kg |
| Boltzmann constant | k_B | 1.380649e-23 | J/K |
| Avogadro constant | N_A | 6.02214076e23 | mol^-1 |
| Hartree energy | E_h | 27.211386245988 | eV |
| Bohr radius | a_0 | 0.529177210903 | Å |
| Speed of light | c | 299792458 | m/s |

## Common unit conversions

- 1 eV = 96.48533212 kJ/mol
- 1 eV/Å^3 = 160.21766208 GPa
- 1 Hartree = 27.211386245988 eV
- 1 Ry = 0.5 Hartree = 13.605693122994 eV
- 1 Å = 1e-10 m = 0.1 nm
- 1 ps = 1e-12 s
- 1 fs = 1e-15 s

## Temperature and energy

- Thermal energy at room temperature (300 K): k_B T ≈ 0.02585 eV ≈ 1/40 eV.
- Use this to judge whether a simulation energy difference is physically significant.

## Advice

Always state units in input files. Many codes (VASP, LAMMPS, Quantum ESPRESSO) use different defaults; mixing units is a common source of wrong results.
