# Physical Constants and Unit Conventions

Use these values and conversions when setting up computational materials simulations.
Values are CODATA 2022 (NIST, public domain). Since the 2019 SI redefinition,
h, e, k_B, N_A are exact; m_e and c carry experimental uncertainty.

## SI 2019 defining constants (exact)

| Constant | Symbol | Exact value | Unit |
|---|---|---|---|
| Planck constant | h | 6.626 070 15 × 10⁻³⁴ | J·s |
| Elementary charge | e | 1.602 176 634 × 10⁻¹⁹ | C |
| Boltzmann constant | k_B | 1.380 649 × 10⁻²³ | J/K |
| Avogadro constant | N_A | 6.022 140 76 × 10²³ | mol⁻¹ |
| Cs hyperfine frequency | Δν_Cs | 9 192 631 770 | Hz |
| Speed of light | c | 299 792 458 | m/s |
| Luminous efficacy | K_cd | 683 | lm/W |

These seven constants define the seven SI base units. m_e is no longer exact
(CODATA 2022 uncertainty 3.0 × 10⁻¹⁰).

## Derived constants (CODATA 2022, with uncertainty)

| Constant | Symbol | Value | Uncertainty | Units |
|---|---|---|---|---|
| Reduced Planck | ℏ | 1.054 571 817… × 10⁻³⁴ | exact (h/2π) | J·s |
| Electron mass | m_e | 9.109 383 7139 × 10⁻³¹ | 2.8 × 10⁻¹³ | kg |
| Proton mass | m_p | 1.672 621 9259 × 10⁻²⁷ | 5.7 × 10⁻¹⁰ | kg |
| Hartree energy | E_h | 4.359 744 7222 × 10⁻¹⁸ | 3.9 × 10⁻¹² | J |
| Bohr radius | a_0 | 5.291 772 1057 × 10⁻¹¹ | 9.3 × 10⁻¹³ | m |
| Magnetic flux quantum | Φ_0 | 2.067 833 848… × 10⁻¹⁵ | exact (h/2e) | Wb |
| Bohr magneton | μ_B | 9.274 010 0657 × 10⁻²⁴ | 2.9 × 10⁻¹³ | J/T |
| Nuclear magneton | μ_N | 5.050 783 7393 × 10⁻²⁷ | 1.6 × 10⁻¹³ | J/T |
| Fine-structure | α | 7.297 352 5643 × 10⁻³ | 1.1 × 10⁻¹² | dimensionless |
| Rydberg constant | R_∞ | 10 973 731.568 157 | 6.4 × 10⁻⁹ | m⁻¹ |
| Molar gas constant | R | 8.314 462 618… | exact (N_A·k_B) | J/(mol·K) |
| Faraday constant | F | 96 485.332 12… | exact (N_A·e) | C/mol |
| Stefan-Boltzmann | σ | 5.670 374 419… × 10⁻⁸ | exact | W/(m²·K⁴) |
| Vacuum permittivity | ε_0 | 8.854 187 8128… × 10⁻¹² | exact (1/μ_0c²) | F/m |
| Vacuum permeability | μ_0 | 1.256 637 06172… × 10⁻⁶ | derived | H/m |

## Atomic units (Hartree units, default in DFT)

| Quantity | Atomic unit | SI equivalent |
|---|---|---|
| Length | a_0 (Bohr) | 0.529 177 210 903 Å |
| Energy | E_h (Hartree) | 27.211 386 245 988 eV |
| Mass | m_e | 9.109 383 7 × 10⁻³¹ kg |
| Charge | e | 1.602 176 634 × 10⁻¹⁹ C |
| Time | ℏ/E_h | 2.418 884 326 × 10⁻¹⁷ s |
| Velocity | a_0·E_h/ℏ = αc | 2.187 691 263 × 10⁶ m/s |

In atomic units, ℏ = m_e = e = a_0 = 1; energies in Hartree, lengths in Bohr.

## Common unit conversions

- 1 eV = 96.485 332 12 kJ/mol = 23.060 549 kcal/mol
- 1 eV/Å³ = 160.217 662 08 GPa
- 1 Hartree = 27.211 386 245 988 eV = 4.359 744 722 2 × 10⁻¹⁸ J
- 1 Ry = 0.5 Hartree = 13.605 693 122 994 eV
- 1 Bohr = 0.529 177 210 903 Å
- 1 Å = 10⁻¹⁰ m = 0.1 nm
- 1 ps = 10⁻¹² s; 1 fs = 10⁻¹⁵ s
- 1 T = 10⁴ G; 1 μ_B = 9.274 × 10⁻²⁴ J/T

## Temperature and energy

- Thermal energy at 300 K: k_B T ≈ 0.025 852 eV ≈ 1/40 eV
- 1 eV/k_B = 11 604.518 K (energy ↔ temperature)
- 1 K = 0.086 173 meV (per particle)
- Use these to judge whether an energy difference is physically significant.

## Useful constants combinations

- m_e c² = 510 998.950 00 eV (electron rest energy)
- E_h/(2ℏ) = 6.579 683 920 × 10¹⁵ Hz (Hartree frequency, for optical transitions)
- α² m_e c² = E_h (fine-structure consistency check)
- k_B T at 298.15 K = 0.025 692 579 eV (room-temperature thermal energy, standard state)
- (k_B T/q) ln(10) = 0.059 157 V at 298.15 K (Nernst slope, 25 °C, 1 electron)

## Advice

- Always state units in input files. VASP, LAMMPS, QE, CP2K use different defaults.
- For DFT, atomic units (Hartree/Bohr) are canonical; report in eV/Å for readability.
- When publishing, cite CODATA 2022 (Tiesinga et al., Rev. Mod. Phys. 93, 025010, 2021).
- For SI-traceable uncertainty propagation, use the full CODATA 2022 covariance matrix
  (download from physics.nist.gov/cuu/Constants), not just the marginal uncertainties above.
