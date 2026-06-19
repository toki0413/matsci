# Electronic Structure Analysis

## Density of States (DOS)
- Total DOS gives the number of states per energy interval.
- Projected DOS (PDOS) decomposes states by atom/orbital, useful for identifying bonding character.
- A band gap is the energy region with zero DOS; compare the DFT gap to experimental values (GGA typically underestimates by 30–50%).

## Band Structure
- Plot energy vs. high-symmetry k-paths to identify direct/indirect gaps, dispersion, and effective masses.
- Effective mass `m*` is obtained by fitting `E(k)` near a band extremum: `1/m* = (1/ℏ²) d²E/dk²`.
- Heavy bands (small dispersion) often correlate with high Seebeck coefficients in thermoelectrics.

## Charge Density and Bader Analysis
- Bader charge partitioning estimates oxidation states and charge transfer.
- Charge-density difference plots (`ρ(AB) - ρ(A) - ρ(B)`) visualize bonding.

## Wavefunction Analysis
- Projected band structures and fatbands reveal orbital contributions.
- ELF (electron localization function) identifies covalent bonds, lone pairs, and ionic cores.

## Functionals and Corrections
- GGA band gaps are usually too small; hybrid functionals (HSE06) and GW improve gaps.
- DFT+U corrects self-interaction for localized d/f orbitals; choose Hubbard U from linear-response or literature.

## Tools
- `pymatgen.electronic_structure`
- `Sumo`
- `VASP` + `p4vasp`
- `Quantum ESPRESSO` + `pw2wannier90`
