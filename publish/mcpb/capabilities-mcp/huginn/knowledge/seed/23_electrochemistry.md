# Electrochemistry for Materials

## Key Quantities
- **Reduction potential (E°)**: Thermodynamic driving force for a redox reaction vs. a reference electrode (SHE, Ag/AgCl, Li/Li⁺).
- **Overpotential (η)**: Extra potential beyond equilibrium required to drive a reaction at a finite rate; η = E_applied − E_eq.
- **Exchange current density (i₀)**: Measures intrinsic electrode kinetic activity.
- **Tafel slope**: Relates overpotential to log current; smaller slope indicates faster kinetics.

## Computational Approaches
- **DFT for electrode potentials**: Calculate free-energy diagrams for reaction intermediates; use computational hydrogen electrode (CHE) for pH/ potential references.
- **Pourbaix diagrams**: Map stable phases as a function of potential and pH.
- **Explicit solvation / implicit solvation models**: Account for electrolyte effects on reaction barriers and adsorption energies.

## Battery-Specific Concepts
- **Voltage profile**: Average voltage = −ΔG / (nF) for Li/Na insertion.
- ** Ionic conductivity**: From AIMD via Nernst–Einstein or hopping models.
- **Solid-electrolyte interphase (SEI)**: Formed by electrolyte reduction at low potentials; controls long-term cycling stability.

## Tools
- `pymatgen.analysis.pourbaix`
- `VASP` + implicit solvation (VASPsol)
- `CP2K` for explicit water/electrolyte interfaces
- `ASE` for reaction-path sampling
