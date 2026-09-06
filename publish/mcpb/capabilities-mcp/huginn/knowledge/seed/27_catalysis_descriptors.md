# Catalysis Descriptors

## Adsorption-Energy Descriptors
- **d-band center**: Often correlates with adsorption strength on transition-metal surfaces.
- **CO / OH / O binding energies**: Common probes for heterogeneous catalysis.
- **Sabatier principle**: Optimal catalyst binds intermediates neither too weakly nor too strongly.

## Activity Metrics
- **Turnover frequency (TOF)**: Rate per active site.
- **Mass activity**: Current per catalyst mass (A g⁻¹).
- **Overpotential at target current density**: Practical figure of merit for ORR/OER/HER.

## Scaling Relations
- Binding energies of reaction intermediates often scale linearly (e.g., ΔG_OH vs. ΔG_O).
- Breaking scaling relations is a key strategy for discovering better catalysts.

## Computational Screening Workflow
1. Build surface slabs for candidate materials.
2. Calculate adsorption energies of key intermediates.
3. Construct free-energy diagrams at applied potential / pH.
4. Compute volcano plots and identify candidates near the peak.

## Tools
- `ASE` + `VASP`/`Quantum ESPRESSO` for slab/adsorption calculations
- `CatKit`, `Pymatgen` for surface generation
- `matplotlib` / `seaborn` for volcano plots
