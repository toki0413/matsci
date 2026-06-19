# Phase Diagram Construction

## Overview
Phase diagrams show thermodynamic stability as a function of composition, temperature, and pressure. Common construction methods include the convex hull (zero-Kelvin), finite-temperature thermodynamic integration, and CALPHAD modeling.

## Convex Hull
- For a set of candidate structures with DFT total energies, the stable phases are the vertices of the lower convex hull in composition-energy space.
- A structure above the hull has an energy of `E_hull = E(structure) - E_hull(surface)` per atom; values > ~20–40 meV/atom often indicate metastability or a missing competing phase.
- Tools: `pymatgen.analysis.phase_diagram`, `qmpy` (OQMD), `ase`.

## Chemical Potential
- Stability against decomposition into elemental reservoirs is governed by the elemental chemical potentials `μ_i`.
- For a compound `A_x B_y`, the formation energy is `E_f = E(A_xB_y) - x μ_A - y μ_B`, usually referenced to the elemental ground states.

## Finite Temperature
- Use the quasiharmonic approximation or phonon free energies to add `F_vib(T)` to DFT static energies.
- For solid solutions, configurational entropy (`k_B ln W`) and cluster expansion / Monte Carlo are common.

## Common Pitfalls
- Missing competing phases (especially polymorphs and ordered structures) can produce incorrect hulls.
- GGA functionals may misrank phases with strong correlation or van der Waals bonding; compare with DFT+U, HSE, or experiment.
