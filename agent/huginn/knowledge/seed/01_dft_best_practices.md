# DFT Best Practices

A short checklist for reliable density-functional-theory calculations.

## 1. Choose the right functional

- **GGA (PBE, PBEsol)**: general geometries, metals, large systems. PBEsol often improves lattice constants for solids.
- **Meta-GGA (SCAN, r2SCAN)**: better thermochemistry and band gaps than GGA, but can be more sensitive to numerical settings.
- **Hybrid functionals (HSE06, PBE0)**: improved band gaps and defect levels. Much more expensive than GGA.
- **DFT+U (Hubbard U)**: localized d/f electrons (transition-metal oxides, rare earths). Validate U values or use linear-response U.
- **vdW corrections (DFT-D3, DFT-D4, vdW-DF, optB88-vdW)**: layered materials, adsorption, soft matter.

## 2. K-point sampling

- Metals: dense k-mesh and small broadening (e.g., 0.02–0.05 eV).
- Semiconductors/insulators: 4×4×4 or denser for bulk; Gamma-centered is usually fine.
- Molecular systems: single Gamma often enough if box is large.
- Converge total energy with respect to k-points before production runs.

## 3. Plane-wave / basis cutoff

- Use at least 1.3× the maximum recommended cutoff (e.g., ENMAX in VASP).
- Converge forces and stress, not only total energy.

## 4. Geometry optimization

- Relax all atomic positions and cell shape/volume when appropriate.
- Check residual forces (< 0.01 eV/Å typical, < 0.001 eV/Å for precise phonons).
- Confirm the structure is a true minimum (phonon frequencies ≥ 0).

## 5. Spin and magnetism

- Use spin-polarized calculations for open-shell systems.
- Initialize magnetic moments sensibly; ferromagnetic and antiferromagnetic configurations may need comparison.

## 6. Common pitfalls

- Wrong functional for the property (e.g., PBE for wide-band-gap oxides).
- Insufficient vacuum for surfaces and molecules.
- Ignoring dispersion interactions in layered systems.
- Bad k-sampling leading to spurious forces or energies.
