# Mechanical Properties of Materials

## Elastic Behavior
- **Young’s modulus (E)**: Stiffness under uniaxial tension/compression.
- **Shear modulus (G)**: Resistance to shear deformation.
- **Bulk modulus (K)**: Resistance to hydrostatic compression.
- **Poisson’s ratio (ν)**: Transverse strain / axial strain; typically 0.25–0.35 for metals.

## Strength and Plasticity
- **Yield strength**: Stress at which permanent deformation begins.
- **Ultimate tensile strength**: Maximum engineering stress before fracture.
- **Ductility**: Strain to fracture; often reported as percent elongation.
- **Hardness**: Resistance to localized plastic deformation (Vickers, Brinell, nanoindentation).

## Computational Prediction
- **DFT**: Elastic constants from strain-energy second derivatives; stress–strain curves from ab-initio MD.
- **Classical MD**: Dislocation nucleation, fracture, nanoindentation at finite temperature.
- **Crystal plasticity / FEM**: Mesoscale and macroscale forming and failure.

## Stability Criteria
- Mechanical stability requires positive definite elastic tensor (all eigenvalues > 0).
- For cubic crystals: C₁₁ > |C₁₂|, C₁₁ + 2C₁₂ > 0, C₄₄ > 0.
