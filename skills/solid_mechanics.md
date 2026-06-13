# Solid Mechanics & FEA Skill

## Overview

Computational solid mechanics using finite element analysis (FEA) for structural,
thermal, and multiphysics problems in materials science.

## Core Methods

### Linear Elasticity
- **Governing equations**: Navier-Cauchy equations (∇·σ + f = 0, σ = C:ε)
- **Material properties**: Young's modulus E, Poisson's ratio ν, shear modulus G = E/(2(1+ν))
- **Validation**: Check stress-strain linearity, compare with analytical solutions for simple geometries

### Elastoplasticity
- **Yield criteria**: von Mises (isotropic), Hill (anisotropic), Drucker-Prager (pressure-dependent)
- **Hardening**: Isotropic (uniform expansion), Kinematic (translation, Bauschinger effect), Combined
- **Return mapping**: Closest-point projection for associative plasticity
- **Validation**: Load-unload-reload cycles, necking prediction in tension

### Crystal Plasticity (CPFEM)
- **Slip systems**: fcc {111}<110>, bcc {110}<111>, hcp basal + prismatic + pyramidal
- **Power-law rate**: γ̇^α = γ̇₀ (|τ^α|/g^α)^(1/m) sign(τ^α)
- **Hardening**: τ̇_c = h(γ̇) Σ q^αβ |γ̇^β|
- **Frameworks**: DAMASK (spectral), ABAQUS+UMAT (FEM), VPFFT

### Fracture Mechanics
- **Stress intensity**: K_I, K_II, K_III for mode mixity
- **J-integral**: Path-independent energy release rate for nonlinear materials
- **CTOD**: Crack tip opening displacement for ductile fracture
- **Cohesive zone**: Traction-separation law for crack propagation

## Software Workflows

### ABAQUS
```python
# Standard workflow
1. Part → Property → Assembly → Step → Interaction → Load → Mesh → Job → Visualization
2. For CPFEM: Write UMAT (Fortran/C++) with constitutive integration
3. For dynamics: Use Explicit for impact, Standard for quasi-static
4. Common errors: "Too many attempts" → reduce increment; "Negative eigenvalues" → check buckling
```

### ANSYS
```python
# MAPDL workflow
1. /PREP7 → geometry, material, mesh
2. /SOLU → boundary conditions, solve
3. /POST1 → results extraction
# Workbench workflow
1. Engineering Data → Geometry → Model → Setup → Solution → Results
```

### Open-Source Alternatives
- **CalculiX**: ABAQUS-compatible open-source FEA
- **FEniCS**: Python-based FEM with automatic code generation
- **deal.II**: C++ FEM library for research
- **MOOSE**: Multiphysics Object-Oriented Simulation Environment

## Validation Checklist

- [ ] Mesh convergence study completed
- [ ] Boundary conditions statically admissible
- [ ] Material properties in consistent units
- [ ] Contact definitions verified
- [ ] Results compared with analytical or experimental data
- [ ] Stress/strain fields physically reasonable (no singularities except at crack tips)
