# Computational Fluid Dynamics Skill

## Overview

CFD for materials processing, heat transfer, and multiphase systems in materials science.

## Core Methods

### Governing Equations
- **Continuity**: ∂ρ/∂t + ∇·(ρu) = 0
- **Momentum (Navier-Stokes)**: ∂(ρu)/∂t + ∇·(ρuu) = -∇p + ∇·τ + ρg
- **Energy**: ∂(ρE)/∂t + ∇·(u(ρE + p)) = ∇·(k∇T) + τ:∇u + Q̇

### Spatial Discretization
- **FVM**: Conservation-form, dominant in engineering CFD (OpenFOAM, Fluent, Star-CCM+)
- **FEM**: Natural for multiphysics coupling (COMSOL)
- **Spectral**: High accuracy for smooth flows, DNS applications
- **Mesh quality criteria**:
  - Orthogonality > 0.5 (or < 70° deviation)
  - Aspect ratio < 100 (except near-wall anisotropic refinement)
  - Skewness < 0.85 (Fluent) or non-orthogonality < 70 (OpenFOAM)

### Turbulence Modeling
| Model | Type | Best For | Limitations |
|-------|------|----------|-------------|
| k-ε (standard) | RANS | Simple shear flows, internal flows | Poor near-wall, poor separation |
| k-ω SST | RANS | Most engineering flows, separation | Slightly more expensive than k-ε |
| Spalart-Allmaras | RANS | Aerospace, external aerodynamics | One-eqn, less general |
| Smagorinsky | LES | Simple geometries, high Re | Too dissipative, constant Cs |
| WALE | LES | Complex flows, transitional | Better near-wall behavior |
| Dynamic Smagorinsky | LES | General LES | More expensive, requires test filtering |
| DDES/IDDES | Hybrid | High-Re external flows with separation | Gray area near RANS-LES interface |

### Multiphase Methods
- **VOF**: Interface-capturing for free-surface flows. Conservative but numerically diffuse.
- **Level-set**: Smooth interface representation. Not mass-conserving without correction.
- **Euler-Euler**: Both phases as interpenetrating continua. Good for fluidized beds.
- **Euler-Lagrange (DPM)**: Fluid continuum + discrete particles. Good for dilute flows.
- **Coupling**: One-way (fluid → particles), two-way (mutual), four-way (+ particle-particle)

## Software Workflows

### OpenFOAM
```bash
# Standard workflow
1. blockMesh / snappyHexMesh → generate mesh
2. setFields → initialize fields (for multiphase)
3. simpleFoam / pimpleFoam / interFoam → solve
4. foamPostProcess / paraFoam → post-process

# Key dictionaries
- constant/transportProperties: material properties
- constant/turbulenceProperties: RAS/LES model selection
- system/fvSchemes: spatial/temporal discretization
- system/fvSolution: solver settings, under-relaxation
```

### ANSYS Fluent
```python
# Workflow
1. Read mesh → Check quality
2. Define models (viscous, multiphase, energy)
3. Material properties
4. Boundary conditions
5. Initialize → Run calculation
6. Post-process (contours, XY plots, reports)

# Convergence checks
- Residuals < 1e-3 to 1e-6 (depending on problem)
- Mass flux imbalance < 0.1%
- Monitor quantities (lift, drag, temperature) stabilized
```

## Critical Checks

- [ ] y+ values appropriate for chosen wall treatment
- [ ] CFL condition satisfied for transient simulations
- [ ] Mass/energy conservation verified
- [ ] Turbulence model appropriate for flow regime
- [ ] Boundary conditions physically consistent
- [ ] Mesh independence demonstrated

## Common Pitfalls

1. **y+ mismatch**: Using wall functions with y+ < 5 or resolved LES with y+ > 10
2. **CFL violation**: Explicit time-stepping with Co > 1 causes instability
3. **Poor mesh quality**: Skewed cells cause convergence failure and unphysical results
4. **Inconsistent BCs**: Mass flux imbalance due to incompatible inlet/outlet conditions
5. **Turbulence model misuse**: RANS for flows with large-scale unsteadiness; LES without adequate resolution
