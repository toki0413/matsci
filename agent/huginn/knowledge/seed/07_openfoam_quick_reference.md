# OpenFOAM Quick Reference

Essential commands, dictionaries, and practices for OpenFOAM v10/11 and OpenFOAM.com versions.

## Case directory structure

```
case/
├── 0/                 # boundary and initial fields
├── constant/
│   ├── transportProperties
│   └── turbulenceProperties
├── system/
│   ├── controlDict
│   ├── fvSchemes
│   ├── fvSolution
│   └── blockMeshDict / snappyHexMeshDict
└── run script
```

## Common solvers

- `icoFoam`: laminar incompressible transient.
- `simpleFoam`: steady incompressible turbulent.
- `pimpleFoam`: transient incompressible turbulent.
- `interFoam`: multiphase (VOF).
- `chtMultiRegionFoam`: conjugate heat transfer.
- `rhoSimpleFoam`: compressible steady-state.

## controlDict key entries

- `application`: solver name.
- `startFrom`, `startTime`: simulation start.
- `stopAt`, `endTime`: simulation end.
- `deltaT`: time step.
- `writeControl`, `writeInterval`: output frequency.
- `runTimeModifiable`: allow runtime dictionary changes.
- `adjustTimeStep`: adaptive timestep with `maxCo` (Courant number).

## Numerical settings

- `fvSchemes`: gradient, divergence, laplacian, interpolation schemes.
  - Common div scheme: `Gauss linear upwind` or `Gauss limitedLinear 1`.
  - Common grad scheme: `Gauss linear`.
- `fvSolution`: linear solver settings and relaxation factors.
  - SIMPLE/PISO/PIMPLE sub-dictionaries for pressure-velocity coupling.

## Mesh generation

- `blockMesh`: structured hexahedral mesh.
- `snappyHexMesh`: automatic unstructured mesh from STL.
- Check mesh quality with `checkMesh`.
- Important metrics: non-orthogonality < 70, skewness < 4.

## Running and monitoring

```bash
blockMesh
simpleFoam
foamLog log.simpleFoam
```

- Residuals: `foamLog` + `gnuplot`, or `pyFoamPlotRunner.py`.
- Post-processing: `paraFoam`, `postProcess -func sample`, `foamToVTK`.

## Common errors

- `AMI: Patch ... has got out of sync`: dynamic mesh/AMI tolerance issue.
- `Maximum number of iterations exceeded`: poor mesh or wrong boundary/initial conditions.
- `Courant number > 1` for explicit/transient: reduce timestep or use adjustable timestep.
