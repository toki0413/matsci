# Abaqus Quick Reference

Common elements of Abaqus/Standard and Abaqus/Explicit workflows.

## Input file (.inp) structure

```abaqus
*Heading
Title of the simulation
*Part, name=Part-1
*Node
1, 0.0, 0.0, 0.0
...
*Element, type=C3D8R
1, 1, 2, 3, ...
*End Part
*Assembly, name=Assembly-1
*Instance, name=Part-1-1, part=Part-1
*End Instance
*End Assembly
*Material, name=Steel
*Elastic
210000.0, 0.3
*Density
7.8e-9
*Solid Section, elset=All, material=Steel
*Step, name=Load, nlgeom=NO
*Static
1.0, 1.0
*Boundary
Set-Fixed, ENCASTRE
*Cload
Set-Load, 2, -1000.0
*Output, field, variable=PRESELECT
*End Step
```

## Common element types

- `C3D8R`: 8-node brick, reduced integration (general solids).
- `C3D10`: 10-node tetrahedron (better for complex geometry).
- `S4R`: 4-node shell, reduced integration.
- `B31`: 2-node beam.
- `CPE4R`: plane-strain quad.
- `CPS4R`: plane-stress quad.

## Analysis procedures

- `*Static`: quasi-static implicit.
- `*Dynamic, explicit`: explicit dynamics/impact.
- `*Heat Transfer`: thermal analysis.
- `*Coupled temperature-displacement`: thermomechanical.
- `*Frequency`: natural frequency extraction.

## Material models

- `*Elastic`: linear elastic (Young's modulus, Poisson's ratio).
- `*Plastic`: isotropic hardening with stress-strain pairs.
- `*Hyperelastic`: rubber-like materials.
- `*Damage`: cohesive/fracture damage.

## Units

Abaqus has no built-in units. Use consistent systems:

| System | Length | Force | Mass | Time | Stress | Energy |
|--------|--------|-------|------|------|--------|--------|
| SI | m | N | kg | s | Pa | J |
| mm-N-s | mm | N | tonne (Mg) | s | MPa | mJ |

Choose one system and stay consistent for all inputs.

## Troubleshooting

- `Too many attempts made for this increment`: convergence issue; check boundary conditions, material, or mesh quality.
- `Hourglassing`: use enhanced hourglass control or full-integration elements.
- `Negative eigenvalues`: buckling or snap-through; consider nonlinear stabilization.
