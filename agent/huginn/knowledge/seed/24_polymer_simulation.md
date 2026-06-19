# Polymer Simulation

## Atomistic Models
- **All-atom (AA)**: Explicit every atom; accurate but expensive for large chains.
- **United-atom (UA)**: Groups CH₃/CH₂/CH into single beads; speeds up alkane polymers.
- **Coarse-grained (CG)**: Maps monomers or groups of atoms to beads; e.g., MARTINI force field.

## Force Fields
- **OPLS-AA / TraPPE**: Common for organic/polymer systems.
- **GAFF / AMBER**: Biopolymers and organic small molecules.
- **MARTINI**: Popular coarse-grained model for soft matter.
- **ReaxFF**: Reactive simulations for polymer decomposition or cross-linking.

## Key Properties
- **Glass transition temperature (Tg)**: From temperature dependence of density or volume in MD.
- **Radius of gyration (Rg)** and **end-to-end distance**: Chain conformation metrics.
- **Mechanical properties**: Stress–strain from uniaxial deformation; modulus from small-strain response.
- **Free volume**: Fractional accessible volume estimated by probe insertion.

## Simulation Setup Tips
- Build equilibrated melts using self-avoiding walks + MD relaxation.
- Use long enough runs to sample chain reptation; CG models help reach longer timescales.
- Validate against experimental Tg, density, and Rg when possible.

## Tools
- `LAMMPS`, `GROMACS`, `NAMD`
- `packmol` for initial configurations
- `MDAnalysis` / `freud` for structural analysis
