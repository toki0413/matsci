# Geometric Invariants in Materials Science

Geometric invariants quantify shapes and spatial relationships that are independent of coordinate transformations. They are essential for comparing structures, building descriptors, and enforcing physical symmetries in ML models.

## Basic Structural Invariants

- **Lattice parameters** (a, b, c, α, β, γ): define the periodic cell up to an affine transformation.
- **Volume per atom**: total cell volume divided by number of atoms; comparable across supercells.
- **Density**: mass per unit volume; depends on composition and cell volume.
- **Coordination number**: number of nearest neighbors within a cutoff; depends on bonding definition.

## Symmetry Invariants

- **Space group** and **point group**: classify crystals by their symmetry operations.
- **Wyckoff positions**: describe atomic sites modulo symmetry.
- **Irreducible representation labels**: encode how orbitals/transforms behave under symmetry operations.

## Curvature and Local Geometry

- **Voronoi cell volume and face areas**: characterize local atomic environment.
- **Voronoi index** (n₃, n₄, n₅, n₆): counts of polygon faces with 3–6 edges; common for liquids and glasses.
- **Steinhardt bond-orientational order parameters** (qₗ, wₗ): rotationally invariant measures of local ordering; q₄ and q₆ distinguish FCC, BCC, HCP, and icosahedral motifs.
- **Local atomic environment vectors** (e.g., SOAP, ACSF): ensure translational, rotational, and permutational invariance.

## Topological–Geometric Coupling

Some descriptors combine topology and geometry:

- **Persistence images** add geometric weighting to persistence diagrams.
- **Ring statistics** (for covalent networks) count n-membered rings in bonding graphs.
- **Cavity/channel descriptors** from alpha shapes give pore size distributions and accessible surface areas.

## Invariance Requirements for ML

When training models on 3D structures, descriptors should ideally be invariant to:

1. Translation and rotation of the whole system.
2. Permutation of identical atoms.
3. Cell replications (supercell invariance) for periodic crystals.

SOAP, many-body tensor representations, and graph neural networks explicitly satisfy these invariances.

## Common Pitfalls

- Using raw Cartesian coordinates as descriptors breaks rotational/translational invariance.
- Choosing a single global cutoff may miss multi-scale features; multi-scale cutoffs or radial basis expansions help.
- Ignoring periodic boundary conditions when computing neighbor lists produces incorrect coordination numbers and descriptors.
