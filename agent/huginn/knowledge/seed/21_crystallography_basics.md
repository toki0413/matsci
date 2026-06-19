# Crystallography Basics

## Lattices and Unit Cells
- A crystal is built by repeating a unit cell in three dimensions.
- 14 Bravais lattices grouped into 7 crystal systems.
- Lattice vectors `a, b, c` and angles `α, β, γ` define the unit-cell metric.

## Miller Indices
- Planes: `(h k l)` represent reciprocal lattice intercepts.
- Directions: `[u v w]` represent direct-space vectors.
- In cubic systems, plane normals and directions with the same indices are parallel.

## Symmetry and Space Groups
- Point symmetry + translational symmetry (glide planes, screw axes) gives 230 space groups.
- Space group determines systematic absences in diffraction and symmetry-equivalent positions.
- Tools: `spglib`, `Bilbao Crystallographic Server`, `pymatgen.symmetry`.

## Reciprocal Lattice
- Defined by vectors `a* = 2π (b × c) / V`, etc.
- Diffraction peaks occur when the scattering vector `Q = h a* + k b* + l c*`.

## Practical Notes
- Always report conventional vs. primitive cells when comparing structures.
- Check for fractional-coordinate wrapping and image conventions (pbc).
- Use Wyckoff positions to reduce the number of independent structural parameters.
