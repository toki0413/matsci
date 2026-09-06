# 3D Structure Validation Best Practices

Guidelines for ensuring that 3D molecular and crystallographic structures are physically and geometrically sound before passing them between tools.

## 1. Enforce structural invariants

The meaning of a 3D structure is unchanged by:

- **Translation** of the whole system.
- **Rotation** of the whole system.
- **Permutation** of identical atom labels.
- Inversion or point-group symmetries, if applicable.

Therefore, never compare two structures by raw coordinate strings. Compare by invariants such as:

- Pair-distance matrices (sorted).
- Bond graphs with edge lengths.
- RMSD after optimal alignment.

## 2. Standardize before conversion

When moving structures between formats (XYZ, POSCAR, CIF, LAMMPS data, SDF):

- Convert all coordinates to the same length unit (recommended: Å).
- Center the system or align it consistently.
- Remove translation/rotation drift if comparing trajectories.
- Preserve atom-element mapping across formats.

## 3. Chemical合理性 checks

| Check | Rule of thumb |
|-------|---------------|
| Bond lengths | Within 0.7–2.5 Å for most covalent bonds; use element-specific tables for accuracy. |
| Bond angles | Reasonable valence angles for the element/hybridization. |
| Close contacts | No non-bonded distances below ~0.8 Å unless intended. |
| Lattice vectors | Positive volume, non-coplanar vectors, consistent units. |
| Periodic images | All fractional coordinates in [0, 1) or explicitly wrapped. |

## 4. Detect equivalent structures

- Use graph-isomorphism on the bonding graph to detect atom-label permutations.
- Use RMSD with Kabsch alignment to detect rotational/translational equivalence.
- Set tolerance: e.g., RMSD < 0.1 Å and matching stoichiometry → equivalent.

## 5. Tool-specific reminders

- **VASP POSCAR**: fractional coordinates must match the scaling factor and lattice vectors.
- **LAMMPS data**: box bounds and atom coordinates must use the declared `units`.
- **CIF**: symmetry operations can generate many atoms; check occupancy and disorder flags.
- **XYZ**: no cell information; add a comment line with lattice if needed.

## 6. Recommended workflow

1. Parse input structure.
2. Validate element symbols and stoichiometry.
3. Wrap periodic coordinates and check cell volume.
4. Build a bond graph using covalent radii.
5. Check for suspicious bonds or close contacts.
6. Standardize and convert to target tool format.
7. Before comparing to another structure, compute invariant descriptors.

These checks prevent "garbage-in, garbage-out" errors when agent-generated structures are passed to DFT, MD, or FEM tools.
