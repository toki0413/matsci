# Crystal Structure Tool

`structure_tool` is a read-only tool for reading, analyzing, and converting
crystal structure files (POSCAR, CIF, XYZ, ...). Safe to auto-execute.

## Actions

| action | what it does | key inputs |
|---|---|---|
| `read` | parse a structure file, report formula / spacegroup / lattice / volume / density | `file_path` |
| `analyze` | deeper structural analysis on top of `read` | `file_path` |
| `convert` | convert between POSCAR / CIF / XYZ / JSON | `file_path`, `output_format` |
| `compare` | compare two structures (e.g. before/after relaxation) | `file_path`, `reference_path` |
| `batch_validate` | validate many structure files at once | `files` (list of paths) |

## Typical use

- Inspect a downloaded CIF before feeding it to a DFT run.
- Convert a CIF from a database into POSCAR for VASP.
- `compare` a relaxed structure against the starting structure to see what
  actually changed.
- `batch_validate` catches malformed/empty structures before a large sweep;
  each file reports its own pass/fail so partial failures stay visible.

## Notes

- Light cost tier; usable in PLANNING and EXECUTION phases.
- `file_path` also accepts an mp-id (e.g. `mp-149`) or a formula (e.g. `Si`);
  if it is in the local structure cache it resolves without a disk file.
- Always read-only — no side effects on disk except explicit `convert` output.
