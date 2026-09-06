# VASP DFT Calculation Tool

`vasp_tool` drives Vienna Ab initio Simulation Package (VASP) calculations —
plane-wave DFT for crystals, surfaces, and molecules.

## Actions

| action | what it does | key inputs |
|---|---|---|
| `relax` | geometry optimization | `working_dir` (POSCAR/INCAR/POTCAR/KPOINTS) |
| `scf` | single-point energy / static run | `working_dir` |
| `band` | band structure along a k-path | `working_dir` |
| `dos` | density of states | `working_dir` |
| `md` | ab-initio molecular dynamics | `working_dir` |
| `phonon` | phonon spectrum (DFPT / finite displacement) | `working_dir` |
| `eos` | fit an equation of state (Birch–Murnaghan / Murnaghan / Vinet) | `working_dir`, `eos_type` |
| `submit_async` | submit a long job, return `job_id` immediately | `compute_action`, `working_dir` |
| `poll_job` | check status of an async job | `job_id` |
| `wait_job` | block until the job finishes or `timeout` | `job_id`, `timeout` |

## Typical use

- Optimize a structure (`relax`) before any property run.
- Chain `relax` -> `scf` -> `band`/`dos` for the standard workflow.
- For HPC-scale runs use `submit_async` + `poll_job`/`wait_job`; short runs
  can run inline.
- `incar_overrides` patches specific INCAR tags without rewriting the file.
- `max_auto_retries` lets a failed run self-diagnose and patch INCAR before
  retrying (0 disables self-healing).

## Notes

- Heavy cost tier; restricted to the EXECUTION phase and the `dft` constraint
  scope. When you only need a quick estimate, prefer `materials_database_tool`
  or `local_structure_db` first.
- Returns energy, convergence flag, and output file paths in `VaspToolOutput`.
- Falls back to mock mode when no real VASP executable is found.
