# LAMMPS Molecular Dynamics Tool

`lammps_tool` runs classical molecular dynamics via LAMMPS — minimization,
equilibration, production runs, and trajectory analysis.

## Actions

| action | what it does | key inputs |
|---|---|---|
| `run` | run a production MD simulation | `input_script`, `structure_file`, `potentials` |
| `minimize` | energy minimization | `input_script`, `structure_file` |
| `equilibrate` | equilibrate at target T/P | `input_script`, `structure_file` |
| `analyze_trajectory` | post-process a trajectory dump | `trajectory_file` |
| `equilibrium_check` | statistically check if a run reached equilibrium | `log_file_path`, `target_temp`, `target_pressure`, `window` |
| `submit_async` | submit a long job, return `job_id` immediately | `compute_action`, `input_script` |
| `poll_job` | check status of an async job | `job_id` |
| `wait_job` | block until the job finishes or `timeout` | `job_id`, `timeout` |

## Typical use

- Build a data file with `structure_tool` / `packing_tool`, then `minimize`
  before any dynamics.
- `equilibrate` -> `run` is the standard NVT/NPT -> production pattern.
- `analyze_trajectory` extracts thermo data and final energies from a dump.
- `equilibrium_check` looks at the trailing window (default 30%) of a log to
  decide if T/P plateaued — use it before trusting production data.
- `fixes` lets you inject patched settings (e.g. a smaller timestep) discovered
  by a diagnosis pass; `max_auto_retries` controls self-healing retries.

## Notes

- Heavy cost tier; EXECUTION phase, `md` constraint scope. For analytic
  estimates prefer `symbolic_math_tool` / `numerical_tool`.
- Output carries `log_path`, `trajectory_path`, `thermo_data`, `final_energy`.
- Mock mode kicks in when no `lmp` executable is available.
