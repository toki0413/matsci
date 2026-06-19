# Workflow Automation Tips for Computational Materials Science

## Reproducibility
- Pin software versions (Python packages, DFT binaries, potentials) in a requirements file or container image.
- Store input files, structures, and job scripts under version control; avoid hand-editing outputs.
- Record the random seed when using stochastic methods (MD, Monte Carlo, structure search).

## Directory Layout
- Use one directory per calculation with a clear naming convention, e.g., `system/functional/kgrid/`.
- Keep a `README` or `metadata.json` describing the purpose, inputs, and runtime environment.

## Convergence Checklist
- Plane-wave cutoff / basis set
- k-point grid density
- Supercell size for defects/interfaces
- Total-energy and force convergence thresholds
- Time step and simulation length for MD

## Error Handling
- Distinguish fatal errors from transient failures (walltime, node failure, license server).
- Capture stdout/stderr and parse for known failure signatures (SCF convergence, missing pseudopotentials).
- Use dependency-aware job chaining when possible.

## Scaling and HPC
- Match node/CPU/GPU requests to the code's parallelization model (k-point, plane-wave, replica).
- Use `mpirun`/`srun` with proper binding; profile with `vtune`, `nsys`, or built-in timers.
- Stage inputs/outputs to fast scratch storage; archive completed jobs to long-term storage.

## Agent Integration
- Break complex workflows into idempotent stages so Huginn can retry or resume after failures.
- Use structured output (JSON) for key results to simplify downstream parsing and decision-making.
