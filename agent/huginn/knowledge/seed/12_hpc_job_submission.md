# HPC Job Submission Best Practices

Most heavy computational materials science jobs should be submitted to a scheduler rather than run interactively.

## Scheduler quick reference

### Slurm

```bash
# Submit
sbatch job.sh

# Monitor
squeue -u $USER
scontrol show job <job_id>
sacct -j <job_id>

# Cancel
scancel <job_id>
```

### PBS

```bash
# Submit
qsub job.sh

# Monitor
qstat -u $USER
qstat -f <job_id>

# Cancel
qdel <job_id>
```

## Sample Slurm script

```bash
#!/bin/bash
#SBATCH --job-name=vasp_run
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --output=vasp-%j.out

module load vasp
mpirun vasp_std
```

## Resource selection guidelines

- **CPU DFT**: 1–2 nodes, 8–32 cores per node, enough memory for the basis set.
- **GPU DFT / ML potentials**: request the `gpu` partition and one or more GPUs.
- **Large MD**: strong scaling often plateaus at 4–8 nodes; prefer more walltime over more nodes.
- **Fat nodes**: use high-memory partitions for dense systems or large databases.

## File staging

- Keep input files in the scratch/work directory referenced by the scheduler.
- Avoid heavy I/O to home directories.
- Archive results after completion.
