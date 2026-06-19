# Machine-Learning Potentials for Materials

Machine-learning potentials (MLPs) interpolate the potential-energy surface using reference data from ab initio calculations.

## Common frameworks

- **NEP (NVIDIA)** — GPUMD / calorine; fast on GPUs; good for elemental systems and alloys.
- **SNAP** — LAMMPS native; built on bispectrum descriptors; robust but can be expensive.
- **ACE** — Atomic Cluster Expansion; highly expressive; many-body order systematically improvable.
- **GAP** — Gaussian Approximation Potential (QUIP); sparse GP; established for materials and molecules.
- **MACE** — Message-passing neural network potential; high accuracy, transferable.

## Typical workflow

1. Generate a diverse training set (DFT snapshots from MD, elastic deformations, surfaces, defects).
2. Choose descriptors/hyperparameters and train.
3. Validate on a held-out test set: energy/force RMSE, radial distribution functions, phonons.
4. Run large-scale MD or Monte Carlo with the MLP.
5. Periodically re-check a few configurations with the reference method.

## Training-data tips

- Include structures at the temperatures and pressures of interest.
- Cover relevant phases, defects, and surfaces.
- Balance the dataset so no single configuration dominates the loss.
- Use active learning (uncertainty sampling) to iteratively expand coverage.

## Validation checks

- Energy and force errors should be low relative to the intended observable.
- Check for unphysical forces or energies in exploratory runs.
- Compare phonon DOS or elastic constants with DFT where possible.
