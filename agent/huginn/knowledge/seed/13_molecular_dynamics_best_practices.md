# Molecular Dynamics Best Practices

MD simulations sample phase space by numerically integrating Newton's equations.

## Preparation

1. Build or import a sensible initial structure.
2. Choose an appropriate force field or potential.
3. Minimize / relax the structure to remove bad contacts.
4. Equilibrate temperature and pressure before production.

## Key choices

- **Ensemble**: NVT for fixed volume/temperature; NPT for fixed pressure/temperature.
- **Thermostat**: Nose-Hoover for equilibrium properties; Langevin or Berendsen for quick thermalization.
- **Barostat**: Parrinello-Rahman or Nose-Hoover for NPT.
- **Timestep**: ~1 fs for all-atom; up to 2–4 fs for constrained bonds; 0.5 fs or less for very stiff potentials.

## Equilibration checklist

- Temperature and pressure have stabilized.
- Energy drift is small and systematic.
- Density / volume has converged (NPT).
- No atoms have moved into unphysical positions.

## Production analysis

- Compute observables only after equilibration.
- Estimate statistical errors with block averaging.
- Check conservation of total energy in NVE microcanonical runs.
- Save trajectories at a reasonable frequency (not every step).

## Common pitfalls

- Starting production too early.
- Using an under-equilibrated barostat.
- Ignoring finite-size effects.
- Force field not validated for the chemistry or phase of interest.
