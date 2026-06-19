# LAMMPS Quick Reference

Quick guide for molecular dynamics and atomistic simulations with LAMMPS.

## Common unit styles

| Style | Length | Energy | Time | Force | Pressure |
|-------|--------|--------|------|-------|----------|
| metal | Å | eV | ps | eV/Å | bar |
| real | Å | kcal/mol | fs | kcal/mol/Å | atm |
| si | m | J | s | N | Pa |
| lj | reduced | reduced | reduced | reduced | reduced |

Most materials simulations use `units metal`.

## Minimal input skeleton

```lammps
units metal
atom_style atomic
boundary p p p
read_data structure.data
pair_style eam/alloy
pair_coeff * * Cu_u3.eam Cu
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

thermo 100
thermo_style custom step temp pe ke etotal press vol

fix 1 all nvt temp 300.0 300.0 0.1
timestep 0.001
run 10000
```

## Common fixes

- `fix nve`: microcanonical.
- `fix nvt temp Tstart Tstop Tdamp`: Nose-Hoover NVT.
- `fix npt temp Tstart Tstop Tdamp iso Pstart Pstop Pdamp`: NPT.
- `fix langevin`: stochastic thermostat.
- `fix spring/self`: harmonic restraint.

## Typical timestep and damping

- `timestep`: 1 fs (metals), 0.5 fs (systems with H/reactions).
- `Tdamp`: 100× timestep ≈ 0.1 ps in metal units.
- `Pdamp`: 1000× timestep ≈ 1.0 ps in metal units.

## Potentials

- `pair_style lj/cut`: Lennard-Jones.
- `pair_style eam/alloy`: embedded-atom method for metals.
- `pair_style reax/c`: reactive force field.
- `pair_style deepmd`: machine-learning potentials via DeePMD-kit.
- `pair_style meam/c`: modified EAM.

## Output

- `dump`: per-atom trajectories.
- `thermo`: global thermodynamic quantities.
- `restart`: save binary restart files.

## Troubleshooting

- `Lost atoms`: atoms moved too far per step or box too small.
- `ERROR on proc 0: Bond atoms missing`: bond topology issue or bad geometry.
- Energy drift: reduce timestep or improve potential.
