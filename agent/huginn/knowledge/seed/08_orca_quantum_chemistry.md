# ORCA Quantum Chemistry Quick Reference

Key settings for ORCA single-point, optimization, frequency, and property calculations.

## Minimal input

```
! B3LYP def2-SVP Opt
* xyz 0 1
C 0.0 0.0 0.0
O 0.0 0.0 1.13
*
```

## Common method/basis keywords

| Task | Suggested input line |
|------|----------------------|
| Geometry opt | `! B3LYP def2-TZVP Opt` |
| Frequency | `! B3LYP def2-TZVP Opt Freq` |
| High accuracy energy | `! DLPNO-CCSD(T) def2-TZVP def2/J TightPNO` |
| UV-Vis | `! B3LYP def2-SVP TDA` |
| NMR | `! B3LYP def2-TZVP NMR` |
| Dispersion | add `D3BJ` or `D4` to method keyword |

## Basis sets

- `def2-SVP`: quick, good for geometry pre-optimization.
- `def2-TZVP`: balanced quality for energies and properties.
- `def2-QZVP`: high accuracy, expensive.
- `cc-pVTZ`, `cc-pVQZ`: correlation-consistent bases.
- Add `C` for DKH/relativistic: `DKH-def2-TZVP`.

## Convergence and grid

- `TightSCF`, `VeryTightSCF`: stricter SCF convergence.
- `Grid5`/`Grid6`: denser DFT integration grid.
- `DefGrid2`, `DefGrid3`: defaults; use `! Grid4 NoFinalGrid` or better for delicate systems.

## Parallelization

```
%pal
  nprocs 8
end
%maxcore 2000
```

- `%maxcore`: max memory per core in MB.
- `%pal nprocs N`: number of MPI processes.

## Solvation

```
%cpcm
  epsilon 80.4
  refrac 1.33
end
```

Use `CPCM` for implicit solvation; set `epsilon` for the solvent.

## Troubleshooting

- SCF convergence problems: try `! Shift Shift`, `SlowConv`, or `UKS` with better guess.
- Negative frequencies after optimization: re-optimize with tighter convergence.
- DLPNO truncation errors: increase `TCutPNO`, `TightPNO`, or use normal CCSD(T).
