# Free Energy Perturbation Tool

`fep_tool` computes alchemical free energy differences along a coupling
parameter λ. Works for both drug binding (morph ligand atoms) and alloy design
(transmute dopant elements).

## Actions

| action | what it does | key inputs |
|---|---|---|
| `lambda_schedule` | generate λ windows (uniform or endpoint-dense nonlinear) | `n_lambda`, `lambda_spacing` |
| `ti` | thermodynamic integration: ΔF = ∫⟨∂U/∂λ⟩ dλ | `lambda_values`, `dU_dlambda` |
| `fep` | Zwanzig FEP: ΔF = −kT·ln⟨exp(−βΔU)⟩ | `lambda_values`, `delta_U` |
| `bar` | Bennett acceptance ratio (bidirectional, lowest variance) | `lambda_values`, `delta_U`, `delta_U_reverse` |
| `jarzynski` | non-equilibrium work: ΔF = −kT·ln⟨exp(−βW)⟩ | `work_values` |

## Typical use

- Start with `lambda_schedule` to plan λ windows, then feed per-window energy
  samples from VASP/LAMMPS into `ti` / `fep` / `bar`.
- `bar` is the recommended estimator when you can afford both forward and
  reverse samples — it has the lowest statistical variance.
- `jarzynski` is for fast-switching / steered simulations where you only have
  work values, not equilibrium samples.
- `n_bootstrap` controls error-bar resampling (0 = skip error bars).

## Notes

- Light cost tier; VALIDATION and REPORTING phases. `gp_tool` is a lighter
  surrogate for rough estimates.
- Output units follow `domain`: `materials` -> eV, `drug_design` -> kcal/mol.
- Math reference: H(λ) = (1−λ)·H_A + λ·H_B; k_B = 8.617e-5 eV/K.
