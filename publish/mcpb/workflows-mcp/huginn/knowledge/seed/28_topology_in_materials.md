# Topology in Materials Science

## Topological Band Theory
- Electronic bands can possess global topological invariants that are robust against smooth deformations and disorder.
- Key invariants: Chern number (2D, broken time-reversal), Z₂ invariant (2D/3D with time-reversal symmetry).
- Materials with non-trivial invariants host protected surface/edge states.

## Topological Insulators
- Bulk is insulating; surfaces conduct via spin-momentum-locked Dirac cones.
- Examples: Bi₂Se₃, Bi₂Te₃, Sb₂Te₃.
- Z₂ calculation from parity eigenvalues (Fu-Kane formula) or from Wannier charge centers.

## Weyl and Dirac Semimetals
- **Weyl semimetals**: 3D gapless phases with non-degenerate band touching points (Weyl nodes) acting as monopoles of Berry curvature.
- **Dirac semimetals**: Four-fold degenerate Dirac points, often protected by crystal symmetry (e.g., Na₃Bi, Cd₃As₂).
- Signature: Fermi-arc surface states connecting projected Weyl nodes.

## Berry Phase and Berry Curvature
- Berry phase `γ = ∮ A(k)·dk` around a closed loop in k-space.
- Berry curvature `Ω(k) = ∇ₖ × A(k)`; integral over a closed surface gives Chern number.
- Useful for anomalous Hall conductivity and orbital magnetization.

## Topological Data Analysis (TDA) for Materials
- **Persistent homology**: Tracks the birth and death of topological features (connected components, loops, voids) across length scales.
- Applications: characterize porous structures, atomic configurations, and local environments.
- Tools: `Ripser`, `Persim`, `Dionysus`, `GUDHI`.

## Practical Computation
- Use `Wannier90` + `Z2Pack` for topological invariants.
- Compare surface band structure with bulk band projection.
- Check symmetries carefully: time-reversal, inversion, and crystal symmetries determine available invariants.

## Pitfalls
- Trivial surface states can mimic topological ones; verify spin texture and k-space connectivity.
- DFT band gaps may be too small; hybrid functionals or GW improve invariant reliability.
