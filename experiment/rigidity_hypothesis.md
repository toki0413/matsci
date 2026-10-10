# Solution Space Rigidity Experiment

## Problem Statement
Test whether small feedforward networks exhibit rigid or fat solution spaces when fitting constraint systems.

## Definitions
- **Zero violation**: max absolute error ≤ 1e-3 on held-out constraint points
- **Rigid**: N_c(w) finite and constant in w
- **Fat**: N_c(w) unbounded (any width fails within scan range)

## Three Hypotheses

### [DIM: optimization] — Rigid (constant)
The optimization landscape has a stable basin independent of w. A fixed-width network can always reach the global minimum.
**Prediction**: N_c(w) = Θ(1), constant across all w.

### [DIM: computation] — Fat (linear growth)
Representational capacity (number of linear regions) must scale with w. A width-n ReLU network has n linear regions in 1D; to fit w points with error ≤ 1e-3, need n ∝ w.
**Prediction**: N_c(w) ∝ w, linear growth.

### [DIM: geometry] — Rigid with warm-up
Solution manifold intrinsic dimension saturates after initial feature resolution. N_c(w) increases for small w then saturates.
**Prediction**: N_c(w) increases for w < w₀ then saturates at constant value.

## Discriminability
A single experiment measuring N_c(w) for w ∈ {10, 20, 50, 100, 200, 500, 1000} distinguishes:
- Constant → Hypothesis 1
- Linear → Hypothesis 2  
- Saturating → Hypothesis 3
