# Three Divergent Candidate Hypotheses on Solution Space Rigidity

## Mathematical Framework

Let {f_i(x) = y_i}_{i=1}^w be w analytic constraint samples on a compact domain X ⊂ ℝ^d.
The solution space is S_w = {f ∈ F : |f(x_i) - y_i| ≤ 1e-3 ∀ i=1..w}.
Rigidity = low dimension of S_w (ideally 0, unique solution); Floppy = high dimension.

N_c(w) = minimum hidden layer width N of a 1-hidden-layer ReLU network such that
         max_{j ∈ held-out} |f_N(x_j) - y_j| ≤ 1e-3.

---

## Hypothesis 1 [COMPUTATION] — Approximation-theoretic redundancy

**Statement:** For rigid solution spaces arising from algebraically dependent constraints, N_c(w) remains bounded and independent of w. For floppy spaces from independent constraints, N_c(w) grows linearly with w.

**Mathematical grounding:** VC dimension of a width-N, 1-hidden-layer ReLU network: d_Vc(N) = Θ(N·dim(X)) = Θ(N). To fit w constraints with zero violation, we need d_Vc(N) ≳ w. If constraints are algebraically dependent (rigid), the effective constraint rank r < w saturates; only O(r) parameters needed → N_c = O(1). If constraints are independent (floppy), each adds new information → N_c(w) ∝ w.

**Prediction:** Rigid: N_c(w) = O(1) (constant); Floppy: N_c(w) ∝ w (linear).

**Pro:** Directly grounded in approximation theory and VC dimension; clear numerical prediction; testable with synthetic constraints of known algebraic structure.
**Con:** VC bounds are often loose; constant factors may obscure asymptotic behavior for small w.

---

## Hypothesis 2 [GEOMETRY] — Polylogarithmic manifold covering

**Statement:** For rigid solution spaces lying on a fixed d-dimensional manifold, N_c(w) = O(1). For floppy spaces where the intrinsic dimension grows logarithmically with w (d ∼ c·log w), N_c(w) grows polylogarithmically: N_c(w) ∼ (log w)^{1/c}.

**Mathematical grounding:** Let the true solution lie on a d-dimensional manifold M ⊂ L∞(X). The covering number of M at precision ε = 1e-3 scales as N_cover(ε, d) ∼ (C/ε)^d. A network of width N can approximate functions on M with error ε if N ∼ d·log(1/ε). For rigid spaces (d fixed), N_c ∼ constant. For floppy spaces where d ∼ c·log w (constraints explore increasingly high-dimensional regions), N_c ∼ (log w)^{1/c}.

**Prediction:** Rigid: N_c(w) = O(1) (constant); Floppy: N_c(w) ∼ (log w)^{1/c} (polylogarithmic, extremely slow growth).

**Pro:** Connects rigidity to intrinsic manifold dimension; distinguishes truly rigid (d=0) from merely sparse (small d) cases; polylogarithmic growth is distinct from both constant and power-law scaling.
**Con:** Estimating intrinsic dimension d from w samples requires additional analysis; the scaling may be hard to distinguish from constant for moderate w.

---

## Hypothesis 3 [OPTIMIZATION] — Square-root effective degrees of freedom

**Statement:** N_c(w) remains constant for rigid solutions (sharp, isolated minimum with few near-zero Hessian eigenvalues) and grows as √w for floppy solutions (degenerate landscape with many near-zero eigenvalues proportional to w).

**Mathematical grounding:** Consider the loss L(θ) = Σ_i (f_θ(x_i) - y_i)^2 at the minimum θ*. The Hessian H = ∂²L/∂θ∂θ* has eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_p. For a rigid solution, only O(1) eigenvalues are near-zero (few flat directions); the rest are O(1). For a floppy solution, the number of near-zero eigenvalues scales with w (each constraint adds a degree of freedom). To resolve the landscape and avoid local minima, network width must satisfy N ≳ √(number of near-zero eigenvalues). Thus N_c(w) ∼ √(number of near-zero eigenvalues) ∼ √w for floppy, and O(1) for rigid.

**Prediction:** Rigid: N_c(w) = O(1) (constant); Floppy: N_c(w) ∼ √w (square-root power law).

**Pro:** √w scaling is distinctly different from both linear (Hypothesis 1) and polylogarithmic (Hypothesis 2) predictions; testable purely by measuring N_c(w) without computing Hessians explicitly; grounded in the geometry of the loss landscape, a less commonly discussed perspective.
**Con:** The threshold for "near-zero" eigenvalues is arbitrary; the √w scaling assumes a specific relationship between constraint count and degeneracy.

---

## Discriminability Check

All three hypotheses agree on rigid families (N_c = O(1)), but predict different scalings for floppy families at w = [10, 100, 1000]:

| w | Computation (linear) | Optimization (√w) | Geometry (polylog, c=1) |
|---|---------------------|-------------------|------------------------|
| 10 | 10 | 3.16 | 2.30 |
| 100 | 100 | 10.0 | 4.61 |
| 1000 | 1000 | 31.6 | 6.91 |

These are numerically distinguishable: linear >> square-root >> polylogarithmic. A single experiment measuring N_c(w) across w ∈ [10, 1000] can distinguish all three.
