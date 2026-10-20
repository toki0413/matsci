# Hypotheses on Solution Space Rigidity Probed by Neural Network Generalization

## Problem Setup

Let f* be a target function in a class C (e.g., C[0,1]). Given w constraint samples {(x_i, y_i)}_{i=1}^w where y_i = f*(x_i), we consider the solution space:

S_w = {f ∈ C : |f(x_i) - y_i| ≤ ε for i=1,...,w}

where ε is a tolerance. The "rigidity" of S_w is its dimension in function space: low dimension = rigid, high dimension = fat.

We train a feedforward network f_θ with hidden width N on the w training samples, and measure the maximum absolute error on a held-out set of validation points. N_c(w) is the minimum width N such that the validation error ≤ 1e-3 ("zero violation").

Three hypotheses predict different scaling of N_c(w) with w.

---

## Hypothesis 1 [STRUCTURE: Algebraic Variety Dimension]

**Statement:** The constraint set defines an algebraic variety V ⊆ ℝ^d in the parameter space of the target function family. The solution space is rigid because the constraints are algebraically dependent, reducing the effective degrees of freedom to a constant d*. Once w ≥ w_min (the minimum number of points needed to determine the algebraic relations), additional constraints do not increase the solution space dimension. The minimal network width needed is determined by the algebraic degree of the variety, not by w.

**Mathematical Grounding:**
- Consider the constraint map Φ: ℝ^d → ℝ^w, θ ↦ (f_θ(x_1), ..., f_θ(x_w))
- The solution set is Φ^{-1}(y) ∩ Θ, where Θ is the parameter space
- By the algebraic implicit function theorem, if the constraints are algebraically dependent, dim(Φ^{-1}(y)) = d - rank(J) where J is the Jacobian
- For a polynomial target of degree d*, the algebraic variety has dimension d* regardless of w ≥ d*+1 (by polynomial interpolation uniqueness)
- Neural networks of width N can approximate polynomials of degree up to O(N) (by the universal approximation theorem with polynomial activations)

**Prediction:** N_c(w) = C (constant) for all w ≥ w_min, where C depends only on the algebraic degree of f*, not on w. Specifically, if f* is a polynomial of degree d*, then N_c(w) = ⌈(d*+1)/2⌉ for w ≥ d*+2.

**Pro:** This prediction is grounded in algebraic geometry, which provides rigorous bounds on solution space dimension. It directly addresses the core definition of rigidity as low-dimensional solution space. The constant prediction is testable and falsifiable.

**Con:** Requires choosing a target function with known algebraic structure. If the target is not algebraic (e.g., transcendental), the algebraic variety framework may not apply directly.

---

## Hypothesis 2 [GEOMETRY: Manifold Curvature and Metric Entropy]

**Statement:** The solution space S_w is a smooth manifold in function space with curvature κ(w) that grows with w. As more constraint points are added, the manifold becomes more tightly curved, increasing its metric entropy. The network width N needed to approximate functions on this manifold with error ≤ ε scales with the covering number of the manifold, which grows polynomially with curvature.

**Mathematical Grounding:**
- The solution manifold M_w = {f ∈ H : |f(x_i) - y_i| ≤ ε, i=1,...,w} where H is a reproducing kernel Hilbert space (RKHS)
- The curvature of M_w is governed by the Hessian of the constraint functionals: κ_ij = ∂²/∂x_i∂x_j of the constraint map
- For an RKHS with kernel K, the metric entropy (log covering number) of M_w scales as log N_ε(M_w) ∝ κ(w)^{d/2} where d is the intrinsic dimension
- By approximation theory, the width N needed to achieve error ε on a manifold of curvature κ scales as N ∝ κ^{d/2}
- As w increases, κ(w) ∝ w^{1/d} (curvature accumulates with more constraints)

**Prediction:** N_c(w) = Θ(w^α) with α ∈ (0,1), specifically α = 1/2 for moderate curvature in 1D. That is, N_c(w) ∝ √w for w ≥ w_min. This sublinear growth reflects that the solution space is "fat" (high-dimensional) but with curvature that constrains it.

**Pro:** This prediction is grounded in differential geometry and metric entropy theory, which naturally captures how solution space complexity grows with constraints. The sublinear prediction is distinct from both constant and linear growth, making it easily distinguishable. It accounts for the geometric intuition that more constraints tighten the solution space but not linearly.

**Con:** Requires careful estimation of curvature, which depends on the distribution of constraint points. The scaling exponent α may vary with the specific geometry, making precise prediction challenging without knowing the exact manifold structure.

---

## Hypothesis 3 [OPTIMIZATION: Landscape Connectivity and Basin Size]

**Statement:** The training objective L(θ) = Σ_i (f_θ(x_i) - y_i)^2 + λΩ(θ) has a loss landscape where the size of the basin of attraction for solutions shrinks exponentially with w. To find a solution that generalizes (i.e., lies in the basin that extends to validation points), the network needs sufficient width to maintain connectivity in the landscape. The minimal width required scales linearly with w to preserve basin size.

**Mathematical Grounding:**
- Consider the loss landscape near a solution θ*: the Hessian H = ∇²L(θ*) has eigenvalues that scale with w
- The basin radius r_basin around θ* scales as r_basin ∝ 1/√λ_max where λ_max is the largest eigenvalue of H
- For w independent constraints, λ_max ∝ w (each constraint adds curvature in a new direction)
- To maintain r_basin large enough to contain the validation set, we need width N that keeps λ_max bounded
- By the neural tangent kernel (NTK) theory, the condition number of the NTK matrix scales as κ(NTK) ∝ N/w for fixed N, requiring N ∝ w to maintain κ = O(1)

**Prediction:** N_c(w) = Θ(w) - linear growth in w. Specifically, N_c(w) = c·w for some constant c > 0, reflecting that each additional constraint requires proportional capacity to maintain optimization connectivity and generalization.

**Pro:** This prediction is grounded in optimization theory and NTK analysis, which directly addresses the computational challenge of finding solutions that generalize. The linear prediction is the most conservative and reflects the intuition that each new constraint adds independent difficulty. It is easily testable and falsifiable.

** Con:** May overestimate the required width if the constraints are correlated (not independent). The linear scaling assumes worst-case independent constraints, which may not hold for structured targets. Optimization artifacts (local minima, bad initialization) could confound the measurement of N_c(w).

---

## Comparison of Predictions

| Hypothesis | Dimension | Scaling of N_c(w) with w | Key Mechanism |
|------------|-----------|--------------------------|---------------|
| 1 (Algebraic) | Structure | N_c(w) = Θ(1) (constant) | Algebraic dependence reduces solution space dimension |
| 2 (Geometric) | Geometry | N_c(w) = Θ(w^{1/2}) (sublinear) | Curvature accumulates, increasing metric entropy |
| 3 (Optimization) | Optimization | N_c(w) = Θ(w) (linear) | Basin shrinks, requiring proportional capacity |

These three predictions are mutually discriminable: a single experiment measuring N_c(w) across a range of w values can distinguish constant, sublinear, and linear growth patterns.

## Experimental Design

**Target Function:** Choose f*(x) = sin(2πx) on [0,1] (smooth, non-polynomial) or f*(x) = x^2 (polynomial, algebraic).

**Constraint Sampling:** For each w ∈ {10, 20, 50, 100, 200, 500}:
- Generate w training points uniformly in [0,1]
- Generate 1000 validation points uniformly in [0,1] (disjoint from training)

**Network Architecture:** Single hidden layer ReLU network: f(x) = Σ_{j=1}^N a_j · ReLU(b_j·x + c_j) + d

**Training:** For each width N ∈ {1, 2, 4, 8, 16, 32, 64, 128}:
- Train for T epochs with Adam optimizer
- Record max absolute error on validation set

**N_c(w):** Minimum N such that max validation error ≤ 1e-3 for all random seeds (≥3 seeds per w).

**Expected Outcome:** If N_c(w) stays constant → Hypothesis 1 (rigid). If N_c(w) ∝ √w → Hypothesis 2 (moderately fat). If N_c(w) ∝ w → Hypothesis 3 (fat, optimization-limited).