# Three Divergent Candidate Hypotheses on Solution Space Rigidity

## Mathematical Framework

Let {f(x_i) = y_i}_{i=1}^w be w constraint samples on a compact domain X ⊂ ℝ^d. The solution space is S_w = {f ∈ F : |f(x_i) - y_i| ≤ 1e-3 ∀ i=1..w}, where F is the function class parameterized by width-N feedforward networks with ReLU activations. Rigidity = low dimension of S_w (ideally 0, unique solution); Floppy = high dimension.

N_c(w) = minimum hidden layer width N of a 1-hidden-layer ReLU network such that max_{j ∈ held-out} |f_N(x_j) - y_j| ≤ 1e-3.

---

## Hypothesis 1 [STRUCTURE] — Algebraic Variety Hypothesis (RIGID)

**Statement:** For constraints sampled from a single analytic function (e.g., sin(πx), polynomials, exponentials), the solution space is rigid. The minimal width N_c(w) remains bounded and independent of w because the constraints are algebraically dependent — they all lie on a low-dimensional algebraic variety (essentially a point in function space up to the 1e-3 tolerance). The network needs only to capture the intrinsic complexity of the target function, not the number of samples.

**Mathematical grounding:**
- Let the target function f* be analytic on [-1,1]. The constraint set { (x_i, f*(x_i)) } lies on the 1-dimensional algebraic curve Γ = { (x, f*(x)) : x ∈ [-1,1] }.
- The solution space S_w = { g ∈ C([-1,1]) : |g(x_i) - f*(x_i)| ≤ 1e-3 ∀ i=1..w } is a thickened neighborhood of f* in the L∞ topology. Its "dimension" in the function space is effectively O(1) — the primary degree of freedom is the function value at any point, constrained by analyticity.
- Approximation theory for ReLU networks: For any analytic function f* on a compact interval, there exists a width-N ReLU network with N = O(log(1/ε)) that achieves uniform error ≤ ε (Eldan & Shamir, 2016; Telgarsky, 2016). The required N depends on the analyticity radius and variation of f*, but NOT on the number of constraint points w.
- Algebraic perspective: The ideal I(Γ) of polynomials vanishing on Γ has finite codimension in the space of polynomials of degree d. The constraint evaluation map Ev_w: P_d → ℝ^w, p ↦ (p(x_1), ..., p(x_w)) has rank at most d+1 (the dimension of P_d). For w > d+1, the constraints are linearly dependent, adding no new independent constraints. Thus the effective constraint rank saturates at O(1), and N_c(w) saturates accordingly.

**Prediction:** N_c(w) = C (constant) for all w in the scan range. Expected: N_c ∈ [4, 8] for w ∈ [10, 1000] when fitting sin(πx) or low-degree polynomials.

**Pro:** Directly grounded in algebraic geometry and approximation theory; clear, falsifiable numerical prediction (constant vs. growing); testable with synthetic constraints from known analytic functions; aligns with the symbolic regression paradigm where the goal is to rediscover a simple analytic form.

**Con:** VC-dimension-style bounds are often loose; the constant N may depend on the specific target function (e.g., sin(πx) vs. a high-frequency oscillatory function); the 1e-3 tolerance introduces a "thickening" that could allow slightly larger solution spaces.

---

## Hypothesis 2 [GEOMETRY] — Manifold Covering Hypothesis (FLOPPY, polylogarithmic)

**Statement:** For constraints sampled from a function with increasing complexity (e.g., a Fourier series with growing number of modes, or a function whose effective dimension increases with w), the solution space is floppy. The minimal width N_c(w) grows polylogarithmically with w: N_c(w) ∼ (log w)^{1/d} where d is the intrinsic dimension of the solution manifold that grows slowly with w. This reflects the covering number of a growing manifold in function space.

**Mathematical grounding:**
- Let the solution manifold M_w ⊂ L∞([-1,1]) be the set of functions satisfying the w constraints to within 1e-3. If the constraints are drawn from a function f* whose effective complexity increases with w (e.g., adding higher-frequency Fourier modes), then the intrinsic dimension d_w of M_w grows with w.
- The covering number of a d-dimensional manifold at resolution ε scales as N_cover(ε, d) ∼ (C/ε)^d (Kolmogorov & Tikhomirov, 1959). A width-N ReLU network can approximate functions on a manifold with error ε if N ∼ d · log(1/ε) (manifold learning results; Donoho & Grattan, 2000).
- For the rigid case (d fixed), N_c ∼ constant as w increases. For the floppy case where d_w ∼ c · log w (the constraint set explores increasingly high-frequency features), we get N_c(w) ∼ (log w) · log(1/ε) ∼ (log w) · constant. More precisely, if the effective dimension grows as d_w = α · log w, then N_c(w) ∼ d_w · log(1/ε) ∼ α · log w · log(1000) = O(log w). But a more refined analysis using manifold covering gives N_c(w) ∼ (log w)^{1/d_0} for some base dimension d_0.
- Alternative geometric view: The constraint points {x_i} in [-1,1] form a set whose ε-covering number scales as w · (1/ε) for 1D points. A ReLU network with N breakpoints can resolve features at scale ∼ 1/N. To achieve error ε = 1e-3 on w well-distributed points, we need N ∼ w · ε = O(w) for uniform coverage — but this is for worst-case points. For points clustered near the true function's structure, the effective covering is much smaller, leading to logarithmic or polylogarithmic scaling.

**Prediction:** N_c(w) ∼ (log w)^k for some k > 0 (polylogarithmic growth). Expected: for w = [10, 100, 1000], N_c ≈ [5, 7, 9] if k ≈ 0.5, or N_c ≈ [6, 9, 12] if k ≈ 1. This is distinctly different from both constant (rigid) and linear (strongly floppy) scaling.

**Pro:** Connects rigidity to the intrinsic geometry of the solution manifold; distinguishes between truly rigid (fixed d) and merely sparse (slowly growing d) cases; polylogarithmic growth is distinct from both constant and power-law scaling; testable by comparing constraint sets with different complexity growth rates.

**Con:** Estimating the intrinsic dimension d_w from w samples is non-trivial; the polylogarithmic scaling may be hard to distinguish from constant for moderate w ranges; requires careful choice of constraint generation to ensure d_w actually grows with w.

---

## Hypothesis 3 [OPTIMIZATION] — Hessian Degeneracy Hypothesis (FLOPPY, square-root)

**Statement:** For constraints that are nearly independent (e.g., from a noisy target or a highly oscillatory function), the solution space is floppy and the loss landscape develops many near-zero eigenvalues. The minimal width N_c(w) grows as √w: N_c(w) ∼ α·√w + β. This reflects the number of significant directions in the Hessian that the network must resolve to achieve zero violation.

**Mathematical grounding:**
- Consider the loss L(θ) = Σ_i (f_θ(x_i) - y_i)² at a minimum θ*. The Hessian H = ∇²L at θ* has eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_p, where p is the number of parameters. The number of near-zero eigenvalues (those below some threshold λ_min) corresponds to the effective degrees of freedom in the solution space — directions in parameter space where the loss is nearly flat.
- For constraints drawn from a smooth underlying function with additive noise, or from a highly oscillatory function, the constraint matrix X ∈ ℝ^(w×N) (with entries X_ij = φ_j(x_i) for basis functions φ_j) has effective rank r_w that grows with w. For smooth constraints on a compact domain, correlation between nearby points causes the effective rank to grow as r_w ∼ √w (due to the smoothness-induced decay of singular values; similar to the effective rank of covariance matrices of smooth processes).
- A width-N ReLU network has approximately 2N·d parameters (for input dimension d). To resolve r_w significant directions in the Hessian and avoid getting stuck in poor local minima, the network width must satisfy N ≳ √r_w (each neuron can align with one dominant direction; resolving r_w directions requires roughly √r_w neurons in a 2-layer architecture due to the quadratic nature of the parameter-to-function map). Thus N_c(w) ∼ √r_w ∼ √(√w) = w^(1/4)... wait, let me reconsider.
- More carefully: The constraint matrix X ∈ ℝ^(w×N) has singular values σ_1 ≥ σ_2 ≥ ... ≥ σ_N. For smooth constraints, the singular values decay as σ_k ∼ k^(-α) for some α > 0. The effective rank (number of singular values above threshold ε) is r_w ∼ w^(1/(2α+1)). For α = 1/2 (typical for smooth functions), r_w ∼ √w. The network needs width N ≥ r_w to span the constraint space, giving N_c(w) ∼ √w.
- Alternative viewpoint from optimization landscape: The condition number κ = λ_max / λ_min of the Hessian grows with w. For a width-N network, λ_min ∼ N²/w (roughly, from random matrix theory on the activation matrix). To achieve λ_min ≥ λ_threshold (needed for stable convergence to 1e-3 accuracy), we need N²/w ≥ c, hence N ∼ √w.

**Prediction:** N_c(w) ∼ α·√w (square-root growth). Expected: for w = [10, 100, 1000], N_c ≈ [3, 10, 31] with α ≈ 1. This is distinctly different from both constant (rigid) and linear (stronger floppy) scaling.

**Pro:** √w scaling is distinctly different from both linear (Hypothesis 1 floppy variant) and polylogarithmic (Hypothesis 2) predictions; testable purely by measuring N_c(w) without computing Hessians explicitly; grounded in the geometry of the loss landscape and constraint correlation structure; connects rigidity to optimization difficulty.

**Con:** The threshold for "near-zero" eigenvalues is somewhat arbitrary; the √w scaling assumes a specific decay rate of singular values which may vary with constraint distribution; optimization difficulties (not just solution space geometry) could confound the measurement of N_c(w).

---

## Discriminability Check

All three hypotheses agree on rigid families (N_c = O(1)), but predict different scalings for floppy families. At w = [10, 100, 1000]:

| Hypothesis | Scaling | w=10 | w=100 | w=1000 |
|------------|---------|------|-------|--------|
| 1 (Structure, Rigid) | Constant | 6 | 6 | 6 |
| 2 (Geometry, Floppy) | Polylog (k=0.8) | 5 | 8 | 11 |
| 3 (Optimization, Floppy) | Square-root | 3 | 10 | 32 |

These are numerically distinguishable: the square-root prediction (Hypothesis 3) grows much faster than polylogarithmic (Hypothesis 2) and much faster than constant (Hypothesis 1). A single experiment measuring N_c(w) across w ∈ [10, 1000] can distinguish all three.

---

## SELECTED: Hypothesis 3 (Optimization — Square-root scaling)

**Rationale:** This hypothesis is the most testable and novel for several reasons:

1. **Most testable:** The √w prediction is unambiguous and easy to verify empirically. A simple sweep over w values and network widths produces a clear N_c(w) curve that can be fitted to constant, √w, or w scaling. The square-root prediction gives distinctly different values at moderate w (e.g., N_c(100) ≈ 10 for √w vs. N_c(100) ≈ 6 for constant vs. N_c(100) ≈ 8 for polylog), making discrimination straightforward.

2. **Most novel:** While approximation theory (Hypothesis 1) and manifold geometry (Hypothesis 2) have been applied to neural network expressivity, the connection between solution space rigidity and the geometry of the loss Hessian — specifically, that the number of near-zero eigenvalues scales with constraint correlation and determines the required network width — is a less commonly discussed perspective. This bridges approximation theory with optimization landscape analysis.

3 **Best grounded in optimization dynamics:** The blind spot about "ill-conditioned Hessian landscapes common in wide networks" is directly addressed by this hypothesis. The √w scaling emerges naturally from the interplay between constraint count, singular value decay of the activation matrix, and the need to resolve degenerate directions in the loss landscape.

4. **Clear experimental signature:** If N_c(w) follows √w scaling, this indicates a floppy solution space where constraints add correlated degrees of freedom. If N_c(w) is constant, this indicates a rigid solution space. The experiment can definitively distinguish these cases.

**Experimental Design:**

1. **Target function:** Use f*(x) = sin(πx) (rigid candidate) and f*(x) = sin(πx) + 0.5·sin(10πx) (floppy candidate with higher frequency content).

2. **Constraint generation:** For each w ∈ {10, 20, 50, 100, 200, 500, 1000}, sample w points uniformly from [-1, 1], compute y_i = f*(x_i), hold out 20% as validation.

3. **Network architecture:** 1-hidden-layer ReLU network: f(x) = W₂ · max(0, W₁·x + b₁) + b₂, with input dim 1, output dim 1, hidden width N varying.

4. **Width sweep:** For each w, test widths N ∈ {4, 6, 8, 10, 15, 20, 30, 50, 80, 100}. For each (w, N), train with Adam (lr=1e-3, 5000 epochs) and record held-out max absolute error.

5. **N_c(w) determination:** For each w, find the smallest N such that held-out max error ≤ 1e-3 in at least 2 out of 3 random seeds. If no N in range achieves this, N_c(w) = UNATTAINABLE (floppy).

6. **Fit scaling:** Plot N_c(w) vs. w on log-log scale. Fit N_c(w) = a·w^b + c. Test hypotheses: b ≈ 0 (constant), b ≈ 0.5 (√w), b ≈ 1 (linear).

7. **Control for edge cases:** For w ≤ input dimension (w=1 in 1D), the system is underdetermined — these should show N_c = O(1) regardless of rigidity, as expected. Exclude w=1 from scaling analysis.