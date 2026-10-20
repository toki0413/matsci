# Selected Hypothesis for Solution Space Rigidity Experiment

## Three Divergent Candidate Hypotheses

### [DIM: optimization] Hypothesis 1 (Optimization Artifact)

**Statement:** The solution space is intrinsically rigid, but gradient-based optimization fails to find solutions for larger w due to increasing landscape complexity with constraint count. The apparent "fatness" (unattainable N_c) in prior experiments is an optimization artifact, not a property of the solution space.

**Mathematical grounding:** The loss landscape of constrained approximation:
$$\min_\theta \frac{1}{w}\sum_{i=1}^w (N_\theta(x_i) - f^*(x_i))^2$$
is non-convex with many shallow local minima. As w increases, the number of constraints increases the conditioning difficulty, making it harder for first-order methods like Adam to find solutions that generalize to held-out points. The gap between training MSE and generalization error at suboptimal stationary points explains why wider networks sometimes perform worse.

**Predict:** With standard Adam training, N_c(w) appears unattainable or grows with w. With improved optimization (LBFGS + continuation on ε from 1e-1 to 1e-3), N_c(w) = O(1) — a small constant (e.g., 8-16) independent of w across the full range of w.

**Pro:** Directly explains why prior experiments failed to find any zero-violation solutions; addresses the computational blind spot; makes a clear prediction about the effect of optimization method.

**Con:** Requires careful experimental design to isolate optimization effects from intrinsic properties; the "improved optimization" protocol must be well-defined and reproducible.

---

### [DIM: computation] Hypothesis 2 (Approximation Scaling)

**Statement:** The solution space is inherently fat due to approximation-theoretic constraints. As the number of constraints w increases, the minimum network width needed to achieve zero violation must grow to accommodate the additional constraints.

**Mathematical grounding:** For a ReLU network of width H, the number of degrees of freedom is O(H). To satisfy w independent constraints with error ≤ ε, we need sufficient degrees of freedom. For smooth target functions (e.g., polynomials of degree d, trigonometric polynomials), approximation theory suggests the sample complexity for uniform approximation scales as H ∝ w^α for α ∈ [0.5, 1]. The VC-dimension of width-H ReLU networks is O(H), so the number of constraints that can be satisfied scales linearly with H.

**Predict:** N_c(w) grows polynomially with w, specifically N_c(w) ∝ w^α for α ∈ [0.5, 1]. On a log-log plot, N_c(w) vs w yields a straight line with slope α ≈ 0.5-1. This scaling is robust across optimization methods.

**Pro:** Grounded in approximation theory and VC-dimension arguments; makes a clear, quantitative prediction; directly testable via width scanning; consistent with the observed failure to achieve zero violation (N_c would exceed the scan range for moderate w).

**Con:** Assumes optimization succeeds in finding the approximating network; may overestimate N_c if optimization gets stuck in suboptimal regions; doesn't explain why even very wide networks failed in prior experiments.

---

### [DIM: structure] Hypothesis 3 (Algebraic Rigidity)

**Statement:** For target functions with low intrinsic algebraic complexity (e.g., low-degree polynomials, trigonometric polynomials with few frequencies), the solution space is rigid: N_c(w) remains bounded by a small constant independent of w. This arises because the piecewise linear structure of ReLU networks can efficiently represent the low-dimensional solution manifold of analytic constraints.

**Mathematical grounding:** Consider a polynomial target of degree d. The solution set S_w = {θ : max_i |N_θ(x_i) - f*(x_i)| ≤ ε} is a semi-algebraic set. For ReLU networks, the function class consists of piecewise linear functions with O(H) linear regions. When f* lies in a low-dimensional subspace (e.g., polynomials of degree d), the constraint manifold has dimension close to d+1. A ReLU network with H ≳ d linear regions can capture this manifold, and once H exceeds this threshold, additional constraints (increasing w) do not require additional width because the same network can interpolate more points on the same low-dimensional manifold. This is the essence of rigidity: the solution space dimension remains low despite increasing w.

**Predict:** N_c(w) = constant (small finite value, e.g., 8-16) for all w up to a large threshold (e.g., w ≤ 1000), yielding a flat N_c(w) curve under standard training. This contrasts with Hypothesis 2 (growing) and Hypothesis 1 (constant only with improved optimization).

**Pro:** Directly tests the core concept of rigidity vs fatness; exploits the algebraic structure of both the target function and the ReLU activation; makes a distinctive prediction that differs from approximation-theoretic expectations; the most novel of the three hypotheses.

**Con:** May only apply to highly structured functions; for generic smooth functions, the solution space may be fat; requires careful choice of target function to validate.

---

## Discriminability Analysis

A single experimental design can distinguish all three hypotheses:

| Experiment Condition | Hypothesis 1 Prediction | Hypothesis 2 Prediction | Hypothesis 3 Prediction |
|---------------------|------------------------|------------------------|------------------------|
| Standard Adam, structured function | N_c(w) unattainable/grows | N_c(w) grows polynomially | N_c(w) = constant |
| LBFGS + continuation, structured function | N_c(w) = constant | N_c(w) grows polynomially | N_c(w) = constant |

**Distinguishing criteria:**
- If N_c(w) is constant under standard training → supports Hypothesis 3 (rejects 2)
- If N_c(w) grows under standard training but becomes constant with improved optimization → supports Hypothesis 1 (rejects 3)
- If N_c(w) grows under both standard and improved optimization → supports Hypothesis 2 (rejects 1 and 3)

---

## SELECTED: Hypothesis 3 (Algebraic/Structural Dimension)

**Rationale:** Hypothesis 3 is the most testable and novel choice for several reasons:

1. **Novelty:** It makes a distinctive prediction (constant N_c(w) for structured functions) that directly challenges the conventional approximation-theoretic expectation (Hypothesis 2, growing N_c(w)). This novelty lies in connecting the algebraic structure of the target function to the rigidity of the solution space, a perspective not typically considered in neural network approximation theory.

2. **Testability:** It can be tested with a single, well-designed experiment: scan network widths for a structured target function (e.g., low-degree polynomial) using standard training and measure N_c(w). A flat N_c(w) curve would provide evidence for rigidity; a growing curve would support Hypothesis 2.

3. **Direct relevance:** It directly addresses the core research question about whether solution space rigidity can be probed via neural network generalization. If confirmed, it would demonstrate that certain structured functions have rigid solution spaces that small neural networks can capture, providing empirical evidence for the rigidity concept.

4. **Clear mathematical grounding:** The hypothesis is grounded in the algebraic structure of both the target function (polynomials/trigonometric polynomials) and the ReLU network (piecewise linear functions), providing a principled basis for the prediction.

5. **Contrast with alternatives:** Unlike Hypothesis 1 (which is a meta-commentary on optimization), Hypothesis 3 makes a substantive claim about the solution space geometry. Unlike Hypothesis 2 (which is a standard approximation-theoretic prediction), Hypothesis 3 offers a novel alternative perspective.

**Experimental recommendation:** Test Hypothesis 3 using a low-degree polynomial target (degree 2-3) with ReLU networks of width ranging from 4 to 128, w values from 10 to 500, and standard Adam training. If N_c(w) remains constant (e.g., N_c ≈ 8-16) across all w, this provides evidence for structural rigidity. If N_c(w) grows, this supports Hypothesis 2 or indicates optimization issues (Hypothesis 1). Follow-up experiments with improved optimization can further discriminate between the hypotheses.
