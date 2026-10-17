Three Divergent Candidate Hypotheses for Solution Space Rigidity in Feedforward Networks

Problem Setup:
- Network: f(x) = ∑_{i=1}^N a_i·ReLU(w_i·x + b_i) + c (one hidden layer, width N)
- w = number of constraint samples (points (x_j, y_j))
- N_c(w) = minimum width N achieving max_j |f(x_j) - y_j| ≤ 1e-3 on held-out points
- Zero violation: max absolute error ≤ 1e-3
- Rigid: N_c(w) stays bounded as w increases
- Floppy: N_c(w) grows without bound or is unattainable

---

Hypothesis 1 [COMPUTATION — VC-Dimension Approximation]:
For a width-N ReLU network with one hidden layer in 1D, the VC dimension is Θ(N²) (each ReLU unit contributes a breakpoint, and N units can create O(N) breakpoints; the VC bound for such networks is VCdim ≤ 2N log(2eN) = Θ(N²)). To achieve max error ≤ 1e-3 on w points, the network must have effective degrees of freedom at least proportional to w (to shatter the relevant patterns). The sample complexity bound for PAC-style learning gives w ≲ VCdim · log(1/δ) ≈ N² · log N. Inverting: N_c(w) ≳ √(w / log w) ≈ √w for moderate w. The governing approximation bound: for smooth targets, uniform approximation error ε scales as O(1/N²), so achieving ε ≤ 1e-3 requires N ≳ const, but the generalization constraint on w points adds the √w dependence.

| predict: **Intermediate family: N_c(w) grows as √w, i.e., N_c(w) = α·√w + β for constants α, β > 0** | pro: Grounded in rigorous VC-dimension theory and approximation bounds; square-root scaling is a natural compromise between constant (too optimistic) and linear (too pessimistic); the prediction is quantitatively distinct from both linear and logarithmic trends and can be distinguished in a finite scan. | con: The VC bound may be loose for structured constraints; actual N_c may be smaller; the √w scaling assumes worst-case VC behavior which may not hold for smooth target functions. |

---

Hypothesis 2 [GEOMETRY — Transversality / Manifold Intersection]:
The function space F_N of width-N ReLU networks is a semi-algebraic set of dimension Θ(N). Each constraint point (x_j, y_j) defines a codimension-1 submanifold M_j = {f ∈ F_N : |f(x_j) - y_j| ≤ 1e-3}. The solution set is S_w = ⋂_{j=1}^w M_j. By the transversality theorem, when N exceeds a critical threshold N_0 determined by the intrinsic dimension d_int of the constraint manifold (not by w), the intersection S_w is generically non-empty and stable under perturbation. Once N ≥ N_0, adding more constraints reduces the dimension of S_w but does not force N to increase—the intersection remains non-empty because the ambient dimension N already exceeds the codimension sum. The governing equation: dim(S_w) = Θ(N) - Θ(w) for w ≤ N, and S_w ≠ ∅ iff N ≥ N_0 where N_0 satisfies N_0 - w_0 = d_int for some reference w_0. For smooth target functions in 1D, d_int is small (determined by the function's curvature and variation), so N_0 is small and constant.

| predict: **Rigid family: N_c(w) = N_0 (constant) for all w in scan range** | pro: Aligns with transversality from differential geometry; once network dimension exceeds constraint codimension, additional constraints don't shrink the intersection; supported by the fact that ReLU networks can represent piecewise-linear functions with fixed width that can interpolate arbitrarily many points in 1D; directly addresses the rigidity definition. | con: Assumes constraints are in "general position"; pathological constraints (e.g., highly oscillatory functions) may require more capacity; the transversality argument assumes smooth manifolds but ReLU networks have piecewise-linear structure with kinks; risk of false rigidity if optimizer gets stuck in local minima that do not correspond to the true manifold. |

---

Hypothesis 3 [MEASURE — Information-Theoretic Capacity]:
Each constraint point (x_j, y_j) with error tolerance ε = 1e-3 provides I = log₂(1/ε) ≈ 10 bits of information about the target function. A width-N ReLU network can represent approximately C·N·log N distinct piecewise-linear configurations (due to the combinatorial choice of breakpoint locations and slopes, where C is a constant depending on input domain range and activation scales). To encode w constraints requiring w·I bits, we need C·N·log N ≥ w·I. Solving asymptotically: N_c(w) ≥ (I/C)·w / log N ≈ (I/C)·w / log w for large w. This gives sublinear growth that is strictly between logarithmic and linear: w/log w grows faster than log w but slower than w. The governing equation: N_c(w) · log N_c(w) = (I/C)·w.

| predict: **Intermediate family: N_c(w) grows as w/log w, i.e., N_c(w) = α·w/log w + β for constants α, β > 0** | pro: Bridges the gap between rigid and floppy predictions; accounts for logarithmic efficiency of network representation; the w/log w scaling is distinct from both constant, √w, and linear trends and can be distinguished in a finite scan with enough data points; grounded in information theory and combinatorial capacity counting. | con: The information capacity argument is heuristic; the constant C is difficult to estimate precisely; w/log w scaling may be subtle to distinguish from √w in finite scans (both are sublinear); sensitive to the specific choice of the 1e-3 threshold which determines I. |

---

SELECTED: [GEOMETRY — Transversality / Manifold Intersection] Hypothesis

Rationale:
1. Novelty: Makes a strong, counterintuitive claim—N_c(w) is constant regardless of w—which would reveal a fundamental rigidity property of neural network solution spaces. This directly tests the core question of whether small feedforward networks can serve as probes for solution space rigidity.
2. Testability: Binary prediction (constant vs. growing) that a single experiment can falsify immediately. If N_c(w) is observed to grow (even sublinearly), the rigid hypothesis is falsified. The clear binary outcome avoids the ambiguity of fitting subtle scaling laws.
3. Mathematical grounding: Uses transversality from differential geometry—a clean framework for "solution space rigidity" (low-dimensional = rigid, high-dimensional = floppy). The governing equation S_w = ⋂ M_j with dim(S_w) = Θ(N) - Θ(w) provides a precise, verifiable mathematical statement.
4. Connection to bootstrap: Suggests a natural bootstrap mechanism—start with small w, find minimal N, then use fixed-N for larger w; if intersection remains non-empty, rigidity is confirmed. This directly addresses the "bootstrap" aspect of the research question.
5. Falsifiability: Even if false, the result is valuable—it would reveal that the solution space is not rigid, prompting investigation into why additional constraints require more capacity (e.g., due to optimization difficulties, lack of general position, or inherent complexity of the constraint manifold).

Governing mathematics:
- Function space: F_N = {f(x) = ∑_{i=1}^N a_i·ReLU(w_i·x + b_i) + c} (semi-algebraic set of dimension Θ(N))
- Constraint submanifold: M_j = {f ∈ F_N : |f(x_j) - y_j| ≤ 1e-3} (codimension-1)
- Solution set: S_w = ⋂_{j=1}^w M_j
- Rigidity condition: ∃ N_0 such that ∀ w, S_w ≠ ∅ when N ≥ N_0 (i.e., N_c(w) ≤ N_0 for all w)
- Transversality condition: dim(S_w) = N - w + O(1) for w ≤ N, with S_w ≠ ∅ iff N ≥ N_0 where N_0 is determined by the intrinsic geometry of the constraint manifold

Experimental verification plan (following the established design from run.log):
1. Constraint function: f(x) = sin(2πx) + 0.5 sin(6πx), x ∈ [0, 1] (smooth, non-piecewise-linear target)
2. Network: Single-hidden-layer ReLU, width N ∈ {4, 8, 16, 32, 64, 128, 256}
3. Training: MAE loss, Adam (lr=1e-3), 10,000 epochs, 5 random initializations per (w, N)
4. Scan: w ∈ {10, 20, 50, 100, 200, 500, 1000}, w_test = 100 held-out points
5. Zero violation: min test MAE ≤ 1e-3 across initializations
6. N_c(w): Minimum N achieving zero violation for each w
7. Discrimination:
   - Constant N_c → Hypothesis 2 (Rigid)
   - N_c ∝ √w → Hypothesis 1 (Intermediate)
   - N_c ∝ w/log w → Hypothesis 3 (Intermediate, different from √w)
   - N_c ∝ w → Floppy (not among our candidates, but would indicate extreme floppiness)