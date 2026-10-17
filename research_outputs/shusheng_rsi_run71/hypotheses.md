Three Divergent Candidate Hypotheses for Solution Space Rigidity in Feedforward Networks

Problem Setup:
- Network: f(x) = ∑_{i=1}^N a_i·ReLU(w_i·x + b_i) + c (one hidden layer, width N)
- w = number of constraint samples (points (x_j, y_j))
- N_c(w) = minimum width N achieving max_j |f(x_j) - y_j| ≤ 1e-3 on held-out points
- Zero violation: max absolute error ≤ 1e-3
- Rigid: N_c(w) stays bounded as w increases
- Floppy: N_c(w) grows without bound or is unattainable

---

Hypothesis 1 [DIM: geometry] — Transversality/Manifold Intersection:
For a width-N ReLU network, the representable function space F_N is a semi-algebraic set of dimension O(N). Each constraint point (x_j, y_j) defines a codimension-1 submanifold M_j = {f ∈ F_N : |f(x_j) - y_j| ≤ 1e-3}. By transversality, when N exceeds a critical threshold N₀ determined by the algebraic structure of the constraints (not their count), the intersection S_w = ⋂_{j=1}^w M_j is generically non-empty and stable under perturbation. Once the ambient dimension exceeds the codimension sum, adding more constraints does not force N to increase because the intersection remains non-empty.

| predict: **rigid family: N_c(w) = constant (independent of w) for all w in scan range** | pro: Aligns with transversality from differential geometry; once network dimension exceeds constraint codimension, additional constraints don't shrink the intersection; supported by the fact that ReLU networks can represent piecewise linear functions with fixed width that can interpolate arbitrarily many points in 1D. | con: Assumes constraints are in "general position"; pathological constraints (e.g., highly oscillatory functions) may require more capacity; the transversality argument assumes smooth manifolds but ReLU networks have piecewise-linear structure with kinks. |

---

Hypothesis 2 [DIM: optimization] — Basin Volume/Curvature:
As w increases, the optimization landscape L(θ) = max_j |f_θ(x_j) - y_j| develops more local minima, saddle points, and narrow valleys. The volume of the parameter region where L(θ) ≤ 1e-3 shrinks exponentially in w due to the curse of dimensionality—each additional constraint reduces the feasible region by a multiplicative factor. To maintain a basin of attraction large enough to contain zero-violation solutions and to provide sufficient navigational freedom through the rugged landscape, network capacity must scale with w.

| predict: **floppy family: N_c(w) grows linearly with w: N_c(w) = α·w + β for some α > 0** | pro: Intuitively matches experience with overparameterized optimization—more constraints = harder landscape = more capacity needed; supported by empirical observations that wider networks generalize better on constrained tasks; the exponential shrinkage of feasible volume suggests linear compensation. | con: May overestimate the effect—neural networks have favorable landscape properties (benign critical points, gradient flow toward global minima) that could mitigate the curse of dimensionality; linear scaling assumes each constraint independently reduces volume, but constraints may be correlated. |

---

Hypothesis 3 [DIM: computation] — Approximation/VC-Dimension:
For a width-N ReLU network in 1D, the VC dimension is O(N²) (each ReLU unit contributes a breakpoint, and N units can create O(N) breakpoints). To achieve 1e-3 error on w points, we need the VC dimension to be at least proportional to w (so the network can shatter the relevant patterns). This gives N_c(w) ~ O(√w). Alternatively, from an information-theoretic viewpoint, each constraint provides O(1) bits of information, and a width-N network can represent ~N log N distinct piecewise-linear configurations; solving N log N ~ w gives N_c(w) ~ O(w/log w), but the VC bound is tighter for smooth targets.

| predict: **intermediate family: N_c(w) grows as √w: N_c(w) = α·√w + β** | pro: Grounded in rigorous approximation theory and VC-dimension bounds; square-root scaling is a natural compromise between constant (too optimistic) and linear (too pessimistic); supported by known results that width-N ReLU networks can approximate functions with O(N²) degrees of freedom. | con: The √w scaling assumes the worst-case VC bound may be loose; actual constraints may have structure that reduces the effective information content; the approximation bound is asymptotic and may not hold in the small-w regime relevant to the experiment. |

---

SELECTED: [DIM: geometry] Transversality/Manifold Intersection Hypothesis

Rationale:
1. Novelty: Makes a strong, counterintuitive claim—N_c(w) is constant regardless of w—which would reveal a fundamental rigidity property of neural network solution spaces.
2. Testability: Binary prediction (constant vs growing) that a single experiment can falsify immediately.
3. Mathematical grounding: Uses transversality from differential geometry—a clean framework for "solution space rigidity" (low-dimensional = rigid, high-dimensional = floppy).
4. Connection to bootstrap: Suggests a natural bootstrap mechanism—start with small w, find minimal N, then use fixed-N for larger w; if intersection remains non-empty, rigidity is confirmed.
5. Falsifiability: Even if false, distinguishes from linear and √w alternatives.

Governing mathematics:
- Function space: F_N = {f(x) = ∑_{i=1}^N a_i·ReLU(w_i·x + b_i) + c}
- Constraint submanifold: M_j = {f ∈ F_N : |f(x_j) - y_j| ≤ 1e-3}
- Solution set: S_w = ⋂_{j=1}^w M_j
- Rigidity condition: ∃ N₀ such that ∀ w, S_w ≠ ∅ when N ≥ N₀ (i.e., N_c(w) ≤ N₀ for all w)
