# Candidate Hypotheses: Solution Space Rigidity in Small Feedforward Networks

## Mathematical Setup

Let f: ℝ^d → ℝ be an analytic constraint function. We sample w constraint points {x_i, y_i = f(x_i)}_{i=1}^w uniformly from a compact domain. A feedforward network with width h (single hidden layer, ReLU activation) is trained to minimize the empirical loss L_h = Σ_i (Network(x_i) - y_i)^2. After training, we evaluate the maximum absolute error on a held-out set of w_hold points: ε_max(h) = max_{j=1}^{w_hold} |Network(x'_j) - f(x'_j)|. The zero-violation condition is ε_max(h) ≤ 1e-3. N_c(w) = min{h : ε_max(h) ≤ 1e-3} over the scan range h ∈ [h_min, h_max].

---

## Hypothesis 1 (Computation / Approximation Theory)

**Statement**: Neural networks have O(d) degrees of freedom for width d (with fixed depth). To satisfy w independent analytic constraints with zero violation, the network must have at least as many effective degrees of freedom as constraints. By the universal approximation theorem and VC-dimension bounds, the minimum width scales linearly with the number of constraints.

**Prediction**: N_c(w) = a·w + b (linear growth in w), indicating a **fat** solution space (the dimension of the solution set grows with w).

**Pro**: 
- Grounded in classical approximation theory: a width-h network with ReLU activation has approximately 2h·d + h + 1 trainable parameters (input-to-hidden: h·d, hidden-to-output: h, biases: h+1). For w constraints, we need sufficient degrees of freedom.
- Consistent with the fact that interpolation of w arbitrary points in ℝ^d typically requires Ω(w) parameters in the worst case.
- The VC dimension of width-h shallow ReLU networks is Θ(h·d), so to shatter w points with zero violation, we need h·d ≳ w, implying h ∝ w.

**Con**:
- Constraints may be correlated (not independent), reducing the effective number of degrees of freedom needed.
- Over-parameterization in neural networks often allows fitting more data than the parameter count suggests, potentially making the scaling sub-linear.

---

## Hypothesis 2 (Geometry / Manifold)

**Statement**: The constraint function f belongs to a low-dimensional function manifold with intrinsic dimension d_intrinsic (e.g., f is a smooth function with bounded Sobolev norm). Once the network width exceeds a threshold proportional to d_intrinsic, the network can uniformly approximate f on the domain regardless of the number of sampled constraints (as long as samples are representative). This reflects the rigidity of the solution space: the set of functions satisfying all w constraints is a low-dimensional manifold independent of w.

**Prediction**: N_c(w) = C (constant for w above a small threshold), indicating a **rigid** solution space (the dimension of the solution set is small and independent of w).

**Pro**:
- Grounded in manifold learning theory: if f lies on a low-dimensional manifold, a sufficiently wide network can capture the intrinsic geometry without needing more capacity as sampling density increases.
- Consistent with the fact that analytic functions are determined by local behavior; once the network can capture the local structure globally, additional samples don't increase the required capacity.
- The Kolmogorov-Tikhomirov ε-entropy of smooth function classes grows slowly with dimension, not with sample count.

**Con**:
- If the constraint function has high-frequency components or singularities, the intrinsic dimension may be large, requiring wide networks.
- The sampling must be sufficiently dense and representative; sparse or biased sampling may require larger networks to generalize.

---

## Hypothesis 3 (Optimization / Landscape)

**Statement**: As w increases, the loss landscape of the neural network becomes increasingly non-convex with more local minima and saddle points. The minimum width needed to find a zero-violation solution grows logarithmically with w, reflecting a critical width phenomenon where each additional order of magnitude in w requires a constant increase in width to escape local minima and navigate to the global solution.

**Prediction**: N_c(w) = a·log(w) + b (logarithmic growth in w), indicating an intermediate regime between rigid and fat.

**Pro**:
- Grounded in optimization theory: wider networks have smoother loss landscapes and fewer spurious local minima (the "over-parameterization effect"). The logarithmic scaling reflects the increasing difficulty of landscape navigation.
- Consistent with mean-field theory of neural networks, where the critical width for global convergence scales logarithmically with problem size.
- The number of local minima in the loss landscape grows combinatorially with w, but width provides enough connectivity to navigate this space with only logarithmic overhead.

**Con**:
- The logarithmic scaling assumes a specific structure to the loss landscape that may not hold for all constraint functions.
- Optimization algorithms (e.g., SGD) may find solutions with smaller widths even if the landscape is complex, making the measured N_c smaller than predicted.

---

## Summary of Predictions

| Hypothesis | Dimension | Prediction for N_c(w) | Solution Space |
|------------|-----------|----------------------|----------------|
| 1 | Computation (Approximation) | N_c(w) ∝ w (linear) | Fat |
| 2 | Geometry (Manifold) | N_c(w) = constant (flat) | Rigid |
| 3 | Optimization (Landscape) | N_c(w) ∝ log(w) (logarithmic) | Intermediate |

These three hypotheses are mutually discriminable: a single experiment measuring N_c(w) at multiple values of w (e.g., w = 10, 50, 100, 500, 1000) can distinguish between linear, constant, and logarithmic scaling through curve fitting.