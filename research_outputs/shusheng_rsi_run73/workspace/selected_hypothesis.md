## SELECTED: Hypothesis 2 (Geometry / Manifold) - Rigid Solution Space

### Mathematical Grounding

**Governing Framework**: Let f ∈ C^k(Ω) be an analytic constraint function on a compact domain Ω ⊂ ℝ^d. The solution space S_w = {g ∈ C^k(Ω) : g(x_i) = f(x_i) for i = 1, ..., w} is the set of functions satisfying all w constraint samples. The rigidity of S_w is measured by its dimension as a function space.

**Key Invariant**: The intrinsic dimension of the function manifold. For analytic functions, the Taylor series at any point determines the function globally. If f is analytic with bounded complexity (e.g., f belongs to a Sobolev space W^{s,p} with finite norm), then the ε-entropy of the class {g : ‖g‖_{W^{s,p}} ≤ M} scales as ε^{-d/s}, independent of w. This implies that once a network can approximate f within ε = 1e-3 uniformly, additional constraint samples do not increase the required capacity.

**Approximation Bound**: For a width-h ReLU network N_h with one hidden layer and activation σ(z) = max(0, z), the universal approximation theorem states that for any f ∈ C(Ω) and ε > 0, there exists h_0 such that for all h ≥ h_0, ‖N_h - f‖_∞ ≤ ε. The critical width h_0 depends on the modulus of continuity of f and the dimension of Ω, but not on the number of constraint samples w.

**Variational Principle**: Consider the functional J(g) = ‖g - f‖_{L^∞(Ω)} subject to the linear constraints g(x_i) = f(x_i). The minimizer of J over the affine subspace defined by the w constraints is f itself (since f satisfies all constraints). The question is whether a width-h network can reach this minimizer via optimization. If the solution space is rigid, a fixed h suffices for all w; if fat, h must grow with w.

**Kolmogorov-Tikhomirov ε-entropy**: For the class of analytic functions with bounded norm on Ω, the ε-entropy H_ε scales as O(|log ε|^{d+1}) as ε → 0, independent of w. This supports the rigidity prediction: the information content needed to represent f to precision ε does not increase with w.

### Experimental Design

**Constraint Function Choice**: Use a smooth, low-intrinsic-dimension analytic function to test rigidity:
- f(x) = sin(π·x_1) + cos(π·x_2) for x ∈ [0, 1]^2 (2D domain)
- This function is analytic, has bounded derivatives, and lies on a 2-dimensional function manifold (spanned by sin(πx_1) and cos(πx_2))

**Network Architecture**: Single hidden layer ReLU network:
- Input dimension: d = 2
- Hidden layer: h neurons with ReLU activation
- Output: 1 neuron (linear activation)
- Total parameters: 2h + h + 1 = 3h + 1

**Sampling Procedure**:
1. Generate w constraint points {x_i} uniformly from [0, 1]^2
2. Compute y_i = f(x_i) for each x_i
3. Split into training set (80% of w) and held-out validation set (20% of w)
4. Generate a separate test set of w_hold = 1000 points uniformly from [0, 1]^2 for measuring ε_max

**Training**:
- Loss: MSE on training set: L = (1/w_train) Σ_{i∈train} (N_h(x_i) - y_i)^2
- Optimizer: Adam with learning rate 1e-3, 10000 epochs
- Early stopping if validation loss plateaus for 100 epochs

**Width Scan**:
- h_min = 10, h_max = 200
- Scan h = [10, 20, 50, 100, 150, 200] (logarithmic spacing)
- For each w ∈ [10, 50, 100, 500, 1000]
- For each (w, h) triplet, run 3 random seeds and report the minimum h achieving ε_max ≤ 1e-3 on the test set

**Measurement**:
- For each (w, h) pair, compute ε_max(h) = max_{j=1}^{1000} |N_h(x'_j) - f(x'_j)| on the test set
- N_c(w) = min{h ∈ scan range : ε_max(h) ≤ 1e-3}
- If no h in scan range achieves ε_max ≤ 1e-3, N_c(w) = h_max + 1 (indicates fat)

**Expected Outcome under Hypothesis 2**: N_c(w) ≈ constant (e.g., N_c ≈ 50) for all w ≥ 10, indicating rigid solution space. The width needed to uniformly approximate f is determined by the function's intrinsic complexity, not by the number of constraint samples.

**Alternative Outcomes**:
- If N_c(w) ∝ w (linear): supports Hypothesis 1 (Computation)
- If N_c(w) ∝ log(w): supports Hypothesis 3 (Optimization)
- If N_c(w) is undefined (no width achieves zero violation): indicates the function may have high-frequency components beyond the network's approximation capability

### Rigor and Edge Cases

1. **Scan Range Sufficiency**: The scan [10, 200] is chosen based on preliminary experiments. If N_c(w) reaches h_max for all w, the scan should be extended to verify rigidity vs. fatness.

2. **Numerical Stability**: Zero violation at ε = 1e-3 requires careful optimization. Use Adam with gradient clipping and learning rate scheduling to ensure stable convergence.

3. **Representative Sampling**: Uniform sampling from the domain ensures constraints are representative of the function's global behavior. For functions with localized features, adaptive sampling may be needed.

4. **Multiple Functions**: To ensure the result is not function-specific, repeat the experiment with several analytic functions of varying intrinsic complexity (e.g., polynomial, trigonometric, Gaussian kernel).