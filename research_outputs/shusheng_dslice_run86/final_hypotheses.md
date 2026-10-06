# Three Divergent Candidate Hypotheses on Solution Space Rigidity

---

## [DIM: STRUCTURE] Algebraic Variety Hypothesis

**Statement**: The constraint set arises from a low-dimensional algebraic variety with finite degrees of freedom; once w exceeds the number of independent parameters, the solution space becomes rigid.

**Predict**: **Rigid family**: N_c(w) = constant in w. After a small threshold w₀, N_c plateaus at a fixed value independent of w.

**Pro**: Grounded in algebraic geometry — algebraic varieties have finite dimension; if constraints sample from such a variety, the solution space is inherently low-dimensional; directly testable via N_c scaling. The prediction is binary and clear: either N_c stays bounded (rigid) or it grows (floppy).

**Con**: Requires the true underlying function to be exactly algebraic; real-world constraints may only approximately satisfy algebraic relations.

**Mathematics**: Let f(x) be an algebraic function satisfying P(x, f(x)) = 0 for some polynomial P. The variety V = {(x,y) | P(x,y)=0} has dimension d. By Bézout's theorem, w generic points determine a unique interpolating polynomial of degree at most d−1, so N_c ~ O(1) for w > d. The solution space is rigid because the algebraic constraint fixes the function up to finitely many parameters.

---

## [DIM: GEOMETRY] Manifold Hypothesis

**Statement**: The constraint points lie on a high-dimensional manifold in function space; as w increases, the manifold's intrinsic dimension grows, preventing finite-dimensional networks from achieving exact interpolation.

**Predict**: **Floppy family**: N_c(w) grows polynomially in w: N_c(w) ~ w^α for some α > 0. This indicates the solution space is "fat" with increasing degrees of freedom.

**Pro**: Grounded in manifold learning theory — data often lies on manifolds whose dimension scales with sample size; captures the geometric intuition that more constraints reveal more structure; polynomial scaling is common in approximation theory.

**Con**: Manifold dimension estimation is sensitive to noise and sampling; polynomial growth could be confused with logarithmic growth in finite regimes.

**Mathematics**: Let the constraint points {(x_i, y_i)} lie on a manifold M ⊂ ℝ^{d×n} of intrinsic dimension d ~ w^α. By the manifold hypothesis in approximation theory, a neural network of width N must satisfy N ≥ C·w^α to achieve ε-accuracy on the manifold, hence N_c(w) ≥ C·w^α. The solution space is fat because the manifold's dimension grows with w.

---

## [DIM: COMPUTATION] Approximation Complexity Hypothesis

**Statement**: The target function belongs to a smooth function class with known approximation rates; small networks can exploit smoothness to generalize well, requiring only logarithmic growth in width with sample size.

**Predict**: **Intermediate regime**: N_c(w) grows logarithmically in w: N_c(w) ~ log(w) or N_c(w) ~ log(w)^β. This indicates a solution space that is neither rigid nor fully floppy but has controlled complexity.

**Pro**: Grounded in approximation theory — smooth functions can be approximated efficiently by neural networks with widths growing slowly in sample size; logarithmic scaling represents a middle ground between rigidity and fatness; directly testable via convergence rates.

**Con**: Logarithmic scaling may be hard to distinguish from constant in finite w regimes; requires smoothness assumptions that may not hold for all constraint sets.

**Mathematics**: Let f be a function in a smoothness class (e.g., Sobolev space W^{k,p}). By Barron's theorem and neural network approximation bounds, the width N needed to achieve error ε on w points satisfies N ~ O(ε^{-d}·log(1/ε)·log w) for certain function classes. Thus N_c(w) ~ O(log w) for fixed ε = 1e-3. The solution space has intermediate complexity because smoothness enables efficient approximation but the degrees of freedom still grow with w.

---

## SELECTED: [DIM: STRUCTURE] Algebraic Variety Hypothesis

**Rationale for selection**: This hypothesis is the most testable and novel because:

1. **Clear binary prediction**: N_c(w) = constant vs. N_c(w) grows — a single experiment with the same measured quantity (N_c as function of w) can definitively distinguish between rigid and floppy families. The algebraic variety hypothesis predicts a plateau (constant N_c), while the manifold hypothesis predicts polynomial growth and the approximation complexity hypothesis predicts logarithmic growth — these three scaling laws (constant, polynomial, logarithmic) are mutually discriminable.

2. **Strong mathematical grounding**: Algebraic geometry provides rigorous bounds on solution space dimension. Bézout's theorem gives a concrete upper bound on the number of parameters needed to interpolate points on an algebraic variety, making the prediction N_c = O(1) mathematically precise rather than heuristic.

3. **Directly falsifiable**: If N_c grows with w, the algebraic structure assumption is violated. This is a clean test without needing to estimate exponents or fit curves.

4. **Concrete experimental design**: Generate constraint points from a known algebraic function (e.g., f(x) = 1/(1+10x²) or f(x) = sin(x)·cos(2x)), vary w from small values up to the scan limit, measure N_c(w). If N_c plateaus at small w, rigidity is confirmed.

5. **Novelty**: Connecting algebraic geometry to neural network expressivity and solution space rigidity is underexplored in the machine learning literature. Most work on neural network approximation focuses on smoothness (Barron spaces, Kolmogorov superposition) rather than algebraic structure.

**Experimental setup**: Use a synthetic algebraic function f(x) = 1/(1+ax²) with a > 0. Generate w constraint points uniformly sampled from [−1, 1]. Train small feedforward networks (1 hidden layer, ReLU activation) of varying widths N to minimize MSE on the w training points. Evaluate on a held-out test set of w_test points (not used in training). Record the minimum N such that max absolute error on test points ≤ 1e-3. This gives N_c(w). Repeat for multiple w values to trace the scaling.
