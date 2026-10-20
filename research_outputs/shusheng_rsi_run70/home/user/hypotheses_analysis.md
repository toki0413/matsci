# Three Candidate Hypotheses on Solution Space Rigidity in Small Feedforward Networks

## Problem Setup
- Target: $f^*(x)$ on domain $\mathcal{X}$
- Constraint points: $w$ samples $\{(x_i, y_i)\}_{i=1}^w$, split into training set $\mathcal{D}_{\text{train}}$ and held-out set $\mathcal{D}_{\text{hold}}$
- Zero violation: $\max_{x \in \mathcal{D}_{\text{hold}}} |f_N(x) - f^*(x)| \le 10^{-3}$
- $N_c(w)$: minimum hidden layer width achieving zero violation

---

## Hypothesis 1: [DIM: structure] (Algebraic Invariants)

**Prediction:** $N_c(w) = O(1)$ — constant in $w$

**Mathematical grounding:** If $f^* \in \mathcal{F}_d = \text{span}\{\phi_1, \ldots, \phi_d\}$ is a finite-dimensional algebraic family, a network of fixed width $N_0(d, \varepsilon)$ can approximate any $f \in \mathcal{F}_d$ to tolerance $\varepsilon$. Additional constraint points are automatically satisfied by the exact representation.

**Pros:** Grounded in universal approximation theory; simple to test.

**Con:** Risk of false rigidity if $f^*$ is only approximately algebraic; edge cases at small $w$.

---

## Hypothesis 2: [DIM: geometry] (Manifold Dimension)

**Prediction:** $N_c(w) = O(w)$ — linear growth in $w$

**Mathematical grounding:** Each independent constraint reduces the solution manifold dimension by approximately one. To represent a $d$-dimensional manifold, width must scale as $O(d)$. Hence $N_c(w) \propto w$.

**Pros:** Aligns with manifold approximation theory; intuitive geometric interpretation.

**Con:** Assumes constraints are independent; may overestimate scaling if constraints are redundant.

---

## Hypothesis 3: [DIM: measure] (Statistical Learning / Generalization Bounds)

**Prediction:** $N_c(w) = O(\sqrt{w})$ — square-root growth in $w$

**Mathematical grounding:** As $w$ increases, the optimization landscape develops more local minima and higher curvature. The generalization error in the NTK regime scales as $O(\sqrt{N/w})$. To maintain zero violation on held-out points, width must scale as $\sqrt{w}$ to compensate for the shrinking generalization gap and increasing landscape complexity.

**Pros:** Connects rigidity to statistical learning theory; addresses optimization sensitivity; sublinear but non-constant scaling is empirically distinguishable.

**Con:** Generalization bounds may be loose; scaling depends on architecture and optimization dynamics.

---

## SELECTED: Hypothesis 3 (Square-root growth)

**Reasoning:** Most testable (log-log plot distinguishes constant/linear/sqroot), novel (links rigidity to generalization bounds), and addresses computational blind spots about optimization sensitivity.