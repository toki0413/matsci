# Candidate Hypotheses for Solution Space Rigidity in Small Feedforward Networks

## Mathematical Framework

Let $f^*: [0,1] \to \mathbb{R}$ be a target analytic function. Given $w$ constraint samples $(x_i, f^*(x_i))$, we seek a neural network $N_\theta$ such that:
$$\max_{i=1,\ldots,w} |N_\theta(x_i) - f^*(x_i)| \leq \epsilon, \quad \epsilon = 10^{-3}$$

Define the solution set:
$$\mathcal{S}_w = \{ \theta \in \mathbb{R}^P : \max_{i=1,\ldots,w} |N_\theta(x_i) - f^*(x_i)| \leq \epsilon \}$$

The "dimension" of $\mathcal{S}_w$ (in an appropriate sense) measures rigidity. We probe this via:
$$N_c(w) = \min \{ H : \exists \text{ network of width } H \text{ with } \max_{x \in \mathcal{X}_{\text{held}}} |N_\theta(x) - f^*(x)| \leq \epsilon \}$$

where $\mathcal{X}_{\text{held}}$ is a set of held-out constraint points not used in training.

---

## Hypothesis 1 (Optimization Dimension)

**Statement:** The apparent "fatness" (unattainable N_c) in prior experiments is an optimization artifact, not a property of the solution space. Gradient-based training converges to local minima that satisfy the training MSE but fail to generalize to held-out points. With improved optimization (e.g., continuation on $\epsilon$, second-order methods, or careful initialization), the true $N_c(w)$ would be small and bounded independently of $w$ for structured target functions.

**Mathematical grounding:** The loss landscape of constrained approximation has many shallow local minima. The optimization problem:
$$\min_\theta \frac{1}{w}\sum_{i=1}^w (N_\theta(x_i) - f^*(x_i))^2$$
is non-convex and may have many stationary points that are not global minimizers. The gap between training error and generalization error at these suboptimal stations explains why wider networks (with more parameters) sometimes perform worse — they have larger search spaces with more traps.

**Prediction:** With standard Adam training (as in prior experiments), $N_c(w)$ appears unattainable or grows with $w$. With improved optimization (e.g., LBFGS with warm-start, continuation from larger $\epsilon$), $N_c(w)$ remains constant (small finite value) for all $w$ in the tested range.

**Pro:** Directly addresses the computational blind spot; explains why previous experiments failed to find any zero-violation solutions despite theoretical capacity. Makes a clear prediction that can be tested by varying optimization methods.

**Con:** Requires careful experimental design to distinguish optimization effects from intrinsic properties; the "improved optimization" must be well-defined and reproducible.

---

## Hypothesis 2 (Computation/Approximation Dimension)

**Statement:** The solution space is inherently "fat." For smooth target functions (e.g., polynomials, trigonometric functions), the minimum width $N_c(w)$ required to achieve zero violation grows polynomially with $w$. Specifically, $N_c(w) \sim O(w^\alpha)$ for some $\alpha \in [0.5, 1]$, reflecting the approximation-theoretic cost of satisfying more constraints.

**Mathematical grounding:** By the universal approximation theorem, a ReLU network with sufficiently many neurons can approximate any continuous function. However, the number of neurons needed to achieve accuracy $\epsilon$ on $w$ constrained points scales with $w$. For a polynomial of degree $d$, the space of interpolating polynomials has dimension $d+1$, and a ReLU network with $H$ neurons has $O(H)$ degrees of freedom. To interpolate $w$ points with error $\leq \epsilon$, we need $H \gtrsim w$ (up to logarithmic factors). More formally, the VC-dimension of width-$H$ ReLU networks is $O(H)$, so the sample complexity for uniform approximation scales with $H$.

**Prediction:** $N_c(w)$ grows linearly (or as $\sqrt{w}$) with $w$. Specifically, plotting $N_c(w)$ vs $w$ on log-log scale yields a straight line with slope $\alpha \in [0.5, 1]$.

**Pro:** Grounded in approximation theory and statistical learning theory; makes a clear, quantitative prediction about scaling; directly testable via width scanning.

**Con:** Assumes that the optimization succeeds in finding the approximating network; may overestimate required width if optimization gets stuck in suboptimal regions.

---

## Hypothesis 3 (Algebraic/Structural Dimension)

**Statement:** For target functions with low intrinsic algebraic complexity (e.g., low-degree polynomials, trigonometric polynomials with few frequencies), the solution space is rigid: $N_c(w)$ remains bounded by a small constant independent of $w$, up to a threshold where $w$ exceeds the representational capacity of the network. This rigidity arises from the piecewise linear structure of ReLU networks, which can efficiently represent low-dimensional solution manifolds of analytic constraints.

**Mathematical grounding:** Consider the constraint set for a polynomial target of degree $d$. The solution set $\mathcal{S}_w$ is a semi-algebraic set defined by polynomial inequalities. For ReLU networks, the function class is the set of piecewise linear functions with $O(H)$ linear regions. When the target function lies in a low-dimensional subspace (e.g., polynomials of degree $d$), the constraint manifold has dimension close to $d+1$. A ReLU network with $H \gtrsim d$ linear regions can capture this manifold, and once $H$ exceeds this threshold, additional constraints (increasing $w$) do not require additional width because the same network can interpolate more points on the same low-dimensional manifold.

**Prediction:** $N_c(w) = \text{constant}$ (e.g., $N_c \approx 8-16$) for all $w$ up to some large threshold (e.g., $w \leq 1000$). The plot of $N_c(w)$ vs $w$ is flat. This contrasts with Hypothesis 2 (growing) and Hypothesis 1 (constant only with improved optimization).

**Pro:** Directly tests the core concept of rigidity; exploits the algebraic structure of both the target function and the network activation; makes a distinctive prediction that differs from the other two hypotheses.

**Con:** May only apply to very structured target functions; for generic smooth functions, the solution space may indeed be fat. Requires careful choice of target function to test the hypothesis.

---

## Discriminability Summary

| Hypothesis | N_c(w) behavior with standard training | N_c(w) behavior with improved optimization |
|------------|----------------------------------------|-------------------------------------------|
| 1 (Optimization) | Appears unattainable or grows | Constant (small) |
| 2 (Approximation) | Grows polynomially with w | Grows polynomially with w |
| 3 (Structure) | Constant (flat) for structured functions | Constant (flat) for structured functions |

A single experiment measuring $N_c(w)$ under standard and improved optimization conditions, using both structured and generic target functions, can distinguish these hypotheses.
