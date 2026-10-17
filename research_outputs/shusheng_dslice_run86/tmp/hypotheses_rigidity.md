# Hypotheses: Solution Space Rigidity in Small Feedforward Networks

## Objective
Investigate whether small feedforward networks can probe solution space rigidity via generalization on held-out constraints. Rigidity is defined by the minimum network width N_c(w) required to achieve zero violation (error ≤ 1e-3) as the number of constraints w increases.

---

## Three Divergent Candidate Hypotheses

### [DIM: STRUCTURE] Algebraic Variety Hypothesis
**Statement:** The space of functions representable by a 2-layer ReLU network with width N forms a semi-algebraic variety of dimension O(N·d). Each constraint f(x_i) = y_i is a polynomial equation in the parameters. For w constraints in general position, the solution variety has dimension O(N·d − w). When N exceeds a threshold N₀ independent of w, the variety remains non-empty and the constraints merely select points within a fixed-dimensional manifold.

**Predict:** N_c(w) = O(1) (constant in w) or O(log w). The solution space is rigid: once width is sufficient, adding more constraints does not require wider networks.

**Pro:** Grounded in algebraic geometry (Bézout-type bounds); polynomial systems have predictable solution-count behavior; clear binary outcome (variety non-empty vs empty).

**Con:** Neural network parameter spaces are not strict algebraic varieties due to ReLU piecewise structure; constraints may not be in general position.

---

### [DIM: GEOMETRY] Manifold Embedding Hypothesis
**Statement:** View the network function class as a finite-dimensional manifold M_N in the space of functions. Each held-out constraint defines a codimension-1 submanifold M_N ∩ {f(x_i) = y_i}. By transversality, intersecting w such submanifolds reduces the effective dimension by w. To maintain a non-empty intersection (a solution), the ambient dimension must scale with w.

**Predict:** N_c(w) ∝ c·w for some constant c > 0. The solution space is floppy: each additional constraint consumes one degree of freedom, requiring proportional width increase.

**Pro:** Clear geometric intuition from transversality theory; linear scaling is easy to test numerically; directly relates width to constraint count.

**Con:** Assumes constraints are independent and in general position; ignores the piecewise linear structure of ReLU networks that may create redundant constraints.

---

### [DIM: OPTIMIZATION] Landscape Conditioning Hypothesis
**Statement:** For small networks, adding constraints creates a loss landscape with increasingly ill-conditioned Hessian. The condition number κ(H) grows with w, and small-width networks lack sufficient degrees of freedom to navigate the rugged landscape. Apparent failure to achieve zero violation may reflect optimization difficulty rather than lack of representational capacity.

**Predict:** N_c(w) grows super-linearly, e.g., N_c(w) ∝ w^α with α > 1, or exponentially N_c(w) ∝ e^{βw}. The apparent floppiness is an artifact of optimization hardness.

**Pro:** Explains why small networks may fail even when solutions theoretically exist; connects to well-studied optimization theory and Hessian conditioning; accounts for training instability noted in blind spots.

**Con:** Hard to distinguish numerically from true geometric floppiness; condition number is not directly observable; may conflate optimization difficulty with representational capacity.

---

## SELECTED: Algebraic Variety Hypothesis

**Rationale:** This hypothesis is the most testable and novel. It makes a sharp, bounded prediction (N_c remains constant or grows very slowly) that is easily distinguishable from linear or super-linear alternatives in a single experiment. The binary outcome (rigid vs. floppy) aligns perfectly with the problem's requirement for a crisp classification. Moreover, it is novel: most neural network expressivity work focuses on width scaling for interpolation of w points, but the algebraic-variety perspective on solution-space dimensionality is underexplored.

**Mathematical grounding:** For a 2-layer ReLU network with input dimension d, hidden width N, the function class is:
$$f(x) = \sum_{j=1}^{N} a_j \cdot \text{ReLU}(w_j^\top x + b_j) + c$$
Each parameter tuple (a_j, w_j, b_j, c) lives in ℝ^{d+2}. The mapping from parameters to functions is semi-algebraic (ReLU = max(0,·) is piecewise linear, hence semi-algebraic). For w constraint pairs {(x_i, y_i)}, the system:
$$f(x_i) = y_i, \quad i = 1,\dots,w$$
is a system of polynomial equations in the parameters. By semi-algebraic geometry, the solution set is a semi-algebraic variety. Its dimension is at most O(N·d − w) when w ≤ N·d. When N is sufficiently large (N ≥ N₀ where N₀ depends only on d and the constraint geometry), the variety remains non-empty for all w, and its dimension stabilizes — this is rigidity.

**Experimental design:** Use a 1D input (d=1), synthetic constraint set from a known low-degree polynomial (e.g., f*(x) = sin(2πx) sampled at w points), train 2-layer ReLU networks with varying width N ∈ [1, 50] for each w ∈ [5, 50, 100, 200, 500], measure the maximum absolute error on a held-out set of 20 constraint points, and record the minimum N achieving error ≤ 1e-3. If N_c stays bounded as w increases, rigidity is confirmed.