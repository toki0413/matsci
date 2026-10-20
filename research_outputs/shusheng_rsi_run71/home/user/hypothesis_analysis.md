# Solution Space Rigidity in Small Feedforward Networks: Candidate Hypotheses

## Problem Definition

- **w**: Number of constraint samples (training points)
- **N_c(w)**: Minimum hidden width achieving max absolute error ≤ 1e-3 on out-of-sample points
- **Rigidity**: N_c(w) remains small and constant as w increases
- **Floppiness**: N_c(w) exceeds scan limits or grows unbounded with w

---

## Hypothesis 1: [COMPUTATION — Approximation Bounds]

**Statement:** ReLU networks possess a finite number of linear regions that grows polynomially with hidden width. To generalize w constraint samples with max error ≤ 1e-3, the network must have sufficient capacity to partition the input space appropriately. By VC dimension theory, the effective degrees of freedom scale linearly with width N, and to control generalization error on w points, we need N ≳ w (modulo log factors).

**Prediction:** **Floppy family:** N_c(w) grows linearly with w: N_c(w) = αw + O(1) for some constant α ≥ 1.

**Mathematical Grounding:**
- VC dimension of width-N ReLU networks: VCdim ≲ O(N log N)
- Generalization bound: w ≲ VCdim · log(1/δ) ≈ O(N log N)
- Inverting: N ≳ w / log w ≈ linear in practical range (w ∈ [10, 1000])

**Pro:** Directly grounded in approximation theory; falsifiable; clear numerical prediction.
**Con:** May overcount capacity if network exploits parameter symmetries.

---

## Hypothesis 2: [GEOMETRY — Solution Manifold Dimension]

**Statement:** The set of functions satisfying all w constraints forms a solution manifold M_w ⊂ F (function space). Adding constraints reduces the dimension of M_w but does not necessarily increase the minimal network width needed to approximate it, provided the network architecture is expressive enough to capture the geometry of M_w from the outset. Once width exceeds a threshold N_min determined by the intrinsic complexity of the constraint function, the network can traverse the shrinking manifold regardless of w.

**Prediction:** **Rigid family:** N_c(w) remains constant (small finite value) for all w in the scan range: N_c(w) = N_0.

**Mathematical Grounding:**
- Constraint function f(x) is fixed; only sampling density changes with w
- A sufficiently wide network can approximate f uniformly regardless of sample count
- Solution manifold dimension decreases with w, but representation capacity needed is unchanged

**Pro:** Aligns with definition of rigidity as low-dimensional solution set; distinguishes finding vs. representing capacity.
**Con:** Risk of false rigidity if optimization gets stuck in local minima.

---

## Hypothesis 3: [OPTIMIZATION — Landscape Conditioning & Generalization Gap]

**Statement:** As w increases, the empirical risk landscape becomes better conditioned (more constraints act as implicit regularization), reducing the generalization gap. However, the strict 1e-3 error floor creates an asymptotic barrier: narrow networks cannot achieve this precision regardless of conditioning, while slightly wider networks benefit from improved optimization dynamics and can cross the threshold. The marginal capacity needed decreases logarithmically with w.

**Prediction:** **Intermediate family:** N_c(w) grows logarithmically with w: N_c(w) = β log w + O(1).

**Mathematical Grounding:**
- More constraints → better-conditioned optimization landscape
- Logarithmic scaling emerges from diminishing returns of regularization
- Strict 1e-3 threshold creates a precision barrier that requires incremental capacity

**Pro:** Bridges rigid/floppy gap; accounts for difficulty of exact zero violation.
**Con:** Log scaling subtle to distinguish from constant in finite scans; sensitive to 1e-3 threshold choice.

---

## Selected Hypothesis

**Hypothesis 1 (COMPUTATION — Approximation Bounds)** is the most testable and novel. It offers a clear, falsifiable numerical prediction (linear growth of N_c with w) that directly contrasts with the "Rigid" null hypothesis. It addresses the blind spot regarding capacity by grounding the prediction in the explicit counting of degrees of freedom (width) versus constraints (w), avoiding reliance on geometric intuitions about manifolds that may be obscured by optimization dynamics. The linear scaling is robust and easily distinguishable from constant or logarithmic trends in experimental data.

---

## Numerical Experiment Design

**Constraint function:** f(x) = sin(2πx) + 0.5 sin(6πx), x ∈ [0, 1]

**Network:** Single-hidden-layer ReLU, width N ∈ {4, 8, 16, 32, 64, 128, 256}

**Training:** MAE loss, Adam (lr=1e-3), 10,000 epochs, 5 random initializations per (w, N)

**Scan:** w ∈ {10, 20, 50, 100, 200, 500, 1000}, w_test = 100 held-out points

**Zero violation:** min test MAE ≤ 1e-3 across initializations

**N_c(w):** Minimum N achieving zero violation for each w

**Discrimination:**
- Constant N_c → Hypothesis 2 (Rigid)
- Linear N_c ∝ w → Hypothesis 1 (Floppy)
- Logarithmic N_c ∝ log w → Hypothesis 3 (Intermediate)