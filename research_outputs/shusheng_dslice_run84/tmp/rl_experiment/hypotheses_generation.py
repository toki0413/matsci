#!/usr/bin/env python3
import json

hypotheses = [
    {
        "dimension": "Computation (Approximation Theory)",
        "statement": "Rigid solution spaces have low Kolmogorov complexity; by universal approximation bounds, the minimum network width needed to satisfy w constraints remains bounded as w increases. Fat solution spaces have high complexity requiring width to scale linearly with constraint count.",
        "predict": "Rigid family: N_c(w) = O(1) (constant in w); Fat family: N_c(w) = O(w) (linear in w)",
        "pro": "Grounded in VC dimension and universal approximation bounds which rigorously link function complexity to network capacity. The O(1) vs O(w) prediction provides a sharp binary threshold for rigidity that aligns perfectly with the 'zero violation' stopping criterion.",
        "con": "Kolmogorov complexity is uncomputable, requiring proxy measures like effective dimension which may not strictly correlate with N_c. Universal approximation bounds are asymptotic and may not hold for small networks."
    },
    {
        "dimension": "Geometry (Manifold Intrinsic Dimension)",
        "statement": "The solution set forms a manifold of intrinsic dimension d. For rigid spaces with small d, the covering number grows logarithmically with sample count. For fat spaces with large d, the covering number grows as a power law with w.",
        "predict": "Rigid family: N_c(w) ~ d*log(w) (logarithmic in w); Fat family: N_c(w) ~ d*w^alpha (power-law in w, alpha > 0)",
        "pro": "Connects to the manifold hypothesis in deep learning where generalization depends on intrinsic rather than ambient dimension. Logarithmic vs power-law scaling provides a clear discriminant that is easier to distinguish than constant vs linear.",
        "con": "Estimating intrinsic dimension d from discrete constraint samples is numerically unstable. High ambient dimension may be confused with high intrinsic dimension, confounding the measurement."
    },
    {
        "dimension": "Optimization (Landscape Conditioning)",
        "statement": "Rigidity corresponds to a sharp, well-conditioned loss minimum where the Neural Tangent Kernel has bounded condition number. Fatness corresponds to a flat, ill-conditioned landscape where the condition number grows with w, requiring increased width to maintain convergence.",
        "predict": "Rigid family: kappa(N_c) = O(1) as w->infinity, so N_c(w) = O(sqrt(w)) (sublinear); Fat family: kappa(N_c) ~ O(w^2), so N_c(w) ~ O(w^2) (superlinear)",
        "pro": "Directly links optimization dynamics (NTK conditioning) to architectural requirements via neural tangent kernel theory. The sublinear vs superlinear scaling provides a dramatic discriminant that is easy to observe numerically.",
        "con": "NTK theory strictly applies to infinite-width regimes; extrapolating to small networks introduces approximation error. The condition number of the discrete empirical NTK may not reflect the continuous NTK condition number."
    }
]

print("=== Three Divergent Candidate Hypotheses ===\n")
for i, h in enumerate(hypotheses, 1):
    print(f"[DIM: {h['dimension']}] {h['statement']} | predict: {h['predict']} | pro: {h['pro']} | con: {h['con']}")
    print()

print("=== SELECTED ===")
print("The most testable and novel hypothesis is:")
print(f"[DIM: {hypotheses[0]['dimension']}] {hypotheses[0]['statement']} | predict: {hypotheses[0]['predict']}")
print("\nRationale: The O(1) vs O(w) prediction from Computation theory offers the cleanest numerical discriminant. A single experiment measuring N_c(w) across increasing w can definitively distinguish constant growth (rigid) from linear growth (fat) without needing to estimate intrinsic dimension (Geometry) or deal with NTK regime mismatch (Optimization). The binary threshold aligns perfectly with the 'zero violation' stopping criterion, enabling definitive classification of the solution space.")

result = {"hypotheses": hypotheses, "selected": 0}
with open("/tmp/rl_experiment/hypotheses.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print("\nHypotheses exported to /tmp/rl_experiment/hypotheses.json")