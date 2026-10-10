# Three Candidate Hypotheses on Solution Space Rigidity

## [DIM: Structure (Approximation Theory)]
**Prediction**: N_c(w) = constant (rigid)

**Mathematical grounding**: Barron's theorem (1993). If target f belongs to Barron class with bounded Fourier norm B = ∫|ω|·|f̂(ω)| dω < ∞, then any f ∈ B can be approximated to error ε by a two-layer network with width N = O(B²/ε²), independent of sample count w.

**Pro**: Concrete functional-analytic class with constructive approximation guarantees; width bound depends only on intrinsic spectral complexity of f, not on w.
**Con**: Barron condition is strong—if target has heavy-tailed Fourier spectrum, bound may be vacuous and N_c may actually grow with w.

## [DIM: Geometry (Parameter Space Manifold)]
**Prediction**: N_c(w) ∝ w (linear growth, floppy)

**Mathematical grounding**: Each constraint f(x_i)=y_i defines a codimension-1 hypersurface in parameter space ℝ^{P(N)} where P(N)∼2N(d+1) is total parameters. Solution manifold is intersection of w hypersurfaces. For non-degenerate intersection with neighborhood of solutions: dim(solution) ≥ 1 ⇒ P(N) - w ≥ 1 ⇒ N_c(w) ≈ (w+1)/(2(d+1)).

**Pro**: Follows from transversality and codimension-counting in finite-dimensional manifolds; each independent constraint removes one degree of freedom.
**Con**: Assumes constraints in general position; adversarially chosen constraints could intersect degenerately, causing N_c to be smaller than linear.

## [DIM: Measure (Statistical Learning / VC-Dimension)]
**Prediction**: N_c(w) ∝ w^{1/2} (square-root growth, floppy)

**Mathematical grounding**: VC-dimension of two-layer ReLU network with N hidden units scales as VC ∼ O(N² d). Uniform convergence bounds require w ≳ VC · log(VC/ε) to achieve error ε on out-of-sample points. Inverting: N_c(w) ∝ √w.

**Pro**: Grounded in classical statistical learning theory; VC dimension grows quadratically in N, so N need only grow as √w to accommodate w samples.
**Con**: VC bounds are often loose in practice; actual optimization may allow smaller networks than VC bound suggests.

## SELECTED: Structure (Approximation Theory)

This is the most testable and novel candidate for proving rigidity. It makes a clear binary prediction: if N_c(w) remains constant as w increases, rigidity is confirmed; if it grows, the rigidity claim fails. The approximation-theoretic grounding is precise, making the experiment falsifiable. The other two hypotheses both predict floppy behavior (linear vs. square-root), making them less discriminative for the binary rigidity question.