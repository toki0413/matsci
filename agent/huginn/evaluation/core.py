"""MCDA core — unified evaluation engine.

Supports: AHP, Entropy, CV, CRITIC, PCA weights;
TOPSIS, VIKOR, TODIM, PROMETHEE, RSR, Grey methods;
All combinations + sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class EvaluationResult:
    """Result of a multi-criteria evaluation."""

    method: str
    weights: dict[str, float]
    scores: dict[str, float]
    ranking: list[str]
    stability: float | None = None  # Spearman correlation for sensitivity
    sensitivity_results: list[dict] | None = None


# ── Utility: Normalization ─────────────────────────────────────


def normalize_matrix(
    X: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
) -> np.ndarray:
    """Normalize decision matrix. X shape: (n_alternatives, n_criteria).

    directions: 'max' = higher is better, 'min' = lower is better.
    """
    n, m = X.shape
    directions = directions or ["max"] * m

    Xn = np.zeros_like(X, dtype=float)
    for j in range(m):
        col = X[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax - cmin < 1e-12:
            Xn[:, j] = 1.0
            continue
        if directions[j] == "max":
            Xn[:, j] = (col - cmin) / (cmax - cmin)
        else:
            Xn[:, j] = (cmax - col) / (cmax - cmin)

    return Xn


def vector_normalize(X: np.ndarray) -> np.ndarray:
    """Vector normalization (for TOPSIS)."""
    norms = np.sqrt(np.sum(X**2, axis=0))
    norms[norms == 0] = 1.0
    return X / norms


# ── Weight Methods ─────────────────────────────────────────────


def weight_entropy(X: np.ndarray) -> np.ndarray:
    """Entropy weight method."""
    n, m = X.shape
    Xn = normalize_matrix(X)
    # Avoid log(0)
    Xn = np.clip(Xn, 1e-10, 1.0)
    p = Xn / Xn.sum(axis=0)
    E = -np.sum(p * np.log(p), axis=0) / np.log(n)
    d = 1 - E
    return d / d.sum()


def weight_cv(X: np.ndarray) -> np.ndarray:
    """Coefficient of variation (CV) weight method."""
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    cv = np.where(means != 0, stds / np.abs(means), 0)
    return cv / cv.sum()


def weight_critic(X: np.ndarray) -> np.ndarray:
    """CRITIC weight method."""
    n, m = X.shape
    Xn = normalize_matrix(X)
    stds = Xn.std(axis=0, ddof=0)

    # Correlation matrix
    corr = np.corrcoef(Xn.T)
    np.fill_diagonal(corr, 0)
    R = np.sum(1 - np.abs(corr), axis=1)

    C = stds * R
    return C / C.sum()


def weight_ahp(pairwise: np.ndarray) -> np.ndarray:
    """AHP weight from pairwise comparison matrix.

    pairwise: n×n matrix where pairwise[i,j] = importance of i vs j.
    """
    # Eigenvector method
    eigenvalues, eigenvectors = np.linalg.eig(pairwise)
    max_idx = np.argmax(eigenvalues.real)
    weights = eigenvectors[:, max_idx].real
    weights = np.abs(weights)
    return weights / weights.sum()


def weight_pca(X: np.ndarray) -> np.ndarray:
    """PCA-based weights from first principal component loadings."""
    Xn = normalize_matrix(X)
    cov = np.cov(Xn.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues.real)[::-1]
    pc1 = eigenvectors[:, idx[0]].real
    pc1 = np.abs(pc1)
    return pc1 / pc1.sum()


def compute_weights(
    X: np.ndarray,
    method: Literal["entropy", "cv", "critic", "ahp", "pca", "equal"],
    ahp_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Compute weights by method name."""
    if method == "entropy":
        return weight_entropy(X)
    elif method == "cv":
        return weight_cv(X)
    elif method == "critic":
        return weight_critic(X)
    elif method == "ahp":
        if ahp_matrix is None:
            raise ValueError("ahp_matrix required for AHP weights")
        return weight_ahp(ahp_matrix)
    elif method == "pca":
        return weight_pca(X)
    elif method == "equal":
        m = X.shape[1]
        return np.ones(m) / m
    else:
        raise ValueError(f"Unknown weight method: {method}")


# ── Evaluation Methods ─────────────────────────────────────────


def method_topsis(
    X: np.ndarray,
    weights: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
) -> np.ndarray:
    """TOPSIS: Technique for Order Preference by Similarity to Ideal Solution."""
    Xn = vector_normalize(X)
    W = weights / weights.sum()
    V = Xn * W

    # Ideal and anti-ideal
    directions = directions or ["max"] * X.shape[1]
    A_plus = np.zeros(X.shape[1])
    A_minus = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        if directions[j] == "max":
            A_plus[j] = V[:, j].max()
            A_minus[j] = V[:, j].min()
        else:
            A_plus[j] = V[:, j].min()
            A_minus[j] = V[:, j].max()

    D_plus = np.sqrt(np.sum((V - A_plus) ** 2, axis=1))
    D_minus = np.sqrt(np.sum((V - A_minus) ** 2, axis=1))

    scores = D_minus / (D_plus + D_minus + 1e-12)
    return scores


def method_vikor(
    X: np.ndarray,
    weights: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
    v: float = 0.5,
) -> np.ndarray:
    """VIKOR: compromise solution method. v=0.5 balances S and R."""
    Xn = normalize_matrix(X, directions)
    W = weights / weights.sum()

    f_plus = Xn.max(axis=0)
    f_minus = Xn.min(axis=0)

    S = np.sum(W * (f_plus - Xn) / (f_plus - f_minus + 1e-12), axis=1)
    R = np.max(W * (f_plus - Xn) / (f_plus - f_minus + 1e-12), axis=1)

    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()

    Q = v * (S - S_star) / (S_minus - S_star + 1e-12) + (1 - v) * (R - R_star) / (
        R_minus - R_star + 1e-12
    )

    # VIKOR scores: lower Q is better → invert for consistent ranking
    return 1.0 / (1.0 + Q)


def method_todim(
    X: np.ndarray,
    weights: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
    theta: float = 2.5,
) -> np.ndarray:
    """TODIM: interactive and multi-criteria decision making.
    theta: attenuation factor (default 2.5).
    """
    n, m = X.shape
    Xn = normalize_matrix(X, directions)
    W = weights / weights.sum()
    w_ref = W.max()

    dominance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            phi = 0.0
            for k in range(m):
                diff = Xn[i, k] - Xn[j, k]
                if diff > 0:
                    phi += np.sqrt(W[k] * diff / w_ref)
                elif diff < 0:
                    phi -= np.sqrt(W[k] * abs(diff) / w_ref) / theta
            dominance[i, j] = phi

    # Global dominance
    delta = np.sum(dominance, axis=1)
    # Normalize to [0, 1]
    delta_min, delta_max = delta.min(), delta.max()
    if delta_max - delta_min < 1e-12:
        return np.ones(n) * 0.5
    return (delta - delta_min) / (delta_max - delta_min)


def method_promethee(
    X: np.ndarray,
    weights: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
    preference_fn: Literal["usual", "linear", "level"] = "linear",
    p: float = 0.5,
    q: float = 0.1,
) -> np.ndarray:
    """PROMETHEE II: net flow score.
    p: preference threshold, q: indifference threshold.
    """
    n, m = X.shape
    Xn = normalize_matrix(X, directions)
    W = weights / weights.sum()

    preference = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pi = 0.0
            for k in range(m):
                diff = Xn[i, k] - Xn[j, k]
                if diff <= q:
                    pref = 0.0
                elif diff <= p:
                    pref = (diff - q) / (p - q)
                else:
                    pref = 1.0
                pi += W[k] * pref
            preference[i, j] = pi

    # Net flow
    positive = preference.sum(axis=1)
    negative = preference.sum(axis=0)
    net_flow = positive - negative

    # Normalize to [0, 1]
    nf_min, nf_max = net_flow.min(), net_flow.max()
    if nf_max - nf_min < 1e-12:
        return np.ones(n) * 0.5
    return (net_flow - nf_min) / (nf_max - nf_min)


def method_rsr(
    X: np.ndarray,
    weights: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
) -> np.ndarray:
    """RSR: Rank Sum Ratio method."""
    n, m = X.shape
    Xn = normalize_matrix(X, directions)

    # Rank each criterion (1 = best)
    ranks = np.zeros((n, m))
    for j in range(m):
        ranks[:, j] = n + 1 - np.argsort(np.argsort(Xn[:, j]))  # Descending rank

    # Weighted rank sum
    W = weights / weights.sum()
    RSR = np.sum(W * ranks / n, axis=1)

    # Normalize: lower RSR is better (lower rank sum = better)
    # Invert so higher score = better
    rsr_min, rsr_max = RSR.min(), RSR.max()
    if rsr_max - rsr_min < 1e-12:
        return np.ones(n) * 0.5
    return 1.0 - (RSR - rsr_min) / (rsr_max - rsr_min)


def method_grey(
    X: np.ndarray,
    weights: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
    rho: float = 0.5,
) -> np.ndarray:
    """Grey relational analysis. rho: distinguishing coefficient (0.5)."""
    Xn = normalize_matrix(X, directions)
    # Reference sequence: best values
    ref = Xn.max(axis=0)

    delta = np.abs(Xn - ref)
    delta_min = delta.min()
    delta_max = delta.max()

    xi = (delta_min + rho * delta_max) / (delta + rho * delta_max)
    W = weights / weights.sum()
    scores = np.sum(W * xi, axis=1)

    return scores


# ── Combinations ───────────────────────────────────────────────


def evaluate(
    alternatives: list[str],
    criteria: list[str],
    matrix: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
    weight_method: Literal[
        "entropy", "cv", "critic", "ahp", "pca", "equal"
    ] = "entropy",
    eval_method: Literal[
        "topsis", "vikor", "todim", "promethee", "rsr", "grey"
    ] = "topsis",
    ahp_matrix: np.ndarray | None = None,
    eval_kwargs: dict | None = None,
) -> EvaluationResult:
    """Unified MCDA evaluation.

    Args:
        alternatives: List of alternative names.
        criteria: List of criterion names.
        matrix: Decision matrix (n_alternatives × n_criteria).
        directions: 'max' or 'min' for each criterion.
        weight_method: Weight calculation method.
        eval_method: Evaluation method.
        ahp_matrix: Required if weight_method='ahp'.
        eval_kwargs: Extra args for eval method (e.g., v=0.5 for VIKOR).

    Returns:
        EvaluationResult with scores, ranking, and weights.
    """
    n, m = matrix.shape
    if len(alternatives) != n or len(criteria) != m:
        raise ValueError("Matrix dimensions must match alternatives and criteria")

    directions = directions or ["max"] * m
    eval_kwargs = eval_kwargs or {}

    # Compute weights
    weights_arr = compute_weights(matrix, weight_method, ahp_matrix)
    weights = {c: float(w) for c, w in zip(criteria, weights_arr)}

    # Evaluate
    method_fn = globals()[f"method_{eval_method}"]
    scores_arr = method_fn(matrix, weights_arr, directions, **eval_kwargs)

    scores = {a: float(s) for a, s in zip(alternatives, scores_arr)}
    ranking = [a for a, _ in sorted(scores.items(), key=lambda x: -x[1])]

    method_label = (
        f"{weight_method}-{eval_method}" if weight_method != "equal" else eval_method
    )

    return EvaluationResult(
        method=method_label,
        weights=weights,
        scores=scores,
        ranking=ranking,
    )


# ── Sensitivity Analysis ───────────────────────────────────────


def sensitivity_random_weights(
    alternatives: list[str],
    criteria: list[str],
    matrix: np.ndarray,
    directions: list[Literal["max", "min"]] | None = None,
    eval_method: Literal[
        "topsis", "vikor", "todim", "promethee", "rsr", "grey"
    ] = "topsis",
    n_trials: int = 1000,
    perturbation: float = 0.3,
    eval_kwargs: dict | None = None,
) -> dict:
    """Sensitivity analysis: random perturbation of weights.

    Returns dict with:
        - original_ranking: list of alternative names
        - stability_score: Spearman-like rank correlation stability
        - trials: list of trial results
        - top_frequency: frequency of each alternative being ranked #1
    """
    n, m = matrix.shape
    directions = directions or ["max"] * m
    eval_kwargs = eval_kwargs or {}

    # Original evaluation with equal weights as baseline
    base_weights = np.ones(m) / m
    method_fn = globals()[f"method_{eval_method}"]
    base_scores = method_fn(matrix, base_weights, directions, **eval_kwargs)
    base_ranking = np.argsort(-base_scores)

    # Run trials with perturbed weights
    rankings = []
    top_counts = dict.fromkeys(alternatives, 0)

    rng = np.random.default_rng(42)
    for _ in range(n_trials):
        # Perturb weights
        noise = rng.uniform(1 - perturbation, 1 + perturbation, size=m)
        w = base_weights * noise
        w = w / w.sum()

        scores = method_fn(matrix, w, directions, **eval_kwargs)
        trial_ranking = np.argsort(-scores)
        rankings.append(trial_ranking)
        top_counts[alternatives[trial_ranking[0]]] += 1

    # Stability: average Spearman correlation
    corr_sum = 0.0
    for trial_rank in rankings:
        # Compute rank correlation (simplified)
        ranks_base = np.argsort(base_ranking)
        ranks_trial = np.argsort(trial_rank)
        # Spearman: 1 - 6*sum(d^2)/(n*(n^2-1))
        d = ranks_base - ranks_trial
        rho = 1.0 - 6.0 * np.sum(d**2) / (n * (n**2 - 1))
        corr_sum += rho

    stability = corr_sum / n_trials

    return {
        "original_ranking": [alternatives[i] for i in base_ranking],
        "stability_score": float(stability),
        "n_trials": n_trials,
        "top_frequency": {a: c / n_trials for a, c in top_counts.items()},
    }
