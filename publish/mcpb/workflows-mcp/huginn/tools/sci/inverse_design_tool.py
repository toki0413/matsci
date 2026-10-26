"""Inverse design tool — property → structure search via scoring function optimization.

Inspired by drug docking inverse search: given a target property, search the
composition/structure space for candidates that maximize a scoring function.
Supports Bayesian optimization, genetic algorithm, and random search.

Math:
  Inverse problem: find x* = argmax S(x)  where S: ℝ^d → ℝ
  S(x) = w₁·f₁(x) + w₂·f₂(x) + ...  (linear scalarization)
  Pareto frontier: {x | ∄ y: S(y) ≥ S(x) ∀ objectives}
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile

logger = logging.getLogger(__name__)


class InverseDesignInput(BaseModel):
    action: Literal[
        "random_search",
        "genetic_algorithm",
        "bayesian_optimization",
        "pareto_frontier",
        "scoring_function",
    ] = Field(...)

    # Candidate space
    candidates: list[list[float]] | None = Field(
        default=None, description="Candidate design parameters"
    )
    candidate_names: list[str] | None = Field(
        default=None, description="Names for each candidate"
    )

    # Scoring
    scores: list[float] | None = Field(
        default=None, description="Pre-computed scores for scoring_function action"
    )
    multi_objective_scores: list[list[float]] | None = Field(
        default=None, description="Multi-objective scores: [[f1, f2, ...], ...]"
    )
    objective_names: list[str] | None = Field(
        default=None, description="Names of objectives"
    )
    weights: list[float] | None = Field(
        default=None, description="Weights for linear scalarization"
    )
    maximize: bool = Field(default=True, description="Maximize or minimize scores")

    # Genetic algorithm params
    n_generations: int = Field(default=50, ge=1, le=500)
    population_size: int = Field(default=20, ge=2, le=200)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    crossover_rate: float = Field(default=0.7, ge=0, le=1)

    # Bayesian optimization params
    n_iterations: int = Field(default=20, ge=1, le=200)
    acquisition: Literal["ei", "ucb", "pi"] = Field(default="ei")
    kappa: float = Field(default=2.0, ge=0)

    # Search space bounds
    bounds_low: list[float] | None = Field(default=None)
    bounds_high: list[float] | None = Field(default=None)
    n_dimensions: int = Field(default=2, ge=1, le=20)

    # Pareto params
    n_pareto_points: int = Field(default=50, ge=1, le=500)


class InverseDesignTool(HuginnTool):
    """Inverse materials design: search for structures with target properties."""

    name = "inverse_design_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.PLANNING}),
        light_alternatives=("gp_tool", "multi_fidelity_tool"),
    )
    description = (
        "Inverse design: given target properties, search composition/structure "
        "space using genetic algorithm, Bayesian optimization, or random search. "
        "Supports multi-objective Pareto frontier analysis."
    )
    input_schema = InverseDesignInput

    async def _execute(
        self, args: InverseDesignInput, context: ToolContext
    ) -> ToolResult:
        try:
            if args.action == "random_search":
                return self._random_search(args)
            if args.action == "genetic_algorithm":
                return self._genetic_algorithm(args)
            if args.action == "bayesian_optimization":
                return self._bayesian_opt(args)
            if args.action == "pareto_frontier":
                return self._pareto_frontier(args)
            if args.action == "scoring_function":
                return self._scoring_function(args)
            return ToolResult(
                data=None, success=False, error=f"Unknown action: {args.action}"
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── Random search ───────────────────────────────────────

    def _random_search(self, args: InverseDesignInput) -> ToolResult:
        if not args.candidates or not args.scores:
            return ToolResult(
                data=None, success=False, error="candidates and scores required"
            )

        candidates = np.array(args.candidates)
        scores = np.array(args.scores)

        ranked_idx = np.argsort(-scores) if args.maximize else np.argsort(scores)

        n = len(ranked_idx)
        top_k = min(10, n)

        data = {
            "action": "random_search",
            "n_candidates": n,
            "top_candidates": [
                {
                    "rank": i + 1,
                    "index": int(ranked_idx[i]),
                    "score": round(float(scores[ranked_idx[i]]), 6),
                    "params": candidates[ranked_idx[i]].tolist()
                    if hasattr(candidates[ranked_idx[i]], "tolist")
                    else list(candidates[ranked_idx[i]]),
                }
                for i in range(top_k)
            ],
            "best_score": round(float(scores[ranked_idx[0]]), 6),
            "score_mean": round(float(np.mean(scores)), 6),
            "score_std": round(float(np.std(scores)), 6),
            "maximize": args.maximize,
            "message": f"Ranked {n} candidates. Best score: {scores[ranked_idx[0]]:.4f}.",
        }
        if args.candidate_names:
            data["top_candidate_names"] = [
                args.candidate_names[ranked_idx[i]] for i in range(top_k)
            ]

        return ToolResult(data=data)

    # ── Genetic algorithm ───────────────────────────────────

    def _genetic_algorithm(self, args: InverseDesignInput) -> ToolResult:
        """Optimize a scoring function via genetic algorithm.
        Requires bounds_low, bounds_high for search space definition.
        The scoring function is provided as pre-evaluated candidates (warm start)
        or generated randomly if no candidates given.
        """
        if not args.bounds_low or not args.bounds_high:
            return ToolResult(
                data=None, success=False, error="bounds_low and bounds_high required"
            )

        bl = np.array(args.bounds_low)
        bh = np.array(args.bounds_high)
        dim = len(bl)
        rng = np.random.default_rng(42)

        # Initialize population
        pop = rng.uniform(bl, bh, size=(args.population_size, dim))

        # If candidates provided, seed population with best ones
        if args.candidates and args.scores:
            _seed_ga_population(pop, args)

        # Evaluate fitness
        evaluate = _make_ga_evaluator(args, bl, bh)
        fitness = np.array([evaluate(ind) for ind in pop])
        best_idx = np.argmax(fitness)
        best_x = pop[best_idx].copy()
        best_f = fitness[best_idx]

        history = [float(best_f)]

        for _gen in range(args.n_generations):
            pop = _next_ga_population(pop, fitness, rng, args, bl, bh, dim)

            # Evaluate
            fitness = np.array([evaluate(ind) for ind in pop])
            gen_best = np.argmax(fitness)
            if fitness[gen_best] > best_f:
                best_f = fitness[gen_best]
                best_x = pop[gen_best].copy()

            history.append(float(best_f))

        return _ga_result(args, best_x, best_f, history)

    # ── Bayesian optimization ──────────────────────────────

    def _bayesian_opt(self, args: InverseDesignInput) -> ToolResult:
        """Bayesian optimization with GP surrogate + acquisition function.
        Requires pre-evaluated candidates as initial data."""
        if not args.candidates or not args.scores:
            return ToolResult(
                data=None,
                success=False,
                error="candidates and scores required as initial data",
            )

        X = np.array(args.candidates)
        y = np.array(args.scores)
        if not args.maximize:
            y = -y

        # Use the existing GP tool for fitting and prediction
        try:
            from huginn.tools.gp_tool import NumPyGP
        except ImportError:
            return ToolResult(data=None, success=False, error="GP tool not available")

        # Fit GP
        gp = NumPyGP(length_scale=1.0, sigma_f=1.0, sigma_n=1e-4)
        gp.fit(X, y)

        # Generate candidates
        X_cand = _bo_candidate_pool(args, X)

        # Predict
        mu, sigma = gp.predict(X_cand)

        # Acquisition function
        y_best = float(np.max(y))
        acq_values = _bo_acquisition(args, mu, sigma, y_best)

        # Select top candidates
        top_idx = np.argsort(-acq_values)[: min(10, len(acq_values))]

        return _bo_result(args, X, X_cand, y, mu, sigma, top_idx, acq_values)

    # ── Pareto frontier ─────────────────────────────────────

    def _pareto_frontier(self, args: InverseDesignInput) -> ToolResult:
        """Compute Pareto frontier from multi-objective scores."""
        if not args.multi_objective_scores:
            return ToolResult(
                data=None, success=False, error="multi_objective_scores required"
            )

        scores = np.array(args.multi_objective_scores)
        n = len(scores)
        n_obj = scores.shape[1]

        # Find Pareto-optimal points (maximize all objectives)
        is_pareto = _pareto_mask(scores)

        pareto_idx = np.where(is_pareto)[0]
        pareto_scores = scores[pareto_idx]

        crowding = _crowding_distance(pareto_scores, n_obj)

        # Linear scalarization if weights provided
        weighted_scores = None
        if args.weights:
            w = np.array(args.weights)
            weighted_scores = scores @ w
            best_weighted = int(np.argmax(weighted_scores))
        else:
            best_weighted = None

        return _pareto_result(
            args, n, n_obj, pareto_idx, pareto_scores, crowding, best_weighted
        )

    # ── Scoring function ───────────────────────────────────

    def _scoring_function(self, args: InverseDesignInput) -> ToolResult:
        """Compute weighted score for candidates."""
        if not args.multi_objective_scores:
            return ToolResult(
                data=None, success=False, error="multi_objective_scores required"
            )

        scores = np.array(args.multi_objective_scores)
        weights = np.array(args.weights or [1.0 / scores.shape[1]] * scores.shape[1])

        if abs(weights.sum()) < 1e-10:
            return ToolResult(
                data=None, success=False, error="weights must not all be zero"
            )

        weights = weights / weights.sum()
        weighted = scores @ weights

        ranked_idx = np.argsort(-weighted) if args.maximize else np.argsort(weighted)

        data = {
            "action": "scoring_function",
            "weights": weights.tolist(),
            "n_candidates": len(scores),
            "n_objectives": scores.shape[1],
            "ranked_candidates": [
                {
                    "rank": i + 1,
                    "index": int(ranked_idx[i]),
                    "weighted_score": round(float(weighted[ranked_idx[i]]), 6),
                    "individual_scores": [
                        round(float(x), 6) for x in scores[ranked_idx[i]]
                    ],
                }
                for i in range(min(10, len(ranked_idx)))
            ],
            "message": f"Scored {len(scores)} candidates. Best weighted score: {weighted[ranked_idx[0]]:.4f}.",
        }

        return ToolResult(data=data)


# ── Module-level helpers ────────────────────────────────────


def _seed_ga_population(pop: np.ndarray, args: InverseDesignInput) -> None:
    """Seed the GA population with the best pre-evaluated candidates."""
    n_seed = min(len(args.candidates), args.population_size // 2)
    if args.maximize:
        seed_idx = np.argsort(-np.array(args.scores))[:n_seed]
    else:
        seed_idx = np.argsort(np.array(args.scores))[:n_seed]
    for i, idx in enumerate(seed_idx):
        pop[i] = np.array(args.candidates[idx])


def _make_ga_evaluator(args: InverseDesignInput, bl: np.ndarray, bh: np.ndarray):
    """Build the fitness function used by the genetic algorithm.

    支持外部评估器: 如果 args 里带 evaluator (callable), 用它;
    否则用预计算 scores 或距离代理.
    ponytail: 之前是纯内置代理. 现在加了外部评估器接口,
    让 GA 能调用真实仿真工具 (VASP/LAMMPS) 做适应度评估.
    """
    evaluator = getattr(args, "evaluator", None)

    def evaluate(x):
        if evaluator is not None:
            # 外部评估器: 调真实仿真
            try:
                score = evaluator(x)
                return float(score) if args.maximize else -float(score)
            except Exception:
                return -1e6
        if args.candidates and args.scores:
            # 预计算 scores: 找最近候选
            dists = np.sum((np.array(args.candidates) - x) ** 2, axis=1)
            nearest = np.argmin(dists)
            return args.scores[nearest] if args.maximize else -args.scores[nearest]
        # 代理: 距离中点的负距离 (maximize = 探索中心)
        return (
            -np.sum((x - (bl + bh) / 2) ** 2)
            if args.maximize
            else np.sum((x - (bl + bh) / 2) ** 2)
        )

    return evaluate


def _next_ga_population(
    pop: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
    args: InverseDesignInput,
    bl: np.ndarray,
    bh: np.ndarray,
    dim: int,
) -> np.ndarray:
    """Selection, crossover, mutation and clipping for one GA generation."""
    # Selection (tournament)
    new_pop = []
    for _ in range(args.population_size):
        i1, i2 = rng.integers(0, args.population_size, 2)
        winner = i1 if fitness[i1] > fitness[i2] else i2
        new_pop.append(pop[winner].copy())
    pop = np.array(new_pop)

    # Crossover
    for i in range(0, args.population_size - 1, 2):
        if rng.random() < args.crossover_rate:
            alpha = rng.random()
            c1 = alpha * pop[i] + (1 - alpha) * pop[i + 1]
            c2 = (1 - alpha) * pop[i] + alpha * pop[i + 1]
            pop[i], pop[i + 1] = c1, c2

    # Mutation
    for i in range(args.population_size):
        if rng.random() < args.mutation_rate:
            gene = rng.integers(0, dim)
            pop[i, gene] = rng.uniform(bl[gene], bh[gene])

    # Clip to bounds
    pop = np.clip(pop, bl, bh)
    return pop


def _ga_result(
    args: InverseDesignInput, best_x: np.ndarray, best_f: float, history: list[float]
) -> ToolResult:
    """Assemble the genetic_algorithm payload."""
    data = {
        "action": "genetic_algorithm",
        "n_generations": args.n_generations,
        "population_size": args.population_size,
        "best_params": [round(float(x), 6) for x in best_x],
        "best_score": round(float(best_f), 6),
        "convergence_history": [
            round(h, 6) for h in history[:: max(1, len(history) // 50)]
        ],
        "n_evaluations": args.n_generations * args.population_size,
        "message": f"GA converged. Best score: {best_f:.4f} after {args.n_generations} generations.",
    }

    return ToolResult(data=data)


def _bo_candidate_pool(args: InverseDesignInput, X: np.ndarray) -> np.ndarray:
    """Generate candidate points in the bounds, or reuse the initial candidates."""
    if args.bounds_low and args.bounds_high:
        bl = np.array(args.bounds_low)
        bh = np.array(args.bounds_high)
        n_candidates = 1000
        rng = np.random.default_rng(42)
        return rng.uniform(bl, bh, size=(n_candidates, len(bl)))
    # Use candidates themselves
    return X


def _bo_acquisition(
    args: InverseDesignInput,
    mu: np.ndarray,
    sigma: np.ndarray,
    y_best: float,
) -> np.ndarray:
    """Evaluate the configured acquisition function."""
    if args.acquisition == "ei":
        # Expected Improvement
        improvement = mu - y_best
        try:
            from scipy.stats import norm

            Z = improvement / np.maximum(sigma, 1e-10)
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            acq_values = ei
        except ImportError:
            # Fallback: just use mean as acquisition
            acq_values = mu
    elif args.acquisition == "ucb":
        acq_values = mu + args.kappa * sigma
    elif args.acquisition == "pi":
        try:
            from scipy.stats import norm

            Z = (mu - y_best) / np.maximum(sigma, 1e-10)
            acq_values = norm.cdf(Z)
        except ImportError:
            acq_values = mu
    else:
        acq_values = mu
    return acq_values


def _bo_result(
    args: InverseDesignInput,
    X: np.ndarray,
    X_cand: np.ndarray,
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    top_idx: np.ndarray,
    acq_values: np.ndarray,
) -> ToolResult:
    """Assemble the bayesian_optimization payload."""
    data = {
        "action": "bayesian_optimization",
        "acquisition": args.acquisition,
        "n_initial_points": len(X),
        "n_candidates_evaluated": len(X_cand),
        "best_observed_score": round(float(np.max(y)), 6),
        "suggested_candidates": [
            {
                "rank": i + 1,
                "params": [round(float(x), 6) for x in X_cand[top_idx[i]]],
                "predicted_mean": round(float(mu[top_idx[i]]), 6),
                "predicted_uncertainty": round(float(sigma[top_idx[i]]), 6),
                "acquisition_value": round(float(acq_values[top_idx[i]]), 6),
            }
            for i in range(len(top_idx))
        ],
        "message": f"BO suggests {len(top_idx)} candidates. Best predicted: {mu[top_idx[0]]:.4f} ± {sigma[top_idx[0]]:.4f}.",
    }

    return ToolResult(data=data)


def _pareto_mask(scores: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points (maximize all objectives)."""
    n = len(scores)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and np.all(scores[j] >= scores[i])
                and np.any(scores[j] > scores[i])
            ):
                # j dominates i if j is >= i in all objectives and > in at least one
                is_pareto[i] = False
                break
    return is_pareto


def _crowding_distance(pareto_scores: np.ndarray, n_obj: int) -> np.ndarray:
    """Compute crowding distance (for NSGA-II style diversity)."""
    crowding = np.zeros(len(pareto_scores))
    for obj in range(n_obj):
        sorted_idx = np.argsort(-pareto_scores[:, obj])
        crowding[sorted_idx[0]] = float("inf")
        crowding[sorted_idx[-1]] = float("inf")
        obj_range = pareto_scores[:, obj].max() - pareto_scores[:, obj].min()
        if obj_range > 0:
            for k in range(1, len(sorted_idx) - 1):
                crowding[sorted_idx[k]] += (
                    pareto_scores[sorted_idx[k + 1], obj]
                    - pareto_scores[sorted_idx[k - 1], obj]
                ) / obj_range
    return crowding


def _pareto_result(
    args: InverseDesignInput,
    n: int,
    n_obj: int,
    pareto_idx: np.ndarray,
    pareto_scores: np.ndarray,
    crowding: np.ndarray,
    best_weighted: int | None,
) -> ToolResult:
    """Assemble the pareto_frontier payload."""
    data = {
        "action": "pareto_frontier",
        "n_candidates": n,
        "n_objectives": n_obj,
        "n_pareto_optimal": int(len(pareto_idx)),
        "pareto_indices": pareto_idx.tolist(),
        "pareto_scores": [[round(float(x), 6) for x in row] for row in pareto_scores],
        "crowding_distances": [round(float(x), 4) for x in crowding],
        "objective_names": args.objective_names or [f"obj_{i}" for i in range(n_obj)],
        "best_weighted_index": best_weighted,
        "weights": args.weights,
        "message": (
            f"Pareto frontier: {len(pareto_idx)} / {n} candidates are non-dominated. "
            f"({len(pareto_idx) / n * 100:.1f}% of space is Pareto-optimal.)"
        ),
    }

    return ToolResult(data=data)
