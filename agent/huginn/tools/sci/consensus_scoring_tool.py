"""Consensus scoring tool — multi-model rank aggregation.

Inspired by drug discovery consensus docking: instead of trusting a single
scoring function, aggregate rankings from multiple models. Complements the
GP uncertainty approach with social choice theory methods.

Math:
  Borda count: candidate i gets score = Σ_models (n - rank_i)
  RRA (Robust Rank Aggregation): down-weights outliers
  Copeland: pairwise win counts
  Kemeny-optimal: NP-hard, approximated via Borda seed + local swaps

Arrow's impossibility theorem: no rank aggregation satisfies all 4 fairness
axioms simultaneously — Borda is the practical compromise.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile
from huginn.types import ToolContext, ToolResult

logger = logging.getLogger(__name__)


class ConsensusScoringInput(BaseModel):
    action: Literal[
        "borda",
        "robust_rank_aggregation",
        "copeland",
        "z_score",
        "rank_uncertainty",
    ] = Field(...)

    # Input: scores from multiple models
    # scores[model_name][candidate_idx] = score
    model_scores: dict[str, list[float]] | None = Field(
        default=None,
        description="Per-model raw scores: {model_name: [score1, score2, ...]}"
    )

    # Candidate names
    candidate_names: list[str] | None = Field(default=None)

    # Weights per model (if some models are trusted more)
    model_weights: dict[str, float] | None = Field(
        default=None, description="Trust weights per model"
    )

    # Direction
    maximize: bool = Field(default=True, description="Higher scores are better")

    # Borda params
    borda_variant: Literal["standard", "dowdall", "fractional"] = Field(
        default="standard",
        description="standard: n-rank; dowdall: 1/rank; fractional: for ties"
    )

    # RRA params
    alpha: float = Field(default=0.05, ge=0, le=1, description="RRA significance level")

    # Z-score params
    use_robust: bool = Field(default=True, description="Use median/MAD instead of mean/std")

    # Rank uncertainty params
    n_bootstrap: int = Field(default=500, ge=0, le=5000)


class ConsensusScoringTool(HuginnTool):
    """Consensus scoring via rank aggregation methods."""

    name = "consensus_scoring_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.VALIDATION, ResearchPhase.REPORTING}),
        light_alternatives=("gp_tool", "multi_fidelity_tool"),
    )
    description = (
        "Consensus scoring: aggregate rankings from multiple models using "
        "Borda count, robust rank aggregation, Copeland method, or z-score "
        "fusion. Complements GP uncertainty with social choice theory."
    )
    input_schema = ConsensusScoringInput

    async def _execute(self, args: ConsensusScoringInput, context: ToolContext) -> ToolResult:
        try:
            if not args.model_scores:
                return ToolResult(data=None, success=False, error="model_scores required")

            if args.action == "borda":
                return self._borda(args)
            if args.action == "robust_rank_aggregation":
                return self._rra(args)
            if args.action == "copeland":
                return self._copeland(args)
            if args.action == "z_score":
                return self._z_score(args)
            if args.action == "rank_uncertainty":
                return self._rank_uncertainty(args)
            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── Borda count ────────────────────────────────────────

    def _borda(self, args: ConsensusScoringInput) -> ToolResult:
        scores = args.model_scores
        model_names = list(scores.keys())
        n_models = len(model_names)
        n_candidates = len(next(iter(scores.values())))

        # Compute ranks per model
        borda_scores = np.zeros(n_candidates)
        for name in model_names:
            model_vals = np.array(scores[name])
            ranks = self._rank_values(model_vals, args.maximize)
            w = args.model_weights.get(name, 1.0) if args.model_weights else 1.0

            if args.borda_variant == "standard":
                # n - rank (top candidate gets n points)
                borda_scores += w * (n_candidates - ranks)
            elif args.borda_variant == "dowdall":
                # 1 / rank (Dowdall system)
                borda_scores += w * (1.0 / np.maximum(ranks + 1, 1))
            elif args.borda_variant == "fractional":
                borda_scores += w * (n_candidates - ranks)

        # Final ranking
        final_ranks = self._rank_values(borda_scores, maximize=True)

        # Also compute per-model agreement
        all_ranks = np.array([
            self._rank_values(np.array(scores[name]), args.maximize)
            for name in model_names
        ])

        # Kendall's W (coefficient of concordance)
        k_w = self._kendalls_w(all_ranks)

        # Identify consensus candidates (ranked high by ALL models)
        consensus = []
        for i in range(n_candidates):
            model_ranks = all_ranks[:, i]
            consensus.append({
                "candidate_index": i,
                "candidate_name": args.candidate_names[i] if args.candidate_names else f"cand_{i}",
                "borda_score": round(float(borda_scores[i]), 4),
                "final_rank": int(final_ranks[i]),
                "model_ranks": {name: int(all_ranks[j, i]) for j, name in enumerate(model_names)},
                "best_rank": int(np.min(model_ranks)),
                "worst_rank": int(np.max(model_ranks)),
                "rank_range": int(np.max(model_ranks) - np.min(model_ranks)),
                "unanimous_top3": bool(np.all(model_ranks < 3)),
            })

        consensus.sort(key=lambda x: x["borda_score"], reverse=True)

        data = {
            "action": "borda",
            "variant": args.borda_variant,
            "n_models": n_models,
            "n_candidates": n_candidates,
            "model_names": model_names,
            "kendalls_w": round(float(k_w), 4),
            "agreement": "strong" if k_w > 0.7 else "moderate" if k_w > 0.4 else "weak",
            "consensus_ranking": consensus[:20],
            "message": (
                f"Borda consensus: top candidate = {consensus[0]['candidate_name']} "
                f"(score={consensus[0]['borda_score']:.2f}). "
                f"Model agreement (Kendall W): {k_w:.3f} ({'strong' if k_w > 0.7 else 'moderate' if k_w > 0.4 else 'weak'})."
            ),
        }

        return ToolResult(data=data)

    # ── Robust Rank Aggregation ─────────────────────────────

    def _rra(self, args: ConsensusScoringInput) -> ToolResult:
        """Robust Rank Aggregation: down-weights outliers using beta distribution model.

        For each candidate, compute a p-value under the null hypothesis that
        all rankings are random. Uses the order statistics of uniform distribution.

        RRA score = -log10(p-value)
        """
        scores = args.model_scores
        model_names = list(scores.keys())
        n_models = len(model_names)
        n_candidates = len(next(iter(scores.values())))

        # Compute normalized ranks [0, 1] per model
        all_ranks = np.zeros((n_models, n_candidates))
        for j, name in enumerate(model_names):
            model_vals = np.array(scores[name])
            all_ranks[j] = self._rank_values(model_vals, args.maximize) / n_candidates

        # For each candidate, the ranks across models are order statistics
        # Under null (random rankings), r_(k) ~ Beta(k, n+1-k)
        # RRA score = -log10(p-value)
        try:
            from scipy.stats import beta as beta_dist
            has_scipy = True
        except ImportError:
            has_scipy = False
            beta_dist = None

        rra_scores = np.zeros(n_candidates)
        rra_pvalues = np.zeros(n_candidates)
        for i in range(n_candidates):
            sorted_ranks = np.sort(all_ranks[:, i])
            # Compute p-value for each order statistic
            pvals = []
            for k in range(1, n_models + 1):
                if has_scipy:
                    p = 1.0 - beta_dist.cdf(sorted_ranks[k - 1], k, n_models + 1 - k)
                else:
                    # Fallback: uniform null model, p = 1 - rank^k
                    p = 1.0 - sorted_ranks[k - 1] ** k
                pvals.append(p)

            # RRA p-value: minimum p-value corrected for multiple testing
            min_p = min(pvals)
            # Bonferroni correction
            corrected_p = min(1.0, min_p * n_models)
            rra_pvalues[i] = corrected_p
            rra_scores[i] = -math.log10(max(corrected_p, 1e-30))

        final_ranks = self._rank_values(rra_scores, maximize=True)

        results = []
        for i in range(n_candidates):
            results.append({
                "candidate_index": i,
                "candidate_name": args.candidate_names[i] if args.candidate_names else f"cand_{i}",
                "rra_score": round(float(rra_scores[i]), 4),
                "p_value": float(rra_pvalues[i]),
                "significant": bool(rra_pvalues[i] < args.alpha),
                "final_rank": int(final_ranks[i]),
                "model_ranks": {name: round(float(all_ranks[j, i]), 3) for j, name in enumerate(model_names)},
            })

        results.sort(key=lambda x: x["rra_score"], reverse=True)

        n_sig = sum(1 for r in results if r["significant"])

        return ToolResult(data={
            "action": "robust_rank_aggregation",
            "n_models": n_models,
            "n_candidates": n_candidates,
            "alpha": args.alpha,
            "n_significant": n_sig,
            "ranking": results[:20],
            "message": (
                f"RRA: {n_sig}/{n_candidates} candidates significant at α={args.alpha}. "
                f"Top: {results[0]['candidate_name']} (score={results[0]['rra_score']:.2f}, p={results[0]['p_value']:.4e})."
            ),
        })

    # ── Copeland's method ──────────────────────────────────

    def _copeland(self, args: ConsensusScoringInput) -> ToolResult:
        """Copeland's method: pairwise comparison, winner gets +1, loser -1, tie 0.

        Copeland score = Σ_j sign(score_i - score_j) averaged over models.
        Satisfies Condorcet criterion.
        """
        scores = args.model_scores
        model_names = list(scores.keys())
        n_models = len(model_names)
        n_candidates = len(next(iter(scores.values())))

        # Build pairwise win matrix
        copeland = np.zeros(n_candidates)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i == j:
                    continue
                wins = 0
                losses = 0
                for name in model_names:
                    si = scores[name][i]
                    sj = scores[name][j]
                    w = args.model_weights.get(name, 1.0) if args.model_weights else 1.0
                    if args.maximize:
                        if si > sj:
                            wins += w
                        elif si < sj:
                            losses += w
                    else:
                        if si < sj:
                            wins += w
                        elif si > sj:
                            losses += w
                if wins > losses:
                    copeland[i] += 1
                elif wins < losses:
                    copeland[i] -= 1
                # tie: 0

        final_ranks = self._rank_values(copeland, maximize=True)

        results = []
        for i in range(n_candidates):
            results.append({
                "candidate_index": i,
                "candidate_name": args.candidate_names[i] if args.candidate_names else f"cand_{i}",
                "copeland_score": int(copeland[i]),
                "final_rank": int(final_ranks[i]),
            })

        results.sort(key=lambda x: x["copeland_score"], reverse=True)

        # Check for Condorcet winner (beats all others)
        condorcet_idx = int(np.argmax(copeland))
        is_condorcet = copeland[condorcet_idx] == n_candidates - 1

        return ToolResult(data={
            "action": "copeland",
            "n_models": n_models,
            "n_candidates": n_candidates,
            "condorcet_winner": results[0]["candidate_name"] if is_condorcet else None,
            "has_condorcet_winner": is_condorcet,
            "ranking": results[:20],
            "message": (
                f"Copeland: top = {results[0]['candidate_name']} "
                f"(score={results[0]['copeland_score']:+d}). "
                f"Condorcet winner: {'yes' if is_condorcet else 'no (cycle exists)'}."
            ),
        })

    # ── Z-score fusion ──────────────────────────────────────

    def _z_score(self, args: ConsensusScoringInput) -> ToolResult:
        """Z-score fusion: normalize each model's scores, then combine.

        Robust version uses median + MAD instead of mean + std.
        """
        scores = args.model_scores
        model_names = list(scores.keys())
        n_models = len(model_names)
        n_candidates = len(next(iter(scores.values())))

        z_scores = np.zeros((n_models, n_candidates))
        for j, name in enumerate(model_names):
            vals = np.array(scores[name])
            if not args.maximize:
                vals = -vals

            if args.use_robust:
                # Robust: median + MAD
                med = np.median(vals)
                mad = np.median(np.abs(vals - med)) * 1.4826  # MAD → std
                z = (vals - med) / (mad + 1e-10)
            else:
                # Standard: mean + std
                z = (vals - np.mean(vals)) / (np.std(vals, ddof=1) + 1e-10)
            z_scores[j] = z

        # Weighted average
        weights = np.ones(n_models)
        if args.model_weights:
            for j, name in enumerate(model_names):
                weights[j] = args.model_weights.get(name, 1.0)
        weights = weights / weights.sum()

        combined = z_scores.T @ weights  # (n_candidates,)

        # Uncertainty: disagreement between models
        disagreement = np.std(z_scores, axis=0)  # per-candidate model spread

        final_ranks = self._rank_values(combined, maximize=True)

        results = []
        for i in range(n_candidates):
            results.append({
                "candidate_index": i,
                "candidate_name": args.candidate_names[i] if args.candidate_names else f"cand_{i}",
                "combined_z": round(float(combined[i]), 4),
                "model_disagreement": round(float(disagreement[i]), 4),
                "final_rank": int(final_ranks[i]),
                "model_z_scores": {
                    name: round(float(z_scores[j, i]), 4)
                    for j, name in enumerate(model_names)
                },
                "confidence": round(float(1.0 / (1.0 + disagreement[i])), 4),
            })

        results.sort(key=lambda x: x["combined_z"], reverse=True)

        return ToolResult(data={
            "action": "z_score",
            "robust": args.use_robust,
            "n_models": n_models,
            "n_candidates": n_candidates,
            "ranking": results[:20],
            "message": (
                f"Z-score fusion: top = {results[0]['candidate_name']} "
                f"(z={results[0]['combined_z']:.2f}, confidence={results[0]['confidence']:.3f})."
            ),
        })

    # ── Rank uncertainty ────────────────────────────────────

    def _rank_uncertainty(self, args: ConsensusScoringInput) -> ToolResult:
        """Bootstrap rank uncertainty: resample models to estimate rank stability."""
        scores = args.model_scores
        model_names = list(scores.keys())
        n_models = len(model_names)
        n_candidates = len(next(iter(scores.values())))

        all_scores = np.array([scores[name] for name in model_names])

        # Bootstrap: resample models with replacement, compute consensus rank
        rank_counts = np.zeros((n_candidates, n_candidates), dtype=int)

        rng = np.random.default_rng(42)
        for _ in range(args.n_bootstrap):
            # Sample models with replacement
            idx = rng.integers(0, n_models, size=n_models)
            sampled = all_scores[idx]

            # Simple average as consensus
            consensus = np.mean(sampled, axis=0) if args.maximize else -np.mean(sampled, axis=0)
            ranks = self._rank_values(consensus, maximize=True)

            for i in range(n_candidates):
                rank_counts[i, ranks[i]] += 1

        # Rank probabilities
        rank_probs = rank_counts / args.n_bootstrap

        # Expected rank and entropy
        expected_ranks = np.sum(rank_probs * np.arange(n_candidates)[:, None], axis=1)
        rank_entropy = -np.sum(rank_probs * np.log(np.maximum(rank_probs, 1e-30)), axis=1)
        max_entropy = math.log(n_candidates)

        results = []
        for i in range(n_candidates):
            most_likely_rank = int(np.argmax(rank_probs[i]))
            results.append({
                "candidate_index": i,
                "candidate_name": args.candidate_names[i] if args.candidate_names else f"cand_{i}",
                "expected_rank": round(float(expected_ranks[i]), 2),
                "most_likely_rank": most_likely_rank,
                "rank_probability": round(float(rank_probs[i, most_likely_rank]), 4),
                "rank_entropy": round(float(rank_entropy[i]), 4),
                "normalized_entropy": round(float(rank_entropy[i] / max_entropy), 4),
                "rank_stability": "stable" if rank_entropy[i] / max_entropy < 0.3
                                  else "moderate" if rank_entropy[i] / max_entropy < 0.6
                                  else "unstable",
                "rank_distribution": [round(float(p), 4) for p in rank_probs[i]],
            })

        results.sort(key=lambda x: x["expected_rank"])

        return ToolResult(data={
            "action": "rank_uncertainty",
            "n_bootstrap": args.n_bootstrap,
            "n_models": n_models,
            "n_candidates": n_candidates,
            "ranking": results[:20],
            "message": (
                f"Rank uncertainty: most stable candidate = {results[0]['candidate_name']} "
                f"(expected rank={results[0]['expected_rank']:.2f}, "
                f"stability={results[0]['rank_stability']})."
            ),
        })

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _rank_values(values: np.ndarray, maximize: bool = True) -> np.ndarray:
        """Return ranks (0=best). Handles ties with average rank."""
        if maximize:
            values = -values  # negate so ascending sort puts largest first
        # Use average rank for ties
        temp = values.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(values))
        # Handle ties: group equal values and assign average rank
        sorted_vals = values[temp]
        i = 0
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                avg_rank = np.mean(ranks[temp[i:j + 1]])
                ranks[temp[i:j + 1]] = avg_rank
            i = j + 1
        return ranks

    @staticmethod
    def _kendalls_w(rank_matrix: np.ndarray) -> float:
        """Kendall's coefficient of concordance W.
        W = 12 * Σ(R_i - R̄)² / (k²(n³-n))
        where R_i = sum of ranks for candidate i, k = number of rankers, n = candidates.
        """
        k, n = rank_matrix.shape
        if n < 2 or k < 2:
            return 1.0
        R = rank_matrix.sum(axis=0)  # sum of ranks per candidate
        R_bar = R.mean()
        S = np.sum((R - R_bar) ** 2)
        W = 12 * S / (k ** 2 * (n ** 3 - n))
        return float(np.clip(W, 0, 1))
