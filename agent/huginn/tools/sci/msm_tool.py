"""Markov State Model tool — trajectory → metastable states → kinetics.

Builds MSMs from MD trajectories: cluster frames into microstates, estimate
transition matrix, extract eigenvalues/eigenvectors for timescale analysis,
and identify metastable macrostates via Perron-cluster cluster analysis (PCCA+).

Math:
  T_ij(τ) = P(S_j at t+τ | S_i at t)  (row-stochastic)
  Perron-Frobenius: eigenvalues 1=λ₁ > λ₂ ≥ ... ≥ λ_k > 0
  Implied timescale: t_k = −τ / ln(λ_k)
  Stationary distribution: π = left eigenvector at λ=1
  PCCA+: macrostate assignment from top eigenvectors (linear algebra on simplex)
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

KB_EV = 8.617333262e-5


class MSMInput(BaseModel):
    action: Literal[
        "build_msm",
        "timescales",
        "metastable_states",
        "stationary_distribution",
        "commitment_probabilities",
        "transition_pathway",
    ] = Field(...)

    # Trajectory data
    trajectory: list[list[float]] | None = Field(
        default=None,
        description="Trajectory frames: trajectory[i] = feature vector at frame i"
    )
    distance_matrix: list[list[float]] | None = Field(
        default=None,
        description="Precomputed N×N distance matrix between frames"
    )

    # Clustering params
    n_microstates: int = Field(default=50, ge=2, le=500, description="Number of microstates (k-means clusters)")
    n_macrostates: int = Field(default=3, ge=2, le=20, description="Number of macrostates for PCCA+")

    # MSM params
    lag_time: int = Field(default=1, ge=1, description="Lag time τ (in frames)")
    reversible: bool = Field(default=True, description="Enforce detailed balance (reversible MSM)")

    # Temperature for rate conversion
    temperature: float = Field(default=300.0, gt=0, description="Temperature (K)")
    dt_per_frame: float = Field(default=1.0, gt=0, description="Time per frame (fs or ps)")

    # Pre-built transition matrix (for eigenvalue analysis only)
    transition_matrix: list[list[float]] | None = Field(
        default=None, description="Pre-built row-stochastic transition matrix"
    )


class MSMTool(HuginnTool):
    """Markov State Model: trajectory clustering, transition matrix, kinetics."""

    name = "msm_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.VALIDATION, ResearchPhase.REPORTING}),
        light_alternatives=("tda_tool", "enhanced_sampling_tool"),
    )
    description = (
        "Markov State Model analysis: cluster MD trajectory into microstates, "
        "estimate transition matrix, compute implied timescales, identify "
        "metastable states via PCCA+, and compute stationary/commitment probabilities."
    )
    input_schema = MSMInput

    async def _execute(self, args: MSMInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "build_msm":
                return self._build_msm(args)
            if args.action == "timescales":
                return self._timescales(args)
            if args.action == "metastable_states":
                return self._metastable_states(args)
            if args.action == "stationary_distribution":
                return self._stationary(args)
            if args.action == "commitment_probabilities":
                return self._commitment(args)
            if args.action == "transition_pathway":
                return self._transition_pathway(args)
            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── Build MSM ───────────────────────────────────────────

    def _build_msm(self, args: MSMInput) -> ToolResult:
        """Cluster trajectory, estimate transition matrix at given lag time."""
        if args.transition_matrix:
            T = np.array(args.transition_matrix)
            # Validate row-stochastic
            row_sums = T.sum(axis=1, keepdims=True)
            T = T / np.maximum(row_sums, 1e-30)
            n_states = T.shape[0]
            # Microstate assignments not needed
            assignments = []
        elif args.trajectory:
            traj = np.array(args.trajectory)
            assignments, centroids = self._kmeans(traj, args.n_microstates)
            T = self._estimate_transition_matrix(assignments, args.lag_time, args.reversible)
            n_states = args.n_microstates
        else:
            return ToolResult(data=None, success=False,
                              error="trajectory or transition_matrix required")

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = self._eigendecompose(T)

        # Implied timescales
        timescales = self._compute_timescales(eigenvalues, args.lag_time, args.dt_per_frame)

        data = {
            "action": "build_msm",
            "n_states": n_states,
            "lag_time_frames": args.lag_time,
            "dt_per_frame": args.dt_per_frame,
            "transition_matrix": [[round(float(x), 6) for x in row] for row in T[:20]],  # cap
            "eigenvalues": [round(float(x), 8) for x in eigenvalues[:20]],
            "implied_timescales": [round(float(t), 4) for t in timescales[:20]],
            "n_macrostates_suggested": self._suggest_macrostates(eigenvalues),
            "reversible": args.reversible,
            "temperature_K": args.temperature,
            "message": f"MSM built: {n_states} states, lag τ={args.lag_time} frames.",
        }
        if args.trajectory:
            data["microstate_assignments"] = assignments[:500]  # cap

        return ToolResult(data=data)

    # ── Timescales ──────────────────────────────────────────

    def _timescales(self, args: MSMInput) -> ToolResult:
        """Compute implied timescales at multiple lag times for validation."""
        if not args.trajectory:
            return ToolResult(data=None, success=False, error="trajectory required")

        traj = np.array(args.trajectory)
        assignments, _ = self._kmeans(traj, args.n_microstates)

        lag_times = [1, 2, 5, 10, 20, 50]
        all_timescales = []
        for lag in lag_times:
            if lag >= len(assignments):
                break
            T = self._estimate_transition_matrix(assignments, lag, args.reversible)
            evals, _ = self._eigendecompose(T)
            ts = self._compute_timescales(evals, lag, args.dt_per_frame)
            all_timescales.append({
                "lag_time": lag,
                "timescales": [round(float(t), 4) for t in ts[:10]],
            })

        return ToolResult(data={
            "action": "timescales",
            "lag_time_results": all_timescales,
            "n_microstates": args.n_microstates,
            "dt_per_frame": args.dt_per_frame,
            "message": "Implied timescales computed at multiple lag times. "
                       "Look for plateau region indicating Markovian behavior.",
        })

    # ── Metastable states (PCCA+) ──────────────────────────

    def _metastable_states(self, args: MSMInput) -> ToolResult:
        """Identify metastable macrostates using PCCA+."""
        if not args.trajectory:
            return ToolResult(data=None, error="trajectory required")

        traj = np.array(args.trajectory)
        assignments, _ = self._kmeans(traj, args.n_microstates)
        T = self._estimate_transition_matrix(assignments, args.lag_time, args.reversible)
        eigenvalues, eigenvectors = self._eigendecompose(T)

        n_macro = args.n_macrostates
        # PCCA+: use top n_macro-1 eigenvectors (excluding stationary)
        # This is a simplified PCCA+ — full implementation needs optimization on the simplex
        # ponytail: simplified PCCA+ via k-means on eigenvector space
        V = eigenvectors[:, 1:n_macro]  # exclude stationary eigenvector
        macro_labels, macro_centers = self._kmeans(V, n_macro)

        # Compute coarse-grained transition matrix
        T_coarse = self._coarse_grain(T, macro_labels, n_macro)

        # Stationary distribution per macrostate
        stat_eigvec = eigenvectors[:, 0]
        stat_dist = np.abs(stat_eigvec) / np.sum(np.abs(stat_eigvec))

        # Macrostate stationary probabilities
        macro_stat = np.zeros(n_macro)
        for i in range(len(macro_labels)):
            macro_stat[macro_labels[i]] += stat_dist[i]

        # Mean first passage times
        mfpt = self._compute_mfpt(T_coarse, macro_stat)

        data = {
            "action": "metastable_states",
            "n_macrostates": n_macro,
            "n_microstates": args.n_microstates,
            "macrostate_labels": macro_labels[:500].tolist(),
            "coarse_transition_matrix": [[round(float(x), 6) for x in row] for row in T_coarse],
            "stationary_probabilities": [round(float(x), 4) for x in macro_stat],
            "mfpt_matrix": [[round(float(x), 4) for x in row] for row in mfpt],
            "lag_time_frames": args.lag_time,
            "dt_per_frame": args.dt_per_frame,
            "temperature_K": args.temperature,
            "message": (
                f"Identified {n_macro} metastable states. "
                f"Most stable: state {np.argmax(macro_stat)} "
                f"(P = {macro_stat[np.argmax(macro_stat)]:.3f})."
            ),
        }

        return ToolResult(data=data)

    # ── Stationary distribution ─────────────────────────────

    def _stationary(self, args: MSMInput) -> ToolResult:
        """Compute stationary distribution."""
        if args.transition_matrix:
            T = np.array(args.transition_matrix)
            T = T / T.sum(axis=1, keepdims=True)
        elif args.trajectory:
            traj = np.array(args.trajectory)
            assignments, _ = self._kmeans(traj, args.n_microstates)
            T = self._estimate_transition_matrix(assignments, args.lag_time, args.reversible)
        else:
            return ToolResult(data=None, success=False, error="trajectory or transition_matrix required")

        eigenvalues, eigenvectors = self._eigendecompose(T)
        stat_dist = np.abs(eigenvectors[:, 0])
        stat_dist = stat_dist / stat_dist.sum()

        # Entropy of stationary distribution (higher = more dispersed)
        entropy = -np.sum(stat_dist * np.log(np.maximum(stat_dist, 1e-30)))
        max_entropy = math.log(T.shape[0])

        data = {
            "action": "stationary_distribution",
            "stationary_distribution": [round(float(x), 6) for x in stat_dist[:50]],
            "n_states": T.shape[0],
            "dominant_state": int(np.argmax(stat_dist)),
            "dominant_probability": round(float(stat_dist[np.argmax(stat_dist)]), 6),
            "entropy": round(float(entropy), 6),
            "normalized_entropy": round(float(entropy / max_entropy), 6),
            "message": (
                f"Stationary distribution: dominant state {np.argmax(stat_dist)} "
                f"(P = {stat_dist[np.argmax(stat_dist)]:.4f}), "
                f"entropy = {entropy/max_entropy:.3f} of max."
            ),
        }

        return ToolResult(data=data)

    # ── Commitment probabilities (committor) ──────────────

    def _commitment(self, args: MSMInput) -> ToolResult:
        """Compute forward/backward committor probabilities.

        Forward committor q_i^+ = P(hit B before A | start at i)
        Solves: (I − T_AA) q_A^+ = T_AB 1_B  with q_B = 1, q_A = 0
        """
        if not args.transition_matrix and not args.trajectory:
            return ToolResult(data=None, success=False, error="transition_matrix or trajectory required")

        if args.transition_matrix:
            T = np.array(args.transition_matrix)
        else:
            traj = np.array(args.trajectory)
            assignments, _ = self._kmeans(traj, args.n_microstates)
            T = self._estimate_transition_matrix(assignments, args.lag_time, args.reversible)

        T = T / T.sum(axis=1, keepdims=True)
        n = T.shape[0]

        # Define A = first state, B = last state (simplified)
        A_idx = [0]
        B_idx = [n - 1]
        intermediate = list(range(1, n - 1))

        if not intermediate:
            return ToolResult(data=None, success=False, error="Need at least 3 states for committor")

        T_ii = T[np.ix_(intermediate, intermediate)]
        T_iB = T[np.ix_(intermediate, B_idx)].sum(axis=1)

        # Solve (I - T_ii) q^+ = T_iB
        q_plus = np.linalg.solve(np.eye(len(intermediate)) - T_ii, T_iB)

        # Backward committor via detailed balance: q^- = 1 - q^+ (for reversible)
        q_minus = 1.0 - q_plus if args.reversible else None

        # Transition path probability: TPP_i = π_i q_i^+ q_i^- / Z
        _, eigenvectors = self._eigendecompose(T)
        pi = np.abs(eigenvectors[:, 0])
        pi = pi / pi.sum()

        q_full = np.zeros(n)
        q_full[B_idx] = 1.0
        q_full[intermediate] = q_plus

        q_minus_full = np.zeros(n)
        q_minus_full[A_idx] = 1.0
        if q_minus is not None:
            q_minus_full[intermediate] = q_minus
        else:
            # Non-reversible: solve backward committor separately
            T_iA = T[np.ix_(intermediate, A_idx)].sum(axis=1)
            q_minus_full[intermediate] = np.linalg.solve(
                np.eye(len(intermediate)) - T_ii.T, T_iA
            )

        tpp = pi * q_full * q_minus_full
        tpp = tpp / tpp.sum() if tpp.sum() > 0 else tpp

        data = {
            "action": "commitment_probabilities",
            "forward_committor": [round(float(x), 6) for x in q_full],
            "backward_committor": [round(float(x), 6) for x in q_minus_full],
            "transition_path_probability": [round(float(x), 6) for x in tpp],
            "state_A": A_idx,
            "state_B": B_idx,
            "n_states": n,
            "message": (
                f"Committor computed. "
                f"Transition path ensemble concentrated at state {np.argmax(tpp)} "
                f"(TPP = {tpp[np.argmax(tpp)]:.4f})."
            ),
        }

        return ToolResult(data=data)

    # ── Transition pathway ──────────────────────────────────

    def _transition_pathway(self, args: MSMInput) -> ToolResult:
        """Find dominant transition pathway between macrostates A → B."""
        if not args.transition_matrix and not args.trajectory:
            return ToolResult(data=None, success=False, error="transition_matrix or trajectory required")

        if args.transition_matrix:
            T = np.array(args.transition_matrix)
        else:
            traj = np.array(args.trajectory)
            assignments, _ = self._kmeans(traj, args.n_microstates)
            T = self._estimate_transition_matrix(assignments, args.lag_time, args.reversible)

        T = T / T.sum(axis=1, keepdims=True)
        n = T.shape[0]

        # Simple pathway: highest-probability path from state 0 to state n-1
        # Using a greedy approach: follow max transition prob (excluding self-loops)
        path = [0]
        current = 0
        visited = {0}
        while current != n - 1 and len(path) < n:
            # Mask visited states and self-transition
            probs = T[current].copy()
            probs[list(visited)] = 0.0  # avoid cycles
            probs[current] = 0.0
            if probs.max() < 1e-10:
                break
            next_state = int(np.argmax(probs))
            path.append(next_state)
            visited.add(next_state)
            current = next_state

        # Compute path probability
        path_prob = 1.0
        for i in range(len(path) - 1):
            path_prob *= T[path[i], path[i + 1]]

        # Branching ratio: how much flux goes through this path vs alternatives
        fluxes = []
        for i in range(len(path) - 1):
            total_flux = T[path[i]].sum() - T[path[i], path[i]]  # exclude self
            if total_flux > 0:
                fluxes.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "probability": round(float(T[path[i], path[i + 1]]), 6),
                    "branching_ratio": round(float(T[path[i], path[i + 1]] / total_flux), 6),
                })

        data = {
            "action": "transition_pathway",
            "pathway": path,
            "path_probability": round(float(path_prob), 8),
            "path_length": len(path),
            "fluxes": fluxes,
            "n_states": n,
            "message": (
                f"Dominant pathway: {' → '.join(str(s) for s in path)} "
                f"(P = {path_prob:.6e}, {len(path)} steps)."
            ),
        }

        return ToolResult(data=data)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _kmeans(data: np.ndarray, k: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering (no sklearn dependency)."""
        n, d = data.shape
        # Initialize with k-means++ style: first random, then farthest points
        rng = np.random.default_rng(42)
        centroids = np.zeros((k, d))
        centroids[0] = data[rng.integers(n)]
        for i in range(1, k):
            dists = np.min(np.sum((data[:, None] - centroids[:i]) ** 2, axis=2), axis=1)
            centroids[i] = data[np.argmax(dists)]

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            new_labels = np.argmin(np.sum((data[:, None] - centroids) ** 2, axis=2), axis=1)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            for j in range(k):
                if np.any(labels == j):
                    centroids[j] = data[labels == j].mean(axis=0)

        return labels, centroids

    @staticmethod
    def _estimate_transition_matrix(assignments: np.ndarray, lag: int,
                                     reversible: bool) -> np.ndarray:
        """Estimate row-stochastic transition matrix at given lag time."""
        n_states = int(assignments.max()) + 1
        counts = np.zeros((n_states, n_states))
        for i in range(len(assignments) - lag):
            counts[assignments[i], assignments[i + lag]] += 1

        # Add pseudocount for zero rows
        counts += 1e-10

        if reversible:
            # Symmetrize: C_sym = (C + C^T) / 2  (maximum likelihood reversible)
            counts = (counts + counts.T) / 2.0

        # Row-normalize
        row_sums = counts.sum(axis=1, keepdims=True)
        T = counts / row_sums
        return T

    @staticmethod
    def _eigendecompose(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute eigenvalues and right eigenvectors of transition matrix.
        Sorts by descending real part."""
        eigenvalues, eigenvectors = np.linalg.eig(T)
        # Sort by descending real part
        idx = np.argsort(-np.real(eigenvalues))
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Take real part (small imaginary parts from numerical noise)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        return eigenvalues, eigenvectors

    @staticmethod
    def _compute_timescales(eigenvalues: np.ndarray, lag: int,
                             dt: float) -> list[float]:
        """Implied timescales: t_k = -τ·dt / ln(|λ_k|) for λ_k < 1."""
        timescales = []
        for lam in eigenvalues[1:]:  # skip λ_1 = 1
            if 0 < lam < 1:
                t = -lag * dt / math.log(lam)
                timescales.append(t)
            elif lam >= 1.0:
                timescales.append(float("inf"))
            else:
                timescales.append(0.0)
        return timescales

    @staticmethod
    def _suggest_macrostates(eigenvalues: np.ndarray) -> int:
        """Suggest number of macrostates based on spectral gap."""
        # Find largest gap in eigenvalues (excluding λ=1)
        evals = eigenvalues[1:]  # skip stationary
        if len(evals) < 2:
            return 2
        gaps = np.diff(evals)
        # spectral gap → n_macrostates = gap_index + 2
        return int(np.argmax(gaps)) + 2

    @staticmethod
    def _coarse_grain(T: np.ndarray, labels: np.ndarray,
                       n_macro: int) -> np.ndarray:
        """Coarse-grain transition matrix by grouping microstates."""
        n_micro = T.shape[0]
        T_cg = np.zeros((n_macro, n_macro))
        for i in range(n_micro):
            for j in range(n_micro):
                T_cg[labels[i], labels[j]] += T[i, j]
        # Row-normalize
        row_sums = T_cg.sum(axis=1, keepdims=True)
        T_cg = T_cg / np.maximum(row_sums, 1e-30)
        return T_cg

    @staticmethod
    def _compute_mfpt(T: np.ndarray, stat_dist: np.ndarray) -> np.ndarray:
        """Mean First Passage Time matrix.
        MFPT(i→j) = 1 / π_j * (Z_jj / Z_ij)  where Z is fundamental matrix.
        """
        n = T.shape[0]
        # Fundamental matrix Z = (I - T + W)^-1  where W = 1 π^T
        W = np.outer(np.ones(n), stat_dist)
        Z = np.linalg.inv(np.eye(n) - T + W)
        mfpt = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    mfpt[i, j] = 1.0 / max(stat_dist[j], 1e-30)
                else:
                    mfpt[i, j] = (Z[j, j] - Z[i, j]) / max(stat_dist[j], 1e-30)
        return mfpt
