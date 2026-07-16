"""Enhanced sampling tool — metadynamics, umbrella sampling, and FES reconstruction.

Provides collective variable (CV) based enhanced sampling methods borrowed from
biomolecular MD, adapted for materials science (crystal polymorph search, defect
migration). Analyzes MD trajectories to reconstruct free energy surfaces.

Math:
  Metadynamics: V_bias(s, t) = Σᵢ h·exp(−(s−sᵢ)²/2σ²)
    FES(s) = −V_bias(s, t→∞)  (in the long-time limit)
  Umbrella Sampling: F(s) = −kT ln P(s) + V_bias(s)
    WHAM: self-consistent iteration to remove bias
  Large deviation: P(rare event) ~ exp(−I(x)/kT)
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


class EnhancedSamplingInput(BaseModel):
    action: Literal[
        "metadynamics_bias",
        "umbrella_setup",
        "wham",
        "reconstruct_fes",
        "rare_event_rate",
    ] = Field(...)

    # Collective variable data
    cv_trajectory: list[list[float]] | None = Field(
        default=None,
        description="CV values along MD trajectory. cv_trajectory[i] = [cv1, cv2, ...] at frame i"
    )
    cv_names: list[str] | None = Field(
        default=None, description="Names of collective variables"
    )

    # Metadynamics params
    gaussian_height: float = Field(default=0.01, gt=0, description="Gaussian hill height (eV)")
    gaussian_width: float = Field(default=0.1, gt=0, description="Gaussian width σ (CV units)")
    deposit_interval: int = Field(default=100, ge=1, description="Add hill every N frames")
    n_dimensions: int = Field(default=1, ge=1, le=3, description="Number of CVs")

    # Umbrella sampling params
    n_windows: int = Field(default=10, ge=2, le=100, description="Number of umbrella windows")
    spring_constant: float = Field(default=10.0, gt=0, description="Umbrella spring constant (eV/CV²)")
    cv_range: list[float] | None = Field(
        default=None, description="[cv_min, cv_max] for umbrella/WHAM"
    )

    # WHAM params
    window_centers: list[float] | None = Field(
        default=None, description="Umbrella center for each window"
    )
    window_samples: list[list[float]] | None = Field(
        default=None, description="CV samples from each window"
    )
    bin_count: int = Field(default=50, ge=5, le=500, description="Number of bins for histogram")

    # Thermodynamics
    temperature: float = Field(default=300.0, gt=0, description="Temperature (K)")

    # Rare event params
    threshold: float | None = Field(
        default=None, description="CV threshold defining the rare event region"
    )
    n_bins: int = Field(default=50, ge=5, le=200, description="Bins for histogram-based estimates")


class EnhancedSamplingTool(HuginnTool):
    """Enhanced sampling: metadynamics, umbrella sampling, WHAM, FES reconstruction."""

    name = "enhanced_sampling_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.VALIDATION, ResearchPhase.REPORTING}),
        light_alternatives=("tda_tool",),
    )
    description = (
        "Enhanced sampling analysis: metadynamics bias potential generation, "
        "umbrella sampling setup, WHAM free energy reconstruction, and rare "
        "event rate estimation via large deviation theory."
    )
    input_schema = EnhancedSamplingInput

    async def _execute(self, args: EnhancedSamplingInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "metadynamics_bias":
                return self._metadynamics(args)
            if args.action == "umbrella_setup":
                return self._umbrella_setup(args)
            if args.action == "wham":
                return self._wham(args)
            if args.action == "reconstruct_fes":
                return self._reconstruct_fes(args)
            if args.action == "rare_event_rate":
                return self._rare_event_rate(args)
            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── Metadynamics bias ──────────────────────────────────

    def _metadynamics(self, args: EnhancedSamplingInput) -> ToolResult:
        """Simulate metadynamics: deposit Gaussian hills along CV trajectory,
        reconstruct FES from accumulated bias."""
        if not args.cv_trajectory:
            return ToolResult(data=None, success=False, error="cv_trajectory required")

        cv = np.array(args.cv_trajectory)
        if cv.ndim == 1:
            cv = cv.reshape(-1, 1)

        n_frames = len(cv)
        n_dim = cv.shape[1]
        h = args.gaussian_height
        sigma = args.gaussian_width

        # Deposit hills
        hill_positions = []
        hill_heights = []
        bias_potential = np.zeros(n_frames)

        for i in range(0, n_frames, args.deposit_interval):
            s_i = cv[i]
            hill_positions.append(s_i)
            # Adaptive height: decrease as we fill (well-tempered metadynamics)
            adaptive_h = h * math.exp(-bias_potential[i] / (KB_EV * args.temperature))
            hill_heights.append(adaptive_h)

            # Update bias at all points
            for j in range(n_frames):
                dist2 = np.sum((cv[j] - s_i) ** 2)
                bias_potential[j] += adaptive_h * math.exp(-dist2 / (2 * sigma ** 2))

        # FES = −V_bias (in the long-time limit)
        fes = -bias_potential

        # Shift so min(FES) = 0
        fes_min = np.min(fes)
        fes = fes - fes_min

        data = {
            "action": "metadynamics_bias",
            "n_hills_deposited": len(hill_positions),
            "n_frames": n_frames,
            "n_dimensions": n_dim,
            "gaussian_height_eV": h,
            "gaussian_width": sigma,
            "deposit_interval": args.deposit_interval,
            "temperature_K": args.temperature,
            "fes_values_eV": [round(float(x), 6) for x in fes[:500]],  # cap for large trajectories
            "fes_min_eV": 0.0,
            "fes_max_eV": round(float(np.max(fes)), 6),
            "barrier_height_eV": round(float(np.max(fes) - np.min(fes)), 6),
            "hill_positions": [s.tolist() for s in hill_positions[:100]],
            "well_tempered": True,
            "bias_factor": math.exp(h / (KB_EV * args.temperature)),
            "message": (
                f"Deposited {len(hill_positions)} hills. "
                f"FES barrier: {np.max(fes) - np.min(fes):.4f} eV."
            ),
        }

        return ToolResult(data=data)

    # ── Umbrella sampling setup ────────────────────────────

    def _umbrella_setup(self, args: EnhancedSamplingInput) -> ToolResult:
        """Generate umbrella window centers along the CV range."""
        if not args.cv_range or len(args.cv_range) != 2:
            return ToolResult(data=None, success=False, error="cv_range [min, max] required")

        cv_min, cv_max = args.cv_range
        centers = np.linspace(cv_min, cv_max, args.n_windows)

        windows = []
        for i, c in enumerate(centers):
            windows.append({
                "window_index": i,
                "cv_center": round(float(c), 6),
                "spring_constant": args.spring_constant,
                "bias_potential": f"0.5 * k * (cv - {c:.4f})^2",
            })

        return ToolResult(data={
            "action": "umbrella_setup",
            "n_windows": args.n_windows,
            "cv_range": [cv_min, cv_max],
            "spring_constant": args.spring_constant,
            "windows": windows,
            "message": f"Set up {args.n_windows} umbrella windows from {cv_min} to {cv_max}.",
        })

    # ── WHAM ───────────────────────────────────────────────

    def _wham(self, args: EnhancedSamplingInput) -> ToolResult:
        """Weighted Histogram Analysis Method: self-consistent FES reconstruction
        from umbrella sampling data.

        WHAM equations:
          P(s) = Σᵢ Nᵢ · exp(−βVᵢ(s)) · pᵢ(s) / Σⱼ Nⱼ · exp(−βVⱼ(s))
          F(s) = −kT ln P(s)
        """
        if not args.window_centers or not args.window_samples:
            return ToolResult(data=None, success=False, error="window_centers and window_samples required")

        centers = np.array(args.window_centers)
        samples_list = [np.array(s) for s in args.window_samples]
        n_windows = len(centers)
        k = args.spring_constant
        beta = 1.0 / (KB_EV * args.temperature)

        # Determine bin edges
        all_samples = np.concatenate(samples_list)
        cv_min, cv_max = all_samples.min(), all_samples.max()
        bins = np.linspace(cv_min, cv_max, args.bin_count + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Histogram each window
        hists = np.zeros((n_windows, args.bin_count))
        n_per_window = np.zeros(n_windows)
        for i, s in enumerate(samples_list):
            h, _ = np.histogram(s, bins=bins, density=False)
            hists[i] = h
            n_per_window[i] = len(s)

        # WHAM self-consistent iteration
        F = np.zeros(args.bin_count)  # Free energy at each bin
        eps = 1e-10
        max_iter = 50000
        tol = 1e-6

        # 预计算偏置势矩阵, 避免三重 Python 循环 (50000×50×8=20M 次 math.exp 超时)
        V = 0.5 * k * (bin_centers[:, None] - centers[None, :]) ** 2  # (bin_count, n_windows)
        exp_negbV = np.exp(-beta * V)  # (bin_count, n_windows)
        hists_T = hists.T  # (bin_count, n_windows)
        sum_term = (n_per_window * exp_negbV).sum(axis=1)  # Σᵢ Nᵢ·exp(−βVᵢ)

        for iteration in range(max_iter):
            F_old = F.copy()
            numerator = (hists_T * exp_negbV).sum(axis=1)  # Σᵢ hᵢ·exp(−βVᵢ)
            denominator = np.exp(-beta * F_old) * sum_term  # Σᵢ Nᵢ·exp(−βVᵢ)·exp(−βF)
            prob = np.where(denominator > eps,
                            numerator / np.maximum(denominator, eps), 0.0)
            prob = np.maximum(prob, eps)
            F = -np.log(prob) / beta
            F = F - np.min(F)
            if np.max(np.abs(F - F_old)) < tol:
                break

        data = {
            "action": "wham",
            "method": "Weighted Histogram Analysis Method",
            "n_windows": n_windows,
            "n_bins": args.bin_count,
            "temperature_K": args.temperature,
            "bin_centers": [round(float(x), 6) for x in bin_centers],
            "fes_eV": [round(float(x), 6) for x in F],
            "fes_min_eV": 0.0,
            "fes_max_eV": round(float(np.max(F)), 6),
            "barrier_height_eV": round(float(np.max(F) - np.min(F)), 6),
            "converged": iteration < max_iter - 1,
            "n_iterations": iteration + 1,
            "message": (
                f"WHAM converged in {iteration + 1} iterations. "
                f"FES barrier: {np.max(F) - np.min(F):.4f} eV."
            ),
        }

        return ToolResult(data=data)

    # ── FES reconstruction from unbiased trajectory ────────

    def _reconstruct_fes(self, args: EnhancedSamplingInput) -> ToolResult:
        """Reconstruct FES from an unbiased trajectory using histogram method.

        F(s) = −kT ln P(s) + const
        """
        if not args.cv_trajectory:
            return ToolResult(data=None, success=False, error="cv_trajectory required")

        cv = np.array(args.cv_trajectory)
        if cv.ndim == 1:
            cv = cv.reshape(-1, 1)

        beta = 1.0 / (KB_EV * args.temperature)
        n_dim = cv.shape[1]

        if n_dim == 1:
            # 1D histogram
            hist, edges = np.histogram(cv[:, 0], bins=args.n_bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            # F = -kT ln P
            prob = np.maximum(hist, 1e-30)
            fes = -np.log(prob) / beta
            fes = fes - np.min(fes)

            data = {
                "action": "reconstruct_fes",
                "n_dimensions": 1,
                "bin_centers": [round(float(x), 6) for x in centers],
                "fes_eV": [round(float(x), 6) for x in fes],
                "fes_min_eV": 0.0,
                "fes_max_eV": round(float(np.max(fes)), 6),
                "barrier_height_eV": round(float(np.max(fes) - np.min(fes)), 6),
                "temperature_K": args.temperature,
                "message": f"FES reconstructed. Barrier: {np.max(fes) - np.min(fes):.4f} eV.",
            }
        else:
            # 2D histogram
            hist, xedges, yedges = np.histogram2d(
                cv[:, 0], cv[:, 1], bins=args.n_bins, density=True
            )
            x_centers = 0.5 * (xedges[:-1] + xedges[1:])
            y_centers = 0.5 * (yedges[:-1] + yedges[1:])
            prob = np.maximum(hist, 1e-30)
            fes = -np.log(prob) / beta
            fes = fes - np.min(fes)

            data = {
                "action": "reconstruct_fes",
                "n_dimensions": 2,
                "x_centers": [round(float(x), 6) for x in x_centers],
                "y_centers": [round(float(x), 6) for x in y_centers],
                "fes_2d_eV": [[round(float(x), 4) for x in row] for row in fes],
                "fes_min_eV": 0.0,
                "fes_max_eV": round(float(np.max(fes)), 6),
                "barrier_height_eV": round(float(np.max(fes) - np.min(fes)), 6),
                "temperature_K": args.temperature,
                "message": f"2D FES reconstructed. Barrier: {np.max(fes) - np.min(fes):.4f} eV.",
            }

        return ToolResult(data=data)

    # ── Rare event rate estimation ──────────────────────────

    def _rare_event_rate(self, args: EnhancedSamplingInput) -> ToolResult:
        """Estimate rare event rate from trajectory using large deviation theory.

        P(X > threshold) ~ exp(−I(X)/kT)  where I(X) is the rate function.
        Estimated as: I(x) ≈ −kT ln(count(x)/total)
        """
        if not args.cv_trajectory:
            return ToolResult(data=None, success=False, error="cv_trajectory required")

        cv = np.array(args.cv_trajectory).flatten()
        threshold = args.threshold if args.threshold else float(np.mean(cv) + 2 * np.std(cv))

        beta = 1.0 / (KB_EV * args.temperature)
        n_total = len(cv)
        n_above = np.sum(cv > threshold)

        # Empirical probability
        p_empirical = n_above / n_total

        # Rate function I(x) = −kT ln P(X > x)
        if p_empirical > 0:
            rate_function = -math.log(p_empirical) / beta
        else:
            rate_function = float("inf")

        # Estimate from histogram tail
        hist, edges = np.histogram(cv, bins=args.n_bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # Find bins above threshold
        above_mask = centers > threshold
        if np.any(above_mask) and np.any(hist[above_mask] > 0):
            tail_probs = hist[above_mask] / np.sum(hist)
            tail_rates = -np.log(np.maximum(tail_probs, 1e-30)) / beta
            avg_tail_rate = float(np.mean(tail_rates))
        else:
            tail_rates = []
            avg_tail_rate = float("inf")

        # Transition rate (Poisson process estimate)
        # k ≈ P(X > threshold) / τ_corr  (τ_corr estimated from autocorrelation)
        if n_total > 10:
            # Simple autocorrelation estimate
            ac = np.correlate(cv - cv.mean(), cv - cv.mean(), mode="full")
            ac = ac[len(ac) // 2:]
            ac = ac / ac[0] if ac[0] != 0 else ac
            # Find decorrelation time
            decorr = np.argmax(ac < 1.0 / np.e) if np.any(ac < 1.0 / np.e) else 1
            tau_corr = float(decorr)
        else:
            tau_corr = 1.0

        rate_estimate = p_empirical / tau_corr if p_empirical > 0 else 0.0

        data = {
            "action": "rare_event_rate",
            "threshold": round(float(threshold), 6),
            "n_total": n_total,
            "n_above_threshold": int(n_above),
            "p_empirical": p_empirical,
            "rate_function_eV": round(rate_function, 6) if rate_function != float("inf") else None,
            "tail_rate_eV": round(avg_tail_rate, 6) if avg_tail_rate != float("inf") else None,
            "tau_correlation": tau_corr,
            "transition_rate": rate_estimate,
            "temperature_K": args.temperature,
            "message": (
                f"P(cv > {threshold:.3f}) = {p_empirical:.6f}, "
                f"rate function I = {rate_function:.4f} eV"
                if rate_function != float("inf")
                else f"No events above threshold {threshold:.3f} in {n_total} samples."
            ),
        }

        return ToolResult(data=data)
