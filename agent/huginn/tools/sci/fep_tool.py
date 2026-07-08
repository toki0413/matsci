"""Alchemical free energy perturbation tool — λ-coupling for materials & drug design.

Implements thermodynamic integration (TI), free energy perturbation (FEP),
and the Jarzynski equality for computing free energy differences along an
alchemical coupling parameter λ. Works for both drug binding (morph ligand
atoms) and alloy design (transmute dopant elements).

Math:
  H(λ) = (1−λ)·H_A + λ·H_B
  TI:  ΔF = ∫₀¹ ⟨∂H/∂λ⟩_λ dλ
  FEP: ΔF = −k_BT · ln⟨exp(−βΔH)⟩
  BAR: ΔF = −k_BT · ln(⟨f(βΔH)⟩_A / ⟨f(−βΔH)⟩_B)  (Bennett acceptance ratio)
  Jarzynski: ΔF = −k_BT · ln⟨exp(−βW)⟩
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

# Boltzmann constant in eV/K (consistent with VASP/QE output)
KB_EV = 8.617333262e-5
# kcal/mol per eV (for drug-design interop)
EV_TO_KCAL = 23.0605


class FEPInput(BaseModel):
    action: Literal["ti", "fep", "bar", "jarzynski", "lambda_schedule"] = Field(
        ..., description="Free energy method: TI (thermodynamic integration), "
        "FEP (Zwanzig), BAR (Bennett), Jarzynski (non-equilibrium), "
        "lambda_schedule (generate λ windows)"
    )

    # Energy data: per-λ-window energy samples
    # For TI: list of (lambda, dU/dlambda samples)
    lambda_values: list[float] = Field(
        default_factory=list,
        description="λ window centers"
    )
    dU_dlambda: list[list[float]] | None = Field(
        default=None,
        description="TI: ∂U/∂λ samples per window. dU_dlambda[i] = samples at λ[i]"
    )

    # For FEP/BAR: energy differences
    delta_U: list[list[float]] | None = Field(
        default=None,
        description="FEP: ΔU = U_B − U_A samples evaluated at each λ window"
    )
    delta_U_reverse: list[list[float]] | None = Field(
        default=None,
        description="BAR reverse: ΔU = U_A − U_B samples from λ+1 window"
    )

    # For Jarzynski: non-equilibrium work values
    work_values: list[float] | None = Field(
        default=None,
        description="Jarzynski: non-equilibrium work W samples from fast switching"
    )

    # Thermodynamic params
    temperature: float = Field(default=298.15, gt=0, description="Temperature (K)")
    n_lambda: int = Field(default=11, ge=2, le=101, description="Number of λ windows for schedule")
    lambda_spacing: Literal["uniform", "nonlinear"] = Field(
        default="uniform",
        description="uniform: Δλ=1/n; nonlinear: denser near endpoints where ∂U/∂λ varies most"
    )

    # Error estimation
    n_bootstrap: int = Field(default=200, ge=0, le=1000, description="Bootstrap samples for error bar")

    # Domain context
    domain: Literal["materials", "drug_design"] = Field(
        default="materials",
        description="materials: output in eV; drug_design: output in kcal/mol"
    )


class FEPTool(HuginnTool):
    """Alchemical free energy computation via TI, FEP, BAR, and Jarzynski."""

    name = "fep_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.VALIDATION, ResearchPhase.REPORTING}),
        light_alternatives=("gp_tool",),
    )
    description = (
        "Alchemical free energy perturbation: thermodynamic integration (TI), "
        "Zwanzig FEP, Bennett acceptance ratio (BAR), and Jarzynski equality. "
        "Computes ΔF along coupling parameter λ for drug binding or alloy formation."
    )
    input_schema = FEPInput

    async def _execute(self, args: FEPInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "lambda_schedule":
                return self._lambda_schedule(args)
            if args.action == "ti":
                return self._thermodynamic_integration(args)
            if args.action == "fep":
                return self._fep_zwanzig(args)
            if args.action == "bar":
                return self._bar(args)
            if args.action == "jarzynski":
                return self._jarzynski(args)
            return ToolResult(data=None, success=False, error=f"Unknown action: {args.action}")
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    # ── λ schedule ──────────────────────────────────────────

    def _lambda_schedule(self, args: FEPInput) -> ToolResult:
        n = args.n_lambda
        if args.lambda_spacing == "uniform":
            lambdas = np.linspace(0, 1, n)
        else:
            # nonlinear: denser near endpoints — sin^(π/2) transformation
            t = np.linspace(0, 1, n)
            lambdas = np.sin(t * np.pi / 2) ** 2
            lambdas[0] = 0.0
            lambdas[-1] = 1.0

        return ToolResult(data={
            "action": "lambda_schedule",
            "lambdas": lambdas.tolist(),
            "n_windows": n,
            "spacing": args.lambda_spacing,
            "message": f"Generated {n} λ windows.",
        })

    # ── Thermodynamic Integration ───────────────────────────

    def _thermodynamic_integration(self, args: FEPInput) -> ToolResult:
        if not args.lambda_values or not args.dU_dlambda:
            return ToolResult(data=None, success=False, error="lambda_values and dU_dlambda required")

        lambdas = np.array(args.lambda_values)
        # Compute mean ⟨∂U/∂λ⟩ at each window
        means = np.array([np.mean(samples) for samples in args.dU_dlambda])
        stds = np.array([np.std(samples, ddof=1) if len(samples) > 1 else 0.0
                         for samples in args.dU_dlambda])

        # Trapezoidal integration: ΔF = ∫₀¹ ⟨∂U/∂λ⟩ dλ
        delta_F = np.trapezoid(means, lambdas)

        # Error estimate: propagate per-window variance through trapezoidal rule
        # Var(ΔF) ≈ Σᵢ wᵢ² Var(⟨∂U/∂λ⟩ᵢ)  where wᵢ are trapezoidal weights
        n_per_window = np.array([len(s) for s in args.dU_dlambda])
        sem_per_window = stds / np.sqrt(np.maximum(n_per_window, 1))
        # Trapezoidal weights
        weights = np.ones(len(lambdas))
        weights[0] = 0.5
        weights[-1] = 0.5
        dlam = np.diff(lambdas)
        # Approx: weight each point by average of adjacent intervals
        var_F = np.zeros(len(lambdas))
        for i in range(len(lambdas)):
            w = 0.0
            if i > 0:
                w += 0.5 * dlam[i - 1]
            if i < len(lambdas) - 1:
                w += 0.5 * dlam[i]
            var_F[i] = w ** 2 * sem_per_window[i] ** 2
        err_F = math.sqrt(np.sum(var_F))

        # Bootstrap for cross-check
        bs_estimate = None
        if args.n_bootstrap > 0:
            bs_samples = []
            for _ in range(args.n_bootstrap):
                bs_means = np.array([
                    np.mean(np.random.choice(s, size=len(s), replace=True))
                    for s in args.dU_dlambda
                ])
                bs_samples.append(np.trapezoid(bs_means, lambdas))
            bs_estimate = {
                "mean": float(np.mean(bs_samples)),
                "std": float(np.std(bs_samples, ddof=1)),
                "ci_lower": float(np.percentile(bs_samples, 2.5)),
                "ci_upper": float(np.percentile(bs_samples, 97.5)),
            }

        # Physics: check curvature for hysteresis sign
        # If ΔF is dominated by endpoint contributions, the transformation is steep
        curvature = np.diff(means, n=2) if len(means) > 2 else np.array([0.0])

        data = {
            "action": "ti",
            "method": "Thermodynamic Integration",
            "delta_F_eV": round(float(delta_F), 6),
            "error_eV": round(float(err_F), 6),
            "lambda_values": lambdas.tolist(),
            "dU_dlambda_means": means.tolist(),
            "dU_dlambda_stds": stds.tolist(),
            "n_windows": len(lambdas),
            "temperature_K": args.temperature,
            "domain": args.domain,
            "bootstrap": bs_estimate,
            "curvature_max": round(float(np.max(np.abs(curvature))), 6) if len(curvature) else 0.0,
        }

        # Convert units for drug design
        if args.domain == "drug_design":
            data["delta_F_kcal_mol"] = round(float(delta_F) * EV_TO_KCAL, 4)
            data["error_kcal_mol"] = round(float(err_F) * EV_TO_KCAL, 4)

        data["message"] = (
            f"ΔF = {data['delta_F_eV']:.4f} eV ± {data['error_eV']:.4f} eV"
            + (f" ({data['delta_F_kcal_mol']:.2f} kcal/mol)" if args.domain == "drug_design" else "")
        )

        return ToolResult(data=data)

    # ── Zwanzig FEP ─────────────────────────────────────────

    def _fep_zwanzig(self, args: FEPInput) -> ToolResult:
        if not args.delta_U:
            return ToolResult(data=None, success=False, error="delta_U required (per-window ΔU samples)")

        beta = 1.0 / (KB_EV * args.temperature)
        lambdas = np.array(args.lambda_values) if args.lambda_values else np.arange(len(args.delta_U))

        # Per-window ΔF: ΔF_i = −kT ln⟨exp(−βΔU)⟩
        window_dFs = []
        for du_samples in args.delta_U:
            du = np.array(du_samples)
            # Zwanzig: ΔF = −kT ln⟨exp(−βΔU)⟩
            arg = np.exp(-beta * du)
            # Guard against overflow/underflow
            if np.any(np.isinf(arg)) or np.any(np.isnan(arg)):
                # Use max-shift trick: ΔF = −kT [max(−βΔU) + ln⟨exp(−βΔU − max)⟩]
                shift = np.max(-beta * du)
                dF = -(1.0 / beta) * (shift + math.log(np.mean(np.exp(-beta * du - shift))))
            else:
                dF = -(1.0 / beta) * math.log(np.mean(arg))
            window_dFs.append(dF)

        total_dF = float(np.sum(window_dFs))

        # Bootstrap error
        bs = None
        if args.n_bootstrap > 0:
            bs_totals = []
            for _ in range(args.n_bootstrap):
                bs_dFs = []
                for du_samples in args.delta_U:
                    du = np.array(du_samples)
                    idx = np.random.choice(len(du), size=len(du), replace=True)
                    du_bs = du[idx]
                    arg = np.exp(-beta * du_bs)
                    if np.any(np.isinf(arg)):
                        shift = np.max(-beta * du_bs)
                        dF = -(1.0 / beta) * (shift + math.log(np.mean(np.exp(-beta * du_bs - shift))))
                    else:
                        dF = -(1.0 / beta) * math.log(np.mean(arg))
                    bs_dFs.append(dF)
                bs_totals.append(np.sum(bs_dFs))
            bs = {
                "mean": float(np.mean(bs_totals)),
                "std": float(np.std(bs_totals, ddof=1)),
                "ci_lower": float(np.percentile(bs_totals, 2.5)),
                "ci_upper": float(np.percentile(bs_totals, 97.5)),
            }

        data = {
            "action": "fep",
            "method": "Zwanzig Free Energy Perturbation",
            "delta_F_eV": round(total_dF, 6),
            "window_dFs_eV": [round(float(x), 6) for x in window_dFs],
            "n_windows": len(window_dFs),
            "temperature_K": args.temperature,
            "domain": args.domain,
            "bootstrap": bs,
        }
        if args.domain == "drug_design":
            data["delta_F_kcal_mol"] = round(total_dF * EV_TO_KCAL, 4)

        data["message"] = f"ΔF = {total_dF:.4f} eV (Zwanzig FEP)"

        return ToolResult(data=data)

    # ── Bennett Acceptance Ratio ───────────────────────────

    def _bar(self, args: FEPInput) -> ToolResult:
        if not args.delta_U or not args.delta_U_reverse:
            return ToolResult(data=None, success=False,
                              error="delta_U (forward) and delta_U_reverse required for BAR")

        beta = 1.0 / (KB_EV * args.temperature)
        # BAR per window pair: ΔF = −kT ln(⟨f(βΔU)⟩_A / ⟨f(−βΔU)⟩_B)
        # f(x) = 1/(1+exp(x)), the Fermi function
        window_dFs = []
        for i in range(len(args.delta_U)):
            fwd = np.array(args.delta_U[i])
            rev = np.array(args.delta_U_reverse[i])
            # BAR: ΔF = −kT ln(⟨1/(1+exp(βΔU))⟩_A / ⟨1/(1+exp(−βΔU))⟩_B)
            fermi_fwd = 1.0 / (1.0 + np.exp(beta * fwd))
            fermi_rev = 1.0 / (1.0 + np.exp(-beta * rev))
            # Guard against log(0)
            eps = 1e-30
            dF = -(1.0 / beta) * math.log(
                max(np.mean(fermi_fwd), eps) / max(np.mean(fermi_rev), eps)
            )
            window_dFs.append(dF)

        total_dF = float(np.sum(window_dFs))

        data = {
            "action": "bar",
            "method": "Bennett Acceptance Ratio",
            "delta_F_eV": round(total_dF, 6),
            "window_dFs_eV": [round(float(x), 6) for x in window_dFs],
            "n_windows": len(window_dFs),
            "temperature_K": args.temperature,
            "domain": args.domain,
            "message": f"ΔF = {total_dF:.4f} eV (BAR)",
        }
        if args.domain == "drug_design":
            data["delta_F_kcal_mol"] = round(total_dF * EV_TO_KCAL, 4)

        return ToolResult(data=data)

    # ── Jarzynski equality ──────────────────────────────────

    def _jarzynski(self, args: FEPInput) -> ToolResult:
        if not args.work_values:
            return ToolResult(data=None, success=False, error="work_values required")

        beta = 1.0 / (KB_EV * args.temperature)
        W = np.array(args.work_values)

        # Jarzynski: ΔF = −kT ln⟨exp(−βW)⟩
        arg = np.exp(-beta * W)
        if np.any(np.isinf(arg)) or np.any(np.isnan(arg)):
            # Use max-shift trick
            shift = np.max(-beta * W)
            delta_F = -(1.0 / beta) * (shift + math.log(np.mean(np.exp(-beta * W - shift))))
        else:
            delta_F = -(1.0 / beta) * math.log(np.mean(arg))

        # Bootstrap error
        bs = None
        if args.n_bootstrap > 0:
            bs_samples = []
            for _ in range(args.n_bootstrap):
                idx = np.random.choice(len(W), size=len(W), replace=True)
                W_bs = W[idx]
                arg_bs = np.exp(-beta * W_bs)
                if np.any(np.isinf(arg_bs)):
                    shift = np.max(-beta * W_bs)
                    dF = -(1.0 / beta) * (shift + math.log(np.mean(np.exp(-beta * W_bs - shift))))
                else:
                    dF = -(1.0 / beta) * math.log(np.mean(arg_bs))
                bs_samples.append(dF)
            bs = {
                "mean": float(np.mean(bs_samples)),
                "std": float(np.std(bs_samples, ddof=1)),
                "ci_lower": float(np.percentile(bs_samples, 2.5)),
                "ci_upper": float(np.percentile(bs_samples, 97.5)),
            }

        # Second-law check: ⟨W⟩ ≥ ΔF (Jarzynski inequality)
        mean_W = float(np.mean(W))
        viol_2nd_law = mean_W < delta_F

        data = {
            "action": "jarzynski",
            "method": "Jarzynski Equality",
            "delta_F_eV": round(float(delta_F), 6),
            "mean_work_eV": round(mean_W, 6),
            "dissipated_work_eV": round(mean_W - float(delta_F), 6),
            "n_samples": len(W),
            "temperature_K": args.temperature,
            "domain": args.domain,
            "second_law_satisfied": not viol_2nd_law,
            "bootstrap": bs,
            "message": f"ΔF = {delta_F:.4f} eV, ⟨W⟩ = {mean_W:.4f} eV (dissipated: {mean_W - delta_F:.4f})",
        }
        if args.domain == "drug_design":
            data["delta_F_kcal_mol"] = round(float(delta_F) * EV_TO_KCAL, 4)

        return ToolResult(data=data)
