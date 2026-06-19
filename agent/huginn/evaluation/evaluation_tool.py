"""Evaluation Tool — multi-criteria decision analysis for material selection."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.evaluation.core import evaluate, sensitivity_random_weights
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class EvaluationToolInput(BaseModel):
    action: Literal["evaluate", "sensitivity", "list_methods"] = Field(...)

    alternatives: list[str] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=list)
    matrix: list[list[float]] = Field(default_factory=list)
    directions: list[Literal["max", "min"]] = Field(default_factory=list)

    weight_method: Literal["entropy", "cv", "critic", "ahp", "pca", "equal"] = Field(
        default="entropy"
    )
    eval_method: Literal["topsis", "vikor", "todim", "promethee", "rsr", "grey"] = (
        Field(default="topsis")
    )

    # AHP pairwise matrix (if weight_method="ahp")
    ahp_matrix: list[list[float]] | None = Field(default=None)

    # Extra params
    v: float = Field(default=0.5, description="VIKOR compromise coefficient")
    theta: float = Field(default=2.5, description="TODIM attenuation factor")
    rho: float = Field(default=0.5, description="Grey relational coefficient")

    # Sensitivity
    n_trials: int = Field(default=1000)
    perturbation: float = Field(default=0.3)


class EvaluationTool(HuginnTool):
    """Multi-criteria decision analysis for material screening and selection."""

    name = "evaluation_tool"
    description = "Evaluate and rank material candidates using MCDA: TOPSIS, VIKOR, TODIM, PROMETHEE, RSR, Grey + Entropy/CV/CRITIC/AHP/PCA weights"
    input_schema = EvaluationToolInput

    def is_read_only(self, args: EvaluationToolInput) -> bool:
        return True

    async def call(self, args: EvaluationToolInput, context: ToolContext) -> ToolResult:
        if args.action == "list_methods":
            return ToolResult(
                data={
                    "weight_methods": [
                        "entropy",
                        "cv",
                        "critic",
                        "ahp",
                        "pca",
                        "equal",
                    ],
                    "eval_methods": [
                        "topsis",
                        "vikor",
                        "todim",
                        "promethee",
                        "rsr",
                        "grey",
                    ],
                    "combinations": [
                        f"{w}-{e}"
                        for w in ["entropy", "cv", "critic"]
                        for e in [
                            "topsis",
                            "vikor",
                            "todim",
                            "promethee",
                            "rsr",
                            "grey",
                        ]
                    ]
                    + ["ahp-topsis", "pca-topsis"],
                },
                success=True,
            )

        if args.action == "evaluate":
            return self._evaluate(args)

        if args.action == "sensitivity":
            return self._sensitivity(args)

        return ToolResult(
            data=None, success=False, error=f"Unknown action: {args.action}"
        )

    def _evaluate(self, args: EvaluationToolInput) -> ToolResult:
        if not args.alternatives or not args.criteria or not args.matrix:
            return ToolResult(
                data=None,
                success=False,
                error="alternatives, criteria, and matrix are required",
            )

        try:
            matrix = np.array(args.matrix, dtype=float)
            ahp = np.array(args.ahp_matrix, dtype=float) if args.ahp_matrix else None

            eval_kwargs = {}
            if args.eval_method == "vikor":
                eval_kwargs["v"] = args.v
            elif args.eval_method == "todim":
                eval_kwargs["theta"] = args.theta
            elif args.eval_method == "grey":
                eval_kwargs["rho"] = args.rho

            result = evaluate(
                alternatives=args.alternatives,
                criteria=args.criteria,
                matrix=matrix,
                directions=args.directions or None,
                weight_method=args.weight_method,
                eval_method=args.eval_method,
                ahp_matrix=ahp,
                eval_kwargs=eval_kwargs,
            )

            return ToolResult(
                data={
                    "method": result.method,
                    "weights": result.weights,
                    "scores": result.scores,
                    "ranking": result.ranking,
                },
                success=True,
            )

        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Evaluation failed: {e}")

    def _sensitivity(self, args: EvaluationToolInput) -> ToolResult:
        if not args.alternatives or not args.criteria or not args.matrix:
            return ToolResult(
                data=None,
                success=False,
                error="alternatives, criteria, and matrix are required",
            )

        try:
            matrix = np.array(args.matrix, dtype=float)
            eval_kwargs = {}
            if args.eval_method == "vikor":
                eval_kwargs["v"] = args.v
            elif args.eval_method == "todim":
                eval_kwargs["theta"] = args.theta

            result = sensitivity_random_weights(
                alternatives=args.alternatives,
                criteria=args.criteria,
                matrix=matrix,
                directions=args.directions or None,
                eval_method=args.eval_method,
                n_trials=args.n_trials,
                perturbation=args.perturbation,
                eval_kwargs=eval_kwargs,
            )

            return ToolResult(data=result, success=True)

        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Sensitivity analysis failed: {e}"
            )
