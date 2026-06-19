"""Active-learning / experiment-design tool for materials synthesis.

Reads tabular experimental data (CSV/Excel), fits a Gaussian-process surrogate,
and recommends the next batch of experiments to maximize or minimize a target
property. Designed to close the compute--experiment loop.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.tools.gp_tool import NumPyGP
from huginn.types import ToolContext, ToolResult


class ActiveLearningInput(BaseModel):
    action: Literal["load_csv", "recommend", "simulate_loop"] = Field(
        default="load_csv"
    )
    data_path: str | None = Field(
        default=None, description="Path to CSV/Excel with training data"
    )
    target_column: str = Field(
        default="target", description="Column name of the property to optimize"
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description=("Input parameter columns; inferred from CSV header if omitted"),
    )
    candidate_path: str | None = Field(
        default=None,
        description="CSV with candidate experiments to score",
    )
    bounds: dict[str, tuple[float, float]] | None = Field(
        default=None,
        description="Parameter bounds used to generate candidates when candidate_path is absent",
    )
    n_recommendations: int = Field(default=3, ge=1, le=100)
    n_candidates: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of candidate points to generate from bounds",
    )
    maximize: bool = Field(default=False)
    length_scale: float = Field(default=1.0, gt=0)
    sigma_f: float = Field(default=1.0, gt=0)
    sigma_n: float = Field(default=1e-5, ge=0)
    n_iterations: int = Field(default=5, ge=1)
    objective_tool: str | None = Field(
        default=None,
        description="Registered tool name used to evaluate proposals in simulation mode",
    )
    objective_path: str = Field(default="data.value")
    tool_input_template: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = Field(default=None)


class ActiveLearningTool(HuginnTool):
    """Recommend next experiments using GP-based active learning."""

    name = "active_learning_tool"
    description = (
        "Load experimental data, fit a GP surrogate, and recommend the next "
        "synthesis conditions to optimize a target property."
    )
    input_schema = ActiveLearningInput
    read_only = True

    def is_read_only(self, args: ActiveLearningInput) -> bool:
        return True

    async def call(self, args: ActiveLearningInput, context: ToolContext) -> ToolResult:
        try:
            if args.action == "load_csv":
                return self._load_csv(args)
            if args.action == "recommend":
                return self._recommend(args, context)
            if args.action == "simulate_loop":
                return await self._simulate_loop(args, context)
            return ToolResult(
                data=None, success=False, error=f"Unknown action: {args.action}"
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=str(exc))

    def _load_csv(self, args: ActiveLearningInput) -> ToolResult:
        if not args.data_path:
            return ToolResult(
                data=None, success=False, error="data_path is required for load_csv"
            )
        data = self._read_table(args.data_path)
        features = self._resolve_features(
            data, args.feature_columns, args.target_column
        )
        summary = {
            "rows": len(data),
            "columns": list(data[0].keys()) if data else [],
            "feature_columns": features,
            "target_column": args.target_column,
        }
        return ToolResult(data=summary)

    def _recommend(
        self,
        args: ActiveLearningInput,
        context: ToolContext,
        data: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        if data is None:
            if not args.data_path:
                return ToolResult(
                    data=None,
                    success=False,
                    error="data_path is required for recommend",
                )
            data = self._read_table(args.data_path)
        features = self._resolve_features(
            data, args.feature_columns, args.target_column
        )

        X_train, y_train = self._extract_xy(data, features, args.target_column)
        if len(X_train) < 2:
            return ToolResult(
                data=None,
                success=False,
                error="Need at least 2 training points to fit a GP",
            )

        candidates = self._load_or_generate_candidates(args, features)
        if not candidates:
            return ToolResult(
                data=None,
                success=False,
                error="No candidate experiments available",
            )

        X_candidates = np.array([[float(c[f]) for f in features] for c in candidates])

        gp = NumPyGP(
            length_scale=args.length_scale,
            sigma_f=args.sigma_f,
            sigma_n=args.sigma_n,
        )
        gp.fit(X_train, y_train)
        mu, sigma = gp.predict(X_candidates)

        ei = self._expected_improvement(mu, sigma, y_train, maximize=args.maximize)
        ranked = sorted(
            zip(ei, mu, sigma, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        recommendations = []
        for ei_val, mean, std, cand in ranked[: args.n_recommendations]:
            row = dict(cand)
            row["predicted_mean"] = float(mean)
            row["predicted_std"] = float(std)
            row["expected_improvement"] = float(ei_val)
            recommendations.append(row)

        return ToolResult(
            data={
                "feature_columns": features,
                "target_column": args.target_column,
                "training_rows": len(X_train),
                "recommendations": recommendations,
            }
        )

    async def _simulate_loop(
        self, args: ActiveLearningInput, context: ToolContext
    ) -> ToolResult:
        """Iteratively recommend and evaluate via a registered tool."""
        from huginn.tools.registry import ToolRegistry

        if not args.objective_tool:
            return ToolResult(
                data=None,
                success=False,
                error="objective_tool is required for simulate_loop",
            )

        data = self._read_table(args.data_path) if args.data_path else []
        features = self._resolve_features(data, args.feature_columns)

        history: list[dict[str, Any]] = []
        for iteration in range(args.n_iterations):
            if len(data) >= 2:
                rec_result = self._recommend(args, context, data=data)
                if not rec_result.success:
                    return rec_result
                recs = rec_result.data.get("recommendations", [])
            else:
                recs = self._load_or_generate_candidates(args, features)[:1]

            if not recs:
                break

            chosen = recs[0]
            template = json.dumps(args.tool_input_template)
            for key, val in chosen.items():
                template = template.replace(f"{{{key}}}", str(val))
            tool_input = json.loads(template)

            objective_tool = ToolRegistry.get(args.objective_tool)
            if objective_tool is None:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Objective tool '{args.objective_tool}' not found",
                )

            if hasattr(objective_tool, "call"):
                if __import__("inspect").iscoroutinefunction(objective_tool.call):
                    out = await objective_tool.call(tool_input, context)
                else:
                    out = objective_tool.call(tool_input, context)
            else:
                return ToolResult(
                    data=None,
                    success=False,
                    error="Objective tool has no callable 'call' method",
                )

            if not out.success:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Objective evaluation failed at iteration {iteration}: {out.error}",
                )

            value = self._get_path(out.data, args.objective_path)
            row = {f: chosen[f] for f in features}
            row[args.target_column] = float(value)
            data.append(row)
            history.append(
                {
                    "iteration": iteration + 1,
                    "parameters": {f: chosen[f] for f in features},
                    "objective": float(value),
                }
            )

        return ToolResult(data={"history": history, "final_dataset": data})

    def _read_table(self, path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        suffix = p.suffix.lower()
        if suffix == ".csv":
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]
        if suffix in {".xlsx", ".xls"}:
            try:
                import pandas as pd
            except ImportError as exc:
                raise RuntimeError(
                    "Excel support requires pandas. Install: pip install pandas"
                ) from exc
            df = pd.read_excel(p)
            return df.to_dict(orient="records")
        raise ValueError(f"Unsupported file format: {suffix}")

    def _resolve_features(
        self,
        data: list[dict[str, Any]],
        feature_columns: list[str] | None,
        target_column: str,
    ) -> list[str]:
        if feature_columns:
            return feature_columns
        if not data:
            return []
        return [k for k in data[0] if k != target_column]

    def _extract_xy(
        self,
        data: list[dict[str, Any]],
        features: list[str],
        target_column: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = []
        y = []
        for row in data:
            try:
                x = [float(row[f]) for f in features]
                yv = float(row[target_column])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid row {row}: {exc}") from exc
            X.append(x)
            y.append(yv)
        return np.array(X), np.array(y)

    def _load_or_generate_candidates(
        self, args: ActiveLearningInput, features: list[str]
    ) -> list[dict[str, Any]]:
        if args.candidate_path:
            return self._read_table(args.candidate_path)
        if not args.bounds:
            return []

        rng = np.random.default_rng(args.seed)
        n = args.n_candidates
        candidates = []
        for _ in range(n):
            row: dict[str, Any] = {}
            for f in features:
                low, high = args.bounds.get(f, (0.0, 1.0))
                row[f] = float(rng.uniform(low, high))
            candidates.append(row)
        return candidates

    def _expected_improvement(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_train: np.ndarray,
        maximize: bool = False,
    ) -> np.ndarray:
        if maximize:
            best = float(np.max(y_train))
            z = (mu - best) / (sigma + 1e-9)
        else:
            best = float(np.min(y_train))
            z = (best - mu) / (sigma + 1e-9)
        from scipy.stats import norm

        return sigma * (norm.pdf(z) + z * norm.cdf(z))

    def _get_path(self, data: Any, path: str) -> Any:
        value = data
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                break
        return value
