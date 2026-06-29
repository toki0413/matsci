"""Symbolic Regression Tool — integrates PSE/PSRN for discovering
analytical physical laws from computational or experimental data.

Wraps the Parallel Symbolic Regression Network (PSRN) with a HuginnTool
interface, enabling the agent to discover constitutive relations,
property prediction formulas, and process-structure-property linkages.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from huginn.security import SafeEvalError, safe_math_eval
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.types import ToolContext, ToolResult


class SymbolicRegressionInput(BaseModel):
    action: str = Field(default="discover", description="discover | evaluate | compare")
    data_file: str | None = Field(
        default=None, description="Path to CSV file (last column = target)"
    )
    data_json: dict[str, list[float]] | None = Field(
        default=None, description="Inline data as {feature: [values], target: [values]}"
    )
    target_column: str = Field(default="", description="Name of target variable")
    feature_columns: list[str] | None = Field(
        default=None, description="Feature names (auto-detect if None)"
    )
    operators: list[str] = Field(
        default=["Add", "Mul", "Identity", "Sin", "Cos", "Exp", "Log"],
        description="Allowed operators for expression search",
    )
    time_limit: int = Field(
        default=300, ge=10, le=3600, description="Search time limit in seconds"
    )
    use_const: bool = Field(default=True, description="Fit numerical constants")
    n_symbol_layers: int = Field(
        default=3, ge=2, le=5, description="PSRN depth (2=speed, 5=expressiveness)"
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of candidate expressions to return"
    )
    probe_expression: str | None = Field(
        default=None, description="Known expression to verify recovery"
    )
    n_down_sample: int | None = Field(
        default=None, description="Subsample large datasets for speed"
    )


class SymbolicRegressionTool(HuginnTool):
    """Discover analytical expressions from data using PSE/PSRN.

    This tool enables the agent to perform scientific law discovery:
    - Constitutive relations (stress-strain, equation of state)
    - Property prediction formulas
    - Process-structure-property mappings
    - Calibration of phenomenological models
    """

    name = "symbolic_regression_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "Discover analytical mathematical expressions from data using "
        "Parallel Symbolic Regression (PSE/PSRN). Returns a Pareto frontier "
        "of expressions trading off accuracy vs. complexity."
    )
    input_schema = SymbolicRegressionInput

    def __init__(self, pse_path: str | None = None):
        super().__init__()
        self.pse_path = (
            Path(pse_path) if pse_path else Path.home() / "Desktop" / "符号回归" / "PSE"
        )
        self._regressor_class = None
        self._model_module = None

    def _ensure_pse_available(self) -> bool:
        """Check if PSE code is importable."""
        if self._regressor_class is not None:
            return True
        try:
            pse_model = self.pse_path / "model"
            if str(pse_model) not in sys.path:
                sys.path.insert(0, str(pse_model.parent))
            from model.regressor import PSRN_Regressor

            self._regressor_class = PSRN_Regressor
            return True
        except Exception:
            return False

    def _load_data(
        self, args: SymbolicRegressionInput
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load X, Y and feature names from file or inline JSON."""
        if args.data_json:
            data = args.data_json
            if args.target_column not in data:
                raise ValueError(
                    f"target_column '{args.target_column}' not found in data_json"
                )
            y = np.array(data[args.target_column], dtype=np.float64)
            feature_cols = args.feature_columns or [
                k for k in data if k != args.target_column
            ]
            x = np.column_stack(
                [np.array(data[c], dtype=np.float64) for c in feature_cols]
            )
            return x, y.reshape(-1, 1), feature_cols

        if not args.data_file:
            raise ValueError("Either data_file or data_json must be provided")

        path = Path(args.data_file)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        # Parse CSV
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError("CSV file is empty")

        fieldnames = reader.fieldnames or []

        # Determine target column
        target = args.target_column
        if not target:
            target = fieldnames[-1]

        if target not in fieldnames:
            raise ValueError(f"Target column '{target}' not found in {fieldnames}")

        # Determine feature columns
        if args.feature_columns:
            feature_cols = [c for c in args.feature_columns if c in fieldnames]
        else:
            feature_cols = [c for c in fieldnames if c != target]

        if not feature_cols:
            raise ValueError("No feature columns found")

        y = np.array([float(r[target]) for r in rows], dtype=np.float64)
        x = np.array(
            [[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64
        )
        return x, y.reshape(-1, 1), feature_cols

    async def call(
        self, args: SymbolicRegressionInput, context: ToolContext
    ) -> ToolResult:
        if args.action == "discover":
            return await self._discover(args)
        elif args.action == "evaluate":
            return await self._evaluate(args)
        elif args.action == "compare":
            return await self._compare(args)
        return ToolResult(
            data=None, success=False, error=f"Unknown action: {args.action}"
        )

    async def _discover(self, args: SymbolicRegressionInput) -> ToolResult:
        try:
            X, Y, features = self._load_data(args)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Data loading failed: {e}"
            )

        # Check PSE availability
        if not self._ensure_pse_available():
            # Fallback: return mock result with helpful message
            return ToolResult(
                data={
                    "pareto_front": [],
                    "best_expression": None,
                    "message": (
                        "PSE/PSRN not available. Install from the符号回归 directory "
                        "or ensure model/regressor.py is on PYTHONPATH."
                    ),
                    "features": features,
                    "samples": X.shape[0],
                },
                success=False,
                error="PSE/PSRN not importable",
            )

        try:
            # Build stage config dynamically
            stage_config = {
                "default": {
                    "operators": args.operators,
                    "time_limit": args.time_limit,
                    "n_psrn_inputs": min(5 + len(features), 10),
                    "n_sample_variables": len(features),
                },
                "stages": [
                    {
                        "time_limit": min(30, args.time_limit // 3),
                        "n_psrn_inputs": min(3 + len(features), 7),
                    },
                    {},
                ],
            }

            # Write temporary stage config
            import yaml

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(stage_config, f)
                stage_path = f.name

            try:
                # Configure PSRN
                regressor = self._regressor_class(
                    variables=features,
                    use_const=args.use_const,
                    n_symbol_layers=args.n_symbol_layers,
                    device="cuda" if self._cuda_available() else "cpu",
                    token_generator="GP",
                    stage_config=stage_path,
                )

                n_down = args.n_down_sample or min(X.shape[0], 1000)

                flag, pareto_ls = regressor.fit(
                    X,
                    Y,
                    n_down_sample=n_down,
                    probe=args.probe_expression,
                    top_k=args.top_k,
                    time_limit=args.time_limit,
                )
            finally:
                Path(stage_path).unlink(missing_ok=True)

            # Format results
            pareto_results = []
            best_expr = None
            best_reward = -float("inf")

            for item in pareto_ls[: args.top_k]:
                expr, reward, mse, complexity = item
                entry = {
                    "expression": str(expr),
                    "reward": float(reward),
                    "mse": float(mse),
                    "complexity": int(complexity),
                }
                pareto_results.append(entry)
                if reward > best_reward:
                    best_reward = reward
                    best_expr = entry

            return ToolResult(
                data={
                    "pareto_front": pareto_results,
                    "best_expression": best_expr,
                    "found_probe": bool(flag),
                    "features": features,
                    "samples": X.shape[0],
                    "target": args.target_column,
                    "search_time": args.time_limit,
                    "operators": args.operators,
                },
                success=True,
            )

        except Exception as e:
            traceback_str = traceback.format_exc()
            return ToolResult(
                data=None,
                success=False,
                error=f"Symbolic regression failed: {e}\n{traceback_str}",
            )

    async def _evaluate(self, args: SymbolicRegressionInput) -> ToolResult:
        """Evaluate a known expression on provided data."""
        try:
            X, Y, features = self._load_data(args)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Data loading failed: {e}"
            )

        if not args.probe_expression:
            return ToolResult(
                data=None, success=False, error="probe_expression required for evaluate"
            )

        try:
            local_vars = {f: X[:, i] for i, f in enumerate(features)}
            predicted = safe_math_eval(args.probe_expression, local_vars)
            predicted = np.asarray(predicted).reshape(-1, 1)
            mse = float(np.mean((predicted - Y) ** 2))
            rmse = float(np.sqrt(mse))
            r2 = float(
                1.0 - np.sum((Y - predicted) ** 2) / np.sum((Y - np.mean(Y)) ** 2)
            )

            return ToolResult(
                data={
                    "expression": args.probe_expression,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "samples": X.shape[0],
                },
                success=True,
            )
        except SafeEvalError as e:
            return ToolResult(
                data=None, success=False, error=f"Expression rejected: {e}"
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Evaluation failed: {e}")

    async def _compare(self, args: SymbolicRegressionInput) -> ToolResult:
        """Compare multiple expressions on the same data."""
        if not args.probe_expression:
            return ToolResult(
                data=None,
                success=False,
                error="probe_expression required for compare (semicolon-separated list)",
            )

        try:
            X, Y, features = self._load_data(args)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Data loading failed: {e}"
            )

        # Parse expressions: semicolon-separated
        expressions = [
            expr.strip()
            for expr in args.probe_expression.split(";")
            if expr.strip()
        ]
        if len(expressions) < 2:
            return ToolResult(
                data=None,
                success=False,
                error="At least 2 expressions required for comparison",
            )

        local_vars = {f: X[:, i] for i, f in enumerate(features)}

        results: list[dict[str, Any]] = []
        for expr in expressions:
            try:
                predicted = safe_math_eval(expr, local_vars)
                predicted = np.asarray(predicted, dtype=np.float64).reshape(-1, 1)
                mse = float(np.mean((predicted - Y) ** 2))
                rmse = float(np.sqrt(mse))
                ss_res = float(np.sum((Y - predicted) ** 2))
                ss_tot = float(np.sum((Y - np.mean(Y)) ** 2))
                r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
                results.append({
                    "expression": expr,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2,
                    "rank": None,
                })
            except Exception as e:
                results.append({
                    "expression": expr,
                    "error": str(e),
                    "rank": None,
                })

        # Rank by R² (descending), errors get lowest rank
        valid = [r for r in results if "error" not in r]
        valid.sort(key=lambda r: r["r2"], reverse=True)
        for i, r in enumerate(valid, 1):
            r["rank"] = i
        # Assign rank after valid for errored ones
        error_rank = len(valid) + 1
        for r in results:
            if r["rank"] is None:
                r["rank"] = error_rank
                error_rank += 1

        return ToolResult(
            data={
                "comparison": results,
                "best_expression": valid[0]["expression"] if valid else None,
                "best_r2": valid[0]["r2"] if valid else None,
                "features": features,
                "samples": X.shape[0],
                "expressions_evaluated": len(results),
            },
            success=True,
        )

    def _cuda_available(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def estimate_cost(self, args: SymbolicRegressionInput) -> dict[str, float] | None:
        """Estimate GPU hours for symbolic regression search."""
        return {
            "cpu_hours": 0.0,
            "gpu_hours": args.time_limit / 3600.0,
            "walltime_hours": args.time_limit / 3600.0,
        }
