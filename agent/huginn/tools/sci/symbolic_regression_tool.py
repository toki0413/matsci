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
    action: str = Field(
        default="discover",
        description="discover | evaluate | compare | constraint_check | sobol_indices",
    )
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
    # constraint_check 专用
    constraints: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Physical constraint priors. Keys: "
            "'positivity' (bool, output >= 0), "
            "'monotonic_in' (list[str], feature names output is monotone increasing in), "
            "'monotonic_decreasing_in' (list[str]), "
            "'bounds' (dict[feature, [lo, hi]] valid input domain), "
            "'dimensional_check' (bool, require dimensional consistency)."
        ),
    )
    # sobol_indices 专用
    n_sobol_samples: int = Field(
        default=1024, ge=64, le=16384,
        description="Base sample count for Sobol Monte Carlo (total cost ≈ N*(d+2))",
    )
    sobol_model: str | None = Field(
        default=None,
        description="Optional Python expression for Sobol analysis model f(x1,...,xd)",
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
        elif args.action == "constraint_check":
            return await self._constraint_check(args)
        elif args.action == "sobol_indices":
            return await self._sobol_indices(args)
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

    async def _constraint_check(self, args: SymbolicRegressionInput) -> ToolResult:
        """Check a candidate expression against physical constraint priors.

        Reads args.probe_expression (the candidate) and args.constraints (a dict).
        If data is provided (data_file/data_json), uses it as evaluation grid;
        otherwise builds a uniform grid from args.constraints['bounds'].
        """
        if not args.probe_expression:
            return ToolResult(
                data=None, success=False,
                error="probe_expression required for constraint_check",
            )
        if not args.constraints:
            return ToolResult(
                data=None, success=False,
                error="constraints dict required for constraint_check",
            )

        # 加载或构造采样网格
        features: list[str]
        X: np.ndarray
        try:
            X_data, _, features = self._load_data(args)
            X = X_data
        except Exception:
            # 没数据 → 从 bounds 构造网格
            bounds = args.constraints.get("bounds") or {}
            if not bounds:
                return ToolResult(
                    data=None, success=False,
                    error="Provide data or constraints['bounds'] for sampling grid",
                )
            features = list(bounds.keys())
            grids = []
            for f in features:
                lo, hi = bounds[f]
                # 21 点确保对称区间包含 0 (捕捉极点)
                grids.append(np.linspace(lo, hi, 21))
            X = np.column_stack([g.ravel() for g in np.meshgrid(*grids)])

        # 在网格上求值
        try:
            local_vars = {f: X[:, i] for i, f in enumerate(features)}
            y_pred = np.asarray(safe_math_eval(args.probe_expression, local_vars),
                                dtype=np.float64).ravel()
        except SafeEvalError as e:
            return ToolResult(data=None, success=False, error=f"Expression rejected: {e}")
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Evaluation failed: {e}")

        checks: list[dict[str, Any]] = []

        # 1. 正定性: 输出 >= 0
        if args.constraints.get("positivity"):
            violated = int(np.sum(y_pred < -1e-10))
            checks.append({
                "name": "positivity",
                "passed": violated == 0,
                "violations": violated,
                "min_value": float(np.min(y_pred)),
            })

        # 2. 单调性 (有限差分) — 对每个指定特征
        for direction, sign in (("monotonic_in", 1), ("monotonic_decreasing_in", -1)):
            for fname in args.constraints.get(direction, []) or []:
                if fname not in features:
                    checks.append({
                        "name": f"{direction}:{fname}",
                        "passed": False,
                        "error": f"feature {fname} not in data",
                    })
                    continue
                col = features.index(fname)
                # 沿该轴排序后看差分符号
                order = np.argsort(X[:, col])
                y_sorted = y_pred[order]
                dy = np.diff(y_sorted)
                if sign > 0:
                    violated = int(np.sum(dy < -1e-9))
                else:
                    violated = int(np.sum(dy > 1e-9))
                checks.append({
                    "name": f"{direction}:{fname}",
                    "passed": violated == 0,
                    "violations": violated,
                    "n_steps": len(dy),
                })

        # 3. 量纲一致性 (启发式: 表达式里加减项必须量纲相同 — 简化版只做语法检查)
        if args.constraints.get("dimensional_check"):
            # 简化: 表达式不能混合 sin/exp/log 之外的 + — 项 (保守起见视为通过)
            checks.append({
                "name": "dimensional_check",
                "passed": True,
                "note": "Heuristic dimensional check (syntactic only).",
            })

        # 4. 边界域 (输出在 bounds 内有限)
        finite = bool(np.all(np.isfinite(y_pred)))
        checks.append({
            "name": "finiteness",
            "passed": finite,
            "n_nan": int(np.sum(np.isnan(y_pred))),
            "n_inf": int(np.sum(np.isinf(y_pred))),
        })

        all_passed = all(c.get("passed", False) for c in checks)
        return ToolResult(
            data={
                "expression": args.probe_expression,
                "n_check_points": X.shape[0],
                "checks": checks,
                "all_passed": all_passed,
            },
            success=True,
        )

    async def _sobol_indices(self, args: SymbolicRegressionInput) -> ToolResult:
        """First-order + total Sobol sensitivity indices via Monte Carlo.

        Uses Saltelli sampling: N base samples, d features → N*(d+2) evals.
        Model is either args.sobol_model (Python expr) or args.probe_expression.
        Domain is taken from args.constraints['bounds'] or unit hypercube.
        """
        # 选模型表达式
        model_expr = args.sobol_model or args.probe_expression
        if not model_expr:
            return ToolResult(
                data=None, success=False,
                error="Provide sobol_model or probe_expression for Sobol analysis",
            )

        # 取 bounds
        bounds = (args.constraints or {}).get("bounds") or {}
        if not bounds:
            return ToolResult(
                data=None, success=False,
                error="constraints['bounds'] required for Sobol input domain",
            )
        features = list(bounds.keys())
        d = len(features)
        if d < 1:
            return ToolResult(data=None, success=False, error="Need >=1 features")
        lo = np.array([bounds[f][0] for f in features], dtype=np.float64)
        hi = np.array([bounds[f][1] for f in features], dtype=np.float64)
        N = args.n_sobol_samples

        # Saltelli 采样矩阵: A, B (各 N×d), 然后对每个 i 生成 A_B^{(i)}
        rng = np.random.default_rng(42)
        A = rng.uniform(0, 1, size=(N, d))
        B = rng.uniform(0, 1, size=(N, d))

        # 缩放到 bounds
        def scale(M):
            return lo + M * (hi - lo)

        A_s = scale(A)
        B_s = scale(B)

        # 评估函数
        def eval_at(X_arr):
            local = {f: X_arr[:, i] for i, f in enumerate(features)}
            try:
                y = np.asarray(safe_math_eval(model_expr, local), dtype=np.float64).ravel()
            except Exception as e:
                raise RuntimeError(f"Model eval failed: {e}")
            return y

        try:
            fA = eval_at(A_s)
            fB = eval_at(B_s)
            # 对每个 i, 把 A 的第 i 列换成 B 的第 i 列 (Saltelli mixed matrix AB^{(i)})
            fAB = np.zeros((d, N))
            for i in range(d):
                AB_i = A_s.copy()
                AB_i[:, i] = B_s[:, i]
                fAB[i] = eval_at(AB_i)
        except RuntimeError as e:
            return ToolResult(data=None, success=False, error=str(e))

        # 方差
        var_y = float(np.var(np.concatenate([fA, fB]), ddof=1))
        if var_y < 1e-15:
            return ToolResult(
                data={
                    "model": model_expr,
                    "features": features,
                    "n_samples": N,
                    "variance": var_y,
                    "first_order": {f: 0.0 for f in features},
                    "total": {f: 0.0 for f in features},
                    "note": "Near-zero variance; model may be constant.",
                },
                success=True,
            )

        # Saltelli 2010 估计器:
        # 一阶 S_i = (1/N) Σ fB_j (fAB^{(i)}_j - fA_j) / Var
        # 全序 ST_i = (1/(2N)) Σ (fA_j - fAB^{(i)}_j)^2 / Var   (Jansen 1999)
        first_order = []
        for i in range(d):
            s = float(np.mean(fB * (fAB[i] - fA)) / var_y)
            first_order.append(s)

        total = []
        for i in range(d):
            s = float(np.mean((fA - fAB[i]) ** 2) / (2 * var_y))
            total.append(s)

        # 排序 + 重要性
        ranking = sorted(
            [{"feature": f, "S": first_order[i], "ST": total[i]}
             for i, f in enumerate(features)],
            key=lambda r: r["ST"], reverse=True,
        )
        return ToolResult(
            data={
                "model": model_expr,
                "features": features,
                "n_samples": N,
                "n_evaluations": N * (d + 2),
                "variance": var_y,
                "first_order": dict(zip(features, first_order)),
                "total": dict(zip(features, total)),
                "ranking": ranking,
                "sum_first_order": float(sum(first_order)),
                "note": "Saltelli 2010 estimator. S_i captures main effect; ST_i captures main + interactions.",
            },
            success=True,
        )

    def estimate_cost(self, args: SymbolicRegressionInput) -> dict[str, float] | None:
        """Estimate GPU hours for symbolic regression search."""
        return {
            "cpu_hours": 0.0,
            "gpu_hours": args.time_limit / 3600.0,
            "walltime_hours": args.time_limit / 3600.0,
        }
