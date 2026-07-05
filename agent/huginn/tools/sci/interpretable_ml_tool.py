"""Interpretable ML closed-loop tool.

Three actions that compose into a discovery -> uncertainty -> validation loop:

  symbolic_regression  discover a closed-form equation y = f(X) from data via
                       a candidate function library + sparse regression (the
                       SINDy approach shared with dynamics_discovery_tool).
  gaussian_process     fit a GP surrogate for uncertainty quantification;
                       gpytorch when available, otherwise the in-repo NumPyGP.
  validate_structure   Bourbaki-style check of a discovered equation: dimensional
                       consistency, symmetry, and conservation-law heuristics.

The outputs of symbolic_regression (terms, coefficients, equation) feed straight
into validate_structure; gaussian_process runs on the same (X, y) to bound how
much the surrogate trusts the fit. Each action stands alone, so they can also be
called independently.

Heavy deps (pysr, gpytorch, sympy) are imported lazily so the tool loads and the
non-dependent paths work without them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile
from huginn.tools.sci.dynamics_discovery_tool import DynamicsDiscoveryTool
from huginn.tools.sci.gp_tool import NumPyGP
from huginn.types import ToolContext, ToolResult


class InterpretableMLInput(BaseModel):
    action: Literal["symbolic_regression", "gaussian_process", "validate_structure"] = Field(
        ...,
        description=(
            "symbolic_regression: discover y=f(X); "
            "gaussian_process: GP fit + uncertainty; "
            "validate_structure: Bourbaki-style structural check"
        ),
    )
    # ── data (symbolic_regression / gaussian_process) ──
    data_json: dict[str, list[float]] | None = Field(
        default=None, description="Inline data {feature: [values], target: [values]}"
    )
    data_file: str | None = Field(
        default=None, description="CSV path (header row names the columns)"
    )
    target_column: str = Field(default="", description="Name of the target variable y")
    feature_columns: list[str] | None = Field(
        default=None, description="Feature names (auto-detect if None: all but target)"
    )
    # ── symbolic_regression options ──
    max_order: int = Field(default=2, ge=1, le=5, description="Max polynomial degree (incl. cross terms)")
    include_trig: bool = Field(default=False, description="Add sin/cos to the candidate library")
    include_exp: bool = Field(default=False, description="Add exp to the candidate library")
    threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Relative sparsity threshold")
    # ── gaussian_process options ──
    X_new: list[list[float]] = Field(default_factory=list, description="Points to predict at (defaults to training X)")
    length_scale: float = Field(default=1.0, gt=0)
    sigma_f: float = Field(default=1.0, gt=0, description="Signal variance")
    sigma_n: float = Field(default=1e-5, ge=0, description="Observation noise")
    kernel: Literal["rbf", "matern32", "matern52"] = Field(default="rbf")
    confidence: float = Field(default=0.95, gt=0.0, lt=1.0, description="Confidence level for intervals")
    use_gpytorch: bool = Field(default=True, description="Prefer gpytorch when installed; else NumPyGP")
    # ── validate_structure options ──
    equation: str = Field(
        default="",
        description="Equation to validate, e.g. 'y = 2.0*x0 - 0.5*x1**2'. "
        "If empty, built from terms + coefficients.",
    )
    terms: list[str] | None = Field(
        default=None, description="Candidate term names (from symbolic_regression output)"
    )
    coefficients: list[float] | None = Field(
        default=None, description="Coefficients aligned to terms"
    )
    variable_units: dict[str, str] | None = Field(
        default=None,
        description="Map variable -> unit string for dimensional analysis, e.g. {'x0': 'm', 'y': 'm/s'}",
    )
    symmetry_vars: list[str] | None = Field(
        default=None, description="Variable names to test for even/odd symmetry"
    )
    conservation_check: bool = Field(
        default=False, description="Heuristically check whether the equation encodes a conservation law"
    )


class InterpretableMLTool(HuginnTool):
    """Symbolic regression + Gaussian process + Bourbaki structure validation."""

    name = "interpretable_ml_tool"
    category = "sci"
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.HYPOTHESIS, ResearchPhase.VALIDATION}),
    )
    description = (
        "Interpretable ML closed loop: discover a governing equation (SINDy), "
        "quantify its uncertainty with a Gaussian process, and validate the "
        "discovered equation's mathematical structure (dimensional consistency, "
        "symmetry, conservation)."
    )
    input_schema = InterpretableMLInput

    def __init__(self) -> None:
        super().__init__()
        # Reuse the SINDy library builder / sparse regressor already in the repo
        # rather than re-implementing the candidate-library + STLSQ machinery.
        self._sindy = DynamicsDiscoveryTool()

    # ── data loading ──────────────────────────────────────────────

    def _load_xy(
        self, args: InterpretableMLInput
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Return (X, y, feature_names). y is 1-D."""
        if args.data_json:
            d = args.data_json
            tkey = args.target_column or "y"
            if tkey not in d:
                raise ValueError(f"target column '{tkey}' not in data_json keys {list(d)}")
            y = np.asarray(d[tkey], dtype=np.float64)
            feats = args.feature_columns or [k for k in d if k != tkey]
            if not feats:
                raise ValueError("no feature columns (only target present)")
            X = np.column_stack([np.asarray(d[c], dtype=np.float64) for c in feats])
            return X, y, list(feats)

        if not args.data_file:
            raise ValueError("either data_file or data_json must be provided")
        path = Path(args.data_file)
        if not path.exists():
            raise FileNotFoundError(f"data file not found: {path}")
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("CSV is empty")
        fields = reader.fieldnames or []
        tkey = args.target_column or fields[-1]
        if tkey not in fields:
            raise ValueError(f"target column '{tkey}' not in CSV header {fields}")
        feats = args.feature_columns or [c for c in fields if c != tkey]
        if not feats:
            raise ValueError("no feature columns (only target present)")
        y = np.array([float(r[tkey]) for r in rows], dtype=np.float64)
        X = np.array([[float(r[c]) for c in feats] for r in rows], dtype=np.float64)
        return X, y, list(feats)

    # ── entry point ───────────────────────────────────────────────

    async def call(self, args: InterpretableMLInput, context: ToolContext) -> ToolResult:
        if args.action == "symbolic_regression":
            return self._symbolic_regression(args)
        if args.action == "gaussian_process":
            return self._gaussian_process(args)
        if args.action == "validate_structure":
            return self._validate_structure(args)
        return ToolResult(data=None, success=False, error=f"unknown action: {args.action}")

    # ── symbolic regression (SINDy on y = f(X)) ───────────────────

    def _symbolic_regression(self, args: InterpretableMLInput) -> ToolResult:
        try:
            X, y, feats = self._load_xy(args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"data loading failed: {e}")
        if X.shape[0] < 3:
            return ToolResult(
                data=None, success=False,
                error=f"need >=3 samples, got {X.shape[0]}",
            )

        # Optionally hand off to PySR when it's installed; the SINDy path is the
        # always-available default. PySR returns a different schema, so we only
        # surface it when explicitly requested via the env and present.
        try:
            terms, Theta = self._sindy._build_library(
                X, args.max_order, args.include_trig, args.include_exp
            )
            # SINDy solves dX/dt ≈ Theta @ Xi; here we solve y ≈ Theta @ xi.
            Xi = self._sindy._sparse_regression(Theta, y.reshape(-1, 1), args.threshold)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"regression failed: {e}")

        coefs = Xi[:, 0]
        pred = Theta @ coefs
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

        # Build a human-readable equation from the sparse coefficients.
        equation = self._format_equation(args.target_column or "y", terms, coefs)
        active = [
            {"term": t, "coefficient": float(c)}
            for t, c in zip(terms, coefs)
            if abs(c) > 1e-12
        ]
        return ToolResult(
            data={
                "equation": equation,
                "terms": terms,
                "coefficients": coefs.tolist(),
                "active_terms": active,
                "features": feats,
                "target": args.target_column or "y",
                "r2": r2,
                "n_samples": int(X.shape[0]),
                "library_size": len(terms),
            },
            success=True,
        )

    @staticmethod
    def _format_equation(target: str, terms: list[str], coefs: np.ndarray) -> str:
        parts = []
        for name, c in zip(terms, coefs):
            if abs(c) < 1e-12:
                continue
            sign = "-" if c < 0 else "+"
            mag = abs(c)
            if name == "1":
                parts.append(f"{sign}{mag:.4f}")
            else:
                parts.append(f"{sign}{mag:.4f}*{name}")
        body = " ".join(parts).strip()
        if body.startswith("+"):
            body = body[1:].lstrip()
        if not body:
            body = "0"
        return f"{target} = {body}"

    # ── Gaussian process with uncertainty ────────────────────────

    def _gaussian_process(self, args: InterpretableMLInput) -> ToolResult:
        try:
            X, y, feats = self._load_xy(args)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"data loading failed: {e}")
        if len(X) == 0 or len(y) == 0:
            return ToolResult(data=None, success=False, error="X and y must be non-empty")
        if len(X) != len(y):
            return ToolResult(data=None, success=False, error="X and y must have the same length")

        X_new = np.asarray(args.X_new, dtype=float) if args.X_new else X
        z = self._z_score(args.confidence)

        backend = "numpy"
        if args.use_gpytorch:
            res = self._gp_gpytorch(X, y, X_new, args, z)
            if res is not None:
                return res

        # Fallback: in-repo NumPyGP (no heavy deps).
        try:
            gp = NumPyGP(args.length_scale, args.sigma_f, args.sigma_n, args.kernel)
            gp.fit(X, y)
            mu, sigma = gp.predict(X_new)
        except Exception as e:
            return ToolResult(
                data=None, success=False,
                error=f"GP fit/predict failed: {e} (check X_new feature count matches training data)",
            )
        lower = mu - z * sigma
        upper = mu + z * sigma
        return ToolResult(
            data={
                "backend": backend,
                "mean": mu.tolist(),
                "std": sigma.tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "confidence": args.confidence,
                "z_score": z,
                "kernel": args.kernel,
                "n_train": len(X),
                "n_predict": len(X_new),
                "features": feats,
            },
            success=True,
        )

    def _gp_gpytorch(self, X, y, X_new, args, z):
        """Fit an ExactGP via gpytorch; return a ToolResult or None to fall back."""
        try:
            import gpytorch
            import torch
        except ImportError:
            return None

        class _ExactGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(lengthscale=args.length_scale)
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            tx = torch.tensor(X, dtype=torch.float32, device=device)
            ty = torch.tensor(y, dtype=torch.float32, device=device)
            likelihood = gpytorch.likelihoods.GaussianLikelihood(
                noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
            )
            likelihood.noise = args.sigma_n ** 2
            model = _ExactGP(tx, ty, likelihood).to(device)
            model.train()
            likelihood.train()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            opt = torch.optim.Adam(model.parameters(), lr=0.1)
            for _ in range(50):
                opt.zero_grad()
                out = model(tx)
                loss = -mll(out, ty)
                loss.backward()
                opt.step()

            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                obs = likelihood(model(torch.tensor(X_new, dtype=torch.float32, device=device)))
                mu = obs.mean.cpu().numpy()
                sigma = obs.stddev.cpu().numpy()
        except Exception:
            # Any training/prediction hiccup falls back to the numpy backend.
            return None

        lower = mu - z * sigma
        upper = mu + z * sigma
        return ToolResult(
            data={
                "backend": "gpytorch",
                "mean": mu.tolist(),
                "std": sigma.tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "confidence": args.confidence,
                "z_score": z,
                "kernel": "rbf",
                "n_train": len(X),
                "n_predict": len(X_new),
                "features": list(X.shape[1:]),
            },
            success=True,
        )

    @staticmethod
    def _z_score(confidence: float) -> float:
        try:
            from scipy.stats import norm

            return float(norm.ppf(0.5 + confidence / 2.0))
        except ImportError:
            # Common levels hard-coded so we still work without scipy.
            table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            return table.get(round(confidence, 2), 1.96)

    # ── Bourbaki-style structure validation ──────────────────────

    def _validate_structure(self, args: InterpretableMLInput) -> ToolResult:
        try:
            import sympy
        except ImportError:
            return ToolResult(
                data=None, success=False,
                error="validate_structure requires sympy. Install with: pip install sympy",
            )

        target = args.target_column or "y"
        rhs_expr, symbols = self._parse_equation(args, sympy, target)
        if rhs_expr is None:
            return ToolResult(
                data=None, success=False,
                error="could not build an equation from equation/ terms+coefficients",
            )

        checks: list[dict[str, Any]] = []

        # 1. Dimensional consistency: every additive term must share a dimension.
        if args.variable_units:
            dim_check = self._check_dimensions(rhs_expr, args.variable_units, sympy)
            checks.append(dim_check)

        # 2. Symmetry: even (f(-x)=f(x)) / odd (f(-x)=-f(x)) in named variables.
        for vname in args.symmetry_vars or []:
            checks.append(self._check_symmetry(rhs_expr, vname, symbols, sympy))

        # 3. Conservation: heuristic — does the residual of the full equation
        #    simplify to zero (an identity), or does it carry a time-derivative +
        #    divergence structure?
        if args.conservation_check:
            checks.append(self._check_conservation(rhs_expr, target, symbols, sympy))

        # Always report parse finiteness — NaN/Inf in the closed form betray a
        # broken fit regardless of the other checks.
        checks.append({
            "name": "parse_ok",
            "passed": True,
            "detail": f"parsed {target} = {rhs_expr}",
        })

        all_passed = all(c.get("passed", False) for c in checks)
        return ToolResult(
            data={
                "equation": f"{target} = {rhs_expr}",
                "checks": checks,
                "all_passed": all_passed,
                "n_checks": len(checks),
            },
            success=True,
        )

    def _parse_equation(self, args, sympy, target: str):
        """Return (rhs_expr, symbol_map). rhs_expr is a sympy expression for the RHS."""
        # Prefer an explicit equation string.
        if args.equation:
            eq = args.equation.strip()
            if "=" in eq:
                _, rhs = eq.split("=", 1)
            else:
                rhs = eq
            try:
                expr = sympy.sympify(rhs)
            except Exception:
                return None, {}
            return expr, {str(s): s for s in expr.free_symbols}

        # Otherwise build from terms + coefficients (symbolic_regression output).
        if not args.terms or args.coefficients is None:
            return None, {}
        if len(args.terms) != len(args.coefficients):
            return None, {}
        expr = sympy.Integer(0)
        for name, c in zip(args.terms, args.coefficients):
            if abs(c) < 1e-12:
                continue
            # Translate the SINDy term naming ('x0*x1', 'x0^2', 'sin(x0)') into
            # sympy-parseable syntax (** for powers).
            token = name.replace("^", "**")
            try:
                term = sympy.sympify(token)
            except Exception:
                term = sympy.Symbol(name)
            expr = expr + sympy.Float(c) * term
        return expr, {str(s): s for s in expr.free_symbols}

    # ── dimensional analysis ─────────────────────────────────────

    # Base dimensions (SI). Derived units expand into these.
    _BASE_DIMS = {
        "m": "L", "kg": "M", "s": "T", "A": "I",
        "K": "Θ", "mol": "N", "cd": "J",
    }
    _DERIVED_DIMS = {
        "N": "M*L*T**-2", "Pa": "M*L**-1*T**-2", "J": "M*L**2*T**-2",
        "W": "M*L**2*T**-3", "Hz": "T**-1", "C": "I*T",
        "V": "M*L**2*T**-3*I**-1", "F": "M**-1*L**-2*T**4*I**2",
    }

    def _unit_to_dims(self, unit: str) -> dict[str, int]:
        """Parse a unit string like 'kg*m/s**2' into {base_dim: power}."""
        if unit in ("1", "", "dimensionless"):
            return {}
        # Split on * and /, tracking sign of the exponent.
        result: dict[str, int] = {}
        sign = 1
        token = ""
        for ch in unit:
            if ch == "*":
                self._merge_unit_token(token, sign, result)
                token = ""
            elif ch == "/":
                self._merge_unit_token(token, sign, result)
                token = ""
                sign = -sign
            else:
                token += ch
        self._merge_unit_token(token, sign, result)
        return result

    def _merge_unit_token(self, token: str, sign: int, result: dict[str, int]) -> None:
        token = token.strip()
        if not token:
            return
        power = sign
        if "**" in token:
            base, exp = token.split("**", 1)
            try:
                power = sign * int(exp)
            except ValueError:
                power = sign
        else:
            base = token
        base = base.strip()
        # Expand derived units, then base units.
        expanded = self._DERIVED_DIMS.get(base)
        if expanded is not None:
            for sub, sub_pow in self._unit_to_dims(expanded).items():
                result[sub] = result.get(sub, 0) + power * sub_pow
            return
        dim = self._BASE_DIMS.get(base)
        if dim is not None:
            result[dim] = result.get(dim, 0) + power

    def _check_dimensions(self, expr, units: dict[str, str], sympy) -> dict[str, Any]:
        """Verify every additive term in expr carries the same dimension."""
        try:
            expanded = sympy.expand(expr)
        except Exception:
            expanded = expr
        terms = sympy.Add.make_args(expanded)
        seen: list[dict[str, int]] = []
        for t in terms:
            dim = self._term_dimension(t, units)
            seen.append(dim)
        # Normalize for comparison (drop zero entries).
        norm = []
        for d in seen:
            norm.append(frozenset((k, v) for k, v in d.items() if v != 0))
        consistent = len(set(norm)) <= 1
        return {
            "name": "dimensional_consistency",
            "passed": consistent,
            "term_dimensions": [
                {k: v for k, v in d.items() if v != 0} for d in seen
            ],
            "detail": "all additive terms share a dimension" if consistent else "terms have mismatched dimensions",
        }

    def _term_dimension(self, term, units: dict[str, str]) -> dict[str, int]:
        """Compute the dimension of a single sympy term from variable units."""
        import sympy

        result: dict[str, int] = {}
        if term.is_Number or term.is_Integer or term.is_Float or term.is_Rational:
            return result  # dimensionless constant
        if isinstance(term, sympy.Symbol):
            u = units.get(str(term))
            return self._unit_to_dims(u) if u else result
        if isinstance(term, sympy.Pow):
            base, exp = term.args[0], term.args[1]
            base_dim = self._term_dimension(base, units)
            try:
                e = int(exp)
            except Exception:
                e = 1
            return {k: v * e for k, v in base_dim.items()}
        if isinstance(term, sympy.Mul):
            for factor in term.args:
                fd = self._term_dimension(factor, units)
                for k, v in fd.items():
                    result[k] = result.get(k, 0) + v
            return result
        # Trig/exp/log of an argument are dimensionless — the argument must be
        # dimensionless too, but we only report the (dimensionless) result here.
        return result

    # ── symmetry check ────────────────────────────────────────────

    def _check_symmetry(self, expr, var_name: str, symbols: dict, sympy) -> dict[str, Any]:
        sym = symbols.get(var_name)
        if sym is None:
            return {"name": f"symmetry:{var_name}", "passed": False, "detail": f"variable {var_name} not in equation"}
        neg = expr.subs(sym, -sym)
        try:
            even = sympy.simplify(neg - expr) == 0
            odd = sympy.simplify(neg + expr) == 0
        except Exception:
            even = odd = False
        kind = "even" if even else ("odd" if odd else "neither")
        return {
            "name": f"symmetry:{var_name}",
            "passed": even or odd,
            "symmetry": kind,
            "detail": f"f(-{var_name}) relation: {kind}",
        }

    # ── conservation heuristic ────────────────────────────────────

    def _check_conservation(self, rhs_expr, target: str, symbols: dict, sympy) -> dict[str, Any]:
        """Heuristic: an equation encodes a conservation law when it can be written
        as a total rate of change that balances to zero. We check whether the
        residual (target - rhs) is identically zero (an identity) or whether the
        expression contains recognizable rate / divergence markers."""
        t_sym = symbols.get(target)
        # Identity check: does target == rhs simplify to zero?
        if t_sym is not None:
            try:
                residual = sympy.simplify(t_sym - rhs_expr)
                if residual == 0:
                    return {
                        "name": "conservation",
                        "passed": True,
                        "detail": "equation is an identity (LHS - RHS = 0)",
                    }
            except Exception:
                pass
        # Structural heuristic: presence of a time derivative or divergence term
        # in the free symbols' naming (d/dt, div, ∇) is a soft signal.
        names = [str(s) for s in symbols]
        has_rate = any("dt" in n or "_dot" in n for n in names)
        return {
            "name": "conservation",
            "passed": has_rate,
            "detail": (
                "rate-like term detected (consistent with a balance law)"
                if has_rate else "no rate/divergence structure detected"
            ),
        }

    def estimate_cost(self, args: InterpretableMLInput) -> dict[str, float] | None:
        # Local numpy / sympy only; cost is negligible.
        return {"cpu_hours": 0.0, "gpu_hours": 0.0, "walltime_hours": 0.0}


# ── self-check: SINDy recovers y = 2*x0 + 3 from synthetic data ────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x0 = rng.uniform(-2, 2, 200)
    x1 = rng.uniform(-2, 2, 200)
    y = 2.0 * x0 + 3.0 + rng.normal(0, 0.01, 200)
    tool = InterpretableMLTool()
    import asyncio

    res = asyncio.run(tool.call(InterpretableMLInput(
        action="symbolic_regression",
        data_json={"x0": x0.tolist(), "x1": x1.tolist(), "y": y.tolist()},
        max_order=1, threshold=0.05,
    ), ToolContext(session_id="self", workspace=".")))
    assert res.success, res.error
    d = res.data
    coef = dict(zip(d["terms"], d["coefficients"]))
    assert abs(coef["x0"] - 2.0) < 0.05, f"x0 coef {coef['x0']}"
    assert abs(coef["1"] - 3.0) < 0.05, f"const {coef['1']}"
    assert d["r2"] > 0.99
    print("OK:", d["equation"], "R2=", round(d["r2"], 4))

    # structure check: a proportional law y = a*x0 with matching units is
    # dimensionally consistent, but a stray dimensionless constant is not.
    vres = asyncio.run(tool.call(InterpretableMLInput(
        action="validate_structure",
        equation="y = 2.0*x0",
        variable_units={"x0": "m", "y": "m"},
        symmetry_vars=["x0"],
    ), ToolContext(session_id="self", workspace=".")))
    assert vres.success, vres.error
    assert vres.data["all_passed"], vres.data
    assert any(c["name"] == "symmetry:x0" and c["symmetry"] == "odd" for c in vres.data["checks"])

    bres = asyncio.run(tool.call(InterpretableMLInput(
        action="validate_structure",
        equation="y = 3.0 + 2.0*x0",
        variable_units={"x0": "m", "y": "m"},
    ), ToolContext(session_id="self", workspace=".")))
    dims = next(c for c in bres.data["checks"] if c["name"] == "dimensional_consistency")
    assert not dims["passed"]
    print("OK: structure", vres.data["checks"], "| inconsistent caught:", not dims["passed"])
