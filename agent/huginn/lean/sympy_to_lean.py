"""SymPy → Lean 4 expression translator.

Translates a subset of SymPy expressions into Lean 4 `Expr` strings.
This is the AST bridge between Phase 1 (symbolic computation) and
Phase 2 (proof assistance).

Supported constructs:
  - Symbols, integers, rationals, floats
  - Add, Mul, Pow
  - sin, cos, exp, log, sqrt, diff, integrate
  - Matrix literals (Fin n → Fin m → ℝ)

Example:
    >>> import sympy as sp
    >>> x, y = sp.symbols('x y')
    >>> expr = sp.sin(x)**2 + sp.cos(x)**2
    >>> SymPyToLean().translate(expr)
    'Real.sin x ^ 2 + Real.cos x ^ 2'
"""

from __future__ import annotations

import sympy as sp


class SymPyToLean:
    """Translate SymPy expressions to Lean 4 source strings."""

    # Mapping of SymPy function names to Lean 4 names
    _FUNC_MAP: dict[str, str] = {
        "sin": "Real.sin",
        "cos": "Real.cos",
        "tan": "Real.tan",
        "exp": "Real.exp",
        "log": "Real.log",
        "sqrt": "Real.sqrt",
        "abs": "abs",
        "sign": "Real.sign",
    }

    def translate(self, expr: sp.Expr) -> str:
        """Convert a SymPy expression to a Lean 4 string."""
        return self._recurse(expr)

    def _recurse(self, expr: sp.Expr) -> str:
        # 0. Matrix (must come first because Matrix is not an Expr and lacks is_Integer)
        if isinstance(expr, sp.Matrix):
            rows = expr.tolist()
            entries = ", ".join(
                ", ".join(self._recurse(entry) for entry in row) for row in rows
            )
            return f"!![{entries}]"

        # 1. Numbers
        if expr.is_Integer:
            return str(int(expr))
        if expr.is_Rational:
            return f"({expr.p} / {expr.q})"
        if expr.is_Float:
            return str(float(expr))

        # 2. Symbols
        if isinstance(expr, sp.Symbol):
            return self._sanitize_name(str(expr))

        # 3. Add
        if expr.is_Add:
            terms = [self._recurse(arg) for arg in expr.args]
            return " + ".join(terms)

        # 4. Mul
        if expr.is_Mul:
            factors = [self._recurse(arg) for arg in expr.args]
            return " * ".join(factors)

        # 5. Pow
        if expr.is_Pow:
            base, exp = expr.args
            base_str = self._recurse(base)
            exp_str = self._recurse(exp)
            return f"{base_str} ^ {exp_str}"

        # 6. Function applications
        if isinstance(expr, sp.Function):
            fname = type(expr).__name__
            lean_fn = self._FUNC_MAP.get(fname, fname)
            args_str = " ".join(self._recurse(arg) for arg in expr.args)
            return f"{lean_fn} {args_str}"

        # 7. Derivative
        if isinstance(expr, sp.Derivative):
            f = expr.args[0]
            vars = expr.args[1:]
            # deriv (deriv f x) y  for ∂²f/∂x∂y
            result = self._recurse(f)
            for v, count in vars:
                for _ in range(count):
                    result = f"deriv {result} {self._recurse(v)}"
            return result

        # 8. Integral (indefinite)
        if isinstance(expr, sp.Integral):
            integrand, *limits = expr.args
            result = self._recurse(integrand)
            for lim in limits:
                if len(lim) == 1:
                    var = lim[0]
                    result = f"∫ {self._recurse(var)}, {result}"
                elif len(lim) == 3:
                    var, a, b = lim
                    result = f"∫ ({self._recurse(a)}) ({self._recurse(b)}) (fun {self._recurse(var)} => {result})"
            return result

        # Fallback
        return f"«{str(expr)}»"

    def _sanitize_name(self, name: str) -> str:
        """Make a SymPy symbol name valid as a Lean 4 identifier."""
        # Replace Greek letters with latin equivalents (simple heuristic)
        replacements = {
            "σ": "sigma",
            "ε": "epsilon",
            "μ": "mu",
            "ρ": "rho",
            "λ": "lambda",
            "α": "alpha",
            "β": "beta",
            "γ": "gamma",
            "θ": "theta",
            "φ": "phi",
            "ψ": "psi",
            "ω": "omega",
            "Δ": "Delta",
            "Σ": "Sigma",
        }
        for gk, lv in replacements.items():
            name = name.replace(gk, lv)
        # Remove primes, subscripts, etc.
        name = name.replace("'", "_prime")
        return name

    def theorem_statement(
        self,
        name: str,
        hypotheses: dict[str, sp.Expr],
        conclusion: sp.Expr,
    ) -> str:
        """Generate a Lean 4 theorem skeleton from SymPy expressions.

        Args:
            name: Theorem name (CamelCase recommended).
            hypotheses: Dict of hypothesis name → SymPy boolean expression.
            conclusion: SymPy boolean expression for the goal.
        """
        lines = [f"theorem {name} :"]
        if hypotheses:
            hyps = " →\n  ".join(
                f"({hname} : {self.translate(hexpr)})"
                for hname, hexpr in hypotheses.items()
            )
            lines.append(f"  {hyps} →\n  {self.translate(conclusion)} := by")
        else:
            lines.append(f"  {self.translate(conclusion)} := by")
        lines.append("  sorry")
        return "\n".join(lines)
