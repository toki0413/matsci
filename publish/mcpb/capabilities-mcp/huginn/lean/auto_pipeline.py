"""Auto-Lean Pipeline — feed SymbolicMathTool results into Lean 4 automatically.

Bridges Phase 1 (symbolic computation) and Phase 2 (proof / verification)
by converting SymPy expressions into Lean 4 `Float` definitions and
compiling them without hand-written Lean code.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import sympy as sp

from huginn.lean.interface import LeanInterface, LeanResult
from huginn.lean.sympy_to_lean import SymPyToLean


class LeanCodeFixer:
    """Rule-based fixer for common Lean 4 compilation errors in generated code."""

    _LEAN_KEYWORDS = {
        "def",
        "theorem",
        "lemma",
        "example",
        "class",
        "structure",
        "inductive",
        "match",
        "if",
        "then",
        "else",
        "let",
        "in",
        "where",
        "namespace",
        "section",
        "end",
        "import",
        "open",
        "variable",
        "variables",
        "universe",
        "universes",
        "axiom",
        "constant",
        "partial",
        "mutual",
        "return",
        "do",
        "for",
        "while",
        "try",
        "catch",
        "finally",
        "throw",
        "macro",
        "syntax",
        "deriving",
        "instance",
        "opaque",
        "abbrev",
    }

    def __init__(self, raw_symbols=None):
        self.raw_symbols = set(raw_symbols or [])
        self._fixes_applied = []

    def fix(self, code, stderr):
        self._fixes_applied = []
        new_code = code
        new_code = self._fix_unknown_identifiers(new_code, stderr)
        new_code = self._fix_keyword_collisions(new_code)
        new_code = self._fix_missing_float_imports(new_code, stderr)
        new_code = self._fix_numeric_literals(new_code, stderr)
        if not self._fixes_applied:
            return None
        return new_code

    def _fix_unknown_identifiers(self, code, stderr):
        for match in re.finditer(r"unknown identifier '([^']+)'", stderr):
            bad = match.group(1)
            for sym in self.raw_symbols:
                if sym.replace("_", "") == bad and "_" in sym:
                    code = code.replace(bad, sym)
                    self._fixes_applied.append(f"restore_underscore: {bad} -> {sym}")
                    break
        return code

    def _fix_keyword_collisions(self, code):
        for kw in self._LEAN_KEYWORDS:
            if kw in self.raw_symbols:
                code = code.replace(f"({kw} : Float)", f"(sym_{kw} : Float)")
                code = re.sub(rf"def\s+{re.escape(kw)}\b", f"def sym_{kw}", code)
        return code

    def _fix_missing_float_imports(self, code, stderr):
        if (
            "Float.sin" in code
            and "unknown identifier 'Float.sin'" in stderr
            and "import Std" not in code
        ):
            code = "import Std\n" + code
            self._fixes_applied.append("add_std_import")
        return code

    def _fix_numeric_literals(self, code, stderr):
        if "has type Nat but is expected to have type Float" in stderr:
            code = re.sub(r"([\+\-\*\/\(\s])(\d+)(?![\.\d])", r"\1(\2 : Float)", code)
            self._fixes_applied.append("cast_nat_to_float")
        return code

    def applied(self):
        return self._fixes_applied


class FloatSymPyToLean(SymPyToLean):
    """SymPy → Lean 4 translator that targets `Float` instead of `Real`."""

    _FUNC_MAP = {
        "sin": "Float.sin",
        "cos": "Float.cos",
        "tan": "Float.tan",
        "exp": "Float.exp",
        "log": "Float.log",
        "sqrt": "Float.sqrt",
        "abs": "Float.abs",
        "sign": "Float.sign",
    }

    @staticmethod
    def _parenthesize_if_needed(s: str) -> str:
        return f"({s})" if " " in s else s

    def _recurse(self, expr) -> str:
        # Matrix
        if isinstance(expr, sp.Matrix):
            return super()._recurse(expr)

        # Mul: parenthesize Add factors so precedence is preserved
        if getattr(expr, "is_Mul", False):
            factors = []
            for arg in expr.args:
                s = self._recurse(arg)
                if getattr(arg, "is_Add", False):
                    s = f"({s})"
                factors.append(s)
            return " * ".join(factors)

        # Non-Pow delegates to base class
        if not getattr(expr, "is_Pow", False):
            return super()._recurse(expr)

        base, exp = expr.args
        base_str = self._parenthesize_if_needed(self._recurse(base))
        if exp.is_Integer:
            n = int(exp)
            if n >= 0:
                return f"{base_str} ^ {n}"
            return f"Float.pow {base_str} ({n} : Float)"
        if exp.is_Float:
            return f"Float.pow {base_str} {float(exp)}"
        exp_str = self._parenthesize_if_needed(self._recurse(exp))
        return f"Float.pow {base_str} {exp_str}"


class AutoLeanPipeline:
    """Automatically verify symbolic-computation results in Lean 4.

    Typical flow:
        1. SymbolicMathTool derives an expression (e.g. stress from free energy).
        2. AutoLeanPipeline converts the expression to Lean 4 `Float` code.
        3. LeanInterface compiles the snippet via `lake env lean --run`.
        4. The caller receives a LeanResult indicating success / failure.
    """

    def __init__(self, project_path: str | Any | None = None):
        if project_path is None:
            from pathlib import Path

            candidates = [
                Path(__file__).parent.parent.parent / "lean" / "HuginnLean",
                Path.cwd() / "lean" / "HuginnLean",
            ]
            for c in candidates:
                if (c / "lakefile.toml").exists():
                    project_path = c.resolve()
                    break
        if project_path is None:
            raise RuntimeError("HuginnLean project not found")
        self._lean = LeanInterface(project_path)
        self._translator = FloatSymPyToLean()

    # ------------------------------------------------------------------
    # Low-level: expression → Lean def
    # ------------------------------------------------------------------

    def verify_expression(
        self,
        expr: sp.Expr | str,
        name: str,
        symbols: list[str] | None = None,
        imports: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Convert a single SymPy expression to a Lean `def` and compile it.

        Free variables in the expression become parameters of the generated
        Lean function so that the definition is well-typed.

        Args:
            expr: SymPy expression or string to parse.
            name: Name for the generated Lean definition (CamelCase recommended).
            symbols: Symbol names to use when parsing a string expression.
            imports: Additional Lean modules to import.
            timeout: Compilation timeout in seconds.
        """
        if isinstance(expr, str):
            sym_dict = {s: sp.Symbol(s) for s in (symbols or [])}
            sym_dict.update(
                {
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "exp": sp.exp,
                    "log": sp.log,
                    "sqrt": sp.sqrt,
                    "pi": sp.pi,
                    "E": sp.E,
                }
            )
            expr = sp.sympify(expr, locals=sym_dict)

        lean_expr = self._translator.translate(expr)
        free = sorted({str(s) for s in expr.free_symbols})
        params = " ".join(f"({s} : Float)" for s in free)
        if params:
            code = f"def {name} {params} : Float := {lean_expr}"
        else:
            code = f"def {name} : Float := {lean_expr}"
        return self._lean.eval_lean_code(code, imports=imports or [], timeout=timeout)

    # ------------------------------------------------------------------
    # Mid-level: batch verify a dict of expressions
    # ------------------------------------------------------------------

    def verify_expression_dict(
        self,
        expressions: dict[str, sp.Expr | str],
        symbols: list[str] | None = None,
        imports: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Batch-convert multiple named expressions to Lean defs and compile.

        Args:
            expressions: Mapping {lean_def_name → sympy_expr_or_str}.
            symbols: Symbols available when parsing string expressions.
            imports: Lean modules to import.
            timeout: Compilation timeout.
        """
        lines = []
        for lean_name, expr in expressions.items():
            if isinstance(expr, str):
                # Defaults first, then user symbols override (so 'E' can be Young's modulus)
                sym_dict = {
                    "sin": sp.sin,
                    "cos": sp.cos,
                    "tan": sp.tan,
                    "exp": sp.exp,
                    "log": sp.log,
                    "sqrt": sp.sqrt,
                    "pi": sp.pi,
                }
                sym_dict.update({s: sp.Symbol(s) for s in (symbols or [])})
                expr = sp.sympify(expr, locals=sym_dict)
            lean_expr = self._translator.translate(expr)
            free = sorted({str(s) for s in expr.free_symbols})
            params = " ".join(f"({s} : Float)" for s in free)
            if params:
                lines.append(f"def {lean_name} {params} : Float := {lean_expr}")
            else:
                lines.append(f"def {lean_name} : Float := {lean_expr}")

        code = "\n\n".join(lines)

        # Reflection loop: compile, and on failure attempt rule-based fixes
        max_retries = 2
        result = self._lean.eval_lean_code(code, imports=imports or [], timeout=timeout)
        if not result.success:
            fixer = LeanCodeFixer(raw_symbols=symbols)
            for attempt in range(1, max_retries + 1):
                fixed = fixer.fix(code, result.stderr)
                if fixed is None:
                    break
                result = self._lean.eval_lean_code(
                    fixed, imports=imports or [], timeout=timeout
                )
                if result.success:
                    # Annotate stdout with fix info for debugging
                    result.stdout = (
                        f"[reflection] Fixed after {attempt} attempt(s): "
                        f"{', '.join(fixer.applied())}\n" + result.stdout
                    )
                    break
        return result

    # ------------------------------------------------------------------
    # High-level: consume SymbolicMathTool results directly
    # ------------------------------------------------------------------

    def verify_constitutive(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify constitutive-relation expressions from SymbolicMathTool.

        Args:
            symbolic_result: The `data` dict returned by SymbolicMathTool
                with action="constitutive".  String values are converted
                to Lean definitions; non-string entries are skipped.
            symbols: Extra symbol names for parsing.
            timeout: Compilation timeout.
        """
        expressions: dict[str, str] = {}
        for key, val in symbolic_result.items():
            if isinstance(val, str) and val.strip():
                lean_key = self._sanitize_lean_name(key)
                expressions[lean_key] = val

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No string expressions found in symbolic_result",
                returncode=-1,
                elapsed_seconds=0.0,
            )

        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.Elasticity"],
            timeout=timeout,
        )

    def verify_tensor_calculus(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify tensor calculus results from SymbolicMathTool.

        Converts invariants, principal values, and Voigt vectors into
        Lean `Float` definitions and compiles them.
        """
        expressions: dict[str, str] = {}

        invariants = symbolic_result.get("invariants", {})
        for key, val in invariants.items():
            if isinstance(val, (int, float)):
                expressions[f"tensorInv{key}"] = str(val)

        principal = symbolic_result.get("principal_values", [])
        for i, val in enumerate(principal):
            expressions[f"principalVal{i}"] = str(val)

        vm = symbolic_result.get("von_mises")
        if isinstance(vm, (int, float)):
            expressions["vonMises"] = str(vm)

        hydro = symbolic_result.get("hydrostatic_pressure")
        if isinstance(hydro, (int, float)):
            expressions["hydrostaticPressure"] = str(hydro)

        dev = symbolic_result.get("deviatoric_voigt", [])
        for i, val in enumerate(dev):
            expressions[f"devVoigt{i}"] = str(val)

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No tensor calculus data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.TensorAlgebra"],
            timeout=timeout,
        )

    def verify_derivative(
        self,
        original: str,
        variable: str,
        expected: str,
        test_points: dict[str, float] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Numerically verify a derivative at sample points.

        Generates Lean code that computes both finite-difference and
        symbolic-derivative values and checks they are close.

        Args:
            original: Original expression string (e.g. "x**3 + 2*x**2").
            variable: Variable name (e.g. "x").
            expected: Expected derivative expression string.
            test_points: Mapping {variable_name → value}. Defaults to {variable: 1.0}.
            timeout: Compilation timeout.
        """
        if test_points is None:
            test_points = {variable: 1.0}

        sym_dict = {s: sp.Symbol(s) for s in test_points}
        sym_dict[variable] = sp.Symbol(variable)
        sym_dict.update(
            {
                "sin": sp.sin,
                "cos": sp.cos,
                "tan": sp.tan,
                "exp": sp.exp,
                "log": sp.log,
                "sqrt": sp.sqrt,
                "pi": sp.pi,
                "E": sp.E,
            }
        )

        orig_expr = sp.sympify(original, locals=sym_dict)
        deriv_expr = sp.sympify(expected, locals=sym_dict)
        var_sym = sym_dict[variable]

        # Build finite-difference expression: (f(x+h) - f(x-h)) / (2h)
        h = sp.Symbol("h")
        fd_expr = (
            orig_expr.subs(var_sym, var_sym + h) - orig_expr.subs(var_sym, var_sym - h)
        ) / (2 * h)

        lean_fd = self._translator.translate(fd_expr)
        lean_deriv = self._translator.translate(deriv_expr)

        x_val = test_points[variable]
        code = f"""def fdVal (x h : Float) : Float := {lean_fd}
def derivVal (x : Float) : Float := {lean_deriv}
#eval fdVal {x_val} 1e-5
#eval derivVal {x_val}
"""

        # Use lake env lean directly (no --quiet) so #eval output is captured
        full_source = code + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=False,
            dir=self._lean.project_path / "HuginnLean",
            encoding="utf-8",
        ) as f:
            f.write(full_source)
            tmp_path = Path(f.name)

        try:
            proc = subprocess.run(
                ["lake", "env", "lean", str(tmp_path)],
                cwd=str(self._lean.project_path),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            tmp_path.unlink(missing_ok=True)
            return LeanResult(
                success=False,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                returncode=-1,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

        return LeanResult(
            success=proc.returncode == 0,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )

    def verify_weak_form(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify weak-form expressions from SymbolicMathTool.

        Converts terms like 'diffusion', 'reaction', 'convection' into
        Lean function definitions.
        """
        expressions: dict[str, str] = {}
        for key, val in symbolic_result.get("weak_form_terms", {}).items():
            if isinstance(val, str) and val.strip():
                expressions[f"weakForm{key.capitalize()}"] = val
        for key, val in symbolic_result.get("boundary_terms", {}).items():
            if isinstance(val, str) and val.strip():
                expressions[f"boundaryTerm{key.capitalize()}"] = val
        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No weak-form terms found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.ContinuumMechanics"],
            timeout=timeout,
        )

    def verify_fem(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 120,
    ) -> LeanResult:
        """Verify FEM results from SymbolicMathTool.

        Converts element stiffness matrices and weak-form expressions into
        Lean definitions that import HuginnLean.FiniteElement.
        """
        expressions: dict[str, str] = {}

        # Weak-form terms
        for key, val in symbolic_result.get("weak_form_terms", {}).items():
            if isinstance(val, str) and val.strip():
                expressions[f"femWeakForm{key.capitalize()}"] = val

        # Element matrix
        mat = symbolic_result.get("element_matrix")
        if isinstance(mat, list):
            # Generate a Lean def for each entry
            for i, row in enumerate(mat):
                for j, entry in enumerate(row):
                    expressions[f"femK{i}{j}"] = str(entry)
            expressions["femMatSize"] = str(len(mat))

        # Bilinear / linear forms
        bf = symbolic_result.get("bilinear_form")
        if isinstance(bf, str) and bf.strip():
            expressions["femBilinearForm"] = bf
        lf = symbolic_result.get("linear_functional")
        if isinstance(lf, str) and lf.strip():
            expressions["femLinearFunctional"] = lf

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No FEM data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.FiniteElement"],
            timeout=timeout,
        )

    def verify_eigenvalue(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify eigenvalue expressions from SymbolicMathTool.

        Converts symbolic eigenvalues, trace and determinant into Lean defs.
        """
        expressions: dict[str, str] = {}
        for i, ev in enumerate(symbolic_result.get("eigenvalues", [])):
            val = ev.get("value", "") if isinstance(ev, dict) else str(ev)
            if val:
                expressions[f"eigenval{i}"] = val
        trace = symbolic_result.get("trace", "")
        det = symbolic_result.get("determinant", "")
        if trace:
            expressions["matTrace"] = trace
        if det:
            expressions["matDet"] = det
        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No eigenvalue data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            timeout=timeout,
        )

    def verify_tensor_ops(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify tensor invariant expressions from SymbolicMathTool."""
        expressions: dict[str, str] = {}
        for key, val in symbolic_result.get("invariants", {}).items():
            if isinstance(val, str) and val.strip():
                expressions[f"invariant{key}"] = val
        trace = symbolic_result.get("trace", "")
        det = symbolic_result.get("determinant", "")
        if trace:
            expressions["tensorTrace"] = trace
        if det:
            expressions["tensorDet"] = det
        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No tensor invariants found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            timeout=timeout,
        )

    def verify_solve(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify solution expressions from SymbolicMathTool.

        Each solution dict like {"x": "2", "y": "3"} becomes a set of defs.
        """
        expressions: dict[str, str] = {}
        solutions = symbolic_result.get("solutions", [])
        if not solutions:
            sol = symbolic_result.get("solution", "")
            if sol:
                expressions["solution0"] = sol
        for i, sol in enumerate(solutions):
            if isinstance(sol, dict):
                for var_name, val in sol.items():
                    if isinstance(val, str):
                        expressions[f"sol{i}{var_name}"] = val
        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No solutions found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            timeout=timeout,
        )

    def verify_linear_algebra(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify linear algebra results from SymbolicMathTool.

        Converts matrix entries, solution vectors, and decomposition
        factors into Lean Float definitions.
        """
        expressions: dict[str, str] = {}

        # LU or Cholesky factor L
        L = symbolic_result.get("L")
        if isinstance(L, list):
            for i, row in enumerate(L):
                for j, entry in enumerate(row):
                    expressions[f"laL{i}{j}"] = str(entry)

        # LU factor U
        U = symbolic_result.get("U")
        if isinstance(U, list):
            for i, row in enumerate(U):
                for j, entry in enumerate(row):
                    expressions[f"laU{i}{j}"] = str(entry)

        # Solution vector
        sol = symbolic_result.get("solution")
        if isinstance(sol, list):
            for i, val in enumerate(sol):
                expressions[f"laX{i}"] = str(val)

        # Matrix-vector product result
        result = symbolic_result.get("result")
        if isinstance(result, list):
            for i, val in enumerate(result):
                expressions[f"laRes{i}"] = str(val)

        # Condition number
        cond = symbolic_result.get("cond_number")
        if isinstance(cond, str) and cond.strip():
            expressions["laCondNumber"] = cond

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No linear algebra data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.NumericalLinearAlgebra"],
            timeout=timeout,
        )

    def verify_dft(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify DFT computation results from SymbolicMathTool.

        Converts energies, DOS values, band structure points, and
        LDA XC energies into Lean Float definitions.
        """
        expressions: dict[str, str] = {}

        fe = symbolic_result.get("fermi_energy")
        if isinstance(fe, (int, float)):
            expressions["dftFermiEnergy"] = str(fe)
        kf = symbolic_result.get("fermi_wavevector")
        if isinstance(kf, (int, float)):
            expressions["dftFermiWavevector"] = str(kf)

        dos = symbolic_result.get("dos")
        if isinstance(dos, (int, float)):
            expressions["dftDOS"] = str(dos)

        levels = symbolic_result.get("levels", [])
        for lvl in levels:
            if isinstance(lvl, dict):
                n = lvl.get("n")
                e = lvl.get("energy")
                if isinstance(e, (int, float)) and n is not None:
                    expressions[f"dftLevel{n}Energy"] = str(e)

        band = symbolic_result.get("band", [])
        for i, pt in enumerate(band[:10]):  # limit to first 10 points
            if isinstance(pt, dict):
                k = pt.get("k")
                e = pt.get("energy")
                if isinstance(k, (int, float)) and isinstance(e, (int, float)):
                    expressions[f"dftBandK{i}"] = str(k)
                    expressions[f"dftBandE{i}"] = str(e)

        xc = symbolic_result.get("xc_energy_density")
        if isinstance(xc, (int, float)):
            expressions["dftXCEnergyDensity"] = str(xc)
        ex = symbolic_result.get("exchange_energy_density")
        if isinstance(ex, (int, float)):
            expressions["dftExchangeEnergyDensity"] = str(ex)
        ec = symbolic_result.get("correlation_energy_density")
        if isinstance(ec, (int, float)):
            expressions["dftCorrelationEnergyDensity"] = str(ec)

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No DFT data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.DFT"],
            timeout=timeout,
        )

    def verify_thermodynamics(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify thermodynamics computation results from SymbolicMathTool.

        Converts pressures, energies, chemical potentials, and partition
        functions into Lean Float definitions.
        """
        expressions: dict[str, str] = {}

        for key in [
            "pressure",
            "internal_energy",
            "volume",
            "temperature",
            "moles",
            "helmholtz_energy",
            "gibbs_energy",
            "enthalpy",
            "entropy",
            "entropy_change",
            "chemical_potential",
            "slope_dPdT",
            "latent_heat",
            "delta_volume",
            "single_partition_function",
            "thermal_wavelength",
            "critical_temperature",
            "critical_pressure",
        ]:
            val = symbolic_result.get(key)
            if isinstance(val, (int, float)):
                expressions[f"thermo{key.title().replace('_', '')}"] = str(val)

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No thermodynamics data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.Thermodynamics"],
            timeout=timeout,
        )

    def verify_probability(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify probability and GP results from SymbolicMathTool.

        Converts PDF values, CDF values, kernel values, integrals,
        and Bayesian posterior parameters into Lean Float definitions.
        """
        expressions: dict[str, str] = {}

        for key in [
            "pdf",
            "cdf",
            "kernel_value",
            "integral",
            "exact",
            "posterior_mean",
            "posterior_variance",
            "prior_mean",
            "prior_variance",
        ]:
            val = symbolic_result.get(key)
            if isinstance(val, (int, float)):
                expressions[f"prob{key.title().replace('_', '')}"] = str(val)

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No probability data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )
        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.Probability"],
            timeout=timeout,
        )

    def verify_unified(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify unified-framework derivation results in Lean 4.

        Converts the energy expression and any algebraic equations into
        Lean `Float` definitions.  Equations containing unevaluated
        derivatives or functions are skipped because they need a prior
        discretization step.
        """
        expressions: dict[str, str] = {}

        energy = symbolic_result.get("energy_expression")
        if isinstance(energy, str) and energy.strip():
            expressions["unifiedEnergy"] = energy

        equations = symbolic_result.get("equations", {})

        def _collect(obj: Any, prefix: str) -> None:
            if isinstance(obj, dict):
                # Eq represented as {"lhs": ..., "rhs": ...}
                if "lhs" in obj and "rhs" in obj:
                    residual = f"({obj['lhs']}) - ({obj['rhs']})"
                    if "Derivative" not in residual:
                        expressions[prefix] = residual
                    else:
                        # Still verify each side separately if it is algebraic
                        for side, key in (("lhs", "Lhs"), ("rhs", "Rhs")):
                            s = obj[side]
                            if isinstance(s, str) and "Derivative" not in s:
                                expressions[f"{prefix}{key}"] = s
                else:
                    for key, val in obj.items():
                        safe = self._sanitize_lean_name(str(key))
                        _collect(val, f"{prefix}_{safe}")
            elif isinstance(obj, list):
                for i, val in enumerate(obj):
                    _collect(val, f"{prefix}_{i}")
            elif isinstance(obj, str) and "Derivative" not in obj:
                expressions[prefix] = obj

        _collect(equations, "unifiedEq")

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No verifiable unified expressions found (algebraic only)",
                returncode=-1,
                elapsed_seconds=0.0,
            )

        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            timeout=timeout,
        )

    def verify_discretization(
        self,
        symbolic_result: dict[str, Any],
        symbols: list[str] | None = None,
        timeout: int = 60,
    ) -> LeanResult:
        """Verify a discretized/solved unified problem in Lean 4.

        Converts stiffness matrix entries and load vector entries into
        Lean `Float` definitions.  If a solution vector is present, it is
        also compiled so downstream checks (e.g. residual) can be added.
        """
        expressions: dict[str, str] = {}

        K = symbolic_result.get("stiffness_matrix")
        if isinstance(K, list):
            for i, row in enumerate(K):
                for j, entry in enumerate(row):
                    expressions[f"discK{i}{j}"] = str(entry)
            expressions["discN"] = str(len(K))

        F = symbolic_result.get("load_vector")
        if isinstance(F, list):
            for i, entry in enumerate(F):
                expressions[f"discF{i}"] = str(entry)

        u = symbolic_result.get("solution")
        if isinstance(u, list):
            for i, entry in enumerate(u):
                expressions[f"discU{i}"] = str(entry)

        if not expressions:
            return LeanResult(
                success=False,
                stdout="",
                stderr="No discretization data found",
                returncode=-1,
                elapsed_seconds=0.0,
            )

        return self.verify_expression_dict(
            expressions,
            symbols=symbols,
            imports=["HuginnLean.NumericalLinearAlgebra"],
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_lean_name(name: str) -> str:
        """Convert a snake_case Python key to a camelCase Lean identifier."""
        parts = name.replace("-", "_").split("_")
        return parts[0] + "".join(p.capitalize() for p in parts[1:])
