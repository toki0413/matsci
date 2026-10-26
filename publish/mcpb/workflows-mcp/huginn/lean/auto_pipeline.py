"""Auto-Lean Pipeline — feed SymbolicMathTool results into Lean 4 automatically.

Bridges Phase 1 (symbolic computation) and Phase 2 (proof / verification)
by converting SymPy expressions into Lean 4 `Float` definitions and
compiling them without hand-written Lean code.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable
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


# sympify 默认符号表: 表达式字符串里的数学函数 / 常量.
_SYMPY_DEFAULTS: dict[str, Any] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "pi": sp.pi,
}

# 热力学结果里可验证的标量字段 (按原顺序; 顺序决定生成的 Lean def 顺序).
_THERMO_KEYS = (
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
)


def _user_symbols(symbols: list[str] | None) -> dict[str, Any]:
    """符号名 → sp.Symbol 映射."""
    return {s: sp.Symbol(s) for s in (symbols or [])}


def _sympy_locals(
    symbols: list[str] | None, *, user_wins: bool, with_e: bool = False
) -> dict[str, Any]:
    """构造 sympify 用的 locals 表.

    user_wins=True: 用户符号覆盖默认表 (dict 批量版本, 允许 E 当弹性模量);
    user_wins=False: 默认表 (及可选 E) 覆盖用户符号 (单表达式 / 导数版本).
    """
    user = _user_symbols(symbols)
    if user_wins:
        merged = dict(_SYMPY_DEFAULTS)
        merged.update(user)
        return merged
    merged = dict(user)
    if with_e:
        merged["E"] = sp.E
    merged.update(_SYMPY_DEFAULTS)
    return merged


def _empty_lean_result(message: str) -> LeanResult:
    """构造一条 "没有可验证数据" 的失败结果."""
    return LeanResult(
        success=False,
        stdout="",
        stderr=message,
        returncode=-1,
        elapsed_seconds=0.0,
    )


def _lean_def_line(name: str, expr: Any, translator: Any) -> str:
    """SymPy 表达式 → 单条 `def name (params) : Float := ...` 源码."""
    lean_expr = translator.translate(expr)
    free = sorted({str(s) for s in expr.free_symbols})
    params = " ".join(f"({s} : Float)" for s in free)
    if params:
        return f"def {name} {params} : Float := {lean_expr}"
    return f"def {name} : Float := {lean_expr}"


def _add_matrix_defs(expressions: dict[str, str], matrix: Any, prefix: str) -> None:
    """二维矩阵逐元素生成 Lean def (prefix{i}{j})."""
    for i, row in enumerate(matrix):
        for j, entry in enumerate(row):
            expressions[f"{prefix}{i}{j}"] = str(entry)


def _add_vector_defs(expressions: dict[str, str], values: Any, prefix: str) -> None:
    """一维向量逐元素生成 Lean def (prefix{i})."""
    for i, value in enumerate(values):
        expressions[f"{prefix}{i}"] = str(value)


def _add_number_def(expressions: dict[str, str], value: Any, name: str) -> None:
    """标量数值生成 def; 非 int/float 跳过."""
    if isinstance(value, int | float):
        expressions[name] = str(value)


def _add_text_def(expressions: dict[str, str], value: Any, name: str) -> None:
    """非空字符串生成 def; 其他跳过."""
    if isinstance(value, str) and value.strip():
        expressions[name] = value


def _title_join(key: str) -> str:
    """snake_case key → Lean 标识符后缀 (Title + 去下划线)."""
    return key.title().replace("_", "")


def _add_number_table(
    expressions: dict[str, str],
    data: dict[str, Any],
    keys: tuple[str, ...],
    prefix: str,
    suffix: Callable[[str], str],
) -> None:
    """按固定 key 表抽取数值 → def (prefix + suffix(key))."""
    for key in keys:
        _add_number_def(expressions, data.get(key), f"{prefix}{suffix(key)}")


def _dft_level_defs(expressions: dict[str, str], levels: Any) -> None:
    """DFT 能级列表 → dftLevel{n}Energy."""
    for lvl in levels:
        if isinstance(lvl, dict):
            n = lvl.get("n")
            energy = lvl.get("energy")
            if isinstance(energy, int | float) and n is not None:
                expressions[f"dftLevel{n}Energy"] = str(energy)


def _dft_band_defs(expressions: dict[str, str], band: Any) -> None:
    """DFT 能带点 → dftBandK{i} / dftBandE{i}; 只取前 10 点."""
    for i, pt in enumerate(band[:10]):  # limit to first 10 points
        if isinstance(pt, dict):
            k = pt.get("k")
            energy = pt.get("energy")
            if isinstance(k, int | float) and isinstance(energy, int | float):
                expressions[f"dftBandK{i}"] = str(k)
                expressions[f"dftBandE{i}"] = str(energy)


def _collect_unified_eq(
    expressions: dict[str, str],
    obj: Any,
    prefix: str,
    sanitize: Callable[[str], str],
) -> None:
    """递归收集 unified 方程里可验证的代数表达式 (跳过含 Derivative 的项)."""
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
            return
        for key, val in obj.items():
            _collect_unified_eq(
                expressions, val, f"{prefix}_{sanitize(str(key))}", sanitize
            )
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            _collect_unified_eq(expressions, val, f"{prefix}_{i}", sanitize)
    elif isinstance(obj, str) and "Derivative" not in obj:
        expressions[prefix] = obj


def _derivative_source(
    translator: Any,
    original: str,
    variable: str,
    expected: str,
    test_points: dict[str, float] | None,
) -> str:
    """构造导数校验用 Lean 源码: 有限差分 + 符号导数各算一次."""
    points = {variable: 1.0} if test_points is None else test_points
    sym_locals = _sympy_locals([*points, variable], user_wins=False, with_e=True)

    orig_expr = sp.sympify(original, locals=sym_locals)
    deriv_expr = sp.sympify(expected, locals=sym_locals)
    var_sym = sp.Symbol(variable)

    # Build finite-difference expression: (f(x+h) - f(x-h)) / (2h)
    h = sp.Symbol("h")
    fd_expr = (
        orig_expr.subs(var_sym, var_sym + h) - orig_expr.subs(var_sym, var_sym - h)
    ) / (2 * h)

    lean_fd = translator.translate(fd_expr)
    lean_deriv = translator.translate(deriv_expr)
    x_val = points[variable]
    return f"""def fdVal (x h : Float) : Float := {lean_fd}
def derivVal (x : Float) : Float := {lean_deriv}
#eval fdVal {x_val} 1e-5
#eval derivVal {x_val}
"""


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
    # Internal plumbing
    # ------------------------------------------------------------------

    def _lean_defs(
        self,
        expressions: dict[str, str],
        symbols: list[str] | None,
        timeout: int,
        missing_msg: str,
        imports: list[str] | None = None,
    ) -> LeanResult:
        """空表达式集 → 失败结果; 否则批量编译并返回."""
        if not expressions:
            return _empty_lean_result(missing_msg)
        return self.verify_expression_dict(
            expressions, symbols=symbols, imports=imports, timeout=timeout
        )

    def _translate_expr(
        self, expr: sp.Expr | str, symbols: list[str] | None
    ) -> sp.Expr:
        """字符串先 sympify (用户符号优先), 再交给翻译器."""
        if isinstance(expr, str):
            return sp.sympify(expr, locals=_sympy_locals(symbols, user_wins=True))
        return expr

    def _apply_reflection_fix(
        self,
        code: str,
        result: LeanResult,
        symbols: list[str] | None,
        imports: list[str] | None,
        timeout: int,
        max_retries: int = 2,
    ) -> LeanResult:
        """编译失败时按规则修复并重试, 成功后把修复记录写进 stdout."""
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

    def _run_lean_source(self, source: str, timeout: int) -> LeanResult:
        """临时文件 + `lake env lean` 直接编译 (不加 --quiet, 以便捕获 #eval)."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            delete=False,
            dir=self._lean.project_path / "HuginnLean",
            encoding="utf-8",
        ) as f:
            f.write(source)
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
            expr = sp.sympify(
                expr, locals=_sympy_locals(symbols, user_wins=False, with_e=True)
            )

        return self._lean.eval_lean_code(
            _lean_def_line(name, expr, self._translator),
            imports=imports or [],
            timeout=timeout,
        )

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
        lines = [
            _lean_def_line(name, self._translate_expr(expr, symbols), self._translator)
            for name, expr in expressions.items()
        ]
        code = "\n\n".join(lines)

        result = self._lean.eval_lean_code(code, imports=imports or [], timeout=timeout)
        if result.success:
            return result
        return self._apply_reflection_fix(code, result, symbols, imports, timeout)

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
            if isinstance(val, int | float):
                expressions[f"tensorInv{key}"] = str(val)

        principal = symbolic_result.get("principal_values", [])
        for i, val in enumerate(principal):
            expressions[f"principalVal{i}"] = str(val)

        vm = symbolic_result.get("von_mises")
        if isinstance(vm, int | float):
            expressions["vonMises"] = str(vm)

        hydro = symbolic_result.get("hydrostatic_pressure")
        if isinstance(hydro, int | float):
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
        code = _derivative_source(
            self._translator, original, variable, expected, test_points
        )
        # Use lake env lean directly (no --quiet) so #eval output is captured
        return self._run_lean_source(code + "\n", timeout)

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

        # LU or Cholesky factor L / LU factor U
        for key, prefix in (("L", "laL"), ("U", "laU")):
            matrix = symbolic_result.get(key)
            if isinstance(matrix, list):
                _add_matrix_defs(expressions, matrix, prefix)

        # Solution vector / matrix-vector product result
        for key, prefix in (("solution", "laX"), ("result", "laRes")):
            vector = symbolic_result.get(key)
            if isinstance(vector, list):
                _add_vector_defs(expressions, vector, prefix)

        # Condition number
        _add_text_def(expressions, symbolic_result.get("cond_number"), "laCondNumber")

        return self._lean_defs(
            expressions,
            symbols,
            timeout,
            "No linear algebra data found",
            imports=["HuginnLean.NumericalLinearAlgebra"],
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

        _add_number_def(
            expressions, symbolic_result.get("fermi_energy"), "dftFermiEnergy"
        )
        _add_number_def(
            expressions, symbolic_result.get("fermi_wavevector"), "dftFermiWavevector"
        )
        _add_number_def(expressions, symbolic_result.get("dos"), "dftDOS")

        _dft_level_defs(expressions, symbolic_result.get("levels", []))
        _dft_band_defs(expressions, symbolic_result.get("band", []))

        _add_number_def(
            expressions, symbolic_result.get("xc_energy_density"), "dftXCEnergyDensity"
        )
        _add_number_def(
            expressions,
            symbolic_result.get("exchange_energy_density"),
            "dftExchangeEnergyDensity",
        )
        _add_number_def(
            expressions,
            symbolic_result.get("correlation_energy_density"),
            "dftCorrelationEnergyDensity",
        )

        return self._lean_defs(
            expressions,
            symbols,
            timeout,
            "No DFT data found",
            imports=["HuginnLean.DFT"],
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

        _add_number_table(
            expressions, symbolic_result, _THERMO_KEYS, "thermo", _title_join
        )

        return self._lean_defs(
            expressions,
            symbols,
            timeout,
            "No thermodynamics data found",
            imports=["HuginnLean.Thermodynamics"],
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
            if isinstance(val, int | float):
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

        _add_text_def(
            expressions, symbolic_result.get("energy_expression"), "unifiedEnergy"
        )
        _collect_unified_eq(
            expressions,
            symbolic_result.get("equations", {}),
            "unifiedEq",
            self._sanitize_lean_name,
        )

        return self._lean_defs(
            expressions,
            symbols,
            timeout,
            "No verifiable unified expressions found (algebraic only)",
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
