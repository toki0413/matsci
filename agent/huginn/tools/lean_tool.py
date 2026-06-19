"""Lean 4 Tool — formal proof verification for materials science.

Bridges symbolic computation (Phase 1) with proof assistance (Phase 2).
Allows the agent to:
  - Verify mathematical statements by compiling them in Lean 4
  - Check that existing theorems in the HuginnLean library compile
  - Translate SymPy expressions to Lean 4 and submit them for proof
  - Auto-verify SymbolicMathTool results via AutoLeanPipeline
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class LeanToolInput(BaseModel):
    action: str = Field(
        ...,
        description="build | verify | translate | prove_snippet | eval | auto_verify",
    )
    theorem_name: str | None = Field(
        default=None, description="Name of theorem to verify (for verify action)"
    )
    module: str = Field(
        default="HuginnLean.ContinuumMechanics", description="Lean module path"
    )
    lean_code: str | None = Field(
        default=None, description="Raw Lean 4 code snippet (for prove_snippet / eval)"
    )
    sympy_expression: str | None = Field(
        default=None, description="SymPy expression string (for translate action)"
    )
    symbols: list[str] = Field(
        default_factory=list, description="Symbol names for translation"
    )
    auto_verify_action: str | None = Field(
        default=None,
        description="Sub-action for auto_verify: constitutive | derivative | weak_form | eigenvalue | tensor_ops | solve | tensor_calculus | fem | linear_algebra | dft | thermodynamics | probability",
    )
    symbolic_result: dict | None = Field(
        default=None,
        description="JSON dict from SymbolicMathTool result.data (for auto_verify)",
    )
    original_expression: str | None = Field(
        default=None, description="Original expression for derivative verification"
    )
    variable: str | None = Field(
        default=None, description="Variable name for derivative verification"
    )
    expected_expression: str | None = Field(
        default=None, description="Expected derivative expression"
    )
    test_points: dict | None = Field(
        default=None,
        description='Test points for derivative verification, e.g. {"x": 2.0}',
    )


class LeanTool(HuginnTool):
    """Formal verification via Lean 4 proof assistant.

    Provides a gateway from computational materials science into
    mathematically-certified statements.
    """

    name = "lean_tool"
    description = (
        "Verify materials-science mathematics using the Lean 4 proof assistant. "
        "Actions: build (compile library), verify (check theorem exists & compiles), "
        "eval (execute Lean code and capture output), translate (SymPy → Lean 4), "
        "prove_snippet (compile ad-hoc Lean code), auto_verify (feed SymbolicMathTool results into Lean)."
    )
    input_schema = LeanToolInput

    def __init__(self):
        super().__init__()
        self._interface = None
        self._project_path = self._resolve_project_path()
        self._auto_pipeline = None

    def _resolve_project_path(self) -> Path | None:
        """Locate the HuginnLean project relative to this file."""
        candidates = [
            Path(__file__).parent.parent.parent / "lean" / "HuginnLean",
            Path.cwd() / "lean" / "HuginnLean",
            Path.cwd().parent / "lean" / "HuginnLean",
        ]
        for c in candidates:
            if (c / "lakefile.toml").exists():
                return c.resolve()
        return None

    def _get_interface(self):
        if self._interface is None:
            from huginn.lean.interface import LeanInterface

            if self._project_path is None:
                raise RuntimeError(
                    "HuginnLean project not found. Run from project root."
                )
            self._interface = LeanInterface(self._project_path)
        return self._interface

    def _get_auto_pipeline(self):
        if self._auto_pipeline is None:
            from huginn.lean.auto_pipeline import AutoLeanPipeline

            if self._project_path is None:
                raise RuntimeError(
                    "HuginnLean project not found. Run from project root."
                )
            self._auto_pipeline = AutoLeanPipeline(self._project_path)
        return self._auto_pipeline

    def is_read_only(self, args: LeanToolInput) -> bool:
        return True

    async def call(self, args: LeanToolInput, context: ToolContext) -> ToolResult:
        action = args.action.lower()

        try:
            if action == "build":
                return self._do_build(args)
            if action == "verify":
                return self._do_verify(args)
            if action == "prove_snippet":
                return self._do_prove_snippet(args)
            if action == "eval":
                return self._do_eval(args)
            if action == "translate":
                return self._do_translate(args)
            if action == "auto_verify":
                return self._do_auto_verify(args)

            return ToolResult(
                data=None, success=False, error=f"Unknown action: {args.action}"
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Lean tool error: {str(e)}"
            )

    def _do_build(self, args: LeanToolInput) -> ToolResult:
        interface = self._get_interface()
        result = interface.build(quiet=True)
        return ToolResult(
            data={
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": result.elapsed_seconds,
            },
            success=result.success,
            error=None if result.success else result.stderr,
        )

    def _do_verify(self, args: LeanToolInput) -> ToolResult:
        if not args.theorem_name:
            return ToolResult(
                data=None,
                success=False,
                error="theorem_name required for verify action",
            )
        interface = self._get_interface()
        result = interface.verify_theorem(args.theorem_name, module=args.module)
        return ToolResult(
            data={
                "theorem": args.theorem_name,
                "module": args.module,
                "verified": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
            success=result.success,
            error=None if result.success else result.stderr,
        )

    def _do_prove_snippet(self, args: LeanToolInput) -> ToolResult:
        if not args.lean_code:
            return ToolResult(
                data=None,
                success=False,
                error="lean_code required for prove_snippet action",
            )
        interface = self._get_interface()
        result = interface.run_lean_code(args.lean_code, timeout=60)
        return ToolResult(
            data={
                "verified": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": result.elapsed_seconds,
            },
            success=result.success,
            error=None if result.success else result.stderr,
        )

    def _do_eval(self, args: LeanToolInput) -> ToolResult:
        if not args.lean_code:
            return ToolResult(
                data=None, success=False, error="lean_code required for eval action"
            )
        interface = self._get_interface()
        imports = [args.module] if args.module else []
        result = interface.eval_lean_code(args.lean_code, imports=imports, timeout=60)
        return ToolResult(
            data={
                "success": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": result.elapsed_seconds,
            },
            success=result.success,
            error=None if result.success else result.stderr,
        )

    def _do_translate(self, args: LeanToolInput) -> ToolResult:
        if not args.sympy_expression:
            return ToolResult(
                data=None,
                success=False,
                error="sympy_expression required for translate action",
            )
        import sympy as sp

        from huginn.lean.sympy_to_lean import SymPyToLean

        sym_dict = {s: sp.Symbol(s) for s in args.symbols}
        local_dict = dict(sym_dict)
        local_dict.update(
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
        expr = sp.sympify(args.sympy_expression, locals=local_dict)
        translator = SymPyToLean()
        lean_str = translator.translate(expr)

        return ToolResult(
            data={
                "sympy_input": str(expr),
                "lean_output": lean_str,
            },
            success=True,
        )

    def _do_auto_verify(self, args: LeanToolInput) -> ToolResult:
        """Consume SymbolicMathTool results and auto-verify in Lean 4."""
        sub = (args.auto_verify_action or "").lower()
        pipe = self._get_auto_pipeline()
        sym = args.symbols or []

        if sub == "constitutive":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify constitutive",
                )
            result = pipe.verify_constitutive(args.symbolic_result, symbols=sym)
        elif sub == "derivative":
            # Support both explicit fields and symbolic_result dict from SymbolicMathTool
            original = args.original_expression
            variable = args.variable
            expected = args.expected_expression
            if args.symbolic_result and isinstance(args.symbolic_result, dict):
                original = (
                    original
                    or args.symbolic_result.get("input")
                    or args.symbolic_result.get("original_expression")
                )
                variable = variable or args.symbolic_result.get("variable")
                expected = (
                    expected
                    or args.symbolic_result.get("result")
                    or args.symbolic_result.get("expected_expression")
                )
            if not original or not variable or not expected:
                return ToolResult(
                    data=None,
                    success=False,
                    error="original_expression, variable, expected_expression required (or pass symbolic_result with input/variable/result)",
                )
            pts = args.test_points or {variable: 1.0}
            result = pipe.verify_derivative(
                original=original,
                variable=variable,
                expected=expected,
                test_points=pts,
            )
        elif sub == "weak_form":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify weak_form",
                )
            result = pipe.verify_weak_form(args.symbolic_result, symbols=sym)
        elif sub == "eigenvalue":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify eigenvalue",
                )
            result = pipe.verify_eigenvalue(args.symbolic_result, symbols=sym)
        elif sub == "tensor_ops":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify tensor_ops",
                )
            result = pipe.verify_tensor_ops(args.symbolic_result, symbols=sym)
        elif sub == "tensor_calculus":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify tensor_calculus",
                )
            result = pipe.verify_tensor_calculus(args.symbolic_result, symbols=sym)
        elif sub == "solve":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify solve",
                )
            result = pipe.verify_solve(args.symbolic_result, symbols=sym)
        elif sub == "fem":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify fem",
                )
            result = pipe.verify_fem(args.symbolic_result, symbols=sym)
        elif sub == "linear_algebra":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify linear_algebra",
                )
            result = pipe.verify_linear_algebra(args.symbolic_result, symbols=sym)
        elif sub == "dft":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify dft",
                )
            result = pipe.verify_dft(args.symbolic_result, symbols=sym)
        elif sub == "thermodynamics":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify thermodynamics",
                )
            result = pipe.verify_thermodynamics(args.symbolic_result, symbols=sym)
        elif sub == "probability":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify probability",
                )
            result = pipe.verify_probability(args.symbolic_result, symbols=sym)
        elif sub == "unified":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify unified",
                )
            result = pipe.verify_unified(args.symbolic_result, symbols=sym)
        elif sub == "discretization":
            if not args.symbolic_result:
                return ToolResult(
                    data=None,
                    success=False,
                    error="symbolic_result required for auto_verify discretization",
                )
            result = pipe.verify_discretization(args.symbolic_result, symbols=sym)
        else:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown auto_verify_action: {args.auto_verify_action}",
            )

        return ToolResult(
            data={
                "verified": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": result.elapsed_seconds,
            },
            success=result.success,
            error=None if result.success else result.stderr,
        )
