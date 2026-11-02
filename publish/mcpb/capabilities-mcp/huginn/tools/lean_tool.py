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

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool


def _pre_lean_dimensional_check(
    symbolic_result: dict | None, unit_symbols: dict | None, expected_units: dict | None
) -> list[dict]:
    """对 auto_verify unified 的方程串做量纲前置自检(契约层, 可选启用, 非学习).

    - ``symbolic_result["equations"]`` 中每个**字符串值**的方程(或 ``{"lhs","rhs"}``
      等式对象的两侧)先过 ``check_expression_dimensions``(sympy + UnitRegistry 推断);
    - 仅当调用方提供了 ``unit_symbols``({符号: 单位标签/SI}) 与 ``expected_units`` 才启用,
      否则返回空列表 → 完全不阻断 Lean 编译, 行为与现状一致;
    - 量纲引擎/表达式不可解析 → 如实记 error(量纲未知), 不硬判失败(诚实边界).

    Returns: 每个被检查的方程一条 {name/expr, inferred, expected, ok, error}.
    """
    from huginn.research.external_validator import check_expression_dimensions

    if not unit_symbols or not expected_units:
        return []
    equations = (symbolic_result or {}).get("equations") or {}
    if not isinstance(equations, dict) or not equations:
        return []

    checks: list[dict] = []
    for name, value in equations.items():
        # 等式对象 {lhs, rhs} → 两侧分别查; 普通字符串 → 整式查
        exprs = _equation_exprs(name, value)
        for label, expr in exprs:
            expected = expected_units.get(name) or expected_units.get(label)
            if not expected:
                checks.append(
                    {
                        "name": label,
                        "expr": expr,
                        "ok": False,
                        "error": "no expected_units declared for this equation",
                    }
                )
                continue
            checks.append(
                {
                    "name": label,
                    "expr": expr,
                    **check_expression_dimensions(expr, unit_symbols, expected),
                }
            )
    return checks


def _equation_exprs(name: str, value) -> list[tuple[str, str]]:
    """等式对象 {lhs, rhs} → 两侧分别查; 普通字符串 → 整式查."""
    exprs = []
    if isinstance(value, dict):
        if value.get("lhs") is not None:
            exprs.append((f"{name}.lhs", str(value["lhs"])))
        if value.get("rhs") is not None:
            exprs.append((f"{name}.rhs", str(value["rhs"])))
    elif isinstance(value, str):
        exprs.append((name, value))
    return exprs


# auto_verify 子动作 → AutoLeanPipeline 上的验证器方法名.
# 未列出的 (derivative / unified) 有专属预处理, 见 LeanTool 的同名私有方法.
_SIMPLE_VERIFIERS: dict[str, str] = {
    "constitutive": "verify_constitutive",
    "weak_form": "verify_weak_form",
    "eigenvalue": "verify_eigenvalue",
    "tensor_ops": "verify_tensor_ops",
    "tensor_calculus": "verify_tensor_calculus",
    "solve": "verify_solve",
    "fem": "verify_fem",
    "linear_algebra": "verify_linear_algebra",
    "dft": "verify_dft",
    "thermodynamics": "verify_thermodynamics",
    "probability": "verify_probability",
    "discretization": "verify_discretization",
}


def _missing_symbolic_result(action: str | None) -> ToolResult:
    """缺 symbolic_result 时统一的失败回执 (文案按子动作区分, 便于调用方纠错)."""
    return ToolResult(
        data=None,
        success=False,
        error=f"symbolic_result required for auto_verify {action}",
    )


class LeanToolInput(BaseModel):
    action: str = Field(
        ...,
        description="build | verify | translate | prove_snippet | eval | auto_verify | constitutive",
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
    unit_symbols: dict | None = Field(
        default=None,
        description="Optional {symbol: unit-tag-or-SI} map for pre-Lean dimensional self-check (auto_verify unified)",
    )
    expected_units: dict | None = Field(
        default=None,
        description="Optional {equation_name_or_variable: expected unit} map for pre-Lean dimensional self-check",
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
    # --- constitutive / variational_principle ---
    sub_action: str | None = Field(
        default=None,
        description="constitutive 下的子动作，目前支持 variational_principle",
    )
    lagrangian: str | None = Field(
        default=None,
        description="拉氏量 L(q, q_dot, t) 的符号表达式，如 1/2*m*v**2 - 1/2*k*x**2",
    )
    coordinates: list[str] = Field(
        default_factory=list, description="广义坐标列表，如 ['x'] 或 ['q1','q2']"
    )
    velocities: list[str] | None = Field(
        default=None,
        description="广义速度列表；不提供则自动按 {coord}_dot 命名（单自由度也接受 v）",
    )
    time_symbol: str = Field(default="t", description="时间变量名")
    claimed_eom: str | None = Field(
        default=None,
        description="用户声明的 Euler-Lagrange 方程，可带 = 号或不带（默认 = 0）",
    )
    return_lean: bool = Field(
        default=False, description="是否同时输出可粘贴到 Lean 4 的命题文本"
    )


class LeanTool(HuginnTool):
    """Formal verification via Lean 4 proof assistant.

    Provides a gateway from computational materials science into
    mathematically-certified statements.
    """

    name = "lean_tool"
    category = "core"
    description = (
        "Verify scientific mathematics using the Lean 4 proof assistant. "
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
            if action == "constitutive":
                return self._do_constitutive(args)

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

        if sub == "derivative":
            return self._auto_verify_derivative(args, pipe, sym)
        if sub == "unified":
            return self._auto_verify_unified(args, pipe, sym)

        verifier = _SIMPLE_VERIFIERS.get(sub)
        if verifier is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown auto_verify_action: {args.auto_verify_action}",
            )
        if not args.symbolic_result:
            return _missing_symbolic_result(sub)
        result = getattr(pipe, verifier)(args.symbolic_result, symbols=sym)

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

    def _auto_verify_derivative(
        self, args: LeanToolInput, pipe, sym: list
    ) -> ToolResult:
        """derivative 子动作: 支持显式字段, 也支持 SymbolicMathTool 的 symbolic_result."""
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

    def _auto_verify_unified(self, args: LeanToolInput, pipe, sym: list) -> ToolResult:
        """unified 子动作: 先做可选量纲前置自检, 再交给 Lean 统一验证器."""
        if not args.symbolic_result:
            return _missing_symbolic_result("unified")
        # 量纲前置自检(可选): 仅当提供 unit_symbols/expected_units 时启用, 否则不阻断.
        dimension_checks = _pre_lean_dimensional_check(
            args.symbolic_result, args.unit_symbols, args.expected_units
        )
        if dimension_checks:
            args.symbolic_result["dimension_checks"] = dimension_checks
        result = pipe.verify_unified(args.symbolic_result, symbols=sym)
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

    # ------------------------------------------------------------------
    # constitutive 动作：本构方程 / 变分原理相关的符号化验证
    # ------------------------------------------------------------------

    def _do_constitutive(self, args: LeanToolInput) -> ToolResult:
        """constitutive 动作分发器，按 sub_action 调对应验证器。"""
        sub = (args.sub_action or "").lower()
        if sub == "variational_principle":
            return self._run_variational_principle(args)
        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown constitutive sub_action: {args.sub_action}",
        )

    def _run_variational_principle(self, args: LeanToolInput) -> ToolResult:
        """用 SymPy 从拉氏量推 Euler-Lagrange 方程，跟用户声明的 EOM 比对。

        核心思路：
          1. 坐标 q_i 映射成 sympy Function(q_i)(t)，速度 v_i 映射成普通 Symbol
          2. 解析 L 后把 v_i 替换成 d q_i/dt，让 L 只依赖 q_i(t) 和 t
          3. 对每个 q_i 算 dL/dq_i - d/dt(dL/dv_i) = 0
          4. 跟 claimed_eom 做符号等价检查（允许差一个整体符号）
        """
        import sympy as sp

        err = _vp_validate_args(args)
        if err is not None:
            return err

        derived_str, local_dict, coords, derived_eoms, err = _vp_derive_equations(
            args, sp
        )
        if err is not None:
            return err

        claimed_expr, claimed_norm_str, err = _vp_parse_claimed(args, local_dict)
        if err is not None:
            return err

        is_consistent, difference_str = _vp_check_consistency(
            coords,
            derived_eoms,
            claimed_expr,
        )
        lean_stmt = _vp_lean_stmt(args, derived_eoms)

        data = {
            "derived_eom": derived_str,
            "claimed_eom_normalized": claimed_norm_str,
            "is_consistent": is_consistent,
            "difference": difference_str,
        }
        if lean_stmt is not None:
            data["lean_statement"] = lean_stmt

        return ToolResult(data=data, success=True)

    # ------------------------------------------------------------------
    # 同步入口：方便脚本和测试直接调用，不用起 asyncio 事件循环
    # ------------------------------------------------------------------

    def run(self, args: dict) -> ToolResult:
        """同步包装 call，给脚本和单测用。

        args 是原始 dict，内部会转成 LeanToolInput。
        """
        import asyncio

        parsed = self.input_schema(**args)
        ctx = ToolContext(session_id="sync", workspace=".")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.call(parsed, ctx))
        # 已经在 async 环境里，丢到新线程跑，避免阻塞当前 loop
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.call(parsed, ctx)).result()


# ── variational_principle 分阶段 helper ──────────────────────────────


def _vp_validate_args(args) -> ToolResult | None:
    """校验必填字段."""
    if not args.lagrangian:
        return ToolResult(data=None, success=False, error="lagrangian 不能为空")
    if not args.coordinates:
        return ToolResult(data=None, success=False, error="coordinates 不能为空")
    return None


def _vp_resolve_velocities(args, coords: list[str]):
    """解析速度符号名，返回 (vel_names, error)."""
    if args.velocities:
        if len(args.velocities) != len(coords):
            return None, ToolResult(
                data=None,
                success=False,
                error="velocities 长度必须和 coordinates 一致",
            )
        return list(args.velocities), None

    vel_names = [f"{c}_dot" for c in coords]
    if len(coords) == 1:
        auto_name = f"{coords[0]}_dot"
        if auto_name not in args.lagrangian and "v" in args.lagrangian:
            vel_names = ["v"]
    return vel_names, None


def _vp_derive_equations(args, sp):
    """解析拉氏量并推导 EOM, 返回 (derived_str, local_dict, coords, eoms, err)."""
    coords = list(args.coordinates)
    t_name = args.time_symbol or "t"
    t = sp.Symbol(t_name)

    vel_names, err = _vp_resolve_velocities(args, coords)
    if err is not None:
        return None, None, None, None, err

    # 坐标用 Function，速度用 Symbol（方便先对 v 求偏导再替换）
    coord_funcs = [sp.Function(c)(t) for c in coords]
    vel_syms = [sp.Symbol(v) for v in vel_names]

    local_dict = _vp_local_dict(
        t_name,
        t,
        coords,
        coord_funcs,
        vel_names,
        vel_syms,
    )

    # 解析拉氏量
    try:
        L = sp.sympify(args.lagrangian, locals=local_dict)
    except Exception as e:
        return (
            None,
            None,
            None,
            None,
            ToolResult(data=None, success=False, error=f"拉氏量解析失败: {e}"),
        )

    # 把速度符号替换成 d(coord)/dt
    subs_map = {v: sp.diff(cf, t) for v, cf in zip(vel_syms, coord_funcs)}
    L_sub = L.subs(subs_map)

    derived_eoms = _vp_derive_eoms(L_sub, L, t, coord_funcs, vel_syms, subs_map)
    derived_str = _vp_format_derived(coords, derived_eoms)
    return derived_str, local_dict, coords, derived_eoms, None


def _vp_local_dict(
    t_name: str,
    t,
    coords: list[str],
    coord_funcs: list,
    vel_names: list[str],
    vel_syms: list,
) -> dict:
    """构造 sympify 的 local_dict（坐标/速度/常用数学函数）."""
    import sympy as sp

    local_dict = {t_name: t}
    for name, func in zip(coords, coord_funcs):
        local_dict[name] = func
    for name, sym in zip(vel_names, vel_syms):
        local_dict[name] = sym
    # 数学函数和常数，方便 claimed_eom 里写 diff(...)
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
            "diff": sp.diff,
        }
    )
    return local_dict


def _vp_derive_eoms(
    L_sub, L, t, coord_funcs: list, vel_syms: list, subs_map: dict
) -> list:
    """逐个广义坐标推 EOM: dL/dq - d/dt(dL/dq_dot) = 0."""
    import sympy as sp

    derived_eoms = []
    for cf, vs in zip(coord_funcs, vel_syms):
        dL_dq = sp.diff(L_sub, cf)
        # 先对速度符号求偏导，再替换成 dx/dt，最后对 t 求全导数
        dL_dvdot = sp.diff(L, vs).subs(subs_map)
        ddt_dL_dvdot = sp.diff(dL_dvdot, t)
        eom = sp.simplify(dL_dq - ddt_dL_dvdot)
        derived_eoms.append(eom)
    return derived_eoms


def _vp_format_derived(coords: list[str], derived_eoms: list) -> str:
    """把推导出的 EOM 列表拼成字符串."""
    if len(derived_eoms) == 1:
        return str(derived_eoms[0])
    return "; ".join(f"{c}: {e}" for c, e in zip(coords, derived_eoms))


def _vp_parse_claimed(args, local_dict: dict):
    """解析用户声明的 EOM，归一化成 lhs - rhs = 0. 返回 (expr, str, error)."""
    import sympy as sp

    claimed_expr = None
    claimed_norm_str = None
    if not args.claimed_eom:
        return claimed_expr, claimed_norm_str, None
    raw = args.claimed_eom.strip()
    try:
        if "=" in raw:
            lhs_str, rhs_str = raw.split("=", 1)
            lhs = sp.sympify(lhs_str, locals=local_dict)
            rhs = sp.sympify(rhs_str, locals=local_dict)
            claimed_expr = sp.simplify(lhs - rhs)
        else:
            claimed_expr = sp.sympify(raw, locals=local_dict)
        claimed_norm_str = f"{claimed_expr} = 0"
    except Exception as e:
        return (
            None,
            None,
            ToolResult(
                data=None,
                success=False,
                error=f"claimed_eom 解析失败: {e}",
            ),
        )
    return claimed_expr, claimed_norm_str, None


def _vp_check_consistency(coords: list[str], derived_eoms: list, claimed_expr):
    """符号等价检查（允许差一个整体符号）。返回 (is_consistent, difference)."""
    import sympy as sp

    is_consistent = None
    difference_str = None
    if claimed_expr is None:
        return is_consistent, difference_str

    if len(derived_eoms) == 1:
        d = derived_eoms[0]
        diff_minus = sp.simplify(d - claimed_expr)
        diff_plus = sp.simplify(d + claimed_expr)
        if diff_minus == 0:
            is_consistent = True
            difference_str = "0"
        elif diff_plus == 0:
            is_consistent = True
            difference_str = "0 (相差整体符号，方程等价)"
        else:
            is_consistent = False
            difference_str = str(diff_minus)
        return is_consistent, difference_str

    # 多自由度：逐条比对
    per_coord = []
    all_ok = True
    for c, d in zip(coords, derived_eoms):
        dm = sp.simplify(d - claimed_expr)
        dp = sp.simplify(d + claimed_expr)
        if dm == 0:
            per_coord.append(f"{c}: 0")
        elif dp == 0:
            per_coord.append(f"{c}: 0 (差符号)")
        else:
            per_coord.append(f"{c}: {dm}")
            all_ok = False
    return all_ok, "; ".join(per_coord)


def _vp_lean_stmt(args, derived_eoms: list) -> str | None:
    """可选：Lean 4 命题文本（不真编译，只给陈述）."""
    if not args.return_lean:
        return None
    if len(derived_eoms) == 1:
        return f"theorem variational_principle : {derived_eoms[0]} = 0 := by sorry"
    conjuncts = " ∧ ".join(f"({e} = 0)" for e in derived_eoms)
    return f"theorem variational_principle : {conjuncts} := by sorry"
