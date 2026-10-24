"""张量类 action: tensor_ops / tensor_calculus / einstein_sum."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy as sp

from huginn.core_types import ToolResult

from ._parsers import parse_einstein_token, parse_symbols, safe_parse

if TYPE_CHECKING:
    from .tool import SymbolicMathInput


def tensor_ops(args: SymbolicMathInput) -> ToolResult:
    """连续介质力学常见张量操作."""
    sym_dict = parse_symbols(args.symbols, args.assumptions)

    if args.matrix:
        M = sp.Matrix(
            [
                [safe_parse(entry, sym_dict) for entry in row]
                for row in args.matrix
            ]
        )
        return ToolResult(
            data={
                "invariants": {
                    "I1": str(sp.trace(M)),
                    "I2": str((sp.trace(M) ** 2 - sp.trace(M**2)) / 2),
                    "I3": str(sp.det(M)),
                },
                "trace": str(sp.trace(M)),
                "determinant": str(sp.det(M)),
            },
            success=True,
        )

    expr = safe_parse(args.expression or "", sym_dict)
    results = {}

    if hasattr(expr, "eigenvals"):
        results["invariants"] = {
            "I1": str(sp.trace(expr)),
            "I2": str((sp.trace(expr) ** 2 - sp.trace(expr**2)) / 2),
            "I3": str(sp.det(expr)),
        }
    else:
        results["factored"] = str(sp.factor(expr))
        results["expanded"] = str(sp.expand(expr))

    return ToolResult(data=results, success=True)


def _tc_parse_inputs(
    args: SymbolicMathInput,
) -> tuple[str, list[float], str, ToolResult | None]:
    """解析公共入参; voigt 长度非法时返回错误结果."""
    operation = (args.expression or "invariants").lower()
    voigt = args.voigt_vector or []
    tensor_type = (args.tensor_type or "stress").lower()
    if len(voigt) not in (6, 21):
        return (
            operation,
            voigt,
            tensor_type,
            _tc_error(
                "voigt_vector must have 6 components (2nd-order) or 21 "
                "components (4th-order stiffness)"
            ),
        )
    return operation, voigt, tensor_type, None


def _tc_serialize(data: dict[str, Any]) -> ToolResult:
    """把成功分支的数据封装成 ToolResult."""
    return ToolResult(data=data, success=True)


def _tc_error(message: str) -> ToolResult:
    """把失败分支的错误封装成 ToolResult."""
    return ToolResult(data=None, success=False, error=message)


def _tc_unknown(operation: str) -> ToolResult:
    """未知 operation 的统一报错."""
    return _tc_error(f"Unknown tensor_calculus operation: {operation}")


def _tc_build_sigma(voigt: list[float]) -> sp.Matrix:
    """把 6 分量 Voigt 向量组装成 3×3 对称应力张量."""
    s11, s22, s33, s23, s13, s12 = (float(v) for v in voigt)
    return sp.Matrix(
        [
            [s11, s12, s13],
            [s12, s22, s23],
            [s13, s23, s33],
        ]
    )


def _tc_build_stiffness(voigt: list[float]) -> sp.Matrix:
    """把 21 分量 Voigt 向量映射成 6×6 对称刚度阵."""
    idx = 0
    stiffness = sp.zeros(6, 6)
    for i in range(6):
        for j in range(i, 6):
            stiffness[i, j] = float(voigt[idx])
            stiffness[j, i] = float(voigt[idx])
            idx += 1
    return stiffness


def _tc_invariants(
    sigma: sp.Matrix, tensor_type: str, voigt: list[float]
) -> ToolResult:
    """2 阶张量的三个不变量."""
    I1 = float(sigma.trace())
    I2 = float((sigma.trace() ** 2 - (sigma**2).trace()) / 2)
    I3 = float(sigma.det())
    return _tc_serialize(
        {
            "tensor_type": tensor_type,
            "voigt_vector": voigt,
            "invariants": {"I1": I1, "I2": I2, "I3": I3},
            "trace": I1,
            "determinant": I3,
        }
    )


def _tc_deviatoric(sigma: sp.Matrix, tensor_type: str) -> ToolResult:
    """应力张量的静水压力与偏量部分."""
    p = float(sigma.trace()) / 3.0
    dev = sigma - p * sp.eye(3)
    return _tc_serialize(
        {
            "tensor_type": tensor_type,
            "hydrostatic_pressure": p,
            "deviatoric_voigt": [
                float(dev[0, 0]),
                float(dev[1, 1]),
                float(dev[2, 2]),
                float(dev[1, 2]),
                float(dev[0, 2]),
                float(dev[0, 1]),
            ],
        }
    )


def _tc_principal(sigma: sp.Matrix, tensor_type: str) -> ToolResult:
    """按降序排列的主应力."""
    ev = sigma.eigenvals()
    principal = sorted([float(v) for v in ev], reverse=True)
    return _tc_serialize(
        {
            "tensor_type": tensor_type,
            "principal_values": principal,
            "max_principal": principal[0] if principal else None,
            "min_principal": principal[-1] if principal else None,
        }
    )


def _tc_von_mises(sigma: sp.Matrix, tensor_type: str) -> ToolResult:
    """von Mises 等效应力."""
    p = float(sigma.trace()) / 3.0
    s = sigma - p * sp.eye(3)
    vm = float(
        sp.sqrt(1.5 * sum(s[i, j] ** 2 for i in range(3) for j in range(3)))
    )
    return _tc_serialize(
        {
            "tensor_type": tensor_type,
            "von_mises": vm,
        }
    )


def _tc_rotate(
    sigma: sp.Matrix,
    tensor_type: str,
    rotation_matrix: list[list[float]] | None,
) -> ToolResult:
    """把应力张量旋转到新坐标系."""
    R = rotation_matrix
    if not R or len(R) != 3 or any(len(r) != 3 for r in R):
        return _tc_error("rotation_matrix must be 3×3 for rotate operation")
    Rm = sp.Matrix(R)
    sigma_rot = Rm * sigma * Rm.T
    return _tc_serialize(
        {
            "tensor_type": tensor_type,
            "rotated_voigt": [
                float(sigma_rot[0, 0]),
                float(sigma_rot[1, 1]),
                float(sigma_rot[2, 2]),
                float(sigma_rot[1, 2]),
                float(sigma_rot[0, 2]),
                float(sigma_rot[0, 1]),
            ],
        }
    )


def _tc_stiffness_invariants(
    stiffness: sp.Matrix, tensor_type: str
) -> ToolResult:
    """4 阶刚度张量的 Voigt 特征值."""
    ev = stiffness.eigenvals()
    eigenvalues = []
    for val, mult in ev.items():
        eigenvalues.extend([float(val)] * mult)
    eigenvalues.sort(reverse=True)
    return _tc_serialize(
        {
            "tensor_type": tensor_type,
            "voigt_eigenvalues": eigenvalues,
            "is_positive_definite": all(ev > 0 for ev in eigenvalues),
        }
    )


def _tc_apply_to_strain(
    stiffness: sp.Matrix,
    tensor_type: str,
    strain_voigt: list[list[float]] | None,
) -> ToolResult:
    """用刚度阵把 Voigt 应变映射成应力."""
    if (
        not strain_voigt
        or len(strain_voigt) != 1
        or len(strain_voigt[0]) != 6
    ):
        return _tc_error(
            "rotation_matrix must contain [ε11, ε22, ε33, 2ε23, 2ε13, 2ε12] "
            "for apply_to_strain"
        )
    eps = sp.Matrix(strain_voigt[0])
    sigma_voigt = stiffness * eps
    return _tc_serialize(
        {
            "tensor_type": tensor_type,
            "stress_voigt": [float(sigma_voigt[i]) for i in range(6)],
        }
    )


def tensor_calculus(args: SymbolicMathInput) -> ToolResult:
    """连续介质力学张量微积分操作."""
    # einstein_sum 子动作走独立分支，不需要 voigt 向量
    if (args.sub_action or "").lower() == "einstein_sum":
        return einstein_sum(args)

    operation, voigt, tensor_type, error = _tc_parse_inputs(args)
    if error is not None:
        return error

    if len(voigt) == 6:
        sigma = _tc_build_sigma(voigt)
        if operation == "invariants":
            return _tc_invariants(sigma, tensor_type, voigt)
        if operation == "deviatoric":
            return _tc_deviatoric(sigma, tensor_type)
        if operation == "principal":
            return _tc_principal(sigma, tensor_type)
        if operation == "von_mises":
            return _tc_von_mises(sigma, tensor_type)
        if operation == "rotate":
            return _tc_rotate(sigma, tensor_type, args.rotation_matrix)
        return _tc_unknown(operation)

    stiffness = _tc_build_stiffness(voigt)
    if operation == "invariants":
        return _tc_stiffness_invariants(stiffness, tensor_type)
    if operation == "apply_to_strain":
        return _tc_apply_to_strain(
            stiffness, tensor_type, args.rotation_matrix
        )
    return _tc_unknown(operation)


def einstein_sum(args: SymbolicMathInput) -> ToolResult:
    """按 Einstein 求和约定对带指标的张量表达式求和并化简."""
    from sympy.tensor.indexed import IndexedBase

    expr_str = args.expression or ""
    if not expr_str.strip():
        return ToolResult(
            data=None,
            success=False,
            error="expression is required for einstein_sum",
        )

    tokens = expr_str.split()
    parsed: list[tuple[str, list[str], list[str]]] = []
    for tok in tokens:
        p = parse_einstein_token(tok)
        if p is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"无法解析的张量 token: {tok}",
            )
        parsed.append(p)

    # 汇总每个指标的出现情况: {idx: [("upper"|"lower", tensor_pos), ...]}
    occurrences: dict[str, list[tuple[str, int]]] = {}
    for tpos, (_name, upper, lower) in enumerate(parsed):
        for idx in upper:
            occurrences.setdefault(idx, []).append(("upper", tpos))
        for idx in lower:
            occurrences.setdefault(idx, []).append(("lower", tpos))

    bad = [idx for idx, occ in occurrences.items() if len(occ) > 2]
    if bad:
        return ToolResult(
            data=None,
            success=False,
            error=f"指标出现超过两次，违反 Einstein 约定: {bad}",
        )

    if args.sum_indices:
        sum_indices = list(args.sum_indices)
    else:
        sum_indices = [
            idx for idx, occ in occurrences.items() if len(occ) == 2
        ]

    free_indices = [
        idx
        for idx, occ in occurrences.items()
        if len(occ) == 1 and idx not in sum_indices
    ]

    if args.indices:
        declared = set(args.indices)
        actual = set(free_indices)
        if declared != actual:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"自由指标不一致: 声明 {sorted(declared)}, "
                    f"实际 {sorted(actual)}"
                ),
            )
        free_indices = list(args.indices)

    metric_used = False
    metric_factors: list[Any] = []
    if args.metric:
        g_name = "g"
        g_entries = None
        if isinstance(args.metric, dict):
            g_name = args.metric.get("name", "g")
            g_entries = (
                args.metric.get("components")
                or args.metric.get("g")
                or args.metric.get(g_name)
            )
        if g_entries is not None:
            g_base = IndexedBase(g_name)
            for idx in sum_indices:
                occ = occurrences.get(idx, [])
                positions = [p for p, _ in occ]
                if len(positions) == 2 and positions[0] == positions[1]:
                    idx_sym = sp.Symbol(idx, integer=True)
                    partner = sp.Symbol(f"{idx}p", integer=True)
                    metric_factors.append(g_base[idx_sym, partner])
                    metric_used = True

    idx_symbols: dict[str, sp.Symbol] = {
        idx: sp.Symbol(idx, integer=True) for idx in occurrences
    }

    bases: dict[str, IndexedBase] = {}
    terms: list[Any] = []
    for name, upper, lower in parsed:
        if name not in bases:
            bases[name] = IndexedBase(name)
        all_idx = [idx_symbols[i] for i in upper + lower]
        if all_idx:
            terms.append(bases[name][tuple(all_idx)])
        else:
            terms.append(sp.Symbol(name))

    expr: sp.Expr = sp.Integer(1)
    for t in terms:
        expr = expr * t
    for gf in metric_factors:
        expr = expr * gf

    implicit_product = str(expr)

    N = sp.Symbol("N", integer=True, positive=True)
    sum_expr = expr
    for idx in sum_indices:
        sym = idx_symbols.get(idx, sp.Symbol(idx, integer=True))
        sum_expr = sp.Sum(sum_expr, (sym, 0, N - 1))

    try:
        evaluated = sum_expr.doit()
    except Exception:
        evaluated = sum_expr

    before = str(evaluated)

    if args.simplify:
        try:
            after_expr = sp.simplify(evaluated)
        except Exception:
            after_expr = evaluated
    else:
        after_expr = evaluated
    after = str(after_expr)

    parsed_summary = [
        {"tensor": name, "upper": upper, "lower": lower}
        for name, upper, lower in parsed
    ]

    return ToolResult(
        data={
            "input": expr_str,
            "result": after,
            "result_latex": sp.latex(after_expr),
            "implicit_product": implicit_product,
            "free_indices": free_indices,
            "sum_indices": sum_indices,
            "before_simplify": before,
            "after_simplify": after,
            "parsed_tensors": parsed_summary,
            "metric_used": metric_used,
        },
        success=True,
    )
