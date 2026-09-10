"""引擎 × 量纲契约接线 —— 符号回归 & Bourbaki 把"学到的/推导的式子"接上量纲自证.

接入点(非学习, 沿用 coldstart_guards 契约层的 UnitRegistry):
  - symbolic_regression_tool._constraint_check 的 ``dimensional_check``: 由旧"语法空壳
    直接过"换成 ``external_validator.check_expression_dimensions`` —— 用 sympy 解析回归式 +
    DimensionalValidator 推断量纲, 与声明目标量纲对账.
  - bourbaki_tool._fallback_dimensional_analysis: 由裸 sympy 五基单位表换成契约层
    ``resolve_unit_dimension``, 支持复合单位(GPa / m/s²)并如实标 valid/unknown.
"""
from __future__ import annotations

import asyncio

from huginn.core_types import ToolContext
from huginn.research.external_validator import (
    check_expression_dimensions,
)

CTX = ToolContext(session_id="test", workspace=".")


# ═══════════════ S1: 契约层 check_expression_dimensions ═══════════════

def test_dimensional_check_kinetic_energy_consistent():
    """E = 0.5*m*v**2 → 量纲 J (M1·L2·T-2) 自洽. 非学习."""
    out = check_expression_dimensions(
        "0.5 * m * v**2",
        {"m": "kg", "v": "m/s"},
        "J",
    )
    assert out["ok"] is True
    assert out["inferred"] == out["expected"]


def test_dimensional_check_wrong_expected_unit_fails():
    """同一表达式却声明目标量纲为 'time' → 硬拒 (捕捉声明笔误)."""
    out = check_expression_dimensions(
        "0.5 * m * v**2",
        {"m": "kg", "v": "m/s"},
        "time",
    )
    assert out["ok"] is False


def test_dimensional_check_derived_velocity():
    """v = x / t → L1·T-1, 与 velocity 量纲一致."""
    out = check_expression_dimensions(
        "x / t",
        {"x": "m", "t": "s"},
        "velocity",
    )
    assert out["ok"] is True


def test_dimensional_check_unparsable_is_unknown_not_hard_blank():
    """不可解析表达式 → 如实报不可用(error 非空), 不偷偷判过."""
    out = check_expression_dimensions("rho * ((", {}, "kg/m3")
    assert out["ok"] is False
    assert out["error"]


# ═══════════════ S2: 符号回归 constraint_check 接线 ═══════════════

def _svr_run(expr, constraints):
    from huginn.tools.sci.symbolic_regression_tool import (
        SymbolicRegressionInput,
        SymbolicRegressionTool,
    )

    # constraint_check 需要 data 或 constraints['bounds'] 以构造采样网格; units 里的
    # feature 必须全部出现在 bounds —— meshgrid 会按 bounds 键数爆炸(21^n), 这里只给
    # 表达式实际用到的 feature 的 bounds, 避免 OOM. 量纲分支不依赖数值.
    c = dict(constraints or {})
    used = [s for s in ("m", "v", "x", "t", "a", "b") if s in expr]
    if not (c.get("bounds") or {}):
        c["bounds"] = {s: [0.1, 5.0] for s in used} or {"a": [0.1, 5.0]}

    inp = SymbolicRegressionInput(
        action="constraint_check",
        probe_expression=expr,
        constraints=c,
    )
    tool = SymbolicRegressionTool()
    return asyncio.run(tool.call(inp, CTX))


def test_svr_dimensional_check_matches_declared():
    """E = 0.5*m*v**2 with constraints(units→kg/m/s, target_unit J) → pass."""
    res = _svr_run(
        "0.5 * m * v**2",
        {
            "dimensional_check": True,
            "units": {"m": "kg", "v": "m/s"},
            "target_unit": "J",
        },
    )
    assert res.success is True
    check = next(c for c in res.data["checks"] if c["name"] == "dimensional_check")
    assert check["passed"] is True


def test_svr_dimensional_check_wrong_declared_fails():
    res = _svr_run(
        "0.5 * m * v**2",
        {
            "dimensional_check": True,
            "units": {"m": "kg", "v": "m/s"},
            "target_unit": "time",  # 声明笔误 → 硬拒
        },
    )
    check = next(c for c in res.data["checks"] if c["name"] == "dimensional_check")
    assert check["passed"] is False


def test_svr_dimensional_check_missing_units_is_not_blank_pass():
    """无完整单位声明 → 如实标 passed=False + 说明, 不硬判过."""
    res = _svr_run("a * b", {"dimensional_check": True})
    check = next(c for c in res.data["checks"] if c["name"] == "dimensional_check")
    assert check["passed"] is False


# ═══════════════ S3: Bourbaki 复合单位解析接线 ═══════════════

def test_bourbaki_dimensional_analysis_resolves_compound_units():
    """复合单位 GPa / kg·m³ 经契约层 UnitRegistry 解析 → all recognized."""
    from huginn.tools.bourbaki_tool import BourbakiInput, BourbakiTool

    tool = BourbakiTool()
    res = asyncio.run(tool.call(
        BourbakiInput(task="dimensional_analysis", domain="continuum_mechanics",
                      variables=[("E", "GPa"), ("rho", "kg/m3"), ("L", "m")]),
        CTX,
    ))
    assert res.success is True
    assert res.data["dimensional_match"] is True
    # E(GPa) 与 rho(kg/m3) 都解析出了非空量纲签名
    sigs = {v["name"]: v["dimension"] for v in res.data["data"]["variables"]}
    assert sigs["E"]
    assert sigs["rho"]
    assert sigs["L"] == "L1"


def test_bourbaki_dimensional_analysis_unknown_unit_false():
    """未识别单位 → dimensional_match=False, 如实暴露而非硬判对."""
    from huginn.tools.bourbaki_tool import BourbakiInput, BourbakiTool

    tool = BourbakiTool()
    res = asyncio.run(tool.call(
        BourbakiInput(task="dimensional_analysis", domain="mystery_domain",
                      variables=[("Q", "frobnicate")]),
        CTX,
    ))
    assert res.success is True
    assert res.data["dimensional_match"] is False


# ═══════════════ S4: Lean auto_verify unified 量纲前置自检 ═══════════════

def test_lean_pre_check_helper_dimensionally_consistent():
    """equations 是字符串值, 提供 unit_symbols+expected_units → 生成量纲自检条目."""
    from huginn.tools.lean_tool import _pre_lean_dimensional_check

    symbolic = {"equations": {"energy": "0.5 * m * v**2"}}
    checks = _pre_lean_dimensional_check(
        symbolic, {"m": "kg", "v": "m/s"}, {"energy": "J"}
    )
    assert len(checks) == 1
    assert checks[0]["name"] == "energy"
    assert checks[0]["ok"] is True


def test_lean_pre_check_helper_wrong_expected_defaults_not_fail():
    """方程未声明 expected_units → 各自带 error, 不静默判过; 空 expected_units → 不启用."""
    from huginn.tools.lean_tool import _pre_lean_dimensional_check

    symbolic = {"equations": {"a": "m * v"}}
    # expected_units 里没给方程 a 声明 → 记录 error 条目, 不静默通过
    checks = _pre_lean_dimensional_check(symbolic, {"m": "kg", "v": "m/s"},
                                         expected_units={"other": "J"})
    assert checks and checks[0]["name"] == "a"
    assert checks[0]["error"] == "no expected_units declared for this equation"
    # 完全没提供 unit_symbols/expected_units → 空列表, 不阻断
    assert _pre_lean_dimensional_check(symbolic, None, None) == []
    assert _pre_lean_dimensional_check(symbolic, {"m": "kg"}, {}) == []


def test_lean_pre_check_helper_eq_object_both_sides():
    """等式对象 {lhs, rhs} → 两侧分别量纲自检."""
    from huginn.tools.lean_tool import _pre_lean_dimensional_check

    symbolic = {"equations": {"f": {"lhs": "E/rho", "rhs": "x/t"}}}
    checks = _pre_lean_dimensional_check(
        symbolic,
        {"E": "N/m2", "rho": "kg/m3", "x": "m", "t": "s"},
        {"f": "1/s"},
    )
    assert len(checks) == 2  # lhs + rhs
    assert all(c["name"] in ("f.lhs", "f.rhs") for c in checks)


def test_lean_auto_verify_unified_disabled_by_default_passthrough():
    """不提供 unit_symbols/expected_units → dimension_checks 空, 行为不变(不破坏既有)."""
    from huginn.tools.lean_tool import LeanToolInput, _pre_lean_dimensional_check

    symbolic = {"equations": {"energy": "0.5 * m * v**2"}}
    inp = LeanToolInput(action="auto_verify", auto_verify_action="unified",
                        symbolic_result=dict(symbolic))
    # 无显式单位 → 前置自检为空, 不往 symbolic_result 里塞 dimension_checks
    d_checks = _pre_lean_dimensional_check(
        inp.symbolic_result, inp.unit_symbols, inp.expected_units
    )
    assert d_checks == []


# ═══════════════ S5: FEM 求解前量纲/合理性前置自检 ═══════════════

def test_fem_precheck_rejects_illegal_poisson_ratio():
    """nu 超 (-1, 0.5) 物理域 → ok=False (硬拒), 规避病态刚度阵."""
    from huginn.tools.fem.tool import FEMInput, _fem_dimensional_precheck

    args = FEMInput(action="static_linear", material={"E": 210e9, "nu": 0.9},
                    dims={"L": 1.0, "H": 0.05},
                    loads=[{"type": "point", "value": 100, "region": "right"}],
                    boundary_conditions=[{"region": "left", "dofs": [0, 1], "value": 0}])
    pre = _fem_dimensional_precheck(args)
    assert pre["ok"] is False
    assert any(c["name"] == "material.nu" for c in pre["checks"])


def test_fem_precheck_rejects_nonpositive_e():
    """E ≤ 0 → ok=False."""
    from huginn.tools.fem.tool import FEMInput, _fem_dimensional_precheck

    args = FEMInput(action="static_linear", material={"E": -5.0, "nu": 0.3},
                    dims={"L": 1.0, "H": 0.05},
                    loads=[{"type": "point", "value": 100, "region": "right"}],
                    boundary_conditions=[{"region": "left", "dofs": [0, 1], "value": 0}])
    assert _fem_dimensional_precheck(args)["ok"] is False


def test_fem_precheck_ok_consistent():
    """合法输入(标准悬臂梁) → 自检通过, 且静力弯曲刚度量纲自洽."""
    import pytest
    pytest.importorskip("skfem")
    from huginn.tools.fem.tool import FEMInput, _fem_dimensional_precheck

    args = FEMInput(action="static_linear", material={"E": 210e9, "nu": 0.3},
                    dims={"L": 1.0, "H": 0.05},
                    loads=[{"type": "point", "value": 100, "region": "right"}],
                    boundary_conditions=[{"region": "left", "dofs": [0, 1], "value": 0}])
    pre = _fem_dimensional_precheck(args)
    assert pre["ok"] is True
    bend = next(c for c in pre["checks"] if c["name"] == "bending_stiffness")
    assert bend["ok"] is True


# ═══════════════ S6: structural_analytical 解析求解量纲自检 ═══════════════

def test_structural_beam_modal_frequency_dimensional():
    """梁模态频率 ω∝β²√(EI/(ρAL⁴)) → 1/s, 解析式量纲自洽."""
    from huginn.tools.structural_analytical.tool import (
        StructuralAnalyticalInput,
        _structural_dimensional_precheck,
    )

    args = StructuralAnalyticalInput(
        action="beam_modal",
        beam={"youngs_modulus": 210e9, "poissons_ratio": 0.3,
              "length": 1.0, "second_moment": 1e-8, "density": 7850, "area": 0.01},
        boundary="simply_supported", n_modes=3,
    )
    pre = _structural_dimensional_precheck(args)
    assert pre["ok"] is True
    chk = next(c for c in pre["checks"] if c["name"] == "beam_modal_frequency")
    assert chk["ok"] is True
    assert chk["inferred"] == "T-1"


def test_structural_beam_buckling_load_dimensional():
    """Euler 屈曲临界载荷 P_cr=π²EI/L² → N, 解析式量纲自洽."""
    from huginn.tools.structural_analytical.tool import (
        StructuralAnalyticalInput,
        _structural_dimensional_precheck,
    )

    args = StructuralAnalyticalInput(
        action="beam_buckling",
        beam={"youngs_modulus": 210e9, "poissons_ratio": 0.3,
              "length": 2.0, "second_moment": 1e-8, "density": 7850, "area": 0.01},
        boundary="simply_supported",
    )
    pre = _structural_dimensional_precheck(args)
    assert pre["ok"] is True
    chk = next(c for c in pre["checks"] if c["name"] == "beam_euler_buckling_load")
    assert chk["ok"] is True
    assert "T-2" in chk["inferred"]  # N = kg·m/s²


def test_structural_shell_buckling_stress_dimensional():
    """Donnell 轴向临界应力 σ_cl=E·h/(R√(3(1-ν²))) → Pa, 解析式量纲自洽."""
    from huginn.tools.structural_analytical.tool import (
        StructuralAnalyticalInput,
        _structural_dimensional_precheck,
    )

    args = StructuralAnalyticalInput(
        action="shell_buckling",
        shell={"youngs_modulus": 70e9, "poissons_ratio": 0.33,
               "radius": 0.5, "length": 1.0, "thickness": 0.002, "density": 2700},
        theory="donnell", shell_load_type="axial",
    )
    pre = _structural_dimensional_precheck(args)
    assert pre["ok"] is True
    chk = next(c for c in pre["checks"] if c["name"] == "shell_axial_buckling_stress")
    assert chk["ok"] is True
    assert "M1" in chk["inferred"] and "L-1" in chk["inferred"]  # Pa = kg/(m·s²)
