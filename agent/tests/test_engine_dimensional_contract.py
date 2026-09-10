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
