"""BourbakiTool._fallback_check_conservation 的 SymPy 升级回归.

锁住:
  * SymPy 能解析的方程: 残差 == 0 → verified True; 残差 != 0 → verified False
  * 含 unicode 算子 (∇· / ∂) 的方程 sympify 失败 → 回退子串启发式
  * 启发式: 同时有散度项和时间导数项才判守恒
"""
from __future__ import annotations

import asyncio

from huginn.tools.bourbaki_tool import BourbakiTool


def _check(equations: str):
    tool = BourbakiTool()
    # Lean 不可用时 call 走 _fallback_check_conservation; 强制关掉 Lean 路径
    tool._lean_available = False
    res = asyncio.run(
        tool.call(
            {"task": "check_conservation", "domain": "continuum_mechanics", "equations": equations},
            context=None,  # fallback 路径不碰 context
        )
    )
    # call 返回 BourbakiResult (pydantic); 统一成 dict 方便断言
    if hasattr(res, "model_dump"):
        return res.model_dump()
    return res


class TestSymPyConservationCheck:
    def test_balanced_equation_verified(self) -> None:
        # lhs - rhs == 0, SymPy 路径判守恒
        res = _check("x + y = x + y")
        assert res["verified"] is True
        assert "SymPy" in res["message"]

    def test_unbalanced_equation_rejected(self) -> None:
        res = _check("x + 1 = x")
        assert res["verified"] is False
        assert "SymPy" in res["message"]

    def test_unicode_continuity_falls_back_to_heuristic_verified(self) -> None:
        # ∇· 和 ∂/∂t 都在 → 启发式判 True
        res = _check("∇·J + ∂ρ/∂t = 0")
        assert res["verified"] is True
        assert "Heuristic" in res["message"]

    def test_unicode_divergence_only_rejected(self) -> None:
        # 只有散度项, 没有时间导数 → 启发式判 False
        res = _check("∇·J = 5")
        assert res["verified"] is False
        assert "Heuristic" in res["message"]
