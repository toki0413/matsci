"""Quickstart 离线冒烟 fixture —— 被 .github/workflows/audit-regression.yml (G38) 引用.

只覆盖 quickstart 里**无需 LLM / 无需 Lean** 的路径:
包可导入 + 符号推导→验证的接线真的通.

刻意不 mock、不假报: 每项断言直接打在真实工具返回值上; 任一失败 -> 退出码 1.
输出约定 (CI 用 `grep -E "Results:.*passed"` 判定):

    Results: <passed> passed, <failed> failed

契约: 该文件必须保持 tracked (曾在 agent/.gitignore 被误忽略, 导致 G38 步骤
恒失败 —— 一个"宣称有 fixture 但仓库里没有"的接线断裂).
"""
from __future__ import annotations

import asyncio
import sys
import traceback
from collections.abc import Callable

from huginn.core_types import ToolContext
from huginn.tools.symbolic_math_tool import SymbolicMathInput, SymbolicMathTool

_CTX = ToolContext(session_id="quickstart-demo", workspace=".")


def _weak_form(**kwargs: object) -> object:
    tool = SymbolicMathTool()
    return asyncio.run(tool.call(SymbolicMathInput(**kwargs), _CTX))


def _check_package_imports() -> None:
    import huginn  # noqa: F401
    import huginn.cli  # noqa: F401

    assert huginn.__file__, "huginn 包无可解析位置"


def _check_weak_form_1d() -> None:
    res = _weak_form(
        action="weak_form",
        expression="-laplacian(u)",
        symbols=["u", "v", "x"],
        target="derivation",
    )
    assert res.success, res.error
    assert "diffusion" in res.data["weak_form_terms"], res.data


def _check_weak_form_verification() -> None:
    res = _weak_form(
        action="weak_form", symbols=["u", "v", "x", "y"], target="verification"
    )
    assert res.success, res.error
    assert res.data["verified_symbolically"] is True, res.data


def _check_heat_conduction() -> None:
    res = _weak_form(
        action="weak_form",
        symbols=["u", "v", "x", "y", "k", "f"],
        target="heat_conduction",
    )
    assert res.success, res.error
    assert "diffusion" in res.data["weak_form_terms"], res.data


CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("package imports", _check_package_imports),
    ("weak form 1D derivation", _check_weak_form_1d),
    ("weak form symbolic verification", _check_weak_form_verification),
    ("weak form heat conduction", _check_heat_conduction),
]


def main() -> int:
    passed = 0
    failed = 0
    for name, check in CHECKS:
        try:
            check()
        except Exception:
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"[ OK ] {name}")

    print(f"Results: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
