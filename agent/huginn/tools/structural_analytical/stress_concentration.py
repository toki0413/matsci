"""应力集中系数解析求解.

按缺口类型分发, 用 Peterson 解析拟合或经典弹性力学解.

公式参考:
- Peterson《Stress Concentration Factors》3rd Ed. (各种缺口 Kt 图表)
- Timoshenko & Goodier《Theory of Elasticity》§7.7 (Inglis 椭圆孔解)
- Pilkey《Peterson's Stress Concentration Factors》3rd Ed. (拟合公式)
"""

from __future__ import annotations

import math
from typing import Any

from huginn.types import ToolResult


def stress_concentration(args: Any) -> ToolResult:
    """按 notch_type 算应力集中系数 Kt."""
    notch_type = args.notch_type
    geom = args.notch_geometry
    load = args.notch_load

    if notch_type == "hole":
        return _kt_hole(geom, load)
    if notch_type == "elliptical_hole":
        return _kt_elliptical_hole(geom, load)
    if notch_type in ("fillet", "groove", "shoulder"):
        return _kt_fillet_groove(notch_type, geom, load)

    return ToolResult(
        data=None,
        success=False,
        error=f"stress_concentration: unknown notch_type={notch_type}",
    )


def _kt_hole(geom: dict[str, float], load: str) -> ToolResult:
    """圆孔应力集中. Kt=3 (单向拉伸), 弯曲 Kt≈2.25."""
    d = float(geom.get("d", 0.0))  # 孔径
    D = float(geom.get("D", 0.0))  # 板宽

    if load == "tension":
        # 无限大板: Kt=3 (经典解). 有限宽板用 Howland 修正.
        if D > 0 and d > 0 and d < D:
            # Howland 级数拟合 (Peterson Fig.4.23):
            # Kt = 3.0 - 3.14·(d/D) + 3.667·(d/D)² - 1.527·(d/D)³
            r = d / D
            Kt = 3.0 - 3.14 * r + 3.667 * r**2 - 1.527 * r**3
        else:
            Kt = 3.0  # 无限大板
    elif load == "bending":
        # 弯曲: Kt ≈ 2.25 (无限大板, Peterson Fig.4.25)
        if D > 0 and d > 0 and d < D:
            r = d / D
            Kt = 2.25 - 2.0 * r + 1.5 * r**2
        else:
            Kt = 2.25
    elif load == "torsion":
        # 圆轴横向孔受扭: Kt ≈ 1.7-2.0 (简化取 1.8)
        Kt = 1.8
    else:
        return ToolResult(
            data=None, success=False,
            error=f"_kt_hole: load={load} not supported (use tension/bending/torsion)",
        )

    return _build_result(Kt, notch_type="hole", load=load, geom=geom)


def _kt_elliptical_hole(geom: dict[str, float], load: str) -> ToolResult:
    """椭圆孔应力集中. Inglis 解: Kt = 1 + 2(a/b)."""
    a = float(geom.get("a", 0.0))  # 半长轴 (垂直于载荷方向)
    b = float(geom.get("b", 0.0))  # 半短轴 (平行于载荷方向)

    if a <= 0 or b <= 0:
        return ToolResult(
            data=None, success=False,
            error="elliptical_hole requires 'a' (>0) and 'b' (>0) in notch_geometry",
        )

    # Inglis 解: 沿载荷方向拉, 孔端点应力集中
    # a 是垂直于载荷的半轴, b 是平行载荷的半轴
    Kt = 1.0 + 2.0 * (a / b)

    return _build_result(Kt, notch_type="elliptical_hole", load=load, geom=geom)


def _kt_fillet_groove(
    notch_type: str, geom: dict[str, float], load: str
) -> ToolResult:
    """圆角/凹槽/轴肩应力集中. Peterson 拟合.

    notch_type: fillet (平板圆角), groove (轴上环槽), shoulder (轴肩)
    geom: {d (小端直径/宽度), D (大端直径/宽度), r (圆角半径)}
    """
    d = float(geom.get("d", 0.0))
    D = float(geom.get("D", 0.0))
    r = float(geom.get("r", 0.0))

    if d <= 0 or D <= 0 or r <= 0 or d >= D:
        return ToolResult(
            data=None,
            success=False,
            error=(
                f"{notch_type} requires d>0, D>0, r>0, d<D in notch_geometry; "
                f"got d={d}, D={D}, r={r}"
            ),
        )

    if load == "torsion" and notch_type == "groove":
        # 圆轴环槽受扭: Kt ≈ 1 + √(r/d)·(D/d - 1)^0.5 (简化)
        Kt = 1.0 + math.sqrt(r / d) * math.sqrt(D / d - 1.0)
    elif load == "torsion" and notch_type == "shoulder":
        Kt = 1.0 + 0.8 * math.sqrt(D / d - 1.0) * math.sqrt(d / r)
    else:
        # 拉伸/弯曲: Peterson 简化拟合
        # fillet: Kt ≈ 1 + 2·√(D/d - 1) / √(2·r/d)
        # groove: Kt ≈ 1 + 2·√(D/d - 1) / √(r/d)
        if notch_type == "fillet":
            Kt = 1.0 + 2.0 * math.sqrt(D / d - 1.0) / math.sqrt(2.0 * r / d)
        elif notch_type == "groove":
            Kt = 1.0 + 2.0 * math.sqrt(D / d - 1.0) / math.sqrt(r / d)
        elif notch_type == "shoulder":
            # 轴肩: 介于 fillet 和 groove 之间
            Kt = 1.0 + 2.0 * math.sqrt(D / d - 1.0) / math.sqrt(1.5 * r / d)
        else:
            return ToolResult(
                data=None, success=False,
                error=f"_kt_fillet_groove: unknown notch_type={notch_type}",
            )

    return _build_result(Kt, notch_type=notch_type, load=load, geom=geom)


def _build_result(
    Kt: float,
    notch_type: str,
    load: str,
    geom: dict[str, float],
) -> ToolResult:
    """组装返回, 顺便算最大应力."""
    # 最大应力 = Kt × 名义应力. 名义应力由调用方按载荷和截面算.
    # 这里只返回 Kt, 附带参考公式.
    return ToolResult(
        data={
            "stress_concentration_factor": float(Kt),
            "notch_type": notch_type,
            "load": load,
            "notch_geometry": geom,
            "note": (
                "Kt 是名义应力倍数. 最大应力 = Kt × 名义应力 (按净截面算). "
                "fillet/groove/shoulder 用 Peterson 简化拟合, 精确值查图."
            ),
            "reference": "Peterson's Stress Concentration Factors 3rd Ed.",
        },
        success=True,
    )
