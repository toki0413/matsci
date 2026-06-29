"""线弹性断裂力学 (LEFM) — K_I / J / G / K_IC 判据.

参考:
- Anderson《Fracture Mechanics》Ch. 2 (K), Ch. 3 (J-integral)
- Tada《The Stress Analysis of Cracks Handbook》表 2.1 (几何因子 Y)
- ASTM E399 (K_IC 测试标准)
"""

from __future__ import annotations

import math
from typing import Any

from huginn.types import ToolResult


# 常见裂纹构型的几何因子 Y (无限大体或标准试件)
_GEOMETRY_FACTORS: dict[str, float] = {
    "edge": 1.12,        # 边裂纹, 单边拉伸 (无限宽板)
    "interior": 1.0,     # 内部中心裂纹 (无限大板, σ √(πa))
    "surface": 1.12,     # 表面裂纹 (近似 edge)
    "three_point_bend": 1.5,  # 三点弯曲 SE(B) 试件
    "compact_tension": 1.0,   # C(T) 试件, 用 a/W 函数修正, 这里给近似
}


def fracture_lefm(args: Any) -> ToolResult:
    """计算 K_I, J, G 并与 K_IC 比较, 给出断裂判据.

    K_I = Y σ √(π a)
    J = K_I² / E'           (平面应变: E' = E/(1-ν²); 平面应力: E' = E)
    G = K_I² / E'           (线弹性下 J = G)

    判据: K_I < K_IC 安全; K_I >= K_IC 失稳扩展
    """
    crack_type = args.crack_type
    a = args.crack_length
    sigma = args.applied_stress
    Y = args.geometry_factor
    k_ic = args.k_ic
    E = args.youngs_modulus
    nu = args.poissons_ratio

    if a <= 0:
        return ToolResult(
            data=None,
            success=False,
            error=f"crack_length must be positive (got {a})",
        )

    # 几何因子: 用户给的 Y 优先, 否则查表
    if Y is None or Y == 1.12:  # 1.12 是 schema 默认值, 若 crack_type 不匹配则查表
        Y = _GEOMETRY_FACTORS.get(crack_type, 1.12)

    # K_I (Mode I 应力强度因子)
    k_i = Y * sigma * math.sqrt(math.pi * a)

    # J-integral 和能量释放率 G (线弹性下 J = G)
    # 平面应变: E' = E / (1 - ν²)
    if E is not None:
        e_prime = E / (1.0 - nu**2)  # 默认平面应变 (保守)
        j = k_i**2 / e_prime
        g = j  # 线弹性 J = G
        plane_assumption = "plane_strain"
    else:
        j = None
        g = None
        e_prime = None
        plane_assumption = "youngs_modulus_not_provided"

    # 判据
    if k_ic is not None:
        safety_factor = k_ic / k_i if k_i > 0 else float("inf")
        if k_i < k_ic:
            assessment = "safe"
            margin = f"K_I / K_IC = {k_i / k_ic:.4f} < 1"
        else:
            assessment = "unstable_propagation"
            margin = f"K_I / K_IC = {k_i / k_ic:.4f} >= 1"
    else:
        safety_factor = None
        assessment = "no_k_ic_provided"
        margin = "provide k_ic for failure assessment"

    return ToolResult(
        data={
            "stress_intensity_factor_ki": k_i,
            "geometry_factor_y": Y,
            "crack_type": crack_type,
            "crack_length": a,
            "applied_stress": sigma,
            "j_integral": j,
            "energy_release_rate_g": g,
            "effective_modulus_e_prime": e_prime,
            "plane_assumption": plane_assumption,
            "fracture_toughness_kic": k_ic,
            "safety_factor": safety_factor,
            "assessment": assessment,
            "margin": margin,
        },
        success=True,
    )
