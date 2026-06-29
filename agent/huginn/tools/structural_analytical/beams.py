"""梁解析求解 — 静态挠度 / 模态 / 屈曲.

公式参考:
- Gere & Timoshenko《Mechanics of Materials》App. G (挠度表)
- Rao《Mechanical Vibrations》Eq.8.38 (模态频率)
- Blevin《Formulas for Natural Frequency and Mode Shape》Table 4-1 (β_nL 系数)
- Timoshenko & Gere《Theory of Elastic Stability》Eq.2.1-2.6 (Euler 屈曲)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from huginn.types import ToolResult


# 不同边界条件下的 β_n L 系数 (前 5 阶), 来自 Blevin Table 4-1
_BETA_NL: dict[str, list[float]] = {
    "simply_supported": [math.pi, 2 * math.pi, 3 * math.pi, 4 * math.pi, 5 * math.pi],
    "cantilever": [1.87510407, 4.69409113, 7.85475744, 10.99554073, 14.13716839],
    "fixed_fixed": [4.73004074, 7.85320462, 10.9956078, 14.1371655, 17.2787596],
    "fixed_pinned": [3.92660231, 7.06858275, 10.21017612, 13.35176879, 16.49336143],
}

# Euler 屈曲有效长度系数 K (Timoshenko & Gere Eq.2.1-2.6)
_BUCKLING_K: dict[str, float] = {
    "simply_supported": 1.0,
    "cantilever": 2.0,
    "fixed_fixed": 0.5,
    "fixed_pinned": 0.699,  # 经验值 0.7
    "clamped": 0.5,  # 等同 fixed_fixed
    "free": 2.0,  # 等同 cantilever
}

# 矩形截面剪切系数 κ = 5/6 (Timoshenko)
_RECT_SHEAR_FACTOR = 5.0 / 6.0


def _shear_factor(section_type: str) -> float:
    # 简化: 矩形 5/6, 圆形 10/9, 其他默认 5/6
    if section_type == "circular":
        return 10.0 / 9.0
    return _RECT_SHEAR_FACTOR


def beam_static(args: Any) -> ToolResult:
    """单跨等截面梁静态挠度. 用经典公式 + 叠加法."""
    beam = args.beam
    E, I = beam.youngs_modulus, beam.resolved_I()
    L = beam.length
    A = beam.resolved_A()
    bc = args.boundary
    theory = args.theory

    xs = np.linspace(0.0, L, args.n_points)
    w = np.zeros_like(xs)

    # 简化: 按几种标准 case 叠加. 不做通用积分, 只识别常见 load/BC 组合.
    for load in args.loads:
        ltype = load.get("type")
        value = float(load.get("value", 0.0))
        pos = float(load.get("position", L if ltype == "udl" else L))

        if bc == "cantilever" and ltype == "point" and abs(pos - L) < 1e-9:
            # 端部集中力: δ(x) = P x^2 (3L - x) / (6 EI)
            w += value * xs**2 * (3 * L - xs) / (6 * E * I)
        elif bc == "cantilever" and ltype == "udl":
            # 全跨均布: δ(x) = q x^2 (6L^2 - 4Lx + x^2) / (24 EI)
            w += value * xs**2 * (6 * L**2 - 4 * L * xs + xs**2) / (24 * E * I)
        elif bc == "simply_supported" and ltype == "point" and abs(pos - L / 2) < 1e-9:
            # 中点集中力: 对 x<=L/2, δ = P x (3L^2 - 4x^2) / (48 EI)
            left = xs <= L / 2
            w[left] += value * xs[left] * (3 * L**2 - 4 * xs[left] ** 2) / (48 * E * I)
            # 对称: 右半段镜像
            right = xs >= L / 2
            xr = L - xs[right]
            w[right] += value * xr * (3 * L**2 - 4 * xr**2) / (48 * E * I)
        elif bc == "simply_supported" and ltype == "udl":
            # 全跨均布: δ(x) = q x (L^3 - 2L x^2 + x^3) / (24 EI)
            w += value * xs * (L**3 - 2 * L * xs**2 + xs**3) / (24 * E * I)
        elif bc == "fixed_fixed" and ltype == "point" and abs(pos - L / 2) < 1e-9:
            # 中点集中力: δ(x) = P x^2 (3L - 4x)^2 / (192 EI) for x<=L/2 (简化的对称形)
            left = xs <= L / 2
            w_l = value * xs[left] ** 2 * (3 * L - 4 * xs[left]) ** 2 / (192 * E * I)
            w[left] += w_l
            right = xs >= L / 2
            xr = L - xs[right]
            w_r = value * xr**2 * (3 * L - 4 * xr) ** 2 / (192 * E * I)
            w[right] += w_r
        else:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"beam_static: load {load} with bc={bc} not supported in "
                    "this simplified implementation. Supported: cantilever+point(端部), "
                    "cantilever+udl, simply_supported+point(中点), simply_supported+udl, "
                    "fixed_fixed+point(中点)."
                ),
            )

    # Timoshenko 剪切修正: w = w_b + w_s, 剪切挠度近似按 load 类型加一个端部挠度修正
    # 简化做法: 对端部/中点挠度加 PL/(κAG) (point) 或 qL²/(8κAG) (udl)
    if theory == "timoshenko":
        G = E / (2 * (1 + beam.poissons_ratio))
        kappa = _shear_factor(beam.section_type)
        w_shear = np.zeros_like(xs)
        for load in args.loads:
            ltype = load.get("type")
            value = float(load.get("value", 0.0))
            if ltype == "point":
                # 简化: 取端部/中点位置的剪切挠度, 整段近似抛物线
                if bc == "cantilever":
                    w_shear += value * xs / (kappa * G * A * L) * (1 - xs / (2 * L))
                elif bc == "simply_supported":
                    # 中点最大: PL/(4 κAG), 沿 x 用 sin(πx/L) 形状近似
                    w_shear += value * L / (4 * kappa * G * A) * np.sin(math.pi * xs / L)
            elif ltype == "udl":
                if bc == "cantilever":
                    w_shear += value * xs * (2 * L - xs) / (2 * kappa * G * A)
                elif bc == "simply_supported":
                    # 中点最大: 5qL²/(48 κAG) (近似)
                    w_shear += (
                        5 * value * L**2 / (48 * kappa * G * A) * np.sin(math.pi * xs / L)
                    )
        w += w_shear

    max_idx = int(np.argmax(np.abs(w)))
    max_deflection = float(w[max_idx])
    max_deflection_location = float(xs[max_idx])

    # 弯矩近似: M = -EI w''  (用有限差分)
    if len(xs) >= 3:
        d2w = np.gradient(np.gradient(w, xs), xs)
        moments = -E * I * d2w
        max_moment = float(np.max(np.abs(moments)))
    else:
        max_moment = 0.0

    # 最大弯曲应力 σ = M·c/I, c = h/2 (矩形) 或 d/2 (圆形)
    if beam.section_type == "rectangular":
        c = beam.section_dims["h"] / 2.0
    elif beam.section_type == "circular":
        c = beam.section_dims["d"] / 2.0
    else:
        c = (I / A) ** 0.5  # 回转半径近似
    max_bending_stress = max_moment * c / I if I > 0 else 0.0

    return ToolResult(
        data={
            "max_deflection": max_deflection,
            "max_deflection_location": max_deflection_location,
            "deflection_curve": {"x": xs.tolist(), "w": w.tolist()},
            "max_bending_stress": float(max_bending_stress),
            "max_moment": float(max_moment),
            "theory": theory,
            "boundary": bc,
            "loads": args.loads,
        },
        success=True,
    )


def beam_modal(args: Any) -> ToolResult:
    """梁自由振动模态. ω_n = (β_n L)^2 sqrt(EI/(ρ A L^4))."""
    beam = args.beam
    E, I = beam.youngs_modulus, beam.resolved_I()
    L = beam.length
    A = beam.resolved_A()
    rho = beam.density
    bc = args.boundary

    if bc not in _BETA_NL:
        return ToolResult(
            data=None,
            success=False,
            error=f"beam_modal: boundary={bc} not supported. Use one of {list(_BETA_NL)}.",
        )

    betas = _BETA_NL[bc]
    n = min(args.n_modes, len(betas))
    # 拿不到足够的预存系数就现场迭代 (前几阶足够)
    while len(betas) < args.n_modes:
        betas.append(betas[-1] + math.pi)  # 近似: 高阶近似按 π 递增 (粗)
        # 注意: 这只是 fallback, 实际工程应用只取前几阶

    # b 已经是 β_n L (无量纲), 直接平方
    omega_eb = [b**2 * math.sqrt(E * I / (rho * A * L**4)) for b in betas[:n]]

    # Timoshenko 修正: ω = ω_EB / sqrt(1 + (βL)^2 (EI/(κAGL²)) + (βL r/L)²)
    if args.theory == "timoshenko":
        G = E / (2 * (1 + beam.poissons_ratio))
        kappa = _shear_factor(beam.section_type)
        r2 = I / A  # 回转半径平方
        omega = []
        for b, w_eb in zip(betas[:n], omega_eb, strict=True):
            beta_L = b  # b 就是 β_n L
            corr = math.sqrt(
                1.0
                + beta_L**2 * E * I / (kappa * G * A * L**2)
                + beta_L**2 * r2 / L**2
            )
            omega.append(w_eb / corr)
    else:
        omega = omega_eb

    freqs_hz = [w / (2 * math.pi) for w in omega]

    # 模态形状 (Euler-Bernoulli 标准)
    xs = np.linspace(0.0, L, 101)
    mode_shapes = []
    for i, b in enumerate(betas[:n]):
        beta_L = b  # b 就是 β_n L
        if bc == "simply_supported":
            phi = np.sin(beta_L * xs / L)
        elif bc == "cantilever":
            # 标准悬臂模态: cosh(βx) - cos(βx) - σ(sinh(βx) - sin(βx))
            sigma = (
                (math.sinh(beta_L) + math.sin(beta_L))
                / (math.cosh(beta_L) + math.cos(beta_L))
            )
            bx = beta_L * xs / L
            phi = np.cosh(bx) - np.cos(bx) - sigma * (np.sinh(bx) - np.sin(bx))
        elif bc == "fixed_fixed":
            sigma = (
                (math.sinh(beta_L) - math.sin(beta_L))
                / (math.cosh(beta_L) - math.cos(beta_L))
            )
            bx = beta_L * xs / L
            phi = np.cosh(bx) - np.cos(bx) - sigma * (np.sinh(bx) - np.sin(bx))
        else:
            # fixed_pinned 等用 sin 近似
            phi = np.sin(beta_L * xs / L)
        # 归一化: 最大幅值=1
        phi = phi / max(abs(phi).max(), 1e-30)
        mode_shapes.append(
            {"n": i + 1, "x": xs.tolist(), "phi": phi.tolist()}
        )

    return ToolResult(
        data={
            "frequencies_hz": freqs_hz,
            "angular_frequencies": omega,
            "mode_shapes": mode_shapes,
            "theory": args.theory,
            "boundary": bc,
            "n_modes": n,
        },
        success=True,
    )


def beam_buckling(args: Any) -> ToolResult:
    """Euler 屈曲临界载荷. P_cr = π² EI / (K L)²."""
    beam = args.beam
    E, I = beam.youngs_modulus, beam.resolved_I()
    L = beam.length
    A = beam.resolved_A()
    bc = args.boundary

    K = _BUCKLING_K.get(bc)
    if K is None:
        return ToolResult(
            data=None,
            success=False,
            error=f"beam_buckling: boundary={bc} not supported.",
        )

    P_cr = math.pi**2 * E * I / (K * L) ** 2
    sigma_cr = P_cr / A
    r = math.sqrt(I / A)  # 回转半径
    slenderness = K * L / r

    # 简化 regime 判定: 弹性 (实际工程还需对照比例极限)
    regime = "elastic"

    return ToolResult(
        data={
            "critical_load": float(P_cr),
            "critical_stress": float(sigma_cr),
            "slenderness_ratio": float(slenderness),
            "effective_length": float(K * L),
            "k_factor": float(K),
            "regime": regime,
            "boundary": bc,
        },
        success=True,
    )
