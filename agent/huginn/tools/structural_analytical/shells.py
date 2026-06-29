"""圆柱壳解析求解 — 屈曲 / 模态.

Donnell 薄壳理论, 适合 h/R << 0.01 的薄壁圆柱壳.

公式参考:
- Donnell《Stability of Thin-Walled Tubes under Torsion》NACA Report 4729 (轴向屈曲)
- NASA SP-8007《Buckling of Thin-Walled Circular Cylinders》(knockdown factor)
- Flügge《Stresses in Shells》§8.3 (外压屈曲)
- Leissa《Vibration of Shells》NASA SP-288, Ch.2 (Donnell-Mushtari 模态)
- Soedel《Vibrations of Shells and Plates》Eq.6.3.7 (频率参数)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from huginn.types import ToolResult


# NASA SP-8007 经验 knockdown 因子下界 (实际壳体有初始缺陷, 低于经典值)
# 这里取一个偏保守的常数. 严格做法是按 (R/h, L/R) 查表, 这里简化.
_DEFAULT_KNOCKDOWN = 0.65


def shell_buckling(args: Any) -> ToolResult:
    """Donnell 圆柱壳屈曲. 轴向 / 外压 / 扭转."""
    shell = args.shell
    E = shell.youngs_modulus
    nu = shell.poissons_ratio
    R = shell.radius
    L = shell.length
    h = shell.thickness
    load_type = args.shell_load_type

    # 经典轴向屈曲临界应力 (Donnell): σ_cl = E h / (R √(3(1-ν²)))
    sigma_classical = E * h / (R * math.sqrt(3.0 * (1.0 - nu**2)))

    if load_type == "axial":
        # 轴向受压: σ_cr = knockdown · σ_cl
        # NASA SP-8007: 实际壳体 knockdown 0.6-0.8, 取 0.65
        knockdown = _DEFAULT_KNOCKDOWN
        sigma_cr = knockdown * sigma_classical
        P_cr = 2.0 * math.pi * R * h * sigma_cr  # 总轴压载荷

        return ToolResult(
            data={
                "critical_stress": float(sigma_cr),
                "critical_load": float(P_cr),
                "classical_critical_stress": float(sigma_classical),
                "knockdown_factor": float(knockdown),
                "load_type": "axial",
                "theory": args.theory,
                "reference": "NASA SP-8007, Donnell NACA Report 4729",
            },
            success=True,
        )

    if load_type == "external_pressure":
        # 外压屈曲: 扫 (m, n) 求 Donnell 方程最小 p_cr
        # p_cr(m,n) = (E/(4(1-ν²))) · (h/R)³ · (1/n²) · ((n²-1+λ²)² + λ²(n²+1))² / ((n²+λ²)² · (n²-1))
        # 简化: 用 Flügge §8.3 的 Donnell 形式, 扫 m=1..5, n=2..30
        # n=0,1 对外压不是控制模态 (n=1 是刚体位移, n=0 是轴对称)
        lambda_vals = [m * math.pi * R / L for m in range(1, 6)]
        best = None
        for m_idx, lam in enumerate(lambda_vals, start=1):
            for n in range(2, 31):
                # Flügge Donnell 外压屈曲方程 (简化形式)
                lam2 = lam**2
                n2 = n**2
                # 弯曲项 + 薄膜项
                bending = (n2 - 1.0 + lam2) ** 2
                membrane = lam2 * (n2 + 1.0) ** 2 / (n2 - 1.0)
                # 临界压力
                p_mn = (E / (12.0 * (1.0 - nu**2))) * (h / R) ** 3 * (
                    (n2 - 1.0) / (R * n2)
                ) * (bending + membrane) / (n2 + lam2) ** 2
                # 上面的公式有点绕, 换一个更标准的写法
                # p_cr = (E/(12(1-ν²))) · (h³/R³) · (1/(n²-1)) · (λ²+n²-1)² / (λ²+n²)²
                p_std = (
                    (E / (12.0 * (1.0 - nu**2)))
                    * (h**3 / R**3)
                    * (1.0 / (n2 - 1.0))
                    * (lam2 + n2 - 1.0) ** 2
                    / (lam2 + n2) ** 2
                )
                if best is None or p_std < best[0]:
                    best = (p_std, m_idx, n)

        p_cr, m_opt, n_opt = best
        # 外压也加 knockdown
        knockdown = _DEFAULT_KNOCKDOWN
        p_cr *= knockdown

        return ToolResult(
            data={
                "critical_pressure": float(p_cr),
                "critical_load": float(p_cr * 2.0 * math.pi * R * L),  # 总外压合力
                "classical_critical_pressure": float(p_cr / knockdown),
                "knockdown_factor": float(knockdown),
                "buckling_mode": {"m": m_opt, "n": n_opt},
                "load_type": "external_pressure",
                "theory": args.theory,
                "reference": "Flügge §8.3, NASA SP-8007",
            },
            success=True,
        )

    if load_type == "torsion":
        # 扭转屈曲: τ_cr ≈ 0.75 · E · (h/R)^(3/2) / √(1+ν) · (R/L)^(1/2)
        # Donnell 扭转屈曲经典解 (简支)
        tau_cr = (
            0.75
            * E
            * (h / R) ** 1.5
            / math.sqrt(1.0 + nu)
            * math.sqrt(R / L)
        )
        knockdown = _DEFAULT_KNOCKDOWN
        tau_cr *= knockdown
        T_cr = 2.0 * math.pi * R**2 * h * tau_cr  # 扭矩

        return ToolResult(
            data={
                "critical_shear_stress": float(tau_cr),
                "critical_torque": float(T_cr),
                "classical_critical_stress": float(tau_cr / knockdown),
                "knockdown_factor": float(knockdown),
                "load_type": "torsion",
                "theory": args.theory,
                "reference": "Donnell NACA Report 4729 (torsion)",
            },
            success=True,
        )

    return ToolResult(
        data=None,
        success=False,
        error=f"shell_buckling: unknown load_type={load_type}",
    )


def shell_modal(args: Any) -> ToolResult:
    """Donnell-Mushtari 圆柱壳自由振动模态.

    扫 (m, n) 求频率. 简支边界 (SS-3).
    ω_mn = √(E/(ρ(1-ν²)R²)) · √( (1-ν²)λ⁴/(λ²+n²)² + k·(λ²+n²)² )

    λ = mπR/L, k = h²/(12R²), n=环向波数, m=轴向半波数.
    参考: Leissa《Vibration of Shells》NASA SP-288, Eq.(2.27).
    """
    shell = args.shell
    E = shell.youngs_modulus
    nu = shell.poissons_ratio
    R = shell.radius
    L = shell.length
    h = shell.thickness
    rho = shell.density

    # 频率参数
    C = math.sqrt(E / (rho * (1.0 - nu**2) * R**2))
    k = h**2 / (12.0 * R**2)  # 薄壳参数

    # 扫 (m, n), m=1..10, n=0..20
    candidates = []
    for m in range(1, 11):
        lam = m * math.pi * R / L
        lam2 = lam**2
        for n in range(0, 21):
            n2 = n**2
            denom = lam2 + n2
            if denom < 1e-12:
                continue
            # Donnell-Mushtari 频率
            omega_sq_factor = (1.0 - nu**2) * lam2**2 / denom + k * denom**2
            omega = C * math.sqrt(max(omega_sq_factor, 0.0))
            candidates.append((omega, m, n))

    candidates.sort(key=lambda t: t[0])
    n_modes = min(args.n_modes, len(candidates))
    top = candidates[:n_modes]

    freqs_hz = [w / (2.0 * math.pi) for w, _, _ in top]
    omegas = [w for w, _, _ in top]
    indices = [{"m": m, "n": n} for _, m, n in top]

    return ToolResult(
        data={
            "frequencies_hz": freqs_hz,
            "angular_frequencies": omegas,
            "mode_indices": indices,
            "theory": args.theory,
            "n_modes": n_modes,
            "reference": "Leissa NASA SP-288, Donnell-Mushtari",
        },
        success=True,
    )
