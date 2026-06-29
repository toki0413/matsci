"""板解析求解 — 静态挠度 / 模态 / 屈曲.

公式参考:
- Timoshenko & Woinowsky-Krieger《Theory of Plates and Shells》Ch.5 (Navier 级数), Ch.3 (圆形板)
- Reddy《Theory and Analysis of Elastic Plates and Shells》Ch.6 (Mindlin)
- Leissa《Vibration of Plates》Ch.11 (简支板频率)
- Timoshenko & Gere《Theory of Elastic Stability》Ch.9 (屈曲)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from huginn.types import ToolResult


def plate_static(args: Any) -> ToolResult:
    """板静态挠度. Kirchhoff (Navier 级数) 或圆形轴对称解析; Mindlin 加剪切修正."""
    plate = args.plate
    D = plate.D()
    nu = plate.poissons_ratio
    h = plate.thickness
    rho = plate.density

    if args.plate_shape == "rectangular":
        a = float(args.plate_dims["a"])
        b = float(args.plate_dims.get("b", a))
        load = args.plate_load
        ltype = load.get("type", "uniform")

        # 网格
        nx, ny = 41, 41
        xs = np.linspace(0.0, a, nx)
        ys = np.linspace(0.0, b, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        W = np.zeros_like(X)

        # 只支持简支 (Navier), 其他 BC 给个保守估计或报错
        if args.plate_bc != "simply_supported":
            return ToolResult(
                data=None,
                success=False,
                error=(
                    f"plate_static rectangular: plate_bc={args.plate_bc} not supported. "
                    "Only simply_supported (Navier) is implemented."
                ),
            )

        # Navier 双正弦级数: w = ΣΣ q_mn sin(mπx/a) sin(nπy/b) / [D π⁴ ((m/a)² + (n/b)²)²]
        n_terms = 15  # 收敛够快
        if ltype == "uniform":
            q = float(load.get("magnitude", 0.0))
            for m in range(1, n_terms + 1, 2):  # 奇数项
                for n in range(1, n_terms + 1, 2):
                    q_mn = 16.0 * q / (math.pi**2 * m * n)
                    denom = D * math.pi**4 * ((m / a) ** 2 + (n / b) ** 2) ** 2
                    W += (
                        q_mn
                        / denom
                        * np.sin(m * math.pi * X / a)
                        * np.sin(n * math.pi * Y / b)
                    )
        elif ltype == "point":
            P = float(load.get("magnitude", 0.0))
            px = float(load.get("position", [a / 2, b / 2])[0])
            py = float(load.get("position", [a / 2, b / 2])[1])
            for m in range(1, n_terms + 1):
                for n in range(1, n_terms + 1):
                    q_mn = (
                        4.0
                        * P
                        / (a * b)
                        * math.sin(m * math.pi * px / a)
                        * math.sin(n * math.pi * py / b)
                    )
                    denom = D * math.pi**4 * ((m / a) ** 2 + (n / b) ** 2) ** 2
                    W += (
                        q_mn
                        / denom
                        * np.sin(m * math.pi * X / a)
                        * np.sin(n * math.pi * Y / b)
                    )
        else:
            return ToolResult(
                data=None, success=False,
                error=f"plate_static: load type {ltype} not supported.",
            )

        # Mindlin 剪切修正: 简化加一个平均剪切挠度
        # 中点最大: w_shear ≈ q·a²/(κ·G·h) (近似, 矩形 κ=5/6)
        if args.theory == "mindlin":
            G = plate.youngs_modulus / (2 * (1 + nu))
            kappa = 5.0 / 6.0
            if ltype == "uniform":
                # 简化: 用 sin(πx/a)sin(πy/b) 形状包络
                W_shear = (
                    q
                    * a
                    * b
                    / (kappa * G * h * math.pi**2)
                    * np.sin(math.pi * X / a)
                    * np.sin(math.pi * Y / b)
                )
                W += W_shear

        max_idx = np.unravel_index(np.argmax(np.abs(W)), W.shape)
        max_deflection = float(W[max_idx])

        # 弯矩近似: Mx = -D (∂²w/∂x² + ν ∂²w/∂y²)
        # 用有限差分
        d2wdx2 = np.gradient(np.gradient(W, xs, axis=0), xs, axis=0)
        d2wdy2 = np.gradient(np.gradient(W, ys, axis=1), ys, axis=1)
        Mx = -D * (d2wdx2 + nu * d2wdy2)
        My = -D * (d2wdy2 + nu * d2wdx2)
        # Mxy = -D(1-ν) ∂²w/∂x∂y (简化取 0)
        Mxy = np.zeros_like(W)
        max_Mx = float(np.max(np.abs(Mx)))
        max_My = float(np.max(np.abs(My)))
        # 最大应力 σ = 6M/h²
        max_stress = 6.0 * max(max_Mx, max_My) / h**2

        return ToolResult(
            data={
                "max_deflection": max_deflection,
                "deflection_field": {
                    "x": xs.tolist(),
                    "y": ys.tolist(),
                    "w": W.tolist(),
                },
                "max_stress": {
                    "sigma_xx": float(6 * max_Mx / h**2),
                    "sigma_yy": float(6 * max_My / h**2),
                    "sigma_vm": float(max_stress),
                },
                "bending_moments": {
                    "Mx": max_Mx,
                    "My": max_My,
                    "Mxy": float(np.max(np.abs(Mxy))),
                },
                "theory": args.theory,
                "plate_bc": args.plate_bc,
                "plate_shape": "rectangular",
                "plate_dims": {"a": a, "b": b},
            },
            success=True,
        )

    if args.plate_shape == "circular":
        R = float(args.plate_dims["radius"])
        load = args.plate_load
        ltype = load.get("type", "uniform")
        bc = args.plate_bc

        # 圆形板轴对称解析 (Timoshenko Ch.3)
        # 均布 q:
        #   clamped: w(r) = q (R²-r²)² / (64 D),  w_max = q R⁴ / (64 D)
        #   simply_supported: w_max = (5+ν) q R⁴ / (64 (1+ν) D)
        nr = 51
        rs = np.linspace(0.0, R, nr)
        if ltype != "uniform":
            return ToolResult(
                data=None, success=False,
                error=f"plate_static circular: load type {ltype} not supported (only uniform).",
            )
        q = float(load.get("magnitude", 0.0))
        if bc == "clamped":
            w = q * (R**2 - rs**2) ** 2 / (64 * D)
            w_max = q * R**4 / (64 * D)
        elif bc == "simply_supported":
            # w(r) = (q (R²-r²) / (64 D)) · ((5+ν)/(1+ν) R² - r²)
            w = q * (R**2 - rs**2) / (64 * D) * ((5 + nu) / (1 + nu) * R**2 - rs**2)
            w_max = (5 + nu) * q * R**4 / (64 * (1 + nu) * D)
        else:
            return ToolResult(
                data=None, success=False,
                error=f"plate_static circular: bc={bc} not supported.",
            )

        # Mindlin 剪切修正 (圆形, 简化)
        if args.theory == "mindlin":
            G = plate.youngs_modulus / (2 * (1 + nu))
            kappa = 5.0 / 6.0
            # clamped: w_s_max = q R² / (8 κ G h); ss: q R² / (8 κ G h) * 系数
            if bc == "clamped":
                w_shear_max = q * R**2 / (8 * kappa * G * h)
            else:
                w_shear_max = q * R**2 / (8 * kappa * G * h) * (1 + 0.5)
            # 沿 r 抛物线分布
            w_shear = w_shear_max * (1 - (rs / R) ** 2)
            w = w + w_shear
            w_max = float(w.max())

        return ToolResult(
            data={
                "max_deflection": float(w_max),
                "deflection_field": {"r": rs.tolist(), "w": w.tolist()},
                "max_stress": {
                    "sigma_vm": float(6 * q * R**2 / h**2 / 8 * (1 + nu)),
                },
                "theory": args.theory,
                "plate_bc": bc,
                "plate_shape": "circular",
                "plate_dims": {"radius": R},
            },
            success=True,
        )

    return ToolResult(
        data=None, success=False,
        error=f"plate_static: unknown plate_shape={args.plate_shape}",
    )


def plate_modal(args: Any) -> ToolResult:
    """板自由振动模态. 简支矩形: ω_mn = π² √(D/(ρh)) ((m/a)²+(n/b)²)."""
    plate = args.plate
    D = plate.D()
    h = plate.thickness
    rho = plate.density

    if args.plate_shape != "rectangular":
        return ToolResult(
            data=None, success=False,
            error="plate_modal: only rectangular simply_supported implemented.",
        )
    if args.plate_bc != "simply_supported":
        return ToolResult(
            data=None, success=False,
            error="plate_modal: only simply_supported bc implemented.",
        )

    a = float(args.plate_dims["a"])
    b = float(args.plate_dims.get("b", a))

    # 生成 (m, n) 组合并算频率, 排序取前 N
    candidates = []
    coef = math.pi**2 * math.sqrt(D / (rho * h))
    for m in range(1, 20):
        for n in range(1, 20):
            omega = coef * ((m / a) ** 2 + (n / b) ** 2)
            candidates.append((omega, m, n))
    candidates.sort(key=lambda t: t[0])
    top = candidates[: args.n_modes]

    freqs = [w / (2 * math.pi) for w, _, _ in top]
    indices = [{"m": m, "n": n} for _, m, n in top]
    omegas = [w for w, _, _ in top]

    return ToolResult(
        data={
            "frequencies_hz": freqs,
            "angular_frequencies": omegas,
            "mode_indices": indices,
            "theory": args.theory,
            "plate_bc": args.plate_bc,
            "n_modes": len(top),
        },
        success=True,
    )


def plate_buckling(args: Any) -> ToolResult:
    """板屈曲临界载荷. 简支单轴: N_cr = π² D · ((m/a)² + (n/b)²) 取最小."""
    plate = args.plate
    D = plate.D()
    h = plate.thickness

    if args.plate_shape != "rectangular":
        return ToolResult(
            data=None, success=False,
            error="plate_buckling: only rectangular implemented.",
        )
    if args.plate_bc != "simply_supported":
        return ToolResult(
            data=None, success=False,
            error="plate_buckling: only simply_supported bc implemented.",
        )

    a = float(args.plate_dims["a"])
    b = float(args.plate_dims.get("b", a))

    # 单轴 x 方向受压: N_cr = π² D / b² · (m b/a + n² a/(m b))²
    # 简化: 取最小化 (m, n=1)
    best = None
    for m in range(1, 10):
        # 单轴, n=1
        N_cr = math.pi**2 * D * ((m / a) ** 2 + (1 / b) ** 2)
        if best is None or N_cr < best[0]:
            best = (N_cr, m, 1)
    N_cr, m_opt, n_opt = best
    sigma_cr = N_cr / h

    return ToolResult(
        data={
            "critical_load_per_unit_length": float(N_cr),
            "critical_stress": float(sigma_cr),
            "buckling_mode": {"m": m_opt, "n": n_opt},
            "theory": args.theory,
            "plate_bc": args.plate_bc,
        },
        success=True,
    )
