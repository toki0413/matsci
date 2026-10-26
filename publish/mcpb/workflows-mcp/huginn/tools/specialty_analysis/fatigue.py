"""疲劳分析 — S-N 曲线 + Paris 律裂纹扩展.

参考:
- Suresh《Fatigue of Materials》Ch. 4 (S-N), Ch. 7 (裂纹扩展)
- Stephens《Metal Fatigue in Engineering》Sec. 4.4 (Basquin + 平均应力修正)
- Paris & Erdogan 1963 (da/dN = C ΔK^m)
"""

from __future__ import annotations

from typing import Any

from huginn.core_types import ToolResult


def _sn_mean_stress_correction(
    args: Any, sigma_a: float, sigma_m: float, sigma_f: float, theory: str
) -> tuple[float, float] | ToolResult:
    """按平均应力理论把应力幅折算到零均值; 缺参时返回失败结果."""
    if abs(sigma_m) < 1e-12:
        return sigma_a, 1.0
    if theory == "goodman":
        sigma_u = args.material_uts
        if sigma_u is None:
            return ToolResult(
                data=None,
                success=False,
                error="goodman correction requires material_uts",
            )
        correction_factor = 1.0 - sigma_m / sigma_u
        return sigma_a * correction_factor, correction_factor
    if theory == "soderberg":
        sigma_y = args.material_yield
        if sigma_y is None:
            return ToolResult(
                data=None,
                success=False,
                error="soderberg correction requires material_yield",
            )
        correction_factor = 1.0 - sigma_m / sigma_y
        return sigma_a * correction_factor, correction_factor
    if theory == "gerber":
        sigma_u = args.material_uts
        if sigma_u is None:
            return ToolResult(
                data=None,
                success=False,
                error="gerber correction requires material_uts",
            )
        correction_factor = 1.0 - (sigma_m / sigma_u) ** 2
        return sigma_a * correction_factor, correction_factor
    if theory == "morrow":
        correction_factor = 1.0 - sigma_m / sigma_f
        return sigma_a * correction_factor, correction_factor
    return ToolResult(
        data=None,
        success=False,
        error=f"unknown mean_stress_theory: {theory}",
    )


def _sn_range_error(
    correction_factor: float, sigma_a_eq: float, sigma_f: float, theory: str
) -> ToolResult | None:
    """等效幅 / 修正因子越界检查, 越界时返回失败结果."""
    if correction_factor <= 0:
        return ToolResult(
            data=None,
            success=False,
            error=f"mean stress too high — {theory} factor {correction_factor:.4f} <= 0 (infinite life region exceeded)",
        )
    if sigma_a_eq <= 0 or sigma_a_eq >= sigma_f:
        return ToolResult(
            data=None,
            success=False,
            error=(
                f"equivalent stress amplitude {sigma_a_eq:.3e} out of valid range "
                f"(0, sigma_f_prime={sigma_f:.3e}) — Basquin not applicable"
            ),
        )
    return None


def _sn_result(
    args: Any,
    n_f: float,
    sigma_a_eq: float,
    correction_factor: float,
    sigma_f: float,
    b: float,
    safety_factor: float,
) -> ToolResult:
    """组装 S-N 疲劳寿命成功结果."""
    return ToolResult(
        data={
            "fatigue_life_cycles": n_f,
            "equivalent_stress_amplitude": sigma_a_eq,
            "correction_factor": correction_factor,
            "mean_stress_theory": args.mean_stress_theory,
            "basquin_params": {"sigma_f_prime": sigma_f, "b": b},
            "applied_stress_amplitude": args.stress_amplitude,
            "applied_mean_stress": args.mean_stress,
            "cycles_limit": args.cycles_limit,
            "safety_factor_vs_limit": safety_factor,
            "regime": "finite_life" if n_f < args.cycles_limit else "infinite_life",
        },
        success=True,
    )


def _crack_growth_invalid(
    a_init: float, a_final: float, dk: float, C: float
) -> ToolResult | None:
    """Paris 律输入合法性检查, 非法时返回失败结果."""
    if a_final <= a_init:
        return ToolResult(
            data=None,
            success=False,
            error=f"a_final ({a_final}) must be > a_init ({a_init})",
        )
    if dk <= 0:
        return ToolResult(
            data=None,
            success=False,
            error=f"dk_range must be positive (got {dk})",
        )
    if C <= 0:
        return ToolResult(
            data=None,
            success=False,
            error=f"Paris C must be positive (got {C})",
        )
    return None


def _crack_growth_result(
    n_f: float,
    growth_rate: float,
    C: float,
    m: float,
    dk: float,
    a_init: float,
    a_final: float,
    final_k: float,
) -> ToolResult:
    """组装 Paris 律裂纹扩展成功结果."""
    return ToolResult(
        data={
            "fatigue_life_cycles": n_f,
            "crack_growth_rate": growth_rate,
            "paris_params": {"C": C, "m": m},
            "dk_range": dk,
            "a_init": a_init,
            "a_final": a_final,
            "crack_extension": a_final - a_init,
            "final_delta_k": final_k,
            "regime": "stable_growth" if dk > 0 else "no_growth",
        },
        success=True,
    )


def fatigue_sn(args: Any) -> ToolResult:
    """Basquin 律 + 平均应力修正, 给应力幅求疲劳寿命 N_f.

    Basquin: σ_a = σ_f' (2 N_f)^b  →  N_f = 0.5 (σ_a / σ_f')^(1/b)
    平均应力修正把 σ_a 等效到 σ_a,eq (零均值):
      - Goodman:    σ_a,eq / σ_a = 1 - σ_m / σ_u
      - Soderberg:  σ_a,eq / σ_a = 1 - σ_m / σ_y
      - Gerber:     σ_a,eq / σ_a = 1 - (σ_m / σ_u)^2
      - Morrow:     σ_a,eq / σ_a = 1 - σ_m / σ_f'
    """
    sn = args.sn_params
    sigma_f = sn.get("sigma_f_prime")
    b = sn.get("b")
    if sigma_f is None or b is None:
        return ToolResult(
            data=None,
            success=False,
            error="sn_params must contain 'sigma_f_prime' and 'b'",
        )

    sigma_a = args.stress_amplitude
    sigma_m = args.mean_stress
    theory = args.mean_stress_theory

    # 计算等效应力幅 (折算到零均值)
    corr = _sn_mean_stress_correction(args, sigma_a, sigma_m, sigma_f, theory)
    if isinstance(corr, ToolResult):
        return corr
    sigma_a_eq, correction_factor = corr

    err = _sn_range_error(correction_factor, sigma_a_eq, sigma_f, theory)
    if err is not None:
        return err

    # Basquin: σ_a_eq = σ_f' (2 N_f)^b  →  N_f = 0.5 (σ_a_eq / σ_f')^(1/b)
    ratio = sigma_a_eq / sigma_f
    n_f = 0.5 * ratio ** (1.0 / b)

    # 安全因子 vs 循环数上限
    cycles_limit = args.cycles_limit
    safety_factor = cycles_limit / n_f if n_f > 0 else float("inf")

    return _sn_result(
        args, n_f, sigma_a_eq, correction_factor, sigma_f, b, safety_factor
    )


def fatigue_crack_growth(args: Any) -> ToolResult:
    """Paris 律积分: da/dN = C (ΔK)^m, 从 a_init 积到 a_final 求 N_f.

    假设 ΔK 不随裂纹长度变化 (恒幅加载, K = Y σ √(πa) 近似简化).
    严格积分需用数值方法处理 Y(a) 依赖, 这里给恒 ΔK 的解析解:
      N_f = (a_final - a_init) / (C ΔK^m)   [恒 ΔK, m != -1]
    """
    paris = args.paris_params
    C = paris.get("C")
    m = paris.get("m")
    if C is None or m is None:
        return ToolResult(
            data=None,
            success=False,
            error="paris_params must contain 'C' and 'm'",
        )

    dk = args.dk_range
    a_init = args.a_init
    a_final = args.a_final

    invalid = _crack_growth_invalid(a_init, a_final, dk, C)
    if invalid is not None:
        return invalid

    # 恒 ΔK 解析解: N = ∫da / (C ΔK^m) = (a_f - a_i) / (C ΔK^m)
    growth_rate = C * dk**m
    n_f = (a_final - a_init) / growth_rate

    # 估算最终裂纹长度时的应力强度因子 (假设 Y σ √(πa))
    # 仅作参考, 不参与寿命计算
    final_k = dk  # 用户给的 ΔK 视为代表性值

    return _crack_growth_result(n_f, growth_rate, C, m, dk, a_init, a_final, final_k)
