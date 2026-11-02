"""MgH2 固态储氢床一维氢热耦合模型 (C2 仿真内核的最小可信内核).

物理与文献量级 (全部为公开量级, 不是"实测值"):
- van't Hoff: ln(p_eq / 1 bar) = -dH/(R T) + dS/R
  dH ≈ -75 kJ/mol H2, dS ≈ -135 J/(mol H2 K)  (MgH2 公开文献量级)
- 吸氢动力学 Arrhenius: k(T) = k0 * exp(-Ea/(R T)), Ea ≈ 120 kJ/mol
- 驱动势: f = clamp(p/p_eq - 1, 0, f_max) (吸氢; 放氢时 f=0)
- 一维床层能量方程 (隐式扩散 + 显式反应源项):
    rho_c * dT/dt = k_eff * d2T/dx2 + |dH| * c_max * dalpha/dt
  两端 Dirichlet 温度 T_bath.
- 由 MgH2 密度 1450 kg/m3 / 摩尔质量 26.3 g/mol 得 c_max ≈ 55.1 kmol H2/m3
  (每个 MgH2 容纳 1 个 H2).

单位: SI. 时间 s, 长度 m, 温度 K, 压力 Pa.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

R = 8.314462618  # J/(mol K)
P0 = 1.0e5  # 1 bar


@dataclass(frozen=True)
class Params:
    dH: float = -75.0e3            # J/mol H2
    dS: float = -135.0             # J/(mol H2 K)
    Ea: float = 120.0e3            # J/mol
    k0: float = 1.0e7              # 1/s
    k_eff: float = 1.0              # W/(m K) 粉末床有效导热
    rho_c: float = 1.45e6          # J/(m3 K) = 1450 kg/m3 * 1000 J/(kg K)
    c_max: float = 55133.0         # mol H2/m3
    L: float = 10.0e-3             # m 床层厚度
    T_bath: float = 573.0          # K
    p_in: float = 1.0e6            # Pa
    f_max: float = 20.0

    def with_(self, **kw) -> "Params":
        return replace(self, **kw)

    @property
    def energy_density(self) -> float:
        """满吸氢焓密度 J/m3."""
        return abs(self.dH) * self.c_max

    @property
    def alpha_th(self) -> float:
        """热扩散率 m2/s."""
        return self.k_eff / self.rho_c


def p_eq(T: float | np.ndarray, p: Params) -> float | np.ndarray:
    """平衡压力 (Pa).

    符号约定: 本文件统一用**吸氢侧**焓/熵 (dH=-75 kJ/mol, dS=-135 J/(mol K)).
    放氢侧 van't Hoff 为 ln(p_eq/P0) = -dH_des/(R T) + dS_des/R, 其中
    dH_des=-dH, dS_des=-dS, 代回即
        ln(p_eq/P0) = dH/(R T) - dS/R = -75000/(R T) + 16.24
    温度升高 → p_eq 升高 (高温不利于吸氢, 反应自限).
    校验点: T=573 K → p_eq ≈ 1.6 bar, 与 MgH2 公开平台压 ~1 bar / 570 K 同量级.
    """
    return P0 * np.exp(p.dH / (R * np.asarray(T, dtype=float)) - p.dS / R)


def rate(alpha, T, p: Params):
    """吸氢速率 dalpha/dt (1/s). alpha 为分数, T 为 K."""
    alpha = np.asarray(alpha, dtype=float)
    T = np.asarray(T, dtype=float)
    peq = p_eq(T, p)
    f = np.clip(p.p_in / peq - 1.0, 0.0, p.f_max)
    k = p.k0 * np.exp(-p.Ea / (R * T))
    return k * f * np.maximum(1.0 - alpha, 0.0)


def _thomas(a, b, c, d):
    """三对角求解 (a 下对角, b 主对角, c 上对角, d 右端). 原地不修改输入."""
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def simulate(
    p: Params,
    N: int = 81,
    dt: float = 1.0,
    t_end: float = 3600.0,
    t90_target: float = 0.9,
    record_every: int = 0,
) -> dict:
    """运行一维床层吸氢, 返回指标.

    数值方案: 扩散项 backward Euler (三对角隐式, 无条件稳定), 反应源项显式.
    两端口 Dirichlet T=T_bath. 初始 T=T_bath, alpha=0.
    """
    if N < 3:
        raise ValueError("N >= 3")
    dx = p.L / (N - 1)
    T = np.full(N, p.T_bath)
    alpha = np.zeros(N)

    a = np.full(N - 2, -p.k_eff / dx**2)
    b0 = p.rho_c / dt + 2.0 * p.k_eff / dx**2
    b = np.full(N - 2, b0)
    c = np.full(N - 2, -p.k_eff / dx**2)

    t = 0.0
    steps = int(round(t_end / dt))
    # 能量账本 (只记内部节点 —— 两端 Dirichlet 半胞的反应热直接进壁面):
    #   q_rel = 内部节点反应放热; q_store = 显热增量; q_loss = 两端导出
    # 守恒式: q_rel = q_store + q_loss (离散格式精确成立)
    q_rel = 0.0
    q_wall_reaction = 0.0
    q_loss = 0.0
    q_store0 = 0.0
    T_max = float(T.max())
    t90 = None
    hist = {"t": [], "alpha_mean": [], "T_mean": [], "T_max": []}
    alpha_mean = float(alpha.mean())
    a_prev = alpha_mean

    for n in range(steps):
        # 显式反应 (用上一时刻 T/alpha)
        r = rate(alpha, T, p)
        dalpha = dt * r
        # 不能吸超过剩余容量
        dalpha = np.minimum(dalpha, np.maximum(1.0 - alpha, 0.0))
        alpha = alpha + dalpha
        q_rel += float(
            np.sum(np.abs(p.dH) * p.c_max * dalpha[1:-1] * dx)
        )
        q_wall_reaction += float(
            np.abs(p.dH) * p.c_max * (dalpha[0] + dalpha[-1]) * dx * 0.5
        )

        S = np.abs(p.dH) * p.c_max * (dalpha / dt)
        rhs = (p.rho_c / dt) * T[1:-1] + S[1:-1]
        # Dirichlet 边界贡献
        rhs[0] += (p.k_eff / dx**2) * p.T_bath
        rhs[-1] += (p.k_eff / dx**2) * p.T_bath
        T_in = _thomas(a, b, c, rhs)
        T_new = np.empty(N)
        T_new[0] = p.T_bath
        T_new[-1] = p.T_bath
        T_new[1:-1] = T_in

        # 边界热流 (显式: 用更新后温度)
        flux_l = p.k_eff * (T_new[1] - T_new[0]) / dx
        flux_r = p.k_eff * (T_new[-2] - T_new[-1]) / dx
        q_loss += float((flux_l + flux_r) * dt)  # 流入为正 → 与 q_rel 同号约定
        T = T_new
        t += dt
        T_max = max(T_max, float(T.max()))
        a_prev = alpha_mean
        alpha_mean = float(alpha.mean())
        if t90 is None and alpha_mean >= t90_target:
            # 亚步长插值: 步长内 alpha 线性增长 → t90 不受 dt 量化限制
            if alpha_mean > a_prev:
                t90 = (t - dt) + dt * (t90_target - a_prev) / (alpha_mean - a_prev)
            else:
                t90 = t
        if record_every and (n % record_every == 0):
            hist["t"].append(t)
            hist["alpha_mean"].append(alpha_mean)
            hist["T_mean"].append(float(T.mean()))
            hist["T_max"].append(float(T.max()))

    q_store = float(np.sum(p.rho_c * (T - p.T_bath) * dx)) - q_store0
    # 能量守恒: 放热 = 显热增量 + 边界净导出 (q_loss 以导出为正)
    imbalance = q_rel - (q_store + q_loss)
    denom = max(abs(q_rel), 1e-30)
    return {
        "t90": t90,
        "alpha_final": float(alpha.mean()),
        "alpha_profile": alpha.tolist(),
        "T_max": T_max,
        "T_mean_final": float(T.mean()),
        "q_rel": q_rel,
        "q_wall_reaction": q_wall_reaction,
        "q_store": q_store,
        "q_loss": q_loss,
        "energy_imbalance_rel": float(imbalance / denom),
        "N": N,
        "dt": dt,
        "steps": steps,
        "hist": hist,
        "T": T.tolist(),
    }


def lumped_isothermal_alpha(t: float, p: Params) -> float:
    """等温集总解析解: dalpha/dt = k f (1-alpha) → alpha = 1-exp(-k f t)."""
    kf = float(rate(0.0, p.T_bath, p))
    return 1.0 - np.exp(-kf * t)


def adiabatic_delta_T(p: Params) -> float:
    """绝热 (k_eff=0) 满吸氢温升解析式: |dH| c_max / rho_c."""
    return p.energy_density / p.rho_c
