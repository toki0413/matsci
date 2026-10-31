"""
NLSE 孤子解析-数值一致性验证
=============================

方程 (无量纲聚焦 NLSE, 标准形式):
    i psi_t + (1/2) psi_xx + |psi|^2 psi = 0

单孤子解析解 (速度 v, 振幅 A=1):
    psi(x,t) = A * sech( A*(x - x0 - v*t) ) * exp( i*v*x + i*(v^2/2 - A^2/2)*t )
本实现取 A=1, x0=0, v=0.5 -> 相位系数 c = v^2/2 - 1/2 = -0.375

数值方法: 分裂步傅里叶法 (SSFM / split-step Fourier)
    psi(t+dt) = exp(i*dt*|psi|^2) * F^{-1}[ exp(-i*k^2*dt/2) * F[psi] ]
    - 线性步 (色散) 精确在频域求解
    - 非线性步 精确在时域求解
    - 一阶 Lie 分裂; 误差 O(dt^2) 每步, 全局 O(dt)

守恒量 (由 Noether / 方程结构直接推导):
    质量 M = int |psi|^2 dx
    能量 E = int [ 0.5*|psi_x|^2 - 0.5*|psi|^4 ] dx
    动量 P = int Im( conj(psi) * psi_x ) dx
出处: 标准 NLSE 守恒律, 见 e.g. Sulem & Sulem "The Nonlinear Schrödinger
      Equation", Ch.2; 或 Ablowitz & Segur "Solitons and the Inverse
      Scattering Transform".

解析解的守恒量 (A=1, v 任意, 全空间积分):
    M = int sech^2 = 2
    E = int[0.5*|psi_x|^2 - 0.5*|psi|^4]
        |psi_x|^2 = v^2*sech^2 + sech^2*tanh^2
        -> E = 0.5*(v^2*2 + 2/3) - 0.5*(4/3) = v^2 - 1/3
    P = int Im(conj(psi)*psi_x) = int v*sech^2 = 2v
验证: v=0.5 -> M=2, E=0.25-0.33333=-0.083333, P=1.0
"""

import json
import os

import numpy as np

# --------------------------------------------------------------------------
# 参数 (全部显式声明, 无魔法数)
# --------------------------------------------------------------------------
L = 80.0          # 计算域长度 (周期边界)
N = 2048          # 空间网格数  -> dx = L/N = 0.0390625
V = 0.5           # 孤子速度
TMAX = 10.0       # 目标演化时间
SEED = 20241028   # 固定随机种子 (本脚本确定性, 但保留以符合交付要求)

DX = L / N
X = np.linspace(-L / 2, L / 2, N, endpoint=False)
K = 2 * np.pi * np.fft.fftfreq(N, d=DX)


# --------------------------------------------------------------------------
# 解析解
# --------------------------------------------------------------------------
def analytic(x, t, v=V, A=1.0, x0=0.0, phase_coef=None):
    """单孤子精确解.

    psi = A * sech(A*(x - x0 - v*t)) * exp(i*v*x + i*c*t)
    代入 PDE 得 c = v^2/2 - A^2/2 (相位系数, 由 residual() 数值验证).
    """
    c = (v * v / 2.0 - A * A / 2.0) if phase_coef is None else phase_coef
    return A / np.cosh(A * (x - x0 - v * t)) * np.exp(1j * (v * x + c * t))


def residual(v=0.5, h=1e-6):
    """用中心差分检验解析解满足 PDE 的程度 (应随 h->0 趋于 0)."""
    out = {}
    for c in [0.5 - 0.5 * v * v, 0.5 - 0.25 * v * v, 1.0 - 0.5 * v * v,
              v * v / 2.0 - 0.5]:
        t = 1.0
        psi = analytic(x=X, t=t, v=v, phase_coef=c)
        psi_t = (analytic(X, t + h, v, phase_coef=c)
                 - analytic(X, t - h, v, phase_coef=c)) / (2 * h)
        psi_xx = (analytic(X + h, t, v, phase_coef=c)
                  - 2 * psi
                  + analytic(X - h, t, v, phase_coef=c)) / h ** 2
        res = 1j * psi_t + 0.5 * psi_xx + np.abs(psi) ** 2 * psi
        out[float(c)] = float(np.max(np.abs(res)))
    return out


# --------------------------------------------------------------------------
# SSFM 求解器
# --------------------------------------------------------------------------
def ssfm(dt=1e-3, tmax=TMAX, v=V, N=N, L=L, order=1):
    """分裂步傅里叶法求解 NLSE.

    order=1: Lie 分裂 (先线性后非线性, 参考脚本的写法).
    order=2: Strang 分裂 (半步非线性-全步线性-半步非线性), 全局 O(dt^2).
    """
    xx = np.linspace(-L / 2, L / 2, N, endpoint=False)
    kk = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
    dx = L / N

    psi = analytic(xx, 0.0, v).copy()
    lin = np.exp(-0.5j * kk ** 2 * dt)          # 线性传播子
    nsteps = int(round(tmax / dt))

    def drifts(p):
        m = np.sum(np.abs(p) ** 2) * dx
        pk = np.fft.fft(p)
        px = np.fft.ifft(1j * kk * pk)
        E = np.sum(0.5 * np.abs(px) ** 2 - 0.5 * np.abs(p) ** 4) * dx
        P = np.sum(np.imag(np.conj(p) * px)) * dx
        return m, E, P

    m0, E0, P0 = drifts(psi)
    snapshots = {}

    def record(n, t):
        tgt = analytic(xx, t, v)
        l2 = np.sqrt(np.sum(np.abs(psi - tgt) ** 2) * dx)
        m, E, P = drifts(psi)
        snapshots[round(t, 6)] = dict(
            L2=l2,
            dM=abs(m - m0) / m0,
            dE=abs(E - E0) / abs(E0),
            dP=abs(P - P0) / abs(P0) if abs(P0) > 0 else float('nan'),
        )

    for n in range(1, nsteps + 1):
        if order == 1:
            psi = np.fft.ifft(lin * np.fft.fft(psi))
            psi = psi * np.exp(1j * np.abs(psi) ** 2 * dt)
        else:  # Strang
            psi = psi * np.exp(1j * np.abs(psi) ** 2 * dt / 2)
            psi = np.fft.ifft(lin * np.fft.fft(psi))
            psi = psi * np.exp(1j * np.abs(psi) ** 2 * dt / 2)
        t = n * dt
        if n % max(1, int(round(2.5 / dt))) == 0 or n == nsteps:
            record(n, t)
    return snapshots, (m0, E0, P0)


# --------------------------------------------------------------------------
# 解析守恒量 (闭式)
# --------------------------------------------------------------------------
def analytic_conserved(v=V, A=1.0):
    """M = 2A, E = A*v^2 - (2/3)A^3, P = 2*A*v  (全空间积分)."""
    M = 2 * A
    E = A * v * v - (2.0 / 3.0) * A ** 3
    P = 2 * A * v
    return M, E, P


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main():
    np.random.seed(SEED)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    os.makedirs(out_dir, exist_ok=True)

    # --- (1) 解析解自洽: 相位系数必须使 PDE 残差 -> 0 ---
    print("=== (1) 解析解残差 (判断相位系数) ===")
    res = residual()
    for c, r in res.items():
        print(f"  c={c:+.6f}  max|residual|={r:.3e}")
    c_true = v2c = V ** 2 / 2 - 0.5
    assert abs(res[c_true]) < 1e-4, f"解析解不自洽: c={c_true} 残差={res[c_true]}"

    # --- (2) 解析守恒量 vs 数值初始值 ---
    M_a, E_a, P_a = analytic_conserved()
    print(f"\n=== (2) 解析守恒量 (v={V}) ===")
    print(f"  M={M_a:.6f}  E={E_a:.6f}  P={P_a:.6f}")

    # --- (3) 主运行: SSFM (Lie, dt=1e-3) 到 t=10 ---
    print("\n=== (3) SSFM 演化 (Lie 分裂, dt=1e-3) ===")
    snaps, (m0, e0, p0) = ssfm(1e-3)
    for t, d in snaps.items():
        print(f"  t={t:5.1f}  L2={d['L2']:.3e}  dM={d['dM']:.1e}  "
              f"dE={d['dE']:.1e}  dP={d['dP']:.1e}")

    # --- (4) 时间步收敛: dt = 1e-3, 5e-4, 2.5e-4 (t=10) ---
    print("\n=== (4) 时间步收敛 (t=10) ===")
    conv = {}
    for dt in [1e-3, 5e-4, 2.5e-4]:
        s, _ = ssfm(dt)
        d = s[max(s)]
        conv[dt] = dict(L2=d['L2'], dE=d['dE'])
        print(f"  dt={dt:g}  L2(t=10)={d['L2']:.3e}  dE={d['dE']:.1e}")

    # --- (5) 独立校验 A: Strang 分裂 (二阶) 在 dt=1e-3 下的结果 ---
    print("\n=== (5) 独立校验: Strang 分裂 (dt=1e-3) ===")
    s2, _ = ssfm(1e-3, order=2)
    d2 = s2[max(s2)]
    print(f"  t=10  L2={d2['L2']:.3e}  dE={d2['dE']:.1e}")

    # --- (6) 独立校验 B: 空间收敛 (N=2048 vs 4096, dt=2.5e-4) ---
    print("\n=== (6) 独立校验: 空间收敛 (dt=2.5e-4) ===")
    spat = {}
    for NN in [2048, 4096]:
        s, _ = ssfm(2.5e-4, N=NN)
        d = s[max(s)]
        spat[NN] = d['L2']
        print(f"  N={NN}  L2(t=10)={d['L2']:.3e}")

    # --- (7) 极限情形: v=0 (静止孤子) 应同样稳定 ---
    print("\n=== (7) 极限情形: v=0 静止孤子 (dt=1e-3) ===")
    s0, _ = ssfm(1e-3, v=0.0)
    d0 = s0[max(s0)]
    print(f"  t=10  L2={d0['L2']:.3e}  dE={d0['dE']:.1e}")

    # --- 组装 results.json ---
    results = dict(
        equation="i psi_t + 0.5 psi_xx + |psi|^2 psi = 0",
        analytic=dict(
            form="A*sech(A*(x-x0-v*t))*exp(i*v*x + i*(v^2/2-A^2/2)*t)",
            A=1.0, x0=0.0, v=V,
            phase_coef=c_true,
            residual_max_h1e6=res[c_true],
            conserved=dict(M=M_a, E=E_a, P=P_a),
        ),
        numerical=dict(
            method="SSFM (split-step Fourier, Lie splitting)",
            L=L, N=N, dx=DX, dt=1e-3, tmax=TMAX,
            nsteps=int(round(TMAX / 1e-3)),
        ),
        main_run={k: v for k, v in snaps.items()},
        time_convergence={str(k): v for k, v in conv.items()},
        validation_strang=dict(L2=d2['L2'], dE=d2['dE']),
        validation_spatial={str(k): v for k, v in spat.items()},
        validation_v0=dict(L2=d0['L2'], dE=d0['dE']),
        initial_numerical=dict(M=m0, E=e0, P=p0),
        success=dict(
            l2_t10=snaps[max(snaps)]['L2'],
            dE_t10=snaps[max(snaps)]['dE'],
            dM_t10=snaps[max(snaps)]['dM'],
            dP_t10=snaps[max(snaps)]['dP'],
            l2_lt_1e6=snaps[max(snaps)]['L2'] < 1e-6,
            dE_lt_1e8=snaps[max(snaps)]['dE'] < 1e-8,
        ),
    )

    path = os.path.join(out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[write] {path}")
    return results


if __name__ == "__main__":
    main()