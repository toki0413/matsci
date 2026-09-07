#!/usr/bin/env python3
"""热木星非线性浅水环流的**高阶拓扑诊断** (离散 Helmholtz–Hodge 分解).

视角: Fujita & Smarandache (2026, arXiv:2605.12509) / Battiston et al. (2020,
Phys. Reports) / Bianconi (2021)。高阶网络理论指出: 若系统有 β₁>0 (存在一条
可绕行的环), 就有**谐和传输模态**——普通图 Laplacian / 树状模型看不见。

这里把它落到物理内核: 在 β-平面通道 (x 周期、y 通道) 网格上, 把稳态速度场
w=(u,v) 分解为正交三分量:

    w = ∇φ (辐散/源汇驱动)  ⊕  rotψ (旋度/涡动)  ⊕  H (谐和 = 拓扑锁定的方位角环流 = 超自转)

  * φ: ∇²φ = div(w)  → 壁面 Neumann (∂φ/∂y=0), x 周期  (昼夜间热强迫源汇)
  * ψ: ∇²ψ = −curl(w)→ 壁面 Dirichlet (ψ=0),   x 周期  (局地涡旋动量输运)
  * H = w − ∇φ − rotψ                                       (赤道超自转喷射的拓扑环)

物理意义 (可证伪):
  超自转喷射是一条**方位角闭合环流** —— 它存在正是因为渐域在 x 方向有 β₁>0
  (动力学模态 = 拓扑许可)。Hodge 三分量定理保证 H ⊥ ∇φ ⊥ rotψ, 因此分解把
  "报表里只有一个 total jet"升级为三个正交机制 + 一个自带校验 (三分量互内积≈0,
  若显著非零则分解失效 = 诊断自我证伪)。

输出:
  * harmonic_amp / harmonic_frac   — 拓扑环流(超自转)振幅与占速度场比重
  * divergent_frac / curl_frac     — 辐散(热强迫)与旋度(涡动)机制占比
  * ortho_*                          — 三分量两两互内积(≈0 即分解正确)
  * sweep_harmonic                  — β₁ 强度随 τ_rad 演化 (强冷却→谐和占比放大)

纯 numpy + scipy(稀疏求解), 确定性、可单测、可证伪。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# 纯环路(常数方位角流)可被谐和分量**精确**捕获 (harmonic_frac→1, ortho≈0);
# 一般混合流场的梯度/旋度占比在粗网格上有 ~10-30% 的壁面模板一致性残差
# (离散 div∘grad ≠ 离散 Laplacian 所致), 模块如实保留 ortho_* 作自我证伪。
__all__ = ["hodge_decompose", "topological_circulation", "sweep_harmonic"]


def _build_laplacian(ny: int, nx: int, dx: float, dy: float, *,
                     neumann_y: bool) -> lil_matrix:
    """2D 5-point 离散 Laplacian (∂xx+∂yy). x 周期 (wrap); y 通道边界自选.

    y 通道二阶差:
      * 内部 : (φ_{j+1}−2φ_j+φ_{j−1})/dy²  → 对角 +2/dy², 北/南 −1/dy²
      * Neumann 墙 (neumann_y=True) : 反射 φ_{±1}=φ_{±∓1}
              j=0 : (2φ_1−2φ_0)/dy²        → 对角 −2/dy², 第1行 +2/dy²
              j=ny−1 : (2φ_{ny−2}−2φ_{ny−1})/dy² → 对角 −2/dy², 第ny−2行 +2/dy²
      * Dirichlet 墙 (False) : φ_{±1}=0
              j=0 : (φ_1−2φ_0)/dy²          → 对角 −2/dy², 第1行 +1/dy²
              j=ny−1 : (−2φ_{ny−1}+φ_{ny−2})/dy² → 对角 −2/dy², 第ny−2行 +1/dy²
    """
    cx, cy = 1.0 / dx ** 2, 1.0 / dy ** 2
    A = lil_matrix((ny * nx, ny * nx))
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            # x (periodic): ∂xx
            A[k, k] += 2 * cx
            A[k, j * nx + ((i - 1) % nx)] -= cx
            A[k, j * nx + ((i + 1) % nx)] -= cx
            # y (channel) 分边界显式装配, 避免符号混淆
            def _y_stencil():
                if 0 < j < ny - 1:
                    return (2 * cy, [(j - 1, -cy), (j + 1, -cy)])
                if j == 0:
                    if neumann_y:
                        return (-2 * cy, [(1, 2 * cy)])
                    return (-2 * cy, [(1, cy)])
                # j == ny-1
                if neumann_y:
                    return (-2 * cy, [(ny - 2, 2 * cy)])
                return (-2 * cy, [(ny - 2, cy)])
            diag_y, offs = _y_stencil()
            A[k, k] += diag_y
            for (jj, coef) in offs:
                A[k, jj * nx + i] += coef
    return A


def _ddx(f: np.ndarray, dx: float):
    """∂/∂x, x 周期 (cell-centered 中心差分)."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)


def _ddy(f: np.ndarray, dy: float):
    """∂/∂y, 通道 (顶/底单侧)."""
    out = np.zeros_like(f, dtype=float)
    if f.shape[0] < 3:
        return out
    out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * dy)
    out[0, :] = (f[1, :] - f[0, :]) / dy
    out[-1, :] = (f[-1, :] - f[-2, :]) / dy
    return out


def _div_curl(u, v, dx, dy):
    du_dx = _ddx(u, dx)
    dv_dx = _ddx(v, dx)
    du_dy = _ddy(u, dy)
    dv_dy = _ddy(v, dy)
    return du_dx + dv_dy, dv_dx - du_dy


def hodge_decompose(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> dict:
    """离散 Helmholtz–Hodge 分解: w = ∇φ ⊕ rotψ ⊕ H."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ny, nx = u.shape
    div, curl = _div_curl(u, v, dx, dy)

    # 注: A 装配为 −∇² (正定; 用于求解). 故:
    #   A φ = −div  ⇔  ∇²φ = div   (Neumann y)
    #   A ψ = +curl ⇔  ∇²ψ = −curl (Dirichlet y)
    A_grad = _build_laplacian(ny, nx, dx, dy, neumann_y=True)
    rhs_g = div.ravel()
    phi = spsolve(A_grad.tocsr(), -(rhs_g - np.mean(rhs_g))).reshape(ny, nx)
    phi -= np.mean(phi)
    grad_x = _ddx(phi, dx)
    grad_y = _ddy(phi, dy)

    # ψ: ∇²ψ = −curl
    A_curl = _build_laplacian(ny, nx, dx, dy, neumann_y=False)
    psi = spsolve(A_curl.tocsr(), curl.ravel()).reshape(ny, nx)
    cur_x = _ddy(psi, dy)
    cur_y = -_ddx(psi, dx)

    # 谐和 H = w − ∇φ − rotψ
    hdr_u = u - grad_x - cur_x
    hdr_v = v - grad_y - cur_y

    def _n2(a, b):
        return float(np.sqrt(np.mean(a * a + b * b)))
    n_total = _n2(u, v) or 1e-30
    n_grad = _n2(grad_x, grad_y)
    n_curl = _n2(cur_x, cur_y)
    n_harm = _n2(hdr_u, hdr_v)

    def _ip(a1, a2, b1, b2):
        return abs(float(np.mean(a1 * a2 + b1 * b2))) / (
            (_n2(a1, b1) * _n2(a2, b2)) or 1e-30)

    return {
        "harmonic_amp": round(n_harm, 6),
        "harmonic_frac": round(n_harm / n_total, 6),
        "divergent_frac": round(n_grad / n_total, 6),
        "curl_frac": round(n_curl / n_total, 6),
        "ortho_grad_curl": round(_ip(grad_x, cur_x, grad_y, cur_y), 6),
        "ortho_grad_harm": round(_ip(grad_x, hdr_u, grad_y, hdr_v), 6),
        "ortho_curl_harm": round(_ip(cur_x, hdr_u, cur_y, hdr_v), 6),
        "components": {"gradient": (grad_x, grad_y),
                       "curl": (cur_x, cur_y),
                       "harmonic": (hdr_u, hdr_v),
                       "field": (u, v)},
    }


def topological_circulation(solver) -> dict:
    """对已积分终态的浅水 solver 输出拓扑诊断 (取 H,U,V → 速度 u,v)."""
    H = np.asarray(solver.H, dtype=float)
    U = np.asarray(solver.U, dtype=float)
    V = np.asarray(solver.V, dtype=float)
    h = H + 1e-30
    r = hodge_decompose(U / h, V / h, getattr(solver, "dx"), getattr(solver, "dy"))
    r["note"] = ("谐和分量 H ⊥ ∇φ ⊥ rotψ (Hodge 三分量): H 是被 β₁>0 拓扑许可的"
                 "方位角环流(超自转), 与昼夜热源辐散流正交分离。")
    return {k: val for k, val in r.items() if k != "components"}


def sweep_harmonic(factory, tau_list: list[float], nx: int = 40, ny: int = 20,
                   t_end_s: float = 6.0e5, **kw) -> dict:
    """τ_rad 扫描里追踪谐和(超自转环)占比 → β₁ 强度随热再分配的演化."""
    rows = []
    for tau in tau_list:
        sol = factory(nx=nx, ny=ny, tau_rad=tau, **kw)
        try:
            sol.integrate(t_end_s)
            d = topological_circulation(sol)
            rows.append({"tau_rad_s": tau, "harmonic_amp": d["harmonic_amp"],
                         "harmonic_frac": d["harmonic_frac"],
                         "divergent_frac": d["divergent_frac"],
                         "curl_frac": d["curl_frac"]})
        except Exception as e:  # noqa: BLE001 — 数值发散时如实标注
            rows.append({"tau_rad_s": tau, "error": str(e)})
    return {"sweep": rows,
            "note": "强冷却(小 τ_rad)→源汇本地化→拓扑环流(谐和分量)占比应被放大。"}


if __name__ == "__main__":
    #  1) 拉普拉斯算子自检: L φ ≈ ∇²φ (周期-x / Neumann-y)
    ny, nx, dx, dy = 20, 40, 0.5, 0.5
    yy = np.arange(ny)[:, None] * dy - (ny * dy) / 2
    xx = np.arange(nx)[None, :] * dx
    phi = np.cos(2 * np.pi * xx / (nx * dx)) * np.exp(-0.5 * (yy ** 2))
    A = _build_laplacian(ny, nx, dx, dy, neumann_y=True)
    Lphi = (A @ phi.ravel()).reshape(ny, nx)
    ana = (-(2 * np.pi / (nx * dx)) ** 2 * np.cos(2 * np.pi * xx / (nx * dx))
           * np.exp(-0.5 * yy ** 2)
           + np.cos(2 * np.pi * xx / (nx * dx)) * ((yy ** 2 - 1) * np.exp(-0.5 * yy ** 2)))
    rel = np.abs(-Lphi - ana).max() / (np.abs(ana).max() + 1e-30)  # A = −∇²
    print("laplacian rel-err:", round(rel, 5))

    #  2) Hodge 自检: 合成 = 辐散 + 旋涡 + 常数方位角环 → 谐和≈常数环, 正交≈0
    u = -np.cos(xx) * np.sin(yy) + np.sin(xx) * np.cos(yy) + 2.0
    v = -np.cos(xx) * np.cos(yy) - np.sin(xx) * np.sin(yy)
    d = hodge_decompose(u, v, dx, dy)
    print("hodge:", {k: d[k] for k in ["harmonic_amp", "harmonic_frac",
          "divergent_frac", "curl_frac", "ortho_grad_curl",
          "ortho_grad_harm", "ortho_curl_harm"]})
    print("harmonic u (-2,1):", round(d["components"]["harmonic"][0][1, 4], 3),
          round(d["components"]["harmonic"][1][1, 4], 3))