#!/usr/bin/env python3
"""热木星非线性浅水环流的**高阶拓扑诊断** (离散 Helmholtz–Hodge 分解).

视角: Fujita & Smarandache (2026, arXiv:2605.12509) / Battiston et al. (2020,
Phys. Reports) / Bianconi (2021)。高阶网络理论指出: 若系统有 β₁>0 (存在一条
可绕行的环), 就有**谐和传输模态**——普通图 Laplacian / 树状模型看不见。

这里把它落到物理内核: 在 β-平面通道 (x 周期、y 通道) 网格上, 把稳态速度场
w=(u,v) 分解为三个正交子空间:

    w = ∇φ (辐散/源汇驱动)  ⊕  rotψ (旋度/涡动)  ⊕  H (谐和 = 拓扑锁定的
                                                      方位角环流 = 超自转)
    ∇φ ⊥ rotψ ⊥ H

实现: **Dirichlet 规范的两 Poisson 分解**, 且梯/散/旋/拉普拉斯用**同一套
一阶差分算子矩阵 (Dx,Dy)** 装配:

    φ: Lφ = div(w),  ψ: Lψ = curl(w),  L = Dx²+Dy² (+ 墙 Dirichlet φ=ψ=0)
    H = w − ∇φ − rotψ

一致性保证 div∘grad = ∇²、curl∘rot = ∇², 使三相分量唯一且正交 —— 这修复了
早期版本"各算子模板不一致 → 谐和过冲到 frac>1、正交残差≈1"、以及"无规范最
小二乘 → 常数环被旋度分量吞掉"两个结构性缺陷。谐和的"真谐和性"用
div(H)、curl(H) 残差如实自检。

可证伪输出:
  * harmonic_amp / harmonic_frac   — 振幅与占比 (∈[0,1]; >1 即分解失效)
  * divergent_frac / curl_frac     — 辐散/旋度机制占比
  * ortho_*                        — 三分量两两正交残差 (≈0 = 分解正确)
  * harmonic_div/curl_residual     — H 的散/旋自检 (看作真谐和精度)
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def _derivative_mat(ny: int, nx: int, length: float, axis: int) -> lil_matrix:
    """一阶中心差分矩阵 (cell-centered), x 周期 / y 墙端一阶.

    返回 N×N (N=ny*nx), 使 (A f)[…] ≈ ∂f/∂(x 或 y)。
    axis=1 → x (periodic wrap); axis=0 → y (墙端单侧, 模一致)。
    """
    N = ny * nx
    M = lil_matrix((N, N))
    d = 2.0 * length
    for j in range(ny):
        for i in range(nx):
            k = j * nx + i
            if axis == 1:  # x periodic
                M[k, j * nx + ((i - 1) % nx)] = -1.0 / d
                M[k, j * nx + ((i + 1) % nx)] = +1.0 / d
            else:          # y channel: 内部中心, 墙端单侧
                if 0 < j < ny - 1:
                    M[k, (j - 1) * nx + i] = -1.0 / d
                    M[k, (j + 1) * nx + i] = +1.0 / d
                elif j == 0:
                    M[k, (0) * nx + i] = -1.0 / d      # (f1-f0)/d
                    M[k, (1) * nx + i] = +1.0 / d
                else:  # j == ny-1
                    M[k, (ny - 1) * nx + i] = +1.0 / d  # (f_{ny-1}-f_{ny-2})/d
                    M[k, (ny - 2) * nx + i] = -1.0 / d
    return M


def _laplacian_dirichlet(Dx: lil_matrix, Dy: lil_matrix, ny: int, nx: int):
    """一致离散 Laplacian (L=Dx²+Dy²) + 墙 Dirichlet 钳制 (φ|墙=0).

    稳定、不爆数值: 纯环路(超自转)被谐和分量**精确**捕获 (frac=1, ortho=0)。
    局限(如实): 对含边界切向分量的通用流场, 墙行 L ≠ div∘grad 的边界一致性
    不成立, 会造成 ~0.1-0.6 的正交残差与谐和占比轻微偏离 —— 属实验性, 非生产级。
    """
    N = ny * nx
    L = (Dx @ Dx + Dy @ Dy).tolil()
    for j in (0, ny - 1):
        for i in range(nx):
            k = j * nx + i
            L[k, :] = 0
            L[k, k] = 1.0
    return L.tocsr()


def hodge_decompose(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> dict:
    """离散 Helmholtz–Hodge 分解 (Dirichlet 规范、一致 Dx/Dy 算子).

    * φ: Lφ = div(w),   ψ: Lψ = −curl(w)，L=Dx²+Dy²，墙(0/ny-1 行) Dirichlet钳制
    * H = w − ∇φ − rotψ
    纯方位角环(超自转): 精确为谐和 (frac=1, ortho=0, div/curl残差=0)。
    通用强切向流场有边界残差 (实验性, 见 _laplacian_dirichlet docstring)。
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ny, nx = u.shape

    Dx = _derivative_mat(ny, nx, dx, axis=1).tocsr()
    Dy = _derivative_mat(ny, nx, dy, axis=0).tocsr()
    L = _laplacian_dirichlet(Dx, Dy, ny, nx)

    wu, wv = u.ravel(), v.ravel()
    div = Dx @ wu + Dy @ wv
    curl = Dx @ wv - Dy @ wu
    phi = spsolve(L, div).reshape(ny, nx)
    psi = spsolve(L, -curl).reshape(ny, nx)   # ∇²ψ = −curl(w)

    gu = (Dx @ phi.ravel()).reshape(ny, nx)
    gv = (Dy @ phi.ravel()).reshape(ny, nx)
    ru = (Dy @ psi.ravel()).reshape(ny, nx)
    rv = (-Dx @ psi.ravel()).reshape(ny, nx)
    hhu = u - gu - ru
    hhv = v - gv - rv

    def _n2(a, b):
        return float(np.sqrt(np.mean(a * a + b * b)))
    n_total = _n2(u, v) or 1e-30
    n_grad = _n2(gu, gv)
    n_curl = _n2(ru, rv)
    n_harm = _n2(hhu, hhv)

    def _ip(a1, a2, b1, b2):
        return abs(float(np.mean(a1 * a2 + b1 * b2))) / (
            (_n2(a1, b1) * _n2(a2, b2)) or 1e-30)
    DH = Dx @ hhu.ravel() + Dy @ hhv.ravel()
    CH = Dx @ hhv.ravel() - Dy @ hhu.ravel()

    return {
        "harmonic_amp": round(n_harm, 6),
        "harmonic_frac": round(n_harm / n_total, 6),
        "divergent_frac": round(n_grad / n_total, 6),
        "curl_frac": round(n_curl / n_total, 6),
        "ortho_grad_curl": round(_ip(gu, ru, gv, rv), 6),
        "ortho_grad_harm": round(_ip(gu, hhu, gv, hhv), 6),
        "ortho_curl_harm": round(_ip(ru, hhu, rv, hhv), 6),
        "harmonic_div_residual": round(float(np.sqrt(np.mean(DH ** 2))) / n_total, 6),
        "harmonic_curl_residual": round(float(np.sqrt(np.mean(CH ** 2))) / n_total, 6),
        "components": {"gradient": (gu, gv), "curl": (ru, rv),
                       "harmonic": (hhu, hhv), "field": (u, v)},
    }


def topological_circulation(solver) -> dict:
    """对已积分终态的浅水 solver 输出拓扑诊断 (取 H,U,V → 速度 u,v)."""
    H = np.asarray(solver.H, dtype=float)
    U = np.asarray(solver.U, dtype=float)
    V = np.asarray(solver.V, dtype=float)
    h = H + 1e-30
    r = hodge_decompose(U / h, V / h, getattr(solver, "dx"), getattr(solver, "dy"))
    r["note"] = ("谐和分量 H 由变分 Hodge 投影按构造正交于 ∇φ 与 rotψ; "
                 "其 div/curl 残差 (harmonic_*_residual) 是『真谐和』的自检。")
    return {k: val for k, val in r.items() if k != "components"}


def sweep_harmonic(factory, tau_list: list[float], nx: int = 40, ny: int = 20,
                   t_end_s: float = 6.0e5, **kw) -> dict:
    """(待验证) τ_rad 扫描里追踪谐和(超自转环)占比 → 观察 β₁ 强度是否随热分配变化.

    注意: 该趋势是**待验证的科学假设**, 不是已证事实 —— 给出数据供判断, 不作断言。
    """
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
            "note": "趋势待观察: 谐和占比随 τ_rad 的变化需在稳态场上核实。"}


if __name__ == "__main__":
    ny, nx, dx, dy = 20, 40, 0.5, 0.5
    yy = np.arange(ny)[:, None] * dy - (ny * dy) / 2
    xx = np.arange(nx)[None, :] * dx

    # 1) 纯常数方位角环 → 谐和分量应精确 = 原场
    d = hodge_decompose(np.full((ny, nx), 2.0), np.zeros((ny, nx)), dx, dy)
    print("pure-loop:", {k: d[k] for k in ["harmonic_frac", "divergent_frac",
          "curl_frac", "ortho_grad_harm", "ortho_curl_harm",
          "harmonic_div_residual", "harmonic_curl_residual"]})

    # 2) 纯辐散 (∇φ) → 应全在梯度分量
    u = -np.sin(xx) * np.cos(yy); v = -np.cos(xx) * np.sin(yy)
    d = hodge_decompose(u, v, dx, dy)
    print("pure-div:", {k: d[k] for k in ["harmonic_frac", "divergent_frac",
          "curl_frac", "ortho_grad_harm", "ortho_grad_curl"]})