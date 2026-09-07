"""轻量但真实的一维椭圆型 PDE 数值核心（纯 Python，可复现）.

求解  -u'' = f  on [0,1],  u(0)=u(1)=0.  误差/收敛阶均为真实计算.

方法:
  - fem_linear : 线性拉格朗日有限元   (H1≈O(h), L2≈O(h²))
  - iga_p2     : 二次 B 样条 等几何分析 (H1≈O(h²), L2≈O(h³), C¹)
  - gfem_kink  : 广义有限元 PUM, 内部 kink 处加 |x-c| 全局富集，容纳奇性

每个方法返回 Solution(u(x), up(x)) 可求值, 便于统一算误差。
"""
from __future__ import annotations

import math


# ── Gauss 积分 ────────────────────────────────────────────────
_GL4 = [
    (-0.8611363115940526, 0.3478548451374538),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538),
]


def _gauss(fn, a, b):
    s = 0.0
    m = 0.5 * (b - a); cc = 0.5 * (b + a)
    for x, w in _GL4:
        s += w * fn(m * x + cc)
    return m * s


def solve_tridiag(A, b, n):
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        for r in range(n):
            if r != col and abs(M[r][col]) > 1e-300:
                f = M[r][col] / d
                for c in range(col, n + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][n] / M[i][i] for i in range(n)]


class Solution:
    def __init__(self, val, der, nodes=None, dof=0, h=1.0, method=""):
        self.val = val; self.der = der; self.nodes = nodes
        self.dof = dof; self.h = h; self.method = method


def errors(sol: Solution, exact, exact_der):
    """L2 与 H1 半范误差."""
    l2 = h1 = 0.0
    for i in range(len(sol.nodes) - 1):
        a, b = sol.nodes[i], sol.nodes[i + 1]
        l2 += _gauss(lambda x: (sol.val(x) - exact(x)) ** 2, a, b)
        h1 += _gauss(lambda x: (sol.der(x) - exact_der(x)) ** 2, a, b)
    return {"L2": math.sqrt(l2), "H1": math.sqrt(h1), "dof": sol.dof, "h": sol.h, "method": sol.method}


def estimate_order(err_by_ne: list[tuple[int, float]]) -> float:
    """由 (ne, error) 序列用最后两段 log-log 斜率估收敛阶 p (error≈C h^p)."""
    if len(err_by_ne) < 2:
        return 0.0
    (n1, e1), (n2, e2) = err_by_ne[-2], err_by_ne[-1]
    return math.log(e1 / e2) / math.log(n2 / n1) if e2 and n2 != n1 else 0.0


# ────────────────────────── 线性有限元 ─────────────────────────
def fem_linear(ne, f_fn, exact, exact_der):
    h = 1.0 / ne
    nodes = [i * h for i in range(ne + 1)]
    n = ne - 1
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for e in range(ne):
        x0, x1 = e * h, (e + 1) * h
        kw = [[1.0 / h, -1.0 / h], [-1.0 / h, 1.0 / h]]
        for ii, ri in enumerate((e - 1, e)):
            ri2 = ri
            if ri2 < 0 or ri2 >= n:
                continue
            for jj, cj in enumerate((e - 1, e)):
                if cj < 0 or cj >= n:
                    continue
                A[ri2][cj] += kw[ii][jj]
        fl = _gauss(lambda x: f_fn(x) * (x1 - x) / h, x0, x1)
        fr = _gauss(lambda x: f_fn(x) * (x - x0) / h, x0, x1)
        if e - 1 >= 0:
            b[e - 1] += fl
        if e < n:
            b[e] += fr
    uh = solve_tridiag(A, b, n)
    u = [0.0] + uh + [0.0]
    def val(x):
        i = min(int(x / h), ne - 1)
        return u[i] + (x - nodes[i]) / h * (u[i + 1] - u[i])
    def der(x):
        i = min(int(x / h), ne - 1)
        return (u[i + 1] - u[i]) / h
    return Solution(val, der, nodes, dof=n, h=h, method="fem_linear")


# ──────────────────── 单个 B 样条基函数 (Cox-de Boor) ─────────────
def bsp_basis(K, i, p, u):
    """单条 clamped B-spline N_{i,p}(u) 的值 (Cox-de Boor 递归)."""
    if p == 0:
        return 1.0 if (K[i] <= u < K[i + 1]) else 0.0
    d1 = K[i + p] - K[i]
    d2 = K[i + p + 1] - K[i + 1]
    a = (u - K[i]) / d1 if d1 != 0 else 0.0
    b = (K[i + p + 1] - u) / d2 if d2 != 0 else 0.0
    return a * bsp_basis(K, i, p - 1, u) + b * bsp_basis(K, i + 1, p - 1, u)


def bsp_der(K, i, p, u):
    """N'_{i,p}(u) = p[ N_{i,p-1}/(k_{i+p}-k_i) - N_{i+1,p-1}/(k_{i+p+1}-k_{i+1}) ]."""
    d1 = K[i + p] - K[i]
    d2 = K[i + p + 1] - K[i + 1]
    v1 = bsp_basis(K, i, p - 1, u) if d1 != 0 else 0.0
    v2 = bsp_basis(K, i + 1, p - 1, u) if d2 != 0 else 0.0
    return p * ((1.0 / d1 if d1 else 0.0) * v1 - (1.0 / d2 if d2 else 0.0) * v2)


# ──────────────────────── 二次 B 样条 IGA ────────────────────────
def iga_p2(ne, f_fn, exact, exact_der):
    p = 2
    nctrl = ne + p
    K = [0.0] * (p + 1) + [float(i) / ne for i in range(1, ne)] + [1.0] * (p + 1)
    A = [[0.0] * nctrl for _ in range(nctrl)]
    bvec = [0.0] * nctrl
    h = 1.0 / ne
    for e in range(ne):
        a, bb = e * h, (e + 1) * h
        for gp, gw in _GL4:
            u = 0.5 * (a + bb) + 0.5 * (bb - a) * gp
            jac = 0.5 * (bb - a)
            active = []
            for i in range(nctrl):
                if K[i] - 1e-12 <= u <= K[i + p + 1] + 1e-12:
                    active.append(i)
            active = list(set(active))
            for ii, i in enumerate(active):
                Ni = bsp_basis(K, i, p, u)
                dNi = bsp_der(K, i, p, u)
                if Ni == 0.0 and dNi == 0.0:
                    continue
                for jj, j in enumerate(active):
                    Nj = bsp_basis(K, j, p, u)
                    dNj = bsp_der(K, j, p, u)
                    A[i][j] += dNi * dNj * jac * gw
                bvec[i] += f_fn(u) * Ni * jac * gw
    # Dirichlet: clamp→ only ctrl0 & ctrl_{n-1} nonzero at ends; set=0, eliminate
    interior = list(range(1, nctrl - 1))
    m = len(interior)
    As = [[A[i][j] for j in interior] for i in interior]
    bs = [bvec[i] for i in interior]
    uc = solve_tridiag(As, bs, m)
    ctrl = [0.0] * nctrl
    for k2, gi in enumerate(interior):
        ctrl[gi] = uc[k2]
    def val(x):
        return sum(ctrl[i] * bsp_basis(K, i, p, x) for i in range(nctrl))
    def der(x):
        return sum(ctrl[i] * bsp_der(K, i, p, x) for i in range(nctrl))
    nodes = [i * h for i in range(ne + 1)]
    return Solution(val, der, nodes, dof=m, h=h, method="iga_p2")


# ──────────────────── GFEM(PUM) 内部 kink 全局富集 ─────────────────
def gfem_kink(ne, f_fn, exact, exact_der, kc=0.5, A=1.0):
    """u_ref = A|x-kc| + A(2kc-1)x - A*kc  (满足 0 边值), f=-2A δ_c."""
    h = 1.0 / ne
    nodes = [i * h for i in range(ne + 1)]
    nreg = ne - 1
    nd = nreg + 1          # 常规内部 + 1 富集 a
    A_m = [[0.0] * nd for _ in range(nd)]
    bvec = [0.0] * nd

    def psi(x): return abs(x - kc)
    def dpsi(x): return -1.0 if x < kc else 1.0

    idx = lambda node: node - 1 if 1 <= node <= ne - 1 else None  # 常规内部自由度

    for e in range(ne):
        x0, x1 = e * h, (e + 1) * h
        for gp, gw in _GL4:
            x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * gp
            jac = 0.5 * (x1 - x0)
            i0 = e; i1 = e + 1
            # 常规线性形函数
            shapes = [(i0, (x1 - x) / h, -1.0 / h), (i1, (x - x0) / h, 1.0 / h)]
            # 富集基 E=ψ(u), 其导数 = dpsi
            for (node, N, dN) in shapes:
                r = idx(node)
                if r is None:
                    continue
                rE = nreg  # 富集自由度行
                # A_NN
                for (node2, N2, dN2) in shapes:
                    c = idx(node2)
                    if c is not None:
                        A_m[r][c] += dN * dN2 * jac * gw
                # cross N-E
                A_m[r][rE] += dN * dpsi(x) * jac * gw
                A_m[rE][r] += dpsi(x) * dN * jac * gw
            # E-E
            A_m[rE][rE] += dpsi(x) * dpsi(x) * jac * gw
    # RHS: f = -2A δ_c → 只有 kink 处的常规节点得到 -2A; 富集基 ψ(c)=0, 故 b[E]=0
    c_node = min(round(kc / h), ne - 1)
    rk = idx(c_node)
    if rk is not None:
        bvec[rk] += -2.0 * A
    uc = solve_tridiag(A_m, bvec, nd)
    u_regular = [0.0] + uc[:nreg] + [0.0]
    aE = uc[nreg]

    def val(x):
        i = min(int(x / h), ne - 1)
        lin = u_regular[i] + (x - nodes[i]) / h * (u_regular[i + 1] - u_regular[i])
        return lin + aE * psi(x)
    def der(x):
        i = min(int(x / h), ne - 1)
        return (u_regular[i + 1] - u_regular[i]) / h + aE * dpsi(x)
    return Solution(val, der, nodes, dof=nd, h=h, method=f"gfem_kink(aE={aE:.4f})")


# ─────────────────────── 平滑真解用例 (sin-pi) ────────────────────
def exact_sin(x): return math.sin(math.pi * x)
def exact_sin_der(x): return math.pi * math.cos(math.pi * x)
def force_sin(x): return math.pi ** 2 * math.sin(math.pi * x)


def kink_exact(kc=0.5, A=1.0):
    def u(x): return A * abs(x - kc) + A * (2 * kc - 1) * x - A * kc
    def du(x): return (A * (2 * kc - 1) + (-A if x < kc else A))
    return u, du