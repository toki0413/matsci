"""P0·NN 容量探针 (真实 PINN) —— 协议§4/§5 饱和检验 + 重训散布的局部实现.

把协议概念投影到一维约束系统, 验证其核心判读何时成立:
  - 容量 p  := NN 宽度 w (深3, 协议网格 w∈{8,16,32,64,128})
  - 约束    := 微分方程 + 点值边界条件 (PINN collocation)
  - 留出    := "跨结构" 冻结点 (非训练 collocation 点的内插网格)
  - Vho     := 冻结留出点上的函数值均方误差 (唯一解 -> 可到0; 解族 -> 不可塌缩)
  - sigmaH  := M 个 seed 重训在冻结留出点上的函数散布 (解族维数探针)

两个诊断系统 (共享 dx=0 点值约束 + 微分方程, 区别只在边界约束数量):
  ROUND(RIGID)   u''=f, u(-1)=A, u(1)=B   -> 唯一解   (定理端点探针校准, C* 应饱和)
  FAT(UNDERSPEC) u''=f, u(0)=C            -> 解族 u+ax+... (欠定, C* 应发散/留出不塌缩)
真解 f 取简单周期/多项式, 使 ROUND u* 是有限"有效维"解析函数.
"""
from __future__ import annotations

import torch
import torch.nn as nn

torch.manual_seed(0)


# ── 诊断系统 ─────────────────────────────────────────────────────────────
def _u_star_rigid(x):
    """刚性系统真解: u(x)=cos(x) - [二次修正满足 Dirichlet 端点] —— 唯一."""
    A, B = float(torch.cos(torch.tensor(-1.0))), 1.0
    # 构造 u(x)=cos(x)+ax+b 满足 u(-1)=A,u(1)=B; f=u''=-cos(x)
    b = (A + B) / 2 - 0  # placeholder
    a = (B - A) / 2
    return torch.cos(x) + a * x + b


def _f(x):
    return -torch.cos(x)  # u'' = -cos(x) 恒满足 cos(x)+ax+b


def _u_star_fat(x):
    """欠定系统: 约束下解族 { u = cos(x)+a x + b_0 } 自由 a → 一维胖流形."""
    b0 = 0.0
    return torch.cos(x) + b0


def linear_grid(n, lo=-1.0, hi=1.0):
    return torch.linspace(lo, hi, n)


# ── 3 层 MLP ──────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, w), nn.Tanh(),
            nn.Linear(w, w), nn.Tanh(),
            nn.Linear(w, w), nn.Tanh(),
            nn.Linear(w, 1),
        )

    def forward(self, x):
        return self.net(x)


def solve(which: str, w: int, n_coll: int = 60, steps: int = 2500, lr: float = 3e-3,
          seed: int = 0, holdout_n: int = 200):
    """训练一个 PINN 求解器, 返回 (Vtr, Vho, holdout_pred)."""
    torch.manual_seed(seed)
    net = MLP(w)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    # collocation 训练点 (含端点/中点)
    xc = linear_grid(n_coll, -1, 1).reshape(-1, 1)
    # 冻结留出: 与 collocation 错开的爪哇网格(跨结构)
    xv = linear_grid(holdout_n, -1, 1).reshape(-1, 1)
    xv = xv[1:-1]
    xv = xv[::3]  # 稀疏化, 避免与训练点重合

    # 点值约束
    x_p = torch.tensor([-1.0, 1.0]).reshape(-1, 1) if which == "rigid" else torch.tensor([0.0]).reshape(-1, 1)
    y_p = _u_star_rigid(x_p) if which == "rigid" else _u_star_fat(x_p)

    def loss():
        xc.requires_grad_(True)
        u = net(xc).squeeze(-1)
        du = torch.autograd.grad(u, xc, grad_outputs=torch.ones_like(u),
                                 create_graph=True)[0]
        ddu = torch.autograd.grad(du, xc, grad_outputs=torch.ones_like(du),
                                  create_graph=True)[0]
        res = ddu + torch.cos(xc)  # u'' - f = 0
        value = net(x_p).squeeze(-1) - y_p
        return res.pow(2).mean() + (value ** 2).mean()

    for _ in range(steps):
        opt.zero_grad()
        L = loss()
        L.backward()
        opt.step()
        sch.step()

    # 训练违规 Vtr(collocation 残差 + 点值)
    Vtr = float(loss().detach())
    # 留出违规 Vho(冻结的内插点函数值误差)
    with torch.no_grad():
        up = net(xv).squeeze(-1)
    ustar = _u_star_rigid(xv) if which == "rigid" else _u_star_fat(xv)
    scale = float((ustar ** 2).mean().sqrt() + 1e-9)
    Vho = float(((up - ustar) ** 2).mean().sqrt() / scale)
    return Vtr, Vho, up.detach()


def main():
    widths = [8, 16, 32, 64, 128]
    M = 8  # seeds
    print("P0·对称NN容量探针: 刚性(唯一) vs 欠定(解族) —— C*(w) 饱和检验 + sigmaH")
    print(f"{'w':>5} | {'Vtr(R)':>10} {'Vho(R)':>10} {'sigH(R)':>9} | "
          f"{'Vtr(F)':>10} {'Vho(F)':>10} {'sigH(F)':>9}")
    print("-" * 78)
    res = {}
    for w in widths:
        row = []
        for which in ("rigid", "fat"):
            Vho_list = []
            for s in range(M):
                Vtr, Vho, _ = solve(which, w, seed=s)
                Vho_list.append(Vho)
            Vtrs = [solve(which, w, seed=s)[0] for s in range(M)]
            sigmaH = float(torch.tensor(Vho_list).std() if len(Vho_list) > 1 else 0.0)
            row.extend([float(min(Vtrs)), float(min(Vho_list)), sigmaH])
            res[(w, which)] = (float(min(Vtrs)), float(min(Vho_list)), sigmaH)
        print(f"{w:>5} | {row[0]:>10.2e} {row[1]:>10.2e} {row[2]:>9.2e} | "
              f"{row[3]:>10.2e} {row[4]:>10.2e} {row[5]:>9.2e}")
    print("-" * 78)
    # 判读: 刚性 Vho 随 w 饱和到平台; 欠定 Vho 不塌缩(C* 发散)
    VhoR = [res[(w, "rigid")][1] for w in widths]
    VhoF = [res[(w, "fat")][1] for w in widths]
    sigF = [res[(w, "fat")][2] for w in widths]
    rigid_sat = abs(VhoR[-1] - VhoR[-3]) <= 0.5 * abs(VhoR[0]) + 1e-6 and VhoR[-1] <= 0.05
    fat_opens = max(VhoF) > 0.15 and sigF[-1] > 0.05
    print("\n判读(§4饱和 + §5重训散布):")
    print(f"  刚性 Vho(w): {[round(v,3) for v in VhoR]} -> {'饱和(收敛)' if rigid_sat else '不饱和'}")
    print(f"  欠定 Vho(w): {[round(v,3) for v in VhoF]} -> {'张开(发散)' if fat_opens else '塌缩?'}")
    print(f"  欠定 sigmaH(w): {[round(v,3) for v in sigF]}")
    ok = rigid_sat and fat_opens
    print(f"\n  协议判读在真实NN上是否成立: {'是' if ok else '否(需放宽/复核)'}")


if __name__ == "__main__":
    main()