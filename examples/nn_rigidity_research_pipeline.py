"""端到端真实科研流程: NN 容量探针判据的可靠性(完整闭环).

流程: 开放问题 -> 3 条真实实验(本地确定性数值) -> 书生 Intern-S2 综合 ->
      声明门禁(claim_grounding) -> 正式研究报告.

研究问题: **NN 容量探针测 bootstrap 解空间刚性时, "饱和判据"应依赖哪个自变量?
           N(样本量) 还是 eps(精度截止)? 及优化器财政(C3)如何污染该判据?**

三个真实实验(均本地可复现, 数值来自真实计算, 非编造):
  X1 (eps-判据)     : 固定 N, 扫 eps -> 刚性 cos vs 胖 |x| 的 C*(eps) 形态.
  X2 (优化器财政)   : torch PINN w=32, 训练预算递增 -> 刚性端 Vtr/Vho 是否收敛.
  X3 (约束维数)     : 点值约束数 k=0/1/2 -> 解族维数 => 同解重算的留出散布 sigmaH.

随后书生综合 X1-X3 的轨迹成文, 经声明门禁核对每个数值是否落在轨迹里。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_AGENT = Path(__file__).resolve().parents[1] / "agent"   # 项目根/agent
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

OUT = Path(__file__).resolve().parent / "out" / "nn_rigidity_research"
OUT.mkdir(parents=True, exist_ok=True)
TRACE: list[str] = []


# ── X1: eps 判据 ----------------------------------------------------------
def _poly_cstar(target: str, eps: float, N: int, pmax: int = 48):
    rng = np.random.default_rng(0)
    f = (lambda t: np.cos(t)) if target == "rigid" else (lambda t: np.abs(t))
    x = rng.uniform(-1.5, 1.5, N)
    y = f(x)
    xv = np.linspace(-1.5, 1.5, 1501); yv = f(xv)
    for p in range(2, pmax + 1):
        A = np.stack([x ** (2 * j) for j in range(p // 2 or 1)], axis=1)
        # 偶多项式: p//2 项 0,2,...,2*(p//2-1); 用 encode 偶容量 = 项数
        k = p // 2
        A = np.stack([x ** (2 * j) for j in range(k)], axis=1)
        coef, *_ = np.linalg.lstsq(A.T @ A + 1e-10 * np.eye(k), A.T @ y, rcond=None)
        B = np.stack([xv ** (2 * j) for j in range(k)], axis=1)
        pred = B @ coef
        if float(np.mean((pred - yv) ** 2)) <= eps:
            return float(k)
    return float(pmax // 2 + 1)


def exp_eps_criterion(N: int = 4096) -> dict:
    eps_list = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
    c_rig, c_fat = [], []
    for e in eps_list:
        rr = _poly_cstar("rigid", e, N); ff = _poly_cstar("fat", e, N)
        c_rig.append(rr); c_fat.append(ff)
        TRACE.append(json.dumps({"exp": "X1", "eps": e, "Cstar_rigid": rr, "Cstar_fat": ff}))
    return {"N": N, "eps": eps_list, "cstar_rigid": c_rig, "cstar_fat": c_fat}


# ── X2: 优化器财政(真实 NN) -------------------------------------------------
class MLP(nn.Module):
    def __init__(self, w: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(),
                                 nn.Linear(w, 1))

    def forward(self, x):
        return self.net(x)


def exp_optimizer_finance(w: int = 32, budgets=(800, 3200, 12800)):
    xc = torch.linspace(-1, 1, 60).reshape(-1, 1).requires_grad_(True)
    xp = torch.tensor([-1.0, 1.0]).reshape(-1, 1)
    yp = torch.cos(xp) + 0.2299 * xp  # u(-1)=cos(-1)+..., u(1) -> Dirichlet 端点 (唯一解)
    yp = torch.cos(xp) + torch.tensor([-0.22985, 0.22985]).reshape(-1, 1) + 0.7701
    xv = torch.linspace(-1, 1, 200).reshape(-1, 1)

    def vho(net):
        u = net(xv); ustar = torch.cos(xv) + 0.22985 * xv + 0.7701
        return float(((u - ustar) ** 2).mean().sqrt() / (ustar ** 2).mean().sqrt())

    rows = []
    torch.manual_seed(1)
    for steps in budgets:
        net = MLP(w)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for _ in range(steps):
            opt.zero_grad()
            u = net(xc).squeeze(-1)
            du = torch.autograd.grad(u, xc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            ddu = torch.autograd.grad(du, xc, grad_outputs=torch.ones_like(du), create_graph=True)[0]
            res = ddu + torch.cos(xc.squeeze(-1))
            val = net(xp).squeeze(-1) - yp.squeeze(-1)
            L = res.pow(2).mean() + val.pow(2).mean()
            L.backward(); opt.step(); sch.step()
        vtr_def = L.item() if 'L' in dir() else None
        vh = vho(net)
        # 重新算 Vtr
        with torch.no_grad():
            uu = net(xc).squeeze(-1)
            # 数值二阶导近似用于 Vtr(直接用最后一次 L 即可)
        rows.append({"steps": steps, "w": w, "vtr": float(L.item()), "vho": vh})
        TRACE.append(json.dumps({"exp": "X2", "w": w, "steps": steps,
                                 "vtr_last": float(L.item()), "vho": vh}))
    return {"w": w, "rows": rows}


# ── X3: 约束维数 -> 解族 σH ------------------------------------------------
def exp_constraint_dimension(N: int = 256):
    from numpy.polynomial import legendre as Lg
    import numpy.polynomial.polynomial as PP
    mesh = np.linspace(-1, 1, N)
    bases = 12
    P = Lg.legval(mesh, np.eye(bases)).T          # (N, bases)  Legendre 值
    D2 = np.zeros((N, bases))
    for j in range(bases):
        cj = Lg.leg2poly(np.eye(j + 1)[:, j])     # P_j 幂基系数
        d2 = PP.polyder(cj, 2)
        acc = np.zeros_like(mesh)
        for kk, cc in enumerate(d2):
            acc += cc * mesh ** kk
        D2[:, j] = acc
    fvec = -np.cos(mesh)
    xv = np.linspace(-1, 1, 321)
    Pv = Lg.legval(xv, np.eye(bases)).T

    def solve(k: int, seed: int):
        # 最小二乘: 满足 u''=-cos + 可选 k 个点值约束(位置随机 -> 重解制造 σH)
        rng = np.random.default_rng(seed)
        A = D2.copy(); b = fvec.copy()
        if k > 0:
            idx = rng.choice(N, size=k, replace=False)
            A = np.vstack([D2, 3.0 * P[idx]])
            b = np.concatenate([fvec, 3.0 * np.cos(mesh[idx])])
        c, *_ = np.linalg.lstsq(A.T @ A + 1e-9 * np.eye(bases), A.T @ b, rcond=None)
        return Pv @ c

    sigs = {}
    for k in (0, 1, 2):
        evs = np.stack([solve(k, s) for s in range(6)])     # (6, 321) 冻结留出
        sigma = float(evs.std(axis=0).mean() / (np.abs(evs).mean() + 1e-9))
        sigs[k] = {"sigmaH": sigma, "n_runs": 6, "nullspace_dim_hint": max(0, 2 - k)}
        TRACE.append(json.dumps({"exp": "X3", "point_constraints": k,
                                 "sigmaH": round(sigma, 4)}))
    return sigs


# ── 书生成文 + 门禁 ----------------------------------------------------------
def finalize(x1, x2, x3):
    try:
        from openai import OpenAI
        from huginn.research.program import grounding_verifier
    except Exception as e:
        print("missing dep:", e); sys.exit(1)

    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: INTERNLM_API_KEY not set"); sys.exit(2)

    trace_txt = "\n".join(f"- {t}" for t in TRACE)
    prompt = f"""你是科研论文作者。基于下方"NN 容量探针测 bootstrap 解空间刚性的饱和判据"
真实实验结果轨迹, 撰写一份**完整研究报告**(不要编造轨迹之外的数值; 所有数字必须来自轨迹).
研究问题: 饱和判据应依赖 N 还是 eps? 优化器财政(C3)如何污染该判据? 约束数如何决定解族维数?

==== 真实实验轨迹(唯一数值来源) ====
{trace_txt}

结构: 0 研究问题与假设 / 1 方法(三实验 X1 X2 X3) / 2 结果(引用轨迹数字) /
3 讨论(饱和判据修正、C3 守卫、约束维数) / 4 结论与局限 / 5 下一步。
输出 markdown。标注哪两个实验支持"判据应看 eps 而非 N"这一结论。
"""
    client = OpenAI(api_key=key, base_url=os.environ.get(
        "INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1"))
    r = client.chat.completions.create(
        model=os.environ.get("INTERNLM_MODEL", "intern-s2-preview"),
        messages=[{"role": "user", "content": prompt}], max_tokens=2800, temperature=0.3,
        extra_body={"thinking_mode": False})
    report = r.choices[0].message.content or ""

    verify = grounding_verifier()
    for _ in range(2):
        g = verify(report, TRACE)
        print(f"\n[门禁] {g['verdict']}  unsubstantiated={g['unsubstantiated']}")
        if g["verdict"] in ("pass", "grounded", "accept"):
            break
        _miss = g.get("unsubstantiated") or []
        prompt2 = (f"门禁未通过: 以下数值不在实验轨迹中, 无法溯源: {_miss}。"
                   "请删除或改用轨迹里的真实数值重写报告, 其余不变。")
        rr = client.chat.completions.create(
            model=os.environ.get("INTERNLM_MODEL", "intern-s2-preview"),
            messages=[{"role": "user", "content": prompt2}],
            max_tokens=2800, temperature=0.2, extra_body={"thinking_mode": False})
        report = rr.choices[0].message.content or ""

    (OUT / "research_report.md").write_text(
        f"# 研究: NN 容量探针的饱和判据可靠性\n\n> 书生 Intern-S2 综合 · 门禁 {g['verdict']}\n\n"
        + "## 实验轨迹(证据)\n" + trace_txt + "\n\n---\n\n" + report.strip() + "\n",
        encoding="utf-8")
    print("== 报告已写入 =="); print((OUT / "research_report.md").resolve())
    return g["verdict"]


def main():
    print("== 完整真实科研流程: 问题->实验->书生综合->声明门禁 ==")
    x1 = exp_eps_criterion()
    print("X1 eps判据: rigid", x1["cstar_rigid"], "| fat", x1["cstar_fat"])
    x2 = exp_optimizer_finance()
    print("X2 优化财政: ", x2["rows"])
    x3 = exp_constraint_dimension()
    print("X3 约束维数 sigmaH:", {k: round(v["sigmaH"], 4) for k, v in x3.items()})
    v = finalize(x1, x2, x3)
    print(f"\n== 流程完成: 门禁 {v} ==")


if __name__ == "__main__":
    main()