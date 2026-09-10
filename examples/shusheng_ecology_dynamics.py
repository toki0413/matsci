"""held-out 泛化域 · 群体生态动力学 (Lotka–Volterra 捕食–被捕食).

目的(泛化验证): 这是**扣出来的第 4 个零族科学域** —— 与既有三域(rigidity 约束/
fracture 断裂力学/quantum_critical 凝聚态)在学科上零血缘。用它跑**同一套 huginn 机制
零改动**, 得到不掺 in-sample 的 held-out 复用分。若该域能零改动通过统一契约(外置严格
验证器), 说明泛化不只是"为同一个 3 域建的机制"; 若不能, 记录 gap(诚实, 我照记)。

域内容(全部真实数值, 不伪造):
  - exp_equilibrium_stability: L-V 平衡点 + Jacobian 特征值实部 → 线性稳定性判据;
  - exp_lotka_period: scipy.solve_ivp 数值积分, 由捕食者穿越平衡点的次数估振荡周期。
契约: 统一 `{success, summary, objectives}`, objectives 全为 Python 纯数值叶。
"""
from __future__ import annotations

import numpy as np

# 代表性种群参数 (a,b,c,d>0): dx/dt = a·x − b·x·y, dy/dt = c·x·y − d·y
_A, _B, _C, _D = 0.6, 0.4, 0.5, 0.3


def exp_equilibrium_stability(cfg=None) -> dict:
    """Lotka–Volterra 平衡点与 Jacobian 稳定性(真实数值)."""
    a, b, c, d = _A, _B, _C, _D
    # 正平衡 (x*, y*) = (d/c, a/b)
    x_star = d / c
    y_star = a / b
    # 平衡处 Jacobian [[a−b·y*, −b·x*], [c·y*, c·x*−d]]；迹=0 → 中性(纯虚特征值, 极限环)
    A = np.array([[a - b * y_star, -b * x_star],
                  [c * y_star, c * x_star - d]], dtype=float)
    ev = np.linalg.eigvals(A)
    tr_x = round(float(ev[0].real), 6)   # 特征值实部(稳定性判据)
    tr_y = round(float(ev[1].real), 6)
    return {
        "success": True,
        "summary": {"x_star": round(x_star, 6), "y_star": round(y_star, 6),
                    "eigen_real": (tr_x, tr_y)},
        "objectives": {"x_star": round(x_star, 6), "y_star": round(y_star, 6),
                       "trace": round(float(ev.real.sum()), 6)},
    }


def exp_lotka_period(cfg=None) -> dict:
    """数值积分 L-V, 由捕食者穿平衡线的次数估振荡周期(真实数值)."""
    from scipy.integrate import solve_ivp
    a, b, c, d = _A, _B, _C, _D
    y_star = d / c

    def rhs(_t, yy):
        x, yp = yy
        return [a * x - b * x * yp, c * x * yp - d * yp]

    y0 = [2.0, 2.0]
    sol = solve_ivp(rhs, (0.0, 60.0), y0, method="RK45",
                    rtol=1e-9, atol=1e-11, dense_output=True)
    ts = np.linspace(0.0, 60.0, 4000)
    prey = sol.sol(ts)[0]
    # 捕食者? 用被捕食者 x(t) 向上穿越平衡 x*=d/c 的频率估周期
    x_star = d / c
    up = np.where((prey[:-1] < x_star) & (prey[1:] > x_star))[0]
    # 相邻两次向上穿越间隔约为一个周期 T
    if len(up) >= 2:
        periods = np.diff(ts[up])
        period_est = round(float(periods[-1]), 4)
    else:
        period_est = 0.0
    return {
        "success": True,
        "summary": {"n_up_crossings": int(len(up)),
                    "final_prey": round(float(prey[-1]), 4)},
        "objectives": {"period_est": period_est,
                       "final_prey": round(float(prey[-1]), 4)},
    }


def _make_experiments(cycle: int = 1) -> list:
    """与既有域统一: 返回真实物理实验列表(2 个)."""
    from huginn.research import Experiment
    return [
        Experiment("eco_stability", "L-V 平衡点与线性稳定性(held-out)",
                   run=exp_equilibrium_stability),
        Experiment("eco_period", "L-V 振荡周期(数值积分, held-out)",
                   run=exp_lotka_period),
    ]


if __name__ == "__main__":
    for e in _make_experiments():
        print(e.id, e.run())