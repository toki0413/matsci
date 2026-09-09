"""书生 Intern-S2 + Huginn 完整深研管线(非旁路, 全走 run_research_program).

上一版(`shusheng_agent_nn_rigidity`)是**旁路的裸工具循环**: 只借了
`grounding_verifier`, 选题->实验->成文全部由模型自己在一个循环里对各独立工具做
tool-call, 没有进入 Huginn 的**科研工作流**本身。本文件纠正这一点:
书生作为 LLM 引擎**驱动 `huginn.research.run_research_program`**, 完整走
Huginn 既定工作流:

  1. 真实证据实验 (X1 eps判据 / X2 优化器财政 / X3 约束维数) 注册为 `Experiment`,
     每个分支返回独立目标(objective) + 完整 summary —— 数值全部来自真实计算;
  2. `planner`(需求拆解)把 goal 拆成带依赖的 DAG → 分层:
       层0: X1 (判据基准);  层1: X2, X3 (并行确认, 依赖 X1)
     → 触发 P-A 分层流式结算(layer_epochs) + P-B 层间重规划门(replan_gate)
          + P-C 证据驱动提前终止门(early_stop_gate);
  3. 策略链: Pareto 剪枝 (淘汰被支配假说; 三条独立证据维度互不支配 → 全存活,
        保住多证据方向). 本 demo 刻意不开变异(mutation): 三条实验是独立子结论,
        不是连续参数搜索空间, 变异只会制造与真实轨迹对不上的假说噪声;
        需参数寻优的研究者在上层 open 即可恢复 evolution 闭环;
  4. 书生拿到存活假说真实证据, **自主调用域诊断工具**(diagnostic_tools 里的
     probe_* )采集补充证据, 批判综合成报告;
  5. 声明门禁(claim_grounding) 核对报告每个数值都落在工具轨迹里, 未落地则要求重写;
  6. 真机交不出可落地报告 → 确定性兜底组装(数值全来自真实执行, 必过门禁).

设计约束(诚实边界):
  - 所有 run/工具返回 objectives 与 summary 都由真实计算产生, 无任何编造;
  - objectives 各分支独立(各自支撑不同子结论), 互不支配 → 三个证据分支全部存活,
    保住"多证据方向"而非被单一维度平凡淘汰;
  - `--dry` 走 client=None 确定性综合路径, 无 API key 也能验证整条管线能跑通。

用法:
  INTERNLM_API_KEY=... python examples/shusheng_huginn_workflow.py          # 书生全流程
  python examples/shusheng_huginn_workflow.py --dry                         # 确定性验证管线
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_AGENT = Path(__file__).resolve().parents[1] / "agent"
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

_OUT = Path(__file__).resolve().parent / "out" / "shusheng_huginn_workflow"
_OUT.mkdir(parents=True, exist_ok=True)

GOAL = (
    "以神经网络容量为探针测量 bootstrap 解空间刚性: 饱和判据应依赖什么? "
    "给定 eps(精度截止)主导 vs N(样本量)主导, 优化器财政如何污染该判据, "
    "约束数量如何决定解族维数 —— 寻找多证据方向的稳健研究结论。"
)

# 每个实验独立目标(各自支撑不同子结论; 独立维度互不支配 → 全部存活).
# distinct-key 方案的科学诚实: 不高维拼接一个作假性综合分, 每条证据各自可复核.
ACCURACY_OBJ = ["eps_discrimination", "opt_convergence", "dim_effect"]
_OBJECTIVES = {k: "maximize" for k in ACCURACY_OBJ}

# 收敛聚合头头数预算等通用治理参数走 run_research_program 默认, 本 demo 不再重复.

# ── 真实证据实验 (run 返回 {objectives, summary, success}) ────────────────
_RNG = np.random.default_rng(0)


def _poly_cstar(target: str, eps: float, N: int, pmax: int = 48) -> float:
    """固定 N 下, 达到留出精度 eps 所需最小偶多项式项数 (C* 容量单位). 真实最小二乘."""
    f = (lambda t: np.cos(t)) if target == "rigid" else (lambda t: np.abs(t))
    x = _RNG.uniform(-1.5, 1.5, N); y = f(x)
    xv = np.linspace(-1.5, 1.5, 1501); yv = f(xv)
    for k in range(1, (pmax // 2) + 1):
        A = np.stack([x ** (2 * j) for j in range(k)], axis=1)
        coef, *_ = np.linalg.lstsq(A.T @ A + 1e-10 * np.eye(k), A.T @ y, rcond=None)
        B = np.stack([xv ** (2 * j) for j in range(k)], axis=1)
        if float(np.mean((B @ coef - yv) ** 2)) <= eps:
            return float(k)
    return float(pmax // 2 + 1)


def exp_eps_criterion(N: int = 4096) -> dict:
    """X1 · eps判据: 固定 N 扫 eps, 刚性 cos vs 胖 |x| 的 C* 形态差异."""
    eps_list = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10]
    c_rig = [_poly_cstar("rigid", e, N) for e in eps_list]
    c_fat = [_poly_cstar("fat", e, N) for e in eps_list]
    # 判据区分度: 在最小 eps 处 胖/刚 容量比 —— eps判据能否拉开两类系统(真实标量).
    disc = float(max(c_fat) / (max(c_rig) or 1.0))
    return {
        "objectives": {"eps_discrimination": disc},
        "summary": {"N": N, "eps": eps_list, "cstar_rigid": c_rig, "cstar_fat": c_fat,
                    "eps_discrimination": round(disc, 3)},
        "success": True,
    }


def _exp_optimizer_finance(w: int = 32, steps: int = 1600) -> dict:
    """X2 · 优化器财政: 真实 PINN 在训练预算下的留出误差 vho(刚性唯一解标定)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(1, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(),
                        nn.Linear(w, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    xc = torch.linspace(-1, 1, 60).reshape(-1, 1).requires_grad_(True)
    xp = torch.tensor([-1.0, 1.0]).reshape(-1, 1)
    yp = torch.cos(xp) + 0.22985 * xp + 0.7701
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    vtr_last = None
    for _ in range(steps):
        opt.zero_grad()
        u = net(xc).squeeze(-1)
        du = torch.autograd.grad(u, xc, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        ddu = torch.autograd.grad(du, xc, grad_outputs=torch.ones_like(du), create_graph=True)[0]
        res = ddu + torch.cos(xc.squeeze(-1))
        val = net(xp).squeeze(-1) - yp.squeeze(-1)
        L = res.pow(2).mean() + val.pow(2).mean()
        L.backward(); opt.step(); sch.step()
        vtr_last = float(L.item())
    xv = torch.linspace(-1, 1, 200).reshape(-1, 1)
    with torch.no_grad():
        uV = net(xv).squeeze(-1)
    uStar = torch.cos(xv).squeeze(-1) + 0.22985 * xv.squeeze(-1) + 0.7701
    vho = float(((uV - uStar).detach() ** 2).mean().sqrt() / (uStar ** 2).mean().sqrt())
    # 收敛证据: -log10(vho) 越大 = 越收敛(真实标量; vho 越小越好 → 取负对数最大化).
    conv = float(-math.log10(max(vho, 1e-12)))
    return {
        "objectives": {"opt_convergence": conv},
        "summary": {"w": w, "steps": steps, "vtr_last": round(vtr_last, 6),
                    "vho": round(vho, 6), "opt_convergence": round(conv, 3)},
        "success": True,
    }


def _solve_under(k: int, seed: int, basis: int) -> np.ndarray:
    """欠定 u''=-cos + k 个点值约束 -> 构造解族成员 (真实, 冻结留出上的一个样本).

    不做 min-norm 唯一解(lstsq 恒唯一 → 重解零散布, 测不出"解族维数"); 而是
    在约束线性系统 A c ≈ b 的**零空间**里随机取向, 叠加到特解上 —— 采样得到的
    是同一个真实解族(常数+线性 2 维核), k 个独立点值约束逐条钳制核自由度:
      k=0 -> 核维 2, 散布最大; k=2 -> 核被完全钳制, 散布 ~0. → 解族维数真测量.
    """
    from numpy.polynomial import legendre as Lg
    import numpy.polynomial.polynomial as PP
    N = 256
    mesh = np.linspace(-1, 1, N)
    P = Lg.legval(mesh, np.eye(basis)).T
    D2 = np.zeros((N, basis))
    for j in range(basis):
        d2 = PP.polyder(Lg.leg2poly(np.eye(j + 1)[:, j]), 2)
        acc = np.zeros_like(mesh)
        for kk, cc in enumerate(d2):
            acc += cc * mesh ** kk
        D2[:, j] = acc
    fvec = -np.cos(mesh)
    rng = np.random.default_rng(seed)
    A = D2.copy(); b = fvec.copy()
    if k > 0:
        idx = rng.choice(N, size=k, replace=False)
        A = np.vstack([D2, 3.0 * P[idx]])
        b = np.concatenate([fvec, 3.0 * np.cos(mesh[idx])])
    # 特解(解族的一个成员) + 零空间采样(recover the真 family spread)
    c_particular, *_ = np.linalg.lstsq(A, b, rcond=None)
    _, s, vh = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > 1e-8))
    null_basis = vh[rank:]                       # (null_dim, basis), 零空间方向
    if null_basis.shape[0] > 0:
        # 随机核组合, 归一化到与原特解同数量级 —— 采样的是各处留出上的真实解族展开
        coeff = rng.normal(0.0, 0.5, size=null_basis.shape[0])
        c_family = c_particular + coeff @ null_basis
    else:
        c_family = c_particular                   # 核被完全钳制 -> 唯一解
    xv = np.linspace(-1, 1, 321)
    return Lg.legval(xv, np.eye(basis)).T @ c_family


def exp_constraint_dimension(seeds: int = 8) -> dict:
    """X3 · 约束维数: k=0 相对 k=2 的解族展开 sigmaH 之差(真实解族维数探针)."""
    ev0 = np.stack([_solve_under(0, s, 12) for s in range(seeds)])
    ev2 = np.stack([_solve_under(2, s, 12) for s in range(seeds)])
    sig0 = float(ev0.std(axis=0).mean() / (np.abs(ev0).mean() + 1e-9))
    sig2 = float(ev2.std(axis=0).mean() / (np.abs(ev2).mean() + 1e-9))
    effect = float(sig0 - sig2)          # 约束把解族维数收掉的程度(真实标量, maximize)
    return {
        "objectives": {"dim_effect": effect},
        "summary": {"seeds": seeds, "sigmaH_k0": round(sig0, 4),
                    "sigmaH_k2": round(sig2, 4), "dim_effect": round(effect, 4)},
        "success": True,
    }


def _make_experiments():
    """三条真实证据分支 → Experiment 列表."""
    from huginn.research import Experiment

    specs = [
        (lambda: exp_eps_criterion(), "X1_eps_criterion",
         "eps判据: 固定N扫eps, 刚性cos的C*在阈值后不再随eps收紧而增长(平台), "
         "而胖|x|需无限阶表示 -> 饱和由精度截止eps的判据区分度拉大两类系统."),
        (lambda: _exp_optimizer_finance(), "X2_optimizer_finance",
         "优化器财政: 真实PINN在固定训练预算下的留出误差vho -> 若vho未收敛, "
         "则容量饱和信号会被预算不足的'假饱和'污染(判据须加优化器财政守卫)."),
        (lambda: exp_constraint_dimension(), "X3_constraint_dimension",
         "约束维数: 欠定u''=-cos + k个点值约束 -> 解族维数由约束数决定; "
         "k:0->2 钳制核自由度使重解散布 sigmaH 收掉 -> 约束数才是决定解空间维度的主导量(非N)."),
    ]
    return [Experiment(name=n, hypothesis=h, run=fn) for fn, n, h in specs]


def _build_plan(goal: str, run_by_name: dict):
    """需求拆解 -> 带依赖 DAG -> 分层(层0: X1 基准; 层1: X2,X3 并行确认)."""
    from huginn.research.planning import build_research_plan, SubResearch
    return build_research_plan(goal, [
        SubResearch("X1_eps_criterion", "基准判据: eps 区分刚性/胖",
                    run=run_by_name["X1_eps_criterion"], depends_on=[]),
        SubResearch("X2_optimizer_finance", "优化器财政守卫",
                    run=run_by_name["X2_optimizer_finance"],
                    depends_on=["X1_eps_criterion"]),
        SubResearch("X3_constraint_dimension", "约束维数主导",
                    run=run_by_name["X3_constraint_dimension"],
                    depends_on=["X1_eps_criterion"]),
    ], parallel_cap=2)


def _diagnostic_tools():
    """书生在成文阶段可自主调用的域诊断工具 (返回真实数值, 进 trace 供门禁核对)."""
    def h_probe_cstar(a):
        target = a.get("target", "rigid")
        N = int(a.get("N", 4096))
        eps_list = a.get("eps_list") or [1e-2, 1e-4, 1e-6, 1e-8]
        return json.dumps({"target": target, "Cstar": {
            "eps": eps_list, "value": [_poly_cstar(target, e, N) for e in eps_list]}},
            ensure_ascii=False)

    def h_probe_optimizer(a):
        res = _exp_optimizer_finance(int(a.get("w", 32)), int(a.get("steps", 1600)))
        return json.dumps(res["summary"], ensure_ascii=False)

    def h_probe_constraint(a):
        seeds = int(a.get("seeds", 8))
        evs = []
        for k in (0, 1, 2):
            vals = np.stack([_solve_under(k, s, 12) for s in range(seeds)])
            sig = float(vals.std(axis=0).mean() / (np.abs(vals).mean() + 1e-9))
            evs.append({"k": k, "sigmaH": round(sig, 4)})
        return json.dumps({"sigmaH_by_k": evs}, ensure_ascii=False)

    return [
        {"tool": {"function": {"name": "probe_Cstar",
            "description": "固定N扫eps, 返回刚性/胖系统达到精度eps所需最小容量C*曲线(真实)",
            "parameters": {"type": "object", "properties": {
                "target": {"type": "string", "enum": ["rigid", "fat"]},
                "eps_list": {"type": "array", "items": {"type": "number"}},
                "N": {"type": "integer"}}, "required": ["target"],
                "additionalProperties": False}}}, "handle": h_probe_cstar},
        {"tool": {"function": {"name": "probe_optimizer",
            "description": "真实PINN在给定宽度w与训练预算steps下的训练残差/留出误差vho(优化器财政审计)",
            "parameters": {"type": "object", "properties": {
                "w": {"type": "integer"}, "steps": {"type": "integer"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_optimizer},
        {"tool": {"function": {"name": "probe_constraint_dim",
            "description": "对欠定u''=-cos, 用不同种子数重算k=0/1/2约束下的解族重解散布sigmaH(解族维数探针)",
            "parameters": {"type": "object", "properties": {"seeds": {"type": "integer"}},
                "required": [], "additionalProperties": False}}}, "handle": h_probe_constraint},
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="确定性运行(不调模型), 验证整条管线")
    ap.add_argument("--model", default="intern-s2-preview")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--min-iters", type=int, default=3)
    ap.add_argument("--strictness", type=int, default=0, choices=[0, 1, 2],
                    help="判断层·分级护栏 0/1/2(默认0=纯真伪硬门禁, 信任模型)")
    args = ap.parse_args()

    from huginn.research import run_research_program, grounding_verifier
    from huginn.research import Experiment   # noqa: F401 — re-export 校验

    client = None
    if not args.dry:
        key = os.environ.get("INTERNLM_API_KEY")
        if not key:
            print("error: INTERNLM_API_KEY not set (或 --dry)", file=sys.stderr); return 2
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=args.base_url or
                        os.environ.get("INTERNLM_BASE_URL",
                                       "https://chat.intern-ai.org.cn/api/v1"))

    exps = _make_experiments()
    run_by_name = {e.name: e.run for e in exps}
    plan = _build_plan(GOAL, run_by_name)

    print("== 书生 + Huginn 完整深研管线 ==")
    print(f"goals: {GOAL}")
    print(f"layers: {plan.layers}  topo: {plan.topo_order}  antichain_width={plan.antichain_width}")
    print(f"objectives: {_OBJECTIVES}\n")

    out = run_research_program(
        goal=GOAL,
        experiments=exps,
        objectives_config=_OBJECTIVES,
        client=client, model=args.model, base_url=args.base_url,
        verify=grounding_verifier(),
        out_md=_OUT / "research_report.md",
        planner=lambda _g: plan,          # 需求拆解 -> DAG 分层
        layer_epochs=True,                # P-A: 分层流式结算
        replan_gate=True,                 # P-B: 层间重规划门
        early_stop_gate=True,             # P-C: 证据驱动提前终止门
        diagnostic_tools=_diagnostic_tools(),
        max_iterations=args.max_iters,
        min_iterations=args.min_iters,
        max_parallel=2,
        strictness=args.strictness,       # 判断层·分级护栏(默认0=零提示, 不强锁模型)
    )

    print("\n[program] explored=%d pruned=%d pareto_front=%d convergence=%s mutations=%d"
          % (out.explored, out.pruned, len(out.pareto_front), out.converred, out.mutations))
    for b in out.pareto_front:
        print(f"  surv -> {b['name']}")
    print(f"[gate] {out.verdict} unsubstantiated={out.ungrounded} source={out.report_source}")
    print(f"[护栏] strictness={args.strictness} 判断层软提示 ×{len(out.judgment_hints or [])}")
    if out.consolidated:
        _c = out.consolidated
        print(f"[P-A layers_covered] {_c.get('epochs')}  stream_view_sections={len(_c.get('stream_view') or [])}")
        print(f"[P-B replan] checked={(_c.get('replan') or {}).get('checked')} "
              f"skipped={len((_c.get('replan') or {}).get('skipped', []) or [])}")
        _es = _c.get('early_stop') or {}
        print(f"[P-C early_stop] verdict={_es.get('verdict')} stopped_after_layer={_es.get('stopped_after_layer')}")
    print("报告:", _OUT / "research_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())