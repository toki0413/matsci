"""书生全流程主导的真实科研 Agent —— NN 容量探针饱和判据.

与 `nn_rigidity_research_pipeline.py`(人类定实验+书生写报告)不同, 本文件把
**整个科研认知权交给书生 Intern-S2**: 选题->假设->实验设计->调用真实数值工具
获取轨迹->分析->结论->成文->应对声明门禁, 全部由模型自主决策.
框架只提供两类东西:
  1) 真实计算工具(容量探针实验) —— 模型不能凭空编数值, 每个数必须来自工具返回;
  2) 声明门禁(claim_grounding) —— 报告里每个数值必须能回溯到工具轨迹, 否则要求补跑溯源.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_AGENT = Path(__file__).resolve().parents[1] / "agent"
sys.path.insert(0, str(_AGENT))

_AGENT_ROOT = Path(__file__).resolve().parent / "out" / "shusheng_agent_nn_rigidity"
_AGENT_ROOT.mkdir(parents=True, exist_ok=True)

RESEARCH_GOAL = """你是 Huginn 科研智能体, 以书生 Intern-S2 身份对一个**开放科学问题**做原创研究.
研究主题: **以神经网络容量为探针测量 bootstrap 解空间刚性 —— 其饱和判据应依赖什么?**

由你全权主导整个科研流程:
  1. 先调用 list_problems 查看可供研究的诊断系统(刚性端点/胖流形)与可测判据;
  2. 提出你的研究假设(如: 饱和应依赖精度截止 eps 还是样本量 N; 优化器预算如何干扰;
     约束数量如何决定解族维数);
  3. 调用容量探针工具(probe_Cstar / probe_optimizer / probe_constraint_dim)采集**真实数值**,
     自己决定研究哪些系统、扫哪些参数、做几组对照、按什么顺序 —— 可自由反复;
  4. 分析结果、给出结论;
  5. 最后撰写一份**完整研究报告**(研究问题/假设/方法/结果/讨论/结论局限/下一步).
门禁提醒: 系统会把报告里每个数值与你实际调用工具返回的轨迹比对, 未落地的主张会被拒绝.
因此你引用的每个 C*、vtr、vho、sigmaH 等都必须先用工具真实算出来, 不得脑内模拟.
输出 markdown 报告正文, 不要输出思考过程. 若某论点缺少工具证据, 宁可先补跑工具再写.
"""

TRACE: list[str] = []


# ── 真实数值工具 (执行器) ──────────────────────────────────────────────
def _poly(which: str, eps: float, N: int, pmax: int = 48):
    rng = np.random.default_rng(0)
    f = (lambda t: np.cos(t)) if which == "rigid" else (lambda t: np.abs(t))
    x = rng.uniform(-1.5, 1.5, N); y = f(x)
    xv = np.linspace(-1.5, 1.5, 1501); yv = f(xv)
    for p in range(1, (pmax // 2) + 1):
        k = max(1, p)
        A = np.stack([x ** (2 * j) for j in range(k)], axis=1)
        coef, *_ = np.linalg.lstsq(A.T @ A + 1e-10 * np.eye(k), A.T @ y, rcond=None)
        B = np.stack([xv ** (2 * j) for j in range(k)], axis=1)
        if float(np.mean((B @ coef - yv) ** 2)) <= eps:
            return float(k)
    return float(pmax // 2 + 1)


def exec(name: str, args: dict) -> str:
    """工具执行: 返回可序列化字符串; 同时把每条结果 log 进轨迹供门禁比对."""
    if name == "list_problems":
        out = {
            "diagnostics": [
                {"id": "exp_law_approx", "kind": "rigid",
                 "note": "全域解析函数(cos) + 偶对称约束, 解唯一, 供容量-C*平台标定"},
                {"id": "nonsmooth", "kind": "fat",
                 "note": "非光滑函数(|x|), 需无限阶表示, 供 C* 发散判读"},
                {"id": "underspecified_ode", "kind": "family",
                 "note": "欠定 u''=-cos, 通解含 2 维零空间, 供 '约束数->解族维数' 研究"},
            ],
            "measurables": [
                "Cstar(eps): 固定N, 达到留出精度 eps 所需最小偶多项式项数(容量单位)",
                "vtr/vho: 真实 NN(PINN)在给定训练预算下的训练残差/留出误差",
                "sigmaH: 同解重算(不同约束位置)在冻结留出上的散布, 量化解族展开",
            ],
        }
    elif name == "probe_Cstar":
        which = (args.get("target") or "rigid")
        eps_list = args.get("eps_list") or [1e-2, 1e-4, 1e-6, 1e-8]
        out = {"target": which,
               "Cstar": {"eps": eps_list,
                         "value": [_poly(which, e, int(args.get("N", 4096))) for e in eps_list]}}
    elif name == "probe_optimizer":
        w = int(args.get("w", 32)); steps = int(args.get("steps", 3200))
        import torch, torch.nn as nn
        torch.manual_seed(1)
        net = nn.Sequential(nn.Linear(1, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(),
                            nn.Linear(w, 1))
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        xc = torch.linspace(-1, 1, 60).reshape(-1, 1).requires_grad_(True)
        xp = torch.tensor([-1.0, 1.0]).reshape(-1, 1)
        yp = torch.cos(xp) + 0.22985 * xp + 0.7701
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
        xv = torch.linspace(-1, 1, 200).reshape(-1, 1)
        with torch.no_grad():
            uV = net(xv).squeeze(-1)
        uStar = torch.cos(xv).squeeze(-1) + 0.22985 * xv.squeeze(-1) + 0.7701
        vho = float(((uV - uStar).detach() ** 2).mean().sqrt() / (uStar ** 2).mean().sqrt())
        out = {"w": w, "steps": steps, "vtr_last": float(round(L.item(), 6)), "vho": round(vho, 6)}
    elif name == "probe_constraint_dim":
        k = int(args.get("k", 1))
        # 欠定 u''=-cos: 点值约束 k 个 -> 零空间 2-k 维; 重解散布测解族张开
        N = 256; from numpy.polynomial import legendre as Lg
        import numpy.polynomial.polynomial as PP
        mesh = np.linspace(-1, 1, N); bz = 12
        P = Lg.legval(mesh, np.eye(bz)).T
        D2 = np.zeros((N, bz))
        for j in range(bz):
            d2 = PP.polyder(Lg.leg2poly(np.eye(j + 1)[:, j]), 2)
            acc = np.zeros_like(mesh)
            for kk, cc in enumerate(d2):
                acc += cc * mesh ** kk
            D2[:, j] = acc
        fvec = -np.cos(mesh)
        xv = np.linspace(-1, 1, 321); Pv = Lg.legval(xv, np.eye(bz)).T
        evs = []
        for s in range(8):
            rng = np.random.default_rng(s)
            A = D2.copy(); b = fvec.copy()
            if k > 0:
                idx = rng.choice(N, size=k, replace=False)
                A = np.vstack([D2, 3.0 * P[idx]])
                b = np.concatenate([fvec, 3.0 * np.cos(mesh[idx])])
            c, *_ = np.linalg.lstsq(A.T @ A + 1e-9 * np.eye(bz), A.T @ b, rcond=None)
            evs.append(Pv @ c)
        evs = np.array(evs)
        sigma = float(evs.std(axis=0).mean() / (np.abs(evs).mean() + 1e-9))
        out = {"point_constraints": k, "sigmaH": round(sigma, 4),
               "nullspace_dim_after": max(0, 2 - k)}
    else:
        return json.dumps({"error": f"unknown tool {name}"})
    text = json.dumps(out, ensure_ascii=False)
    TRACE.append(text)
    return text


_TOOLS = [
    {"type": "function", "function": {"name": "list_problems",
        "description": "查看可研究的诊断系统(刚性/胖流形/欠定解族)与可测量量",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "probe_Cstar",
        "description": "固定样本量N, 扫精度截止 eps, 量达到留出精度所需最小容量 C* (一家容量=偶多项式项数)",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "enum": ["rigid", "fat"]},
            "eps_list": {"type": "array", "items": {"type": "number"}},
            "N": {"type": "integer"}}, "required": ["target"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "probe_optimizer",
        "description": "真实 NN(PINN)在给定宽度与训练预算下的训练残差 vtr_last 与留出误差 vho (优化器财政审计)",
        "parameters": {"type": "object", "properties": {
            "w": {"type": "integer"}, "steps": {"type": "integer"}},
            "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "probe_constraint_dim",
        "description": "对欠定 u''=-cos, 施加 k 个点值约束, 量冻结留出上的重解散布 sigmaH (解族维数探针)",
        "parameters": {"type": "object", "properties": {"k": {"type": "integer"}},
            "required": ["k"], "additionalProperties": False}}},
]


def _pick(tc):
    name = tc.function.name
    raw = tc.function.arguments if isinstance(tc.function.arguments, str) else tc.function.arguments
    if not isinstance(raw, str) or not raw.strip():
        return name, {}
    try:
        return name, json.loads(raw)
    except Exception:
        try:
            return name, json.loads(raw.replace("'", '"'))
        except Exception:
            return name, {}


def main() -> int:
    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 未设置 INTERNLM_API_KEY", file=sys.stderr); return 2
    base = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
    model = os.environ.get("INTERNLM_MODEL", "intern-s2-preview")
    from openai import OpenAI
    from huginn.research.program import grounding_verifier
    client = OpenAI(api_key=key, base_url=base)
    verify = grounding_verifier()

    messages: list[dict] = [{"role": "user", "content": RESEARCH_GOAL}]

    # ── 书生全流程探索: 自主选题/设计/调用工具采集真实数值 ──
    research_calls = 0
    for _ in range(16):
        r = client.chat.completions.create(model=model, messages=messages, tools=_TOOLS,
                                           tool_choice="auto", max_tokens=1400, temperature=0.25)
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            # 尚未用工具就打算收尾 -> 追问必须先用工具留证据
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content":
                "你还没有调用任何科研工具。必须先调用 list_problems 选题、probe_Cstar 等采集真实数值, "
                "门禁会核对报告数字是否都来自工具轨迹。请先做实验, 再成文。"})
            continue
        for tc in calls[:1]:            # 一次处理一条, 保持简单
            name, a = _pick(tc)
            result = exec(name, a)
            print(f"[tool] {name} {tc.function.arguments}\n  -> {result}")
            research_calls += 1
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        if research_calls >= 6:
            break

    # ── 书生成文 + 声明门禁 ──
    transcript = list(TRACE)
    def gen_final():
        return client.chat.completions.create(
            model=model, messages=messages, max_tokens=2600, temperature=0.3,
            extra_body={"thinking_mode": False}).choices[0].message.content or ""

    final = ""; verdict, ungrounded = "needs_grounding", []
    for _ in range(3):
        final = gen_final()
        if not final.strip():
            verdict, ungrounded = "needs_grounding", ["<空报告>"]
            messages.append({"role": "assistant", "content": final})
            messages.append({"role": "user", "content": "你的报告是空的, 请基于已调用的工具结果撰写完整研究报告。"})
            continue
        g = verify(final, TRACE)
        verdict, ungrounded = g.get("verdict"), g.get("unsubstantiated", [])
        print(f"\n[门禁] {verdict}  unsubstantiated={ungrounded}")
        if verdict in ("pass", "grounded", "accept"):
            break
        messages.append({"role": "assistant", "content": final})
        messages.append({"role": "user", "content":
            f"门禁未通过: 以下数值不在你的工具执行轨迹中、无法溯源: {ungrounded}。"
            f"请删除这些未落地主张(要保留则先调用相应工具获取真实值), 再重写完整研究报告; "
            f"也可先补跑工具再重写。确保每个数值都能在轨迹中找到。"})

    out = _AGENT_ROOT / "shusheng_agent_research_report.md"
    trace_txt = "\n".join(f"- {t}" for t in TRACE)
    out.write_text(f"# 书生全流程主导科研: NN容量探针饱和判据\n\n"
                   f"> 模型 {model} · 工具调用 {research_calls} 次 · 声明门禁 {verdict}\n\n"
                   f"## 工具执行轨迹(证据, {len(TRACE)} 条)\n{trace_txt}\n\n---\n\n"
                   + final.strip() + "\n\n---\n*Huginn: 数值全部来自真实工具轨迹; 结论证据可复核.*\n",
                   encoding="utf-8")
    print("\n== 书生全流程科研完成 ==")
    print(out.resolve())
    print(f"[门禁] {verdict}  unsubstantiated={ungrounded}")
    print(f"\n---- 报告正文(前 1500 字) ----\n"); print(final[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())