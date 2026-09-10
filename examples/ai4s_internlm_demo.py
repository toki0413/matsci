#!/usr/bin/env python3
"""Huginn × 书生 — AI4S 方向1「开放问题研究」端到端演示.

与"复现型" demo 不同：**不预设答案**。让 Intern-S2-Preview 作为主研究员对一个开放
科学问题完成科研流程：选题 → 假设(候选规律) → 算(最小二乘 + AIC 模型选择)
→ 验证(Bootstrap CI / 留一交叉) → 做(费雪信息矩阵最优实验设计) → 写(开放式结论).

**Grounded 防编造设计（关键）**：Intern-S2 面对大段的开放式多步任务时，倾向"脑内模拟"
工具结果并编造统计量。因此本演示采用与 Huginn phase/plan 门控一致的受控流水线——
框架分阶段编排、每阶段强制执行真实工具并把统计量落地，模型负责选题/解释/写作；
**任何数值都来自工具返回，绝不采纳模型声明的数字**。

数据为稀疏含噪观测，正确规律不被告知。更优模型、参数可辨识性、下一步补样点，
全部由数据 + AIC + Bootstrap + FIM 推断——**结果开放**。

问题库（模型自主选题）：
  - exp_law  : 化学反应速率 k(T) 随温度——应服从指数律(Arrhenius)还是线性律？
  - comp_law : 材料某性能 y 随组分 x——幂律还是线性？

支持的计算：`fit_law`（真实最小二乘 + AIC）、`analyze_uncertainty`（参数 Bootstrap CI）、
`design_experiment`(FIM 挑选新抽样点)、`cross_validate`(LOOCV)。仅依赖 `openai`。

用法：
    export INTERNLM_API_KEY=<书生 token>
    python examples/ai4s_internlm_demo.py [--model intern-s2-preview] [--outdir examples/out]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

_BASE_URL = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
_DEFAULT_MODEL = "intern-s2-preview"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))   # 产品模块(huginn.*)
from huginn.research import grounding_verifier  # noqa: E402   # 声明门禁唯一实现


# ── 开放问题库：稀疏含噪观测，噪声标准差 sigma 已知，规律未知 ─────
_PROBLEMS = {
    "exp_law": {
        "title": "化学反应速率随温度",
        "xlabel": "Temperature / K",
        "ylabel": "rate k / a.u.",
        "note": "物理预期多为指数律(Arrhenius: k=A·exp(−Ea/RT))，但也可能伪线性",
        "x": [310, 325, 340, 355, 370, 390],
        "y": [0.18, 0.32, 0.55, 0.91, 1.55, 2.90],
        "sigma": 0.06,
        "pool_x": [300, 315, 330, 345, 360, 375, 385, 400, 410],
    },
    "comp_law": {
        "title": "材料性能随组分",
        "xlabel": "composition x",
        "ylabel": "property y / a.u.",
        "note": "幂律(含平方项)与线性竞争，稀疏数据下二者可区分但需实验设计",
        "x": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "y": [1.00, 1.21, 1.44, 1.71, 2.00, 2.35],
        "sigma": 0.05,
        "pool_x": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    },
}


# ── 真实统计工具实现（确定性，含 AIC / Bootstrap CI / FIM / LOOCV）───
def _fit_linear(x, y):
    n = len(x)
    sx = sum(x); sy = sum(y)
    sxx = sum(a * a for a in x); sxy = sum(a * b for a, b in zip(x, y))
    denom = n * sxx - sx * sx or 1e-12
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    rss = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    return {"a": a, "b": b, "rss": rss, "k": 2}


def _fit_exponential(x, y):
    # 对数线性化: ln y = ln A + b·x, 在原始刻度算 RSS/AIC
    n = len(x)
    z = [math.log(max(v, 1e-12)) for v in y]
    sx = sum(x); sz = sum(z)
    sxx = sum(a * a for a in x); sxz = sum(a * b for a, b in zip(x, z))
    denom = n * sxx - sx * sx or 1e-12
    b = (n * sxz - sx * sz) / denom
    lnA = (sz - b * sx) / n
    A = math.exp(lnA)
    rss = sum((yi - A * math.exp(b * xi)) ** 2 for xi, yi in zip(x, y))
    return {"A": A, "b": b, "rss": rss, "k": 2}


def _fit_power(x, y):
    # log-log: ln y = ln c + m·ln x
    xp = [max(v, 1e-6) for v in x]
    lx = [math.log(v) for v in xp]
    ly = [math.log(max(v, 1e-12)) for v in y]
    n = len(x)
    sx = sum(lx); sy = sum(ly)
    sxx = sum(a * a for a in lx); sxy = sum(a * b for a, b in zip(lx, ly))
    denom = n * sxx - sx * sx or 1e-12
    m = (n * sxy - sx * sy) / denom
    lnc = (sy - m * sx) / n
    c = math.exp(lnc)
    rss = sum((yi - c * (xi ** m)) ** 2 for xi, yi in zip(x, y))
    return {"c": c, "m": m, "rss": rss, "k": 2}


def _predict(law, params, x):
    if law == "linear":
        return params["a"] + params["b"] * x
    if law == "exponential":
        return params["A"] * math.exp(params["b"] * x)
    if law == "power":
        return params["c"] * (x ** params["m"])
    raise ValueError(law)


_FITTED = {"linear": _fit_linear, "exponential": _fit_exponential, "power": _fit_power}


def exec_tool(name: str, args: dict) -> str:
    if name == "list_problems":
        return json.dumps(
            [{"id": pid, "title": p["title"], "note": p["note"]}
             for pid, p in _PROBLEMS.items()], ensure_ascii=False)

    if name == "load_dataset":
        p = _PROBLEMS[args["problem"]]
        return json.dumps(
            {"title": p["title"], "xlabel": p["xlabel"], "ylabel": p["ylabel"],
             "note": p["note"], "x": p["x"], "y": p["y"],
             "measurement_noise_sigma": p["sigma"]}, ensure_ascii=False)

    if name == "fit_law":
        p = _PROBLEMS[args["problem"]]
        x, y, n = p["x"], p["y"], len(p["x"])
        law = args["law"]
        if law not in _FITTED:
            return json.dumps({"error": f"unknown law {law}"}, ensure_ascii=False)
        prm = _FITTED[law](x, y)
        lse = _predict(law, prm, x[-1])
        aic = n * math.log(max(prm["rss"] / n, 1e-12)) + 2 * prm["k"]
        sst = sum((yy - (sum(y) / n)) ** 2 for yy in y)
        r2 = 1.0 - prm["rss"] / max(sst, 1e-12)
        return json.dumps(
            {"law": law, "params": prm, "AIC": round(aic, 3),
             "R2": round(r2, 3), "residual_ss": round(prm["rss"], 4),
             "tip": ("更小 AIC 更优; 若指数律(Arrhenius)占优且参数 CI 不含 0, "
                     "说明速率随温度呈指数依赖")}, ensure_ascii=False)

    if name == "analyze_uncertainty":
        p = _PROBLEMS[args["problem"]]
        law = args["law"]; n_boot = int(args.get("n_boot", 300))
        x, y, sigma = p["x"], p["y"], p["sigma"]
        rng = random.Random(args.get("seed", 7))
        params = _FITTED[law](x, y)
        key = {"linear": "b", "exponential": "b", "power": "m"}[law]
        pred = [_predict(law, params, xi) for xi in x]
        boot = []
        for _ in range(n_boot):
            # 参数自助: 按各律的正向模型加高斯噪声重构样本后重拟
            ys = [pi + rng.gauss(0.0, sigma) for pi in pred]
            fp = _FITTED[law](x, ys)
            boot.append(fp[key])
        boot.sort()
        lo, hi = boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]
        return json.dumps(
            {"law": law, "param_name": key, "point_estimate": round(params[key], 4),
             "bootstrap_95ci": [round(lo, 4), round(hi, 4)],
             "includes_zero": (lo <= 0 <= hi),
             "verdict": ("参数可信(CI 不含 0)" if not (lo <= 0 <= hi)
                         else "参数未识别(CI 含 0)——需更多/更优数据")}, ensure_ascii=False)

    if name == "design_experiment":
        p = _PROBLEMS[args["problem"]]
        law = args["law"]; pool = args.get("pool_x") or p["pool_x"]
        # 用费雪信息矩阵近似: 在候选 x 上评估其对 key 参数方差(FIM 对角)的贡献
        # 取使 α(x)²/σ² 最大(即最敏感)的前几个点 → 最能压缩参数 CI.
        params = _FITTED[law](p["x"], p["y"])
        sigma = p["sigma"]
        def sens(xx):
            if law == "linear":
                return [xx, 1.0]          # d/db=x, d/da=1
            if law == "exponential":       # d/db = A x e^(bx)
                return [params["A"] * xx * math.exp(params["b"] * xx), math.exp(params["b"] * xx)]
            return [params["c"] * (xx ** params["m"]) * math.log(max(xx, 1e-12)), xx ** params["m"]]
        scored = []
        for xx in pool:
            g = sens(xx)
            fi = sum(v * v / (sigma * sigma) for v in g)  # FIM 增量(对角近似)
            scored.append((fi, xx))
        scored.sort(reverse=True)
        top = [round(xx, 2) for _, xx in scored[:3]]
        return json.dumps(
            {"law": law, "recommended_next_sample_x": top,
             "rationale": "FIM 对角近似: 优先在模型对关键参数最敏感、而当前数据最稀疏的 x 处补样, "
                          "可在等成本下最大压缩参数 CI."}, ensure_ascii=False)

    if name == "cross_validate":
        p = _PROBLEMS[args["problem"]]
        law = args["law"]
        x, y = p["x"], p["y"]
        n = len(x)
        errs = []
        for i in range(n):
            xo = x[:i] + x[i + 1:]; yo = y[:i] + y[i + 1:]
            prm = _FITTED[law](xo, yo)
            errs.append((y[i] - _predict(law, prm, x[i])) ** 2)
        mse = sum(errs) / n
        return json.dumps({"law": law, "loocv_mse": round(mse, 4)}, ensure_ascii=False)

    raise AssertionError(name)


def _make_fit_svg(path, problem, law, params):
    p = _PROBLEMS[problem]
    W, H, L, R, T0, B = 640, 400, 60, 610, 30, 360
    xs, ys = p["x"], p["y"]
    xmax = max(xs); xmin = min(xs)
    px = lambda x: L + (x - xmin) / (xmax - xmin) * (R - L)
    ymax = max(ys) * 1.1
    ymin = min(ys) * 0.5
    py = lambda y: B - (y - ymin) / (ymax - ymin) * (B - T0)
    pts = "\n".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#e2574c"/>'
        for x, y in zip(xs, ys))
    seg = []
    xg = [xmin + (xmax - xmin) * t / 40 for t in range(41)]
    yy = [_predict(law, params, xx) for xx in xg]
    for i in range(1, len(xg)):
        seg.append(f'<line x1="{px(xg[i-1]):.1f}" y1="{py(yy[i-1]):.1f}" '
                   f'x2="{px(xg[i]):.1f}" y2="{py(yy[i]):.1f}" stroke="#2a7de1" stroke-width="2.5"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial">'
           f'<rect width="{W}" height="{H}" fill="white"/>'
           f'<text x="{L}" y="20" font-size="14" fill="#222">{p["title"]} · {law} law fit</text>'
           f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="#333" stroke-width="1.5"/>'
           f'<line x1="{L}" y1="{B}" x2="{L}" y2="{T0}" stroke="#333" stroke-width="1.5"/>'
           + "".join(seg) + pts +
           f'<text x="{(L+R)/2}" y="{H-12}" text-anchor="middle" font-size="13">{p["xlabel"]}</text>'
           f'<text x="14" y="{(T0+B)/2}" transform="rotate(-90 14 {(T0+B)/2})" text-anchor="middle" font-size="13">{p["ylabel"]}</text>'
           f'</svg>')
    path.write_text(svg, encoding="utf-8")


def _write_report(path, final, transcript, title, trace, verdict, ungrounded):
    gate_note = (
        f"**结论证伪门禁：{verdict}**（未落地主张: {ungrounded or '无'}。每个报告数值均可"
        f"回溯到下方工具执行轨迹。）" if verdict == "pass" else
        f"**结论证伪门禁：{verdict}** ⚠ 以下数值不在工具轨迹中、未被采纳: {ungrounded}。")
    header = (
        "# 开放问题研究 — 书生 Intern-S2 × Huginn 自主科研报告\n\n"
        f"> **研究性质：开放问题（无预设答案）** · {gate_note}\n\n"
        "## 一、研究问题\n\n" + title + "\n\n## 二、工具执行轨迹（门禁证据）\n\n"
    )
    trace = "\n".join(f"- {t}" for t in trace)
    body = ("\n\n## 三、方法\n\n"
            "方法由模型自主决定：候选规律比较(AIC)、可辨识性(Bootstrap CI)、"
            "泛化(留一交叉)、最优补样(费雪信息)。任何统计量以工具轨迹落地的真值为准。\n\n"
            "## 四、结论(开放性)\n\n")
    footer = ("\n\n---\n*Huginn: 不假装一致，也不捏造自由度。模型自由假设，框架用"
              "结论证伪门禁锁住每一个数值的可复现性。*\n")
    path.write_text(header + trace + body + final.strip() + footer, encoding="utf-8")


# ── 自由探索 + 结论证伪门禁 ──────────────────────────────────────
# 方法完全自由：模型自行决定调哪些工具、用什么参数、按什么顺序走；
# 框架不设阶段墙、不覆盖模型参数。
# 只在交付结论时收紧：verify_claims 将报告里每个数值与「工具执行轨迹」比对，
# 未落地主张 → needs_grounding，回给模型要求删除或补跑工具溯源。
# 复用产品模块已落地的声明门禁唯一实现 ground_verifier。


def _pick(tc):
    """把 model 的工具调用规整成 (name, args). 对非法 JSON 参数做容错解析。"""
    import ast

    name = tc.function.name
    raw = tc.function.arguments if isinstance(tc.function.arguments, str) else tc.function.arguments
    if not isinstance(raw, str):
        return name, raw
    candidates = [raw, raw.replace("'", '"')]
    for c in candidates:
        try:
            return name, json.loads(c)
        except Exception:
            pass
        try:
            return name, ast.literal_eval(c)
        except Exception:
            pass
    return name, {}


_OPEN_GOAL = (
    "你是 Huginn 科研智能体，以 Intern-S2 身份对一个**开放科学问题**做原创研究。\n"
    "问题库里有多个稀疏含噪、规律未知的数据集。研究步骤完全由你决定：\n"
    "  - 用 list_problems 看有哪些问题，用 load_dataset 选定一个；\n"
    "  - 自行提出候选规律并拟合比较（fit_law），评估参数可辨识性与泛化"
    "（analyze_uncertainty / cross_validate），并规划最优补样（design_experiment）；\n"
    "  - 你决定用什么工具、什么参数、什么顺序，可自由反复。\n"
    "最后撰写开放式科研报告：结论、置信区间、局限性与下一步实验。\n"
    "门禁提醒：系统会把报告里每个数值与你实际调用工具返回的轨迹比对，"
    "未落地的主张会被拒绝。因此你引用的每个统计量都必须先用工具真实算出。\n"
    "最终请撰写一份**完整研究报告**，结构：研究问题 / 数据与方法 / 结果分析 / "
    "结论与局限 / 下一步实验。只输出报告正文，不要输出思考过程。"
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "out"))
    args = ap.parse_args()

    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 请设置环境变量 INTERNLM_API_KEY", file=sys.stderr); return 2

    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url or _BASE_URL)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    verify = grounding_verifier()

    messages: list[dict] = [{"role": "user", "content": _OPEN_GOAL}]
    trace: list[str] = []          # 工具执行轨迹(每条工具结果) → 门禁证据
    transcript: list[str] = []
    problem_title = "（模型自选）"

    # ── 自由探索：模型自主调工具/参数/顺序，框架只收集轨迹 ──
    for _ in range(14):
        r = client.chat.completions.create(
            model=args.model, messages=messages, tools=_TOOLS,
            tool_choice="auto", max_tokens=1200, temperature=0.2)
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            # 无工具调用: 追问必须先用真实工具获取数据(防"脑内"臆造), 而非直接进入成文.
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user", "content":
                "你还没有调用任何科研工具。必须先用 list_problems 查看问题库、"
                "load_dataset 加载真实数据(而这个 load_dataset 数据是 title 与 x/y 观测), "
                "再调用 fit_law / analyze_uncertainty / cross_validate / design_experiment "
                "获取真实统计量。门禁会把报告里每个数值与实际工具轨迹比对: 凭记忆或脑内"
                "模拟编写的数字一律拦截。请立即开始调用工具。"})
            continue  # 继续探索而不是直接成文
        for tc in calls:
            name, a = _pick(tc)
            result = exec_tool(name, a)
            print(f"[tool] {name} {tc.function.arguments}\n  -> {result}")
            trace.append(result)
            transcript.append(f"`{name}` {tc.function.arguments} → {result}")
            if name == "load_dataset":
                problem_title = _PROBLEMS.get(a.get("problem", ""), {}).get("title", problem_title)
            if name == "fit_law":
                try:
                    prm = _FITTED[a["law"]](_PROBLEMS[a["problem"]]["x"], _PROBLEMS[a["problem"]]["y"])
                    _make_fit_svg(outdir / "open_research_fit.svg", a["problem"], a["law"], prm)
                except Exception:
                    pass
            messages.append({"role": "assistant", "content": msg.content or "",
                            "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            break  # 一次只回执一条，保持展开简单

    # ── 结论证伪门禁：模型交报告 → 核对每个数值是否落在轨迹里 ──
    def _gen_final() -> str:
        _kw = {}
        try:
            if "intern-ai.org.cn" in str(client.base_url):
                _kw = {"extra_body": {"thinking_mode": False}}
        except Exception:  # noqa: BLE001
            _kw = {}
        return client.chat.completions.create(
            model=args.model, messages=messages, max_tokens=1800, temperature=0.2,
            **_kw).choices[0].message.content or ""

    final = ""
    verdict_state, ungrounded = "needs_grounding", []
    for _ in range(3):
        final = _gen_final()
        if not final.strip():
            # 空报告不视为通过(没有数值不代表正确, 代表没干活) —— 追问重写.
            verdict_state, ungrounded = "needs_grounding", ["<空报告>"]
            messages.append({"role": "assistant", "content": final})
            messages.append({"role": "user", "content":
                "你的报告是空的。请基于上面已调用的真实工具结果, 撰写一份**完整研究报告**"
                "(研究问题/数据与方法/结果分析/结论与局限/下一步实验), 所有数字必须来自工具轨迹。"})
            continue
        g = verify(final, trace)
        transcript.append(f"门禁: verify={g['verdict']} unsubstantiated={g['unsubstantiated']}")
        print(f"\n[门禁] {g['verdict']}  unsubstantiated={g['unsubstantiated']}")
        if g["verdict"] == "pass":
            verdict_state, ungrounded = "pass", []
            break
        ungrounded = g["unsubstantiated"]
        # 自由但不得绕过门禁：请删除未落地主张或补跑工具溯源
        messages.append({"role": "user", "content":
            "门禁未通过：以下数值不在你的工具执行轨迹中、无法溯源: "
            f"{ungrounded}。请删除这些未落地的主张（若要保留，先用对应工具获得真实值），"
            "然后重写一份**完整研究报告**（研究问题/数据与方法/结果分析/结论与局限/"
            "下一步实验），确保每个数值都能在工具轨迹中找到。"})
        messages.append({"role": "assistant", "content": final})

    report = outdir / "ai4s_open_research_report.md"
    _write_report(report, final, transcript, problem_title, trace, verdict_state, ungrounded)
    print("\n==== 开放研究报告已写入 ====")
    print(report.resolve())
    print(f"\n结论证伪门禁: {verdict_state}  未落地主张: {ungrounded}")
    print("\n---- 正文（前 1200 字）----\n")
    print(final[:1200])
    return 0


_TOOLS = [
    {"type": "function", "function": {"name": "list_problems", "description": "查看开放问题库(候选科研问题 id 与简述)",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "load_dataset", "description": "加载某问题的稀疏含噪观测数据",
        "parameters": {"type": "object", "properties": {"problem": {"type": "string", "enum": ["exp_law", "comp_law"]}},
            "required": ["problem"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "fit_law", "description": "用最小二乘拟合并返回 AIC/R²(residual_ss)",
        "parameters": {"type": "object", "properties": {"problem": {"type": "string"},
            "law": {"type": "string", "enum": ["linear", "exponential", "power"]}},
            "required": ["problem", "law"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "analyze_uncertainty", "description": "bootstrap 估计关键参数 95% CI, 判定可识别性",
        "parameters": {"type": "object", "properties": {"problem": {"type": "string"},
            "law": {"type": "string", "enum": ["linear", "exponential", "power"]},
            "n_boot": {"type": "integer"}, "seed": {"type": "integer"}},
            "required": ["problem", "law"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "design_experiment", "description": "FIM 挑选下一步最该补样的 x 点",
        "parameters": {"type": "object", "properties": {"problem": {"type": "string"},
            "law": {"type": "string", "enum": ["linear", "exponential", "power"]},
            "pool_x": {"type": "array", "items": {"type": "number"}}},
            "required": ["problem", "law"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "cross_validate", "description": "某规律的留一交叉验证误差",
        "parameters": {"type": "object", "properties": {"problem": {"type": "string"},
            "law": {"type": "string", "enum": ["linear", "exponential", "power"]}},
            "required": ["problem", "law"], "additionalProperties": False}}},
]


if __name__ == "__main__":
    raise SystemExit(main())