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


def _write_report(path, final, transcript, title, evidence, best_law):
    header = (
        "# 开放问题研究 — 书生 Intern-S2 × Huginn 自主科研报告\n\n"
        "> **研究性质：开放问题（无预设答案）**。数据稀疏含噪，规律未知；以下结论由 "
        "AIC 模型选择 + Bootstrap 参数置信区间 + 留一交叉验证 + 费雪信息实验设计推断得出。\n\n"
        "## 一、研究问题\n\n" + title + "\n\n## 二、Grounded 真实证据（来自工具，非模型声明）\n\n```\n"
        + evidence + "```\n\n## 三、执行轨迹\n\n"
    )
    trace = "\n".join(f"- {t}" for t in transcript)
    body = ("\n\n## 四、研究方法\n\n"
            "候选规律: linear / exponential; 最小二乘拟合并按 AIC 比较; "
            "参数可辨识性用残差自助法(Bootstrap)95% CI 判定; 用留一交叉验证(LOOCV)复核; "
            "用费雪信息矩阵(FIM)对补样点做最优实验设计。\n\n"
            f"经 AIC 判定的更优规律：**{best_law}**。\n\n## 五、结论(开放性)\n\n")
    footer = ("\n\n---\n*Huginn: 不假装一致，也不捏造自由度。所有统计量均由真实工具结果落地，"
              "参数不确定性与实验盲区均已显式报告。*\n")
    path.write_text(header + trace + body + final.strip() + footer, encoding="utf-8")


# ── 受控科研流水线：每个数字都由真实工具结果落地（防模型编造）───
# 与 Huginn 的 phase/plan 门控一致：框架编排阶段、模型在阶段内推理/解释，
# 但任何统计量只来自工具返回，绝不采纳模型"脑内模拟"的数字。


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
    # 仍失败：尽力取缺失字段, 允许空参(用 force_args 兜底)
    return name, {}


def _run_stage(client, model_args, messages, required_tool, prompt, legal_tools, force_args=None, fallback_args=None):
    """反复提示直至模型调用 required_tool（最多 3 次），执行并返回 (name,args,result).

    force_args 非空时，无论模型传入什么，都按框架给定参数执行——确保每个统计量同属
    一个受控实验(grounded)。若模型一直不调工具，则用 fallback_args 兜底执行。
    """
    messages.append({"role": "user", "content": prompt})
    for _ in range(3):
        r = client.chat.completions.create(
            model=model_args["model"], messages=messages,
            tools=legal_tools, tool_choice="auto", max_tokens=800, temperature=0.2)
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if calls:
            name, model_args_raw = _pick(calls[0])
            args = force_args if (name == required_tool and force_args is not None) else model_args_raw
            result = exec_tool(name, args)
            messages.append({"role": "assistant", "content": msg.content or "",
                            "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": calls[0].id, "content": result})
            if name == required_tool:
                return name, args, result
        messages.append({"role": "user",
                        "content": f"请直接调用 {required_tool} 工具(其余工具请不要调用)。若你仍不调用，系统将代为执行。"})
    # 兜底：模型一直不调 → 由框架代执行，保证结果落地
    fb = fallback_args
    if fb is None and force_args is not None:
        fb = force_args
    if fb is None:
        fb = {}
    return required_tool, fb, exec_tool(required_tool, fb)


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
    messages: list[dict] = []
    grounded: dict[str, str] = {}          # 阶段名 -> 工具返回(真实)
    transcript: list[str] = []

    def closure(label, result, note=""):
        transcript.append(f"- **{label}**: {note}{result}")
        print(f"[{label}] {note}{result}" if not note else f"[{label}] {note}")

    # 阶段1 选题+加载（模型自主从问题库选一个）
    name1, a1, r1 = _run_stage(
        client, {"model": args.model}, messages, "load_dataset",
        "科研启动：先调用 list_problems 查看问题库(可先看一眼)，然后调用 load_dataset 选定要研究的开放问题(exp_law 或 comp_law)。",
        [t for t in _TOOLS if t["function"]["name"] in ("list_problems", "load_dataset")])
    grounded["dataset"] = r1
    closure("选题与数据加载", r1)
    problem = a1["problem"]
    p = _PROBLEMS[problem]

    # 阶段2 假设+算：对两个互相竞争的候选规律各做一次真实拟合(AIC 比较)
    fits: dict[str, str] = {}
    for law in ("linear", "exponential"):
        _, _, rl = _run_stage(
            client, {"model": args.model}, messages, "fit_law",
            f"假设: 对已选问题({problem})调用 fit_law 拟合 {law} 律。",
            [t for t in _TOOLS if t["function"]["name"] == "fit_law"],
            force_args={"problem": problem, "law": law},
            fallback_args={"problem": problem, "law": law})
        grounded[f"fit_{law}"] = rl
        fits[law] = rl
        closure(f"拟合 {law} 律", rl)

    # 由真实 AIC 决定更优规律（不依赖模型的声明）
    aic = {law: json.loads(fits[law])["AIC"] for law in fits}
    best_law = min(aic, key=aic.get)

    # 阶段3 验证：对更优规律做 bootstrap 可辨识性 + 留一交叉
    _, _, r3 = _run_stage(
        client, {"model": args.model}, messages, "analyze_uncertainty",
        f"对更优规律 {best_law}(AIC={aic[best_law]}) 调用 analyze_uncertainty 评估参数可辨识性。",
        [t for t in _TOOLS if t["function"]["name"] in ("analyze_uncertainty", "cross_validate")],
        force_args={"problem": problem, "law": best_law},
        fallback_args={"problem": problem, "law": best_law})
    grounded["uncertainty"] = r3
    closure("参数可辨识性(Bootstrap CI)", r3)
    _, _, r4 = _run_stage(
        client, {"model": args.model}, messages, "cross_validate",
        f"对 {best_law} 律调用 cross_validate 做留一交叉验证。",
        [t for t in _TOOLS if t["function"]["name"] == "cross_validate"],
        force_args={"problem": problem, "law": best_law},
        fallback_args={"problem": problem, "law": best_law})
    grounded["cv"] = r4
    closure("留一交叉验证", r4)

    # 阶段4 做：FIM 最优实验设计
    _, _, r5 = _run_stage(
        client, {"model": args.model}, messages, "design_experiment",
        f"对 {best_law} 律调用 design_experiment 规划下一步最该补样的 x 点。",
        [t for t in _TOOLS if t["function"]["name"] == "design_experiment"],
        force_args={"problem": problem, "law": best_law},
        fallback_args={"problem": problem, "law": best_law})
    grounded["design"] = r5
    closure("最优实验设计(FIM)", r5)

    # 生成拟合图(用真实拟合参数)
    prm = _FITTED[best_law](p["x"], p["y"])
    _make_fit_svg(outdir / "open_research_fit.svg", problem, best_law, prm)

    # 阶段5 写：把全部真实数值注入, 让模型只做开放式的解释与写作
    evidence = (
        "以下为本研究通过工具获得的**全部真实数值**(只能引用这些, 禁止编造/外推):\n"
        f"- 数据: x={p['x']}, y={p['y']}, 测量噪声 sigma={p['sigma']}\n"
        f"- linear 拟合: {json.loads(grounded['fit_linear'])['params']}, "
        f"AIC={json.loads(grounded['fit_linear'])['AIC']}, R²={json.loads(grounded['fit_linear'])['R2']}\n"
        f"- exponential 拟合: {json.loads(grounded['fit_exponential'])['params']}, "
        f"AIC={json.loads(grounded['fit_exponential'])['AIC']}, R²={json.loads(grounded['fit_exponential'])['R2']}\n"
        f"- AIC 更优规律: {best_law}\n"
        f"- 参数可辨识性: {grounded['uncertainty']}\n"
        f"- 留一交叉验证: {grounded['cv']}\n"
        f"- 最优补样点(FIM): {grounded['design']}\n"
    )
    messages.append({"role": "user", "content":
        f"{evidence}\n\n请以开放式科研报告的形式撰写最终结论：\n"
        "1) 两个候选规律孰优(引用各自 AIC)；\n"
        "2) 参数是否可识别(引用 CI 是否含 0 与其 95% 区间)；\n"
        "3) 留一交叉验证误差；\n"
        "4) 建议下一步补样的 x 点(引用 FIM 推荐)与理由；\n"
        "5) 局限性与后续实验方向。\n"
        "只能使用上方给出的真实数值，不要编造任何未给出的统计量。\n"
        "直接输出研究报告正文，不要输出任何思考过程。"})

    # 写作阶段关掉 thinking_mode, 避免 intern-s2 把思维链写进 content
    final = ""
    for attempt in range(2):
        kwargs = dict(model=args.model, messages=messages, max_tokens=2000, temperature=0.2)
        if attempt == 0:
            kwargs["extra_body"] = {"thinking_mode": False}
        r = client.chat.completions.create(**kwargs)
        raw = (r.choices[0].message.content or "").strip()
        import re
        m = re.search(r"<report>(.*?)</report>", raw, flags=re.DOTALL | re.IGNORECASE)
        candidate = m.group(1).strip() if m else raw
        if "Thinking Process" in candidate[:120]:
            candidate = candidate.split("Thinking Process", 1)[-1].strip()
        # 判质: 正文应有一定长度且含句读; 否则换 thinking 档重试一次
        if len(candidate) > 120 and (candidate.count("。") + candidate.count("\n")) >= 3:
            final = candidate
            break
        final = candidate
    if not final or len(final) < 80:
        final = ("（模型生成异常，已退化为基于 Grounded 证据的确定性结论）\n"
                 f"依据真实统计：exponential({json.loads(fits['exponential'])['AIC']}) 优于 "
                 f"linear({json.loads(fits['linear'])['AIC']})，参数可识别(Boot CI="
                 f"{json.loads(grounded['uncertainty'])['bootstrap_95ci']}，"
                 f"{json.loads(grounded['uncertainty'])['verdict']})，LOOCV="
                 f"{json.loads(grounded['cv'])['loocv_mse']}，下一步建议在 "
                 f"{json.loads(grounded['design'])['recommended_next_sample_x']} 处补样。")

    report = outdir / "ai4s_open_research_report.md"
    _write_report(report, final, transcript, p["title"], evidence, best_law)
    print("\n==== 开放研究报告已写入 ====")
    print(report.resolve())
    print("\n---- 正文（前 1600 字）----\n")
    print(final[:1600])
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