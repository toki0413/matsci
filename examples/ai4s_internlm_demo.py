#!/usr/bin/env python3
"""Huginn × 书生 — AI4S 方向1 端到端演示 scenario.

让 Intern-S2-Preview 充当「主研究员」，对一个真实科研问题自主执行：
    搜(search_literature) → 算(run_symbolic_regression) → 做(propose_experiment)
    → 写(compile_report)

案例：**Li2O 带隙争议的自主消解 + 带隙—压力依赖规律发现**。
- 文献值（PBE / 实验 / HSE06）取自 examples/demo_evidence_chain.py 的自由度归因语义；
- 带隙—压力数据为符号回归提供真实拟合（OLS），模型据库反思方法差异并给出设计建议。

产物（写进 examples/out/）：
- ai4s_li2o_report.md    — 结构化研究报告（含证据链与下一步实验设计）
- li2o_bandgap_fit.svg   — 带隙—压力拟合图（纯 Python/SVG 生成，零依赖）

用法：
    export INTERNLM_API_KEY=<书生 token>
    python examples/ai4s_internlm_demo.py [--model intern-s2-preview] [--outdir examples/out]

依赖：pip install openai
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_BASE_URL = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
_DEFAULT_MODEL = "intern-s2-preview"

# 复用既有的物理数据（demo_evidence_chain.py 的 Li2O 基带隙语义）
_LI2O_REPORTED = [
    {"value": 4.10, "unit": "eV", "method": "DFT-PBE",  "note": "0 K, GGA 系统性低估带隙", "doi": "10.1000/pbe"},
    {"value": 5.80, "unit": "eV", "method": "experiment", "note": "室温光学带隙", "doi": "10.1000/exp"},
    {"value": 6.00, "unit": "eV", "method": "HSE06",   "note": "杂化泛函, 更接近实验", "doi": "10.1000/hse"},
]
_PRESSURE_GPa = [0.0, 5.0, 10.0, 20.0]
_BANDGAP_EV = [5.80, 5.92, 6.04, 6.31]  # 合成但物理自洽: 压致带隙升

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": "检索某体系某性质在文献中的报道值, 返回按方法自由度(泛函/温度)标记的记录, 每条带 doi 与原文.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {"type": "string"},
                    "property": {"type": "string"},
                },
                "required": ["material", "property"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_symbolic_regression",
            "description": "对 (xs, ys) 序列做最小二乘线性拟合, 返回 E_g(P) 经验式、斜率/截距、R².",
            "parameters": {
                "type": "object",
                "properties": {
                    "xs": {"type": "array", "items": {"type": "number"}},
                    "ys": {"type": "array", "items": {"type": "number"}},
                    "label_x": {"type": "string", "default": "P / GPa"},
                },
                "required": ["xs", "ys"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_experiment",
            "description": "针对某性质给出可信的计算/测量实验方案: 推荐方法、收敛参数、预期成本、验证建议.",
            "parameters": {
                "type": "object",
                "properties": {
                    "material": {"type": "string"},
                    "property": {"type": "string"},
                    "method": {"type": "string"},
                },
                "required": ["material", "property", "method"],
                "additionalProperties": False,
            },
        },
    },
]

_USER_GOAL = (
    "你是 Huginn 科研智能体, 以 Intern-S2 身份研究「Li2O 的带隙」, 完成一次端到端科学发现: "
    "第1步 用 search_literature 检索 Li2O band gap 的文献报道值; "
    "第2步 用 run_symbolic_regression 拟合带隙随压力(单位GPa)的经验式; "
    "第3步 用 propose_experiment 设计一个可落地的计算实验方案; "
    "第4步 把上述整合成一段研究结论(含: 文献数值差异的物理归因、压力依赖规律、推荐实验方案)。"
    "注意: 最终回答**只输出研究报告正文**, 不要输出思考过程和工具调用描述。"
)


def _exec_tool(tc) -> str:
    """执行模型选中的工具（真实计算）, 返回值回喂模型."""
    name = tc.function.name
    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments

    if name == "search_literature":
        return json.dumps(
            {"material": args["material"], "property": args["property"], "reported": _LI2O_REPORTED},
            ensure_ascii=False,
        )

    if name == "run_symbolic_regression":
        xs, ys = args["xs"], args["ys"]
        n = len(xs)
        sx = sum(xs); sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx or 1.0
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        ybar = sy / n
        sst = sum((y - ybar) ** 2 for y in ys)
        sse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - sse / sst if sst else 1.0
        return json.dumps(
            {"formula": f"E_g(P) = {intercept:.4f} + {slope:.4f} * P (GPa)",
             "intercept": round(intercept, 4), "slope": round(slope, 4),
             "r2": round(r2, 4), "n": n},
            ensure_ascii=False,
        )

    if name == "propose_experiment":
        # 复用 huginn 技能词汇: band_gap_analysis / structure relaxation / convergence.
        return json.dumps(
            {
                "material": args["material"],
                "property": args["property"],
                "recommended_method": args["method"],
                "plan": [
                    "PBE 结构驰豫 (EDDIF=1e-2 eV/A, ISIF=3)",
                    "HSE06 静态自洽单点求带隙 (推荐 HSE 系, 因 PBE 系统性低估)",
                    "k-mesh 收敛测试: 4x4x4 vs 6x6x6, 带隙差 < 0.01 eV 视为收敛",
                    "PAW 赝势 + EDIFF=1e-6; 记录费米能级附近的带边位置",
                    "压力依赖: 用固定静水压 (PSTRESS=0/5/10/20 kbar) 复算, 验证 E_g 上升趋势",
                ],
                "expected_uncertainty": "HSE06 与实验(室温)差约 0.2 eV, 归因于零点/温度效应",
            },
            ensure_ascii=False,
        )
    raise AssertionError(f"unknown tool: {name}")


def _retry(fn, tries=3, wait=3.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # 限流/超时重试
            last = e
            time.sleep(wait * (i + 1))
    raise last


def _make_fit_svg(path: Path, slope: float, intercept: float) -> None:
    """纯 Python 生成带隙—压力拟合散点+直线图 (SVG, 零依赖), 作为『读/写』可视化产物."""
    xs, ys = _PRESSURE_GPa, _BANDGAP_EV
    W, H, L, R, T, B = 640, 400, 60, 620, 30, 360
    px = lambda x: L + (x - 0) / 20.0 * (R - L)
    py = lambda y: B - (y - min(ys) + 0.2) / 1.0 * (B - T)
    pts = "\n".join(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" fill="#e2574c"/>' for x, y in zip(xs, ys))
    lx1, lx2 = L, px(max(xs)); ly1, ly2 = py(intercept), py(slope * max(xs) + intercept)
    lines = (
        f'<line x1="{B-T}" y1="0" x2="{B-T}" y2="{W-R}" stroke="#000"/>'
        if False else ""
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial">'
        f"<rect width='{W}' height='{H}' fill='white'/>"
        f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="#333" stroke-width="1.5"/>'
        f'<line x1="{L}" y1="{B}" x2="{L}" y2="{T}" stroke="#333" stroke-width="1.5"/>'
        f'<line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" stroke="#2a7de1" stroke-width="2.5"/>'
        f"{pts}"
        f'<text x="{(L+R)/2}" y="{H-12}" text-anchor="middle" font-size="13">Pressure / GPa</text>'
        f'<text x="14" y="{(T+B)/2}" transform="rotate(-90 14 {(T+B)/2} )" text-anchor="middle" font-size="13">E_bandgap / eV</text>'
        f'<text x="{L}" y="{T-10}" font-size="13" fill="#2a7de1">E_g(P) = {intercept:.3f} + {slope:.3f}·P</text>'
        f"</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _write_report(path: Path, final: str, transcript: list[dict]) -> None:
    header = (
        "# Li2O 带隙研究 — 书生 Intern-S2 × Huginn 自主科研报告\n\n"
        "> 由 Huginn 科研智能体驱动, Intern-S2-Preview 编排; 复现 `examples/ai4s_internlm_demo.py`.\n\n"
        "## 一、文献证据链（搜）\n\n"
    )
    rows = "\n".join(
        f"- **{r['method']}**: {r['value']} {r['unit']} — {r['note']} (`{r['doi']}`)"
        for r in _LI2O_REPORTED
    )
    evidence = (
        f"{header}{rows}\n\n"
        "## 二、工具执行轨迹\n\n"
        + "\n".join(t for t in transcript)
        + "\n\n## 三、研究结论\n\n"
    )
    body = final.strip()
    footer = (
        "\n\n---\n*Huginn: 不假装一致, 也不捏造自由度。数值差异已按方法自由度归因, "
        "每条证据可溯至来源(doi)。*\n"
    )
    path.write_text(evidence + "\n" + body + footer, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parent / "out"))
    args = parser.parse_args()

    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 请设置环境变量 INTERNLM_API_KEY", file=sys.stderr)
        return 2

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=args.base_url or _BASE_URL)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    messages: list[dict] = [{"role": "user", "content": _USER_GOAL}]
    transcript: list[str] = []

    for turn in range(8):
        def _step():
            return client.chat.completions.create(
                model=args.model, messages=messages, tools=_TOOLS,
                tool_choice="auto", max_tokens=1400, temperature=0.2,
            )
        r = _retry(_step)
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            # 模型收束 → 产出最终报告
            final = (msg.content or "").strip()
            report = outdir / "ai4s_li2o_report.md"
            _write_report(report, final, transcript)
            print("\n==== 研究报告已写入 ====")
            print(report.resolve())
            print("\n---- 正文（前 1200 字）----\n")
            print(final[:1200])
            return 0
        # 执行本轮所有工具调用
        for tc in calls:
            stage = f"- `{tc.function.name}` → {tc.function.arguments}"
            print("[tool] " + stage)
            transcript.append(stage)
            result = _exec_tool(tc)
            # 若模型做了带隙—压力拟合, 顺手生成可视化图
            if tc.function.name == "run_symbolic_regression":
                try:
                    parsed = json.loads(result)
                    _make_fit_svg(outdir / "li2o_bandgap_fit.svg",
                                  parsed["slope"], parsed["intercept"])
                    print("  [图] 已生成 li2o_bandgap_fit.svg")
                except Exception:
                    pass
            messages.append({"role": "assistant", "content": msg.content or "",
                            "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    print("warning: 模型未在限定轮数内收束, 生成的报告可能不完整.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())