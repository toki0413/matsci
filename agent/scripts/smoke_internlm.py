#!/usr/bin/env python3
"""书生 InternLM 接入冒烟 — 验证『Intern-S2-Preview → 工具调用 → 执行 → 归报』AI4S 闭环.

背景: Huginn 的 model registry 已把书生 ChatAPI 注册为原生 ``internlm`` provider
(见 huginn/models/registry.py 的 ``_DOMESTIC_OPENAI_COMPATIBLE`` 与能力表). 本脚本不依赖
Huginn 运行时, 仅用 openai SDK 直连书生端点, 复现一条最小 AI4S 认知链:

    「搜」(search_literature) → 「算」(run_symbolic_regression) → 「写」(结论报告)

用的工具语义分别复用仓库里两个既有案例的思路: demo_evidence_chain (文献按自由度归因)
与符号回归 (symbolic_regression_tool). 目标固定为 Li2O 带隙 — 深度思考模型应能自主
编排这两步工具调用并给出归因结论.

用法:
    export INTERNLM_API_KEY=<你的书生 token>          # 必填
    python scripts/smoke_internlm.py                  # 模型默认 intern-s2-preview
    python scripts/smoke_internlm.py --model intern-s1-pro

依赖: pip install openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_BASE_URL = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
_DEFAULT_MODEL = "intern-s2-preview"

# ── 暴露给模型的科学工具集 (与 Huginn 能力一一对应) ─────────────
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_literature",
            "description": (
                "检索某体系某性质在文献中的报道值, 返回按方法自由度(泛函/温度/实验)标记的 "
                "结构化记录, 每条带 doi 与原文摘要."
            ),
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
            "description": (
                "对 (x, y) 序列做符号回归, 返回拟合系数与公式度数. "
                "用于从数据中挖掘解析规律(如带隙随组分的经验式)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "xs": {"type": "array", "items": {"type": "number"}},
                    "ys": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["xs", "ys"],
                "additionalProperties": False,
            },
        },
    },
]

_USER_GOAL = (
    "研究 Li2O 带隙: (1) 先用 search_literature 检索 PBE/实验/HSE 的报道值; "
    "(2) 再用 run_symbolic_regression 拟合带隙随参考压力 P(单位 GPa) 的经验式, "
    "xs 给定压力 [0,5,10,20], ys 给定带隙 [5.80, 5.92, 6.04, 6.31]; "
    "(3) 最后用一句话给出带隙数值与\"不同报告中值为何不一致\"的归因. 依次调用工具, 别跳过."
)


# ── 两个模拟工具的执行端 (纯 Python, 复用既有案例的物理语义) ─────
def _exec_tool(tc) -> str:
    fn = tc.function.name
    args = (
        json.loads(tc.function.arguments)
        if isinstance(tc.function.arguments, str)
        else tc.function.arguments
    )

    if fn == "search_literature":
        # 语义复刻 demo_evidence_chain 的 Li2O 基带隙, 按方法自由度归因.
        rows = [
            {"value": 4.10, "unit": "eV", "method": "DFT-PBE", "note": "0 K, GGA 低估带隙", "doi": "10.1000/pbe"},
            {"value": 5.80, "unit": "eV", "method": "experiment", "note": "室温光学带隙", "doi": "10.1000/exp"},
            {"value": 6.00, "unit": "eV", "method": "HSE06", "note": "杂化泛函, ~实验结果", "doi": "10.1000/hse"},
        ]
        return json.dumps({"material": args["material"], "property": args["property"],
                           "reported": rows}, ensure_ascii=False)
    if fn == "run_symbolic_regression":
        xs, ys = args["xs"], args["ys"]
        # 简单线性拟合 E_g(P) = a + b*P (P 单位 GPa).
        n = len(xs)
        sx = sum(xs); sy = sum(ys); sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx or 1.0
        b = (n * sxy - sx * sy) / denom
        a = (sy - b * sx) / n
        return json.dumps({"formula": f"E_g(P) = {a:.4f} + {b:.4f} * P (GPa)",
                           "degree": 1, "coefficients": [a, b]}, ensure_ascii=False)
    raise AssertionError(f"unknown tool: {fn}")


def _run_loop(client_args: dict, model: str, verbose: bool = True) -> int:
    from openai import OpenAI

    client = OpenAI(**client_args)
    messages: list[dict] = [{"role": "user", "content": _USER_GOAL}]

    for turn in range(6):
        r = client.chat.completions.create(
            model=model, messages=messages, tools=_TOOLS,
            tool_choice="auto", max_tokens=1024, temperature=0.2,
        )
        msg = r.choices[0].message
        calls = msg.tool_calls or []

        if not calls:
            print("\n[final] 模型结论:")
            print(msg.content)
            print("\n[ok] InternLM 工具调用 + 归报闭环通过.")
            return 0

        for tc in calls:
            name = tc.function.name
            if verbose:
                print(f"[tool_calls] {name} args={tc.function.arguments}")
            result = _exec_tool(tc)
            if verbose:
                print(f"  -> result: {result}")
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in calls],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
            break  # 一轮只回执一条, 保持 dp 简单; 模型会继续下一轮.
    print("[fail] 模型未在限定轮数内收敛.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_DEFAULT_MODEL, help="书生模型名")
    parser.add_argument("--base-url", default=None, help="覆盖端点 (默认环境 INTERNLM_BASE_URL)")
    args = parser.parse_args()

    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 请设置环境变量 INTERNLM_API_KEY=你的书生 token", file=sys.stderr)
        return 2

    code = _run_loop(
        {"api_key": key, "base_url": args.base_url or _BASE_URL}, args.model)
    sys.exit(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())