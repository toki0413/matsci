#!/usr/bin/env python3
"""Huginn × 书生 — 计算数学「新型有限元/等几何分析」开放研究范例.

真实数值方法科研：求解 -u''=π²sin(πx), u(0)=u(1)=0，用
  - fem_linear ：线性拉格朗日有限元（理论 H1≈O(h), L2≈O(h²)）
  - iga_p2     ：二次 B 样条等几何分析（理论 H1≈O(h²), L2≈O(h³)，C¹）
做收敛率研究（refinement 后 log-log 估阶），对更细网格的误差做【可证伪预测】，
再用真实求解回算对账。数值核心 examples/numerics.py 为纯 Python 真实装配/求解。

「结论证伪门禁」沿用 huginn/validation/claim_grounding：报告每个数值必须在工具轨迹中。

用法: export INTERNLM_API_KEY=<token>; python examples/ai4s_numerics_demo.py
依赖: pip install requests openai
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

_BASE_URL = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
_DEFAULT_MODEL = "intern-s2-preview"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

_HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(_HERE))
import numerics as nm  # noqa: E402

METHODS = {"fem_linear": "线性拉格朗日有限元", "iga_p2": "二次 B 样条等几何分析(IGA)"}
THEORY = {"fem_linear": {"H1": 1, "L2": 2}, "iga_p2": {"H1": 2, "L2": 3}}
MESHES = [8, 16, 32, 64]


def _load_gate():
    try:
        from huginn.validation.claim_grounding import verify_claims
        return verify_claims
    except Exception:
        import importlib.util
        src = Path(__file__).resolve().parents[1] / "agent/huginn/validation/claim_grounding.py"
        spec = importlib.util.spec_from_file_location("_cg", str(src))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.verify_claims


_TOOLS = [
    {"type": "function", "function": {"name": "load_pde", "description": "加载椭圆型 PDE、可用方法与理论收敛阶。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "convergence", "description": "对某方法在不同网格做收敛研究, 返回各网格 H1/L2 误差与 log-log 估阶。",
        "parameters": {"type": "object", "properties": {"method": {"type": "string", "enum": list(METHODS)}},
            "required": ["method"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "compare_methods", "description": "对比 FEM-linear 与 IGA-p2 在同自由度下的误差(谁更高效)。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "predict_error", "description": "用给定网格的误差拟合收敛阶, 预测更细网格(target_ne)的 H1 误差(可证伪)。",
        "parameters": {"type": "object", "properties": {"method": {"type": "string", "enum": list(METHODS)},
            "target_ne": {"type": "integer", "default": 128}}, "required": ["method"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "verify_predict", "description": "回算: 真正求解 target_ne, 返回预测vs实际误差与差距(回算对账)。",
        "parameters": {"type": "object", "properties": {"method": {"type": "string", "enum": list(METHODS)},
            "target_ne": {"type": "integer", "default": 128}}, "required": ["method"], "additionalProperties": False}}},
]

_GOAL = (
    "你是 Huginn 科研智能体, 以 Intern-S2 身份对【计算数学：新型有限元 vs 等几何分析】做真实数值研究。\n"
    "研究问题: 对 -u''=π²sin(πx), u(0)=u(1)=0, '线性有限元(FEM)' 与 '二次 B 样条等几何(IGA)' 谁收敛更快？"
    "数值估的收敛阶是否与理论一致？同自由度下谁更高效？\n"
    "建议: 用 load_pde 看方法与理论阶 → 用 convergence 分别对 fem_linear 与 iga_p2 做收敛研究(估H1/L2阶) "
    "→ 用 compare_methods 在同自由度下对比 → 用 predict_error 对更细网格预测 H1 误差 → 用 verify_predict 回算对账。\n"
    "门禁提醒: 报告中每个数值必须落在你实际调用工具返回的真实结果中; 未落地主张会被拒绝。请务必真正调用工具。\n"
    "最终输出完整研究报告(研究问题/数据与方法/结果分析/预测-对账/结论与局限/下一步), 只输出正文。"
)


def _pick(tc):
    import ast
    name = tc.function.name
    raw = tc.function.arguments if isinstance(tc.function.arguments, str) else tc.function.arguments
    if not isinstance(raw, str):
        return name, raw
    for c in (raw, raw.replace("'", '"')):
        for fn in (json.loads, ast.literal_eval):
            try:
                return name, fn(c)
            except Exception:
                continue
    return name, {}


def _run_conv(method):
    rows = []
    for ne in MESHES:
        s = nm.fem_linear(ne, nm.force_sin, nm.exact_sin, nm.exact_sin_der) if method == "fem_linear" \
            else nm.iga_p2(ne, nm.force_sin, nm.exact_sin, nm.exact_sin_der)
        e = nm.errors(s, nm.exact_sin, nm.exact_sin_der)
        rows.append({"ne": ne, "dof": e["dof"], "H1": round(e["H1"], 6), "L2": round(e["L2"], 6)})
    oh1 = nm.estimate_order([(r["ne"], r["H1"]) for r in rows])
    ol2 = nm.estimate_order([(r["ne"], r["L2"]) for r in rows])
    return {"method": method, "name": METHODS[method], "theory": THEORY[method],
            "estimated_H1_order": round(oh1, 2), "estimated_L2_order": round(ol2, 2), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()
    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: INTERNLM_API_KEY not set", file=sys.stderr); return 2
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url or _BASE_URL)
    verify = _load_gate()
    state = {}  # 记录 predict 供 verify 用

    def exec_tool(name, a):
        if name == "load_pde":
            return json.dumps({"pde": "-u'' = π² sin(πx), u(0)=u(1)=0, 精确解 u=sin(πx)",
                               "methods": METHODS, "theory_orders": THEORY,
                               "meshes_tested": MESHES}, ensure_ascii=False)
        if name == "convergence":
            return json.dumps(_run_conv(a["method"]), ensure_ascii=False)
        if name == "compare_methods":
            c_l = _run_conv("fem_linear"); c_i = _run_conv("iga_p2")
            # 同自由度对比:找 dof 相近的两档
            return json.dumps({
                "fem_linear": {"name": METHODS["fem_linear"], "estimated_H1": c_l["estimated_H1_order"],
                               "H1_at_dof": [(r["dof"], r["H1"]) for r in c_l["rows"]]},
                "iga_p2": {"name": METHODS["iga_p2"], "estimated_H1": c_i["estimated_H1_order"],
                           "H1_at_dof": [(r["dof"], r["H1"]) for r in c_i["rows"]]},
                "read": "同样多自由度下 H1 误差越小越高效; 注意估计收敛阶是否与理论 O(h) / O(h²) 一致。"},
                ensure_ascii=False)
        if name == "predict_error":
            m = a["method"]; tgt = int(a.get("target_ne", 128))
            errors_h = []
            for ne in MESHES:
                s = nm.fem_linear(ne, nm.force_sin, nm.exact_sin, nm.exact_sin_der) if m == "fem_linear" \
                    else nm.iga_p2(ne, nm.force_sin, nm.exact_sin, nm.exact_sin_der)
                errors_h.append((ne, nm.errors(s, nm.exact_sin, nm.exact_sin_der)["H1"]))
            p = nm.estimate_order(errors_h)
            last_ne, last_err = errors_h[-1]
            pred = last_err * (last_ne / tgt) ** p
            state["pred"] = {"method": m, "target": tgt, "predicted": pred}
            return json.dumps({"method": m, "target_ne": tgt, "estimated_order": round(p, 2),
                               "predicted_H1_error": round(pred, 6)}, ensure_ascii=False)
        if name == "verify_predict":
            m = a.get("method", state.get("pred", {}).get("method", "fem_linear"))
            tgt = int(a.get("target_ne", state.get("pred", {}).get("target", 128)))
            s = nm.fem_linear(tgt, nm.force_sin, nm.exact_sin, nm.exact_sin_der) if m == "fem_linear" \
                else nm.iga_p2(tgt, nm.force_sin, nm.exact_sin, nm.exact_sin_der)
            actual = nm.errors(s, nm.exact_sin, nm.exact_sin_der)["H1"]
            pred = state.get("pred", {}).get("predicted")
            ratio = pred / actual if pred else None
            return json.dumps({"method": m, "target_ne": tgt, "predicted_h1": round(pred, 6) if pred else None,
                               "actual_h1": round(actual, 6), "pred_to_actual": round(ratio, 3) if ratio else None,
                               "verdict": ("预测已由真实求解对账" if ratio and 0.7 <= ratio <= 1.4 else
                                           "预测与真实求解偏差明显")}, ensure_ascii=False)
        raise AssertionError(name)

    messages = [{"role": "user", "content": _GOAL}]
    trace = []; transcript = []
    for _ in range(14):
        r = client.chat.completions.create(model=args.model, messages=messages, tools=_TOOLS,
                                           tool_choice="auto", max_tokens=1100, temperature=0.2)
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if not calls:
            break
        for tc in calls[:1]:
            name, a = _pick(tc)
            result = exec_tool(name, a)
            print(f"[tool] {name} {tc.function.arguments}\n  -> {result}")
            trace.append(result); transcript.append(f"`{name}` {tc.function.arguments} → {result}")
            messages.append({"role": "assistant", "content": msg.content or "",
                            "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # 引导完成数值研究必需的真实工具步奏（防模型跳过工具而捏造数值）
    required = ["convergence", "compare_methods", "predict_error", "verify_predict"]
    if not all(any(t.startswith(f"`{k}") for t in transcript) for k in required):
        missing = [k for k in required if not any(t.startswith(f"`{k}") for t in transcript)]
        messages.append({"role": "user", "content":
            "你的数值研究尚未走真实工具, 当前缺少: " + ", ".join(missing) +
            "。请用工具依次完成: 对 fem_linear 与 iga_p2 各调用一次 convergence; 用 compare_methods 对比; "
            "用 predict_error 选一个更细网格预测 H1 误差; 用 verify_predict 回算对账。完成后再等我指令写报告。"})
        for _ in range(12):
            rr = client.chat.completions.create(model=args.model, messages=messages, tools=_TOOLS,
                                                tool_choice="auto", max_tokens=900, temperature=0.2)
            mm = rr.choices[0].message
            cc = mm.tool_calls or []
            if not cc:
                break
            for tc in cc[:1]:
                name, a = _pick(tc)
                result = exec_tool(name, a)
                print(f"[tool][补] {name} {tc.function.arguments}\n  -> {result}")
                trace.append(result); transcript.append(f"`{name}` {tc.function.arguments} → {result}")
                messages.append({"role": "assistant", "content": mm.content or "",
                                "tool_calls": [tc.model_dump() for tc in cc]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            if all(any(t.startswith(f"`{k}") for t in transcript) for k in required):
                break

    def _gen():
        return client.chat.completions.create(model=args.model, messages=messages, max_tokens=1800,
                                              temperature=0.2, extra_body={"thinking_mode": False}).choices[0].message.content or ""
    final = ""; verdict, ungrounded = "needs_grounding", []
    did_verify = any(t.startswith("`verify_predict") for t in transcript)
    for _ in range(3):
        final = _gen()
        g = verify(final, trace)
        print(f"\n[门禁] {g['verdict']} unsubstantiated={g['unsubstantiated']}")
        reasons = []
        if not did_verify: reasons.append("还差可证伪验证: 请 predict_error 后 verify_predict 回算对账")
        if g["verdict"] != "pass": reasons.append(f"未落地数值: {g['unsubstantiated']}")
        if len(final.strip()) < 200: reasons.append("报告过短, 请按完整结构重写")
        if not reasons:
            verdict, ungrounded = "pass", []; break
        ungrounded = g["unsubstantiated"]
        messages.append({"role": "user", "content": "未交付: " + "; ".join(reasons) + "。补齐后按(研究问题/数据与方法/结果分析/预测-对账/结论与局限/下一步)重写完整报告。"})
        messages.append({"role": "assistant", "content": final})

    rep = OUT / "ai4s_numerics_fem_iga_report.md"
    header = (f"# 计算数学开放研究 — 书生 Intern-S2 × Huginn · 新型有限元/等几何\n\n"
              f"> **结论证伪门禁: {verdict}**（未落地主张: {ungrounded or '无'}）\n"
              f"> 数值核心: examples/numerics.py（纯 Python 真实装配/求解，可复现）\n\n"
              f"## 一、工具执行轨迹（含真实误差与收敛阶）\n\n")
    body = "\n".join(f"- {t}" for t in transcript)
    rep.write_text(header + body + "\n\n## 二、结论(开放)\n\n" + final.strip() + "\n", encoding="utf-8")
    print("\n报告:", rep.resolve(), "| 门禁:", verdict, ungrounded)
    print("\n---- 正文(前1200字) ----\n", final[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())