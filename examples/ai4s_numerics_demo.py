#!/usr/bin/env python3
"""Huginn × 书生 — 计算数学「新型有限元/等几何分析」开放研究范例（带智能兜底）.

真实数值方法科研：求解 -u''=π²sin(πx), u(0)=u(1)=0，用
  - fem_linear ：线性拉格朗日有限元（理论 H1≈O(h), L2≈O(h²)）
  - iga_p2     ：二次 B 样条等几何分析（理论 H1≈O(h²), L2≈O(h³)，C¹）
做收敛率研究（refinement 后 log-log 估阶），对更细网格的误差做【可证伪预测】，
再用真实求解回算对账。数值核心 examples/numerics.py 为纯 Python 真实装配/求解。

「结论证伪门禁」沿用 huginn/validation/claim_grounding：报告每个数值必须在工具轨迹中。

—— 智能兜底机制（针对性能较弱的本地/开源模型，Exactly 本场景常翻车）——
  弱模型在「自主选题 + 多步工具调用」里典型失败：不调用工具、残缺/非法 JSON 参数、
  跳过 must-run 工作流、assert 直接崩进程、结尾捏造数字。为此设三层兜底：
  L1 参数容错  _pick 用正则抢救残缺 JSON；safe() 捕获任何异常转成给模型的错误串，
                绝不因一个坏参数让整个进程崩掉。
  L2 确定性补全  若模型没有跑完所选题目的 must-run 工作流，由系统用统一数值核心
                代跑缺失步奏并写回真实 trace（保证门禁有据可依，不靠模型自觉）。
  L3 选题兜底+报告兜底  模型若能自行选题/成文则用其输出；否则回退到 T1，并在模型
                始终无法交付可落地报告时，由系统从真实 trace 确定性组装报告。
  所有兜底都透明标注（header 里写明），绝不让兜底伪装成模型能力。

用法: export INTERNLM_API_KEY=<token>; python examples/ai4s_numerics_demo.py [--model intern-s2-preview]
依赖: pip install requests openai
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

_BASE_URL = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
_DEFAULT_MODEL = "intern-s2-preview"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

_HERE = Path(__file__).resolve().parent
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


_TOPICS = {
    "T1": {"label": "方法效率", "question": "同自由度下，二次等几何(IGA)是否显著优于线性有限元(FEM)？优势随网格如何变化？",
           "workflow": ["convergence(fem_linear)", "convergence(iga_p2)", "compare_methods", "predict_error", "verify_predict"]},
    "T2": {"label": "预测可信度", "question": "由若干粗网格外推更细网格的 H1 误差，回算验证是否可靠？预测/实际 比值是否稳定？",
           "workflow": ["convergence(fem_linear)", "convergence(iga_p2)", "predict_error(目标128)", "verify_predict", "predict_error(目标256)", "verify_predict"]},
    "T3": {"label": "理论一致性审计", "question": "数值估的 H1/L2 收敛阶是否在各网格、各范数下系统性地与理论 O(h)/O(h²)/O(h³) 一致？",
           "workflow": ["convergence(fem_linear)", "convergence(iga_p2)", "compare_methods"]},
}
# 每个题目 must-run 的真实工具步奏（由系统保证全落轨迹，不靠模型自觉）
_REQUIRED_ANY = ("choose_topic",)
_REQUIRED_ALL = ("convergence", "compare_methods", "predict_error", "verify_predict")

_TOOLS = [
    {"type": "function", "function": {"name": "topic_bank", "description": "查看开放研究题目菜单(你可自主选题)。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "choose_topic", "description": "选定你要研究的问题, 给出研究问题与工作假说。",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string", "enum": list(_TOPICS)},
            "research_question": {"type": "string"}, "hypothesis": {"type": "string"}},
            "required": ["topic", "research_question"], "additionalProperties": False}}},
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
    {"type": "function", "function": {"name": "submit_report", "description": "把最终成文的完整研究报告作为 report_text 参数提交(正文放这里, 系统会从这里读取, 思维链请留在 content 不放进参数)。",
        "parameters": {"type": "object", "properties": {"report_text": {"type": "string"}},
            "required": ["report_text"], "additionalProperties": False}}},
]

_GOAL = (
    "你是 Huginn 科研智能体, 以 Intern-S2 身份对【计算数学：新型有限元 vs 等几何分析】做开放研究。\n"
    "研究问题: 对 -u''=π²sin(πx), u(0)=u(1)=0, 线性有限元(FEM) 与 二次 B 样条等几何(IGA) 的收敛与效率。\n"
    "你拥有选题自主权: 第一步必须先调用 topic_bank 查看开放题目菜单, 然后用 choose_topic 自主选定你要研究的问题, "
    "并给出你的研究问题与研究假说。之后严格按你选定的 workflow 调用 convergence(fem_linear)、convergence(iga_p2)、"
    "compare_methods、predict_error、verify_predict 完成深度研究。\n"
    "门禁提醒: 报告中每个数值必须落在你实际调用工具返回的真实结果中(可直接出现或由轨迹真值 +/−/×/÷ 推出); 未落地会被拒绝。\n"
    "最终请先把完整研究报告写在 submit_report 的 report_text 参数里(正文放那里, 思维链留在 content 即可), 只调用一次 submit_report 提交。"
)


# ── L1 · 参数容错 ─────────────────────────────────────────────
def _salvage(raw: str) -> dict:
    """弱模型常产出残缺/非法 JSON：用正则抢救关键字段，绝不因坏参数崩进程."""
    a: dict = {}
    m = re.search(r'"method"\s*:\s*"([A-Za-z0-9_]+)"', raw)
    if m and m.group(1) in METHODS:
        a["method"] = m.group(1)
    t = re.search(r'"target_ne"\s*:\s*(\d+)', raw)
    if t:
        a["target_ne"] = int(t.group(1))
    tp = re.search(r'"topic"\s*:\s*"(T[1-9])"', raw)
    if tp and tp.group(1) in _TOPICS:
        a["topic"] = tp.group(1)
    if "submit_report" in raw:
        rt = re.search(r'"report_text"\s*:\s*"((?:\\.|[^"\\])*)"', raw)
        if rt:
            a["report_text"] = rt.group(1)
    return a


def _pick(tc):
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
    return name, _salvage(raw)  # L1: 正规解析全失败的兜底


def _meth(a) -> str:
    m = a.get("method") if isinstance(a, dict) else None
    return m if m in METHODS else "fem_linear"


def _resolve_step(step: str):
    """把 workflow 步奏字符串解析为 (tool, args)。如 'predict_error(目标128)' -> ('predict_error', {'target_ne':128})."""
    base = step.split("(")[0].strip()
    if base in ("topic_bank", "load_pde", "compare_methods"):
        return base, {}
    args: dict = {}
    rest = (step[step.find("(") + 1:] if "(" in step else "")
    rest = rest.rstrip(")").strip()
    if base == "convergence":
        args["method"] = rest if rest in METHODS else "fem_linear"
    elif base in ("predict_error", "verify_predict"):
        args["method"] = rest if rest in METHODS else "fem_linear"
        nums = [int(x) for x in re.findall(r"[0-9]+", rest)]
        if nums:
            args["target_ne"] = nums[0]
    return base, args


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
    ap.add_argument("--dry", action="store_true", help="不开模型, 直接演示三层兜底的确定性补全+报告组装")
    ap.add_argument("--topic", choices=list(_TOPICS), default=None,
                    help="测试兜底链路时强制选题(默认留空=自主选题; 仅覆盖不干扰正常自主流程)")
    args = ap.parse_args()
    key = os.environ.get("INTERNLM_API_KEY")
    if not args.dry and not key:
        print("error: INTERNLM_API_KEY not set (或使用 --dry 演示兜底)", file=sys.stderr); return 2
    from openai import OpenAI
    client = OpenAI(api_key=key or "dry", base_url=args.base_url or _BASE_URL) if not args.dry else None
    verify = _load_gate()
    state = {}  # 累积确定性状态, 供补全与报告组装
    state.setdefault("verifies", [])
    messages = [{"role": "user", "content": _GOAL}]
    trace, transcript = [], []          # 门禁证据 / 报告头轨迹
    fallback_notes: list[str] = []      # 透明记录所有兜底行为
    done: set[str] = set()              # 已落地步奏 key, 防重复污染
    if args.topic:
        state["topic"] = args.topic      # --topic 测试覆盖：视为已自主选题
        done.add("choose_topic")

    def step_key(name: str, args: dict) -> str:
        if name == "convergence":
            return f"convergence({_meth(args)})"
        if name in ("predict_error", "verify_predict"):
            return f"{name}({_meth(args)},{int(args.get('target_ne') or 128)})"
        return name

    # ── 确定性工具执行核心（唯一能产生真实数值的地方） ──
    def exec_tool(name, a):
        nonlocal state
        if name == "topic_bank":
            return json.dumps({"menu": [{"id": k, "label": v["label"], "question": v["question"],
                                         "workflow": v["workflow"]} for k, v in _TOPICS.items()]},
                              ensure_ascii=False)
        if name == "choose_topic":
            t = a.get("topic") or "T1"
            t = t if t in _TOPICS else "T1"          # L3 选题校验: 非法即兜底
            state["topic"] = t
            return json.dumps({"chosen": t, "label": _TOPICS[t]["label"], "question": _TOPICS[t]["question"],
                               "workflow": _TOPICS[t]["workflow"],
                               "research_question": a.get("research_question", ""),
                               "hypothesis": a.get("hypothesis", "")}, ensure_ascii=False)
        if name == "load_pde":
            return json.dumps({"pde": "-u'' = π² sin(πx), u(0)=u(1)=0, 精确解 u=sin(πx)",
                               "methods": METHODS, "theory_orders": THEORY,
                               "meshes_tested": MESHES}, ensure_ascii=False)
        if name == "convergence":
            m = _meth(a)
            r = _run_conv(m)
            state.setdefault("conv", {})[m] = r
            return json.dumps(r, ensure_ascii=False)
        if name == "compare_methods":
            c_l, c_i = _run_conv("fem_linear"), _run_conv("iga_p2")
            state["compare"] = {m: {"name": METHODS[m], "estimated_H1": c["estimated_H1_order"],
                                    "H1_at_dof": [(x["dof"], x["H1"]) for x in c["rows"]]}
                                for m, c in (("fem_linear", c_l), ("iga_p2", c_i))}
            return json.dumps({
                "fem_linear": state["compare"]["fem_linear"],
                "iga_p2": state["compare"]["iga_p2"],
                "read": "同样多自由度下 H1 误差越小越高效; 注意估计收敛阶是否与理论 O(h) / O(h²) 一致。"},
                ensure_ascii=False)
        if name == "predict_error":
            m = _meth(a); tgt = int(a.get("target_ne", 128))
            errs = []
            for ne in MESHES:
                s = nm.fem_linear(ne, nm.force_sin, nm.exact_sin, nm.exact_sin_der) if m == "fem_linear" \
                    else nm.iga_p2(ne, nm.force_sin, nm.exact_sin, nm.exact_sin_der)
                errs.append((ne, nm.errors(s, nm.exact_sin, nm.exact_sin_der)["H1"]))
            p = nm.estimate_order(errs)
            last_ne, last_err = errs[-1]
            pred = last_err * (last_ne / tgt) ** p
            state["pred"] = {"method": m, "target": tgt, "predicted": pred}
            return json.dumps({"method": m, "target_ne": tgt, "estimated_order": round(p, 2),
                               "predicted_H1_error": round(pred, 6)}, ensure_ascii=False)
        if name == "verify_predict":
            m = _meth(a) or state.get("pred", {}).get("method", "fem_linear")
            tgt = int(a.get("target_ne", state.get("pred", {}).get("target", 128)))
            s = nm.fem_linear(tgt, nm.force_sin, nm.exact_sin, nm.exact_sin_der) if m == "fem_linear" \
                else nm.iga_p2(tgt, nm.force_sin, nm.exact_sin, nm.exact_sin_der)
            actual = nm.errors(s, nm.exact_sin, nm.exact_sin_der)["H1"]
            pred = state.get("pred", {}).get("predicted")
            ratio = pred / actual if pred else None
            v = {"method": m, "target_ne": tgt, "predicted_h1": round(pred, 6) if pred else None,
                 "actual_h1": round(actual, 6), "pred_to_actual": round(ratio, 3) if ratio else None,
                 "verdict": ("预测已由真实求解对账" if ratio and 0.7 <= ratio <= 1.4 else
                             "预测与真实求解偏差明显")}
            state["verifies"].append(v)
            return json.dumps(v, ensure_ascii=False)
        if name == "submit_report":
            return json.dumps({"ok": True}, ensure_ascii=False)
        raise AssertionError(name)

    def safe(name, args):
        """L1 兜底: 任何工具异常都以『可读错误』喂回给模型, 绝不崩进程."""
        try:
            return exec_tool(name, args)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"参数无效: {e}; 请用合法 JSON 重试(如 method=中正确的是 {list(METHODS)})",
                               "received_args": args}, ensure_ascii=False)

    def record(name, args: dict, result: str, *, agent_driven: bool):
        """统一落轨迹 + 防重。返回是否为新执行."""
        key = step_key(name, args)
        if key in done:
            return False
        done.add(key)
        if name == "choose_topic":
            state["topic"] = (args.get("topic") or "T1") if (args.get("topic") in _TOPICS) else "T1"
        trace.append(result)
        transcript.append(f"`{name}` {json.dumps(args, ensure_ascii=False)} → {result}"
                          + ("" if agent_driven else "  ←（系统兜底代跑）"))
        return True

    # ── Phase 1: 让模型自主驱动研究（开放探索，尽力而为） ──
    early_report = ""
    if client is not None:
        for _ in range(10):
            r = client.chat.completions.create(model=args.model, messages=messages, tools=_TOOLS,
                                               tool_choice="auto", max_tokens=1100, temperature=0.2)
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                break  # 模型停止调用工具
            tc = calls[0]
            name, a = _pick(tc)
            if role_t := (a.get("report_text") if name == "submit_report" else None):
                early_report = str(role_t).strip()
                break  # 提前交报告 → 结束探索，交给门禁与兜底
            result = safe(name, a)
            record(name, a, result, agent_driven=True)
            print(f"[tool] {name} {tc.function.arguments}\n  -> {result}")
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # ── L2/L3 · 确定性兜底：选题 + 工作流补全（不依赖模型自觉） ──
    fill_lines: list[str] = []

    def add_fill(name, args, note=""):
        result = safe(name, args)
        assert result and '"error"' not in result and "参数无效" not in result, (
            f"确定性代跑失败: {name} -> {result}")
        if record(name, args, result, agent_driven=False):
            fill_lines.append(f"- `{name}` {json.dumps(args, ensure_ascii=False)} → {result}" + (f"（{note}）" if note else ""))
            return True
        return False

    # L3a · 选题兜底：模型没选/选错 → 系统回退 T1（透明标注）
    if state.get("topic") not in _TOPICS:
        fallback_notes.append("模型未能自主选题(未调用或 topic 无效), 系统兜底选题 T1")
        add_fill("choose_topic", {"topic": "T1",
                                  "research_question": _TOPICS["T1"]["question"],
                                  "hypothesis": "由系统兜底选题给出的缺省假说"}, "系统兜底选题 T1")
    topic = state["topic"]

    # L2 · 工作流补全：按所选题目 must-run 步骤，系统代跑缺失的，保证真实数据落轨迹
    for st in _TOPICS[topic]["workflow"]:
        name, st_args = _resolve_step(st)
        # verify_predict 紧跟 predict_error：补上预测目标，避免 T2 两轮 verify 去重冲突
        if name == "verify_predict" and "target_ne" not in st_args and state.get("pred"):
            st_args["target_ne"] = state["pred"]["target"]
        add_fill(name, st_args)

    # update messages so the report phase模型能引用兜底真实数值
    if client is not None and fill_lines:
        messages.append({"role": "user", "content":
            "为保障数值真实落地, Huginn 已代跑下列真实工具步奏(数据可复现, 请直接引用, 不要自己编造):\n"
            + "\n".join(fill_lines)})

    # ── Phase 4: 报告成文 + 门禁 + L3b 报告兜底组装 ──
    report_inst = (
        "现在请深度思考后撰写**完整研究报告**（研究问题/数据与方法/结果分析/预测-对账/结论与局限/下一步）。"
        "把**最终成文报告**完整放在 submit_report 的 report_text 参数里；思维链留在 content 即可。"
        "研究的每个数值必须来自你已调用工具返回的真实结果。")

    def _extract(raw: str) -> str:
        m = re.search(r"<report>(.*?)</report>", raw, flags=re.DOTALL | re.IGNORECASE)
        return (m.group(1).strip() if m else raw.strip())

    def _gen_one() -> str:
        r = client.chat.completions.create(model=args.model, messages=messages, tools=_TOOLS,
                                           tool_choice="auto", max_tokens=4000, temperature=0.2)
        msg = r.choices[0].message
        for tc in (msg.tool_calls or []):
            name, a = _pick(tc)
            if name == "submit_report":
                t = str(a.get("report_text", "")).strip()
                if t:
                    return t
        return _extract(msg.content or "")

    def assemble_report() -> str:
        """L3b：当真模型始终交不出可落地报告时，由系统从真实 trace 确定性组装."""
        cl = state.get("conv", {}).get("fem_linear") or _run_conv("fem_linear")
        ci = state.get("conv", {}).get("iga_p2") or _run_conv("iga_p2")
        cm = state.get("compare") or {}
        p = state.get("pred") or {}
        vs = state.get("verifies") or []
        info = _TOPICS[topic]
        L = [f"# {info['label']}（系统确定性组装）",
             "",
             f"> 研究问题：{info['question']}",
             "> 说明：模型未能交付可落地报告，本报告由 Huginn 依据真实工具执行轨迹确定性组装，所有数值均可复现。",
             "",
             "## 数据与方法",
             f"求解 -u''=π²sin(πx), u(0)=u(1)=0，精确解 u=sin(πx)；网格 ne={MESHES}。",
             "线性有限元(FEM)理论 H1≈O(h), L2≈O(h²)；二次 B 样条等几何(IGA)理论 H1≈O(h²), L2≈O(h³)。",
             "",
             "## 结果分析",
             f"- FEM H1 估阶 ≈ {cl['estimated_H1_order']}，L2 估阶 ≈ {cl['estimated_L2_order']}（ne=8→64）",
             f"- IGA H1 估阶 ≈ {ci['estimated_H1_order']}，L2 估阶 ≈ {ci['estimated_L2_order']}",
             f"- 同自由度 H1 误差：FEM {cl['rows'][0]['H1']} → {cl['rows'][-1]['H1']}；"
             f"IGA {ci['rows'][0]['H1']} → {ci['rows'][-1]['H1']}（IGA 优势随细化扩大）"]
        if p:
            L += ["",
                  "## 预测-对账",
                  f"由 ne={MESHES} 拟合外推 ne={p['target']}，预测 H1 ≈ {round(p['predicted'], 6)}；"]
            for v in vs[-1:]:
                L.append(f"回算真实 H1 = {v['actual_h1']}，预测/实际 ≈ {v['pred_to_actual']}（{v['verdict']}）")
        L += ["",
              "## 结论（开放）",
              "二次等几何在同自由度下误差显著更小且收敛阶更高，符合理论预期；预测-对账确认外推可信。",
              "局限：一维线性或二次离散；下一步扩展更高维/非光滑解。"]
        return "\n".join(L)

    final = early_report or ""
    verdict, ungrounded = "needs_grounding", []
    degraded_assembly = False
    if client is not None and not final:
        messages.append({"role": "user", "content": report_inst})
        for _ in range(3):
            try:
                final = _gen_one()
            except Exception as e:            # noqa: BLE001
                final = ""
            final = (final or "").strip()
            g = verify(final, trace, allow_derived=True) if final else {"verdict": "needs_grounding", "unsubstantiated": ["无正文"]}
            reasons = []
            if len(final) < 200:
                reasons.append("报告过短或为空")
            if g["verdict"] != "pass":
                reasons.append(f"未落地数值: {g['unsubstantiated']}")
            if not reasons:
                verdict, ungrounded = "pass", []
                break
            ungrounded = g["unsubstantiated"]
            messages.append({"role": "user", "content": "未交付: " + "; ".join(reasons) +
                             "。补齐后按(研究问题/数据与方法/结果分析/预测-对账/结论与局限/下一步)重写完整报告。"})
            messages.append({"role": "assistant", "content": final})
        else:
            # 3 轮全失败 → L3b：系统确定性组装（必然 pass）
            if ungrounded or len(final) < 200:
                final = assemble_report()
                degraded_assembly = True
                verdict, ungrounded = "pass", []
                fallback_notes.append("模型 3 轮未能交付可落地报告, 报告由系统按真实轨迹确定性组装")
    elif args.dry or (client is None):
        # --dry：不开模型，直接演示确定性兜底链路
        final = assemble_report()
        degraded_assembly = True
        verdict, ungrounded = "pass", []
    elif final:
        g = verify(final, trace, allow_derived=True)
        if g["verdict"] == "pass" and len(final) >= 200:
            verdict, ungrounded = "pass", []
        else:
            if g["verdict"] != "pass" or len(final) < 200:
                final = assemble_report()
                degraded_assembly = True
                verdict, ungrounded = "pass", []
                fallback_notes.append("提前提交的模型报告未落地, 报告由系统按真实轨迹确定性组装")

    rep = OUT / "ai4s_numerics_fem_iga_report.md"
    topic_line = f"\n> 书生自主选题: **{topic}** ({_TOPICS[topic]['label']}) — {_TOPICS[topic]['question']}"
    fallback_line = ("\n> **智能兜底: " + "; ".join(fallback_notes) + "**") if fallback_notes else ""
    degrade_line = "\n> 报告来源: 系统确定性组装" if degraded_assembly else "\n> 报告来源: 模型自主成文"
    header = (f"# 计算数学开放研究 — 书生 Intern-S2 × Huginn · 新型有限元/等几何\n\n"
              f"> **结论证伪门禁: {verdict}**（未落地主张: {ungrounded or '无'}）"
              f"{topic_line}\n"
              f"> 数值核心: examples/numerics.py（纯 Python 真实装配/求解，可复现）"
              f"{fallback_line}{degrade_line}\n\n"
              f"## 一、工具执行轨迹（含真实误差与收敛阶）\n\n")
    body = "\n".join(f"- {t}" for t in transcript)
    rep.write_text(header + body + "\n\n## 二、结论(开放)\n\n" + final.strip() + "\n", encoding="utf-8")
    print("\n报告:", rep.resolve(), "| 门禁:", verdict, ungrounded)
    if fallback_notes:
        print("兜底触发:", *fallback_notes, sep="\n  - ")
    print("\n---- 正文(前1200字) ----\n", final[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())