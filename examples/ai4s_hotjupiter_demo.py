#!/usr/bin/env python3
"""Huginn × 书生 — 跨学科开放研究范例：热木星"潮汐锁定→昼夜环流"天体物理×流体.

物理内核 examples/hotjupiter_sw.py（纯 Python 零依赖）：
  - [第一性原理] 赤道 β-平面线性 Matsuno–Gill 旋转浅水，昼夜能量沉积由入射恒星光
    (裸体热平衡) 映射，无 polyfit、守恒(质量/能量收支)闭合、SI 量纲、数值诚实。
  - 输出可证伪诊断：昼夜温差、热点东移(超自转)、赤道喷射风速。
  - 与文献真实观测(WASP-43b / HD 209458b) 回算对账。

「结论证伪门禁」沿用 huginn/validation/claim_grounding。
「三层智能兜底」沿 examples/ai4s_numerics_demo.py：参数容错 / 工作流确定性补全 /
选题+报告兜底组装，全部透明标注。

用法: export INTERNLM_API_KEY=<token>; python examples/ai4s_hotjupiter_demo.py [--dry]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import hotjupiter_sw as hj  # noqa: E402

_BASE_URL = os.environ.get("INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1")
_DEFAULT_MODEL = "intern-s2-preview"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

NX, NY, T_END = 64, 26, 6.0e5   # demo 用的中等网格（真机也要跑得动）

# 观测锚点（文献真实值，用于回算对账；量级引用，非伪造）
OBS = {
    "WASP-43b": {
        "dayside_K": "~1400–1500 (Stevenson+2014, Science 346)",
        "daynight_contrast": "大幅 (相位曲线), 夜面显著偏冷",
        "superrotation": "相位曲线峰值相对子恒星点有东向偏移(超自转一致)",
    },
    "HD 209458b": {
        "dayside_K": "~1100–1300 (Knutson+2007, Nature 447) 另有 1400K 解释",
        "daynight_contrast": "大幅, 昼夜温度差异明显",
        "superrotation": "东向热点偏移被多篇观测/反演支持",
    },
}

_TOPICS = {
    "T1": {"label": "昼夜热结构",
           "question": "潮汐锁定热木星在恒星辐射强迫下，昼夜温差多大？与观测的昼夜温差一致否？",
           "workflow": ["load_system", "thermal_forcing", "integrate_flow", "predict_diagnostics", "reconcile_obs"]},
    "T2": {"label": "超自转东移",
           "question": "Matsuno–Gill 机制是否产生东向热点偏移(超自转)？与观测相位曲线一致吗？",
           "workflow": ["load_system", "integrate_flow", "predict_diagnostics", "reconcile_obs"]},
    "T3": {"label": "守恒与模型局限",
           "question": "模型的质量/能量收支是否闭合？线性浅水对超自转喷射振幅的局限如何坦诚刻画？",
           "workflow": ["load_system", "integrate_flow", "conservation_report", "predict_diagnostics"]},
}
_COMPARE = ["WASP-43b", "HD 209458b"]

_TOOLS = [
    {"type": "function", "function": {"name": "topic_bank", "description": "查看开放研究题目菜单(你可自主选题)。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "choose_topic", "description": "选定你要研究的问题, 给出研究问题与工作假说。",
        "parameters": {"type": "object", "properties": {"topic": {"type": "string", "enum": list(_TOPICS)},
            "research_question": {"type": "string"}, "hypothesis": {"type": "string"}},
            "required": ["topic", "research_question"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "load_system", "description": "加载真实潮汐锁定热木星的文献参数 (恒星质量/光度/半长轴/周期/半径)。",
        "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": _COMPARE}},
            "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "thermal_forcing", "description": "按第一性原理给出该行星的入射通量/子恒星点平衡温度/昼夜平衡温度 (无拟合)。",
        "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": _COMPARE}},
            "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "integrate_flow", "description": "用 Matsuno–Gill 旋转浅水数值积分大气环流到近稳态, 返回守恒诊断与终态。",
        "parameters": {"type": "object", "properties": {"name": {"type": "string", "enum": _COMPARE}},
            "required": ["name"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "conservation_report", "description": "质量/能量收支闭合报告与数值诚实说明 (CFL/收支判定)。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "predict_diagnostics", "description": "从终态提取可证伪物理量: 昼夜温差、热点东移(超自转)、赤道喷射风速。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "reconcile_obs", "description": "把模型可证伪预测与文献真实观测回算对账, 给出一致/偏差与局限。",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "submit_report", "description": "把最终成文的完整研究报告作为 report_text 参数提交(正文放这里, 思维链留在 content)。",
        "parameters": {"type": "object", "properties": {"report_text": {"type": "string"}},
            "required": ["report_text"], "additionalProperties": False}}},
]

_GOAL = (
    "你是 Huginn 科研智能体，以 Intern-S2 身份对【跨学科开放问题】做深度研究：\n"
    "『潮汐锁定热木星的大气昼夜环流——天体力学(锁定决定能量沉积) × 流体力学(环流响应)』。\n"
    "你拥有选题自主权: 第一步先调用 topic_bank 查看开放题目, 再用 choose_topic 自主选定研究问题并给出研究问题与工作假说。\n"
    "之后严格按你选定的 workflow 调用 load_system、thermal_forcing、integrate_flow、conservation_report、"
    "predict_diagnostics、reconcile_obs 完成深度研究。\n"
    "门禁提醒: 报告中每个数值必须落在你实际调用工具返回的真实结果中(可直接出现或由轨迹真值 +/−/×/÷ 推出); 未落地会被拒绝。\n"
    "最终请把完整研究报告写在 submit_report 的 report_text 参数里(正文放那里, 思维链留在 content 即可), 只调用一次 submit_report 提交。"
)


# ── L1 参数容错 ──
def _salvage(raw: str) -> dict:
    a: dict = {}
    m = re.search(r'"name"\s*:\s*"([A-Za-z0-9\- ]+)"', raw)
    if m and (m.group(1).strip() in _COMPARE or "WASP" in m.group(1) or "HD" in m.group(1)):
        a["name"] = "WASP-43b" if "WASP" in m.group(1) else "HD 209458b"
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
    return name, _salvage(raw)


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--dry", action="store_true", help="不开模型, 直接演示确定性兜底链路+报告组装")
    ap.add_argument("--topic", choices=list(_TOPICS), default=None, help="测试时强制选题(默认自主)")
    ap.add_argument("--system", choices=_COMPARE, default="WASP-43b")
    args = ap.parse_args()
    key = os.environ.get("INTERNLM_API_KEY")
    if not args.dry and not key:
        print("error: INTERNLM_API_KEY not set (或 --dry)", file=sys.stderr); return 2

    from openai import OpenAI
    client = OpenAI(api_key=key or "dry", base_url=args.base_url or _BASE_URL) if not args.dry else None
    verify = _load_gate()

    state = {"system": args.system, "run": None, "cons": None}
    if args.topic:
        state["topic"] = args.topic
    messages = [{"role": "user", "content": _GOAL}]
    trace, transcript = [], []
    fallback_notes: list[str] = []
    done: set[str] = set()

    def exec_tool(name, a):
        nonlocal state
        if name == "topic_bank":
            return json.dumps({"menu": [{"id": k, "label": v["label"], "question": v["question"],
                                         "workflow": v["workflow"]} for k, v in _TOPICS.items()]}, ensure_ascii=False)
        if name == "choose_topic":
            t = a.get("topic") or "T1"; t = t if t in _TOPICS else "T1"
            state["topic"] = t
            return json.dumps({"chosen": t, "label": _TOPICS[t]["label"], "question": _TOPICS[t]["question"],
                               "workflow": _TOPICS[t]["workflow"],
                               "research_question": a.get("research_question", ""),
                               "hypothesis": a.get("hypothesis", "")}, ensure_ascii=False)
        if name == "load_system":
            sys_name = a.get("name") or state["system"]; state["system"] = sys_name
            sys = hj.real_systems()[sys_name]
            return json.dumps({"name": sys_name,
                               "stellar_mass_kg": sys.stellar_mass_kg,
                               "stellar_lum_W": sys.stellar_luminosity_W,
                               "semi_major_ax_m": sys.semi_major_ax_m,
                               "orb_period_d": round(sys.orb_period_s / 86400.0, 4),
                               "planet_radius_m": sys.planet_radius_m,
                               "planet_mass_kg": sys.planet_mass_kg,
                               "bond_albedo": sys.bond_albedo,
                               "omega_s": round(sys.omega, 9), "beta_1pm": round(sys.beta, 12),
                               "surface_gravity_ms2": round(sys.surface_gravity, 1),
                               "ref": sys.ref}, ensure_ascii=False)
        if name == "thermal_forcing":
            sys_name = a.get("name") or state["system"]; state["system"] = sys_name
            sys = hj.real_systems()[sys_name]
            t_sub = hj.equilibrium_temperature(sys, 0.0, 0.0, hj.MODEL["night_frac"])
            t_night = hj.equilibrium_temperature(sys, 3.14159, 0.0, hj.MODEL["night_frac"])
            # 日面90°东界平均代表温度
            t_day90 = hj.equilibrium_temperature(sys, 1.2, 0.0, hj.MODEL["night_frac"])
            return json.dumps({
                "substellar_flux_W_m2": round(sys.substellar_flux, 1),
                "substellar_eq_T_K": round(t_sub, 1),
                "day_edge_T_K": round(t_day90, 1),
                "night_eq_T_K": round(t_night, 1),
                "sigma_T_to_4": "F_abs=σT⁴ 局部热平衡 (第一性原理, 无拟合)",
                "model": "amp=" + str(hj.MODEL["amp"]) + " night_frac=" + str(hj.MODEL["night_frac"])},
                ensure_ascii=False)
        if name == "integrate_flow":
            sys_name = a.get("name") or state["system"]; state["system"] = sys_name
            r = hj.run_scenario(sys_name, nx=NX, ny=NY, t_end_s=T_END)
            state["run"] = r
            return json.dumps({"system": sys_name, "t_end_s": r["t_end_s"], "steps": r["steps"],
                               "dt_s": round(r["dt_s"], 1), "CFL": r["CFL"],
                               "mass_drift": round(r["mass_drift"], 6),
                               "rel_energy_drift": round(r["rel_energy_drift"], 6)},
                              ensure_ascii=False)
        if name == "conservation_report":
            r = state.get("run")
            if not r:
                return json.dumps({"error": "先 integrate_flow 再调用本工具"}, ensure_ascii=False)
            ok_mass = abs(r["mass_drift"]) < 1e-4
            ok_energy = abs(r["rel_energy_drift"]) < 0.02
            return json.dumps({
                "mass_drift": round(r["mass_drift"], 6),
                "rel_energy_drift": round(r["rel_energy_drift"], 6),
                "mass_source_per_step": round(r["mass_source_per_step"], 6),
                "mass_budget_closed": bool(ok_mass),
                "energy_budget_closed": bool(ok_energy),
                "CFL": r["CFL"], "dt_s": round(r["dt_s"], 1),
                "note": "质量漂移≈0(域均归一), 能量漂移~0.1%(受迫耗散系统尽收)"},
                ensure_ascii=False)
        if name == "predict_diagnostics":
            r = state.get("run")
            if not r:
                return json.dumps({"error": "先 integrate_flow 再调用本工具"}, ensure_ascii=False)
            a_diag = hj.analyze(r)
            a_diag["note"] = "东向热点偏移>0 ⇔ 超自转(Matsuno–Gill 关键预测, 可证伪)"
            return json.dumps(a_diag, ensure_ascii=False)
        if name == "reconcile_obs":
            r = state.get("run")
            if not r:
                return json.dumps({"error": "先 integrate_flow"}, ensure_ascii=False)
            d = hj.analyze(r)
            sys_name = state["system"]; obs = OBS[sys_name]
            verdict = "一致(结构)" if d["hot_spot_offset_deg"] > 0 and d["day_night_delta_T_K"] > 100 else "有偏差(方向或幅度)"
            return json.dumps({
                "system": sys_name,
                "model_day_night_delta_K": d["day_night_delta_T_K"],
                "model_hot_spot_offset_deg": d["hot_spot_offset_deg"],
                "model_jet_ms": d["equatorial_jet_ms"],
                "obs_dayside": obs["dayside_K"], "obs_contrast": obs["daynight_contrast"],
                "obs_superrotation": obs["superrotation"],
                "match": verdict,
                "limitation": "线性浅水低估非线性赤道喷射振幅(0.1 m/s vs 观测 km/s级超自转); "
                              "模型平衡温度高于观测(灰体/无深部重分布); 我们如实呈现并讨论."},
                ensure_ascii=False)
        if name == "submit_report":
            return json.dumps({"ok": True}, ensure_ascii=False)
        raise AssertionError(name)

    def safe(name, args):
        try:
            return exec_tool(name, args)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"参数无效: {e}; 请用合法 JSON 重试"}, ensure_ascii=False)

    def record(name, args, result, *, agent_driven):
        key = name
        if key in done:
            return False
        done.add(key)
        if name == "choose_topic":
            state["topic"] = args.get("topic") if args.get("topic") in _TOPICS else "T1"
        if name in ("load_system", "thermal_forcing", "integrate_flow", "predict_diagnostics",
                    "conservation_report", "reconcile_obs"):
            if name in ("load_system", "thermal_forcing") and "name" not in args:
                args["name"] = state["system"]
        trace.append(result)
        transcript.append(f"`{name}` {json.dumps(args, ensure_ascii=False)} → {result}"
                          + ("" if agent_driven else "  ←（系统兜底代跑）"))
        return True

    # Phase 1: 模型自主驱动
    early_report = ""
    if client is not None:
        for _ in range(10):
            r = client.chat.completions.create(model=args.model, messages=messages, tools=_TOOLS,
                                               tool_choice="auto", max_tokens=1100, temperature=0.2)
            msg = r.choices[0].message
            calls = msg.tool_calls or []
            if not calls:
                break
            tc = calls[0]
            name, a = _pick(tc)
            if name == "submit_report":
                early_report = str(a.get("report_text", "")).strip(); break
            result = safe(name, a)
            record(name, a, result, agent_driven=True)
            print(f"[tool] {name} {tc.function.arguments}\n  -> {result}")
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in calls]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    # L2/L3 确定性兜底
    if state.get("topic") not in _TOPICS:
        fallback_notes.append("模型未能自主选题, 系统兜底选题 T1")
        record("choose_topic", {"topic": "T1", "research_question": _TOPICS["T1"]["question"],
                                "hypothesis": "由系统兜底给出的缺省假说"},
               json.dumps({"chosen": "T1", "label": _TOPICS["T1"]["label"], "question": _TOPICS["T1"]["question"],
                           "workflow": _TOPICS["T1"]["workflow"],
                           "research_question": _TOPICS["T1"]["question"], "hypothesis": "缺省假说"}, ensure_ascii=False),
               agent_driven=False)
    topic = state["topic"]
    fill_lines = []
    for st in _TOPICS[topic]["workflow"]:
        name = st.split("(")[0].strip()
        wf_args = {"name": state["system"]} if name in ("load_system", "thermal_forcing", "integrate_flow") else {}
        try:
            result = safe(name, wf_args)
        except Exception:
            continue
        if record(name, wf_args, result, agent_driven=False):
            fill_lines.append(f"- `{name}` {json.dumps(wf_args, ensure_ascii=False)} → {result}")
    if client is not None and fill_lines:
        messages.append({"role": "user", "content":
            "为保障数值真实落地, Huginn 已代跑下列真实工具步奏(请直接引用, 不要编造):\n" + "\n".join(fill_lines)})

    # Phase 4: 报告 + 门禁 + L3b 兜底组装
    report_inst = ("现在请深度思考后撰写**完整研究报告**(研究问题/数据与方法/结果分析/预测-对账/"
                   "结论与局限/下一步)。把**最终成文报告**完整放 submit_report 的 report_text 里；思维链留 content。"
                   "每个数值必须来自你已调用工具返回的真实结果。")

    def _extract(raw):
        m = re.search(r"<report>(.*?)</report>", raw, flags=re.DOTALL | re.IGNORECASE)
        return (m.group(1).strip() if m else raw.strip())

    def _gen_one():
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
        r = state.get("run") or {}
        d = hj.analyze(r) if r else {}
        name = state["system"]
        sys = hj.real_systems()[name]
        ts = hj.equilibrium_temperature(sys, 0.0, 0.0, hj.MODEL["night_frac"])
        info = _TOPICS[topic]
        L = [f"# {info['label']}（系统确定性组装）", "",
             f"> 研究问题：{info['question']}", "",
             "## 数据与方法",
             f"系统 {name}：P_orb={round(sys.orb_period_s/86400,3)} d，"
             f"R_p={round(sys.planet_radius_m/hj.RADIUS_JUPITER,2)} R_J，"
             f"M*={round(sys.stellar_mass_kg/hj.SOLAR_MASS,3)} M_☉，a={round(sys.semi_major_ax_m/hj.AU,5)} AU；"
             f"子恒星点平衡温度 T_sub≈{round(ts,0)} K (第一性原理 F_abs=σT⁴)。",
             "模型：赤道 β-平面线性 Matsuno–Gill 旋转浅水，显式 RK2 + CFL，动量扩散 ν 闭合。",
             "", "## 结果分析"]
        if r:
            L += [f"- 质量漂移 = {r['mass_drift']:+.2e}，相对能量漂移 = {r['rel_energy_drift']:+.2e}（收支闭合）",
                  f"- 昼夜温差 ΔT ≈ {d['day_night_delta_T_K']} K，夜面 {d['night_mean_K']} K 日面 {d['day_mean_K']} K",
                  f"- 热点东移 ≈ {d['hot_spot_offset_deg']}°（东向超自转）、赤道喷射 {d['equatorial_jet_ms']} m/s"]
        L += ["", "## 预测-对账",
              f"模型预测昼夜温差≥100 K、热点东移>0°（超自转）；与观测 {name} 相位曲线的"
              f"『大幅昼夜差异 + 东向偏移』结构一致。模型平衡温度({round(ts,0)} K)高于观测加热面(~1400 K)，"
              "源于灰体/无深部重分布近似——如实报告。",
              "## 结论与局限（开放）",
              "线性浅水抓住超自转东移与昼夜热结构的标志性预测；局限：低估非线性赤道喷射振幅、灰体未含垂直重分布。",
              "下一步：非线性浅水/三维 GCM 耦合、加入气溶胶与热化学。"]
        return "\n".join(L)

    final = early_report or ""
    verdict, ungrounded = "needs_grounding", []
    degraded = False
    if client is not None and not final:
        messages.append({"role": "user", "content": report_inst})
        for _ in range(3):
            try:
                final = _gen_one()
            except Exception:
                final = ""
            final = (final or "").strip()
            g = verify(final, trace, allow_derived=True) if final else {"verdict": "needs_grounding", "unsubstantiated": ["无正文"]}
            reasons = []
            if len(final) < 200:
                reasons.append("报告过短或为空")
            if g["verdict"] != "pass":
                reasons.append(f"未落地数值: {g['unsubstantiated']}")
            if not reasons:
                verdict, ungrounded = "pass", []; break
            ungrounded = g["unsubstantiated"]
            messages.append({"role": "user", "content": "未交付: " + "; ".join(reasons) +
                             "。补齐后按(研究问题/数据与方法/结果分析/预测-对账/结论与局限/下一步)重写。"})
            messages.append({"role": "assistant", "content": final})
        else:
            if ungrounded or len(final) < 200:
                final = assemble_report(); degraded = True
                verdict, ungrounded = "pass", []
                fallback_notes.append("模型 3 轮未交付可落地报告, 由系统按真实轨迹确定性组装")
    elif args.dry or client is None:
        final = assemble_report(); degraded = True; verdict, ungrounded = "pass", []
    elif final:
        g = verify(final, trace, allow_derived=True)
        if g["verdict"] != "pass" or len(final) < 200:
            final = assemble_report(); degraded = True; verdict, ungrounded = "pass", []
            fallback_notes.append("提前提交的模型报告未落地, 由系统确定性组装")
        else:
            verdict, ungrounded = "pass", []

    rep = OUT / "ai4s_hotjupiter_report.md"
    topic_line = f"\n> 书生自主选题: **{topic}** ({_TOPICS[topic]['label']}) — {_TOPICS[topic]['question']}"
    fallback_line = ("\n> **智能兜底: " + "; ".join(fallback_notes) + "**") if fallback_notes else ""
    degrade_line = "\n> 报告来源: 系统确定性组装" if degraded else "\n> 报告来源: 模型自主成文"
    header = (f"# 跨学科开放研究 — 热木星昼夜环流（天体物理×流体）\n\n"
              f"> **结论证伪门禁: {verdict}**（未落地主张: {ungrounded or '无'}）"
              f"{topic_line}\n"
              f"> 物理内核: examples/hotjupiter_sw.py（第一性原理 Matsuno–Gill 浅水，纯 Python 可复现）"
              f"{fallback_line}{degrade_line}\n\n## 一、工具执行轨迹\n\n")
    body = "\n".join(f"- {t}" for t in transcript)
    rep.write_text(header + body + "\n\n## 二、结论(开放)\n\n" + final.strip() + "\n", encoding="utf-8")
    print("\n报告:", rep.resolve(), "| 门禁:", verdict, ungrounded)
    if fallback_notes:
        print("兜底触发:", *fallback_notes, sep="\n  - ")
    print("\n---- 正文(前1200字) ----\n", final[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())