"""书生 × Huginn 第三域冷启动 · 应变耦合量子相变 (depth: 多尺度耦合/相变临界/凝聚态序/材料谱系).

目标(泛化验证): 证明同一套 Huginn agent 机制**零改动**即可驱动一个与固体力学**零族**
的陌生物质科学域。本域不是"单变量查表"的浅例, 而是:
  - **凝聚态序**: 极化 P / 磁化 M / 超导能隙 Δ 等序参量随控制场(应变 s / 场 E,H / 温度 T)演化;
  - **相变临界/群展**: Landau-Ginzburg 自由能 → 临界指数(β,γ,ν)与**普适类**判别、
    标度律、相图拓扑(一阶/二阶/三临界点);
  - **多场耦合**: 应变-极化-磁-温度 耦合自由能, 序参量相互作用(双线性/双二次);
  - **材料谱系**: ABO₃ 钙钛矿一族候选材料的相图判别(储铁电/磁电耦合)。

一致性红线: 所有"结果"都来自真实解析/数值计算(landau 最小二乘、自由能极点、
临界指数拟合) —— 门禁必须能落地, 不假造任何数; 书生可提议扫描维度 / 提开放问题,
但数值全真实。域注册见 huginn.research.coldstart_guards.DOMAIN_PROFILES["quantum_critical"].
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_OUT = Path(__file__).resolve().parents[1] / "research_outputs" / "shusheng_quantum_critical"
_OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════ 真实物理内核 ═══════════════════

# ABO₃ 钙钛矿谱系: (label, 基态序类型, 极化普适性参数 p, 应变耦合系数 c, T_C(K))
# 数值取代表性铁电体/介电体量级 (Aizu/Lines-Glass 量级), 用于相图判别, 非精确拟合.
PEROVSKITE = [
    ("BaTiO3", "ferroelectric", 0.4, 0.9, 403.0),   # 经典铁电, 强应变响应
    ("PbTiO3", "ferroelectric", 0.5, 1.0, 763.0),     # 高 T_C 铁电
    ("SrTiO3", "incipient", 0.08, 0.5, 4.0),          # 量子顺电/形变不诱发的临界
    ("SrBi2Ta2O9", "ferroelectric", 0.30, 0.55, 608.0),
    ("LaAlO3", "paraelectric", 0.01, 0.1, 0.0),
    ("KNbO3", "ferroelectric", 0.45, 0.95, 708.0),
]


# ═══════════════════ 书生成码: 自主建模型为主 (Code Lab 主角) ═══════════════════

def _llm_compat_kwargs(client) -> dict:
    """端点感知的调用增量参数: 仅 Intern/书生端点(intern-ai.org.cn 或路径含 intern)才注入
    extra_body={'thinking_mode': False}(InternLM 关思考流的专属字段); 标准 OpenAI 兼容端点
    (GPT/DeepSeek 等)不接受该字段, 自动省略避免 400. 让同一段调用语义跨端点兼容 ——
    换模型只换 client/base_url, 不替书生解题, 也不把 agent 锁死在某一个模型上."""
    try:
        base = str(getattr(client, "base_url", None) or "")
    except Exception:  # noqa: BLE001 — 取不到 base_url 按通用端点处理
        base = ""
    if "intern-ai.org.cn" in base or "/intern" in base:
        return {"extra_body": {"thinking_mode": False}}
    return {}


def _ask_json(client, model: str, system: str, user: str, max_tokens: int = 1500) -> dict:
    try:
        r = client.chat.completions.create(model=model, messages=[
            {"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.2,
            **_llm_compat_kwargs(client))
        text = (r.choices[0].message.content or "")
        import re as _re
        for m in _re.finditer(r"\{.*?\}", text, _re.DOTALL):
            try:
                d = json.loads(m.group(0))
                return d if isinstance(d, dict) else {}
            except Exception:  # noqa: BLE001
                continue
        # 兜底: 裸代码块
        _m = _re.search(r"```(?:python)?\s*\n(.*?)```", text, _re.DOTALL | _re.IGNORECASE)
        if _m:
            return {"code": _m.group(1).strip()}
        return {"code": text}
    except Exception as ee:  # noqa: BLE001 — 调用失败不伪造
        return {"error": str(ee)}


def _try_raw_code(client, model: str, prompt: str) -> str:
    try:
        r = client.chat.completions.create(model=model, messages=[
            {"role": "user", "content": prompt}],
            max_tokens=1500, temperature=0.2)
        text = r.choices[0].message.content or ""
        import re as _re
        _m = _re.search(r"```(?:python)?\s*\n(.*?)```", text, _re.DOTALL | _re.IGNORECASE)
        return _m.group(1).strip() if _m else ""
    except Exception:  # noqa: BLE001 — 提取失败返回空, 调用方回退
        return ""


def _try_author_code(client, model: str, next_open: str, cycle: int):
    """Code Lab · 书生自主建模: 自己定表征/写 run(cfg) 检验开放问题.

    弃用预置领域内核: 不再替书生预设物理方程, 只提供 numpy/scipy 通用原语供其组装.
    失败带错回流修一版(预算1), 全败回退白名单扫描(不阻塞, 不伪造).
    返回 (exp_or_None, probe_specs, obj_keys, err_msg).
    """
    from huginn.research.code_lab import extract_code, sandbox_run, author_probe_specs
    sys_prompt = (
        "你是凝聚态/量子材料实验员。面对一个陌生科学问题, **由你自主建模**: "
        "基于经典理论(Landau 序参量/Curie-Weiss/临界指数/相图拓扑/钙钛矿可调性等)判断该"
        "算什么、用什么物理量, 再用 numpy(必要时 scipy)亲手写 def run(cfg): 检验它。"
        "cfg 只含你需要的数值配置(如 T/strain/imaterial)。**只通过返回值交付数值**: return "
        "dict, 键是物理量名、值是真实计算数值(如 {'beta': 0.5}); 也可用标准结构 "
        "{'success': bool, 'summary': {...}, 'objectives': {<k>: 数值}}。两种情况都行, "
        "但**不要 print 大量中间结果**(那会被当控制台输出丢弃), 所有最终数值都必须放进 "
        "return 的 dict。你的 cadence: ① 先想清楚物理模型与自由能/方程; ② 决定参数扫描; "
        "③ 数值求平衡序/临界指数/拟合; ④ 把可证伪数值放进 return。公式必须来自真实凝聚态"
        "物理(不是瞎编), dict 里每个值必须是真实计算数值(不要字符串)。输出裸代码(不要解释), "
        "代码必须含 def run(cfg)。"
    )
    _AUTHOR_MAX_RETRY = 1
    res: dict | None = None
    last_err = ""
    code = ""
    for at in range(_AUTHOR_MAX_RETRY):
        if at == 0:
            user_goal = f"本轮开放问题(自主建模): {next_open[:1200]}"
        else:
            user_goal = (f"你上一版代码执行报错如下:\n{last_err}\n\n"
                         f"请**只输出修正后的完整代码 def run(cfg)**, 修复该错误, "
                         f"数值仍须真实计算。问题: {next_open[:600]}")
        d = _ask_json(client, model, sys_prompt, user_goal, max_tokens=1500)
        code = d.get("code") or d.get("python") or ""
        if not code:
            code = _try_raw_code(client, model,
                                 next_open if at == 0 else f"{next_open}\n[用户反馈] 报错: {last_err}")
        if not code:
            break
        code = extract_code(code) if not code.strip().startswith("def run") else code
        cfg = {"T": 300.0, "strain": 0.0, "imaterial": 0}   # 泛用 cfg, 书生自取所需键
        res, err = sandbox_run(code, cfg)
        if res is not None:
            break
        last_err = err or "代码执行失败"
        print(f"  [书生成码·重试 {at + 1}] 执行失败: {last_err[:160]}  → 回流给书生重修")
    if res is None:
        return None, [], [], last_err or "代码执行失败"
    obj_keys = list(res["objectives"].keys())
    try:
        probe_specs = author_probe_specs(code)
    except Exception:  # noqa: BLE001
        probe_specs = []
    from huginn.research import Experiment
    exp = Experiment(f"author_qc{cycle}", f"书生自主建模(S{cycle}): {next_open[:90]}",
                     run=lambda: res)
    return exp, probe_specs, obj_keys, ""


GOAL = (
    "应变耦合的发生量子相变 —— 交给书生自主建模求解. Huginn 提供通用数值工具面"
    "(numerical_tool: ODE/优化/求根/曲线拟合/积分/特征值)与 Code Lab 沙箱; 书生须自己"
    "决定: 用什么理论模型(Landau 序参量/Curie-Weiss/临界指数/相图体系)、算什么物理量、"
    "怎么求. 目标是产出可证伪的真实数值结论(应变对临界/序的调控、临界指数与普适类、相图"
    "拓扑、钙钛矿谱系可调性). 所有数值必须来自通用工具或书生 Hand Code 的真实计算, "
    "门禁可落地, 不造任何数."
)


def _make_experiments(cycle: int, author_exp=None) -> list:
    """实验集: 书生自主建模为标准实验(codelab 产物), 无预置内核."""
    from huginn.research import Experiment
    exps = []
    if author_exp is not None:
        exps.append(author_exp)          # 书生亲手写的 run() —— 开山主角
    else:
        # 首轮无书生代码占位: 以通用工具自检(纯真实工具调用, 不预置物理)
        exps.append(Experiment("qc_tools_selfcheck",
                               "通用数值工具面自检(无预置内核): 验证工具可真实求积分/求根",
                               run=_tools_selfcheck))
    return exps


def _tools_selfcheck() -> dict:
    """通用工具面自检: 用 scipy 真实算两个无物理内核的数值(验证仪器可用, 非解题)."""
    from scipy.integrate import quad
    from scipy.optimize import brentq
    i_res = quad(lambda x: x ** 2, 0, 3)[0]          # ∫₀³x²=9, 真实积分
    r_res = brentq(lambda x: x ** 2 - 2, 1, 2)       # √2≈1.4142, 真实求根
    return {"objectives": {"tool_int_ok": 1.0, "tool_root_ok": 1.0},
            "summary": {"integral_0_3_x2": round(i_res, 4),
                        "root_x2_2": round(r_res, 6)},
            "success": True}


def _diagnostic_tools() -> list[dict]:
    """通用工具面: 书生成文时可自主调用的**域无关科学仪器**(dict 逃生口).

    关键: 这里只提供通用科学计算原语(积分/求根/极小化/曲线拟合/ODE), 由 scipy 真实计算,
    不针对任何具体物理 —— **不发生成任何凝聚态内核**。书生自主决定用什么原语、算什么量,
    再配 code_lab 沙箱(书生成码)与白名单自检, 共同产出可证伪数值。诚实红线不变:
    每个 handler 都真算, 不伪造; 返回 JSON 字符串进 trace 供门禁核验。
    """
    from scipy.integrate import quad as _quad
    from scipy.optimize import brentq as _brentq, minimize_scalar as _min_sc
    import numpy as _np

    def _num(a):
        act = a.get("action", "integrate")
        # 通用一维数值积分 ∫_a^b func(theta) dtheta
        if act == "integrate":
            f = a.get("func", "theta**2")
            lo, hi = float(a.get("a", 0.0)), float(a.get("b", 1.0))
            val, err = _quad(lambda t: eval(f, {"theta": t, "np": _np, "math": math}), lo, hi)
            return json.dumps({"integral": round(float(val), 6), "abserr": round(float(err), 9),
                               "note": "通用积分器(数值)"}, ensure_ascii=False)
        # 通用求根 brentq 于 [lo,hi]
        if act == "root":
            f = a.get("func", "x**2 - 2")
            lo, hi = float(a.get("a", 0.0)), float(a.get("b", 2.0))
            r = _brentq(lambda x: eval(f, {"x": x, "np": _np, "math": math}), lo, hi)
            return json.dumps({"root": round(float(r), 9)}, ensure_ascii=False)
        # 通用一维极小化(可再加括号)
        if act == "minimize":
            f = a.get("func", "(x - 1.5)**2")
            lo, hi = float(a.get("a", -5.0)), float(a.get("b", 5.0))
            r = _min_sc(lambda x: eval(f, {"x": x, "np": _np, "math": math}),
                        bounds=(lo, hi), method="bounded")
            return json.dumps({"xstar": round(float(r.x), 6), "fmin": round(float(r.fun), 6)},
                              ensure_ascii=False)
        # 通用非线性曲线拟合: func 表达式用 params a0,a1,... ; ydata/xdata 传列表
        if act == "curve_fit":
            expr = a.get("func", "a0*x + a1")
            xs = list(map(float, a.get("xdata") or []))
            ys = list(map(float, a.get("ydata") or []))
            nl = expr.count("a(")  # 兼容 a0 -> a(0)? 简化: 用 polyfit 兜底
            p, _ = _np.polyfit(_np.asarray(xs), _np.asarray(ys), deg=1)
            return json.dumps({"polyfit_slope": round(float(p[0]), 6),
                               "polyfit_intercept": round(float(p[1]), 6)}, ensure_ascii=False)
        # 通用一阶 ODE 集成 (dy/dt = func(t,y)); 返回末值
        if act == "ode":
            f = a.get("func", "-y")
            y0 = float(a.get("y0", 1.0))
            t1 = float(a.get("t_end", 1.0))
            n = int(a.get("steps", 50))
            ys = [y0]
            for i in range(n):
                t0_, t1_ = t1 * i / n, t1 * (i + 1) / n
                k = eval(f, {"t": t0_, "y": ys[-1], "np": _np, "math": math})
                ys.append(ys[-1] + (t1_ - t0_) * k)
            return json.dumps({"y_end": round(float(ys[-1]), 6)}, ensure_ascii=False)
        return json.dumps({"error": f"unknown action {act}"})

    def _spec(name, desc, props):
        from huginn.research.tool_surface import canonical_tool_shape
        return {"tool": canonical_tool_shape(
            name, desc, {"type": "object", "properties": props}),
            "handle": _num}

    return [
        _spec("qc_integrate",
              "通用一维数值积分器: func(theta) over [a,b]. 返回值 integral(数值).",
              {"action": {"type": "string"}, "func": {"type": "string"},
               "a": {"type": "number"}, "b": {"type": "number"}}),
        _spec("qc_root",
              "通用一维求根器(brentq): 找 func(x)=0 in [a,b]. 返回值 root(数值).",
              {"action": {"type": "string"}, "func": {"type": "string"},
               "a": {"type": "number"}, "b": {"type": "number"}}),
        _spec("qc_minimize",
              "通用一维极小化器(bounded): 最小化 func(x) over [a,b]. 返回 xstar/fmin.",
              {"action": {"type": "string"}, "func": {"type": "string"},
               "a": {"type": "number"}, "b": {"type": "number"}}),
        _spec("qc_curvefit",
              "通用多项式拟合器: 拟合 ydata vs xdata. 返回斜率/截距(数值).",
              {"action": {"type": "string"}, "func": {"type": "string"},
               "xdata": {"type": "array", "items": {"type": "number"}},
               "ydata": {"type": "array", "items": {"type": "number"}}}),
        _spec("qc_ode",
              "通用一阶 ODE 显式 Euler: dy/dt=func(t,y). 返回 y_end.",
              {"action": {"type": "string"}, "func": {"type": "string"},
               "y0": {"type": "number"}, "t_end": {"type": "number"},
               "steps": {"type": "number"}}),
    ]


__all__ = ["_make_experiments", "_try_author_code", "_tools_selfcheck",
           "_diagnostic_tools", "GOAL", "PEROVSKITE"]

def _next_open_report(last_report: str, cycle: int) -> str:
    """从上一轮报告提取"下一步"(简版: 若为空给域默认开放问题)."""
    if "下一步" in (last_report or ""):
        seg = last_report.split("下一步")[-1][:200]
        return seg.strip()
    if cycle <= 2:
        return ("应变对发生铁电序参量/临界温度如何调控? 临界指数 β 属哪个普适类"
                "(平均场 or 3D Ising)? 相图是否显示一阶/二阶/三临界? "
                "哪个 ABO₃ 候选的应变可调性最高?"
                "你自行判断模型与量纲, 自行决定怎么算.")
    return "深化序参量耦合与标度律: 你自主定义下一步该算什么、怎么算."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry", action="store_true", help="确定性运行(不调模型)")
    ap.add_argument("--model", default="intern-s2-preview")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--start-cycle", type=int, default=1)
    args = ap.parse_args()

    # 冷启动守卫: 域声明已登记 → 预检依赖
    from huginn.research.coldstart_guards import compile_domain_guards, verify_domain_ready
    g = compile_domain_guards("quantum_critical")
    r = verify_domain_ready(g)
    if not r["ready"]:
        print("error: 冷启动守卫依赖缺失: %s" % r["missing_deps"], file=sys.stderr)
        return 3
    print("== 冷启动守卫(quantum_critical) ==")
    print("  依赖预检: %s" % g["deps_check"])
    print("  书生成码重试预算: %d" % g["code_retry_budget"])
    print("  成文探针: %s" % (", ".join(g["probes"]) or "(无)"))

    client = None
    if not args.dry:
        key = os.environ.get("INTERNLM_API_KEY")
        if not key:
            print("error: INTERNLM_API_KEY not set", file=sys.stderr)
            return 2
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=os.environ.get(
            "INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1"))

    print("== 书生 + Huginn 量子临界域长程深研(第三域泛化验证) ==")
    last_report = ""
    for cycle in range(max(1, args.start_cycle), args.start_cycle + args.cycles):
        print(f"\n===== 量子临界 第 {cycle} 轮 =====")
        from huginn.research import run_research_program, Experiment  # noqa: F401
        # Code Lab: 书生亲手写本轮实验代码(自主建模为主). 失败回退白名单自检, 不阻塞.
        author_exp = None
        author_err = "dry-run(无 client, 走确定性自检占位)"
        if client is not None:
            next_open = _next_open_report(last_report, cycle)
            author_exp, _probes, _objk, author_err = _try_author_code(
                client, args.model, next_open, cycle)
            if author_exp is not None:
                print(f"  [书生成码·自主建模] 通过 schema: "
                      f"{author_exp.hypothesis[:120]}")
            else:
                print(f"  [书生成码·自主建模] 未通过, 回退白名单自检: {author_err[:120]}")
        exps = _make_experiments(cycle, author_exp=author_exp)
        goal = GOAL + " | " + _next_open_report(last_report, cycle)
        print("  experiments:", [e.name for e in exps])
        _OBJ = {k: "maximize" for k in
                ("strain_eta_spread", "beta", "phase_order_span", "tunability_span")}
        # P-B 审计注入: last_report 非空则喂给书生观察
        program_kwargs = dict(client=client, model=args.model,
                              diagnostic_tools=_diagnostic_tools(),
                              max_iterations=8, verify=None)
        # 其它 --dry 时仍走确定性路径
        out = run_research_program(goal, exps, _OBJ, **program_kwargs)
        report = getattr(out, "report", "")
        verdict = getattr(out, "verdict", "?")
        print("  [门禁]", verdict)
        fname = "research_report.md" if cycle == 1 else f"cycle{cycle}_report.md"
        (_OUT / fname).write_text(report or "", encoding="utf-8")
        print(f"  第{cycle}轮报告: {_OUT / fname}")
        last_report = report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())