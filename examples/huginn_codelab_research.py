"""命题无关的 Huginn 原生 Code Lab 自主研究循环.

与 `shusheng_huginn_workflow.py`(逐轮硬编码的 X7 研究日志)不同, 本文件把**研究
目标**当作外部输入(`--objective`), 不绑定任何具体命题:

  书生(Intern)读目标 → 提开放问题 → 在 Code Lab 里**亲手写** numpy 实验代码 →
  `huginn.research.code_lab` 安全沙箱真实执行 → 数值进 `run_research_program`
  的分层流式结算(P-A)+ 层间重规划(P-B)+ 提前终止(P-C)+ claim_grounding 门禁 →
  成文报告 → 下一轮读报告里的开放问题继续. 目标换一行参数即可复用整条管线.

诚实边界: 代码执行异常/超时/schema 不过 → 该分支弃用, **不产生证据也不伪造**;
所有数值都来自沙箱真实执行, 经门禁核对才成文.

用法:
  INTERNLM_API_KEY=... python examples/huginn_codelab_research.py \
      --objective "你的研究目标" --domain rigidity --cycles 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_AGENT = Path(__file__).resolve().parents[1] / "agent"
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

_OUT = Path(__file__).resolve().parents[1] / "research_outputs" / "shusheng_huginn_native"

_AUTHOR_TEMPLATE = (
    "import numpy as np\n"
    "def run(cfg):\n"
    "    seed = int(cfg.get('seed', 0))\n"
    "    rng = np.random.default_rng(seed)\n"
    "    # ... 你的真实数值实验逻辑(纯 numpy) ...\n"
    "    return {\"success\": True,\n"
    "            \"summary\": {\"computed\": True},\n"
    "            \"objectives\": {\"score\": 0.0}}\n"
    "def probe_author_probe(cfg):\n"
    "    return {\"note\": \"可选诊断探针; 成文期可自主调用\"}"
)


def _llm_compat_kwargs(client) -> dict:
    """仅 Intern/书生端点注入 extra_body={'thinking_mode': False}(关思考流).

    Intern ChatAPI 的 thinking_mode 把思维链写进 content 字段, 会吃满预算拖空正文;
    标准 OpenAI 兼容端点不接受该字段 → 自动省略, 跨端点调用面不变.
    """
    try:
        base = str(getattr(client, "base_url", None) or "")
    except Exception:  # noqa: BLE001
        base = ""
    if "intern-ai.org.cn" in base or "/intern" in base:
        return {"extra_body": {"thinking_mode": False}}
    return {}


def _extract_next_open(report_text: str) -> str:
    """从报告里提取「下一步/局限/不确定性」段(供下一轮目标拼接)."""
    pat = re.compile(r"#+\s*(?:[0-9]+[.、)]?\s*)*(下一步|局限|后续工作|未来工作|不确定性)"
                     r"[^\n]*\n(.*?)(?=\n[#]{1,3}\s|\Z)", flags=re.DOTALL)
    m = pat.search(report_text or "")
    if m:
        return m.group(2).strip()[:2000]
    return (report_text or "").strip()[-900:]


def _author_code(client, model: str, goal: str, guards: dict, cycle: int):
    """书生在 Code Lab 亲手写本轮实验代码 → 沙箱试跑校验.

    返回 (Experiment|None, probe_specs, objectives_keys, err).
    失败即回退(不伪造): (None, [], [], 原因).
    """
    from huginn.research import Experiment
    from huginn.research.code_lab import author_probe_specs, extract_code, sandbox_run

    guard_block = "\n".join(f"- {g}" for g in (guards.get("prompt_guards") or [])[:8])
    base_prompt = (
        "你是实验代码作者。用一段纯 numpy 的短函数 run(cfg) 做真实数值实验, 推进下面的研究目标。\n"
        "硬约束: 禁止 IO/网络/读写文件; 不要 try/except、不要 class、不要嵌套函数; "
        "单行 <= 88 字符; 每个 for/if/def 后紧跟缩进 4 空格; 结尾必须有 return。\n"
        "cfg 是 dict(可能只含 seed); 读参数请写 cfg.get('x', 默认值), 其余实验参数直接写在代码里。"
        "严禁把 cfg 整体解包成多个变量。\n"
        "只实现 <=30 行核心计算。返回 {\"success\": True, \"summary\": {可证伪中间量}, "
        "\"objectives\": {\"指标名\": 数值}}; objectives 每个值须为 float, 越大越支持你要验证的结论。\n"
        "可选: 再写 1 个 probe_<name>(cfg) 返回 dict 作为诊断探针。\n"
        + (("参考冷启动守卫(软提示):\n" + guard_block + "\n") if guard_block else "")
        + "模板:\n" + _AUTHOR_TEMPLATE +
        "\n只输出 <code>...</code> 内的**完整可用代码**, 不要任何多余文字。\n\n研究目标:\n"
        + goal[:1500]
    )
    cfg = {"seed": 0}
    extra = tuple(guards.get("imports_whitelist_extra") or ())
    aliases = guards.get("cfg_aliases") or None

    def _ask(extra_ctx: str = "") -> str:
        r = client.chat.completions.create(
            model=model, max_tokens=6000, temperature=0.2,
            **_llm_compat_kwargs(client),
            messages=[{"role": "user", "content": base_prompt + extra_ctx}])
        return extract_code(r.choices[0].message.content or "")

    try:
        code = _ask()
    except Exception as e:  # noqa: BLE001
        return None, [], [], f"书生写码调用失败: {e}"
    if not code.strip():
        return None, [], [], "书生未输出可解析的 <code> 块"
    res, reason = sandbox_run(code, dict(cfg), imports_whitelist_extra=extra,
                              cfg_aliases=aliases)
    budget = max(0, int(guards.get("code_retry_budget") or 0))
    for _ in range(budget):
        if reason is None:
            break
        try:
            fixed = _ask(extra_ctx=(f"\n\n你上次的代码没通过校验, 错误:\n{reason}\n"
                                    f"请只修正错误, 重输出 <code>...</code> 完整代码, 保持短小。"))
        except Exception:  # noqa: BLE001
            break
        if not fixed.strip() or fixed == code:
            break
        code = fixed
        res, reason = sandbox_run(code, dict(cfg), imports_whitelist_extra=extra,
                                  cfg_aliases=aliases)
    if reason or res is None:
        return None, [], [], f"CodeLab 校验失败: {reason}"
    name = f"author_c{cycle}"
    exp = Experiment(
        name=name,
        hypothesis=(f"书生亲手编写的实验代码(Code Lab): 针对目标 '{goal[:60]}' "
                    f"自研数值实验, 沙箱真实执行, 数值进 trace 过门禁."),
        run=lambda cc=code, cg=dict(cfg), ex=extra, al=aliases:
            sandbox_run(cc, cg, imports_whitelist_extra=ex, cfg_aliases=al)[0]
            or {"success": False, "summary": {"error": "author run failed"}, "objectives": {}},
    )
    probes = author_probe_specs(code)
    print(f"  [书生成码] 分支 {name} 校验通过, 探针 ×{len(probes)}, "
          f"objectives_keys={sorted(res['objectives'])}")
    return exp, probes, sorted(res["objectives"]), ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--objective", required=True, help="研究目标(命题无关, 由外部传入)")
    ap.add_argument("--domain", default="", help="冷启动守卫域(如 rigidity; 留空=无域守卫)")
    ap.add_argument("--cycles", type=int, default=2, help="自主轮数(每轮=写码→实验→报告)")
    ap.add_argument("--max-iters", type=int, default=8)
    ap.add_argument("--min-iters", type=int, default=2)
    ap.add_argument("--strictness", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--model", default=os.environ.get("INTERNLM_MODEL", "intern-s2-preview"))
    ap.add_argument("--base-url", default=os.environ.get(
        "INTERNLM_BASE_URL", "https://chat.intern-ai.org.cn/api/v1"))
    ap.add_argument("--session", default="run", help="产物文件名后缀")
    args = ap.parse_args()

    from huginn.research import grounding_verifier, run_research_program
    from huginn.research.coldstart_guards import compile_domain_guards, verify_domain_ready
    from huginn.research.planning import SubResearch, build_research_plan

    guards = compile_domain_guards(args.domain) if args.domain else {}
    if guards:
        ready = verify_domain_ready(guards)
        if not ready["ready"]:
            print(f"error: 冷启动守卫依赖缺失({args.domain}): {ready['missing_deps']}",
                  file=sys.stderr)
            return 3
        print(f"== 冷启动守卫({args.domain}) ==\n  依赖: {guards['deps_check']}  "
              f"写码重试预算: {guards['code_retry_budget']}")

    key = os.environ.get("INTERNLM_API_KEY")
    if not key:
        print("error: 未设置 INTERNLM_API_KEY", file=sys.stderr)
        return 2
    from openai import OpenAI
    client = OpenAI(api_key=key, base_url=args.base_url)

    _OUT.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.session}" if args.session else ""
    transcript = [f"# Huginn 原生 Code Lab 自主循环 · {args.session}", "",
                  f"> 目标(外部传入, 命题无关): {args.objective}", "",
                  f"> 模型: 书生 `{args.model}` @ `{args.base_url}`; 域: {args.domain or '(无)'}", ""]
    last_report = ""
    for cycle in range(1, args.cycles + 1):
        t0 = time.time()
        print(f"\n===== 第 {cycle} 轮(自主) =====")
        goal = args.objective
        if cycle > 1 and last_report:
            nxt = _extract_next_open(last_report)
            if nxt:
                goal = (f"{args.objective}\n\n【上一轮报告的开放问题/局限(优先推进)】\n{nxt[:1200]}")
        exp, probes, obj_keys, err = _author_code(client, args.model, goal, guards, cycle)
        transcript += [f"## 第 {cycle} 轮 · 书生成码", ""]
        if exp is None:
            note = f"书生成码未通过, 本轮无真实证据, 停止(不伪造): {err}"
            print(f"  !! {note}", file=sys.stderr)
            transcript += [f"**{note}**", ""]
            break
        plan = build_research_plan(goal, [
            SubResearch(exp.name, exp.hypothesis, run=exp.run, depends_on=[])])
        report_md = _OUT / f"report_cycle{cycle}{sfx}.md"
        objectives = {k: "maximize" for k in obj_keys}
        print(f"  goal: {goal[:100]}...")
        out = run_research_program(
            goal=goal, experiments=plan.experiments, objectives_config=objectives,
            client=client, model=args.model, base_url=args.base_url,
            verify=grounding_verifier(), out_md=report_md,
            planner=lambda _g, _p=plan: _p, layer_epochs=True, replan_gate=True,
            early_stop_gate=True, diagnostic_tools=probes or None,
            max_iterations=args.max_iters,
            min_iterations=min(args.min_iters, max(1, len(plan.experiments))),
            max_parallel=2, strictness=args.strictness)
        print(f"  [program] explored={out.explored} pruned={out.pruned} "
              f"pareto={len(out.pareto_front)} convergence={out.converred}")
        print(f"  [gate] {out.verdict} unsubstantiated={out.ungrounded} source={out.report_source}")
        last_report = out.report or ""
        transcript += [
            f"**objectives_keys**: {obj_keys}", "",
            f"**program**: explored={out.explored} pareto={len(out.pareto_front)} "
            f"verdict={out.verdict}", "",
            f"**报告**: `{report_md.name}`  ({time.time() - t0:.0f}s)", "",
            "```", (out.report or "(空)")[:4000], "```", ""]
        (_OUT / f"transcript{sfx}.md").write_text("\n".join(transcript), encoding="utf-8")

    print(f"\n产物: {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())