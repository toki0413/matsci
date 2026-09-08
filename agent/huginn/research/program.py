"""Huginn Research — 域无关的自主深研统一管线(产品能力).

这不是参赛演示, 而是 Huginn 产品里的一条通用能力:
给定一个研究目标, 自动完成 假说生成 → 真实实验执行 → Pareto 演化/淘汰 → 
批判综合 → 声明门禁 → 兜底组装 的完整深研闭环.

设计(对齐 2026 Co-Scientist / GPT-6 Astra / Fable 的 product-grade 做法):
  - 域无关: 只依赖 ``Experiment`` 抽象与一个可插拔执行函数, 不碰具体物理.
  - 复用真实 ``ExplorationOrchestrator``(min_iterations 防早停 + max_iterations 预算).
  - 批判/演化: Pareto 剪枝维持前沿 = "debate 淘汰"的存活假说.
  - 综合: supervisor 可选(传 client 则真机成文; 否则确定性组装), 再走 claim_grounding 门禁.
  - 兜底: 真机交不出可落地报告 → 确定性组装(数值全来自真实执行, 必过门禁).
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class Experiment:
    """一个可执行的研究假说/实验.

    name: 唯一标识; hypothesis: 人类可读假说; run: 执行真实实验,
        返回 dict(含 summary 与 objectives[magnet maximize]).
    parametrize: 可选 —— 给定 MutationStrategy 的变异参数(如 {"a_scale": 1.05}),
        返回一个执行**真实变异实验**的 run 闭包。缺省 None 时, 变异子代以父实验
        的 run 真实重跑(诚实回退, 不再让子代"被创建却无法执行")。
    """
    name: str
    hypothesis: str
    run: Callable[[], dict[str, Any]]
    parametrize: Callable[[dict[str, float]], Callable[[], dict[str, Any]]] | None = None


@dataclass
class ResearchOutcome:
    converred: str = ""
    explored: int = 0
    pruned: int = 0
    pareto_front: list[dict] = field(default_factory=list)   # 存活假说 [{name, hypothesis, objectives}]
    cache: dict[str, dict] = field(default_factory=dict)     # name -> {summary, objectives}
    report: str = ""
    verdict: str = "needs_grounding"
    ungrounded: list = field(default_factory=list)
    report_source: str = "deterministic"
    mutations: int = 0                                 # 进化变异生成的子代数
    supervision_log: list = field(default_factory=list)  # HITL 人工反馈记录
    plan_summary: dict | None = None                   # 需求拆解/自主规划摘要(planner) — 若提供
    structural_gate: dict | None = None                # 结构闸门(交互等效/多元论)审计结果
    structural_aligned: bool | None = None             # 代理是否通过结构对齐(Shortcut 探测)
    workspace_verified: bool | None = None             # C-Space 工作区门: 报告断言是否作为在场落地


def _build_trace(cache: dict[str, dict]) -> list[str]:
    return [json.dumps(v, ensure_ascii=False) for v in cache.values()]


def _slug_goal(goal: str, limit: int = 40) -> str:
    """把 goal 规整成工作区 Being id 用的小写 slug (供报告在场命名)."""
    s = re.sub(r"[^0-9a-z_]+", "_", (goal or "final").lower()).strip("_")
    return s[:limit] or "final"


def grounding_verifier() -> Callable[[str, list[str]], dict]:
    """声明门禁唯一实现 (claim_grounding). 所有 demo/管线统一从这里取, 不再各自 bootstrap."""
    def _v(text: str, trace: list[str]) -> dict:
        try:
            from huginn.validation.claim_grounding import verify_claims
            return verify_claims(text, trace, allow_derived=True)
        except Exception:  # noqa: BLE001 — claim_grounding 不可用时拉文件级兜底
            src = Path(__file__).resolve().parents[1] / "validation/claim_grounding.py"
            spec = util.spec_from_file_location("_cg", str(src))
            mod = util.module_from_spec(spec); spec.loader.exec_module(mod)
            return mod.verify_claims(text, trace, allow_derived=True)
    return _v


def run_research_program(
    goal: str,
    experiments: list[Experiment],
    objectives_config: dict[str, str],
    *,
    max_iterations: int = 40,
    min_iterations: int = 6,
    max_parallel: int = 2,
    client: Any = None,            # OpenAI 兼容 client; None → 确定性综合
    model: str = "intern-s2-preview",
    base_url: str | None = None,
    verify: Callable[[str, list[str]], dict] | None = None,
    out_md: Path | None = None,
    mutation_config: dict | None = None,   # {"param_space":{p:(lo,hi)}, "mutation_rate":0.5, "max_children":3}
    human_review: Callable[[dict], dict] | None = None,  # 人工/专家评审回调(card)->{drop,keep,guidance}
    supervisor_every: int = 0,             # 每 N 轮一次 HITL 评审; 0=关闭
    debate: bool = False,                  # 用 client 对存活想法做 LLM tournament 筛选
    diagnostic_tools: list[dict] | None = None,  # 域诊断工具能力 [{"tool":schema,"handle":fn}], LLM 可自主发现并调用
    self_audit: Callable[[], list[str]] | None = None,  # 能力自省 §3: 返回需并入 trace 的可证伪工件(能力缺口提案)
    structural_audit: Callable[[list[dict], str], dict] | None = None,  # 结构闸门: (survivors, goal)->{"pass",...} 交互等效审计(张拳石/多元论治理)
    workspace: Any = None,  # C-Space 工作区: 若提供, 最终报告须经 workspace.broadcast 作为"在场断言"落地才成文
    planner: Callable[[str], "ResearchPlan"] | None = None,  # 需求拆解/自主规划: goal->{experiments, max_parallel, plan_summary}
) -> ResearchOutcome:
    """跑一条完整深研管线并返回结果."""
    from huginn.exploration.orchestrator import ExplorationOrchestrator
    import huginn.exploration.strategies as S
    from huginn.exploration.strategies import MutationStrategy
    from huginn.exploration.supervisor import SupervisorStrategy

    # 需求拆解/自主规划: planner 把高层 goal 解成子任务 DAG, 调整实验序列与并行度
    plan_summary: dict | None = None
    if planner is not None:
        plan = planner(goal)
        if plan is not None:
            experiments = list(plan.experiments or experiments)
            max_parallel = int(plan.max_parallel or max_parallel)
            plan_summary = plan.to_dict()

    cache: dict[str, dict] = {}
    spec_by_name = {e.name: e for e in experiments}

    def _resolve_variant(branch_name: str, branch) -> Experiment | None:
        """把动态子代(变异/演化)解析成可真实执行的实验 spec.

        迭代闭环的关键: MutationStrategy 生成的 `{parent}~mut{i}` 子代不在初始
        `experiments` 里, 以前 executor 直接返回 failure → 子代被创建却永不产生
        真实 objectives(「评估→变异→再执行」断裂)。这里按 branch.metadata 里的
        世系与参数重建变体:

          - 父实验声明了 parametrize → 用变异参数生成真实变异实验(参数真正生效);
          - 未声明 → 复用父实验 run 真实重跑(诚实回退, 不伪造数值);
          - 参数化失败 → 丢弃该子代(不产生假数据), 由 orchestrator 正常失败兜底。
        """
        meta = branch.metadata or {}
        parent = meta.get("mutation_of")
        if parent is None:
            return None  # 非变异子代(如 refine 缺参), 保持原行为
        spec = spec_by_name.get(parent)
        if spec is None:
            return None
        params = meta.get("params") or {}
        if spec.parametrize is not None:
            try:
                variant_run = spec.parametrize(params)
                if variant_run is None:
                    raise TypeError("parametrize 返回 None")
            except Exception:  # noqa: BLE001 — 参数化失败: 不伪造, 子代丢弃
                return None
        else:
            variant_run = spec.run  # 诚实回退: 父实验真实重跑
        return dataclasses.replace(
            spec,
            run=variant_run,
            name=branch_name,
            hypothesis=branch.hypothesis or spec.hypothesis,
        )

    async def executor(branch):
        spec = spec_by_name.get(branch.name)
        if spec is None:  # 动态子代(变异/演化新分支)
            spec = _resolve_variant(branch.name, branch)
            if spec is not None:
                spec_by_name[branch.name] = spec  # 注册, 供综合阶段取假说/摘要
        if spec is None:
            return {"success": False, "objectives": {}, "results": {}}
        res = await asyncio.to_thread(spec.run)
        cache[branch.name] = res
        return {"success": bool(res.get("success", True)),
                "objectives": res.get("objectives", {}),
                "results": res.get("summary", {})}

    def _llm_tournament(ideas: list[dict]) -> list[dict]:
        """用 client 做 LLM 想法 tournament: 返回存活想法 (解析失败则全存活, 保证稳健)."""
        bullets = "\n".join(
            f"- {it['name']}: hypothesis={it['hypothesis']} obj={json.dumps(it['objectives'], ensure_ascii=False)}"
            for it in ideas)
        prompt = (
            f"你是科研创意评审委员。目标: {goal}。\n候选存活想法:\n{bullets}\n\n"
            "请交叉评审, 淘汰缺乏证据、重复或低可行性的想法, 返回存活想法的 name 列表, "
            '只输出 JSON: {"survivors": ["name", ...]}。'
        )
        try:
            r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=500, temperature=0.2)
            text = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            survivors = set(json.loads(m.group(0))["survivors"])
            return [it for it in ideas if it["name"] in survivors] or ideas
        except Exception:  # noqa: BLE001 — 评审失败不退化为空, 全存活
            return ideas

    # 策略链: Pareto → (可选)变异 → (可选)HITL/辩论评审
    strategy: S.ExplorationStrategy = S.ParetoPruningStrategy(max_active=max_parallel + 6)
    if mutation_config:
        strategy = MutationStrategy(
            wrapped=strategy,
            param_space=mutation_config.get("param_space"),
            mutation_rate=mutation_config.get("mutation_rate", 0.5),
            max_children=mutation_config.get("max_children", 3),
        )
    if human_review or debate:
        debate_eval = _llm_tournament if (debate and client is not None) else None
        strategy = SupervisorStrategy(
            wrapped=strategy,
            human_review=human_review,
            review_every=supervisor_every,
            debate_evaluator=debate_eval,
        )

    orch = ExplorationOrchestrator(
        strategy=strategy,
        branch_executor=executor,
        max_parallel=max_parallel,
    )

    # 声明门禁: 默认从产品内单一实现取
    verify = verify or grounding_verifier()

    result = asyncio.run(orch.explore(
        objective=goal,
        initial_branches=[{"name": e.name, "hypothesis": e.hypothesis} for e in experiments],
        objectives_config=objectives_config,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
    ))

    trace = _build_trace(cache)
    # 能力自省 §3: 把能力缺口提案的可证伪工件并入 trace, 使报告引用可被 grounding 门禁核实
    if self_audit is not None:
        try:
            trace += list(self_audit())
        except Exception:  # noqa: BLE001 — 自省失败不阻断主流程
            pass
    front = result.pareto_front or []
    out = ResearchOutcome(converred=result.convergence_reason,
                          explored=result.n_branches_explored,
                          pruned=result.n_branches_pruned,
                          pareto_front=front, cache=cache,
                          plan_summary=plan_summary)
    # 进化/监督衍生量: 变异子代数 + HITL 反馈记录(若启用了对应策略)
    mstrat = getattr(strategy, "wrapped", None) if isinstance(strategy, SupervisorStrategy) else None
    if isinstance(strategy, MutationStrategy):
        out.mutations = strategy.created
    elif isinstance(mstrat, MutationStrategy):
        out.mutations = mstrat.created
    if isinstance(strategy, SupervisorStrategy):
        out.supervision_log = list(strategy.review_log)
    # 结构闸门(交互等效/多元论治理): 代理/结论进报告前, 自动过一遍结构对齐审计.
    audit: dict | None = None
    if structural_audit is not None:
        try:
            audit = structural_audit(list(front), goal)
        except Exception as exc:  # noqa: BLE001 — 结构审计异常不阻断主流程, 如实记为未通过
            audit = {"pass": False, "reason": f"structural_audit failed: {exc}",
                     "surrogate_only": [], "law_only": []}
        if isinstance(audit, dict):
            audit.setdefault("pass", False)
            audit.setdefault("surrogate_only", [])
            audit.setdefault("law_only", [])
            # 可证伪工件并入 trace —— 报告里对结构风险的引用即可被 grounding 门禁核实.
            trace.append(json.dumps({"type": "structural_gate", **audit}, ensure_ascii=False))
            out.structural_gate = audit
            out.structural_aligned = bool(audit["pass"])

    survivors = [(b["name"], cache.get(b["name"], {})) for b in front]

    def _synthesize() -> str:
        L = [f"# 自主深研 — {goal}", "",
             "> 存活假说(Pareto 前沿, debate 淘汰后): " + ", ".join(n for n, _ in survivors) + "", ""]
        for name, res in survivors:
            hyp = spec_by_name.get(name).hypothesis if spec_by_name.get(name) else ""
            L.append(f"## {name}")
            L.append(f"- 假说: {hyp}")
            if res.get("summary"):
                L.append(f"- 真实结果: {json.dumps(res['summary'], ensure_ascii=False)}")
            L.append("")
        L.append("## 结论(开放)")
        L.append("存活假说覆盖目标下的多证据方向, 数值均来自真实执行、可复现。")
        # 结构闸门: 代理结论进决策前的交互等效审计(通过/未通过如实写明)
        if audit is not None:
            if out.structural_aligned is not False:
                L.append("## 结构对齐闸门")
                L.append("代理经交互等效审计与定律结构一致(无虚假交互)。")
            else:
                L.append("## 结构对齐闸门(未通过)")
                L.append(f"- 原因: {audit.get('reason', '')}")
                L.append(f"- 代理独有交互(surrogate_only): {audit.get('surrogate_only')}")
                L.append(f"- 定律独有交互(law_only): {audit.get('law_only')}")
                L.append("**代理结论存在与定律不符的交互结构(疑似 shortcut/混淆), "
                         "不应直接进入决策。**")
        return "\n".join(L)

    final, verdict, ungrounded = "", "needs_grounding", []
    if client is not None:
        survivors_text = "\n".join(
            f"- {name}: {json.dumps(res.get('summary', {}), ensure_ascii=False)}" for name, res in survivors)
        prompt = (f"你是 Huginn 科研智能体。目标: {goal}。\n"
                  f"以下是存活假说的真实数值证据(可复现、非伪造):\n{survivors_text}\n\n"
                  f"请撰写跨学科深度研究报告(研究问题/数据与方法/结果分析/对账与局限/下一步)。"
                  f"每个数值必须来自上面真实结果, 不许编造。把最终报告放在 <report> 与 </report> 之间。")
        # 结构闸门提示: 让 LLM 成文时如实反映交互等效审计结果, 不隐瞒 shortcut 风险.
        if audit is not None:
            if out.structural_aligned is not False:
                prompt += "\n【结构对齐闸门】代理经交互等效审计与定律一致(无虚假交互)。"
            else:
                prompt += (f"\n【结构对齐闸门】代理未通过: {audit.get('reason', '')}; "
                           f"surrogate_only={audit.get('surrogate_only')}。报告中须如实说明此结构风险, 不许隐瞒。")
        # 统一诊断工具挂载面 —— 走单一薄控制点 (resolve_diagnostic_tools), 不内联再造 schema
        from huginn.research.tool_surface import resolve_diagnostic_tools
        tool_schemas, tool_handlers = resolve_diagnostic_tools(diagnostic_tools)
        if tool_schemas:
            prompt += ("\n可自主调用的域诊断工具(结果作为可证伪证据进报告): "
                       + ", ".join(t["function"]["name"] for t in tool_schemas) + "。")

        def _one_attempt(msgs):
            for _round in range(4):
                r = client.chat.completions.create(model=model, messages=msgs,
                                                   tools=tool_schemas or None,
                                                   tool_choice="auto" if tool_schemas else None,
                                                   max_tokens=4000, temperature=0.2)
                msg = r.choices[0].message
                calls = msg.tool_calls or []
                if not calls:
                    return msg.content or ""
                for tc in calls[:4]:
                    name = tc.function.name
                    a = {}
                    try:
                        a = json.loads(tc.function.arguments) if isinstance(
                            tc.function.arguments, str) else (tc.function.arguments or {})
                    except Exception:  # noqa: BLE001 — 参数容错
                        a = {}
                    h = tool_handlers.get(name)
                    if h is None:
                        res = json.dumps({"error": f"unknown tool {name}"}, ensure_ascii=False)
                    else:
                        try:
                            res = h(a)
                        except Exception as ee:  # noqa: BLE001 — 工具调用异常如实入 trace 供门禁对照
                            res = json.dumps({"error": str(ee)}, ensure_ascii=False)
                    trace.append(res)  # 门禁证据: 诊断工具真实 return 落 trace
                    msgs += [{"role": "assistant", "content": msg.content or "",
                              "tool_calls": [tc.model_dump()]},
                             {"role": "tool", "tool_call_id": tc.id, "content": res}]
            return msg.content or ""

        msgs = [{"role": "user", "content": prompt}]
        for _ in range(3):
            final = _one_attempt(msgs).strip()
            m = re.search(r"<report>(.*?)</report>", final, flags=re.DOTALL | re.IGNORECASE)
            if m:
                final = m.group(1).strip()
            g = verify(final, trace)
            if g["verdict"] == "pass" and len(final.strip()) > 200:
                verdict, ungrounded = "pass", []; break
            ungrounded = g["unsubstantiated"]
            msgs = [{"role": "user",
                     "content": f"未交付(未落地:{ungrounded})。请只用真实证据重写:\n" + prompt}]
        else:
            final = _synthesize()      # L3b 兜底组装, 必过门禁
            out.report_source = "fallback_assembly(agent failed)"
    else:
        final = _synthesize()
        out.report_source = "deterministic"

    g2 = verify(final, trace)
    if g2["verdict"] == "pass":
        verdict, ungrounded = "pass", []
    else:
        ungrounded = g2["unsubstantiated"]

    # C-Space 工作区门: 若注入工作区, 先以存活结论的真实执行证据"灌出在场 + trace"，
    # 再把最终报告作为一条 broadcast 经其声明门禁落地才成文. 未过 → 如实醒目标注.
    if workspace is not None:
        for name, res in survivors:
            workspace.register(name, "state", payload=dict(res.get("summary", {})),
                               source=f"survivor:{name}", falsifiable=True)
            if res.get("summary"):
                workspace.trace.append(json.dumps(res["summary"], ensure_ascii=False))
        ws_res = workspace.broadcast(final)
        out.workspace_verified = bool(ws_res.get("verified"))
        if out.workspace_verified:
            workspace.register("report_" + (_slug_goal(goal) or "final"), "concept",
                               payload={"source": "run_research_program.final"}, source="report",
                               falsifiable=True)
        else:
            _miss = ws_res.get("unsubstantiated", [])
            final += ("\n\n## 工作区门禁(未确认在场)\n报告断言未作为可证伪在场落地; "
                      "未落地主张: " + str(_miss) + "\n")

    out.report = final
    out.verdict, out.ungrounded = verdict, ungrounded

    if out_md is not None:
        header = (f"# 自主深研(Huginn×书生)\n\n> **门禁: {verdict}** (未落地: {ungrounded or '无'})\n"
                  f"> real orchestration: explored={out.explored} pruned={out.pruned} "
                  f"convergence={out.converred}\n> 报告来源: {out.report_source}"
                  + (f"\n> 结构对齐闸门: {out.structural_aligned}" if audit is not None else "")
                  + (f"\n> 工作区门: {out.workspace_verified}" if workspace is not None else "")
                  + (f"\n> 进化变异: {out.mutations} 子代; HITL 反馈: {len(out.supervision_log)} 轮"
                     if (out.mutations or out.supervision_log) else "")
                  + "\n\n")
        out_md.write_text(header + final.strip() + "\n", encoding="utf-8")
    return out