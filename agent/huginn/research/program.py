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
import contextlib
import dataclasses
import functools
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huginn.research._scalars import first_scalar as _first_scalar

if TYPE_CHECKING:
    from huginn.research.planning import ResearchPlan


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
    parametrize: Callable[[dict[str, float]], Callable[[], dict[str, Any]]] | None = (
        None
    )


@dataclass
class ResearchOutcome:
    converred: str = ""
    explored: int = 0
    pruned: int = 0
    pareto_front: list[dict] = field(
        default_factory=list
    )  # 存活假说 [{name, hypothesis, objectives}]
    cache: dict[str, dict] = field(
        default_factory=dict
    )  # name -> {summary, objectives}
    report: str = ""
    verdict: str = "needs_grounding"
    ungrounded: list = field(default_factory=list)
    report_source: str = "deterministic"
    mutations: int = 0  # 进化变异生成的子代数
    supervision_log: list = field(default_factory=list)  # HITL 人工反馈记录
    plan_summary: dict | None = (
        None  # (P2 遗存) 需求拆解/自主规划摘要(planner) — 聚合头 head 源
    )
    structural_gate: dict | None = None  # (P2 遗存) 结构闸门(交互等效/多元论)审计结果
    structural_aligned: bool | None = (
        None  # (P2 遗存) 代理是否通过结构对齐(Shortcut 探测)
    )
    workspace_verified: bool | None = (
        None  # (P2 遗存) C-Space 工作区门: 报告断言是否作为在场落地
    )
    harness: dict | None = None  # Self-Harness 六维报告(dict) — demo 一键出报告
    law_model_used: dict | None = (
        None  # (P2 遗存) 世界模型真用证据: predict 产物/是否参与决策 (真深思 D)
    )
    plan_revision: dict | None = (
        None  # (P2 遗存) 漏A: plan 修订门(新证据→显式复盘初始计划)审计
    )
    grounding_audit: dict | None = (
        None  # (P2 遗存) 漏C: "得分≠使用"(高分存活项是否真进最终报告)审计
    )
    # 注意: 判断层·分级护栏的注入软提示不再挂成 god-object 字段, 而是经聚合头注册
    # 为 `guardrail.judgment` head (见下) —— 遵守 P3 "新视角只许经聚合头"红线.
    consolidated: dict | None = (
        None  # 收敛聚合头(P1/P2): 现有治理头统一注册后的收敛视图.
    )
    # P3 起为唯一治理出口 —— 新视角只许在此注册, 不再加 out.* 字段.


def _build_trace(cache: dict[str, dict]) -> list[str]:
    return [json.dumps(v, ensure_ascii=False) for v in cache.values()]


def _slug_goal(goal: str, limit: int = 40) -> str:
    """把 goal 规整成工作区 Being id 用的小写 slug (供报告在场命名)."""
    s = re.sub(r"[^0-9a-z_]+", "_", (goal or "final").lower()).strip("_")
    return s[:limit] or "final"


# 世界模型对账容差: 相对误差 <= 此值视为预测被真实执行"证实"(默认 3%, 同 law_model.reconcile)
_WM_TOL = 0.03

# 缺陷七 · 聚合头头数预算: 超过即"建制膨胀"亮灯(second system effect 反制).
# 当前既有 11 头 + meta.overbuild_guard 自身 = 12; 预算留 2 个余量,
# 新增审计视角先评审其增益, 再考虑提预算 —— 与依赖白名单同哲学.
_HEAD_BUDGET = 14

# 缺陷一(P-A): 流式层摘要单条上限(字符). 决策上下文 = 各层增量摘要之和, 有界;
# 与漏B 先导摘要同语义 —— 只压缩表达冗余, 全量证据始终留在 cache/trace.
_STREAM_SUMMARY_CHARS = 1200

# 缺陷二(P-B): 层间重规划门 —— 假说 Jaccard 重叠阈值(>= 视为"该方向已被前序层
# 覆盖采样", 跳过=预算再分配, 不伪造). 与 distill_tool_output 同关键词切法.
_REPLAN_SIM = 0.55

# 缺陷一(P-C): 证据驱动提前终止门 —— 防早停: 至少 N 个有分观测层才允许查稳定;
# 稳定判据: 最后两层 top-1 得分相对变化 <= margin(分数高原) → 提前终止剩余层.
_EARLY_STOP_MIN_LAYERS = 2
_EARLY_STOP_MARGIN = 0.02


def _wm_predict(world_model: Any, spec: Experiment) -> dict:
    """统一世界模型预筛入口: 接受 LawModel 或普通 callable, 产出**可判伪**预测.

    - LawModel 形态: model.predict(LawState, LawAction) -> next state. 这里用 spec 的
      假说/参数构造一个可量的 saliency 向量喂入, 取返回 state 的 as_dict(); 失败降级为 {}.
    - callable 形态: 兼容已符合 ``predict(spec)->{predicted:...}`` 契约的用户自定义模型.
    predict 的产物只作为**假说**(预判), 绝不替代真实执行(execute 才是真相) —— 与
    LawModel "predict 只预告" 的诚实边界一致, 也是"训练/能力存在 ≠ 部署时在规划"的
    判据落点: 提供了 world_model 且本 run 真实调用, 才算真深思 D.
    """
    if hasattr(world_model, "predict"):
        try:
            # 尝试 LawModel 形态: predict(state, action) -> next state
            if hasattr(world_model, "law"):
                from huginn.research.law_model import LawAction, LawState

                action = LawAction(config={"a_scale": 1.0}, label=spec.name)
                state = (
                    world_model.seed({"orbper_d": 365.25})
                    if hasattr(world_model, "seed")
                    else LawState({}, domain="")
                )
                nxt = world_model.predict(state, action)
                d = getattr(
                    nxt, "as_dict", lambda: dict(getattr(nxt, "__dict__", {}))
                )()
                return {
                    "pred": d if isinstance(d, dict) else {"predicted": d},
                    "law": getattr(world_model, "law", lambda: "")(),
                }
            # 否则当作 predict(spec)->{predicted:...}
            out = world_model.predict(spec)
            return out if isinstance(out, dict) else {"predicted": out}
        except Exception:  # noqa: BLE001 — 形态推断失败: 返回空预测, 由 caller 保留纯真实结果
            return {"pred": {}}
    # 普通 callable: predict(spec)->{predicted:...}
    out = world_model(spec)
    return out if isinstance(out, dict) else {"predicted": out}


def grounding_verifier() -> Callable[[str, list[str]], dict]:
    """声明门禁唯一实现 (claim_grounding). 所有 demo/管线统一从这里取, 不再各自 bootstrap."""

    def _v(text: str, trace: list[str]) -> dict:
        try:
            from huginn.validation.claim_grounding import verify_claims

            return verify_claims(text, trace, allow_derived=True)
        except Exception:  # noqa: BLE001 — claim_grounding 不可用时拉文件级兜底
            import importlib.util as util  # noqa: F401 — 文件级加载绕开 package 深层 import

            src = Path(__file__).resolve().parents[1] / "validation/claim_grounding.py"
            spec = util.spec_from_file_location("_cg", str(src))
            mod = util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.verify_claims(text, trace, allow_derived=True)

    return _v


@dataclass
class _RPState:
    """run_research_program 的阶段间上下文(显式承载可变状态, 不靠跨函数闭包)."""

    goal: str
    experiments: list[Experiment]
    objectives_config: dict[str, str]
    max_parallel: int
    plan_summary: dict | None
    spec_by_name: dict[str, Experiment]
    layers_map: list[list[str]]
    layer_epochs: bool
    replan_enabled: bool
    replan_similarity: float
    hypotheses: dict[str, str]
    early_stop_enabled: bool
    early_stop_meta: dict
    early_stop_min_layers: int
    early_stop_margin: float
    prior_args: dict
    stream_summary_chars: int
    world_model: Any = None
    cache: dict[str, dict] = field(default_factory=dict)
    reuse_plan: dict[str, str] = field(default_factory=dict)
    runs_by_base: dict[str, list[dict]] = field(default_factory=dict)
    stream_rows: list[dict] = field(default_factory=list)
    replan_log: list[dict] = field(default_factory=list)
    replan_checked: int = 0
    early_stop_active: bool = False
    judgment_hints: list[str] = field(default_factory=list)
    stream_view: list[dict] | None = None
    n_epochs: int = 0


def run_research_program(
    goal: str,
    experiments: list[Experiment],
    objectives_config: dict[str, str],
    *,
    max_iterations: int = 40,
    min_iterations: int = 6,
    max_parallel: int = 2,
    client: Any = None,
    model: str = "intern-s2-preview",
    base_url: str | None = None,
    verify: Callable[[str, list[str]], dict] | None = None,
    out_md: Path | None = None,
    mutation_config: dict | None = None,
    human_review: Callable[[dict], dict] | None = None,
    supervisor_every: int = 0,
    debate: bool = False,
    diagnostic_tools: list[dict] | None = None,
    self_audit: Callable[[], list[str]] | None = None,
    structural_audit: Callable[[list[dict], str], dict] | None = None,
    workspace: Any = None,
    planner: Callable[[str], ResearchPlan] | None = None,
    harness_agent: str = "",
    harness_machine: str = "",
    world_model: Any = None,
    external_verifier: Callable[[dict], dict] | None = None,
    layer_epochs: bool = False,
    replan_gate: bool = False,
    early_stop_gate: bool = False,
    stream_summary_chars: int = _STREAM_SUMMARY_CHARS,
    replan_similarity: float = _REPLAN_SIM,
    early_stop_min_layers: int = _EARLY_STOP_MIN_LAYERS,
    early_stop_margin: float = _EARLY_STOP_MARGIN,
    prior: dict | None = None,
    strictness: int = 0,
) -> ResearchOutcome:
    """跑一条完整深研管线并返回结果."""
    return _run_research_pipeline(locals())


def _run_research_pipeline(kw: dict[str, Any]) -> ResearchOutcome:
    """管线主体: 初始化 → 探索/收敛 → 成文 → 落盘."""
    st = _rp_init_state(kw)
    verify, strategy = _rp_prepare(st, kw)
    result = _rp_explore(st, strategy, kw["max_iterations"], kw["min_iterations"])
    front, trace, out, audit, survivors = _rp_collect(
        st, result, strategy, kw["self_audit"], kw["structural_audit"]
    )
    final, verdict, ungrounded = _rp_final_report(
        st, out, survivors, audit, trace, kw, verify
    )
    _rp_persist(st, out, final, verdict, ungrounded, front, audit, survivors, kw)
    return out


def _rp_init_state(kw: dict[str, Any]) -> _RPState:
    """按管线参数初始化状态."""
    return _rp_build_state(
        goal=kw["goal"],
        experiments=kw["experiments"],
        objectives_config=kw["objectives_config"],
        max_parallel=kw["max_parallel"],
        planner=kw["planner"],
        layer_epochs=kw["layer_epochs"],
        replan_gate=kw["replan_gate"],
        early_stop_gate=kw["early_stop_gate"],
        early_stop_min_layers=kw["early_stop_min_layers"],
        early_stop_margin=kw["early_stop_margin"],
        stream_summary_chars=kw["stream_summary_chars"],
        prior=kw["prior"],
        replan_similarity=kw["replan_similarity"],
        world_model=kw["world_model"],
    )


def _rp_prepare(st: _RPState, kw: dict[str, Any]):
    """构建校验器 / 辩论评估 / 执行策略."""
    verify = kw["verify"] or grounding_verifier()
    debate_eval = (
        functools.partial(
            _llm_tournament, goal=kw["goal"], model=kw["model"], client=kw["client"]
        )
        if (kw["debate"] and kw["client"] is not None)
        else None
    )
    strategy = _build_strategy(
        mutation_config=kw["mutation_config"],
        human_review=kw["human_review"],
        debate=kw["debate"],
        debate_eval=debate_eval,
        supervisor_every=kw["supervisor_every"],
        max_parallel=st.max_parallel,
    )
    return verify, strategy


def _rp_final_report(st, out, survivors, audit, trace, kw, verify):
    """成文阶段: 组装 prompt → LLM 出文 → 校验收敛."""
    return _rp_generate_final(
        st,
        out,
        survivors,
        audit,
        trace,
        client=kw["client"],
        model=kw["model"],
        diagnostic_tools=kw["diagnostic_tools"],
        strictness=kw["strictness"],
        verify=verify,
        workspace=kw["workspace"],
    )


def _rp_persist(
    st, out, final, verdict, ungrounded, front, audit, survivors, kw
) -> None:
    """落盘阶段: 聚合视图 + 报告写文件."""
    _rp_finish(
        st,
        out,
        final,
        verdict,
        ungrounded,
        front,
        audit,
        survivors,
        kw["workspace"],
        kw["external_verifier"],
        kw["harness_agent"],
        kw["harness_machine"],
        kw["out_md"],
    )


def _rp_build_state(
    *,
    goal: str,
    experiments: list[Experiment],
    objectives_config: dict[str, str],
    max_parallel: int,
    planner: Callable[[str], ResearchPlan] | None,
    layer_epochs: bool,
    replan_gate: bool,
    early_stop_gate: bool,
    early_stop_min_layers: int,
    early_stop_margin: float,
    stream_summary_chars: int,
    replan_similarity: float,
    prior: dict | None,
    world_model: Any,
) -> _RPState:
    """初始化/预算阶段: 规划拆解、层拓扑、门控状态与先验参数."""
    experiments, max_parallel, plan_summary, layers_map, has_layers = _rp_plan_setup(
        planner, goal, experiments, max_parallel
    )
    prior_args = _rp_prior_args(prior, early_stop_min_layers, early_stop_margin, goal)
    return _RPState(
        goal=goal,
        experiments=experiments,
        objectives_config=objectives_config,
        max_parallel=max_parallel,
        plan_summary=plan_summary,
        spec_by_name={e.name: e for e in experiments},
        layers_map=layers_map,
        layer_epochs=bool(layer_epochs) and has_layers,
        replan_enabled=bool(replan_gate) and has_layers,
        replan_similarity=replan_similarity,
        hypotheses={e.name: e.hypothesis for e in experiments},
        early_stop_enabled=bool(early_stop_gate) and has_layers,
        early_stop_meta={"enabled": bool(early_stop_gate) and has_layers},
        early_stop_min_layers=int(prior_args["min_layers"]),
        early_stop_margin=float(prior_args["margin"]),
        prior_args=prior_args,
        stream_summary_chars=stream_summary_chars,
        world_model=world_model,
    )


def _rp_plan_setup(
    planner: Callable[[str], ResearchPlan] | None,
    goal: str,
    experiments: list[Experiment],
    max_parallel: int,
) -> tuple[list[Experiment], int, dict | None, list, bool]:
    """规划拆解(planner 可选) + 层拓扑推导: 覆盖实验集/并行度, 返回 plan 摘要与层信息."""
    plan_summary: dict | None = None
    if planner is not None:
        plan = planner(goal)
        if plan is not None:
            experiments = list(plan.experiments or experiments)
            max_parallel = int(plan.max_parallel or max_parallel)
            plan_summary = plan.to_dict()
    layers_map = (plan_summary.get("layers") or []) if plan_summary is not None else []
    has_layers = plan_summary is not None and bool(plan_summary.get("layers"))
    return experiments, max_parallel, plan_summary, layers_map, has_layers


def _rp_prior_args(
    prior: dict | None,
    default_min_layers: int,
    default_margin: float,
    goal: str,
) -> dict:
    """从 prior_store 解析提前终止先验参数(按 goal 相似度取先验)."""
    from huginn.research.prior_store import resolve_early_stop_args

    return resolve_early_stop_args(
        prior,
        default_min_layers=default_min_layers,
        default_margin=default_margin,
        current_goal_slug=_slug_goal(str(goal)),
        current_goal=str(goal),
    )


def _rp_resolve_variant(
    st: _RPState, branch_name: str, branch: Any
) -> Experiment | None:
    """把动态子代(变异/演化)解析成可真实执行的实验 spec (失败则丢弃, 不伪造)."""
    meta = branch.metadata or {}
    parent = meta.get("mutation_of")
    if parent is None:
        return None  # 非变异子代(如 refine 缺参), 保持原行为
    spec = st.spec_by_name.get(parent)
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
        st.reuse_plan[branch_name] = parent  # §10: 同体重复执行指纹 = 父实验
    return dataclasses.replace(
        spec,
        run=variant_run,
        name=branch_name,
        hypothesis=branch.hypothesis or spec.hypothesis,
    )


def _rp_layer_of(st: _RPState, name: str) -> int:
    for i, lay in enumerate(st.layers_map):
        if name in lay:
            return i
    return -1


def _rp_replan_decision(st: _RPState, name: str) -> dict | None:
    """单个后序实验的层间重规划判定 (纯函数托管在 replan_gate.decide_replan_skip)."""
    from huginn.research.replan_gate import decide_replan_skip

    i = _rp_layer_of(st, name)
    if i <= 0:
        return None
    prior = [n for j in range(i) for n in st.layers_map[j]]
    return decide_replan_skip(
        name,
        i,
        prior,
        st.cache,
        st.hypotheses,
        already_decided=set(st.cache) | {r["name"] for r in st.replan_log},
        similarity=st.replan_similarity,
        tol=_WM_TOL,
    )


def _rp_settle_layer_check(st: _RPState, settled_until: int) -> None:
    """层 settled_until 刚结算: 用稳定度判据决定是否激活提前终止(预算决策)."""
    if not st.early_stop_enabled or st.early_stop_active:
        return
    from huginn.research.early_stop_gate import stability_check

    settled_layers: list[list[tuple[str, float]]] = []
    for i in range(min(settled_until, len(st.layers_map) - 1) + 1):
        rows = []
        for n in st.layers_map[i]:
            res = st.cache.get(n) or {}
            s = _first_scalar(res.get("objectives") or res.get("summary"))
            if s is not None:
                rows.append((n, s))
        settled_layers.append(rows)
    _st = stability_check(
        settled_layers, min_layers=st.early_stop_min_layers, margin=st.early_stop_margin
    )
    st.early_stop_meta["stability"] = _st
    if _st["stable"] and settled_until + 1 < len(st.layers_map):
        st.early_stop_active = True
        st.early_stop_meta["stopped_after_layer"] = settled_until
        st.early_stop_meta["skipped"] = []
        st.early_stop_meta["verdict"] = "early_stopped"
    else:
        st.early_stop_meta.setdefault("verdict", "checked_and_continued")


def _rp_spec_for(st: _RPState, branch: Any) -> Experiment | None:
    spec = st.spec_by_name.get(branch.name)
    if spec is None:  # 动态子代(变异/演化新分支)
        spec = _rp_resolve_variant(st, branch.name, branch)
        if spec is not None:
            st.spec_by_name[branch.name] = spec  # 注册, 供综合阶段取假说/摘要
    return spec


def _rp_early_stop_skip(st: _RPState, name: str) -> dict | None:
    """P-C: 占优方向已稳定 → 剩余层实验终止 (未执行=不产生证据, 不伪造)."""
    if not st.early_stop_active:
        return None
    if _rp_layer_of(st, name) <= st.early_stop_meta.get("stopped_after_layer", -1):
        return None
    st.early_stop_meta.setdefault("skipped", []).append(
        {
            "name": name,
            "layer": _rp_layer_of(st, name),
            "reason": "early_termination",
            "evidence_from": f"layers 0..{st.early_stop_meta['stopped_after_layer']}",
        }
    )
    return {"success": False, "objectives": {}, "results": {"early_stopped": True}}


def _rp_replan_skip(st: _RPState, name: str) -> dict | None:
    """P-B: 前序层真实证据结算后判定该实验是否仍值得执行."""
    if not st.replan_enabled:
        return None
    st.replan_checked += 1
    _rp = _rp_replan_decision(st, name)
    if _rp is None:
        return None
    st.replan_log.append(_rp)
    return {
        "success": False,
        "objectives": {},
        "results": {"replanned": _rp["reasons"]},
    }


def _rp_attach_prediction(st: _RPState, res: dict, spec: Experiment) -> dict:
    """世界模型预筛(真 D): 预测值作为可证伪证据并入结果."""
    if st.world_model is None:
        return res
    res = dict(res)
    try:
        res["predicted"] = _wm_predict(st.world_model, spec)
    except Exception:  # noqa: BLE001 — 世界模型预筛失败: 不伪造, 保留纯真实结果
        res.setdefault("predicted", {})
    return res


def _rp_maybe_settle_layer(st: _RPState, name: str) -> None:
    if not st.early_stop_enabled or st.early_stop_active:
        return
    _i = _rp_layer_of(st, name)
    if _i < 0:
        return
    _layer = st.layers_map[_i]
    _es_skipped = {s.get("name") for s in st.early_stop_meta.get("skipped", [])}
    _rp_skipped = {r["name"] for r in st.replan_log}
    if all(n in st.cache or n in _es_skipped or n in _rp_skipped for n in _layer):
        _rp_settle_layer_check(st, _i)


def _rp_record_result(st: _RPState, name: str, res: dict, spec: Experiment) -> None:
    st.cache[name] = res
    # §10 扩展: 按同体指纹聚合 objectives(变异回退 → 父名; 普通 → 自身).
    _base = st.reuse_plan.get(name, name)
    st.runs_by_base.setdefault(_base, []).append(res.get("objectives") or {})
    _rp_maybe_settle_layer(st, name)
    if st.layer_epochs:
        try:
            from huginn.research.decision_gate import distill_tool_output

            _d = distill_tool_output(
                spec.name,
                json.dumps(res.get("summary", {}), ensure_ascii=False),
                goal=str(st.goal),
                max_chars=st.stream_summary_chars,
            )
            st.stream_rows.append(
                {
                    "layer": _rp_layer_of(st, spec.name),
                    "experiment": spec.name,
                    "summary": _d["front"],
                    "relevance": _d["relevance"],
                }
            )
        except Exception:  # noqa: BLE001 — 流式记录失败不影响执行结果
            pass


async def _rp_execute_branch(st: _RPState, branch: Any) -> dict:
    """主循环执行体: observe/decide/execute/validate 的单分支落地."""
    spec = _rp_spec_for(st, branch)
    if spec is None:
        return {"success": False, "objectives": {}, "results": {}}
    skipped = _rp_early_stop_skip(st, branch.name) or _rp_replan_skip(st, branch.name)
    if skipped is not None:
        return skipped
    res = await asyncio.to_thread(spec.run)
    res = _rp_attach_prediction(st, res, spec)
    _rp_record_result(st, branch.name, res, spec)
    return {
        "success": bool(res.get("success", True)),
        "objectives": res.get("objectives", {}),
        "results": res.get("summary", {}),
    }


def _rp_explore(
    st: _RPState, strategy: Any, max_iterations: int, min_iterations: int
) -> Any:
    """计划与团队阶段: 在预算内跑真实编排循环(branch_executor 绑定本上下文)."""
    from huginn.exploration.orchestrator import ExplorationOrchestrator

    orch = ExplorationOrchestrator(
        strategy=strategy,
        branch_executor=functools.partial(_rp_execute_branch, st),
        max_parallel=st.max_parallel,
    )
    return asyncio.run(
        orch.explore(
            objective=st.goal,
            initial_branches=[
                {"name": e.name, "hypothesis": e.hypothesis} for e in st.experiments
            ],
            objectives_config=st.objectives_config,
            max_iterations=max_iterations,
            min_iterations=min_iterations,
        )
    )


def _rp_collect(
    st: _RPState,
    result: Any,
    strategy: Any,
    self_audit: Callable[[], list[str]] | None,
    structural_audit: Callable[[list[dict], str], dict] | None,
) -> tuple[list, list[str], ResearchOutcome, dict | None, list]:
    """复盘与学习前置: 流式视图/证据 trace/outcome/结构闸门/存活集."""
    from huginn.exploration.strategies import MutationStrategy
    from huginn.exploration.supervisor import SupervisorStrategy

    st.stream_view, st.n_epochs = (
        _group_stream_view(st.stream_rows) if st.layer_epochs else (None, 0)
    )
    trace = _build_trace(st.cache)
    # 能力自省 §3: 把能力缺口提案的可证伪工件并入 trace, 使报告引用可被门禁核实
    if self_audit is not None:
        with contextlib.suppress(Exception):
            trace += list(self_audit())
    front = result.pareto_front or []
    out = ResearchOutcome(
        converred=result.convergence_reason,
        explored=result.n_branches_explored,
        pruned=result.n_branches_pruned,
        pareto_front=front,
        cache=st.cache,
        plan_summary=st.plan_summary,
    )
    _apply_strategy_metrics(out, strategy, MutationStrategy, SupervisorStrategy)
    audit = _run_structural_audit(structural_audit, front, st.goal, trace, out)
    survivors = [(b["name"], st.cache.get(b["name"], {})) for b in front]
    return front, trace, out, audit, survivors


def _rp_generate_final(
    st: _RPState,
    out: ResearchOutcome,
    survivors: list,
    audit: dict | None,
    trace: list[str],
    *,
    client: Any,
    model: str,
    diagnostic_tools: list[dict] | None,
    strictness: int,
    verify: Callable[[str, list[str]], dict],
    workspace: Any,
) -> tuple[str, str, list]:
    """汇总与报告阶段: 成文 + 门禁重试 + 门控审计段 + 工作区落地."""
    synthesize = functools.partial(
        _synthesize_report, st.goal, survivors, st.spec_by_name, audit, out
    )
    final, verdict, ungrounded = _generate_report(
        client=client,
        model=model,
        goal=st.goal,
        survivors=survivors,
        stream_view=st.stream_view,
        audit=audit,
        out=out,
        diagnostic_tools=diagnostic_tools,
        strictness=strictness,
        trace=trace,
        synthesize=synthesize,
        verify=verify,
        judgment_hints=st.judgment_hints,
    )
    final = _append_gate_audits(
        final, trace, st.replan_log, st.early_stop_active, st.early_stop_meta
    )
    g2 = verify(final, trace)
    if g2["verdict"] == "pass":
        verdict, ungrounded = "pass", []
    else:
        ungrounded = g2["unsubstantiated"]
    # C-Space 工作区门: 先以存活结论真实证据灌在场, 再让报告经 broadcast 落地.
    if workspace is not None:
        final = _apply_workspace_gate(workspace, survivors, final, out, st.goal)
    return final, verdict, ungrounded


def _rp_finish(
    st: _RPState,
    out: ResearchOutcome,
    final: str,
    verdict: str,
    ungrounded: list,
    front: list,
    audit: dict | None,
    survivors: list,
    workspace: Any,
    external_verifier: Callable[[dict], dict] | None,
    harness_agent: str,
    harness_machine: str,
    out_md: Path | None,
) -> None:
    """收尾/落盘阶段: 使用审计、收敛聚合头、harness 与报告文件."""
    out.report = final
    out.verdict, out.ungrounded = verdict, ungrounded
    _audit_usage_and_plan(out, final, front, st.plan_summary, st.cache, st.goal)
    if st.world_model is not None:
        _record_world_model_usage(out, st.cache, st.world_model)
    ctx = _rp_finish_ctx(
        st, front, verdict, ungrounded, audit, workspace, external_verifier
    )
    _consolidate_outcome(out, ctx)
    _attach_harness(out, st.goal, harness_agent, harness_machine)
    _write_report_file(
        out_md,
        final,
        out,
        verdict,
        ungrounded,
        audit,
        workspace,
        st.replan_log,
        st.early_stop_meta,
        st.early_stop_active,
    )


def _rp_finish_ctx(
    st: _RPState,
    front: list,
    verdict: str,
    ungrounded: list,
    audit: dict | None,
    workspace: Any,
    external_verifier: Callable[[dict], dict] | None,
) -> dict[str, Any]:
    """收敛聚合头的输入上下文: 前沿/缓存/门禁/层门控/判断软提示."""
    return {
        "front": front,
        "cache": st.cache,
        "verdict": verdict,
        "ungrounded": ungrounded,
        "audit": audit,
        "workspace": workspace,
        "stream_view": st.stream_view,
        "n_epochs": st.n_epochs,
        "replan_enabled": st.replan_enabled,
        "replan_checked": st.replan_checked,
        "replan_log": st.replan_log,
        "early_stop_enabled": st.early_stop_enabled,
        "early_stop_meta": st.early_stop_meta,
        "layers_map": st.layers_map,
        "early_stop_min_layers": st.early_stop_min_layers,
        "early_stop_margin": st.early_stop_margin,
        "prior_args": st.prior_args,
        "runs_by_base": st.runs_by_base,
        "judgment_hints": st.judgment_hints,
        "external_verifier": external_verifier,
    }


def _llm_tournament(
    ideas: list[dict], *, goal: str, model: str, client: Any
) -> list[dict]:
    """用 client 做 LLM 想法 tournament: 返回存活想法 (解析失败则全存活, 保证稳健)."""
    bullets = "\n".join(
        f"- {it['name']}: hypothesis={it['hypothesis']} "
        f"obj={json.dumps(it['objectives'], ensure_ascii=False)}"
        for it in ideas
    )
    prompt = (
        f"你是科研创意评审委员。目标: {goal}。\n候选存活想法:\n{bullets}\n\n"
        "请交叉评审, 淘汰缺乏证据、重复或低可行性的想法, 返回存活想法的 name 列表, "
        '只输出 JSON: {"survivors": ["name", ...]}。'
    )
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        text = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        survivors = set(json.loads(m.group(0))["survivors"])
        return [it for it in ideas if it["name"] in survivors] or ideas
    except Exception:  # noqa: BLE001 — 评审失败不退化为空, 全存活
        return ideas


def _build_strategy(
    *,
    mutation_config: dict | None,
    human_review: Callable[[dict], dict] | None,
    debate: bool,
    debate_eval: Callable[[list[dict]], list[dict]] | None,
    supervisor_every: int,
    max_parallel: int,
) -> Any:
    """策略链: Pareto → (可选)变异 → (可选)HITL/辩论评审."""
    import huginn.exploration.strategies as S
    from huginn.exploration.strategies import MutationStrategy
    from huginn.exploration.supervisor import SupervisorStrategy

    strategy: Any = S.ParetoPruningStrategy(max_active=max_parallel + 6)
    if mutation_config:
        strategy = MutationStrategy(
            wrapped=strategy,
            param_space=mutation_config.get("param_space"),
            mutation_rate=mutation_config.get("mutation_rate", 0.5),
            max_children=mutation_config.get("max_children", 3),
        )
    if human_review or debate:
        strategy = SupervisorStrategy(
            wrapped=strategy,
            human_review=human_review,
            review_every=supervisor_every,
            debate_evaluator=debate_eval,
        )
    return strategy


def _group_stream_view(rows: list[dict]) -> tuple[list[dict] | None, int]:
    """P-A 流式结算视图: 逐实验增量行按层分组为有界 stream_view(仅记录, 不改执行)."""
    if not rows:
        return None, 0
    by_layer: dict[int, list[dict]] = {}
    for _r in rows:
        by_layer.setdefault(_r["layer"], []).append(
            {"experiment": _r["experiment"], "summary": _r["summary"]}
        )
    view = [
        {"layer": i, "experiments": len(layer_rows), "rows": layer_rows}
        for i, layer_rows in sorted(by_layer.items())
    ]
    return view, len(view)


def _apply_strategy_metrics(
    out: ResearchOutcome, strategy: Any, mutation_cls: type, supervisor_cls: type
) -> None:
    """进化/监督衍生量: 变异子代数 + HITL 反馈记录(若启用了对应策略)."""
    mstrat = (
        getattr(strategy, "wrapped", None)
        if isinstance(strategy, supervisor_cls)
        else None
    )
    if isinstance(strategy, mutation_cls):
        out.mutations = strategy.created
    elif isinstance(mstrat, mutation_cls):
        out.mutations = mstrat.created
    if isinstance(strategy, supervisor_cls):
        out.supervision_log = list(strategy.review_log)


def _run_structural_audit(
    structural_audit: Callable[[list[dict], str], dict] | None,
    front: list[dict],
    goal: str,
    trace: list[str],
    out: ResearchOutcome,
) -> dict | None:
    """结构闸门(交互等效/多元论治理): 代理/结论进报告前, 自动过一遍结构对齐审计."""
    if structural_audit is None:
        return None
    audit: dict | None
    try:
        audit = structural_audit(list(front), goal)
    except Exception as exc:  # noqa: BLE001 — 结构审计异常不阻断主流程, 如实记为未通过
        audit = {
            "pass": False,
            "reason": f"structural_audit failed: {exc}",
            "surrogate_only": [],
            "law_only": [],
        }
    if isinstance(audit, dict):
        audit.setdefault("pass", False)
        audit.setdefault("surrogate_only", [])
        audit.setdefault("law_only", [])
        # 可证伪工件并入 trace —— 报告里对结构风险的引用即可被 grounding 门禁核实.
        trace.append(
            json.dumps({"type": "structural_gate", **audit}, ensure_ascii=False)
        )
        out.structural_gate = audit
        out.structural_aligned = bool(audit["pass"])
    return audit


def _synthesize_report(
    goal: str,
    survivors: list,
    spec_by_name: dict[str, Experiment],
    audit: dict | None,
    out: ResearchOutcome,
) -> str:
    """确定性兜底成文: 数值全来自真实执行, 必过门禁."""
    L = [
        f"# 自主深研 — {goal}",
        "",
        "> 存活假说(Pareto 前沿, debate 淘汰后): "
        + ", ".join(n for n, _ in survivors)
        + "",
        "",
    ]
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
            L.append(
                "**代理结论存在与定律不符的交互结构(疑似 shortcut/混淆), "
                "不应直接进入决策。**"
            )
    return "\n".join(L)


def _append_gate_audits(
    final: str,
    trace: list[str],
    replan_log: list,
    early_stop_active: bool,
    early_stop_meta: dict,
) -> str:
    """P-B / P-C 审计段: 只陈述预算决策, 不捏造任何实验结果."""
    # P-B 层间重规划审计: 报告如实说明被门控跳过的实验(证据进 trace 供 grounding 核对).
    # 跳过 = 未执行 = 无证据 —— 报告只陈述"重规划决策", 不捏造任何实验结果.
    if replan_log:
        trace.append(
            json.dumps({"type": "replan_gate", "log": replan_log}, ensure_ascii=False)
        )
        final += (
            "\n\n## 层间重规划(P-B)\n"
            f"{len(replan_log)} 个计划内实验经前序层真实证据门控判定冗余/被证伪方向, "
            "重规划跳过(预算再分配; 未执行 = 不产生证据, 不伪造): "
            + ", ".join(
                f"{r['name']}({'; '.join(x['rule'] for x in r['reasons'])})"
                for r in replan_log
            )
            + "。完整理由见 trace.replan_gate。"
        )

    # P-C 证据驱动提前终止审计: 报告如实说明被终止的实验(证据进 trace 供 grounding 核对).
    # 终止 = 未执行 = 无证据 —— 报告只陈述"稳定度预算决策", 不捏造任何实验结果.
    if early_stop_active:
        trace.append(
            json.dumps(
                {"type": "early_stop_gate", "meta": early_stop_meta}, ensure_ascii=False
            )
        )
        _st = early_stop_meta.get("stability", {})
        _es_n = len(early_stop_meta.get("skipped", []))
        final += (
            "\n\n## 证据驱动提前终止(P-C)\n"
            f"执行到层 {early_stop_meta.get('stopped_after_layer')} 后占优方向进入分数高原"
            f"(top={_st.get('top')}, score={_st.get('score')}, "
            f"rel_change={_st.get('relative_change')}), 剩余 {_es_n} 个计划内实验提前终止"
            "(预算回收; 未执行 = 不产生证据, 不伪造)。证据见 trace.early_stop_gate。"
        )
    return final


def _audit_usage_and_plan(
    out: ResearchOutcome,
    final: str,
    front: list,
    plan_summary: dict | None,
    cache: dict[str, dict],
    goal: str,
) -> None:
    """漏C 得分≠使用 + 漏A plan 修订门 (审计失败不阻断)."""
    # ── 漏C · "得分≠使用" grounding 审计(看过≠用过) ─────────────────────
    # 高价值(存活/高分)假说的数值是否真写进最终报告. **不**因为被检索到就当作用过.
    try:
        from huginn.research.decision_gate import grounding_audit

        out.grounding_audit = grounding_audit(final, list(front))
    except Exception:  # noqa: BLE001 — 审计为附加值, 失败不应阻断管线
        out.grounding_audit = None

    # ── 漏A · plan 修订门(锚定反制) ─────────────────────────────────────
    # 若跑过 planner, 用执行后证据显式复盘初始计划是否仍成立(而非默认上下文覆盖).
    if plan_summary is not None:
        try:
            from huginn.research.decision_gate import revision_gate

            out.plan_revision = revision_gate(plan_summary, cache, goal)
        except Exception:  # noqa: BLE001 — 修订审计失败不阻断, 如实留空
            out.plan_revision = None


def _attach_harness(
    out: ResearchOutcome, goal: str, harness_agent: str, harness_machine: str
) -> None:
    """M2 Self-Harness 五维报告: 复用 out 已记录的 gate 结果, 不重复计算."""
    try:
        from huginn.research.harness import build_harness_report

        out.harness = build_harness_report(
            goal, out, agent=harness_agent, machine=harness_machine
        ).to_dict()
    except Exception:  # noqa: BLE001 — harness 为可选附加值, 失败不应阻断管线
        out.harness = None


def _diag_tool_messages(
    calls: list[Any],
    tool_handlers: dict[str, Any],
    trace: list[str],
    goal: str,
    assistant_text: str,
) -> list[dict]:
    """执行一轮诊断工具调用, 返回要追加到 messages 的 assistant+tool 消息."""
    msgs: list[dict] = []
    for tc in calls:
        msgs += _diag_tool_message(tc, tool_handlers, trace, goal, assistant_text)
    return msgs


def _diag_tool_message(
    tc: Any,
    tool_handlers: dict[str, Any],
    trace: list[str],
    goal: str,
    assistant_text: str,
) -> list[dict]:
    """单个诊断工具调用: 执行 → 落 trace → 先导摘要 → 组装 assistant+tool 消息."""
    name = tc.function.name
    a: Any = {}
    try:
        a = (
            json.loads(tc.function.arguments)
            if isinstance(tc.function.arguments, str)
            else (tc.function.arguments or {})
        )
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
    # 漏B 决策先导摘要: 长工具回调进决策上下文前先净化(先导摘要),
    # 防止均匀灌注稀释; 完整原始证据仍在 trace 供 grounding 门禁核对.
    try:
        from huginn.research.decision_gate import distill_tool_output

        _g = distill_tool_output(name, res, goal=str(goal), max_chars=4000)
        _message = _g["front"]
    except Exception:  # noqa: BLE001 — 门控不可用时退原样, 不阻断
        _message = res
    trace.append(
        json.dumps(
            {"gate": "front_summary", "tool": name, "len": len(res)},
            ensure_ascii=False,
        )
    )
    return [
        {
            "role": "assistant",
            "content": assistant_text,
            "tool_calls": [tc.model_dump()],
        },
        {"role": "tool", "tool_call_id": tc.id, "content": _message},
    ]


def _stream_view_text(stream_view: list) -> str:
    """把分层流式视图压成有界文本 (每层经漏B 摘要)."""
    return "\n".join(
        f"- {_lay['experiments']} 实验(层 {_lay['layer']}): "
        + "; ".join(f"{_r['experiment']}={_r['summary']}" for _r in _lay["rows"])
        for _lay in stream_view
    )


def _audit_prompt_block(out: ResearchOutcome, audit: dict) -> str:
    """结构闸门提示: 让 LLM 成文时如实反映交互等效审计结果."""
    if out.structural_aligned is not False:
        return "\n【结构对齐闸门】代理经交互等效审计与定律一致(无虚假交互)。"
    return (
        f"\n【结构对齐闸门】代理未通过: {audit.get('reason', '')}; "
        f"surrogate_only={audit.get('surrogate_only')}。"
        "报告中须如实说明此结构风险, 不许隐瞒。"
    )


def _build_report_prompt(
    *,
    goal: str,
    survivors: list,
    stream_view: list | None,
    audit: dict | None,
    out: ResearchOutcome,
    diagnostic_tools: list[dict] | None,
    strictness: int,
    trace: list[str],
    judgment_hints: list[str],
) -> tuple[str, list, dict]:
    """组装成文 prompt + 诊断工具挂载面 (流式视图/结构闸门/分级护栏)."""
    prompt = _report_prompt_base(goal, survivors)
    # 缺陷一(P-A) 分层流式结算: 决策上下文改为"增量层摘要视图(有界)"而非全 trace——
    # LLM 只读 stream_view(每层经漏B 压缩), 全量证据始终留 trace 供 grounding 门禁.
    if stream_view:
        prompt += (
            "\n\n【分层流式证据视图(P-A)】已按执行层增量结算的摘要(有界上下文; "
            "完整可证伪证据在 trace 中):\n" + _stream_view_text(stream_view)
        )
    # 结构闸门提示: 让 LLM 成文时如实反映交互等效审计结果, 不隐瞒 shortcut 风险.
    if audit is not None:
        prompt += _audit_prompt_block(out, audit)
    prompt, tool_schemas, tool_handlers = _diagnostic_tools_block(
        prompt, diagnostic_tools
    )
    prompt = _guardrail_block(
        prompt, goal, trace, survivors, strictness, judgment_hints
    )
    return prompt, tool_schemas, tool_handlers


def _report_prompt_base(goal: str, survivors: list) -> str:
    """成文 prompt 头: 目标 + 存活假说真实数值证据 + 写作约束."""
    survivors_text = "\n".join(
        f"- {name}: {json.dumps(res.get('summary', {}), ensure_ascii=False)}"
        for name, res in survivors
    )
    return (
        f"你是 Huginn 科研智能体。目标: {goal}。\n"
        f"以下是存活假说的真实数值证据(可复现、非伪造):\n{survivors_text}\n\n"
        f"请撰写跨学科深度研究报告(研究问题/数据与方法/结果分析/对账与局限/下一步)。"
        f"每个数值必须来自上面真实结果, 不许编造。报告只允许引用上面这些真实结果里的数值;"
        f"任何来自历史报告/上下文对话的数值(如结论段外引用的旧统计量)一律禁止引用。"
        f"把最终报告放在 <report> 与 </report> 之间。"
    )


def _diagnostic_tools_block(
    prompt: str, diagnostic_tools: list[dict] | None
) -> tuple[str, list, dict]:
    # 统一诊断工具挂载面 —— 走单一薄控制点 (resolve_diagnostic_tools), 不内联再造 schema
    from huginn.research.tool_surface import resolve_diagnostic_tools

    tool_schemas, tool_handlers = resolve_diagnostic_tools(diagnostic_tools)
    if tool_schemas:
        prompt += (
            "\n可自主调用的域诊断工具(结果作为可证伪证据进报告): "
            + ", ".join(t["function"]["name"] for t in tool_schemas)
            + "。"
        )
    return prompt, tool_schemas, tool_handlers


def _guardrail_block(
    prompt: str,
    goal: str,
    trace: list[str],
    survivors: list,
    strictness: int,
    judgment_hints: list[str],
) -> str:
    # 判断层·分级护栏: 只追加进 prompt (软提示), 不参与 verify —— 强模型不受锁.
    # strictness=0 → 空串, 零行为变化(默认信任模型推理).
    if strictness > 0:
        from huginn.research.judgment_guardrail import hint_block

        _hs_block = hint_block(
            str(goal), trace=trace, survivors=survivors, strictness=int(strictness)
        )
        if _hs_block:
            prompt += "\n" + _hs_block
            # 聚合头载体: 写进闭包列表(经 guardrail.judgment head 入 consolidated)——
            # 不往 ResearchOutcome 挂字段(P3 红线: 新视角只许经聚合头)。
            judgment_hints[:] = [h for h in _hs_block.split("\n") if h.startswith("- ")]
    return prompt


def _attempt_with_tools(
    *,
    client: Any,
    model: str,
    msgs: list[dict],
    tool_schemas: list,
    tool_handlers: dict[str, Any],
    trace: list[str],
    goal: str,
) -> str:
    """最多 4 轮工具调用后取正文; 若一路调工具未收尾则强制一次"只写报告"."""
    msg: Any = None
    for _round in range(4):
        msg = _chat_tool_round(client, model, msgs, tool_schemas, trace)
        if msg is None:
            return ""
        calls = msg.tool_calls or []
        if not calls:
            return msg.content or ""
        msgs += _diag_tool_messages(
            calls[:4], tool_handlers, trace, goal, msg.content or ""
        )
    if msg is not None and not (msg.content or "").strip():
        return _forced_report_write(client, model, msgs, trace)
    return msg.content if msg is not None else ""


def _chat_tool_round(
    client: Any,
    model: str,
    msgs: list[dict],
    tool_schemas: list,
    trace: list[str],
) -> Any:
    """单轮带工具调用; 调用失败落 trace 并返回 None (由调用方终止成文)."""
    try:
        r = client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=tool_schemas or None,
            tool_choice="auto" if tool_schemas else None,
            max_tokens=4000,
            temperature=0.2,
            extra_body={"thinking_mode": False},
        )
        return r.choices[0].message
    except Exception as ee:
        # Client API 调用失败（权限/网络/base-url 错） → 记 trace 并进下一轮
        trace.append(
            json.dumps(
                {"error": f"client.chat.completions.create: {ee}"},
                ensure_ascii=False,
            )
        )
        return None


def _forced_report_write(
    client: Any, model: str, msgs: list[dict], trace: list[str]
) -> str:
    # 工具轮结束后若仍无正文(如模型一路调工具未收尾), 强制一次"只写报告"调用,
    # 避免返回空 content → 门禁因空报告打回 → 白白兜底.
    try:
        rr = client.chat.completions.create(
            model=model,
            messages=msgs
            + [
                {
                    "role": "user",
                    "content": "不要再调用工具。基于轨迹里的真实数值, "
                    "现在直接撰写完整研究报告, 把正文放在 "
                    "<report> 与 </report> 之间。",
                }
            ],
            max_tokens=4000,
            temperature=0.2,
        )
        return rr.choices[0].message.content or ""
    except Exception as ee:  # noqa: BLE001 — 兜底成文失败如实返回空
        trace.append(json.dumps({"error": f"final write: {ee}"}, ensure_ascii=False))
        return ""


def _report_retry_loop(
    *,
    client: Any,
    model: str,
    prompt: str,
    trace: list[str],
    verify: Callable[[str, list[str]], dict],
    synthesize: Callable[[], str],
    out: ResearchOutcome,
    tool_schemas: list,
    tool_handlers: dict[str, Any],
    goal: str,
) -> tuple[str, str, list]:
    """三轮「成文 → 声明门禁」重试; 全失败则确定性兜底组装(必过门禁)."""
    final, verdict, ungrounded = "", "needs_grounding", []
    msgs = [{"role": "user", "content": prompt}]
    for _i in range(3):
        final = _attempt_with_tools(
            client=client,
            model=model,
            msgs=msgs,
            tool_schemas=tool_schemas,
            tool_handlers=tool_handlers,
            trace=trace,
            goal=goal,
        ).strip()
        m = re.search(r"<report>(.*?)</report>", final, flags=re.DOTALL | re.IGNORECASE)
        if m:
            final = m.group(1).strip()
        g = verify(final, trace)
        if g["verdict"] == "pass" and len(final.strip()) > 200:
            verdict, ungrounded = "pass", []
            out.report_source = "agent"
            break
        ungrounded = g["unsubstantiated"]
        msgs = [
            {
                "role": "user",
                "content": f"未交付(未落地:{ungrounded})。请只用真实证据重写:\n"
                + prompt,
            }
        ]
        out.report_source = "agent(retry)"
    else:
        final = synthesize()  # L3b 兜底组装, 必过门禁
        out.report_source = "fallback_assembly(agent failed)"
    return final, verdict, ungrounded


def _generate_report(
    *,
    client: Any,
    model: str,
    goal: str,
    survivors: list,
    stream_view: list | None,
    audit: dict | None,
    out: ResearchOutcome,
    diagnostic_tools: list[dict] | None,
    strictness: int,
    trace: list[str],
    synthesize: Callable[[], str],
    verify: Callable[[str, list[str]], dict],
    judgment_hints: list[str],
) -> tuple[str, str, list]:
    """LLM 成文 + 声明门禁重试; 无 client 时走确定性兜底组装."""
    if client is None:
        out.report_source = "deterministic"
        return synthesize(), "needs_grounding", []
    prompt, tool_schemas, tool_handlers = _build_report_prompt(
        goal=goal,
        survivors=survivors,
        stream_view=stream_view,
        audit=audit,
        out=out,
        diagnostic_tools=diagnostic_tools,
        strictness=strictness,
        trace=trace,
        judgment_hints=judgment_hints,
    )
    return _report_retry_loop(
        client=client,
        model=model,
        prompt=prompt,
        trace=trace,
        verify=verify,
        synthesize=synthesize,
        out=out,
        tool_schemas=tool_schemas,
        tool_handlers=tool_handlers,
        goal=goal,
    )


def _apply_workspace_gate(
    workspace: Any, survivors: list, final: str, out: ResearchOutcome, goal: str
) -> str:
    """C-Space 工作区门: 存活结论先灌在场, 报告再经 broadcast 落地."""
    if workspace is not None:
        for name, res in survivors:
            workspace.register(
                name,
                "state",
                payload=dict(res.get("summary", {})),
                source=f"survivor:{name}",
                falsifiable=True,
            )
            if res.get("summary"):
                workspace.trace.append(json.dumps(res["summary"], ensure_ascii=False))
        ws_res = workspace.broadcast(final)
        out.workspace_verified = bool(ws_res.get("verified"))
        if out.workspace_verified:
            workspace.register(
                "report_" + (_slug_goal(goal) or "final"),
                "concept",
                payload={"source": "run_research_program.final"},
                source="report",
                falsifiable=True,
            )
        else:
            _miss = ws_res.get("unsubstantiated", [])
            final += (
                "\n\n## 工作区门禁(未确认在场)\n报告断言未作为可证伪在场落地; "
                "未落地主张: " + str(_miss) + "\n"
            )
    return final


def _record_world_model_usage(
    out: ResearchOutcome, cache: dict[str, dict], world_model: Any
) -> None:
    """世界模型真用证据: 预测 vs 真实执行对账 (偏差如实记 falsified)."""
    if world_model is not None:
        preds = []
        reconcile_rows: list[dict] = []
        for name, res in cache.items():
            p = res.get("predicted")
            if not p:
                continue
            preds.append({"experiment": name, "predicted": p})
            # 尝试与真实 summary 数值对账 (law_model.reconcile 语义): 预测里挑第一个
            # 标量指标 vs 真实结果 --- 可行则记 reconcile; 对不上/无数不去伪造。
            try:
                pred_scalar = _first_scalar(p)
                actual_scalar = _first_scalar(res.get("summary", {}))
                if pred_scalar is not None and actual_scalar is not None:
                    err = abs(pred_scalar - actual_scalar) / (abs(actual_scalar) or 1.0)
                    reconcile_rows.append(
                        {
                            "experiment": name,
                            "rel_err": round(err, 4),
                            "borne_out": err <= _WM_TOL,
                            "predicted": round(pred_scalar, 4),
                            "actual": round(actual_scalar, 4),
                        }
                    )
            except Exception:  # noqa: BLE001 — 对账失败不伪造, 只跳过
                pass
        out.law_model_used = {
            "count": len(preds),
            "mock": [p["predicted"] for p in preds[:3]],
            "reconcilable": bool(preds),
            "reconcile": reconcile_rows,
            "borne_out_all": bool(reconcile_rows)
            and all(r["borne_out"] for r in reconcile_rows),
        }


def _early_stop_summary(ctx: dict[str, Any]) -> dict[str, Any]:
    """P-C 提前终止门的可证伪明细(是否启用/稳定判据/终止名单/先验来源)."""
    _early_stop_enabled = ctx["early_stop_enabled"]
    _early_stop_meta = ctx["early_stop_meta"]
    _layers_map = ctx["layers_map"]
    _early_stop_min_layers = ctx["early_stop_min_layers"]
    _early_stop_margin = ctx["early_stop_margin"]
    _prior_args = ctx["prior_args"]
    return {
        "enabled": _early_stop_enabled,
        "layers": len(_layers_map) if _early_stop_enabled else 0,
        "stopped_after_layer": _early_stop_meta.get("stopped_after_layer"),
        "skipped": len(_early_stop_meta.get("skipped", [])),
        "log": list(_early_stop_meta.get("skipped", [])),  # 完整终止名单(可证伪)
        "stability": dict(
            _early_stop_meta.get("stability") or {}
        ),  # 稳定判据细节(可证伪)
        "prior_used": {
            "min_layers": _early_stop_min_layers,
            "margin": _early_stop_margin,
            "source": _prior_args["source"],
            "note": _prior_args["note"],
            "goal_matched": _prior_args["goal_matched"],
            "goal_level": _prior_args.get("goal_level"),
            "goal_overlap": _prior_args.get("goal_overlap"),
            "decay_weight": _prior_args.get("decay_weight"),
            "prior_age": _prior_args.get("prior_age"),
            "weight": _prior_args.get("weight"),
        },
        "verdict": (
            "no_early_stop"
            if not _early_stop_enabled
            else _early_stop_meta.get("verdict", "checked_and_continued")
        ),
    }


def _replan_summary(ctx: dict[str, Any]) -> dict[str, Any]:
    """P-B 层间重规划门的可证伪明细(检查数/跳过名单/结论)."""
    _replan_enabled = ctx["replan_enabled"]
    _replan_checked = ctx["replan_checked"]
    _replan_log = ctx["replan_log"]
    _layers_map = ctx["layers_map"]
    return {
        "enabled": _replan_enabled,
        "layers": len(_layers_map) if _replan_enabled else 0,
        "checked": _replan_checked,
        "skipped": len(_replan_log),
        "log": list(_replan_log),  # 完整审计: 被跳过实验+理由+证据面(可证伪)
        "verdict": (
            "no_replan"
            if not _replan_enabled
            else ("replanned" if _replan_log else "all_proceed")
        ),
    }


def _head_replan(HeadResult, enabled, checked, log, obs, unobs):
    # gate.replan —— P-B 层间重规划门 (缺陷二, 建议级: DAG 边在证据到达后可修订).
    # 视为 unobserved = 未启用(默认全执行, 零行为变化); observed = 真实完成门控判定
    # 并留下可证伪记录(跳过项+理由)—— 无论是否跳过, 门工作即 passed, 明细供复核.
    if enabled:
        return HeadResult(
            "gate.replan",
            "层间重规划门(缺陷二: DAG 边不再冻结)",
            obs,
            "passed",
            detail=(
                f"checked={checked} skipped={len(log)}; "
                + str([(r["name"], [x["rule"] for x in r["reasons"]]) for r in log])
            ),
            ref="out.consolidated.replan",
        )
    return HeadResult(
        "gate.replan",
        "层间重规划门(缺陷二: DAG 边可在证据后修订)",
        unobs,
        "unobserved",
        detail="replan_gate 未启用(默认全执行, 零行为变化)",
        ref="out.consolidated.replan",
    )


def _head_early_stop(HeadResult, early_stop_head, enabled, obs, unobs):
    # gate.early_stop —— P-C 证据驱动提前终止门 (缺陷一, 建议级: 稳定后不等全层).
    # unobserved = 未启用(默认全 BSP, 零行为变化); observed = 真实完成稳定度
    # 判定并留下可证伪记录 —— 无论是否终止, 门工作即 passed, 明细供复核.
    return HeadResult(
        "gate.early_stop",
        "证据驱动提前终止门(P-C: 分数高原后不等全层)",
        obs if enabled else unobs,
        "passed" if enabled else "unobserved",
        detail=str(early_stop_head),
        ref="out.consolidated.early_stop",
    )


def _head_claim_grounding(HeadResult, verdict, ungrounded, obs, unobs):
    # gate.claim_grounding —— 声明门禁 (对报告是否成文有否决权)
    if verdict != "needs_grounding" or ungrounded:
        return HeadResult(
            "gate.claim_grounding",
            "声明门禁(claim_grounding)",
            obs,
            "passed"
            if verdict in ("grounded", "pass", "accept", "confirmed")
            else "failed",
            detail=f"verdict={verdict}",
            ref="out.verdict",
            gate=True,
        )
    return HeadResult(
        "gate.claim_grounding",
        "声明门禁(claim_grounding)",
        unobs,
        "unobserved",
        detail="verdict=needs_grounding 且无 ungrounded 素材",
        ref="out.verdict",
        gate=True,
    )


def _head_structural(HeadResult, out, audit, obs, unobs):
    # gate.structural —— 结构闸门 (交互等效/多元论审计)
    if audit is not None:
        return HeadResult(
            "gate.structural",
            "结构闸门(交互等效/多元论审计)",
            obs,
            "passed" if out.structural_aligned is not False else "failed",
            detail=str(audit),
            ref="out.structural_gate",
            gate=True,
        )
    return HeadResult(
        "gate.structural",
        "结构闸门(交互等效/多元论审计)",
        unobs,
        "unobserved",
        detail="未注入 structural_audit",
        ref="out.structural_gate",
        gate=True,
    )


def _head_workspace(HeadResult, out, workspace, obs, unobs):
    # gate.workspace —— 工作区广播门 (C-Space 在场断言)
    if workspace is not None:
        return HeadResult(
            "gate.workspace",
            "工作区广播门(C-Space 在场断言)",
            obs,
            "passed" if out.workspace_verified else "failed",
            detail=f"workspace_verified={out.workspace_verified}",
            ref="out.workspace_verified",
            gate=True,
        )
    return HeadResult(
        "gate.workspace",
        "工作区广播门(C-Space 在场断言)",
        unobs,
        "unobserved",
        detail="未注入 workspace",
        ref="out.workspace_verified",
        gate=True,
    )


def _head_wm(HeadResult, out, obs, unobs):
    # wm.actually_used —— 世界模型真用 (误区二审计, 建议级)
    if out.law_model_used is not None:
        wm_ok = out.law_model_used.get("borne_out_all") is not False
        return HeadResult(
            "wm.actually_used",
            "世界模型真用?(predict 参与决策 = 真深思 D)",
            obs,
            "passed" if wm_ok else "failed",
            detail=str(out.law_model_used),
            ref="out.law_model_used",
        )
    return HeadResult(
        "wm.actually_used",
        "世界模型真用?(predict 参与决策 = 真深思 D)",
        unobs,
        "unobserved",
        detail="未见 predict/reconcile 产物",
        ref="out.law_model_used",
    )


def _head_plan_revision(HeadResult, out, obs):
    # audit.plan_revision —— 锚定反制 (漏A, 建议级)
    if out.plan_revision is not None:
        return HeadResult(
            "audit.plan_revision",
            "plan 修订门(新证据→显式复盘初始计划)",
            obs,
            "passed" if out.plan_revision.get("verdict") == "plan_holds" else "failed",
            detail=str(out.plan_revision),
            ref="out.plan_revision",
        )
    return None


def _head_score_usage(HeadResult, out, obs):
    # audit.score_usage —— 得分≠使用 (漏C, 对疑似 lipstick 有否决权)
    if out.grounding_audit is not None:
        return HeadResult(
            "audit.score_usage",
            "得分≠使用(高分存活项真进最终报告?)",
            obs,
            "passed"
            if out.grounding_audit.get("verdict") == "proper_use"
            else "failed",
            detail=str(out.grounding_audit),
            ref="out.grounding_audit",
            gate=True,
        )
    return None


def _head_supervision(HeadResult, out, obs, unobs):
    # controlled.supervision —— 受控执行 (建议级, 供 P2 六维投影)
    if getattr(out, "supervision_log", None):
        return HeadResult(
            "controlled.supervision",
            "受控执行(HITL/权限评审记录)",
            obs,
            "passed",
            detail=f"记录 {len(out.supervision_log)} 条",
            ref="out.supervision_log",
        )
    return HeadResult(
        "controlled.supervision",
        "受控执行(HITL/权限评审记录)",
        unobs,
        "unobserved",
        detail="supervisor_every=0 或未触发人工评审",
        ref="out.supervision_log",
    )


def _head_evidence_cache(HeadResult, cache, obs, unobs):
    # reliable.evidence_cache —— 真实执行证据 (建议级, 供 P2 六维投影)
    if cache:
        return HeadResult(
            "reliable.evidence_cache",
            "真实执行证据(缓存/expts 实录)",
            obs,
            "passed",
            detail=f"{len(cache)} 个实验",
            ref="out.cache",
        )
    return HeadResult(
        "reliable.evidence_cache",
        "真实执行证据(缓存/expts 实录)",
        unobs,
        "unobserved",
        detail="cache 为空",
        ref="out.cache",
    )


def _head_self_audit(HeadResult, out, obs, unobs):
    # learning.self_audit —— 自省与经验沉淀 (建议级, 供 P2 六维投影)
    artifacts = (
        out.structural_gate,
        out.plan_summary,
        out.plan_revision,
        out.grounding_audit,
        getattr(out, "supervision_log", None),
    )
    if any(a not in (None, []) for a in artifacts):
        return HeadResult(
            "learning.self_audit",
            "能力自省闭环(缺口→提案)",
            obs,
            "passed",
            detail="该 run 产生可复用审计/规划产物",
            ref="out.structural_gate",
        )
    return HeadResult(
        "learning.self_audit",
        "能力自省闭环(缺口→提案)",
        unobs,
        "unobserved",
        detail="自省已接线但本 run 无审计产物",
        ref="capabilities/introspection",
    )


def _head_guardrail(HeadResult, judgment_hints, obs, unobs):
    # guardrail.judgment —— 判断层·分级护栏软提示(注入 prompt 但不参与 verify).
    # 建议级: 反映"强护栏下注入了几分提示", 供复核; 不具否决权, strictness=0 → 未启用.
    # 不再往 ResearchOutcome 挂 judgment_hints 字段 —— 遵守 P3 新视角只许经聚合头.
    if judgment_hints:
        return HeadResult(
            "guardrail.judgment",
            "判断层·分级护栏(soft hints 注入 prompt)",
            obs,
            "passed",
            detail=f"注入 {len(judgment_hints)} 条: " + "; ".join(judgment_hints),
            ref="out.consolidated.head_details",
        )
    return HeadResult(
        "guardrail.judgment",
        "判断层·分级护栏(soft hints 注入 prompt)",
        unobs,
        "unobserved",
        detail="strictness=0 或该 run 未触发护栏(零行为变化)",
        ref="out.consolidated.head_details",
    )


def _meta_common(
    ctx, verdict, front, replan_meta, early_stop_head, replication_view
) -> dict[str, Any]:
    return {
        "grounding_verdict": verdict,
        "role_view": list(front),
        "epochs": ctx["n_epochs"],
        "stream_view": ctx["stream_view"],
        "replan": replan_meta,
        "early_stop": early_stop_head,
        "replication": replication_view,
    }


def _head_external_verify(HeadResult, ext, obs, unobs):
    return HeadResult(
        "governance.external_verify",
        "独立验证方(可替换的外部复核)",
        ext.get("evidence", unobs),
        ("passed" if ext.get("verified") else "failed")
        if ext.get("verified") is not None
        else "unobserved",
        detail=str(ext),
        ref="out.consolidated.external_verify",
        gate=True,
    )


def _head_team_diversity(HeadResult, role, front, obs, unobs):
    if front:
        return HeadResult(
            "team.diversity",
            "团队视角分离度(防多头塌缩)",
            obs,
            {"diverse": "passed", "collapsed_single_view": "failed"}.get(
                role.get("verdict"), "unobserved"
            ),
            detail=str(role),
            ref="out.pareto_front",
        )
    return HeadResult(
        "team.diversity",
        "团队视角分离度(防多头塌缩)",
        unobs,
        "unobserved",
        detail="无存活假说, 无从度量视角分化",
        ref="out.pareto_front",
    )


def _head_overbuild(HeadResult, ob, obs):
    return HeadResult(
        "meta.overbuild_guard",
        "过度建制审计(second system effect 反制)",
        obs,
        "passed" if ob.get("verdict") == "healthy" else "failed",
        detail=str(ob),
        ref="out.consolidated.overbuild",
    )


class _ConsolidationHeads:
    """聚合物头构造器: 门禁组 / 审计组 / 元头组, 每组批量产出 HeadResult."""

    def __init__(self, agg: Any) -> None:
        self.HeadResult = agg.HeadResult
        self.observed = agg.EVIDENCE_OBSERVED
        self.unobserved = agg.EVIDENCE_UNOBSERVED
        self.consolidate = agg.consolidate
        self.oracle_verify = agg.oracle_verify_consolidated
        self.replication_head = agg.build_replication_head
        self.replication_view = agg.build_replication_view

    def gates(
        self,
        out: ResearchOutcome,
        ctx: dict[str, Any],
        early_stop_head: dict[str, Any],
        replication_view: Any,
    ) -> list[Any]:
        """门禁组: 重规划/提前终止/复现/声明/结构/工作区."""
        HeadResult = self.HeadResult
        obs, unobs = self.observed, self.unobserved
        _replan_enabled = ctx["replan_enabled"]
        _replan_checked = ctx["replan_checked"]
        _replan_log = ctx["replan_log"]
        _early_stop_enabled = ctx["early_stop_enabled"]
        _runs_by_base = ctx["runs_by_base"]
        verdict = ctx["verdict"]
        ungrounded = ctx["ungrounded"]
        audit = ctx["audit"]
        workspace = ctx["workspace"]
        heads: list[Any] = []
        heads.append(
            _head_replan(
                HeadResult, _replan_enabled, _replan_checked, _replan_log, obs, unobs
            )
        )
        heads.append(
            _head_early_stop(
                HeadResult, early_stop_head, _early_stop_enabled, obs, unobs
            )
        )
        # audit.replication —— §10 扩展: 重复实验一致性 (同体多次执行是否可复现).
        # 无重复执行 → unobserved(默认零噪音); 有重复 → observed + 明细(供人类决策,
        # 一致性不是质量判据, 无否决权).
        heads.append(self.replication_head(_runs_by_base))
        heads.append(_head_claim_grounding(HeadResult, verdict, ungrounded, obs, unobs))
        heads.append(_head_structural(HeadResult, out, audit, obs, unobs))
        heads.append(_head_workspace(HeadResult, out, workspace, obs, unobs))
        return heads

    def audits(
        self,
        out: ResearchOutcome,
        cache: dict[str, dict],
        judgment_hints: list[str],
    ) -> list[Any]:
        """审计组: 世界模型真用/plan 修订/得分≠使用/受控/证据/自省/判断护栏."""
        HeadResult = self.HeadResult
        obs, unobs = self.observed, self.unobserved
        heads: list[Any] = []
        heads.append(_head_wm(HeadResult, out, obs, unobs))
        plan_revision_head = _head_plan_revision(HeadResult, out, obs)
        if plan_revision_head is not None:
            heads.append(plan_revision_head)
        score_usage_head = _head_score_usage(HeadResult, out, obs)
        if score_usage_head is not None:
            heads.append(score_usage_head)
        heads.append(_head_supervision(HeadResult, out, obs, unobs))
        heads.append(_head_evidence_cache(HeadResult, cache, obs, unobs))
        heads.append(_head_self_audit(HeadResult, out, obs, unobs))
        heads.append(_head_guardrail(HeadResult, judgment_hints, obs, unobs))
        return heads

    def meta(
        self,
        heads: list[Any],
        out: ResearchOutcome,
        ctx: dict[str, Any],
        replan_meta: dict[str, Any],
        early_stop_head: dict[str, Any],
        replication_view: Any,
    ) -> Any:
        """元头组: 外部验证 + 团队视角分离度 + 过度建制, 三轮合并后返回最终视图."""
        HeadResult = self.HeadResult
        obs, unobs = self.observed, self.unobserved
        front = ctx["front"]
        verdict = ctx["verdict"]
        _ver = (
            ctx["external_verifier"]
            if ctx["external_verifier"] is not None
            else self.oracle_verify
        )
        common = _meta_common(
            ctx, verdict, front, replan_meta, early_stop_head, replication_view
        )
        base = self.consolidate(heads, external_verifier=_ver, **common)
        ext = base.external_verify or {}
        heads.append(_head_external_verify(HeadResult, ext, obs, unobs))
        _role = base.role_diversity or {}
        heads.append(_head_team_diversity(HeadResult, _role, front, obs, unobs))
        final_cons = self.consolidate(heads, head_budget=_HEAD_BUDGET, **common)
        final_cons.external_verify = ext  # 第二轮不重跑验证方, 保留第一轮独立复核结果
        # 缺陷七: 治理自身是否过度建制 —— 建议级元头(不否决, 只亮灯).
        _ob = final_cons.overbuild or {}
        heads.append(_head_overbuild(HeadResult, _ob, obs))
        final_cons2 = self.consolidate(heads, head_budget=_HEAD_BUDGET, **common)
        final_cons2.external_verify = ext  # 保留第一轮独立复核结果(第二轮不重跑验证方)
        return final_cons2


def _consolidate_outcome(out: ResearchOutcome, ctx: dict[str, Any]) -> None:
    """收敛聚合头: 把治理字段收敛成 out.consolidated (失败不阻断)."""
    try:
        from huginn.research import aggregation_head as agg
    except Exception:  # noqa: BLE001 — 聚合头不可用: 如实留空, 不阻断管线
        out.consolidated = None
        return
    try:
        builder = _ConsolidationHeads(agg)
        early_stop_head = _early_stop_summary(ctx)
        replan_meta = _replan_summary(ctx)
        replication_view = builder.replication_view(ctx["runs_by_base"])
        heads = builder.gates(out, ctx, early_stop_head, replication_view)
        heads += builder.audits(out, ctx["cache"], ctx["judgment_hints"])
        out.consolidated = agg.consolidate(
            heads,
            grounding_verdict=ctx["verdict"],
            epochs=ctx["n_epochs"],
            stream_view=ctx["stream_view"],
            replan=replan_meta,
            early_stop=early_stop_head,
            replication=replication_view,
        ).as_dict()
        # 缺陷三/五接缝: 在第一轮聚合视图上追加"元头"(外部验证 + 团队视角分离度).
        # 第一轮先用可替换外部验证方(缺省出厂 oracle)复核; 派生两个头后第二轮合并,
        # 再以第一轮验证结果覆盖 external_verify(第二轮不重跑验证方, 幂等).
        try:
            merged = builder.meta(
                heads, out, ctx, replan_meta, early_stop_head, replication_view
            )
            out.consolidated = merged.as_dict()  # overbuild 用含全部头的最终视图(自洽)
        except Exception:  # noqa: BLE001 — 元头派生失败: 保留第一轮聚合视图, 不阻断
            pass
    except Exception:  # noqa: BLE001 — 聚合头为新增视图, 失败不阻断管线
        out.consolidated = None


def _write_report_file(
    out_md: Path | None,
    final: str,
    out: ResearchOutcome,
    verdict: str,
    ungrounded: list,
    audit: dict | None,
    workspace: Any,
    replan_log: list,
    early_stop_meta: dict,
    early_stop_active: bool,
) -> None:
    """报告页眉 + 正文落盘 (页眉在聚合头产出后再写)."""
    if out_md is not None:
        header = _report_header(
            out,
            verdict,
            ungrounded,
            audit,
            workspace,
            replan_log,
            early_stop_meta,
            early_stop_active,
        )
        out_md.write_text(header + final.strip() + "\n", encoding="utf-8")


def _report_header(
    out: ResearchOutcome,
    verdict: str,
    ungrounded: list,
    audit: dict | None,
    workspace: Any,
    _replan_log: list,
    _early_stop_meta: dict,
    _early_stop_active: bool,
) -> str:
    """报告页眉: 门禁/聚合视图/结构闸门/工作区门/变异/重规划/提前终止."""
    _con = out.consolidated or {}
    _con_line = f"verdict={_con.get('verdict', 'n/a')}"
    if _con.get("gates_failed"):
        _con_line += f", gates_failed={_con['gates_failed']}"
    return (
        f"# 自主深研(Huginn×书生)\n\n> **门禁: {verdict}** (未落地: {ungrounded or '无'})\n"
        f"> 聚合视图: {_con_line}"
        f"> real orchestration: explored={out.explored} pruned={out.pruned} "
        f"convergence={out.converred}\n> 报告来源: {out.report_source}"
        + (f"\n> 结构对齐闸门: {out.structural_aligned}" if audit is not None else "")
        + (f"\n> 工作区门: {out.workspace_verified}" if workspace is not None else "")
        + (
            f"\n> 进化变异: {out.mutations} 子代; HITL 反馈: {len(out.supervision_log)} 轮"
            if (out.mutations or out.supervision_log)
            else ""
        )
        + (
            f"\n> 层间重规划(P-B): {len(_replan_log)} 实验跳过(预算再分配, 见 out.replan_log)"
            if _replan_log
            else ""
        )
        + (
            f"\n> 证据驱动提前终止(P-C): 层 {_early_stop_meta.get('stopped_after_layer')} 后稳定, "
            f"{len(_early_stop_meta.get('skipped', []))} 实验终止(见 out.consolidated.early_stop)"
            if _early_stop_active
            else ""
        )
        + "\n\n"
    )
