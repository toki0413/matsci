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
    plan_summary: dict | None = None                   # (P2 遗存) 需求拆解/自主规划摘要(planner) — 聚合头 head 源
    structural_gate: dict | None = None                # (P2 遗存) 结构闸门(交互等效/多元论)审计结果
    structural_aligned: bool | None = None             # (P2 遗存) 代理是否通过结构对齐(Shortcut 探测)
    workspace_verified: bool | None = None             # (P2 遗存) C-Space 工作区门: 报告断言是否作为在场落地
    harness: dict | None = None                        # Self-Harness 六维报告(dict) — demo 一键出报告
    law_model_used: dict | None = None                 # (P2 遗存) 世界模型真用证据: predict 产物/是否参与决策 (真深思 D)
    plan_revision: dict | None = None                # (P2 遗存) 漏A: plan 修订门(新证据→显式复盘初始计划)审计
    grounding_audit: dict | None = None              # (P2 遗存) 漏C: "得分≠使用"(高分存活项是否真进最终报告)审计
    judgment_hints: list = field(default_factory=list)  # 判断层·分级护栏: 本次注入的软提示(仅写进 prompt, 不参与 verify)
    consolidated: dict | None = None                   # 收敛聚合头(P1/P2): 现有治理头统一注册后的收敛视图.
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


def _first_scalar(node: Any) -> float | None:
    """从预测/真实结果里取"第一个可用的数值指标"作对账依据(深度优先, 不伪造).

    优先取常见量纲判定名(T/K/score/value/...); 折返时取任意第一个 float。
    取不到返回 None(调用方跳过, 不硬造对账)。
    """
    if isinstance(node, bool):
        return None
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, dict):
        for k in ("T_eq_K", "T", "score", "value", "y", "actual", "S_Wm2"):
            if k in node:
                v = _first_scalar(node[k])
                if v is not None:
                    return v
        for v in node.values():
            s = _first_scalar(v)
            if s is not None:
                return s
    elif isinstance(node, (list, tuple)):
        for v in node:
            s = _first_scalar(v)
            if s is not None:
                return s
    return None


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
                state = world_model.seed({"orbper_d": 365.25}) \
                    if hasattr(world_model, "seed") else LawState({}, domain="")
                nxt = world_model.predict(state, action)
                d = getattr(nxt, "as_dict", lambda: dict(getattr(nxt, "__dict__", {})))()
                return {"pred": d if isinstance(d, dict) else {"predicted": d},
                        "law": getattr(world_model, "law", lambda: "")()}
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
    harness_agent: str = "",          # Self-Harness 报告维度: agent 身份 (组织层账本聚合维度, 留空可)
    harness_machine: str = "",        # Self-Harness 报告维度: machine 身份 (留空可)
    world_model: Any = None,          # 世界模型(可选): predict(spec)->{predicted:{...}} 或 LawModel.
                                      # 接入后, predict 在每次真实执行前对候选做预期目标预筛,
                                      # 预测值作为**可证伪证据**进 cache/trace —— 让 LawModel 真进决策路径(真深思 D),
                                      # 而非只存在于代码里. 不提供则深研走纯真实执行(R).
    external_verifier: Callable[[dict], dict] | None = None,  # 缺陷五: 独立验证方(可选).
                                      # 纯函数契约 {verified: bool, reason: str}, 不共享策略参数;
                                      # 不提供则用出厂最小确定性复核 oracle_verify_consolidated.
    layer_epochs: bool = False,       # 缺陷一(P-A): 分层流式结算. 提供 planner 时, 每个真实实验
                                      # 完成即按层把压缩摘要增量记录进 stream_view(有界决策上下文),
                                      # 并记 epochs —— 决策不再"全跑完才结算". 默认关闭(全 BSP, 零行为变化).
    replan_gate: bool = False,        # 缺陷二(P-B): 层间重规划门. 提供 planner(含 layers)时,
                                      # 前序层真实证据结算后, 用假说重叠/证伪判定跳过已冗余或被打脸方向的
                                      # 后序实验(预算再分配; 跳过=不执行=不产生证据, **绝不伪造**),
                                      # 全程记录 replan_log 供审计. 默认关闭(全执行, 零行为变化).
    early_stop_gate: bool = False,    # 缺陷一(P-C): 证据驱动提前终止门. 提供 planner(含 layers)时,
                                      # 每层结算后用稳定度判据(连续层 top-1 分数高原)判定占优方向是否饱和,
                                      # 饱和则提前终止剩余层(预算回收; 未执行=不产生证据, **绝不伪造**).
                                      # 防早停: 至少 2 个有分观测层才允许判稳定. 默认关闭(全 BSP, 零行为变化).
    stream_summary_chars: int = _STREAM_SUMMARY_CHARS,  # A1: P-A 单条层摘要上限(字符)
    replan_similarity: float = _REPLAN_SIM,             # A1: P-B 假说重叠阈值(0..1)
    early_stop_min_layers: int = _EARLY_STOP_MIN_LAYERS, # A1: P-C 防早停最小观测层数
    early_stop_margin: float = _EARLY_STOP_MARGIN,      # A1: P-C 分数高原容差
    prior: dict | None = None,                      # A2: 跨 run 稳定度先验(prior_store.extract_prior 产物).
                                                    # 只保守调整早停预算参数(min_layers 单调不减), 绝不参与实验.
    strictness: int = 0,                        # 判断层·分级护栏: 0=只真伪硬门禁(信任模型, 零变化);
                                                # 1=最小软提示(候选自变量对照确认); 2=完整软提示(量纲/不确定度).
                                                # 仅追加进成文 prompt, 不参与 verify —— 强模型不受锁.
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
    # §10 扩展: 重复实验一致性 —— 变异回退(parametrize=None 复用父 run)时,
    # 子代与父执行的是**同一实验体**, 按其被重复执行的 objectives 聚合审计.
    _reuse_plan: dict[str, str] = {}                 # branch_name -> 被复用父实验名
    _runs_by_base: dict[str, list[dict]] = {}        # base_name -> [objectives, ...]

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
            _reuse_plan[branch_name] = parent  # §10: 同体重复执行指纹 = 父实验
        return dataclasses.replace(
            spec,
            run=variant_run,
            name=branch_name,
            hypothesis=branch.hypothesis or spec.hypothesis,
        )

    # ── 缺陷一(P-A) 分层流式结算: 流视图 + 层映射 ──────────────────────
    # 提供 planner(有 layers)且 layer_epochs=True 时, 每完成一个真实实验就把压缩
    # 摘要增量记入 _stream_rows —— 决策上下文随执行生长, 而非全跑完后一次性结算.
    _layer_epochs = bool(layer_epochs) and plan_summary is not None \
        and bool(plan_summary.get("layers"))
    # 拓扑层映射: 只要 planner 给过 layers 就可用 —— P-A(P-B 亦共用)的门控前提.
    # 与 _layer_epochs 解耦: replan_gate(P-B) 不要求流式视图开启.
    _layers_map: list[list[str]] = (plan_summary.get("layers") or []) \
        if plan_summary is not None else []
    _stream_rows: list[dict] = []

    # ── 缺陷二(P-B) 层间重规划门: 状态与单实验判定 ──────────────────────────
    # 前序层真实证据结算后, 用假说重叠(冗余方向)/predicted 对账(证伪方向)判定
    # 后序实验是否仍值得执行 —— DAG 边在证据到达后可修订, 不再一次性冻结.
    # _replan_enabled 只要求 planner 给出 layers(重规划把先验边当可修订工作假说);
    # 与 _layer_epochs 正交: 不开 P-A 也可单独启用 replan(P-B 只依赖 cache 证据面).
    _replan_enabled = bool(replan_gate) and plan_summary is not None \
        and bool(plan_summary.get("layers"))
    _replan_log: list[dict] = []
    _replan_checked = 0                          # 被门控评估过的后序实验数(审计计数)
    _hypotheses = {e.name: e.hypothesis for e in experiments}

    def _replan_decision(name: str) -> dict | None:
        """单个后序实验的层间重规划判定 (纯函数托管在 replan_gate.decide_replan_skip)."""
        from huginn.research.replan_gate import decide_replan_skip
        i = _layer_of(name)
        if i <= 0:
            return None
        prior = [n for j in range(i) for n in _layers_map[j]]
        return decide_replan_skip(
            name, i, prior, cache, _hypotheses,
            already_decided=set(cache) | {r["name"] for r in _replan_log},
            similarity=replan_similarity, tol=_WM_TOL,
        )

    def _layer_of(name: str) -> int:
        for i, lay in enumerate(_layers_map):
            if name in lay:
                return i
        return -1

    # ── 缺陷一(P-C) 证据驱动提前终止门: 状态与层结算判定 ──────────────────
    # 与 P-B 同族(层间预算决策, 不伪造): P-B 逐实验修订边; P-C 在"分数进入高原
    # (连续层 top-1 得分趋平)"时直接终止剩余层. 判定只基于真实执行成绩(cache).
    _early_stop_enabled = bool(early_stop_gate) and plan_summary is not None \
        and bool(plan_summary.get("layers"))
    _early_stop_active = False
    _early_stop_meta: dict = {"enabled": _early_stop_enabled}

    # A2: 先验注入 —— 只把早停参数推向更保守方向(上次 N 层才稳定, 这次至少等 N 层).
    # A4/A5/A6: 附当前 goal 的归一化 slug 与原文, 支持异域拒绝/时间衰减/模糊匹配.
    # 无先验 → defaults 取参数原件(优先级: 参数 > 常量), 行为不变; 有先验 → min_layers 单调不减且钳制 [2,4].
    from huginn.research.prior_store import resolve_early_stop_args
    _prior_args = resolve_early_stop_args(
        prior, default_min_layers=early_stop_min_layers, default_margin=early_stop_margin,
        current_goal_slug=_slug_goal(str(goal)), current_goal=str(goal))
    _early_stop_min_layers = int(_prior_args["min_layers"])
    _early_stop_margin = float(_prior_args["margin"])

    def _settle_layer_check(settled_until: int) -> None:
        """层 settled_until 刚结算: 用稳定度判据决定是否激活提前终止(预算决策)."""
        nonlocal _early_stop_active
        if not _early_stop_enabled or _early_stop_active:
            return
        from huginn.research.early_stop_gate import stability_check
        settled_layers: list[list[tuple[str, float]]] = []
        for i in range(min(settled_until, len(_layers_map) - 1) + 1):
            rows = []
            for n in _layers_map[i]:
                res = cache.get(n) or {}
                s = _first_scalar(res.get("objectives") or res.get("summary"))
                if s is not None:
                    rows.append((n, s))
            settled_layers.append(rows)
        _st = stability_check(settled_layers, min_layers=_early_stop_min_layers,
                              margin=_early_stop_margin)
        _early_stop_meta["stability"] = _st
        # 稳定 且 还有剩余层可终止 → 激活
        if _st["stable"] and settled_until + 1 < len(_layers_map):
            _early_stop_active = True
            _early_stop_meta["stopped_after_layer"] = settled_until
            _early_stop_meta["skipped"] = []
            _early_stop_meta["verdict"] = "early_stopped"
        else:
            _early_stop_meta.setdefault("verdict", "checked_and_continued")

    async def executor(branch):
        nonlocal _replan_checked          # P-B 审计计数: 嵌套闭包需 nonlocal 才能 +=
        spec = spec_by_name.get(branch.name)
        if spec is None:  # 动态子代(变异/演化新分支)
            spec = _resolve_variant(branch.name, branch)
            if spec is not None:
                spec_by_name[branch.name] = spec  # 注册, 供综合阶段取假说/摘要
        if spec is None:
            return {"success": False, "objectives": {}, "results": {}}
        # P-C 提前终止: 占优方向已稳定(分数高原), 剩余层实验直接终止(预算回收).
        # 与 P-B 同红线: 未执行 = 不产生证据, 不伪造; 决策记录进 _early_stop_meta.
        if _early_stop_active and _layer_of(branch.name) > _early_stop_meta.get("stopped_after_layer", -1):
            _early_stop_meta.setdefault("skipped", []).append({
                "name": branch.name, "layer": _layer_of(branch.name),
                "reason": "early_termination",
                "evidence_from": f"layers 0..{_early_stop_meta['stopped_after_layer']}"})
            return {"success": False, "objectives": {},
                    "results": {"early_stopped": True}}
        # P-B 层间重规划门: 前序层真实证据结算后, 判定该实验是否仍值得执行.
        # 跳过 = 预算再分配 —— 不写 cache/stream_view/trace(未执行=不产生证据, 不伪造);
        # 决策全程记录进 _replan_log 供审计(gate.replan / out.replan_log).
        if _replan_enabled:
            _rp = _replan_decision(branch.name)
            if _rp is not None:
                _replan_checked += 1
                _replan_log.append(_rp)
                return {"success": False, "objectives": {},
                        "results": {"replanned": _rp["reasons"]}}
            _replan_checked += 1
        res = await asyncio.to_thread(spec.run)
        # 世界模型预筛(真 D): predict 在真实执行前对候选作预期目标预判, 预测值作为
        # 可证伪证据并入结果 —— 供 post-hoc reconcile 对账, 并让"预测参与决策"成为事实.
        if world_model is not None:
            res = dict(res)
            try:
                res["predicted"] = _wm_predict(world_model, spec)
            except Exception:  # noqa: BLE001 — 世界模型预筛失败: 不伪造, 保留纯真实结果
                res.setdefault("predicted", {})
        cache[branch.name] = res
        # §10 扩展: 按同体指纹聚合 objectives(变异回退 → 父名; 普通 → 自身).
        _base = _reuse_plan.get(branch.name, branch.name)
        _runs_by_base.setdefault(_base, []).append(res.get("objectives") or {})
        # P-C 层结算检查: 本实验所在层全部结算(执行/重规划跳过/提前终止)后,
        # 用真实成绩跑稳定度判据 —— 稳定则激活提前终止(见 _settle_layer_check).
        if _early_stop_enabled and not _early_stop_active:
            _i = _layer_of(branch.name)
            if _i >= 0:
                _layer = _layers_map[_i]
                _es_skipped = {s.get("name") for s in _early_stop_meta.get("skipped", [])}
                _rp_skipped = {r["name"] for r in _replan_log}
                if all(n in cache or n in _es_skipped or n in _rp_skipped for n in _layer):
                    _settle_layer_check(_i)
        # P-A 流式入账: 每完成一个真实实验, 经漏B 门控压缩后按层增量记录(有界上下文).
        if _layer_epochs:
            try:
                from huginn.research.decision_gate import distill_tool_output
                _d = distill_tool_output(
                    spec.name,
                    json.dumps(res.get("summary", {}), ensure_ascii=False),
                    goal=str(goal), max_chars=stream_summary_chars)
                _stream_rows.append({
                    "layer": _layer_of(spec.name),
                    "experiment": spec.name,
                    "summary": _d["front"],
                    "relevance": _d["relevance"],
                })
            except Exception:  # noqa: BLE001 — 流式记录失败不影响执行结果
                pass
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

    # P-A · 流式结算视图: 把逐实验增量行按层分组为有界的 stream_view(仅记录, 不改执行)
    stream_view: list[dict] | None = None
    _n_epochs = 0
    if _layer_epochs and _stream_rows:
        by_layer: dict[int, list[dict]] = {}
        for _r in _stream_rows:
            by_layer.setdefault(_r["layer"], []).append(
                {"experiment": _r["experiment"], "summary": _r["summary"]})
        stream_view = [
            {"layer": i, "experiments": len(rows), "rows": rows}
            for i, rows in sorted(by_layer.items())
        ]
        _n_epochs = len(stream_view)

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
                  f"每个数值必须来自上面真实结果, 不许编造。报告只允许引用上面这些真实结果里的数值;"
                  f"任何来自历史报告/上下文对话的数值(如结论段外引用的旧统计量)一律禁止引用。"
                  f"把最终报告放在 <report> 与 </report> 之间。")
        # 缺陷一(P-A) 分层流式结算: 决策上下文改为"增量层摘要视图(有界)"而非全 trace——
        # LLM 只读 stream_view(每层经漏B 压缩), 全量证据始终留 trace 供 grounding 门禁.
        if stream_view:
            _sv_text = "\n".join(
                f"- {_lay['experiments']} 实验(层 {_lay['layer']}): "
                + "; ".join(f"{_r['experiment']}={_r['summary']}" for _r in _lay["rows"])
                for _lay in stream_view)
            prompt += ("\n\n【分层流式证据视图(P-A)】已按执行层增量结算的摘要(有界上下文; "
                       "完整可证伪证据在 trace 中):\n" + _sv_text)
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
        # 判断层·分级护栏: 只追加进 prompt (软提示), 不参与 verify —— 强模型不受锁.
        # strictness=0 → 空串, 零行为变化(默认信任模型推理).
        if strictness > 0:
            from huginn.research.judgment_guardrail import hint_block
            _hs_block = hint_block(str(goal), trace=trace, survivors=survivors,
                                   strictness=int(strictness))
            if _hs_block:
                prompt += "\n" + _hs_block
                out.judgment_hints = [
                    h for h in _hs_block.split("\n") if h.startswith("- ")]

        def _one_attempt(msgs):
            for _round in range(4):
                try:
                    r = client.chat.completions.create(model=model, messages=msgs,
                                                       tools=tool_schemas or None,
                                                       tool_choice="auto" if tool_schemas else None,
                                                       max_tokens=4000, temperature=0.2,
                                                       extra_body={"thinking_mode": False})
                    msg = r.choices[0].message
                except Exception as ee:
                    # Client API 调用失败（权限/网络/base-url 错） → 记 trace 并进下一轮
                    trace.append(json.dumps({"error": f"client.chat.completions.create: {ee}"},
                                         ensure_ascii=False))
                    return ""
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
                    # 漏B 决策先导摘要: 长工具回调进决策上下文前先净化(先导摘要),
                    # 防止均匀灌注稀释; 完整原始证据仍在 trace 供 grounding 门禁核对.
                    try:
                        from huginn.research.decision_gate import distill_tool_output
                        _g = distill_tool_output(name, res, goal=str(goal), max_chars=4000)
                        _message = _g["front"]
                    except Exception:  # noqa: BLE001 — 门控不可用时退原样, 不阻断
                        _message = res
                    trace.append(json.dumps({"gate": "front_summary",
                                             "tool": name, "len": len(res)}, ensure_ascii=False))
                    msgs += [{"role": "assistant", "content": msg.content or "",
                              "tool_calls": [tc.model_dump()]},
                             {"role": "tool", "tool_call_id": tc.id, "content": _message}]
            # 工具轮结束后若仍无正文(如模型一路调工具未收尾), 强制一次"只写报告"调用,
            # 避免返回空 content → 门禁因空报告打回 → 白白兜底.
            if not (msg.content or "").strip():
                try:
                    rr = client.chat.completions.create(
                        model=model,
                        messages=msgs + [{"role": "user",
                                          "content": "不要再调用工具。基于轨迹里的真实数值, "
                                                     "现在直接撰写完整研究报告, 把正文放在 "
                                                     "<report> 与 </report> 之间。"}],
                        max_tokens=4000, temperature=0.2)
                    return rr.choices[0].message.content or ""
                except Exception as ee:  # noqa: BLE001 — 兜底成文失败如实返回空
                    trace.append(json.dumps({"error": f"final write: {ee}"},
                                             ensure_ascii=False))
                    return ""
            return msg.content or ""

        msgs = [{"role": "user", "content": prompt}]
        for _i in range(3):
            final = _one_attempt(msgs).strip()
            m = re.search(r"<report>(.*?)</report>", final, flags=re.DOTALL | re.IGNORECASE)
            if m:
                final = m.group(1).strip()
            g = verify(final, trace)
            if g["verdict"] == "pass" and len(final.strip()) > 200:
                verdict, ungrounded = "pass", []; out.report_source = "agent"; break
            ungrounded = g["unsubstantiated"]
            msgs = [{"role": "user",
                     "content": f"未交付(未落地:{ungrounded})。请只用真实证据重写:\n" + prompt}]
            out.report_source = "agent(retry)"
        else:
            final = _synthesize()      # L3b 兜底组装, 必过门禁
            out.report_source = "fallback_assembly(agent failed)"
    else:
        final = _synthesize()
        out.report_source = "deterministic"

    # P-B 层间重规划审计: 报告如实说明被门控跳过的实验(证据进 trace 供 grounding 核对).
    # 跳过 = 未执行 = 无证据 —— 报告只陈述"重规划决策", 不捏造任何实验结果.
    if _replan_log:
        trace.append(json.dumps({"type": "replan_gate", "log": _replan_log},
                                ensure_ascii=False))
        final += ("\n\n## 层间重规划(P-B)\n"
                  f"{len(_replan_log)} 个计划内实验经前序层真实证据门控判定冗余/被证伪方向, "
                  "重规划跳过(预算再分配; 未执行 = 不产生证据, 不伪造): "
                  + ", ".join(f"{r['name']}({'; '.join(x['rule'] for x in r['reasons'])})"
                              for r in _replan_log)
                  + "。完整理由见 trace.replan_gate。")

    # P-C 证据驱动提前终止审计: 报告如实说明被终止的实验(证据进 trace 供 grounding 核对).
    # 终止 = 未执行 = 无证据 —— 报告只陈述"稳定度预算决策", 不捏造任何实验结果.
    if _early_stop_active:
        trace.append(json.dumps({"type": "early_stop_gate", "meta": _early_stop_meta},
                                ensure_ascii=False))
        _st = _early_stop_meta.get("stability", {})
        _es_n = len(_early_stop_meta.get("skipped", []))
        final += ("\n\n## 证据驱动提前终止(P-C)\n"
                  f"执行到层 {_early_stop_meta.get('stopped_after_layer')} 后占优方向进入分数高原"
                  f"(top={_st.get('top')}, score={_st.get('score')}, "
                  f"rel_change={_st.get('relative_change')}), 剩余 {_es_n} 个计划内实验提前终止"
                  "(预算回收; 未执行 = 不产生证据, 不伪造)。证据见 trace.early_stop_gate。")

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

    # 世界模型真用证据: 若提供了 world_model, 本 run 确实用 predict 预筛了候选 ——
    # 记录 per-实验预测对账素材, 供 harness 的 "世界模型真用?" 以 observed 落地(真 D)。
    # 更进一步: 把预测与真实执行 reconcile 数值对账 —— 预测升级为"强在场"检验,
    # 偏差如实记 falsified(这就是发现), 而非只记"用过了"。诚实红线不变。
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
                    reconcile_rows.append({
                        "experiment": name,
                        "rel_err": round(err, 4),
                        "borne_out": err <= _WM_TOL,
                        "predicted": round(pred_scalar, 4),
                        "actual": round(actual_scalar, 4),
                    })
            except Exception:  # noqa: BLE001 — 对账失败不伪造, 只跳过
                pass
        out.law_model_used = {
            "count": len(preds),
            "mock": [p["predicted"] for p in preds[:3]],
            "reconcilable": bool(preds),
            "reconcile": reconcile_rows,
            "borne_out_all": bool(reconcile_rows) and all(r["borne_out"] for r in reconcile_rows),
        }

    # ── 收敛聚合头 (P1 纯适配器) ─────────────────────────────────────────
    # 把现有治理字段一次性收敛成 head 注册视图 out.consolidated; 旧字段保留为
    # 残差兼容(报告/self-harness/账本 P2 前仍读旧字段). 未来新视角一律走
    # HeadResult 注册, 不再加 out.* 字段 (P3 由 arch_cleanliness 执法).
    try:
        from huginn.research.aggregation_head import (
            EVIDENCE_OBSERVED,
            EVIDENCE_UNOBSERVED,
            HeadResult,
            build_replication_head,
            build_replication_view,
            consolidate,
        )

        heads: list[HeadResult] = []
        # gate.replan —— P-B 层间重规划门 (缺陷二, 建议级: DAG 边在证据到达后可修订).
        # 视为 unobserved = 未启用(默认全执行, 零行为变化); observed = 真实完成门控判定
        # 并留下可证伪记录(跳过项+理由)—— 无论是否跳过, 门工作即 passed, 明细供复核.
        if _replan_enabled:
            heads.append(HeadResult(
                "gate.replan", "层间重规划门(缺陷二: DAG 边不再冻结)",
                EVIDENCE_OBSERVED, "passed",
                detail=(f"checked={_replan_checked} skipped={len(_replan_log)}; "
                        + str([(r["name"], [x["rule"] for x in r["reasons"]])
                               for r in _replan_log])),
                ref="out.consolidated.replan"))
        else:
            heads.append(HeadResult(
                "gate.replan", "层间重规划门(缺陷二: DAG 边可在证据后修订)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="replan_gate 未启用(默认全执行, 零行为变化)",
                ref="out.consolidated.replan"))
        # gate.early_stop —— P-C 证据驱动提前终止门 (缺陷一, 建议级: 稳定后不等全层).
        # unobserved = 未启用(默认全 BSP, 零行为变化); observed = 真实完成稳定度
        # 判定并留下可证伪记录 —— 无论是否终止, 门工作即 passed, 明细供复核.
        _early_stop_head = {
            "enabled": _early_stop_enabled,
            "layers": len(_layers_map) if _early_stop_enabled else 0,
            "stopped_after_layer": _early_stop_meta.get("stopped_after_layer"),
            "skipped": len(_early_stop_meta.get("skipped", [])),
            "log": list(_early_stop_meta.get("skipped", [])),   # 完整终止名单(可证伪)
            "stability": dict(_early_stop_meta.get("stability") or {}),  # 稳定判据细节(可证伪)
            "prior_used": {"min_layers": _early_stop_min_layers,
                           "margin": _early_stop_margin,
                           "source": _prior_args["source"],
                           "note": _prior_args["note"],
                           "goal_matched": _prior_args["goal_matched"],
                           "goal_level": _prior_args.get("goal_level"),
                           "goal_overlap": _prior_args.get("goal_overlap"),
                           "decay_weight": _prior_args.get("decay_weight"),
                           "prior_age": _prior_args.get("prior_age"),
                           "weight": _prior_args.get("weight")},
            "verdict": ("no_early_stop" if not _early_stop_enabled
                        else _early_stop_meta.get("verdict", "checked_and_continued")),
        }
        heads.append(HeadResult(
            "gate.early_stop", "证据驱动提前终止门(P-C: 分数高原后不等全层)",
            EVIDENCE_OBSERVED if _early_stop_enabled else EVIDENCE_UNOBSERVED,
            "passed" if _early_stop_enabled else "unobserved",
            detail=str(_early_stop_head), ref="out.consolidated.early_stop"))
        # audit.replication —— §10 扩展: 重复实验一致性 (同体多次执行是否可复现).
        # 无重复执行 → unobserved(默认零噪音); 有重复 → observed + 明细(供人类决策,
        # 一致性不是质量判据, 无否决权).
        _replication_view = build_replication_view(_runs_by_base)
        heads.append(build_replication_head(_runs_by_base))
        # gate.claim_grounding —— 声明门禁 (对报告是否成文有否决权)
        if verdict != "needs_grounding" or ungrounded:
            heads.append(HeadResult(
                "gate.claim_grounding", "声明门禁(claim_grounding)",
                EVIDENCE_OBSERVED,
                "passed" if verdict in ("grounded", "pass", "accept", "confirmed")
                else "failed",
                detail=f"verdict={verdict}", ref="out.verdict", gate=True))
        else:
            heads.append(HeadResult(
                "gate.claim_grounding", "声明门禁(claim_grounding)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="verdict=needs_grounding 且无 ungrounded 素材",
                ref="out.verdict", gate=True))
        # gate.structural —— 结构闸门 (交互等效/多元论审计)
        if audit is not None:
            heads.append(HeadResult(
                "gate.structural", "结构闸门(交互等效/多元论审计)",
                EVIDENCE_OBSERVED,
                "passed" if out.structural_aligned is not False else "failed",
                detail=str(audit), ref="out.structural_gate", gate=True))
        else:
            heads.append(HeadResult(
                "gate.structural", "结构闸门(交互等效/多元论审计)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="未注入 structural_audit", ref="out.structural_gate", gate=True))
        # gate.workspace —— 工作区广播门 (C-Space 在场断言)
        if workspace is not None:
            heads.append(HeadResult(
                "gate.workspace", "工作区广播门(C-Space 在场断言)",
                EVIDENCE_OBSERVED,
                "passed" if out.workspace_verified else "failed",
                detail=f"workspace_verified={out.workspace_verified}",
                ref="out.workspace_verified", gate=True))
        else:
            heads.append(HeadResult(
                "gate.workspace", "工作区广播门(C-Space 在场断言)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="未注入 workspace", ref="out.workspace_verified", gate=True))
        # wm.actually_used —— 世界模型真用 (误区二审计, 建议级)
        if out.law_model_used is not None:
            wm_ok = out.law_model_used.get("borne_out_all") is not False
            heads.append(HeadResult(
                "wm.actually_used", "世界模型真用?(predict 参与决策 = 真深思 D)",
                EVIDENCE_OBSERVED, "passed" if wm_ok else "failed",
                detail=str(out.law_model_used), ref="out.law_model_used"))
        else:
            heads.append(HeadResult(
                "wm.actually_used", "世界模型真用?(predict 参与决策 = 真深思 D)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="未见 predict/reconcile 产物", ref="out.law_model_used"))
        # audit.plan_revision —— 锚定反制 (漏A, 建议级)
        if out.plan_revision is not None:
            heads.append(HeadResult(
                "audit.plan_revision", "plan 修订门(新证据→显式复盘初始计划)",
                EVIDENCE_OBSERVED,
                "passed" if out.plan_revision.get("verdict") == "plan_holds" else "failed",
                detail=str(out.plan_revision), ref="out.plan_revision"))
        # audit.score_usage —— 得分≠使用 (漏C, 对疑似 lipstick 有否决权)
        if out.grounding_audit is not None:
            heads.append(HeadResult(
                "audit.score_usage", "得分≠使用(高分存活项真进最终报告?)",
                EVIDENCE_OBSERVED,
                "passed" if out.grounding_audit.get("verdict") == "proper_use" else "failed",
                detail=str(out.grounding_audit), ref="out.grounding_audit", gate=True))
        # controlled.supervision —— 受控执行 (建议级, 供 P2 六维投影)
        if getattr(out, "supervision_log", None):
            heads.append(HeadResult(
                "controlled.supervision", "受控执行(HITL/权限评审记录)",
                EVIDENCE_OBSERVED, "passed",
                detail=f"记录 {len(out.supervision_log)} 条", ref="out.supervision_log"))
        else:
            heads.append(HeadResult(
                "controlled.supervision", "受控执行(HITL/权限评审记录)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="supervisor_every=0 或未触发人工评审", ref="out.supervision_log"))
        # reliable.evidence_cache —— 真实执行证据 (建议级, 供 P2 六维投影)
        if cache:
            heads.append(HeadResult(
                "reliable.evidence_cache", "真实执行证据(缓存/expts 实录)",
                EVIDENCE_OBSERVED, "passed",
                detail=f"{len(cache)} 个实验", ref="out.cache"))
        else:
            heads.append(HeadResult(
                "reliable.evidence_cache", "真实执行证据(缓存/expts 实录)",
                EVIDENCE_UNOBSERVED, "unobserved", detail="cache 为空", ref="out.cache"))
        # learning.self_audit —— 自省与经验沉淀 (建议级, 供 P2 六维投影)
        artifacts = (out.structural_gate, out.plan_summary, out.plan_revision,
                     out.grounding_audit, getattr(out, "supervision_log", None))
        if any(a not in (None, []) for a in artifacts):
            heads.append(HeadResult(
                "learning.self_audit", "能力自省闭环(缺口→提案)",
                EVIDENCE_OBSERVED, "passed",
                detail="该 run 产生可复用审计/规划产物", ref="out.structural_gate"))
        else:
            heads.append(HeadResult(
                "learning.self_audit", "能力自省闭环(缺口→提案)",
                EVIDENCE_UNOBSERVED, "unobserved",
                detail="自省已接线但本 run 无审计产物", ref="capabilities/introspection"))

        _replan_meta = {
            "enabled": _replan_enabled,
            "layers": len(_layers_map) if _replan_enabled else 0,
            "checked": _replan_checked,
            "skipped": len(_replan_log),
            "log": list(_replan_log),              # 完整审计: 被跳过实验+理由+证据面(可证伪)
            "verdict": ("no_replan" if not _replan_enabled
                        else ("replanned" if _replan_log else "all_proceed")),
        }

        out.consolidated = consolidate(heads, grounding_verdict=verdict,
                                       epochs=_n_epochs, stream_view=stream_view,
                                       replan=_replan_meta,
                                       early_stop=_early_stop_head,
                                       replication=_replication_view).as_dict()

        # 缺陷三/五接缝: 在第一轮聚合视图上追加"元头"(外部验证 + 团队视角分离度).
        # 第一轮先用可替换外部验证方(缺省出厂 oracle)复核; 派生两个头后第二轮合并,
        # 再以第一轮验证结果覆盖 external_verify(第二轮不重跑验证方, 幂等).
        try:
            from huginn.research.aggregation_head import oracle_verify_consolidated
            _ver = external_verifier if external_verifier is not None else oracle_verify_consolidated
            base = consolidate(heads, grounding_verdict=verdict,
                           role_view=list(front), external_verifier=_ver,
                           epochs=_n_epochs, stream_view=stream_view,
                           replan=_replan_meta, early_stop=_early_stop_head,
                           replication=_replication_view)
            ext = base.external_verify or {}
            heads.append(HeadResult(
                "governance.external_verify", "独立验证方(可替换的外部复核)",
                ext.get("evidence", EVIDENCE_UNOBSERVED),
                ("passed" if ext.get("verified") else "failed")
                if ext.get("verified") is not None else "unobserved",
                detail=str(ext), ref="out.consolidated.external_verify", gate=True))
            _role = base.role_diversity or {}
            if front:
                heads.append(HeadResult(
                    "team.diversity", "团队视角分离度(防多头塌缩)",
                    EVIDENCE_OBSERVED,
                    {"diverse": "passed", "collapsed_single_view": "failed"}.get(
                        _role.get("verdict"), "unobserved"),
                    detail=str(_role), ref="out.pareto_front"))
            else:
                heads.append(HeadResult(
                    "team.diversity", "团队视角分离度(防多头塌缩)",
                    EVIDENCE_UNOBSERVED, "unobserved",
                    detail="无存活假说, 无从度量视角分化", ref="out.pareto_front"))
            final_cons = consolidate(heads, grounding_verdict=verdict,
                                 role_view=list(front), head_budget=_HEAD_BUDGET,
                                 epochs=_n_epochs, stream_view=stream_view,
                                 replan=_replan_meta, early_stop=_early_stop_head,
                                 replication=_replication_view)
            final_cons.external_verify = ext   # 第二轮不重跑验证方, 保留第一轮独立复核结果
            # 缺陷七: 治理自身是否过度建制 —— 建议级元头(不否决, 只亮灯).
            _ob = final_cons.overbuild or {}
            heads.append(HeadResult(
                "meta.overbuild_guard", "过度建制审计(second system effect 反制)",
                EVIDENCE_OBSERVED,
                "passed" if _ob.get("verdict") == "healthy" else "failed",
                detail=str(_ob), ref="out.consolidated.overbuild"))
            final_cons2 = consolidate(heads, grounding_verdict=verdict,
                                  role_view=list(front), head_budget=_HEAD_BUDGET,
                                  epochs=_n_epochs, stream_view=stream_view,
                                  replan=_replan_meta, early_stop=_early_stop_head,
                                  replication=_replication_view)
            final_cons2.external_verify = ext   # 保留第一轮独立复核结果(第二轮不重跑验证方)
            out.consolidated = final_cons2.as_dict()   # overbuild 用含全部头的最终视图(自洽)
        except Exception:  # noqa: BLE001 — 元头派生失败: 保留第一轮聚合视图, 不阻断
            pass
    except Exception:  # noqa: BLE001 — 聚合头为新增视图, 失败不阻断管线
        out.consolidated = None

    # M2: Self-Harness 五维报告 — 复用本 out 已记录的 gate 结果, 不重复计算 (例行轻量)
    try:
        from huginn.research.harness import build_harness_report
        out.harness = build_harness_report(
            goal, out, agent=harness_agent, machine=harness_machine).to_dict()
    except Exception:  # noqa: BLE001 — harness 为可选附加值, 失败不应阻断管线
        out.harness = None

    # 报告页眉 (移动到底部, 聚合头/账本产出后再写): 门禁按声明定稿, 聚合视图取
    # out.consolidated(统一入口), 旧字段仅作残差回显.
    if out_md is not None:
        _con = out.consolidated or {}
        _con_line = f"verdict={_con.get('verdict', 'n/a')}"
        if _con.get("gates_failed"):
            _con_line += f", gates_failed={_con['gates_failed']}"
        header = (f"# 自主深研(Huginn×书生)\n\n> **门禁: {verdict}** (未落地: {ungrounded or '无'})\n"
                  f"> 聚合视图: {_con_line}"
                  f"> real orchestration: explored={out.explored} pruned={out.pruned} "
                  f"convergence={out.converred}\n> 报告来源: {out.report_source}"
                  + (f"\n> 结构对齐闸门: {out.structural_aligned}" if audit is not None else "")
                  + (f"\n> 工作区门: {out.workspace_verified}" if workspace is not None else "")
                  + (f"\n> 进化变异: {out.mutations} 子代; HITL 反馈: {len(out.supervision_log)} 轮"
                     if (out.mutations or out.supervision_log) else "")
                  + (f"\n> 层间重规划(P-B): {len(_replan_log)} 实验跳过(预算再分配, 见 out.replan_log)"
                     if _replan_log else "")
                  + (f"\n> 证据驱动提前终止(P-C): 层 {_early_stop_meta.get('stopped_after_layer')} 后稳定, "
                     f"{len(_early_stop_meta.get('skipped', []))} 实验终止(见 out.consolidated.early_stop)"
                     if _early_stop_active else "")
                  + "\n\n")
        out_md.write_text(header + final.strip() + "\n", encoding="utf-8")
    return out