"""RCB Step 2 执行 — 依赖 rcb_utils / rcb_cognition / rcb_audit."""
from __future__ import annotations

import contextlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from huginn.cli.rcb.audit import _rcb_drift_check
from huginn.cli.rcb.prompt_builders import _legacy_build_step2_prompt
from huginn.cli.rcb_audit import (
    _ChecklistItem,
    _checklist_item_parser,
    _derivation_chain_audit,
    _llm_coverage_audit,
    _rcb_effort_floor,
    _report_coverage_compass,
    _time_slot_index,
)
from huginn.cli.rcb_fork_merge import (
    _FCM_PERSPECTIVES,
    anneal_fork_count,
    judge_fork_reports,
)
from huginn.cli.rcb_cognition import (
    _append_observations_log,
    _collect_observations,
    _compute_v15_fields,
    _init_hypothesis_manifold,
    _record_abduction,
    _trigger_anomaly_hypothesis,
    _write_cognitive_evidence,
)
from huginn.cli.rcb_utils import (
    _detect_file_rewrite_stagnation,
    _infer_domain,
    _infer_task_id_from_workspace,
    _make_simplex_id,
    _MODEL_VERSION,
    _save_manifold,
    _cross_task_store,
)
from huginn.utils.runtime import HUGINN_DIR_NAME, get_runtime_home

logger = logging.getLogger(__name__)


# v14 Task 2: darwin_score 真实计算 (StepEvaluator gap_severity 反向打分).
# ponytail: top-level try-except 跟 line 599 defensive 模式一致 — step_evaluator
#   依赖较重, import 失败回退 0.5 (天花板: 全 0.5 不区分, 升级路径: 修 import).
try:
    from huginn.metacog.step_evaluator import _compute_darwin_score
except Exception:  # pragma: no cover
    def _compute_darwin_score(_step_eval: Any) -> float:
        return 0.5


class _RCBStep2Ctx:
    """Step 2 执行循环的上下文 — 核心对象 + Step 1 产物 + 闭包."""
    ws: Path
    model: Any
    agent: Any
    kb: Any
    mem_mgr: Any
    persona: Any
    kg: Any
    thread_id: str
    task_id: str
    resume_from_iter: int
    extreme: bool
    checklist: str
    scan_text: str
    fcm: dict
    target_chains: list
    instructions: Any
    stream_chat_fn: Any
    rcb_csm_advance_fn: Any
    # P5 挂钟预算守卫: run() 创建 goal 后传入, _step2_execute 每轮查.
    # None 表示未开 P5, 守卫总返 False (不阻断).
    wall_expired_fn: Any = None
    # persona_name: run() 从 ws.name 推断后传入, _step2_execute 写 iteration_result 用.
    persona_name: str = "default"
    # P3-7: run 开始的 ISO 时间戳, 传给 build_pmk_state 的 since 参数,
    # 限定 KB query / Memory recall 只看本 run 之后的新知识.
    run_start_iso: str | None = None


async def _step2_execute(ctx: _RCBStep2Ctx) -> list:
    """Step 2: 执行任务 — setup + 迭代执行循环.

    从 run() 抽出, 680 行逻辑保持不变. 闭包 _stream_chat / _rcb_csm_advance
    通过 ctx 传入, 其他 setup 初始化的变量是函数局部变量.
    返回 _evals_history 供 Step 3 的 _step3_adversarial 使用.
    """
    ws = ctx.ws
    model = ctx.model
    agent = ctx.agent
    kb = ctx.kb
    _mem_mgr = ctx.mem_mgr
    persona = ctx.persona
    _kg = ctx.kg
    thread_id = ctx.thread_id
    _task_id = ctx.task_id
    # v14 Task 1: trace 里存短 task_id (Astronomy_000), 不是 ws.name (带时间戳).
    _trace_task_id = _infer_task_id_from_workspace(_task_id)
    # v14 Task 15: 跨 task darwin prior — 取本 domain 历史 high darwin entry,
    # 传给 HintCoordinator boost 当前 hint. 跨 domain 隔离在 CrossTaskStore 层做.
    _trace_domain = _infer_domain(_trace_task_id)
    _cross_task_prior_entries = (
        _cross_task_store.query_high_darwin(domain=_trace_domain, top_k=5)
        if _cross_task_store is not None else []
    )
    _resume_from_iter = ctx.resume_from_iter
    extreme = ctx.extreme
    checklist = ctx.checklist
    scan_text = ctx.scan_text
    fcm = ctx.fcm
    _target_chains = ctx.target_chains
    instructions = ctx.instructions

    # Step 2: 执行任务 (v7 P3: 迭代执行 + Meta-Trace 蒸馏)
    # checklist 已在 thread_id 的对话历史里, agent 能看到. 不需要显式注入.
    #
    # 对标 Oxelra 206 步: 单次 chat() 已能跑 150-300 tool calls (langgraph 内部循环),
    # 但单次 chat() 会因 context 溢出或 agent 主动 emit text-only 提前终止.
    # 迭代执行让 agent 在多次 chat() 间累积进展, 每轮间写 Meta-Trace entry,
    # 下一轮 chat() 的 build_meta_trace_text (P1) 会读回来注入 prompt,
    # 同时 compaction 因 trace 存在会更激进 drop raw messages.
    #
    # ponytail: 不接 AutoloopEngine (它用 CoderRunner/WorkflowEngine, 不写
    #   report/report.md, 会破坏 RCBench 评分). 用 mini-loop + 手写 trace.
    #   升级路径: full AutoloopEngine.run_cognitive() + 自定义 report writer.
    print("\n=== Step 2: Execution (iterative) ===\n", flush=True)
    ctx.rcb_csm_advance_fn("user_confirmed", {"plan": "execute methodology checklist"})
    # _scan_hint 按 verifiable_via 分档:
    # - hard_check (dimensional/exact_formula/conservation_law): 必须验证, 违反则 debug
    # - soft_check (asymptotic/symmetry/topological): 建议验证, 违反 warn 不 block
    # - none/empirical: 不约束, 按数值精度处理
    # 这呼应物理 precheck "警告 + force_proceed" 偏好 — 结构违反先 warn, 不强制拦截.
    if scan_text and scan_text.strip():
        _scan_hint = (
            f"\n\n## Intuitive Gamer + Math Structure Scan (Step 1.5 result)\n{scan_text}\n\n"
            f"## Execution Guidance\n"
            f"Follow the STRATEGY line above: hard_check items first (bank structural wins),\n"
            f"then soft_check, then empirical/none last.\n\n"
            f"## Invariant Self-Check (per item)\n"
            f"- hard_check (dimensional/exact_formula/conservation_law): result MUST satisfy the invariant.\n"
            f"  Violation → debug and fix, do NOT silent-substitute. This is non-negotiable.\n"
            f"- soft_check (asymptotic/symmetry/topological): result SHOULD satisfy. Violation →\n"
            f"  warn in report.md under 'Limitations' section, continue if fix is expensive.\n"
            f"- none/empirical: no structural constraint, focus on numerical accuracy.\n\n"
            f"## Anti-Fabrication\n"
            f"Do NOT report metrics that violate hard_check invariants. Self-check before writing report.md:\n"
            f"  for each hard_check item, verify result respects invariant. Violations must be fixed, not hidden."
        )
    else:
        _scan_hint = ""
    if fcm["winner_plan"]:
        _insights = "\n".join(f"- {x}" for x in fcm["merge_insights"]) or "- (none)"
        _fcm_hint = (
            f"\n\n## Selected Execution Plan (Step 1.7 fork-critique-merge, "
            f"winner perspective: {fcm['winner_perspective']})\n{fcm['winner_plan']}\n\n"
            f"## Merge insights from rejected candidates\n{_insights}\n\n"
            f"Follow this plan unless execution proves it infeasible — "
            f"if you deviate, explain why in report.md."
        )
    else:
        _fcm_hint = ""
    _step2_prompt_base = (
        "Now execute the task following your methodology checklist. "
        "Implement each [EXACT] component as-specified in the paper. "
        "If a component fails, debug and push through — do NOT silently substitute a simpler model. "
        "Write report/report.md with your results, referencing the checklist items you covered. "
        "Use file_write_tool for report.md, code_tool for analysis/plotting, bash_tool for running scripts.\n\n"
        "## METHOD SUBSTITUTE discipline (critical — agent repeatedly over-invests in one component)\n"
        "If a checklist component is infeasible in-sandbox (e.g. 150M-param transformer training, "
        "100M synthetic data generation, GPU-required training), do NOT keep rewriting the same "
        "solver file hoping heuristic search will close the gap. Instead:\n"
        "  1. Add a header line at the TOP of report/report.md: "
        "'METHOD SUBSTITUTE: <X> replaced <Y> because <reason>'.\n"
        "  2. Move on to the NEXT checklist item that does NOT need the infeasible component. "
        "Many tasks have items independent of the ML core (traceback algorithm, symbolic engine, "
        "data analysis, ablation studies) — leaving these unattempted = 0 score for that criterion.\n"
        "  3. If you've rewritten the same file >3 times (e.g. solver_v5.py -> v6.py -> v7.py) "
        "without measurable progress (same benchmark score, same error), STOP refining that file. "
        "Pivot to a different checklist item. Persistent rewriting of one file is stagnation, "
        "not progress — the wall_clock budget is better spent on other items.\n"
    )
    # v14 Task 6: 14 hint 走 HintCoordinator Hodge 正交分解合并.
    # ponytail: env HUGINN_HINT_COORDINATOR=0 走旧拼接路径, 留对照基线 / 回退兜底.
    _hint_coord_enabled = os.environ.get(
        "HUGINN_HINT_COORDINATOR", "1").lower() not in ("0", "false", "no")
    if _hint_coord_enabled:
        from huginn.agent.hint_coordinator import HintCoordinator
        _hint_coord = HintCoordinator()
        step2_prompt, _hint_trace_events = _hint_coord.coordinate(
            iter_n=0,
            # ponytail: 字符串硬编码, Task 2 CSM 完善后接 csm.current_state.
            csm_state="S4_CONSTRUCT",
            # ponytail: Task 4 未完, 占位 (1, 0). 升级路径: 接 betti 计算.
            beta=(1, 0),
            last_verdict=None,
            fcm_winner=fcm.get("winner_plan") or None,
            scan_text=scan_text,
            step2_prompt=_step2_prompt_base,
            iter_prompt=None,
            compass=None,
            step_eval=None,
            drift_info=None,
            imagination=None,
            meta_agent=None,
            cross_task_prior=_cross_task_prior_entries,
            # v15 Task 4: iter 0 不传 manifold — _hypo_manifold 在下方才 init,
            # 且 iter 0 无 observations 不需要 posterior lift. ponytail: 默认 None
            # 走 v14 keyword overlap, 不破坏 v14 行为.
        )
    else:
        step2_prompt = _legacy_build_step2_prompt(
            _step2_prompt_base, _scan_hint, _fcm_hint)

    import hashlib as _hashlib
    import json as _json
    import time as _time
    _trace_path = ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
    _trace_path.parent.mkdir(parents=True, exist_ok=True)
    # v26 Task 26.11: meta_trace 也走分片. RCB 跑 700 万 call 时单 jsonl 撑爆磁盘,
    # 每 N iter (HUGINN_TRACE_SHARD_INTERVAL, 默认 100) 切一个文件, 老分片 gzip 到
    # .huginn/archive/<task_id>/. 跟 audit_log 复用 _ShardState / write_sharded_jsonl,
    # 不引新依赖. ponytail: task_id 取 _trace_task_id (短 id, 已 strip 时间戳).
    # 升级路径: 整套换 Postgres/Cassandra, 文件分片只在单机 dev 场景留作 fallback.
    from huginn.events.audit_log import _ShardState as _MetaTraceShard
    from huginn.events.audit_log import write_sharded_jsonl as _mt_write_shard
    _meta_trace_shard = _MetaTraceShard(
        base_dir=_trace_path.parent,
        default_path=_trace_path,
        shard_interval=int(os.environ.get("HUGINN_TRACE_SHARD_INTERVAL", "100")) or 100,
        filename_prefix="meta_trace",
        task_id=_trace_task_id,
    )
    _max_exec_iters = int(os.environ.get(
        "HUGINN_RCB_EXEC_ITERS",
        "20" if extreme else "2",
    ))
    _prev_report_hash: str | None = None
    _stagnation_count = 0
    # P1-B: PMK 闭环反向边用 — 记上一轮 darwin_score, 本轮对比判断升/降.
    _prev_darwin: float | None = None

    # P4 Task 21: score_history 滑动窗口 — 监测 darwin_score 振荡/单调下降,
    # 振荡大降 LLM temperature, 单调下降达阈值补一刀终止. stream_chat_fn 不接
    # temperature 参数, 这里只记到 meta_trace + stats, 不影响实际 LLM 调参.
    # ponytail: 不引入完整控制论框架 (Lyapunov/PID/状态空间), 只用最小机制.
    _score_history = None
    try:
        from huginn.runtime.score_history import ScoreHistory
        _score_history = ScoreHistory()
    except Exception as _she:
        print(f"[score_history init skipped: {_she}]", flush=True)

    # P4 Task 22: cumulative output audit — 每轮扫 outputs/, 超阈值就阻断.
    # ponytail: 与 AuditLogger 解耦, 后者按事件追加 jsonl, 这里管累积轨迹.
    #   task_type 从 checklist 关键词推断, 默认 repro (RCBench 多为论文复现).
    _cumulative_auditor = None
    _audit_task_type = "repro"  # 默认 repro, 下方 checklist 扫到 optim 覆盖
    try:
        from huginn.security.cumulative_audit import CumulativeAuditor
        _cumulative_auditor = CumulativeAuditor()
        _ck_lower = (checklist or "").lower()
        if "optim" in _ck_lower and "repro" not in _ck_lower:
            _audit_task_type = "optim"
        elif "repro" not in _ck_lower and "reproduction" not in _ck_lower:
            _audit_task_type = "default"
    except Exception as _cae:
        print(f"[cumulative_audit init skipped: {_cae}]", flush=True)

    # 认知退火: T_hot 代理控制轨迹分叉数 (anneal_fork_count).
    # 任务开始 = 1.0 (最热, 多轨迹探索), 每轮几何降温 ×0.5, 停滞重热 +0.5.
    # ponytail: 不接 belief_entropy 测量 — RCB mini-loop 没有 hypothesis graph,
    #   熵无来源. 用模拟退火 schedule + 停滞信号做代理. 升级路径: 7-phase
    #   autoloop 接入后换 belief_entropy 驱动 (cognitive_heat_engine 已有接口).
    _t_hot = 1.0
    _fork_k_max = int(os.environ.get("HUGINN_RCB_FORK_K_MAX", "3"))
    _fork_enabled = os.environ.get(
        "HUGINN_RCB_FORK_ENABLED", "1").lower() not in ("0", "false", "no")
    _merge_hint = ""

    # Task 11+3: StepEvaluator 历史 + ProspectiveMemory (Step 2 循环外初始化)
    _evals_history: list = []
    # v14 Task 4: trace entry 累积, 每轮 append 主 entry 供 compute_betti 用.
    # ponytail: Task 3 supported_ratio 也复用这个 list (若实现). n≤50 by 截断.
    _trace_history: list = []
    # v14 Task 4: betti 计算 lazy import — 失败不阻塞主循环, betti jsonl 跳过.
    _compute_betti = None
    try:
        from huginn.metacog.trace_topology import compute_betti as _compute_betti
    except Exception as _tte:
        print(f"[betti init skipped: {_tte}]", flush=True)
    _betti_path = ws / HUGINN_DIR_NAME / "meta_trace_betti.jsonl"
    _prospective_mem = None
    try:
        from huginn.memory.prospective import ProspectiveMemory
        _prospective_mem = ProspectiveMemory(workspace=ws)
    except Exception as _e:
        print(f"[Prospective] init warning: {_e}", flush=True)

    # Task 18: 上一轮 StepEvaluation, 首轮 None. 每轮结束更新, 下一轮注入 prompt.
    _last_step_eval = None

    # G70: TaskMetrics — 循环外初始化, 每轮 update_metrics + save_metrics 落盘.
    # resume 时从盘上 load, 否则新建. total_steps 用 _max_exec_iters 估上界.
    # ponytail: task_state 用 SimpleNamespace(created_at=start_ts) 给 update_metrics
    #   算 estimated_remaining 用; 不接 TaskLifecycle.created_at 避免 lifecycle 时序
    #   依赖 (lifecycle 在 pause block 才创建, metrics 更早).
    _run_started_at = _time.time()
    try:
        from types import SimpleNamespace as _NS  # noqa: N814

        from huginn.runtime.task_metrics import (
            TaskMetrics,
            load_metrics,
        )
        _task_metrics = load_metrics(_task_id, ws) or TaskMetrics(
            task_id=_task_id, total_steps=_max_exec_iters)
        # 跨领域: 用 suggest_domain 推断 domain_label (材料/物理/化学/医学/数学)
        # ponytail: keyword 匹配, 不上分类器. 失败默认 unknown.
        try:
            from huginn.personas import PersonaManager
            _pm_for_domain = PersonaManager()
            _domain = _pm_for_domain.suggest_domain(
                (checklist or "") + " " + (instructions or ""))
            _task_metrics.domain_label = _domain
        except Exception:
            logger.debug("task domain label inference skipped", exc_info=True)
        _task_state_for_metrics = _NS(created_at=_run_started_at)
        _metrics_ok = True
    except Exception as _e:
        print(f"[task_metrics init skipped: {_e}]", flush=True)
        _task_metrics = None
        _task_state_for_metrics = None
        _metrics_ok = False

    # G62: detect_drift 结果缓存 — 每轮 evaluate_step 后算, 下一轮 prompt 注入用.
    # 首轮 None; build_meta_agent_text(drift_info=None) 走空路径不报错.
    _drift_info: tuple | None = None

    # P1: 想象力机制耦合 — rcb_runner 之前只接了 detect_drift, 没接 heat_engine.
    # autoloop 已有完整 _should_imaginate + _IMAGINATION_PROMPT_BLOCK, 但 rcb_runner
    # 走自己 6-step 不调 autoloop, 想象力断开. 这里直接复用 heat_engine + 复制 prompt block.
    # ponytail: 不引 autoloop 依赖, 直接 from cognitive_heat_engine import. prompt block
    #   复制一份 (autoloop 的 _IMAGINATION_PROMPT_BLOCK 是 class attr, 不便直接引用).
    _heat_engine = None
    try:
        from huginn.metacog.cognitive_heat_engine import get_heat_engine
        _heat_engine = get_heat_engine()
    except Exception as _he:
        print(f"[heat_engine init skipped: {_he}]", flush=True)

    _IMAGINATION_BLOCK = """
Imagination directive (speculative mode activated):
- Your prediction was significantly off, or your hypotheses keep getting refuted.
- Consider a counterfactual: what if the governing structure is different from what you assumed?
- Try shifting between mathematical structure families: PDE ↔ variational, continuum ↔ discrete, deterministic ↔ stochastic, linear ↔ nonlinear.
- This is NOT random guessing — the shift must be between mathematically valid structure families, grounded in the domain context.

LUCID review (mandatory after generating hypothesis):
- You are allowed an absurd premise, but the reasoning must be rigorous.
- State ONE necessary condition: without it, your hypothesis definitely fails.
- State ONE hidden assumption from the source domain that may not hold here.
- State ONE falsifiable test: if result is X, hypothesis is refuted.
- If you cannot state these, the hypothesis is dream-only and must be discarded.
"""

    # v15 Phase 2 Task 3.1: HypothesisManifold 接入 — init + 持久化
    # 失败降级到 None, 后续 _compute_v15_fields / _record_abduction 走空路径,
    # 不阻塞主循环. _last_abduction_result 给 Task 4 HintCoordinator 读.
    _hypo_manifold = None
    _hypo_manifold_path = ws / HUGINN_DIR_NAME / "hypothesis_manifold.jsonl"
    _hypo_obs_path = ws / HUGINN_DIR_NAME / "observations.jsonl"
    _last_abduction_result: dict | None = None
    # v15 Task 4: 上一轮 observations, 给 _build_posterior_guided_hint 用.
    # iter 0 时为空 (还没跑过), iter 1+ 持有上一轮的 observations.
    _iter_observations: list = []
    # P1-A: MCMC 状态 — 跨 iter 保持, 每 _mcmc_interval 步跑一次 mcmc_step.
    # 之前 mcmc_step 只在 _selfcheck 调, 主循环从不调, manifold 是静态的.
    # ponytail: MCMC 是 advisory only, 不强制 agent 用 MCMC 选的 h.
    # rcb_runner 不接 AutoloopEngine, 用 SimpleNamespace 当 holder 让 save_engine_state 能拉到字段
    import random as _mcmc_random
    import types as _mcmc_types
    _mcmc_engine = _mcmc_types.SimpleNamespace(
        _mcmc_current=None,
        _mcmc_rng=_mcmc_random.Random(
            int(os.environ.get("HUGINN_MCMC_SEED", "42"))),
        _mcmc_rng_state=None,
        _mcmc_accept_count=0,
        _mcmc_step_count=0,
        _mcmc_chains={},
        _iteration=0,
        workspace=ws,
        hypothesis_graph=None,
    )
    # Step 2: 跨步缓存 current 的 log_posterior, 拒绝时复用避免重算
    _mcmc_cached_log_p: float | None = None
    _mcmc_interval = int(os.environ.get("HUGINN_MCMC_INTERVAL", "5"))
    _mcmc_ckpt_interval = int(os.environ.get(
        "HUGINN_MCMC_CHECKPOINT_INTERVAL", "10000"))
    # 触觉层 env var: extreme 默认开, 非 extreme 默认关 (行为不变).
    # ponytail: haptic_layers.json 不存在时 _haptic_proposal 全返 None, 自动退化 fisher.
    _mcmc_haptic_enabled = os.environ.get(
        "HUGINN_MCMC_HAPTIC", "1" if extreme else "0") == "1"
    _mcmc_haptic_temperature = float(
        os.environ.get("HUGINN_MCMC_HAPTIC_TEMPERATURE", "1.0"))
    _mcmc_alignment_enabled = os.environ.get("HUGINN_MCMC_ALIGNMENT", "0") == "1"
    _mcmc_alignment_temperature = float(
        os.environ.get("HUGINN_MCMC_ALIGNMENT_TEMPERATURE", "1.0"))
    # P1-C: 完成度审计周期触发 — 之前 metacog_check_completion 只在 agent 声称
    # TASK COMPLETE 时跑, 长任务里 agent 一直不说完成 → 审计门永不跑.
    # ponytail: advisory only, 不阻断. 结果写 cognitive_evidence.md.
    _completion_interval = int(os.environ.get("HUGINN_COMPLETION_CHECK_INTERVAL", "10"))
    _prev_completion_hint = ""  # 跨 iter 传递审计 gap 提示
    # v15 Phase 4 Task 8: stagnation 检测 — best h_id 连续 N 轮不变触发 imagination
    _stagnation_history: list[str] = []
    _imagination_log_path = ws / HUGINN_DIR_NAME / "imagination_log.jsonl"

    # v15 Phase 5 Task 10: SelfModel 接入 — agent 对自己能力的 internal model.
    # 失败降级到 None, 后续 update / hint 注入 / imagine_from_blind_spot 都跳过.
    # 跨 task 路径跟 mem_mgr 同款 env var 约定, 避开沙箱拦截.
    _self_model = None
    _self_model_path = ws / HUGINN_DIR_NAME / "self_model.json"
    _self_model_cross_path: Path | None = None
    try:
        if os.environ.get("HUGINN_RCB_CROSS_TASK", "1") == "1":
            _sm_cross_dir = Path(
                os.environ.get(
                    "HUGINN_RCB_CROSS_TASK_DIR",
                    str(get_runtime_home() / "rcb_cross_task"),
                )
            )
            _sm_cross_dir.mkdir(parents=True, exist_ok=True)
            _self_model_cross_path = _sm_cross_dir / "self_model_cross_task.json"
        else:
            _self_model_cross_path = ws / HUGINN_DIR_NAME / "self_model_cross_task.json"
        from huginn.metacog.self_model import SelfModel
        _self_model = SelfModel(
            task_local_path=_self_model_path,
            cross_task_path=_self_model_cross_path,
            model=model,
        )
        print(
            f"[v15] SelfModel init: {len(_self_model._skills)} skills "
            f"(cross-task={_self_model_cross_path})",
            flush=True,
        )
    except Exception as _e:
        print(f"[v15] SelfModel init skipped: {_e}", flush=True)
        _self_model = None
    try:
        _hypo_manifold = _init_hypothesis_manifold(
            ws=ws,
            task_id=_trace_task_id,
            checklist=checklist or "",
            instructions=instructions,
            scan_text=scan_text or "",
            model=model,
            task_ctx=checklist or "",
        )
        if _hypo_manifold is not None:
            print(
                f"[v15] HypothesisManifold init: "
                f"{len(_hypo_manifold._hyp)} hypotheses",
                flush=True,
            )
    except Exception as _e:
        print(f"[v15] HypothesisManifold init skipped: {_e}", flush=True)
        _hypo_manifold = None


    # v18: bandit effort controller — register checklist items, init runtime.
    # ponytail: 不再 time-slice (v17 dead code), bandit 通过 reward 学习何时 switch.
    # 天花板: bandit state space 稀疏, 升级路径 = tile coding. 跨任务持久化缓解.
    _budget_items = None
    try:
        _budget_items = _checklist_item_parser(checklist or "")
        if _budget_items and len(_budget_items) >= 2:
            from huginn.agent.bandit_controller import EffortBandit
            _bandit = EffortBandit.get_instance()
            _bandit.set_items(_budget_items)
            _bandit.switch_item(0)
            print(f"[v18] bandit registered {len(_budget_items)} items, "
                  f"Q-table size={len(_bandit._Q)}", flush=True)
        else:
            _budget_items = None
    except Exception as _e:
        print(f"[v18] bandit init skipped (fallback to no-bandit): {_e}", flush=True)
        _budget_items = None

    # Task 3: 从 resume 的 iter 开始, 不重跑已 checkpoint 的轮次
    # 连续驳回计数: agent 已尽力但 gates 太严时, 第 3 次接受 TASK COMPLETE.
    # ponytail: 修 Math_003 iter 2-5 反复 'TASK COMPLETE' 不调工具被驳回循环.
    #   ceiling: 让 LLM judge 区分 '真尽力' vs '偷懒', 这里用计数兜底.
    _consecutive_complete_rejections = 0
    _MAX_COMPLETE_REJECTIONS = int(os.environ.get("HUGINN_RCB_MAX_COMPLETE_REJECTIONS", "3"))
    for _iter_n in range(_resume_from_iter, _max_exec_iters):
        # ponytail: v16 引入 _derivation_audit 只在 else 分支赋值, iter 0
        # 走 if 分支跳过赋值, line 1170 引用时 UnboundLocalError 崩溃.
        # 在 for 开头初始化, iter 0 默认空串不触发 derivation gate.
        _derivation_audit = ""
        # v15 Phase 3: 更新 LLM likelihood 的 iter counter (决定本轮是否调 LLM)
        _llm_lik_handle = getattr(_hypo_manifold, "_llm_likelihood", None)
        if _llm_lik_handle is not None:
            _llm_lik_handle.iter_n = _iter_n
        # P5 wall_clock 守卫: 挂钟预算耗尽就停, 不再跑新轮.
        # 比 max_tool_calls / stagnation / TASK COMPLETE 优先级高 — 跑满 timeout.
        if ctx.wall_expired_fn and ctx.wall_expired_fn():
            print(f"[P5] wall_clock budget expired at iter {_iter_n}, stopping.", flush=True)
            break
        # v14 Task 6: iter>0 的 hint 由 HintCoordinator Hodge 分解合并输出.
        # 把原来 _iter_prompt += X 的累积式改成 gather → dispatch, 让 HintCoordinator
        # 接管 gradient/curl/harmonic 三族, retrieval 族 (kb_chunks) 仍直接拼.
        if _iter_n == 0:
            _iter_prompt = step2_prompt
        else:
            _iter_base = (
                f"Continue execution. Iteration {_iter_n + 1}/{_max_exec_iters}.\n"
                f"Review the Research Trace section above for what you've already tried.\n"
                f"Identify the NEXT gap from your checklist (missing component, weak metric, "
                f"untested claim) and address it.\n"
                f"OVERWRITE report/report.md with updated results as you make progress.\n"
                f"If the report is complete and covers ALL checklist items, respond with "
                f"'TASK COMPLETE' followed by a one-paragraph summary. No tool call needed."
            )
            # M2Flow profiling-guided compass — Anthropic 文章启发:
            # cursor/compass > 视觉辅助, 显式状态信息比原始数据更有用.
            # 每轮注入 report.md 覆盖度 compass, 让 agent 看到"自己在哪",
            # 调整策略 (补缺 / 收尾 / 深化). ponytail: 只读, 不改控制流.
            _compass = ""
            _derivation_audit = ""
            try:
                _compass = _report_coverage_compass(ws, checklist) or ""
                # v8: 每 5 轮做一次 LLM 语义深度审计 (规则版每轮都跑, LLM 版成本高)
                # 解决规则版 keyword 命中漏同义改写的天花板 (MAE vs mean absolute error)
                if _iter_n > 0 and _iter_n % 5 == 0 and model and _compass:
                    _llm_compass = await _llm_coverage_audit(
                        model, ws, checklist, _compass,
                    )
                    if _llm_compass:
                        _compass = _llm_compass
                # v16: derivation chain audit — 区分 discussed vs executed.
                # 触发: iter>=2 每 3 轮一次, 或 drift fire 时强制触发.
                # 检查 outputs/ 是否有对应 checklist item 的实验产物.
                _drift_fire = False
                with contextlib.suppress(Exception):
                    _drift_fire, _ = _rcb_drift_check(_evals_history)
                if (model and _compass and ws / "outputs" and
                        ((_iter_n >= 2 and _iter_n % 3 == 0) or _drift_fire)):
                    _derivation_audit = await _derivation_chain_audit(
                        model, ws, checklist, _compass,
                    ) or ""
                    if _derivation_audit:
                        print(
                            f"[v16] derivation audit iter {_iter_n}: "
                            f"{'drift-fire' if _drift_fire else 'scheduled'}",
                            flush=True,
                        )
                        # 抽 derivation 百分比打印
                        import re as _re_d
                        _dm = _re_d.search(
                            r"DERIVATION:\s*(\d+)%", _derivation_audit)
                        if _dm:
                            print(f"[v16] derivation: {_dm.group(1)}%", flush=True)
            except Exception as _e:
                print(f"[v16] derivation audit skipped: {_e}", flush=True)

            # F4: FCM winner_plan 每轮提醒 — Step 1.7 选出的执行方案只在 iter 0
            # 注入 step2_prompt, compaction 后会丢失. 每轮追加避免 agent 漂移到
            # rejected fork 的思路上. ponytail: 不重复 merge_insights (iter 0 已有).
            _fcm_winner = (fcm.get("winner_plan") or "").strip() if fcm else ""
            _fcm_winner_reminder = (
                "\n\n## Selected Execution Plan (reminder — Step 1.7 FCM winner)\n"
                + _fcm_winner[:1200]
            ) if _fcm_winner else ""

            # F5: KB chunk 注入 — 让 RAG 真 augment RCB 生成.
            # 之前 rcb_runner 每轮查 KB 只用于 PMK pause 决策, 检索结果不进 prompt.
            # 现在每轮基于上一轮 attempted 做 KB 检索, top-2 chunk 注入 prompt.
            # ponytail: top_k=2 控成本, 截 400 字防 prompt 膨胀. 失败只跳过.
            # P3 Task 16: 优先走 UnifiedComplexView 双源融合 (KG + Meta-Trace),
            #   kb.query 作为 fallback. cross_task_store / kg 任一为 None 模块自兜底.
            _kb_chunks_text = ""
            if kb is not None:
                try:
                    _gap_query = ""
                    if _last_step_eval is not None:
                        _gap_query = (
                            getattr(_last_step_eval, "attempted", "")
                            or getattr(_last_step_eval, "gap", "")
                            or ""
                        )[:200]
                    if not _gap_query:
                        _gap_query = checklist[:200]
                    _kb_chunks: list[str] = []
                    _ucv_used = False
                    # 先试 UnifiedComplexView (真 import, 不走子进程)
                    try:
                        from huginn.metacog.unified_complex import UnifiedComplexView
                        _view = UnifiedComplexView(
                            cross_task_store=_cross_task_store, kg=_kg,
                        )
                        _vertices = _view.query(
                            domain=_trace_domain,
                            task_id=_trace_task_id,
                            keyword=_gap_query,
                            top_k=2,
                        )
                        for _v in _vertices[:2]:
                            _txt = getattr(_v, "content", "") or ""
                            if _txt:
                                _kb_chunks.append(_txt[:400])
                        if _kb_chunks:
                            _ucv_used = True
                            print(
                                f"[Step 2] UnifiedComplexView: "
                                f"{len(_vertices)} vertices (kg={_kg is not None}, "
                                f"store={_cross_task_store is not None})",
                                flush=True,
                            )
                    except Exception as _ucv_e:
                        print(
                            f"[UnifiedComplexView skipped: {_ucv_e}]",
                            flush=True,
                        )
                    # fallback: kb.query
                    if not _ucv_used:
                        _kb_hits = kb.query(_gap_query, top_k=2) or []
                        for _h in _kb_hits[:2]:
                            _txt = _h.get("content", "") if isinstance(_h, dict) else str(_h)
                            if _txt:
                                _kb_chunks.append(_txt[:400])
                    if _kb_chunks:
                        _kb_chunks_text = (
                            "\n\n## Domain Knowledge (KB retrieval, top-2)\n"
                            + "\n---\n".join(_kb_chunks)
                        )
                except Exception:
                    logger.debug("kb gap retrieval skipped", exc_info=True)

            # v14 Task 6: dispatch. HintCoordinator 接 gradient/curl/harmonic,
            # retrieval 族 (kb_chunks) 不归它管, 直接拼后面.
            # ponytail: step_eval / imagination / meta_agent 还没算 (下方 SHARED 块才算),
            #   先传 None. 当前 β=(1,0) + last_verdict=None 也不触发 curl/harmonic,
            #   所以这里 None 不影响输出. 升级路径 Task 4 完成后把 3 个搬上来再调,
            #   并在 SHARED 块加 _hint_coord_handled flag 跳过重复 append.
            if _hint_coord_enabled:
                _drift_text = None
                if isinstance(_drift_info, tuple) and _drift_info and _drift_info[0]:
                    _drift_text = str(_drift_info[1]) if len(_drift_info) > 1 else None
                # v15 Phase 2 Task 4: 构造 posterior-guided hint (核心 + 探索).
                # 用上一轮的 _iter_observations (循环顶部仍持有上轮值). 失败降级
                # 返回空串, coordinate 跳过注入, 不阻塞主循环.
                _posterior_hint_v15 = ""
                if _hypo_manifold is not None and _iter_observations:
                    try:
                        from huginn.agent.hint_coordinator import (
                            _build_posterior_guided_hint as _build_pg_hint,
                        )
                        _posterior_hint_v15 = _build_pg_hint(
                            manifold=_hypo_manifold,
                            observations=_iter_observations,
                            history_entries=_trace_history,
                            mcmc_current=(
                                _mcmc_engine._mcmc_current
                                if "_mcmc_engine" in dir()
                                and _mcmc_engine is not None
                                else None
                            ),
                        )
                    except Exception as _pe:
                        print(
                            f"[v15] posterior hint build skipped: {_pe}",
                            flush=True,
                        )
                        _posterior_hint_v15 = ""
                _iter_prompt, _hint_trace_events = _hint_coord.coordinate(
                    iter_n=_iter_n,
                    csm_state="S4_CONSTRUCT",
                    beta=(1, 0),
                    last_verdict=None,
                    fcm_winner=_fcm_winner or None,
                    scan_text=None,
                    step2_prompt=_step2_prompt_base,
                    iter_prompt=_iter_base,
                    compass=_compass or None,
                    step_eval=None,
                    drift_info=_drift_text,
                    imagination=None,
                    meta_agent=None,
                    cross_task_prior=_cross_task_prior_entries,
                    # v15 Task 4: manifold 给 boost posterior lift 用;
                    # posterior_hint 是预构造的核心+探索 hint 段.
                    manifold=_hypo_manifold,
                    posterior_hint=_posterior_hint_v15 or None,
                )
                if _kb_chunks_text:
                    _iter_prompt += _kb_chunks_text
            else:
                _iter_prompt = _iter_base
                if _compass:
                    _iter_prompt += "\n\n" + _compass
                if _fcm_winner_reminder:
                    _iter_prompt += _fcm_winner_reminder
                if _kb_chunks_text:
                    _iter_prompt += _kb_chunks_text
        _iter_prompt += _merge_hint
        _merge_hint = ""

        # P6 修复: 文件反复重写 stagnation 检测 — Math_003 暴露的问题.
        # agent 在 iter 2 卡 60 分钟反复重写 solver_v5→v6→v7.py, persistent goal
        # mode 不让 break, 但也没引导 pivot. 这里 advisory 注入 pivot 提示.
        # ponytail: 不阻断主流程, 只注入提示. 升级路径: 配合 darwin_score 无提升
        #   才触发, 当前 iter 边界 darwin_score 可能未算完, 先纯文件名触发.
        if _iter_n >= 1:
            _fr_stuck, _fr_msg = _detect_file_rewrite_stagnation(ws / "code")
            if _fr_stuck:
                _iter_prompt += "\n\n" + _fr_msg + "\n"
                print(f"[iter {_iter_n}] {_fr_msg}", flush=True)
        # P1-C: 上轮完成度审计的 gap 提示注入 — advisory, 不阻断.
        if _prev_completion_hint:
            _iter_prompt += _prev_completion_hint

        # P1: 想象力机制 — 每轮检查 should_imaginate, True 时注入 imagination block.
        # 触发条件: Re_cog > Re_crit (概念湍流) 或 T_hot > 0.7 (高熵).
        # P0-B: 加 HUGINN_USE_MENTAL_IMAGERY=1 守卫, 非 extreme 模式默认关.
        # ponytail: 失败只跳过, 不阻塞主流程. heat_engine.update_kinematics 在
        #   StepEvaluator 后调 (有 idea_count/stable_principles_count 数据才更新).
        if _heat_engine is not None and os.environ.get("HUGINN_USE_MENTAL_IMAGERY", "0") == "1":
            try:
                if _heat_engine.should_imaginate(_iter_n):
                    _iter_prompt += _IMAGINATION_BLOCK
                    print(f"[Step 2] imagination triggered at iter {_iter_n}", flush=True)
                    # P3 Task 15: 接 mental_imagery sketch→verify 闭环, 草图入 RAG KB.
                    # ponytail: 3 种预设模板 (lattice/particles/spectrum) 合成入 RAG 价值
                    #   有限, 升级路径换 LLM 判断 spec 而非正则. 渲染失败兜底文本不阻塞.
                    try:
                        from huginn.metacog import mental_imagery as _mi
                        # spec 从上一轮 best hypothesis 提取, 没有就用 checklist 兜底
                        _mi_spec = ""
                        if _hypo_manifold is not None and "_iter_best_h_id" in dir() \
                                and _iter_best_h_id is not None:  # noqa: F821
                            _prev_h = _hypo_manifold._hyp.get(_iter_best_h_id)  # noqa: F821
                            if _prev_h is not None:
                                _mi_spec = getattr(_prev_h, "statement", "") or ""
                        if not _mi_spec:
                            _mi_spec = (checklist or _iter_prompt or "")[:200]
                        _img_bytes = _mi.sketch(_mi_spec)
                        if _img_bytes:
                            _verify_res = _mi.verify(
                                _img_bytes, {"kind": "unknown"})
                            # 草图作为 visual primitive 注入 RAG KB (G4 auto-ingest)
                            if kb is not None:
                                with contextlib.suppress(Exception):
                                    kb.add_text(
                                        f"[mental_imagery sketch] spec={_mi_spec[:80]} "
                                        f"verified={_verify_res.get('verified', False)} "
                                        f"n_regions={_verify_res.get('n_regions_detected', 0)}",
                                        filename=f"mental_imagery_iter{_iter_n}.txt",
                                        metadata={
                                            "source": "mental_imagery",
                                            "iter": _iter_n,
                                            "verified": str(_verify_res.get("verified", False)),
                                        },
                                    )
                            print(
                                f"[Step 2] mental_imagery: verified="
                                f"{_verify_res.get('verified', False)} "
                                f"n_regions={_verify_res.get('n_regions_detected', 0)} "
                                f"at iter {_iter_n}",
                                flush=True,
                            )
                    except Exception as _mie:
                        print(f"[mental_imagery skipped: {_mie}]", flush=True)
            except Exception as _ie:
                print(f"[should_imaginate skipped: {_ie}]", flush=True)

        # v15 Phase 5 Task 11: 注入 blind_spot hint — agent 看到自己在哪有盲点 + 绕法.
        # 失败降级空串, 不阻塞. iter 0 跳过 (SelfModel 还没工具调用记录).
        # ponytail: 直接拼到 _iter_prompt 末尾, 跟 _IMAGINATION_BLOCK 同款注入.
        if _iter_n > 0 and _self_model is not None:
            try:
                from huginn.metacog.blind_spot_mapper import (
                    infer_blind_spots as _infer_bs,
                )
                from huginn.metacog.blind_spot_mapper import (
                    map_blind_spots_to_hint as _map_bs_hint,
                )
                _bs_list = _infer_bs(_self_model)
                _bs_hint = _map_bs_hint(_bs_list, max_n=3)
                if _bs_hint:
                    _iter_prompt += "\n\n" + _bs_hint
                    _n_high = sum(1 for b in _bs_list if b.priority == "high")
                    print(
                        f"[v15] blind_spot hint injected: "
                        f"{len(_bs_list)} spots ({_n_high} high)",
                        flush=True,
                    )
            except Exception as _be:
                print(f"[v15] blind_spot hint skipped: {_be}", flush=True)

        # Task 18 / G66: 注入 prospective / target_chain / step_eval 文本.
        # ponytail: 直接拼到 _iter_prompt 末尾 — fork / 主路径都吃同一份 prompt,
        #   注入一次覆盖两条路. 不重构 prompt 构造, 不新增抽象.
        #   天花板: 文本块顺序固定 (tc → pro → se → meta_agent), 不做动态优先级;
        #   recall 只传 current_step, 不带 events/variables (RCB mini-loop 没有结构化信号源).
        try:
            _ctx_b = getattr(agent, "_ctx_builder", None)
            _fired: list = []
            if _mem_mgr is not None:
                try:
                    _fired = _mem_mgr.recall_prospective(
                        {"current_step": _iter_n})
                except Exception as _e:
                    print(f"[prospective recall skipped: {_e}]", flush=True)
            if _ctx_b is not None:
                _tc_text = _ctx_b.build_target_chain_text(
                    _target_chains, _iter_n) or ""
                _pro_text = _ctx_b.build_prospective_text(_fired) or ""
                _se_text = _ctx_b.build_step_eval_text(_last_step_eval) or ""
                # 25.1: episode 历史 — kg 没建起来就空串, 不影响其他注入.
                _eh_text = _ctx_b.build_episode_history_text(_kg, _iter_n) if _kg else ""
                # 25.5: 元 Agent 视角重组 (Planner/Adviser/Reflector 三段).
                # drift_info 来自上一轮 detect_drift (_drift_info 缓存), 首轮 None.
                _ma_text = ""
                try:
                    _ma_text = _ctx_b.build_meta_agent_text(
                        target_chains=_target_chains,
                        last_step_evaluation=_last_step_eval,
                        tool_call_health=getattr(
                            _last_step_eval, "tool_call_health", None),
                        drift_info=_drift_info,
                    ) or ""
                except Exception as _e:
                    print(f"[meta_agent_text skipped: {_e}]", flush=True)
                # 25.6: PMK 三路立场显式呈现 — 给 LLM 看 persona/memory/kb 各自
                # 什么立场 + 一致性标签 (H¹ proxy). 不一致时 LLM 会看到 ⚠ 提示.
                _pmk_text = ""
                try:
                    _pmk_text = _ctx_b.build_pmk_text(
                        persona=persona,
                        memory=_mem_mgr,
                        kb=kb,
                        last_step_evaluation=_last_step_eval,
                    ) or ""
                except Exception as _e:
                    print(f"[pmk_text skipped: {_e}]", flush=True)
                _ctx_inject = (
                    _tc_text + _pro_text + _se_text + _eh_text
                    + _ma_text + _pmk_text
                )
                if _ctx_inject:
                    _iter_prompt += "\n\n" + _ctx_inject
        except Exception as _e:
            print(f"[ctx inject skipped: {_e}]", flush=True)

        # 跨任务 curiosity hint — 跨任务共享 db 时, 注入历史弱 persona.
        # 单任务首次跑 self_model 空 → 返空串, 不影响. 跨任务积累后, agent
        # 知道"哪些 persona 历史成功率低", 主动 seek 而非被动 escape.
        if os.environ.get("HUGINN_CURIOSITY_HINT", "0") == "1" and _mem_mgr is not None:
            try:
                _sm = _mem_mgr.longterm.get_self_model()
                if _sm:
                    _weak = [
                        f"- {v.get('dimension', '?')}/{v.get('hyp_type', '?')}: "
                        f"rate={v.get('rate', 0):.2f} (n={v.get('success', 0) + v.get('failure', 0)})"
                        for v in _sm.values()
                        if isinstance(v.get("rate"), (int, float))
                        and v["rate"] < 0.4
                        and v.get("success", 0) + v.get("failure", 0) >= 3
                    ]
                    if _weak:
                        _iter_prompt += (
                            "\n\n[CURIOSITY] 历史预测不准的簇 (主动探索方向):\n"
                            + "\n".join(_weak[:5]) + "\n"
                        )
                        print(f"[curiosity] injected {len(_weak)} weak personas", flush=True)
            except Exception as _ce:
                print(f"[curiosity skipped: {_ce}]", flush=True)

        # Deliverable 物理检查 — 从 BenchmarkOrchestrator 复用 RCB_DELIVERABLES.
        # checklist 系统 check 内容覆盖度 (keyword 命中), 这里 check 文件存在性.
        # 两者互补: compass 说 "报告提到了 X" 但 report.md 不存在 → deliverable 提示.
        # iter 0 跳过 (agent 还没开始), iter>=1 才检查.
        if _iter_n > 0:
            try:
                from huginn.bench.orchestrator import RCB_DELIVERABLES, _triage_prompt
                _missing = RCB_DELIVERABLES.missing(ws)
                if _missing:
                    _iter_prompt += "\n\n" + _triage_prompt(_missing)
                    print(f"[deliverable] missing: {_missing}", flush=True)
            except Exception as _de:
                print(f"[deliverable check skipped: {_de}]", flush=True)

        # v16: derivation chain audit 注入 — 区分 discussed vs executed.
        # derivation_audit 在前面 compass 块算出. 有 DISCUSSED_ONLY 段时
        # 强制注入 MISSING DERIVATION 提示, 让 agent 知道"光讨论不够, 要产 artifact".
        # ponytail: 只在有 DISCUSSED_ONLY 段时注入, 全 FULFILLED 不打扰 agent.
        if _derivation_audit and "DISCUSSED_ONLY:" in _derivation_audit:
            _iter_prompt += (
                "\n\n## DERIVATION CHAIN AUDIT (v16 gate — agent repeatedly "
                "discusses checklist items in report without executing them)\n"
                + _derivation_audit
                + "\n\n→ Do NOT just rewrite report sections. PRODUCE the missing "
                "artifacts in outputs/ first (e.g. outputs/imo_results.json with "
                "your own run results, NOT report citing paper values). "
                "TASK COMPLETE will be BLOCKED until artifacts exist."
            )

        # TFM: T_hot 决定这轮是单轨迹还是 k 路分叉
        _k = anneal_fork_count(_t_hot, _fork_k_max) if _fork_enabled else 1
        _tree = agent._conversation_tree
        _branch_point = _tree.active_leaf_id if _tree is not None else None
        if _branch_point is None:
            _k = 1  # 树是空的没法分叉 (Step 1 都失败了)

        if _k > 1:
            print(
                f"\n--- Step 2 iter {_iter_n + 1}/{_max_exec_iters} "
                f"[TFM: T_hot={_t_hot:.2f} → {_k} forks] ---\n",
                flush=True,
            )
            # TFM fork 顺序执行共享 workspace, 后序 fork 覆盖主路径 report.md.
            # winner=None 时不合并, 但 report.md 已被 fork 破坏 → Step 2.5 落回
            # fallback. 备份主 report.md, winner=None 时恢复. ponytail: 最小备份,
            # 不改 fork 执行逻辑.
            _main_rp_backup = ws / "report" / "report.md"
            _main_rp_backup_text = (
                _main_rp_backup.read_text(encoding="utf-8")
                if _main_rp_backup.exists() else None
            )
            _fork_reports: dict[str, str] = {}
            _fork_leaves: dict[str, str] = {}
            for _persp, _bias in _FCM_PERSPECTIVES[:_k]:
                _tree.set_active_leaf(_branch_point)
                _fork_tid = f"{thread_id}_f{_iter_n}_{_persp}"
                # ponytail: 不让 fork 改写 report_fork_X.md — system prompt 里
                # "写 report.md" 的先验太强, fork 会无视改名指令 (σ: prompt
                # 对抗). 顺着先验让它写 report.md, runner 在 fork 结束后自己
                # 快照成 report_fork_X.md 供 verifier 评审.
                _fork_prompt = (
                    f"[Trajectory fork — '{_persp}' bias] {_bias}\n\n"
                    f"{_iter_prompt}"
                )
                await ctx.stream_chat_fn(_fork_prompt, f"fork_{_persp}", tid=_fork_tid)
                _fork_leaves[_persp] = _tree.active_leaf_id or _branch_point
                _fork_reports[_persp] = ""
                _main_rp = ws / "report" / "report.md"
                if _main_rp.exists():
                    _snap = ws / "report" / f"report_fork_{_persp}.md"
                    try:
                        _snap.write_text(
                            _main_rp.read_text(encoding="utf-8"), encoding="utf-8")
                        _fork_reports[_persp] = _main_rp.read_text(encoding="utf-8")
                    except Exception as _e:
                        print(f"[tfm snapshot {_persp} skipped: {_e}]", flush=True)
                # fork 的计算产物也快照 — 复现门禁的对照集. 顺序执行的 fork
                # 共享 outputs/, 不快照前序 fork 的产物会被后序覆盖.
                _out_src = ws / "outputs"
                if _out_src.is_dir():
                    try:
                        shutil.copytree(
                            _out_src, ws / "report" / f"outputs_fork_{_persp}",
                            dirs_exist_ok=True)
                    except Exception as _e:
                        print(f"[tfm outputs snap {_persp} skipped: {_e}]", flush=True)

            _verdict = await judge_fork_reports(
                _fork_reports, checklist, model,
                artifact_dirs={
                    p: ws / "report" / f"outputs_fork_{p}" for p in _fork_reports
                })
            _winner = _verdict["winner"]
            print(f"[tfm: winner={_winner} scores={_verdict['scores']}]", flush=True)
            if _verdict.get("gate"):
                print(f"[tfm gate: {_verdict['gate']}]\n", flush=True)
            if _winner and _winner in _fork_leaves:
                # 合并: winner 报告入主路径, 轨迹沿 winner 分支继续
                _rd = ws / "report"
                _rd.mkdir(parents=True, exist_ok=True)
                (_rd / "report.md").write_text(
                    _fork_reports[_winner], encoding="utf-8")
                _tree.set_active_leaf(_fork_leaves[_winner])
                thread_id = f"{thread_id}_f{_iter_n}_{_winner}"
                _ai_text = f"[tfm: merged fork '{_winner}' into main trajectory]"
                if _verdict["merge_notes"]:
                    _notes = "\n".join(f"- {x}" for x in _verdict["merge_notes"])
                    _merge_hint = (
                        f"\n\n## Merge notes from rejected forks (trajectory fork-merge)\n"
                        f"{_notes}\nSalvage these into report/report.md where applicable."
                    )
            else:
                _tree.set_active_leaf(_branch_point)
                _ai_text = "[tfm: all forks produced empty reports, main trajectory unchanged]"
                # winner=None 时 fork 已破坏主 report.md, 从备份恢复. 不恢复的话
                # Step 2.5 看到的是最后一个 fork 的残缺 report (不是空), 不会触发
                # emergency write, 最终评分拿到残缺 report.
                if _main_rp_backup_text is not None:
                    try:
                        _main_rp_backup.write_text(
                            _main_rp_backup_text, encoding="utf-8")
                    except Exception as _e:
                        print(f"[tfm restore main report failed: {_e}]", flush=True)

            # Meta-Trace: 分叉轮也留痕 (role=trajectory_fork_merge)
            try:
                _tfm_entry = {
                    "iteration": _iter_n + 1,
                    "ts": _time.time(),
                    "role": "trajectory_fork_merge",
                    "attempted": f"{_k}-fork trajectory exploration (T_hot={_t_hot:.2f})",
                    "found": f"winner={_winner} scores={_verdict.get('scores')} gate={_verdict.get('gate')}",
                    "evidence": [_fork_reports.get(_winner, "")[:150]] if _winner else [],
                    "limitations": ["sequential forks share workspace; outputs snapshotted per fork"],
                    "artifacts": [f"report/report_fork_{p}.md" for p in _fork_reports],
                    "next_hint": "salvage merge_notes" if _verdict.get("merge_notes")
                                 else "continue winner trajectory",
                    "darwin_score": 0.0,
                    "supported_ratio": 0.0,
                    # v14 Task 1: simplicial complex schema. trajectory_fork_merge
                    # 跟 FCM winner 同族 (gradient) — 都是 task-driven 选路.
                    "simplex_id": _make_simplex_id(_trace_task_id, _iter_n + 1, "trajectory_fork_merge"),
                    "cochain_type": "gradient",
                    "domain": _infer_domain(_trace_task_id),
                    "task_id": _trace_task_id,
                    "model_version": _MODEL_VERSION,
                }
                # v15: upgrade_entry 补 v15 默认字段 (tfm entry 不填 abduction 结果,
                # 只保证 schema 一致, 让 Task 4 读 trace 时不用判字段存在)
                try:
                    from huginn.metacog.trace_topology import (
                        upgrade_entry as _upgrade_entry,
                    )
                    _upgrade_entry(_tfm_entry)
                except Exception:
                    logger.debug("tfm trace entry upgrade skipped", exc_info=True)
                with _trace_path.open("a", encoding="utf-8") as f:
                    f.write(_json.dumps(_tfm_entry, ensure_ascii=False) + "\n")
            except Exception as _e:
                print(f"[tfm trace skipped: {_e}]", flush=True)
        else:
            print(f"\n--- Step 2 iter {_iter_n + 1}/{_max_exec_iters} ---\n", flush=True)
            _ai_text = await ctx.stream_chat_fn(_iter_prompt, f"step2_iter{_iter_n + 1}")

        # 退火降温 (在停滞重热之前, 停滞信号下一轮生效)
        _t_hot = max(0.0, _t_hot * 0.5)

        # agent 这轮没说 TASK COMPLETE → 连续驳回计数清零. agent 做了实际工作,
        # 之前的驳回不算"陷入循环". ponytail: 字符串匹配, 简单粗暴.
        if not (_ai_text and "TASK COMPLETE" in _ai_text.upper()):
            _consecutive_complete_rejections = 0

        # v15 Phase 2 Task 3.2+3.3: 收集 observations + abductive inference
        # 失败降级到空 observations + 不写 abduction entry, 不阻塞主循环.
        # _last_abduction_result 给 Task 4 HintCoordinator 读 (下一轮 prompt 注入).
        _iter_observations: list = []
        _iter_best_h_id: str | None = None
        _iter_log_post: float = 0.0
        _iter_fisher_info: float = 0.0
        if _hypo_manifold is not None:
            try:
                _report_text_for_obs = ""
                _report_path_for_obs = ws / "report" / "report.md"
                if _report_path_for_obs.exists():
                    _report_text_for_obs = _report_path_for_obs.read_text(
                        encoding="utf-8")
                _iter_observations = _collect_observations(
                    step_result=_ai_text or "",
                    report_text=_report_text_for_obs,
                    checklist=checklist or "",
                )
                if _iter_observations:
                    _iter_best_h_id, _iter_log_post, _iter_fisher_info = (
                        _compute_v15_fields(_hypo_manifold, _iter_observations)
                    )
                    _append_observations_log(
                        _iter_observations, _hypo_obs_path,
                        iteration=_iter_n + 1,
                    )
                    _record_abduction(
                        _hypo_manifold,
                        _iter_observations,
                        trace_path=_trace_path,
                        task_id=_trace_task_id,
                        iteration=_iter_n + 1,
                        ts=_time.time(),
                    )
                    # 持久化 manifold (predictions 可能被外部更新, 此处保状态)
                    _save_manifold(_hypo_manifold, _hypo_manifold_path)
                    # 缓存 abduction 结果给 Task 4 HintCoordinator 读
                    _last_abduction_result = {
                        "best_h_id": _iter_best_h_id,
                        "log_posterior": _iter_log_post,
                        "fisher_info": _iter_fisher_info,
                        "n_observations": len(_iter_observations),
                        "iteration": _iter_n + 1,
                    }
            except Exception as _e:
                print(f"[v15] observation/abduction skipped: {_e}", flush=True)

        # v15 Phase 4 Task 8: stagnation 检测 + imagination 触发
        # best h_id 连续 3 轮不变 -> 在 manifold 上做 structure-preserving transform
        # 失败一律降级 (imagine_with_checks 返回 None), 不阻塞主循环.
        if _iter_best_h_id is not None and _hypo_manifold is not None:
            _stagnation_history.append(_iter_best_h_id)
            # P4 Task 23: cycle_detect 顶层函数接入 — hypothesis id 序列卡顿检测.
            # ponytail: 不扩展 VF2 cycle 检测到更长周期轨道, 保持现状 (Floyd 算法).
            # 升级路径: 接 trajectory_pattern.VF2 做语义级 cycle 检测.
            try:
                from huginn.runtime.cycle_detect import is_stuck as _cycle_is_stuck
                if _cycle_is_stuck(_stagnation_history, min_cycle_len=2, min_repeats=2):
                    print(f"[P4] hypothesis cycle detected: {_stagnation_history[-6:]}", flush=True)
            except Exception:
                logger.debug("hypothesis cycle detection skipped", exc_info=True)
            try:
                from huginn.metacog.imagination import (
                    detect_stagnation as _detect_stagnation,
                )
                from huginn.metacog.imagination import (
                    imagine_with_checks as _imagine_with_checks,
                )
                if _detect_stagnation(_stagnation_history, N=3):
                    print(
                        f"[v15] stagnation detected (best h={_iter_best_h_id} "
                        f"for 3 iters), triggering imagination",
                        flush=True,
                    )
                    _parent_h = _hypo_manifold._hyp.get(_iter_best_h_id)
                    if _parent_h is not None:
                        # 轮试三族变换, 第一个成功就用
                        for _t_type in ("algebraic", "topological", "order"):
                            _new_h = _imagine_with_checks(
                                _parent_h, _t_type, _hypo_manifold,
                                model=model,
                                sigma=0.5,
                                log_path=_imagination_log_path,
                            )
                            if _new_h is not None:
                                try:
                                    _hypo_manifold.add(_new_h)
                                    _save_manifold(_hypo_manifold, _hypo_manifold_path)
                                except ValueError:
                                    logger.debug("best-effort op failed", exc_info=True)  # duplicate h_id, 跳过
                                # v15 Phase 4 Task 9.2: harmonic trace entry
                                _img_entry = {
                                    "iteration": _iter_n + 1,
                                    "ts": _time.time(),
                                    "role": "imagination",
                                    "attempted": f"imagine ({_t_type}) on {_parent_h.h_id}",
                                    "found": f"new_h={_new_h.h_id} desc={_new_h.description[:80]}",
                                    "evidence": [
                                        f"transform={_t_type}",
                                        f"n_params={_new_h.n_params}",
                                        f"predictions={list(_new_h.predictions.keys())}",
                                    ],
                                    "limitations": [],
                                    "artifacts": [],
                                    "next_hint": f"test {_new_h.h_id} predictions",
                                    "darwin_score": 0.0,
                                    "supported_ratio": 0.0,
                                    "simplex_id": _make_simplex_id(
                                        _trace_task_id, _iter_n + 1, f"imagination_{_t_type}"),
                                    "cochain_type": "harmonic",
                                    "domain": _infer_domain(_trace_task_id),
                                    "task_id": _trace_task_id,
                                    "model_version": _MODEL_VERSION,
                                    "hypothesis_id": _new_h.h_id,
                                    "log_posterior": 0.0,
                                    "fisher_info": 0.0,
                                    "imagination_parent": _parent_h.h_id,
                                }
                                try:
                                    from huginn.metacog.trace_topology import (
                                        upgrade_entry as _upgrade_entry,
                                    )
                                    _upgrade_entry(_img_entry)
                                except Exception:
                                    logger.debug("imagination trace entry upgrade skipped", exc_info=True)
                                with _trace_path.open("a", encoding="utf-8") as f:
                                    f.write(_json.dumps(_img_entry, ensure_ascii=False) + "\n")
                                print(
                                    f"[v15] imagination: {_parent_h.h_id} -> "
                                    f"{_new_h.h_id} ({_t_type})",
                                    flush=True,
                                )
                                break
            except Exception as _e:
                print(f"[v15] imagination skipped: {_e}", flush=True)

        # v15 Phase 5 Task 12.2: blind_spot 触发 imagination — 跟 stagnation 并行.
        # 选 high priority blind spot 作种子, 调 imagine_from_blind_spot.
        # ponytail: 每 3 轮试一次, 避免 LLM 调用 spam. 失败降级, 不阻塞.
        #   天花板: 没跟踪已试过的 skill, 同一 blind spot 可能被反复试. 升级路径:
        #   _tried_bs_skills set 去重, + feedback_from_imagination 闭环.
        if (
            _self_model is not None
            and _hypo_manifold is not None
            and _iter_n > 0
            and _iter_n % 3 == 0
        ):
            try:
                from huginn.metacog.blind_spot_mapper import (
                    infer_blind_spots as _infer_bs,
                )
                from huginn.metacog.blind_spot_mapper import (
                    pick_imagination_seed as _pick_bs_seed,
                )
                from huginn.metacog.imagination import (
                    imagine_from_blind_spot as _img_from_bs,
                )
                _bs_list = _infer_bs(_self_model)
                _bs_seed = _pick_bs_seed(_bs_list)
                if _bs_seed is not None:
                    _bs_new_h = _img_from_bs(
                        _bs_seed, _hypo_manifold, model=model,
                        log_path=_imagination_log_path, sigma=0.5,
                    )
                    if _bs_new_h is not None:
                        try:
                            _hypo_manifold.add(_bs_new_h)
                            _save_manifold(_hypo_manifold, _hypo_manifold_path)
                        except ValueError:
                            logger.debug("best-effort op failed", exc_info=True)  # duplicate h_id, 跳过
                        # v15 Phase 5 Task 12.3: imagination 成功 → feedback 升级
                        # uncertain → capable (绕过盲点 = 实际能做).
                        # ponytail: 乐观反馈, 新 h 进 manifold 就算 success.
                        #   天花板: 真 success 要等下轮 step_eval 验证. 升级路径:
                        #   下一轮 on_track=true 时才 feedback(success=True).
                        with contextlib.suppress(Exception):
                            _self_model.feedback_from_imagination(
                                _bs_seed.skill, success=True)
                        print(
                            f"[v15] blind_spot imagination: {_bs_seed.skill} "
                            f"-> {_bs_new_h.h_id}",
                            flush=True,
                        )
            except Exception as _e:
                print(f"[v15] blind_spot imagination skipped: {_e}", flush=True)

        # 写 Meta-Trace entry — P1 的 build_meta_trace_text 下一轮会读到.
        # ponytail: 字段从 self/agent 状态抽, 不调 LLM. RCB mini-loop 不跑 darwin
        #   ratchet, darwin_score 用 _compute_darwin_score, supported_ratio 见下方.
        try:
            _report_text = ""
            _report_path_iter = ws / "report" / "report.md"
            if _report_path_iter.exists():
                _report_text = _report_path_iter.read_text(encoding="utf-8")
            # v14 Task 3: supported_ratio = 命中数(overlap>0.7) / max(历史总数, 1).
            # 当前 entry.attempted 跟 _trace_history 里每条 entry.evidence 算 TF-IDF cosine.
            # 首轮历史为空 → 0.0. ponytail: evidence 是 list, join 成 str 喂 overlap 函数.
            # P6 修复: LLM 回答多是控制消息 ([tfm:...] / TASK COMPLETE), 跟 evidence
            # (report.md 前 150 字符) TF-IDF 必然失配. 控制消息 fallback 到 report.md
            # 前 200 字符, 让 supported_ratio 反映 report.md 自我一致性. 升级路径: 用
            # tool_call 描述 (从 audit_log 提取) 作 attempted, 而非 LLM 回答文本.
            _ai_text_stripped = (_ai_text or "").strip()
            _is_control_msg = (
                not _ai_text_stripped
                or _ai_text_stripped.startswith("[tfm:")
                or _ai_text_stripped.upper().startswith("TASK COMPLETE")
                or len(_ai_text_stripped) < 50
            )
            if _is_control_msg and _report_text:
                _attempted_text = _report_text[:200].replace("\n", " ")
            else:
                _attempted_text = (_ai_text[:200] if _ai_text else _iter_prompt[:200]).replace("\n", " ")
            _supported_hits = 0
            if _trace_history:
                try:
                    from huginn.context_builder import _compute_semantic_overlap
                    for _hist in _trace_history:
                        _hev = _hist.get("evidence") or []
                        _hev_text = " ".join(_hev) if isinstance(_hev, list) else str(_hev)
                        if _compute_semantic_overlap(_attempted_text, _hev_text) > 0.7:
                            _supported_hits += 1
                except Exception as _oe:
                    print(f"[supported_ratio skipped: {_oe}]", flush=True)
            _supported_ratio = _supported_hits / max(len(_trace_history), 1)
            _entry = {
                "iteration": _iter_n + 1,
                "ts": _time.time(),
                "role": "rcb_exec",
                "attempted": _attempted_text,
                "found": (_ai_text or "")[:300],
                "evidence": [_report_text[:150]] if _report_text else [],
                "limitations": [],
                "artifacts": ["report/report.md"] if _report_path_iter.exists() else [],
                "next_hint": "continue execution" if _iter_n < _max_exec_iters - 1 else "step3 critique",
                # v14 Task 2: darwin = 1 - gap_severity, 首轮 _last_step_eval=None → 0.5.
                "darwin_score": _compute_darwin_score(_last_step_eval),
                # v14 Task 3: supported_ratio 跨轮语义重叠 (TF-IDF cosine > 0.7 视为支持).
                "supported_ratio": _supported_ratio,
                # v14 Task 1: 主循环 entry = gradient (task-driven).
                "simplex_id": _make_simplex_id(_trace_task_id, _iter_n + 1, "rcb_exec"),
                "cochain_type": "gradient",
                "domain": _infer_domain(_trace_task_id),
                "task_id": _trace_task_id,
                "model_version": _MODEL_VERSION,
            }
            # v15 Phase 2 Task 3.4: upgrade_entry 补 v15 默认值 + 填本轮 abduction 结果
            # upgrade 失败不阻塞, v14 entry 仍可写; 字段缺失时 upgrade_entry 补默认.
            try:
                from huginn.metacog.trace_topology import (
                    upgrade_entry as _upgrade_entry,
                )
                _upgrade_entry(_entry)
                _entry["hypothesis_id"] = _iter_best_h_id
                _entry["log_posterior"] = _iter_log_post
                _entry["fisher_info"] = _iter_fisher_info
                # imagination_parent 留 None (Phase 4 的工作)
            except Exception:
                logger.debug("meta trace entry upgrade skipped", exc_info=True)
            # v26 Task 26.11: 走分片写入. _meta_trace_shard 在 _trace_path 初始化处
            # 创建, task_id 已设, 跨 shard 边界自动 gzip 归档老分片. 老路径
            # _trace_path 仍作 default_path 兜底 (task_id 没设时). 失败只 log debug,
            # 不抛 — 跟原 with open 一致的容错语义.
            _mt_write_shard(_meta_trace_shard, _entry, _entry.get("iteration"))
            # v14 Task 4: 累积 entry 到 _trace_history, 算 betti 写 jsonl.
            # ponytail: 只对主 entry 做, step_evaluation 等辅助 entry 不 append
            #   (避免噪声边). betti 失败只 warn, 不阻塞主循环.
            _trace_history.append(_entry)
            if _compute_betti is not None:
                try:
                    _b0, _b1 = _compute_betti(_trace_history)
                    with _betti_path.open("a", encoding="utf-8") as _bf:
                        _bf.write(_json.dumps({
                            "ts": _time.time(),
                            "iteration": _iter_n + 1,
                            "beta_0": _b0,
                            "beta_1": _b1,
                            "n_entries": len(_trace_history),
                        }, ensure_ascii=False) + "\n")
                except Exception as _be:
                    print(f"[betti write skipped: {_be}]", flush=True)
        except Exception as _e:
            print(f"[meta_trace write skipped: {_e}]", flush=True)

        # v18: bandit iter-end — reward_slow = β * Δdarwin_score.
        # ponytail: 失败静默, bandit 内部 catch. _entry 可能未定义, 用 get 兜.
        try:
            if _budget_items:
                from huginn.agent.bandit_controller import EffortBandit
                _darwin = float(_entry.get("darwin_score", 0.5)) if "_entry" in dir() else 0.5
                EffortBandit.get_instance().update_iter_end(_darwin)
        except Exception:
            logger.debug("bandit iter-end update skipped", exc_info=True)

        # StepEvaluator 评估 + Checkpoint 保存 (G63 + G59)
        # 失败只 warn, 不影响主循环. _entry 可能因 meta_trace try 失败而未定义,
        # 用 try 兜住 NameError.
        try:
            from huginn.metacog.step_evaluator import (
                evaluate_step,
                should_continue,
            )
            from huginn.metacog.target_chain import update_progress
            # ponytail: scan_text 是 Step 1.5 的纯文本输出, 不是 list[dict],
            # 没法直接喂给 _check_structure — 传 None 走 LLM 兜底路径.
            # 升级路径: Step 1.5 改输出结构化 JSON (verifiable_via per item).
            # 25.3: tool_call_health=None 让 evaluate_step 从 audit_log 自动算;
            #   audit_log 路径取 audit_log._resolve_audit_path(), 失败回 None.
            # 25.1: kg=_kg (上面初始化的 ProjectKnowledgeGraph). None 时
            #   evaluate_step 跳过 episode + dep edge 写入.
            try:
                from huginn.events.audit_log import _resolve_audit_path as _rap
                _audit_path = _rap()
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                _audit_path = None
            # ponytail: prev_step_id 用上一轮的 iteration; 首轮 None, 不写 dep edge.
            _prev_sid = _iter_n if _iter_n > 0 else None
            _step_eval = evaluate_step(
                meta_trace_entry=_entry,
                target_chains=_target_chains,
                verification_signals=None,
                memory=_mem_mgr,
                kb=kb,
                persona=persona,
                model=model,
                tool_call_health=None,
                kg=_kg,
                prev_step_id=_prev_sid,
                audit_log_path=_audit_path,
            )
            _evals_history.append(_step_eval)
            _last_step_eval = _step_eval  # Task 18: 供下一轮 prompt 注入
            for _tc in _target_chains:
                update_progress(_tc, _step_eval.found)

            # v15 Phase 5 Task 10.2: 从 audit_log 抓工具调用记录, 更新 SelfModel.
            # 失败降级静默, 不阻塞主循环. _audit_path 上面 evaluate_step 已算过.
            # ponytail: 复用 evaluate_step 的 audit_log 路径, 不重复扫文件.
            #   天花板: audit_log schema 没 step_id 时全收 (跟 step_evaluator 一致),
            #   会让 SelfModel 把别 step 的工具调用也算进来. 升级路径: schema 加 step_id.
            if _self_model is not None and _audit_path is not None:
                try:
                    from huginn.metacog.self_model import (
                        extract_step_result_from_audit as _extract_step,
                    )
                    _step_records = _extract_step(_audit_path, step_id=_iter_n + 1)
                    if _step_records:
                        _self_model.update_from_step(_step_records)
                except Exception as _se:
                    print(f"[v15] self_model update skipped: {_se}", flush=True)

            # AV6: ProspectiveMemory 闭环 — on_track=false/unsure 时记一条 intention,
            # 下一轮 recall_prospective 触发, 经 build_prospective_text 注入 prompt.
            # description 不含 "用户决策", 走 reminder 路径不触发 pause (RCB 跑分要顺跑).
            # ponytail: trigger=dependency 保证下一轮必触发; 天花板: 同偏差连续多轮
            #   会写多条 jsonl, 升级路径 store 前 list_pending() 去重.
            if _mem_mgr is not None and _step_eval.on_track in ("false", "unsure"):
                try:
                    from huginn.memory.prospective import _new_intention_id
                    _mem_mgr.remember_prospective({
                        "intention_id": _new_intention_id(),
                        "description": (
                            f"上一步脱轨需复核: attempted={_step_eval.attempted[:80]}"
                            f"; deviation={_step_eval.deviation[:80]}"
                        ),
                        "trigger_type": "dependency",
                        "trigger_payload": {"depends_on_step": _iter_n},
                        "priority": 5,
                        "created_at": _time.time(),
                        "source_step": _iter_n,
                    })
                except Exception as _pe:
                    print(f"[prospective store skipped: {_pe}]", flush=True)

            # P1+AV8+AV4: heat_engine 闭环 — 调 cognitive_loop 共享函数.
            # 之前 4 档映射逻辑两边各写一份 (rcb_runner + autoloop reflect_fn),
            # 现在抽到 update_heat_engine_after_step 共享.
            if _heat_engine is not None:
                from huginn.autoloop.cognitive_loop import update_heat_engine_after_step
                _idea_count = sum(
                    len(getattr(tc, "completed_results", set()) or set())
                    for tc in _target_chains
                )
                _sp_len = len(step2_prompt) if _iter_n == 0 else len(_iter_prompt)
                update_heat_engine_after_step(
                    _heat_engine, _step_eval, _sp_len, _idea_count,
                )

            # G62+G70: detect_drift + TaskMetrics 滚动更新 — 复用 AV4 共享函数.
            # ponytail: update_drift_and_metrics 内部 try 兜住 import 失败,
            #   _metrics_ok=False 时 metrics 静默跳过 (跟原逻辑等价), drift 仍算.
            #   drift detected 的 print 保留 (共享函数只 logger.debug, advisory only).
            try:
                from huginn.autoloop.cognitive_loop import update_drift_and_metrics
                # target_chain_progress 取所有 chain 的平均 progress — 整体任务完成度.
                # ponytail: 算术平均, 不加权 (chain 重要性相近).
                _tc_prog = (
                    sum(getattr(tc, "progress", 0.0) for tc in _target_chains)
                    / len(_target_chains)
                ) if _target_chains else None
                _drift_info, _task_metrics = update_drift_and_metrics(
                    evals_history=_evals_history,
                    step_eval=_step_eval,
                    task_metrics=_task_metrics,
                    task_state=_task_state_for_metrics,
                    workspace=ws,
                    run_id=_task_id,
                    max_iterations=_max_exec_iters,
                    target_chain_progress=_tc_prog,
                )
                if _drift_info and _drift_info[0]:
                    print(f"[Step 2] drift detected: {_drift_info[1]}", flush=True)
            except Exception as _de:
                print(f"[Step 2] drift/metrics update skipped: {_de}", flush=True)
                _drift_info = None

            _eval_entry = {
                "iteration": _iter_n + 1,
                "ts": _time.time(),
                "role": "step_evaluation",
                "attempted": f"evaluate step {_iter_n + 1}",
                "found": f"on_track={_step_eval.on_track} evidence={_step_eval.evidence_quality}",
                "evidence": [],
                "limitations": [],
                "artifacts": [],
                "next_hint": "",
                "darwin_score": 0.0,
                "supported_ratio": 0.0,
                # v14 Task 1: step_eval = curl (critique-driven).
                "simplex_id": _make_simplex_id(_trace_task_id, _iter_n + 1, "step_evaluation"),
                "cochain_type": "curl",
                "domain": _infer_domain(_trace_task_id),
                "task_id": _trace_task_id,
                "model_version": _MODEL_VERSION,
            }
            # v15: upgrade_entry 补 v15 默认字段 (step_eval entry 不填 abduction 结果)
            try:
                from huginn.metacog.trace_topology import (
                    upgrade_entry as _upgrade_entry,
                )
                _upgrade_entry(_eval_entry)
            except Exception:
                logger.debug("step_eval trace entry upgrade skipped", exc_info=True)
            with _trace_path.open("a", encoding="utf-8") as _f:
                _f.write(_json.dumps(_eval_entry, ensure_ascii=False) + "\n")
            _cont, _msg = should_continue(_evals_history)
            if not _cont:
                print(f"[Step 2] should_continue=False: {_msg}", flush=True)
                # 不 break, 重定向提示拼到 _merge_hint, 下一轮 prompt 注入
                _merge_hint = (_merge_hint or "") + f"\n\n[REDIRECT] {_msg}\n"
                # 25.4: 工具异常时让 Reflector 介入, 文本拼到 _merge_hint.
                # 不破坏重定向逻辑, Reflector 文本是补充建议.
                if "Reflector" in _msg or "工具调用异常" in _msg:
                    try:
                        from huginn.metacog.reflector import (
                            format_reflector_text,
                            reflect,
                        )
                        _actions = reflect(
                            tool_call_health=getattr(
                                _step_eval, "tool_call_health", None),
                            last_step_evaluations=_evals_history,
                        )
                        _refl_text = format_reflector_text(_actions)
                        if _refl_text:
                            _merge_hint = (
                                (_merge_hint or "") + f"\n\n{_refl_text}\n"
                            )
                    except Exception as _re:
                        print(
                            f"[Step 2] reflector warning: {_re}", flush=True)
        except Exception as _e:
            print(f"[Step 2] step evaluator warning: {_e}", flush=True)

        # Task 29: 人机协同 pause 检查 (G71)
        # ponytail: RCB 是自动化测试环境, 真人在环是 v8 升级路径.
        #   这里把 pause/resume 接口跑通, 默认选 A 自动 resume. 失败只 warn 不阻塞.
        try:
            from huginn.runtime.task_lifecycle import (
                DecisionRequest,
                TaskLifecycle,
                TaskState,
                load_task_lifecycle,
                save_task_lifecycle,
            )
            # AV4: PMK 状态构建 + pause 判定走 cognitive_loop 共享函数.
            # _fired 在上面 ctx inject 块里定义, 正常路径一定有; 兜底 NameError
            try:
                _fired_local = _fired
            except NameError:
                _fired_local = []
            from huginn.autoloop.cognitive_loop import (
                build_pmk_state,
                check_pause_decision,
            )
            _pmk_state = build_pmk_state(
                persona, _last_step_eval, kb,
                since=ctx.run_start_iso,
                mem_mgr=_mem_mgr,
            )
            # P1-B: PMK 闭环反向边 — build_pmk_state 之前是只读快照, persona/
            # memory/kb 三路信号读完就丢. 现在把三路反向写回:
            #   1. PMK → persona.adaptive_layer (累计本轮 memory+kb 摘要)
            #   2. PMK → prospective memory (memory 段转 intention, 下轮 recall 触发)
            #   3. PMK → KB (darwin 上升时把本步 evidence 入库, 供后续 iter 检索)
            # ponytail: 三段反向边都用 try/except 兜底, 失败只 warn 不阻塞主循环.
            #   升级路径: 反向边触发频率/成功率为 PMK 闭环健康度指标.
            if _pmk_state is not None:
                try:
                    _cur_darwin = (
                        float(_entry.get("darwin_score", 0.5))
                        if "_entry" in dir() else 0.5
                    )
                    # 反向边 1: PMK memory+kb → persona.adaptive_layer
                    # 不覆盖原 adaptive_layer, 而是前缀拼接本轮 PMK 摘要.
                    if "_pm" in dir() and _pm is not None and persona is not None:  # noqa: F821
                        _pmk_summary_bits = []
                        if _pmk_state.get("memory"):
                            _pmk_summary_bits.append(
                                f"[iter {_iter_n+1} memory] {_pmk_state['memory'][:120]}")
                        if _pmk_state.get("kb"):
                            _pmk_summary_bits.append(
                                f"[iter {_iter_n+1} kb] {_pmk_state['kb'][:120]}")
                        if _pmk_summary_bits:
                            _old_adaptive = getattr(persona, "adaptive_layer", "") or ""
                            _new_adaptive = (
                                " | ".join(_pmk_summary_bits)
                                + (f" | {_old_adaptive}" if _old_adaptive else "")
                            )
                            # 截断防膨胀: 上限 800 字符, 老摘要自然滚出.
                            _new_adaptive = _new_adaptive[:800]
                            _pm.update(_persona_name, adaptive_layer=_new_adaptive)  # noqa: F821
                            print(
                                f"[PMK reverse edge] persona.adaptive_layer updated "
                                f"(len={len(_new_adaptive)})", flush=True)

                    # 反向边 2: PMK memory → prospective memory
                    # memory 段含上一步偏差描述, 转 intention 让下轮 recall 触发复核.
                    if _mem_mgr is not None and _pmk_state.get("memory"):
                        from huginn.memory.prospective import _new_intention_id
                        _mem_mgr.remember_prospective({
                            "intention_id": _new_intention_id(),
                            "description": (
                                f"PMK memory reverse edge: "
                                f"{_pmk_state['memory'][:160]}"
                            ),
                            "trigger_type": "dependency",
                            "trigger_payload": {"depends_on_step": _iter_n},
                            "priority": 3,
                            "created_at": _time.time(),
                            "source_step": _iter_n,
                        })

                    # 反向边 3: PMK → KB (darwin 上升时)
                    # darwin 上升 = 本步 evidence 有价值, 入库供后续 iter 检索.
                    # 下降/持平不入库, 避免污染 KB with 低质 evidence.
                    if (kb is not None and _prev_darwin is not None
                            and _cur_darwin > _prev_darwin
                            and "_entry" in dir()):
                        _evidence_text = (
                            f"[iter {_iter_n+1} darwin={_cur_darwin:.3f}] "
                            f"{(_entry.get('attempted') or '')[:200]}"
                        )
                        kb.add_text(
                            _evidence_text,
                            metadata={
                                "source": "pmk_reverse_edge",
                                "iter": _iter_n + 1,
                                "darwin_score": _cur_darwin,
                                "task_id": _task_id,
                            },
                        )
                        print(
                            f"[PMK reverse edge] KB add_text "
                            f"(darwin {_prev_darwin:.3f}→{_cur_darwin:.3f})",
                            flush=True)

                    _prev_darwin = _cur_darwin
                except Exception as _pmk_e:
                    print(
                        f"[PMK reverse edge] skipped: {_pmk_e}",
                        flush=True)
            _pause, _pause_reason, _pause_opts = check_pause_decision(
                _evals_history, _target_chains, kb,
                _fired_local, _pmk_state,
            )
            if _pause:
                _pause_step = _iter_n + 1
                _dr = DecisionRequest(
                    step_id=_pause_step,
                    question=_pause_reason,
                    options=_pause_opts,
                    context_summary=(
                        f"iter={_pause_step}, "
                        f"evals={len(_evals_history)}, "
                        f"kb={'empty' if kb is None else 'available'}"
                    ),
                )
                # 加载或新建 lifecycle, 确保 RUNNING 态才能 pause
                _lifecycle = load_task_lifecycle(_task_id, ws) or TaskLifecycle(
                    task_id=_task_id)
                if _lifecycle.state != TaskState.RUNNING:
                    try:
                        _lifecycle.transition(TaskState.RUNNING)
                    except ValueError:
                        # 终态/暂停态 → 新建一个干净的 lifecycle
                        _lifecycle = TaskLifecycle(task_id=_task_id)
                        _lifecycle.transition(TaskState.RUNNING)
                _lifecycle.pause_for_decision(_dr)
                save_task_lifecycle(_lifecycle, ws)
                print(
                    f"\n[human-in-loop] PAUSE step={_pause_step}: "
                    f"{_pause_reason}", flush=True)
                for _opt in _pause_opts:
                    print(
                        f"  {_opt.get('id', '?')}: {_opt.get('label', '')} "
                        f"(pros: {_opt.get('pros', '')}, "
                        f"cons: {_opt.get('cons', '')})", flush=True)
                # RCB 测试环境: 默认选 A 自动 resume, 不等真人
                _auto_choice = _pause_opts[0].get("id", "A") if _pause_opts else "A"
                _lifecycle.resume(answer=_auto_choice)
                save_task_lifecycle(_lifecycle, ws)
                print(
                    f"[human-in-loop] auto-resume with {_auto_choice}: "
                    f"{_pause_opts[0].get('label', '') if _pause_opts else ''}",
                    flush=True)
                # 记录到 meta_trace, role=human_decision
                try:
                    _hd_entry = {
                        "iteration": _pause_step,
                        "ts": _time.time(),
                        "role": "human_decision",
                        "attempted": f"pause: {_pause_reason}",
                        "found": (
                            f"auto-resume {_auto_choice}: "
                            f"{_pause_opts[0].get('label', '') if _pause_opts else ''}"
                        ),
                        "evidence": [],
                        "limitations": [
                            "RCB auto-resume, no real human in loop"],
                        "artifacts": [],
                        "next_hint": "continue after decision",
                        "darwin_score": 0.0,
                        "supported_ratio": 0.0,
                        # v14 Task 1: human_decision 不在 spec 三族主映射里 → legacy.
                        "simplex_id": _make_simplex_id(_trace_task_id, _iter_n + 1, "human_decision"),
                        "cochain_type": "legacy",
                        "domain": _infer_domain(_trace_task_id),
                        "task_id": _trace_task_id,
                        "model_version": _MODEL_VERSION,
                    }
                    # v15: upgrade_entry 补 v15 默认字段
                    try:
                        from huginn.metacog.trace_topology import (
                            upgrade_entry as _upgrade_entry,
                        )
                        _upgrade_entry(_hd_entry)
                    except Exception:
                        logger.debug("human_decision trace entry upgrade skipped", exc_info=True)
                    with _trace_path.open("a", encoding="utf-8") as _f:
                        _f.write(
                            _json.dumps(_hd_entry, ensure_ascii=False) + "\n")
                except Exception as _e:
                    print(
                        f"[human-in-loop] meta_trace write skipped: {_e}",
                        flush=True)
        except Exception as _e:
            print(f"[Step 2] human-in-loop pause warning: {_e}", flush=True)

        # Checkpoint 保存 (G59) — 每轮后落盘, 供下次 resume
        # P4 Task 26.6: 补 engine_state_digest (darwin_score + supported_ratio hash),
        # resume 时 resume_from_checkpoint 会校验防 drift.
        # ponytail: _darwin/_supported_ratio 从 _entry 取, _entry 在 line 1672 已创建,
        # 之前 _darwin 在 line 2052 才赋值导致 referenced-before-assignment.
        try:
            from huginn.runtime.checkpoint import save_checkpoint
            _tc_progress = {tc.target_id: tc.progress for tc in _target_chains}
            _pending = (
                [i.intention_id for i in _prospective_mem.list_pending()]
                if _prospective_mem is not None else []
            )
            _darwin_cp = float(_entry.get("darwin_score", 0.5)) if "_entry" in dir() else 0.5
            _sup_ratio_cp = float(_entry.get("supported_ratio", 0.0)) if "_entry" in dir() else 0.0
            _es_digest = _hashlib.md5(
                f"{_darwin_cp:.6f}|{_sup_ratio_cp:.6f}".encode(),
                usedforsecurity=False,
            ).hexdigest()
            save_checkpoint(
                task_id=_task_id,
                step_id=_iter_n + 1,
                phase="execute",
                workspace=ws,
                context_digest=_hashlib.md5((_ai_text or "").encode(), usedforsecurity=False).hexdigest(),
                memory_cursor=None,
                target_chain_progress=_tc_progress,
                prospective_queue=_pending,
                engine_state_digest=_es_digest,
            )
        except Exception as _e:
            print(f"[Step 2] checkpoint save warning: {_e}", flush=True)

        # 跨任务 iteration_result 写入 — curiosity hint 的数据源.
        # status 由 darwin_score 映射: >=0.5 supported, <0.5 refuted.
        # persona_id 用 _persona_name, 让 get_self_model() 按 persona 聚合.
        # ponytail: 只在 cross-task db 开时写, 单任务模式跳过 (无读者).
        if _mem_mgr is not None and os.environ.get("HUGINN_RCB_CROSS_TASK", "1") == "1":
            try:
                _darwin = float(_entry.get("darwin_score", 0.5)) if "_entry" in dir() else 0.5
                _typed_status = "supported" if _darwin >= 0.5 else "refuted"
                _mem_mgr.remember_typed(
                    content=f"RCB iter {_iter_n+1} task={ws.name} darwin={_darwin:.2f} supported_ratio={float(_entry.get('supported_ratio', 0.0)) if '_entry' in dir() else 0.0:.2f}",
                    memory_type="iteration_result",
                    run_id=ws.name,
                    persona_id=ctx.persona_name,
                    status=_typed_status,
                    importance=_darwin,
                    tier="mid",
                    tags=["rcb", f"task:{ws.name}", f"iter:{_iter_n+1}"],
                    source="rcb_runner",
                )
            except Exception as _me:
                print(f"[Memory] iteration_result write skipped: {_me}", flush=True)

        # 停滞检测: report.md 内容 hash 不变 → 升温分叉探索, 不再 break.
        # C7: 删 stagnation early-stop — 与 P5 "跑满 timeout" 矛盾, 是早交卷根因.
        #   升温 + imagination/pivot 分流负责重热, wall_clock 守卫唯一控制退出.
        _curr_hash = (
            _hashlib.md5(_report_text.encode(), usedforsecurity=False).hexdigest()
            if _report_text else None
        )
        if _curr_hash == _prev_report_hash and _curr_hash is not None:
            _stagnation_count += 1
            # 停滞重热: 报告没变化 = 轨迹卡住, 升温让下轮分叉探索
            _t_hot = min(1.0, _t_hot + 0.5)
            print(
                f"[stagnation: report.md unchanged for {_stagnation_count} iters, heating]",
                flush=True,
            )
        else:
            _stagnation_count = 0
        _prev_report_hash = _curr_hash

        # P4 Task 21: score_history 滑动窗口 — push 本轮 darwin_score, 只监测.
        # C7: 早退已删, 仅记 stats + 日志, wall_clock 守卫唯一控制退出.
        if _score_history is not None:
            try:
                _sh_darwin = (
                    _entry.get("darwin_score", 0.5) if "_entry" in dir() else 0.5
                )
                _score_history.push(_sh_darwin)
                _sh_temp = _score_history.get_temperature()
                _sh_stats = _score_history.stats()
                if _sh_stats["n_samples"] >= 2:
                    print(
                        f"[score_history] iter {_iter_n}: temp={_sh_temp:.2f} "
                        f"mean={_sh_stats['mean']:.3f} var={_sh_stats['variance']:.3f} "
                        f"streak={_sh_stats['monotonic_streak']}",
                        flush=True,
                    )
                # C7: 删 monotonic-decrease early-stop — 同 stagnation, 与 P5 矛盾.
                #   保留监测 + 日志, wall_clock 守卫唯一控制退出.
                if _score_history.should_terminate():
                    print(
                        f"[score_history] monotonic decrease "
                        f"{_sh_stats['monotonic_streak']} iters (advisory, not stopping)",
                        flush=True,
                    )
            except Exception as _she:
                print(f"[score_history push skipped: {_she}]", flush=True)

        # P4 Task 22: cumulative output audit — 每轮扫 outputs/, 累积越权就阻断.
        # ponytail: 中间步无害但累积成论文复现包越权产出才拦, 单步不拦.
        if _cumulative_auditor is not None:
            try:
                _audit_res = _cumulative_auditor.audit_step(
                    ws / "outputs", task_type=_audit_task_type,
                )
                if _audit_res.get("blocked"):
                    print(
                        f"[cumulative_audit] BLOCKED at iter {_iter_n}: "
                        f"{_audit_res.get('reason')}",
                        flush=True,
                    )
                    break
            except Exception as _cae:
                print(f"[cumulative_audit skipped: {_cae}]", flush=True)

        # 认知反馈: darwin ratchet 检查 stagnation, 触发 pivot/counterexample/stop 分流.
        # ponytail: 顶层函数不依赖 AutoloopEngine 实例, hypothesis_graph=None 退化到
        #   score 阈值. advisory only — pivot/counterexample 不 break, 让 agent 继续
        #   (stagnation 早退逻辑上面已处理). 只有 action=stop 才 break.
        try:
            from huginn.autoloop.cognitive_loop import (
                classify_stall,
                darwin_ratchet_check,
            )
            _dr_darwin = _entry.get("darwin_score", 0.5) if "_entry" in dir() else 0.5
            _dr_ratio = _entry.get("supported_ratio", 0.0) if "_entry" in dir() else 0.0
            _action, _state = darwin_ratchet_check(
                darwin_score=_dr_darwin,
                supported_ratio=_dr_ratio,
                stagnation_count=_stagnation_count,
            )
            if _action == "stop":
                print(
                    f"[Step2] darwin_ratchet: stop at iter {_iter_n}, "
                    f"score={_dr_darwin:.3f}",
                    flush=True,
                )
                break
            elif _action in ("pivot", "counterexample"):
                _stall_type = classify_stall([_dr_darwin], [_dr_ratio])
                print(
                    f"[Step2] darwin_ratchet: {_action} "
                    f"(stall_type={_stall_type}) at iter {_iter_n}",
                    flush=True,
                )
                # 不 break, 让 agent 继续 (advisory only)
            _stagnation_count = _state.get("stagnation_count", _stagnation_count)
        except Exception as _e:
            print(f"[Step2] darwin_ratchet_check failed: {_e}", flush=True)

        # P1-A: manifold MCMC 接入主循环 — 每 _mcmc_interval 步跑一次 mcmc_step.
        # 之前 mcmc_step 只在 _selfcheck 调, 主循环从不调, manifold 是静态的.
        # ponytail: MCMC 是 advisory only, 不强制 agent 用 MCMC 选的 h.
        #   升级路径: MCMC 接受率作为 exploration vs exploitation 信号注入 bandit.
        if (_hypo_manifold is not None
                and _iter_observations
                and _iter_n % _mcmc_interval == 0):
            try:
                # 初始化 _mcmc_current: 用本轮 best_h_id (posterior 最高的)
                if _mcmc_engine._mcmc_current is None \
                        and "_iter_best_h_id" in dir() \
                        and _iter_best_h_id is not None:
                    _mcmc_engine._mcmc_current = _iter_best_h_id
                if _mcmc_engine._mcmc_current is not None \
                        and len(_hypo_manifold._hyp) >= 2:
                    _mcmc_prev = _mcmc_engine._mcmc_current
                    # Step 2: 增量路径 — cached_log_p 跨步复用, 不再调 log_posterior 全量
                    # 触觉层: 只在 extreme 模式 (_mcmc_haptic_enabled=True) 传 haptic 参数,
                    # 非 extreme 不传, mcmc_step 走默认 haptic_enabled=False (行为不变).
                    _mcmc_step_kwargs = {
                        "rng": _mcmc_engine._mcmc_rng,
                        "cached_log_p_current": _mcmc_cached_log_p,
                    }
                    if _mcmc_haptic_enabled:
                        _mcmc_step_kwargs["haptic_enabled"] = True
                        _mcmc_step_kwargs["haptic_temperature"] = _mcmc_haptic_temperature
                    if _mcmc_alignment_enabled:
                        _mcmc_step_kwargs["alignment_enabled"] = True
                        _mcmc_step_kwargs["alignment_temperature"] = _mcmc_alignment_temperature
                    _next_h, _next_log_p = _hypo_manifold.mcmc_step(
                        _iter_observations, _mcmc_engine._mcmc_current,
                        **_mcmc_step_kwargs,
                    )
                    _mcmc_engine._mcmc_current = _next_h
                    _mcmc_cached_log_p = _next_log_p
                    _mcmc_accepted = (_mcmc_engine._mcmc_current != _mcmc_prev)
                    if _mcmc_accepted:
                        _mcmc_engine._mcmc_accept_count += 1
                    _mcmc_engine._mcmc_step_count += 1
                    _mcmc_engine._iteration = _iter_n
                    # #2 打通: MCMC 接受率注入 bandit 作为探索/利用信号.
                    # 每 _mcmc_interval 步只跑一次 step, 接受率是单步 0/1;
                    # bandit 内部做滑动平均平滑. 失败静默, 不阻塞 MCMC.
                    try:
                        from huginn.agent.bandit_controller import EffortBandit
                        EffortBandit.get_instance().update_mcmc_acceptance(
                            1.0 if _mcmc_accepted else 0.0
                        )
                    except Exception:
                        logger.debug("bandit mcmc acceptance update skipped", exc_info=True)
                    _mcmc_llh = _next_log_p  # mcmc_step 已返回, 不再重算
                    print(
                        f"[mcmc] iter {_iter_n}: "
                        f"{'accepted' if _mcmc_accepted else 'rejected'} "
                        f"h={_mcmc_engine._mcmc_current} llh={_mcmc_llh:.3f}",
                        flush=True)
                    _mcmc_current_info = {
                        "accepted": _mcmc_accepted,
                        "h_id": _mcmc_engine._mcmc_current,
                        "llh": _mcmc_llh,
                    }
                    # 周期 checkpoint — 每 HUGINN_MCMC_CHECKPOINT_INTERVAL 步落盘
                    # ponytail: save_engine_state 直接调, 失败只 warn 不阻塞 MCMC
                    if _mcmc_ckpt_interval > 0 and \
                            _mcmc_engine._mcmc_step_count % _mcmc_ckpt_interval == 0:
                        try:
                            from huginn.runtime.engine_state import save_engine_state
                            _mcmc_engine._mcmc_rng_state = \
                                _mcmc_engine._mcmc_rng.getstate()
                            save_engine_state(_mcmc_engine, _task_id, ws)
                        except Exception:
                            print(
                                "[mcmc] checkpoint save failed (non-fatal)",
                                flush=True)
                else:
                    _mcmc_current_info = None
            except Exception as _mcmc_e:
                print(f"[mcmc] skipped: {_mcmc_e}", flush=True)
                _mcmc_current_info = None
        else:
            _mcmc_current_info = None

        # P1-C: 周期性完成度审计 — 每 _completion_interval 步强制跑一次,
        # 不等 agent 声称 TASK COMPLETE. 长任务里 agent 一直不说完成时, 审计门
        # 也能定期跑, gap 写 cognitive_evidence.md + 注入下轮 prompt.
        # ponytail: 复用 metacog_check_completion, 不新写审计逻辑. advisory only.
        _completion_audit = None
        if _iter_n > 0 and _iter_n % _completion_interval == 0:
            try:
                from huginn.autoloop.cognitive_loop import metacog_check_completion
                _rep_md = _report_text or (
                    _report_path_iter.read_text(encoding="utf-8")
                    if _report_path_iter.exists() else ""
                )
                _completion_audit = metacog_check_completion(
                    report_md=_rep_md,
                    outputs_dir=ws / "outputs",
                )
                if not _completion_audit.get("passed", True):
                    _gaps = _completion_audit.get("block_reasons", [])
                    print(
                        f"[completion audit iter {_iter_n}] NOT passed: {_gaps}",
                        flush=True,
                    )
                    _prev_completion_hint = (
                        f"\n\n## Completion Audit Advisory (iter {_iter_n})\n"
                        f"Auto-audit found gaps: {_gaps}. "
                        f"Address these before claiming TASK COMPLETE."
                    )
                else:
                    print(
                        f"[completion audit iter {_iter_n}] passed",
                        flush=True,
                    )
                    _prev_completion_hint = ""
            except Exception as _e:
                print(f"[completion audit skipped: {_e}]", flush=True)

        # 触觉层: 同构不同性异常检测 — 每 _completion_interval 步跑一次.
        # detect_isomorphic_anomaly 抓 "结构同 + 力学不同" (石墨 vs 金刚石),
        # trigger_isomorphic_anomaly_hypothesis 生成解释差异的新 hypothesis.
        # ponytail: advisory only, 失败只 warn 不阻塞. 结果写 cognitive_evidence.
        _anomaly_info = None
        if _hypo_manifold is not None and _iter_n > 0 \
                and _iter_n % _completion_interval == 0:
            try:
                _anomaly_pairs = _hypo_manifold.detect_isomorphic_anomaly()
                _anomaly_generated: list[str] = []
                if _anomaly_pairs:
                    _anomaly_generated = await _trigger_anomaly_hypothesis(
                        _anomaly_pairs, model)
                if _anomaly_pairs or _anomaly_generated:
                    _anomaly_info = {
                        "pairs": _anomaly_pairs,
                        "generated": _anomaly_generated,
                    }
                    print(
                        f"[anomaly] iter {_iter_n}: {len(_anomaly_pairs)} pair(s), "
                        f"{len(_anomaly_generated)} new hypothesis", flush=True)
            except Exception as _e:
                print(f"[anomaly] detection skipped: {_e}", flush=True)

        # P0-C: 每 iter 结束写 cognitive_evidence.md, 让 score.py judge 能看到
        # agent 跑分过程中的认知层证据. ponytail: 追加写, 失败只 warn 不阻塞.
        _write_cognitive_evidence(
            ws, _iter_n,
            entry=_entry if "_entry" in dir() else None,
            pmk_state=_pmk_state if "_pmk_state" in dir() else None,
            hypo_manifold=_hypo_manifold if "_hypo_manifold" in dir() else None,
            heat_engine=_heat_engine if "_heat_engine" in dir() else None,
            bandit_controller=_bandit if "_bandit" in dir() else None,
            mcmc_info=_mcmc_current_info if "_mcmc_current_info" in dir() else None,
            completion_audit=_completion_audit if "_completion_audit" in dir() else None,
            anomaly_info=_anomaly_info if "_anomaly_info" in dir() else None,
        )

        # 早速: agent 明确说完成 — P0.3: 先过 RCB effort floor 硬下限.
        # 防止 agent 一轮就收敛到"看起来完整"的 report.md 但 checklist 还缺关键项.
        # AV7 autoloop _validate 已接 MinEffortFloor, RCB 路径对齐.
        # P5 守卫: wall_clock 未耗尽才允许 TASK COMPLETE break.
        # v16: TASK COMPLETE 时强制重跑 derivation audit (即使非 3 轮调度点),
        # 因这是最后一次放行机会. audit 失败 (空串) 不阻塞, fallback 到 keyword.
        if _ai_text and "TASK COMPLETE" in _ai_text.upper() and not (ctx.wall_expired_fn and ctx.wall_expired_fn()):
            # 连续驳回计数: agent 反复 'TASK COMPLETE' 不调工具时, 第 N+1 次接受.
            # 修 Math_003 iter 2-5 反复驳回循环: agent 已尽力, gates 太严只会空转.
            _consecutive_complete_rejections += 1
            _force_accept = _consecutive_complete_rejections > _MAX_COMPLETE_REJECTIONS
            if _force_accept:
                print(
                    f"[effort floor] 接受 TASK COMPLETE: 连续驳回 "
                    f"{_consecutive_complete_rejections - 1} 次达上限 "
                    f"{_MAX_COMPLETE_REJECTIONS}. agent 已尽力, 强制收尾.",
                    flush=True,
                )
            _final_derivation = _derivation_audit
            if not _force_accept and model and not _final_derivation:
                try:
                    _final_derivation = await _derivation_chain_audit(
                        model, ws, checklist,
                        _report_coverage_compass(ws, checklist) or "",
                    ) or ""
                except Exception as _e:
                    print(f"[v16] final derivation audit skipped: {_e}", flush=True)
            _eff_ok, _eff_reason = (True, "force_accept skipped") if _force_accept else _rcb_effort_floor(
                ws, checklist, derivation_audit=_final_derivation or None,
            )
            if not _eff_ok:
                print(
                    f"[effort floor] TASK COMPLETE 被驳回: {_eff_reason}. "
                    f"继续迭代补缺.",
                    flush=True,
                )
                # 把驳回原因作为下一轮 prompt, agent 必须先补缺再声称完成.
                _iter_prompt_override = (
                    f"Previous TASK COMPLETE was rejected by effort floor: "
                    f"{_eff_reason}. Address the MISSING items and re-claim "
                    f"TASK COMPLETE only when report.md covers them."
                )
                # 覆盖下一轮的 _iter_prompt (否则 agent 会继续说 TASK COMPLETE)
                # ponytail: 直接改 _iter_prompt 变量, 下一轮 for 循环用它
                with contextlib.suppress(NameError):
                    _iter_prompt = (
                        f"Continue execution. Iteration {_iter_n + 2}/{_max_exec_iters}.\n"
                        f"{_iter_prompt_override}\n\n"
                        f"Review the Research Trace section and Coverage Compass above."
                    )
                # 不 break, 继续下一轮
                continue
            # Task 5+10: 反完成审计 — 4 层完成度 + 拓扑坍缩. 任一阻断 → continue.
            # ponytail: 顶层函数不依赖 engine, hypothesis_graph=None 退化到启发式.
            #   阻断时覆盖下一轮 prompt, 让 agent 补缺而非重复 TASK COMPLETE.
            _metacog_blocked = False
            if not _force_accept:
                try:
                    from huginn.autoloop.cognitive_loop import (
                        metacog_check_completion,
                        metacog_check_selection_bias,
                        metacog_check_topology_collapse,
                    )
                    # _report_text 在 iter 头部算, 可能空; 兜底重读文件.
                    _rep_md = _report_text or (
                        _report_path_iter.read_text(encoding="utf-8")
                        if _report_path_iter.exists() else ""
                    )
                    _completion = metacog_check_completion(
                        report_md=_rep_md,
                        outputs_dir=ws / "outputs",
                    )
                    # P1-C: 别名给 final cognitive evidence write 用 (修变量名不匹配 bug)
                    _completion_audit = _completion
                    if not _completion.get("passed", True):
                        print(
                            f"[TaskComplete] blocked by metacog_check: "
                            f"{_completion.get('block_reasons', [])}",
                            flush=True,
                        )
                        _metacog_blocked = True
                    # 拓扑层反完成审计: hypothesis space 多样性检测.
                    _topo = metacog_check_topology_collapse(
                        hypothesis_graph=None, hypothesis_list=None,
                    )
                    if _topo.get("collapsed", False):
                        print(
                            f"[TaskComplete] blocked by topology collapse: "
                            f"{_topo.get('reason')}",
                            flush=True,
                        )
                        _metacog_blocked = True
                    # 选择偏差反完成审计: 样本系统性缺一类 (幸存者偏差) 时阻断.
                    # 用本轮收集的 observations (manifold 空 / 无观测时自动不命中).
                    _sb = metacog_check_selection_bias(
                        observations=_iter_observations if "_iter_observations" in dir()
                        else None,
                    )
                    if _sb.get("biased", False):
                        print(
                            f"[TaskComplete] blocked by selection bias: "
                            f"{_sb.get('reason')}",
                            flush=True,
                        )
                        _metacog_blocked = True
                except Exception as _e:
                    print(f"[TaskComplete] metacog audit failed: {_e}", flush=True)
            if _metacog_blocked:
                # 阻断时覆盖下一轮 prompt, 让 agent 补缺而非重复 TASK COMPLETE
                with contextlib.suppress(NameError):
                    _iter_prompt = (
                        f"Continue execution. Iteration {_iter_n + 2}/{_max_exec_iters}.\n"
                        f"Previous TASK COMPLETE blocked by metacog audit "
                        f"(completion/topology). Address the gaps and re-claim "
                        f"TASK COMPLETE when done.\n\n"
                        f"Review the Research Trace section and Coverage Compass above."
                    )
                continue
            print("[agent signalled TASK COMPLETE, breaking]", flush=True)
            # P0-C: TASK COMPLETE 时写 final cognitive evidence snapshot.
            _write_cognitive_evidence(
                ws, _iter_n,
                entry=_entry if "_entry" in dir() else None,
                pmk_state=_pmk_state if "_pmk_state" in dir() else None,
                hypo_manifold=_hypo_manifold if "_hypo_manifold" in dir() else None,
                heat_engine=_heat_engine if "_heat_engine" in dir() else None,
                bandit_controller=_bandit if "_bandit" in dir() else None,
                completion_audit=_completion_audit if "_completion_audit" in dir() else None,
                mcmc_info=_mcmc_current_info if "_mcmc_current_info" in dir() else None,
                anomaly_info=_anomaly_info if "_anomaly_info" in dir() else None,
                is_final=True,
            )
            break

    return _evals_history
