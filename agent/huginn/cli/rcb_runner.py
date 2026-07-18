"""RCB harness 入口: 读 workspace/INSTRUCTIONS.md, 跑 huginn agent, 输出到 stdout.

ResearchClawBench 的 TaskRunner 通过 subprocess 跑 agent_cmd, 捕获 stdout.
本脚本作为 huginn 的 RCB adapter:
  python huginn/cli/rcb_runner.py --workspace <workspace>

agent 在 workspace 里工作 (cwd=workspace), 用 code_tool/bash_tool 读写文件,
最终产出 report/report.md. RCB 的 INSTRUCTIONS.md 模板已经很详细, system
prompt 只需简短研究导向.

ponytail: 不重复 RCB prompt 已有的内容, 不加交互式渲染, 纯文本输出.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 在 import huginn 之前关掉秒级限流 — RCB 任务的 prompt 长 + 工具多,
# 默认 5000 tokens/s 会在第一轮就超限. RCB 是离线评测, 不需要限流.
os.environ.setdefault("HUGINN_RATE_LIMIT_ENABLED", "0")
os.environ.setdefault("HUGINN_RATE_LIMIT_TOKENS_PER_TURN", "500000")
# 允许本地沙箱执行 code_tool/bash_tool — RCB subprocess 没有 docker, 用本地 python
os.environ.setdefault("HUGINN_ALLOW_LOCAL_BASH", "1")
# HUGINN_CACHE_DIR 可能被设成空串, 导致 LongTermMemory 用相对路径 memory.db,
# 在 RCB workspace cwd 下 sqlite WAL 创建失败. 强制用绝对路径.
if not os.environ.get("HUGINN_CACHE_DIR"):
    os.environ["HUGINN_CACHE_DIR"] = str(Path.home() / ".huginn")
# RCB 场景用 CSM 子集: 3-step 映射 S1/S4/S6+S7, 不再全 skip (Task 18, R8 减法修正).
# ponytail: S7 自修改仍走 (Task 2), 只跳过 compaction — 见 reflection.py L245.
os.environ.setdefault("HUGINN_RCB_CSM_SUBSET", "1")
# RCB 场景 compaction 保留前 2 条 root (task + Step 1 checklist) — 修同伦断裂 (σ₂)
os.environ.setdefault("HUGINN_KEEP_ROOT_N", "2")
# F3: σ₂ 半修补全 — 位置切片保不到 Step 1 checklist prompt (在 msgs[2:4]),
# 改用内容 marker 标 root. 同时保 FCM winner_plan (F4) 和 compass 不被 compaction drop.
# ponytail: marker 选 ## 标题, 足够独特不会误匹配普通正文.
os.environ.setdefault(
    "HUGINN_ROOT_MARKERS",
    "## Methodology Checklist;## Selected Execution Plan;## Report Coverage Compass;## Intuitive Gamer",
)
# RCB 场景跳过 Rust sandbox — 它在 RDKit+sklearn GPR 场景静默崩溃返回空 stderr
os.environ.setdefault("HUGINN_NO_RUST_SANDBOX", "1")
# RCB 跑分关掉 LLM decider — 跑分需要确定性, run_cognitive 的规则版 decide_fn 已够用.
# 生产路径 (deli_research/cli/routes) 不设这个变量, 默认开 LLM decider.
os.environ.setdefault("HUGINN_COGNITIVE_LLM_DECIDER", "0")
# RCB 场景关熔断器 — file_read_tool 误触发 circuit_open 阻止 agent 读文件 (σ₇)
os.environ.setdefault("HUGINN_HEALTH_MONITOR", "0")
# RCB 场景关循环检测 — agent 反复跑 code_tool 是正常行为, 误判为 loop (σ₈)
os.environ.setdefault("HUGINN_SKIP_LOOP_DETECTOR", "1")



# === 认知原语: adversarial_critique + critique_decision (抽到 rcb_critique.py) ===
# ponytail: 单一职责拆分, 减少 rcb_runner.py 行数. 原 L56-513 抽到 rcb_critique.py.
from huginn.cli.rcb_critique import (
    adversarial_critique, critique_decision, format_critique_for_agent,
    Decision, CritiqueResult,
)


# === 认知原语: fork_critique_merge (FCM) — verifier 下沉到 plan 决策点 ===

# === 认知原语: fork_critique_merge (FCM) 抽到 rcb_fork_merge.py ===
# ponytail: 单一职责拆分. 原 L62-357 抽到 rcb_fork_merge.py.
from huginn.cli.rcb_fork_merge import (
    fork_critique_merge, anneal_fork_count,
    _extract_sci_numbers, _collect_artifact_numbers,
    _reproduction_gate, judge_fork_reports,
    _FCM_PERSPECTIVES,
)

@dataclass
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
    step2_prompt = (
        "Now execute the task following your methodology checklist. "
        "Implement each [EXACT] component as-specified in the paper. "
        "If a component fails, debug and push through — do NOT silently substitute a simpler model. "
        "Write report/report.md with your results, referencing the checklist items you covered. "
        "Use file_write_tool for report.md, code_tool for analysis/plotting, bash_tool for running scripts."
        + _scan_hint + _fcm_hint
    )

    import hashlib as _hashlib
    import json as _json
    import time as _time
    _trace_path = ws / ".huginn" / "meta_trace.jsonl"
    _trace_path.parent.mkdir(parents=True, exist_ok=True)
    _max_exec_iters = int(os.environ.get(
        "HUGINN_RCB_EXEC_ITERS",
        "4" if extreme else "2",
    ))
    _prev_report_hash: str | None = None
    _stagnation_count = 0

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
        from huginn.runtime.task_metrics import (
            TaskMetrics, load_metrics, save_metrics, update_metrics,
        )
        from types import SimpleNamespace as _NS
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
            pass
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

    # Task 3: 从 resume 的 iter 开始, 不重跑已 checkpoint 的轮次
    for _iter_n in range(_resume_from_iter, _max_exec_iters):
        if _iter_n == 0:
            _iter_prompt = step2_prompt
        else:
            _iter_prompt = (
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
            try:
                _compass = _report_coverage_compass(ws, checklist)
                # v8: 每 5 轮做一次 LLM 语义深度审计 (规则版每轮都跑, LLM 版成本高)
                # 解决规则版 keyword 命中漏同义改写的天花板 (MAE vs mean absolute error)
                if _iter_n > 0 and _iter_n % 5 == 0 and model and _compass:
                    _llm_compass = await _llm_coverage_audit(
                        model, ws, checklist, _compass,
                    )
                    if _llm_compass:
                        _compass = _llm_compass
                if _compass:
                    _iter_prompt += "\n\n" + _compass
            except Exception:
                pass  # compass 是增强, 失败不阻塞

            # F4: FCM winner_plan 每轮提醒 — Step 1.7 选出的执行方案只在 iter 0
            # 注入 step2_prompt, compaction 后会丢失. 每轮追加避免 agent 漂移到
            # rejected fork 的思路上. ponytail: 不重复 merge_insights (iter 0 已有).
            _fcm_winner = (fcm.get("winner_plan") or "").strip() if fcm else ""
            if _fcm_winner:
                _iter_prompt += (
                    "\n\n## Selected Execution Plan (reminder — Step 1.7 FCM winner)\n"
                    + _fcm_winner[:1200]
                )

            # F5: KB chunk 注入 — 让 RAG 真 augment RCB 生成.
            # 之前 rcb_runner 每轮查 KB 只用于 PMK pause 决策, 检索结果不进 prompt.
            # 现在每轮基于上一轮 attempted 做 KB 检索, top-2 chunk 注入 prompt.
            # ponytail: top_k=2 控成本, 截 400 字防 prompt 膨胀. 失败只跳过.
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
                    _kb_hits = kb.query(_gap_query, top_k=2) or []
                    _kb_chunks = []
                    for _h in _kb_hits[:2]:
                        _txt = _h.get("content", "") if isinstance(_h, dict) else str(_h)
                        if _txt:
                            _kb_chunks.append(_txt[:400])
                    if _kb_chunks:
                        _iter_prompt += (
                            "\n\n## Domain Knowledge (KB retrieval, top-2)\n"
                            + "\n---\n".join(_kb_chunks)
                        )
                except Exception:
                    pass
        _iter_prompt += _merge_hint
        _merge_hint = ""

        # P1: 想象力机制 — 每轮检查 should_imaginate, True 时注入 imagination block.
        # 触发条件: Re_cog > Re_crit (概念湍流) 或 T_hot > 0.7 (高熵).
        # ponytail: 失败只跳过, 不阻塞主流程. heat_engine.update_kinematics 在
        #   StepEvaluator 后调 (有 idea_count/stable_principles_count 数据才更新).
        if _heat_engine is not None:
            try:
                if _heat_engine.should_imaginate(_iter_n):
                    _iter_prompt += _IMAGINATION_BLOCK
                    print(f"[Step 2] imagination triggered at iter {_iter_n}", flush=True)
            except Exception as _ie:
                print(f"[should_imaginate skipped: {_ie}]", flush=True)

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
                }
                with _trace_path.open("a", encoding="utf-8") as f:
                    f.write(_json.dumps(_tfm_entry, ensure_ascii=False) + "\n")
            except Exception as _e:
                print(f"[tfm trace skipped: {_e}]", flush=True)
        else:
            print(f"\n--- Step 2 iter {_iter_n + 1}/{_max_exec_iters} ---\n", flush=True)
            _ai_text = await ctx.stream_chat_fn(_iter_prompt, f"step2_iter{_iter_n + 1}")

        # 退火降温 (在停滞重热之前, 停滞信号下一轮生效)
        _t_hot = max(0.0, _t_hot * 0.5)

        # 写 Meta-Trace entry — P1 的 build_meta_trace_text 下一轮会读到.
        # ponytail: 字段从 self/agent 状态抽, 不调 LLM. RCB mini-loop 不跑 darwin
        #   ratchet, darwin_score/supported_ratio 留 0 (trace 段仍显示 iteration).
        try:
            _report_text = ""
            _report_path_iter = ws / "report" / "report.md"
            if _report_path_iter.exists():
                _report_text = _report_path_iter.read_text(encoding="utf-8")
            _entry = {
                "iteration": _iter_n + 1,
                "ts": _time.time(),
                "role": "rcb_exec",
                "attempted": (_iter_prompt[:200]).replace("\n", " "),
                "found": (_ai_text or "")[:300],
                "evidence": [_report_text[:150]] if _report_text else [],
                "limitations": [],
                "artifacts": ["report/report.md"] if _report_path_iter.exists() else [],
                "next_hint": "continue execution" if _iter_n < _max_exec_iters - 1 else "step3 critique",
                "darwin_score": 0.0,
                "supported_ratio": 0.0,
            }
            with _trace_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(_entry, ensure_ascii=False) + "\n")
        except Exception as _e:
            print(f"[meta_trace write skipped: {_e}]", flush=True)

        # StepEvaluator 评估 + Checkpoint 保存 (G63 + G59)
        # 失败只 warn, 不影响主循环. _entry 可能因 meta_trace try 失败而未定义,
        # 用 try 兜住 NameError.
        try:
            from huginn.metacog.step_evaluator import (
                ToolCallHealth, evaluate_step, should_continue,
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

            # G62: detect_drift — 连续 window 步 on_track=false → 漂移告警.
            # 结果缓存到 _drift_info, 下一轮 build_meta_agent_text 会读到 Adviser 段.
            # ponytail: 直接传 _evals_history (list[StepEvaluation]), detect_drift
            #   已兼容 dataclass 对象. 失败只 warn, 不影响 should_continue / Reflector.
            try:
                from huginn.metacog.target_chain import detect_drift as _detect_drift
                _drift_info = _detect_drift(_evals_history, window=3)
                if _drift_info[0]:
                    print(
                        f"[Step 2] drift detected: {_drift_info[1]}",
                        flush=True)
            except Exception as _de:
                print(f"[Step 2] detect_drift skipped: {_de}", flush=True)
                _drift_info = None

            # G70: TaskMetrics 滚动更新 + 落盘.
            # target_chain_progress 取所有 chain 的平均 progress — 整体任务完成度.
            # ponytail: 用算术平均, 不加权 (chain 之间重要性相近). 失败只 warn.
            if _metrics_ok and _task_metrics is not None:
                try:
                    _tc_prog = (
                        sum(getattr(tc, "progress", 0.0) for tc in _target_chains)
                        / len(_target_chains)
                    ) if _target_chains else None
                    _task_metrics = update_metrics(
                        _task_metrics, _step_eval,
                        task_state=_task_state_for_metrics,
                        target_chain_progress=_tc_prog,
                    )
                    save_metrics(_task_metrics, ws)
                except Exception as _me:
                    print(f"[Step 2] metrics update skipped: {_me}", flush=True)

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
            }
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
                            reflect, format_reflector_text,
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
                TaskLifecycle, TaskState, DecisionRequest,
                save_task_lifecycle,
                load_task_lifecycle,
            )
            # AV4: PMK 状态构建 + pause 判定走 cognitive_loop 共享函数.
            # _fired 在上面 ctx inject 块里定义, 正常路径一定有; 兜底 NameError
            try:
                _fired_local = _fired
            except NameError:
                _fired_local = []
            from huginn.autoloop.cognitive_loop import (
                build_pmk_state, check_pause_decision,
            )
            _pmk_state = build_pmk_state(persona, _last_step_eval, kb)
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
                    }
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
        try:
            from huginn.runtime.checkpoint import save_checkpoint
            _tc_progress = {tc.target_id: tc.progress for tc in _target_chains}
            _pending = (
                [i.intention_id for i in _prospective_mem.list_pending()]
                if _prospective_mem is not None else []
            )
            save_checkpoint(
                task_id=_task_id,
                step_id=_iter_n + 1,
                phase="execute",
                workspace=ws,
                context_digest=_hashlib.md5((_ai_text or "").encode()).hexdigest(),
                memory_cursor=None,
                target_chain_progress=_tc_progress,
                prospective_queue=_pending,
            )
        except Exception as _e:
            print(f"[Step 2] checkpoint save warning: {_e}", flush=True)

        # 停滞检测: report.md 内容 hash 不变 → 可能卡住, 早停
        _curr_hash = (
            _hashlib.md5(_report_text.encode()).hexdigest()
            if _report_text else None
        )
        if _curr_hash == _prev_report_hash and _curr_hash is not None:
            _stagnation_count += 1
            # 停滞重热: 报告没变化 = 轨迹卡住, 升温让下轮分叉探索
            _t_hot = min(1.0, _t_hot + 0.5)
            if _stagnation_count >= 2:
                print(
                    f"[stagnation: report.md unchanged for {_stagnation_count} iters, breaking]",
                    flush=True,
                )
                break
        else:
            _stagnation_count = 0
        _prev_report_hash = _curr_hash

        # 早停: agent 明确说完成 — P0.3: 先过 RCB effort floor 硬下限.
        # 防止 agent 一轮就收敛到"看起来完整"的 report.md 但 checklist 还缺关键项.
        # AV7 autoloop _validate 已接 MinEffortFloor, RCB 路径对齐.
        if _ai_text and "TASK COMPLETE" in _ai_text.upper():
            _eff_ok, _eff_reason = _rcb_effort_floor(ws, checklist)
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
                try:
                    _iter_prompt = (
                        f"Continue execution. Iteration {_iter_n + 2}/{_max_exec_iters}.\n"
                        f"{_iter_prompt_override}\n\n"
                        f"Review the Research Trace section and Coverage Compass above."
                    )
                except NameError:
                    pass
                # 不 break, 继续下一轮
                continue
            print("[agent signalled TASK COMPLETE, breaking]", flush=True)
            break

    return _evals_history


def _rcb_effort_floor(
    ws: Path, checklist: str, *, min_cov_pct: int = 70,
) -> tuple[bool, str]:
    """P0.3: RCB 跑分路径的 effort floor 硬下限 — 对齐 AV7 autoloop MinEffortFloor.

    复用 _report_coverage_compass 的 keyword 覆盖度, 不达标 → 驳回 TASK COMPLETE.
    ponytail: keyword 命中有天花板 (同义改写漏判), 但比 LLM 版成本低.
    升级路径: B3 LLM compass 替代 keyword compass 做硬下限.
    """
    if not checklist:
        return True, ""  # 无 checklist 不约束
    _compass = _report_coverage_compass(ws, checklist)
    if not _compass:
        return True, ""  # report.md 不存在或无 keyword → 放行 (避免误杀)
    # 从 compass 文本抽 cov_pct: 标题格式 "(NN% — M/N keywords found)"
    import re as _re
    _m = _re.search(r"\((\d+)%\s*—\s*(\d+)/(\d+)", _compass)
    if not _m:
        return True, ""  # 格式不符 → 放行
    _cov = int(_m.group(1))
    if _cov >= min_cov_pct:
        return True, ""
    # 抽 Missing 段
    _missing = ""
    for _line in _compass.split("\n"):
        if _line.lower().startswith("missing:"):
            _missing = _line[len("missing:"):].strip()
            break
    return False, f"coverage={_cov}% < {min_cov_pct}%, missing: {_missing}"


def _report_coverage_compass(ws: Path, checklist: str) -> str:
    """M2Flow compass — 扫 report.md 找已覆盖的 checklist item, 给 agent 显式状态.

    Anthropic robotics 文章核心发现: cursor/compass > 深度图/分割/第三人称视角.
    给模型显式状态信息 ("你在哪, 还差什么") 比给原始数据更有用.

    这里不做 NLP 语义匹配, 只做 keyword 命中 — ponytail: 规则版, 升级路径才换 LLM.
    天花板: keyword 命中可能漏掉同义改写 (如 "MAE" vs "mean absolute error").
    升级: 调 LLM 做语义覆盖度判断 (v8 候选).

    返回 compass 文本, 注入 _iter_prompt. report.md 不存在或 checklist 为空返回 "".
    """
    report_path = ws / "report" / "report.md"
    if not report_path.exists() or not checklist:
        return ""
    try:
        report_text = report_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return ""
    # 从 checklist 提取 keyword — 抓 [EXACT] 标记的组件名 + 数字指标
    keywords = set()
    for line in checklist.split("\n"):
        line = line.strip()
        if not line:
            continue
        # [EXACT] 标记的组件
        if "[exact]" in line.lower():
            # 取 [EXACT] 后面的词组 (最多 4 个词)
            idx = line.lower().find("[exact]")
            tail = line[idx + 7:].strip().split()
            for w in tail[:4]:
                w = w.strip(".,;:()[]")
                if len(w) >= 3:  # >=3 保留 VAE/GNN/CNN 等缩写
                    keywords.add(w.lower())
        # 数字指标 (MAE, R², accuracy 等)
        for metric in ("mae", "rmse", "r²", "r2", "accuracy", "precision", "recall", "f1"):
            if metric in line.lower():
                keywords.add(metric)
    if not keywords:
        return ""
    covered = [kw for kw in keywords if kw in report_text]
    missing = [kw for kw in keywords if kw not in report_text]
    total = len(keywords)
    cov_pct = int(100 * len(covered) / total) if total else 0
    lines = [
        f"## Report Coverage Compass ({cov_pct}% — {len(covered)}/{total} checklist keywords found in report.md)",
    ]
    if covered:
        lines.append(f"Covered: {', '.join(sorted(covered))}")
    if missing:
        lines.append(f"Missing: {', '.join(sorted(missing))}")
        lines.append("→ Address a MISSING item next.")
    else:
        lines.append("→ All keywords covered. Verify quality and respond TASK COMPLETE if done.")
    return "\n".join(lines)


# B3: LLM compass 缓存 — report.md 未变不重审.
# key = (mtime, size, checklist_hash), value = LLM 审计结果. 进程级, RCB subprocess 一次性.
# B6: cache key 加入 checklist hash — checklist 变了但 report.md 没变时 cache 应失效.
_LLM_COVERAGE_CACHE: dict[tuple[float, int, int], str] = {}


async def _llm_coverage_audit(
    model: Any, ws: Path, checklist: str, rule_compass: str,
) -> str:
    """v8: LLM 语义深度审计 report.md 覆盖度. 规则版兜底.

    规则版 keyword 命中漏同义改写 (MAE vs mean absolute error, CGCNN vs Crystal GCN).
    LLM 版做语义判断: 读 report.md + checklist, 判断每个 checklist item 是否被覆盖.

    B3 增强: 用 (mtime, size, checklist_hash) 作为 cache key, report.md 未变直接返回缓存.
    每 5 轮调一次, 但只有 report.md 真的变化才调 LLM, 否则用缓存.
    B6: checklist hash 也进 key — checklist 变 (跨任务 resume) 但 report.md 没变时,
    旧 cache 会误命中, 导致审计用旧 checklist. 加 hash 修这个洞.

    ponytail: 失败返回空串, 调用方 fallback 到 rule_compass.
    """
    report_path = ws / "report" / "report.md"
    if not report_path.exists():
        return ""
    # B3+B6: cache 检查 — report.md 未变 + checklist 未变 才返回缓存
    try:
        stat = report_path.stat()
        checklist_hash = hash(checklist or "")
        cache_key = (stat.st_mtime, stat.st_size, checklist_hash)
        if cache_key in _LLM_COVERAGE_CACHE:
            return _LLM_COVERAGE_CACHE[cache_key]
    except Exception:
        cache_key = None
    try:
        report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(report_text) > 8000:
        report_text = report_text[:8000] + "\n... (truncated)"
    prompt = f"""Audit report.md coverage against checklist. Rule-based compass says:

{rule_compass}

Checklist:
{checklist[:3000]}

Report.md (first 8000 chars):
{report_text}

Task: For each checklist item, judge if it's COVERED / PARTIALLY / MISSING in the report.
Pay attention to synonyms and paraphrases (e.g. "MAE" = "mean absolute error", "CGCNN" = "Crystal Graph Convolutional Network").

Respond in this exact format (no prose):
COVERAGE: X% (M/N items)
COVERED: item1, item2
PARTIAL: item3 (what's missing: ...)
MISSING: item4, item5
NEXT: the single most important missing/partial item to address next"""
    try:
        # model 是 HuginnModel, 有 async chat 或 sync invoke
        if hasattr(model, "chat"):
            resp = await model.chat(prompt)
        elif hasattr(model, "ainvoke"):
            resp = await model.ainvoke(prompt)
        elif hasattr(model, "invoke"):
            resp = model.invoke(prompt)
        else:
            return ""
        resp_text = resp if isinstance(resp, str) else str(getattr(resp, "content", resp))
        if not resp_text or len(resp_text) < 20:
            return ""
        # 包装成 compass 格式
        result = f"## LLM Coverage Audit (semantic, cached)\n{resp_text.strip()}"
        # B3: 存缓存
        if cache_key is not None:
            _LLM_COVERAGE_CACHE[cache_key] = result
            # 限制缓存大小 (RCB 任务通常 < 50 次 LLM 审计)
            if len(_LLM_COVERAGE_CACHE) > 100:
                # 删最老的 key (mtime 最小)
                oldest = min(_LLM_COVERAGE_CACHE.keys(), key=lambda k: k[0])
                del _LLM_COVERAGE_CACHE[oldest]
        return result
    except Exception as e:
        logger.debug("LLM coverage audit failed: %s", e)
        return ""


async def _step2_5_report_fallback(
    ws: Path,
    stream_chat_fn,
) -> None:
    """Step 2.5: report.md 兜底 — agent 没写就强制写, 仍不写就自动生成.

    σ₆ 修复: 减 CSM (σ₃) 后失去 completion guidance, 加 lightweight gate.
    agent 可能在 Step 2 提前终止 (text-only response), 没写 report.md.
    """
    report_path = ws / "report" / "report.md"
    if not report_path.exists():
        print("\n=== Step 2.5: report.md Emergency Write ===\n", flush=True)
        await stream_chat_fn(
            "CRITICAL: report/report.md does NOT exist. Session scores ZERO without it.\n"
            "Write report/report.md NOW using file_write_tool. Base it on:\n"
            "- Your Step 1 methodology checklist\n"
            "- Your code in code/ and results in outputs/\n"
            "Minimum: # Title, ## Methodology, ## Results (images/*.png), ## Discussion.\n"
            "Be HONEST. A short honest report beats no report. Write it NOW.",
            "step2.5"
        )
    if not report_path.exists():
        print("[fallback: auto-generating minimal report.md]", flush=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _metrics_parts = []
        for _p in (ws / "outputs").glob("*.json"):
            try:
                _metrics_parts.append(f"### {_p.name}\n```json\n{_p.read_text(encoding='utf-8')}\n```")
            except Exception:
                pass
        _metrics = "\n".join(_metrics_parts) or "None"
        _imgs_dir = ws / "report" / "images"
        _imgs = "\n".join(f"![{p.name}](images/{p.name})" for p in _imgs_dir.glob("*.png")) or "None" if _imgs_dir.exists() else "None"
        _code_dir = ws / "code"
        _code = "\n".join(f"- `{p.name}`" for p in _code_dir.glob("*.py")) or "None" if _code_dir.exists() else "None"
        report_path.write_text(
            f"# Research Report (Auto-generated Fallback)\n\n"
            f"## Methodology\nAgent did not write report.md; auto-generated from artifacts.\n\n"
            f"### Code\n{_code}\n\n### Metrics\n{_metrics}\n\n## Results\n{_imgs}\n",
            encoding="utf-8"
        )


async def _step3_adversarial(
    ws: Path,
    model: Any,
    agent: Any,
    checklist: str,
    evals_history: list,
    stream_chat_fn,
    rcb_csm_advance_fn,
) -> None:
    """Step 3: 对抗式自检 — skeptical reviewer 视角找 gap.

    ponytail: 治 3 个系统性短板 (跨 4 题评分发现的共性 gap):
      A. sanity check — 治 "不可信结果不自检"
      B. substitution audit — 治 "沉默方法降级"
      C. hard push — 治 "硬组件轻易放弃"
    双层 critique: object mode (report) + meta mode (directive).
    """
    print("\n=== Step 3: Adversarial Self-Critique ===\n", flush=True)
    rcb_csm_advance_fn("tool_failure", {"reason": "adversarial critique — find gaps"})

    report_path = ws / "report" / "report.md"
    external_critique_block = ""
    object_verdict = None
    if report_path.exists() and checklist:
        try:
            report_text = report_path.read_text(encoding="utf-8")
            print(f"[adversarial_critique: reading {len(report_text)} chars of report.md]", flush=True)
            try:
                from huginn.metacog.step_evaluator import check_uncertainty_propagation
                _unc_issues = check_uncertainty_propagation(evals_history)
                if _unc_issues:
                    _unc_block = "\n\n## 误差建模缺失 (P2 check)\n"
                    for iss in _unc_issues:
                        _unc_block += f"- step {iss['step_id']}: {iss['issue']} — {iss['detail']}\n"
                    checklist = checklist + _unc_block
                    print(f"[P2: {len(_unc_issues)} uncertainty issue(s) injected to critique]", flush=True)
            except Exception as _ue:
                print(f"[P2 uncertainty check skipped: {_ue}]", flush=True)
            object_verdict = await adversarial_critique(
                model, report_text, checklist, mode="object",
            )
            try:
                recomputed = _recompute_report_metrics(report_text, ws)
                if recomputed:
                    object_verdict.setdefault("recomputed_red_flags", []).extend(recomputed)
                    if object_verdict.get("overall_verdict") == "pass":
                        object_verdict["overall_verdict"] = "fix_needed"
                    print(f"[G28: {len(recomputed)} metric claim(s) mismatch recomputed values]", flush=True)
                    external_critique_block = format_critique_for_agent(object_verdict)
            except Exception as e:
                print(f"[G28: recompute skipped: {e}]", flush=True)
            external_critique_block = format_critique_for_agent(object_verdict)
            print(f"[adversarial_critique: verdict={object_verdict.get('overall_verdict', '?')}]", flush=True)
        except Exception as e:
            print(f"[adversarial_critique: skipped due to error: {e}]", flush=True)
    else:
        print("[adversarial_critique: skipped — report.md or checklist missing]", flush=True)

    # Layer 2 — meta mode: 触发 CSM 进 S6_FEEDBACK → S7_SELF_MODIFY
    try:
        from huginn.cognitive_engine import TransitionSignal, CognitiveState
        csm = getattr(agent, "_csm", None)
        if csm is not None and object_verdict is not None:
            verdict_flag = object_verdict.get("overall_verdict", "fix_needed")
            sig = "tool_failure" if verdict_flag != "pass" else "tool_success"
            new_state = csm.transition(TransitionSignal(sig, {
                "objective": "step3_critique",
                "result_summary": f"object_verdict={verdict_flag}",
            }))
            if new_state == CognitiveState.S6_FEEDBACK and verdict_flag != "pass":
                csm.transition(TransitionSignal("gap_found", {
                    "gap": external_critique_block[:200] or "step3 object critique red flags",
                }))
    except Exception:
        logger.debug("Step 3 CSM S6/S7 trigger failed", exc_info=True)

    step3_prompt = (
        "ADVERSARIAL SELF-CRITIQUE. You are now a SKEPTICAL REVIEWER who wants to score this report LOW. "
        "Do NOT be lenient with yourself.\n\n"
        "## A. Sanity Check (do this FIRST — catches fabricated/impossible results)\n"
        "Read your report/report.md. Extract EVERY quantitative claim (MAE, R², accuracy, loss, etc.).\n"
        "Compare each to the paper's baseline value from your Step 1 checklist.\n"
        "Build a table: | Metric | Paper Value | Your Value | Better? |\n"
        "If ANY of your metrics is BETTER than the paper's — that is a RED FLAG.\n"
        "Investigate why: data leakage? wrong train/test split? simplified geometry? fabricated?\n"
        "Fix the bug, or honestly document the discrepancy. "
        "Implausibly good results get ZERO from reviewers.\n\n"
        "## B. Substitution Audit (catches silent methodology downgrade)\n"
        "List every [EXACT] component from your Step 1 checklist.\n"
        "For each, answer honestly: did I implement it AS-SPECIFIED, or did I substitute a simpler alternative?\n"
        "  - Substituted WITHOUT trying the real implementation → FAILURE. Implement it now.\n"
        "  - Substituted AFTER ≥2 genuine failed attempts → document the attempts with error messages.\n"
        "  'I used Random Forest instead of VAE because VAE is hard' is NOT acceptable.\n"
        "  'I used GCNConv instead of CGCNNConv because it was easier' is NOT acceptable.\n\n"
        "## C. Coverage Check\n"
        "List checklist items COVERED (with evidence from report) vs MISSING/WEAK.\n\n"
        "## D. Fix & Rewrite\n"
        "For each gap found in A/B/C:\n"
        "  - Missing metric → compute it now (run code_tool)\n"
        "  - Missing [EXACT] component → implement it (push through, try ≥2 approaches before giving up)\n"
        "  - Implausible result → fix the bug or document honestly why it's off\n"
        "OVERWRITE report/report.md with: improved results + baseline comparison table + "
        "honest Limitations section (only for items where you tried ≥2 approaches and genuinely failed).\n"
        "Use file_write_tool for the rewrite."
    )
    if external_critique_block:
        step3_prompt = external_critique_block + "\n\n## Now act on the critique above:\n" + step3_prompt
    await stream_chat_fn(step3_prompt, "step3")


async def run(workspace: str, extreme: bool = False) -> int:
    ws = Path(workspace).resolve()
    instructions = ws / "INSTRUCTIONS.md"
    if not instructions.exists():
        print(f"ERROR: {instructions} not found", file=sys.stderr)
        return 1

    # RCB subprocess 跑时主 memory.db 可能被 IDE/桌面端锁定 (sqlite WAL),
    # 改用 workspace 下的独立缓存目录. RCB 是无状态离线评测, 不需要跨任务记忆.
    rcb_cache = ws / ".huginn_cache"
    rcb_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HUGINN_CACHE_DIR"] = str(rcb_cache)

    # Task 3: Checkpoint resume — task_id 用 ws.name (RCB workspace 目录名).
    # 找到上次的 checkpoint 就接着跑, audit chain 校验失败则从头开始.
    _task_id = ws.name
    _resume_from_iter = 0
    try:
        from huginn.runtime.checkpoint import load_checkpoint, resume_from_checkpoint
        _resumed = load_checkpoint(_task_id, ws)
        if _resumed is not None:
            print(f"[Resume] found checkpoint at step {_resumed.step_id}", flush=True)
            _resume_from_iter = max(0, resume_from_checkpoint(_resumed, ws) - 1)
            print(f"[Resume] continuing from iter {_resume_from_iter}", flush=True)
    except Exception as _e:
        print(f"[Resume] failed, starting fresh: {_e}", flush=True)
        _resume_from_iter = 0

    prompt = instructions.read_text(encoding="utf-8")

    from huginn.agent import HuginnAgent
    from huginn.config import HuginnConfig
    from huginn.models.registry import ModelRegistry
    from huginn.tools import register_all_tools

    # snapshot 默认用 ~/.huginn/snapshots, RCB subprocess 跑时该目录可能被
    # IDE/桌面端锁定 (PermissionError). 重定向到 workspace 下的独立目录.
    from huginn.snapshot import file_snapshot as _fs
    _fs._SNAPSHOT_ROOT = rcb_cache / "snapshots"

    cfg = HuginnConfig.from_env()
    # v6 极限模式: 解除一切限制, 性能优先. 更高思考强度 + 更长任务轨迹.
    # ponytail: 不改默认值, 只在 --extreme 时 override. 升级路径是加 profile 系统.
    if extreme:
        os.environ.setdefault("HUGINN_THINKING", "high")
        # v7 长任务: extreme 模式同时放宽 autoloop stop 阈值, 允许 200+ 步轨迹.
        # 对标 Oxelra 206 步. 默认值已放宽 (20/20/10/5), extreme 再翻倍.
        os.environ.setdefault("HUGINN_MAX_CONSECUTIVE_FAILURES", "50")
        os.environ.setdefault("HUGINN_MAX_REFINES", "50")
        os.environ.setdefault("HUGINN_MAX_PIVOTS", "20")
        os.environ.setdefault("HUGINN_DARWIN_STAGNATION_LIMIT", "15")
        cfg = HuginnConfig.from_env()  # 重读 env 拿 thinking
        print("[EXTREME MODE] thinking=high, max_tool_calls=300, context_budget=200K, autoloop thresholds 50/50/20/15", flush=True)

    registry = ModelRegistry.from_config(cfg)
    alias = registry.default_alias()
    if alias:
        model = registry.resolve(alias)
    elif cfg.provider and cfg.provider != "default":
        model = registry.resolve(f"{cfg.provider}/{cfg.model or 'auto'}")
    else:
        print("ERROR: no model configured", file=sys.stderr)
        return 1

    # RCB harness 从 stdout 检测 model 名 (run_task._detect_model)
    model_name = getattr(model, "name", None) or getattr(model, "model_id", None) or str(model)
    print(f'model: {model_name}', flush=True)

    # system prompt: workspace 路径 + 工具操作事实. 让 INSTRUCTIONS.md 做 task gradient.
    # ponytail: 删 CRITICAL override 层 (σ₄) — control loop 不替 LLM 决策"什么重要".
    system_prompt = (
        f"You are an autonomous scientific research agent. "
        f"Your workspace is: {ws}\n"
        f"Current working directory IS the workspace. All relative paths "
        f"(data/, related_work/, code/, outputs/, report/) resolve from here.\n"
        f"Follow INSTRUCTIONS.md as your primary guide — it defines the task.\n"
        f"Prefer real implementations over shortcuts; document failures honestly.\n\n"
        "## Tool facts (sandbox constraints, not priorities)\n"
        "- code_tool: run Python. Sandbox BLOCKS open() and os — CANNOT write files via code_tool.\n"
        "- bash_tool: pip install, run scripts.\n"
        "- file_write_tool: CREATE or OVERWRITE text files (report.md, code/*.py). "
        "Pass FULL content each time.\n"
        "- matplotlib.savefig() WORKS (library code, not AST-scanned) — use it for figures.\n"
        "- code_tool security scanner may false-positive on eval() in torch/numpy — "
        "if so, write script via file_write_tool and run with bash_tool.\n"
        "- file_read_tool/glob/grep: explore data/ and related_work/.\n"
        "- web_search_tool: verify constants, methods, or edge cases.\n\n"
        "## Operating rules\n"
        "- Every response before task completion MUST include a tool call. "
        "Text-only response = task termination.\n"
        "- Push through errors: debug, install missing packages, try alternatives.\n"
        "- Write report/report.md EARLY, then OVERWRITE as you add results.\n"
    )

    # 先注册工具到 ToolRegistry, 再让 agent 从 registry 拉取
    register_all_tools()

    # v6 极限模式: max_tool_calls 300 + context_budget 200K + 每 tool 上限 100
    # 默认 150 / 0 / 50. 极限模式拉满, 让 agent 能跑更长任务轨迹.
    _max_calls = 300 if extreme else 150
    _max_per_tool = 100 if extreme else 50
    _ctx_budget = 200000 if extreme else cfg.context_budget_tokens

    # Task 12: Memory 接线 — 用 workspace 内独立 memory dir, 避免跨任务污染.
    # 失败降级 None, agent 走无 memory 路径 (原行为).
    _mem_mgr = None
    try:
        from huginn.memory.manager import MemoryManager, MemoryConfig
        _mem_cfg = MemoryConfig(memory_dir=ws / ".huginn" / "memory")
        _mem_mgr = MemoryManager(config=_mem_cfg, llm=model)
    except Exception as _e:
        print(f"[Memory] init warning: {_e}", flush=True)

    # 25.1: project KG 实例. 落 ws/.huginn/, 和 ContextBuilder.build_kg_text
    # 用同一路径, 复用同一份持久化. 失败降级 None, evaluate_step 和 episode
    # history 注入都跳过 — ponytail: kg 是可选增强, 失败不阻塞主流程.
    _kg = None
    try:
        from huginn.kg.graph import ProjectKnowledgeGraph
        _kg = ProjectKnowledgeGraph(ws / ".huginn")
    except Exception as _e:
        print(f"[KG] init warning: {_e}", flush=True)

    # Task 13: Persona 接线 — 从 ws.name 推断领域, 选对应 built-in persona.
    # HuginnAgent 只接 persona_name (str), persona 对象留给 StepEvaluator 用.
    _persona_name = "default"
    persona = None
    try:
        from huginn.personas import PersonaManager
        _pm = PersonaManager(workspace=str(ws))
        _ws_name_lower = ws.name.lower()
        if any(k in _ws_name_lower for k in ("astronom", "cosmo", "galaxy", "star")):
            _persona_name = "reviewer"
        elif any(k in _ws_name_lower for k in ("material", "dft", "vasp", "crystal")):
            _persona_name = "dft_expert"
        elif any(k in _ws_name_lower for k in ("md", "lammps", "molecular")):
            _persona_name = "md_expert"
        persona = _pm.get(_persona_name)
    except Exception as _e:
        print(f"[Persona] init warning: {_e}", flush=True)

    agent = HuginnAgent(
        model=model,
        system_prompt=system_prompt,
        memory_manager=_mem_mgr,
        persona_name=_persona_name,
        workspace=ws,
        max_tool_output_tokens=cfg.max_tool_output_tokens,
        context_budget_tokens=_ctx_budget,
        max_tool_calls=_max_calls,
        max_tool_calls_per_tool=_max_per_tool,
        # file_write_tool 写文本文件 (report.md, code/*.py);
        # code_tool 的 sandbox 禁 open(), 只能跑分析/画图 (savefig 库代码不受限).
        # 不给 file_edit_tool: 它要求文件已存在, agent 误用 edit 写新文件会失败.
        tool_filter=[
            "code_tool", "bash_tool",
            "file_read_tool", "file_write_tool",
            "glob", "grep", "web_search_tool",
            "self_observe",
            # G27: 数学工具解除 filter 屏蔽 — repro 数量级错误 (χ=1.0 vs 0.004) 的根因之一
            # 是四个外部适配器 tool_filter 把数学工具整体摘除 (audit 13 F1).
            "symbolic_math_tool", "lean_tool", "validate_tool",
        ],
        # RCB 是无人工 subprocess, 所有工具自动 approve
        auto_approve=True,
    )
    agent.register_tools_from_registry()

    # 3 步认知循环: 论文方法论提取 → 执行 → 自验证
    # ponytail: 不走 autoloop 7 阶段 (太重), 用 3 步循环治 3 个短板:
    #   Step 1 治 "不读论文就动手" — 强制先提取方法核心组件 + baseline 指标
    #   Step 2 治 "方法降级" — checklist 注入, agent 对照方法约束执行
    #   Step 3 治 "不自验证" — 对照 checklist 检查 report 覆盖度, 补缺
    # 用同一 thread_id 保持上下文连续, Step 2 能看到 Step 1 的 checklist.
    from langchain_core.messages import AIMessage

    thread_id = f"rcb_{ws.name}"

    async def _stream_chat(msg: str, step_label: str, tid: str | None = None) -> str:
        """跑一轮 agent.chat, 流式打印 AIMessage, 返回最后的 AI 文本.

        tid: TFM 分叉用独立 thread 隔离 graph 内态 (历史从 ConversationTree
        重建, thread 只影响 checkpoint 内态, 换 tid 无历史损失).

        视觉接入: msg 里含图片路径 (xxx.png/jpg/...) 时透传 image_path 给
        agent.chat, streaming.py 的 VisionRouter 自动接管 (CV 预分析 +
        visual primitives 注入). RCB 任务通常无图, 但 related_work/ 下的
        论文图表路径若被 agent 引用就会触发. ponytail: 0 行额外配置.
        """
        ai_text = ""
        try:
            # 扫 msg 里的图片路径 — 命中就透传给 VisionRouter
            _image_path = None
            try:
                from huginn.vision.router import _IMAGE_PATH_RE
                _m = _IMAGE_PATH_RE.search(msg or "")
                if _m:
                    _candidate = _m.group(0)
                    # 相对路径补成 workspace 绝对路径, VisionRouter 需要能 open
                    _p = Path(_candidate)
                    if not _p.is_absolute():
                        _p = (ws / _candidate).resolve()
                    if _p.exists():
                        _image_path = str(_p)
            except Exception:
                pass  # 视觉接入是增强, 失败不阻塞文本路径
            async for chunk in agent.chat(
                msg, thread_id=tid or thread_id, image_path=_image_path,
            ):
                msgs = chunk.get("messages", [])
                if not msgs:
                    continue
                last = msgs[-1]
                if not isinstance(last, AIMessage):
                    continue
                content = getattr(last, "content", "")
                if content:
                    print(content, flush=True)
                    ai_text = content
        except Exception as e:
            print(f"ERROR [{step_label}]: {e}", file=sys.stderr)
        return ai_text

    # RCB 3-step 映射 CSM: Step1→S1_DISCOVER, Step2→S4_CONSTRUCT, Step3→S6+S7 (Task 18)
    # ponytail: transition 是 advisory — 不允许就 no-op, 不破坏现有 3-step 流程.
    from huginn.cognitive_engine import TransitionSignal as _RCB_TS

    def _rcb_csm_advance(signal_type: str, ctx: dict) -> None:
        """RCB step 开始时手动推 CSM 状态. advisory: 不允许就 no-op."""
        csm = getattr(agent, "_csm", None)
        if csm is None:
            return
        try:
            csm.transition(_RCB_TS(signal_type, ctx))
        except Exception:
            logger.debug("RCB CSM transition failed", exc_info=True)

    # Step 0: KB 摄入任务数据 (Task 13)
    # 把 related_work/ + INSTRUCTIONS.md + data/ 灌进 workspace KB,
    # 给后续 target_chain 反推和 StepEvaluator 提供先验. 失败只 warn, 不阻塞主流程.
    kb = None
    try:
        from huginn.knowledge.store import get_knowledge_base
        kb = get_knowledge_base(str(ws))
        for _rw_file in (ws / "related_work").glob("*"):
            if _rw_file.is_file() and _rw_file.suffix in (".md", ".txt", ".pdf"):
                kb.add_document(_rw_file.name, _rw_file.read_bytes())
        if (ws / "INSTRUCTIONS.md").exists():
            kb.add_text((ws / "INSTRUCTIONS.md").read_text(encoding="utf-8"),
                        filename="INSTRUCTIONS.md")
        _data_dir = ws / "data"
        if _data_dir.exists():
            for _data_file in _data_dir.glob("*"):
                if _data_file.is_file() and _data_file.suffix in (".csv", ".json", ".txt", ".md"):
                    kb.add_text(
                        _data_file.read_text(encoding="utf-8", errors="ignore"),
                        filename=f"data/{_data_file.name}",
                    )
        print("[Step 0] KB ingest done", flush=True)
    except Exception as _e:
        print(f"[Step 0] KB ingest warning: {_e}", flush=True)

    # Step 1: 论文方法论提取
    # agent 读 INSTRUCTIONS.md + related_work/, 输出方法核心组件 + baseline 指标 checklist
    print("\n=== Step 1: Methodology Extraction ===\n", flush=True)
    _rcb_csm_advance("user_goal", {"goal": "understand problem and extract methodology"})
    step1_prompt = (
        f"Read the task instructions below AND explore related_work/ directory for reference papers.\n"
        f"Extract a METHODOLOGY CHECKLIST from the paper:\n"
        f"1. Core method components (model architecture, training protocol, key algorithms).\n"
        f"   For EACH component, label it [EXACT] (must reproduce as-specified) or [VARIANT]\n"
        f"   (justified deviation with reason). Default to [EXACT]. The label forces honesty\n"
        f"   about substitutions — Step 3 will audit them.\n"
        f"2. Key quantitative metrics with the paper's BASELINE VALUES (e.g. 'R²=0.79, MAE=48K').\n"
        f"   These are the targets your results will be compared against in Step 3.\n"
        f"3. Critical implementation details that must be reproduced\n\n"
        f"Output the checklist as a numbered list. Be SPECIFIC (e.g. 'CGCNNConv with gating "
        f"and residual connections', not just 'GNN'). This checklist will guide your implementation "
        f"and will be used in Step 3's substitution audit and sanity check.\n\n"
        f"Task instructions:\n{prompt}"
    )
    checklist = await _stream_chat(step1_prompt, "step1")
    print(f"\n[checklist extracted: {len(checklist)} chars]\n", flush=True)

    # G29: checklist 永驻 system_prompt — 写入 stable_principles (source="checklist"),
    # context.py 的 STABLE_PRINCIPLES 段每轮 build_prompt 重读, 不进 compaction 范围.
    # 修 audit 09: RCB 长任务 compaction 跳过后 checklist 丢失, Step 2/3 看不到方法论约束.
    # ponytail: checklist 是 persona 级输入 (跨 step 不变), 走 stable_principles 通道
    # 比改 prompt_builder 加新段更省代码. 任务结束不清除, 下一任务 init 时会被覆盖语义
    # (新 checklist 会被 store 进来, 旧的仍在文件里但 LLM 会以新为准).
    if checklist and checklist.strip():
        try:
            from huginn.memory import store_stable_principle
            # 截断到 2000 字符防 persona 膨胀, 完整 checklist 在 ws/checklist.md
            store_stable_principle(
                f"[METHODOLOGY CHECKLIST]\n{checklist[:2000]}",
                source="rcb_step1_checklist",
            )
            # 同时写到 ws/checklist.md 让 agent 能 file_read_tool 读完整版
            (ws / "checklist.md").write_text(checklist, encoding="utf-8")
            print(f"[G29: checklist stored as stable_principle + ws/checklist.md]", flush=True)
        except Exception as e:
            print(f"[G29: checklist store skipped: {e}]", flush=True)

    # Step 1.2: 目标链反推 (G62, Task 14)
    # 把 checklist 每条 Mode-A 目标反推成 required_results/methods/data/verification 链.
    # checklist 是 Step 1 输出的文本, 这里包成单条 Mode-A item 让 LLM 自己分解.
    # ponytail: 天花板是单条粗粒度 item — LLM 拿到整段 checklist 做分解, 不会逐条对齐;
    #           升级路径是先用结构化 prompt 让 Step 1 直接输出 list[dict] (mode/item).
    _target_chains = []
    try:
        from huginn.metacog.target_chain import build_target_chains
        _checklist_items = (
            checklist if isinstance(checklist, list)
            else [{"mode": "A", "item": (checklist or "")[:2000]}]
        )
        _task_ctx = (
            (ws / "INSTRUCTIONS.md").read_text(encoding="utf-8")[:2000]
            if (ws / "INSTRUCTIONS.md").exists() else ""
        )
        _target_chains = await build_target_chains(
            _checklist_items, kb, model, _task_ctx,
        )
        _tc_entry = {
            "iteration": 0,
            "ts": _time.time() if "_time" in dir() else __import__("time").time(),
            "role": "target_chain",
            "attempted": f"build_target_chains for {len(_checklist_items)} item(s)",
            "found": f"{len(_target_chains)} chains built",
            "evidence": [],
            "limitations": [],
            "artifacts": [],
            "next_hint": "step1.5 structure scan",
            "darwin_score": 0.0,
            "supported_ratio": 0.0,
        }
        try:
            _tc_trace = ws / ".huginn" / "meta_trace.jsonl"
            _tc_trace.parent.mkdir(parents=True, exist_ok=True)
            with _tc_trace.open("a", encoding="utf-8") as _f:
                _f.write(json.dumps(_tc_entry, ensure_ascii=False, default=str) + "\n")
        except Exception as _e:
            print(f"[Step 1.2] trace write skipped: {_e}", flush=True)
        print(f"[Step 1.2] built {len(_target_chains)} target chains", flush=True)
    except Exception as _e:
        print(f"[Step 1.2] target chain warning: {_e}", flush=True)

    # Step 1.5: Intuitive Gamer + 数学直觉结构识别
    # 两层结合 (arXiv:2510.11503 fast-flat scan + 数学结构识别):
    # - fast flat scan: 不深挖, 快速过一遍 checklist
    # - 数学直觉: 识别每个 item 的数学结构 + invariant, 而非只标难度
    #
    # 平衡点 (advisory + audited, not enforced):
    # - 保守默认: 每个 item 默认 structure=empirical, invariant=none
    # - verifiable_via 枚举 gate: 拿不出验证方法的退回 empirical
    # - 分档处理: hard check (dimensional/exact_formula/conservation_law)
    #            vs soft check (asymptotic/symmetry/topological) vs none
    # - exploratory 合法: 不强制每个 item 都有数学结构
    #
    # ponytail: v7 只做 prompt + 枚举约束, 不写 Lean, 不写 pydantic schema.
    #   v8 升级: 接 cognitive_heat_engine should_imaginate, hard check 失败
    #   触发 imagination; 接 LeanInterface 做形式化验证.
    print("\n=== Step 1.5: Intuitive Gamer + math structure scan ===\n", flush=True)
    scan_prompt = (
        "FAST FLAT SCAN with mathematical structure identification.\n"
        "Goal: identify structure + invariants for each checklist item, not just difficulty.\n\n"
        "For EACH checklist item, output a block:\n"
        "  [item N] structure: <type>\n"
        "    invariant: <one-line statement, or 'none'>\n"
        "    verifiable_via: <method, or 'none'>\n"
        "    anchor: <first-principles reference, or 'exploratory'>\n\n"
        "structure types (enum, pick one):\n"
        "  empirical | symmetry | asymptotic | dimensional | topological | probabilistic | algebraic\n"
        "  - empirical: pure data/observation, no known mathematical structure\n"
        "  - symmetry: invariant under transformation group (rotation, gauge, etc)\n"
        "  - asymptotic: limit behavior (t->inf, x->0) constrains the answer\n"
        "  - dimensional: Buckingham Pi / dimensional homogeneity must hold\n"
        "  - topological: invariant under continuous deformation (winding number, etc)\n"
        "  - probabilistic: distributional constraint (normalization, Bayes consistency, posterior contractibility)\n"
        "  - algebraic: equation/identity must hold exactly (eigenvalue eq, etc)\n\n"
        "verifiable_via (enum, pick one):\n"
        "  none | dimensional | asymptotic_limit | exact_formula | conservation_law | symmetry_argument | topological_invariant\n"
        "  - 'none' only if you genuinely cannot specify a verification method\n"
        "  - must correspond to the structure type (e.g. structure=dimensional → verifiable_via=dimensional)\n\n"
        "anchor:\n"
        "  - cite first-principles reference (e.g. 'black hole thermodynamics', 'Noether theorem', 'Bayes rule')\n"
        "  - 'exploratory' is valid — accept that structure may be uncertain at this stage\n\n"
        "Guidance (calibrated — avoid both over-claiming and under-claiming):\n"
        "- If the item involves a physical quantity with units → structure=dimensional, verifiable_via=dimensional.\n"
        "- If the item involves a Bayesian posterior / probability distribution → structure=probabilistic, verifiable_via=exact_formula (Bayes rule) or asymptotic_limit (posterior concentration).\n"
        "- If the item involves a conservation law (energy, charge, angular momentum) → structure=symmetry, verifiable_via=conservation_law.\n"
        "- If the item involves an exact equation (eigenvalue, polynomial root) → structure=algebraic, verifiable_via=exact_formula.\n"
        "- If the item involves a limit behavior (large-N, t→∞, x→0) → structure=asymptotic, verifiable_via=asymptotic_limit.\n"
        "- Only use empirical/none when the item is genuinely pure observation with no mathematical constraint.\n"
        "- Do NOT fabricate invariants you cannot verify — but DO identify invariants that genuinely apply.\n\n"
        "Constraints:\n"
        "- 1 tool call MAX (file_read or code_tool for quick check). Prefer 0.\n"
        "- Do NOT execute analysis. Do NOT write report.md.\n\n"
        "After all items, output a STRATEGY line:\n"
        "  STRATEGY: <one-line plan — order items by verifiable_via priority:\n"
        "    hard_check (dimensional/exact_formula/conservation_law) first to bank structural wins,\n"
        "    then soft_check (asymptotic/symmetry/topological), then empirical/none last>\n"
        f"\nChecklist:\n{checklist[:4000]}"
    )
    # Step 1.5 用单次 LLM 调用, 绕过 agent.chat 的 ReAct loop.
    # 原因: ReAct agent 拿到 prompt 后会直接调 tool 执行, 不给文本规划输出.
    # Step 1.5 要的是纯文本 structure scan, 不允许 tool call.
    # ponytail: 不另建 agent 实例 (省 memory), 直接调 model.ainvoke.
    #   升级路径: 建专用 "planner" agent (无 tools), 复用 thread_id 上下文.
    scan_text = ""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        _scan_msgs = [
            SystemMessage(content=(
                "You are a mathematical structure scanner. Output ONLY text, "
                "no tool calls. Identify the mathematical structure of each "
                "checklist item and the invariant it must satisfy."
            )),
            HumanMessage(content=scan_prompt),
        ]
        _scan_resp = await asyncio.to_thread(model.invoke, _scan_msgs)
        scan_text = _scan_resp.content if hasattr(_scan_resp, "content") else str(_scan_resp)
        print(scan_text, flush=True)
    except Exception as _e:
        print(f"[Step 1.5 LLM call failed: {_e}]", flush=True)
        scan_text = ""
    print(f"\n[structure scan done: {len(scan_text)} chars]\n", flush=True)

    # 写 Meta-Trace entry — role="intuitive_gamer", 带 structure 信息
    try:
        import json as _ig_json
        import time as _ig_time
        _ig_entry = {
            "iteration": 0,
            "ts": _ig_time.time(),
            "role": "intuitive_gamer",
            "attempted": "fast flat scan with mathematical structure identification",
            "found": (scan_text or "")[:500],
            "evidence": [],
            "limitations": [
                "single-sample, no k-sampling (v8 upgrade)",
                "structure labels not schema-validated (v8: pydantic + Lean)",
            ],
            "artifacts": [],
            "next_hint": "execute hard_check items first to bank structural wins",
            "darwin_score": 0.0,
            "supported_ratio": 0.0,
        }
        _ig_trace_path = ws / ".huginn" / "meta_trace.jsonl"
        _ig_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with _ig_trace_path.open("a", encoding="utf-8") as f:
            f.write(_ig_json.dumps(_ig_entry, ensure_ascii=False) + "\n")
        print("[intuitive_gamer + math structure trace entry written]", flush=True)
    except Exception as _e:
        print(f"[intuitive_gamer trace skipped: {_e}]", flush=True)

    # Step 1.7: fork-critique-merge — verifier 下沉到 plan 决策点 (v8)
    # k 路视角采样执行方案, 独立评审选优, winner 注入 Step 2.
    # 成本: k+1 次 cheap LLM 调用 (无 tool, 纯文本).
    print("\n=== Step 1.7: Plan Fork-Critique-Merge ===\n", flush=True)
    fcm = await fork_critique_merge(checklist, scan_text, model, k=3)
    if fcm["winner_plan"]:
        print(f"[fcm: winner={fcm['winner_perspective']} scores={fcm.get('scores')}]\n", flush=True)
        print(fcm["winner_plan"], flush=True)
    else:
        print("[fcm: all forks failed, Step 2 runs plan-free]", flush=True)
    try:
        _fcm_entry = {
            "iteration": 0,
            "ts": _ig_time.time(),
            "role": "fork_critique_merge",
            "attempted": f"{len(_FCM_PERSPECTIVES)}-perspective plan sampling + independent critique",
            "found": f"winner={fcm['winner_perspective']} scores={fcm.get('scores')}",
            "evidence": [fcm["winner_plan"][:300]] if fcm["winner_plan"] else [],
            "limitations": ["plan-level critique only; trajectory-level fork deferred (v8 fcm-2)"],
            "artifacts": [],
            "next_hint": "execute winner plan; merge insights as fallback options",
            "darwin_score": 0.0,
            "supported_ratio": 0.0,
        }
        with _ig_trace_path.open("a", encoding="utf-8") as f:
            f.write(_ig_json.dumps(_fcm_entry, ensure_ascii=False) + "\n")
    except Exception as _e:
        print(f"[fcm trace skipped: {_e}]", flush=True)

    # Step 2 setup + 循环抽到模块级函数 _step2_execute.
    _step2_ctx = _RCBStep2Ctx(
        ws=ws, model=model, agent=agent, kb=kb,
        mem_mgr=_mem_mgr, persona=persona, kg=_kg,
        thread_id=thread_id, task_id=_task_id,
        resume_from_iter=_resume_from_iter, extreme=extreme,
        checklist=checklist, scan_text=scan_text, fcm=fcm,
        target_chains=_target_chains,
        instructions=instructions,
        stream_chat_fn=_stream_chat,
        rcb_csm_advance_fn=_rcb_csm_advance,
    )
    _evals_history = await _step2_execute(_step2_ctx)  # 返回 _evals_history 供 Step 3 用

    # Step 2.5 + Step 3 抽到模块级函数 — 闭包 _stream_chat / _rcb_csm_advance 作参数传入.
    await _step2_5_report_fallback(ws, _stream_chat)
    await _step3_adversarial(
        ws, model, agent, checklist, _evals_history, _stream_chat, _rcb_csm_advance,
    )

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Huginn RCB runner")
    parser.add_argument("--workspace", required=True, help="RCB workspace path")
    parser.add_argument(
        "--extreme", action="store_true",
        help="v6 极限模式: thinking=high, max_tool_calls=300, context_budget=200K",
    )
    args = parser.parse_args()

    rc = asyncio.run(run(args.workspace, extreme=args.extreme))
    sys.exit(rc)


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        # Task 3 self-check: meta mode 早期拒绝 (不调 LLM)
        # ponytail: 命中查重直接返回, llm_client=None 也能跑, 验证 ponytail 优化没退化.
        # 用 asyncio.run 包裹因 adversarial_critique 是 async (object mode 调用点 L434 依赖)
        rejections = ["always use Tanimoto kernel for GP", "add CRITICAL: never use RBF"]
        proposal = "always use Tanimoto kernel for GP regression"
        result = asyncio.run(adversarial_critique(
            mode="meta",
            proposal=proposal,
            recent_rejections=rejections,
            system_prompt_summary="",
            llm_client=None,
        ))
        assert result["verdict"] == "reject", f"expected reject, got {result}"
        assert result.get("early_reject") is True, "should be early_reject"
        print("Task 3 self-check PASS")

        # FCM self-check: fake model 验证 winner 选择 / 越界保护 / 空候选退化
        class _FakeResp:
            def __init__(self, text): self.content = text

        class _FakeModel:
            """前 3 次 (fork) 返回方案, 第 4 次 (critique) 返回评审 JSON."""
            def __init__(self, critique_json): self.calls = 0; self._cj = critique_json
            def invoke(self, msgs):
                self.calls += 1
                if self.calls <= 3:
                    return _FakeResp(f"plan from call {self.calls}")
                return _FakeResp(self._cj)

        # 1. winner=2 → 选第二个候选, merge_insights 透传
        m = _FakeModel('{"scores": [3, 9, 5], "winner": 2, '
                       '"merge_insights": ["use robust split"], "fatal_flaws": {}}')
        r = asyncio.run(fork_critique_merge("checklist text", "scan text", m, k=3))
        assert r["winner_plan"] == "plan from call 2", r
        assert r["merge_insights"] == ["use robust split"], r
        assert r["scores"] == [3.0, 9.0, 5.0], r

        # 2. winner 越界 → 保护性回落第一个候选
        m = _FakeModel('{"scores": [1, 2, 3], "winner": 99}')
        r = asyncio.run(fork_critique_merge("c", "s", m, k=3))
        assert r["winner_plan"] == "plan from call 1", r

        # 3. critique JSON 坏掉 → fallback 第一个候选, 带 error 字段
        m = _FakeModel("not json at all")
        r = asyncio.run(fork_critique_merge("c", "s", m, k=3))
        assert r["winner_plan"] == "plan from call 1" and "error" in r, r

        # 4. k=1 → 单候选直接过, 不调 critique (calls==1)
        m = _FakeModel('{"winner": 1}')
        r = asyncio.run(fork_critique_merge("c", "s", m, k=1))
        assert r["winner_plan"] == "plan from call 1" and m.calls == 1, r
        print("FCM self-check PASS (4 cases)")

        # TFM self-check: 退火边界 + judge 评审
        assert anneal_fork_count(1.0, 3) == 3
        assert anneal_fork_count(0.7, 3) == 3
        assert anneal_fork_count(0.5, 3) == 2
        assert anneal_fork_count(0.4, 3) == 2
        assert anneal_fork_count(0.2, 3) == 1
        assert anneal_fork_count(0.9, 1) == 1

        class _FakeJudge:
            def __init__(self, text): self._t = text
            def invoke(self, msgs): return _FakeResp(self._t)

        # 1. 正常 winner 选择 + merge_notes 透传
        r = asyncio.run(judge_fork_reports(
            {"fast": "report A", "robust": "report B"}, "checklist",
            _FakeJudge('{"scores": {"fast": 5, "robust": 8}, "winner": "robust", '
                       '"merge_notes": ["use CV split"]}')))
        assert r["winner"] == "robust" and r["merge_notes"] == ["use CV split"], r
        # 2. LLM 编了不存在的 fork 名 → fallback 第一份非空
        r = asyncio.run(judge_fork_reports(
            {"fast": "A", "exact": "B"}, "c", _FakeJudge('{"winner": "nope"}')))
        assert r["winner"] == "fast", r
        # 3. 空报告过滤 → 单候选直接过, 不调 LLM
        r = asyncio.run(judge_fork_reports(
            {"fast": "", "robust": "B"}, "c", _FakeJudge("bad")))
        assert r["winner"] == "robust", r
        # 4. 全空 → winner None
        r = asyncio.run(judge_fork_reports(
            {"fast": " ", "robust": ""}, "c", _FakeJudge("bad")))
        assert r["winner"] is None, r
        print("TFM self-check PASS (anneal 6 + judge 4)")

        # 复现门禁 self-check: 编数字的 fork 被门禁淘汰, 真实数字的免 LLM 直接胜
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            import numpy as np
            ad = Path(td)
            np.save(ad / "stats.npy", np.array([2.6e-20, 1.4e-13, 0.95]))
            # 三种 sci 写法都要抓到
            assert len(_extract_sci_numbers("μ < 2.6e-20 eV, 1.4×10^-13, 3×10⁻⁵")) == 3
            ok, note = _reproduction_gate("limits 2.6e-20 eV and 1.4e-13 eV", ad)
            assert ok and "2/2" in note, note
            ok, note = _reproduction_gate("we found 9.9e-55 and 8.8e-44", ad)
            assert not ok and "FAIL" in note, note
            # 无承重数字 → 不可验证, 放行不误杀
            ok, _ = _reproduction_gate("qualitative discussion only", ad)
            assert ok
            # judge 集成: fabricator 淘汰后单候选, 不调 LLM
            r = asyncio.run(judge_fork_reports(
                {"fast": "we found 9.9e-55 and 8.8e-44",
                 "robust": "limits 2.6e-20 eV and 1.4e-13 eV"},
                "c", _FakeJudge("bad"), artifact_dirs={"fast": ad, "robust": ad}))
            assert r["winner"] == "robust" and r["gate"]["fast"].startswith("FAIL"), r
        print("reproduction gate self-check PASS (4 cases)")
        sys.exit(0)
    main()
