"""RCB Step 3 对抗审计 — 依赖 rcb_utils / rcb_critique / rcb_fork_merge."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from huginn.cli.rcb.audit import (
    _derive_gap_type,
    _infer_beta_1_simple,
    _recompute_report_metrics,
    _should_retry_execute,
    _write_directive_rejection,
)
from huginn.cli.rcb_critique import adversarial_critique, format_critique_for_agent
from huginn.cli.rcb_utils import (
    _MODEL_VERSION,
    _infer_domain,
    _infer_task_id_from_workspace,
    _make_simplex_id,
)
from huginn.utils.runtime import HUGINN_DIR_NAME

logger = logging.getLogger(__name__)


async def _step3_adversarial(
    ws: Path,
    model: Any,
    agent: Any,
    checklist: str,
    evals_history: list,
    stream_chat_fn,
    rcb_csm_advance_fn,
    persona: Any = None,
    kb: Any = None,
    mem_mgr: Any = None,
    cross_task_store: Any = None,
    task_id: str = "",
    persona_name: str = "default",
) -> str | None:
    """Step 3: 对抗式自检 — skeptical reviewer 视角找 gap.

    返回最终 critique verdict ("pass" / "fix_needed" / None), 供 v14 Task 18
    失败 trace 进训练池判定 task score 是否 <20 的代理.

    ponytail: 治 3 个系统性短板 (跨 4 题评分发现的共性 gap):
      A. sanity check — 治 "不可信结果不自检"
      B. substitution audit — 治 "沉默方法降级"
      C. hard push — 治 "硬组件轻易放弃"
    双层 critique: object mode (report) + meta mode (directive).

    v14 Task 8: critique 后若 verdict=fix_needed + β_1>0 + gap 类型匹配,
    触发 Step3→Step2 reverse 1-simplex 让 agent 回 execute 重跑. 最多 2 次
    (spec §"回退次数上限"), 超过写 directive_rejections.jsonl 强制 finalize.
    """
    report_path, checklist, final_verdict = await _step3_first_pass(
        ws, model, agent, checklist, evals_history, stream_chat_fn, rcb_csm_advance_fn
    )

    retry_ctx = (persona, kb, mem_mgr, evals_history)
    final_verdict = await _run_retry_stage(
        ws, model, checklist, report_path, stream_chat_fn, retry_ctx, final_verdict
    )

    fin_ctx = (persona_name, task_id, cross_task_store)
    return await _finalize_step3(
        ws, checklist, final_verdict, mem_mgr, stream_chat_fn, fin_ctx
    )


async def _step3_first_pass(
    ws: Path,
    model: Any,
    agent: Any,
    checklist: str,
    evals_history: list,
    stream_chat_fn: Any,
    rcb_csm_advance_fn: Any,
) -> tuple[Path, str, str | None]:
    """首轮: object/meta 双层 critique + execute."""
    (
        report_path,
        checklist,
        final_verdict,
        external_critique_block,
    ) = await _critique_step3(
        ws, model, agent, checklist, evals_history, rcb_csm_advance_fn
    )

    await _run_step3_execute(
        ws, stream_chat_fn, _build_step3_prompt(external_critique_block)
    )
    return report_path, checklist, final_verdict


async def _critique_step3(
    ws: Path,
    model: Any,
    agent: Any,
    checklist: str,
    evals_history: list,
    rcb_csm_advance_fn: Any,
) -> tuple[Path, str, str | None, str]:
    """首轮: 打印头 + CSM 推进 + 对象层 critique + meta 层触发."""
    print("\n=== Step 3: Adversarial Self-Critique ===\n", flush=True)
    rcb_csm_advance_fn("tool_failure", {"reason": "adversarial critique — find gaps"})

    report_path = ws / "report" / "report.md"
    (
        checklist,
        object_verdict,
        external_critique_block,
        final_verdict,
    ) = await _run_object_critique(ws, model, checklist, evals_history, report_path)

    # Layer 2 — meta mode: 触发 CSM 进 S6_FEEDBACK → S7_SELF_MODIFY
    _trigger_meta_feedback(agent, object_verdict, external_critique_block)
    return report_path, checklist, final_verdict, external_critique_block


async def _run_retry_stage(
    ws: Path,
    model: Any,
    checklist: str,
    report_path: Path,
    stream_chat_fn: Any,
    retry_ctx: tuple,
    initial_verdict: str | None,
) -> str | None:
    """v14 Task 8: Step3→Step2 回退通道 (拓扑许可动力学)."""
    # 重新 critique 看 agent 修没修好; 仍 fix_needed + β_1>0 + gap 类型匹配
    # 则发 step3_retry 让 agent 回 execute 模式重跑. 硬上限 2 次.
    persona, kb, mem_mgr, evals_history = retry_ctx
    xretry_path = _prepare_cross_retry_path(ws)
    return await _run_step3_retry_loop(
        ws=ws,
        model=model,
        checklist=checklist,
        report_path=report_path,
        stream_chat_fn=stream_chat_fn,
        persona=persona,
        kb=kb,
        mem_mgr=mem_mgr,
        evals_history=evals_history,
        xretry_path=xretry_path,
        initial_verdict=initial_verdict,
    )


async def _finalize_step3(
    ws: Path,
    checklist: str,
    final_verdict: str | None,
    mem_mgr: Any,
    stream_chat_fn: Any,
    fin_ctx: tuple,
) -> str | None:
    """收尾: evidence manifest + learn 回流 + 盲重构验证."""
    persona_name, task_id, cross_task_store = fin_ctx
    # 生成 evidence manifest 用于复现性审计.
    _write_evidence_manifest(ws)

    # P4 Task 24: 接 autoloop _learn — 写 long-term memory + persona_history +
    # cross_task_store. ponytail: 抽顶层 learn_from_rcb wrapper, 不实例化 engine.
    #   失败 try/except 不阻塞 finalize. skill abstraction / evolution 留升级路径.
    _run_learn_from_rcb(
        ws=ws,
        checklist=checklist,
        final_verdict=final_verdict,
        mem_mgr=mem_mgr,
        persona_name=persona_name,
        task_id=task_id,
        cross_task_store=cross_task_store,
    )

    # P4 Task 25: 盲重构验证 — 验证通过才记 final score, 失败要求 agent 重做.
    if not await _run_blind_reconstruct_verify(ws, final_verdict, stream_chat_fn):
        return "blind_reconstruct_failed"

    return final_verdict


async def _run_object_critique(
    ws: Path,
    model: Any,
    checklist: str,
    evals_history: list,
    report_path: Path,
) -> tuple[str, dict | None, str, str | None]:
    """Layer 1 — object mode critique: 注入 A2/A3/P2 信号后审 report.

    返回 (可能被追加信号的 checklist, object_verdict, 外部 critique 文本, verdict).
    """
    if not (report_path.exists() and checklist):
        print(
            "[adversarial_critique: skipped — report.md or checklist missing]",
            flush=True,
        )
        return checklist, None, "", None
    try:
        report_text = report_path.read_text(encoding="utf-8")
        print(
            f"[adversarial_critique: reading {len(report_text)} chars of report.md]",
            flush=True,
        )
        checklist = _inject_repro_gate(report_text, ws, checklist)
        checklist = _inject_step2_signals(ws, checklist)
        checklist = _inject_uncertainty_signals(evals_history, checklist)
        object_verdict = await adversarial_critique(
            model,
            report_text,
            checklist,
            mode="object",
        )
        object_verdict = _apply_metric_recompute(object_verdict, report_text, ws)
        external_critique_block = format_critique_for_agent(object_verdict)
        print(
            f"[adversarial_critique: verdict={object_verdict.get('overall_verdict', '?')}]",
            flush=True,
        )
        final_verdict = (
            object_verdict.get("overall_verdict", "fix_needed")
            if object_verdict
            else None
        )
        return checklist, object_verdict, external_critique_block, final_verdict
    except Exception as e:
        print(f"[adversarial_critique: skipped due to error: {e}]", flush=True)
        return checklist, None, "", None


async def _run_step3_execute(ws: Path, stream_chat_fn: Any, step3_prompt: str) -> None:
    """Step 3 主审 — 独立 thread 隔离 dangling tool_calls, 异常不炸整个 run."""
    # Step 3 用独立 thread 隔离 Step 1/2 末尾可能 dangling 的 tool_calls 历史.
    # DeepSeek API 严格校验 assistant(tool_calls) 必须紧跟 tool message, Step 2
    # 跑完若 agent 调工具被中断 (timeout/exception), thread 历史会留 dangling,
    # 同 thread 调 Step 3 直接 400. ConversationTree 全局重建历史, 换 thread 不丢上下文.
    # ponytail: 单独 _step3 后缀, 不污染 Step 1/2 thread. 升级: 修 checkpointer
    # 的 history restorer 自动过滤 dangling tool_calls (留后续).
    _step3_tid = f"rcb_{ws.name}_step3"
    # Step 3 stream_chat_fn 异常不能炸整个 run — v5/v7 都死在这里, LLM 调用
    # 抛异常 (timeout/recursion limit/dangling tool_calls 400) → 冒泡到 run()
    # → 跳过所有评分逻辑 → 0 分. 兜底: 记录异常, 继续走 retry loop + 评分.
    try:
        await stream_chat_fn(step3_prompt, "step3", tid=_step3_tid, fresh_history=True)
    except Exception as _e:
        import traceback as _tb

        print(f"[step3] stream_chat_fn failed: {_e}\n{_tb.format_exc()}", flush=True)


def _inject_repro_gate(report_text: str, ws: Path, checklist: str) -> str:
    """复现门禁: report.md 承重数字查 outputs/ 复现性 (advisory, 非否决)."""
    # ponytail: _reproduction_gate 内部已抽 sci-notation (|exp|>=3), 不重复抽取.
    try:
        from huginn.cli.rcb_fork_merge import _reproduction_gate

        _repro_ok, _repro_note = _reproduction_gate(report_text, ws / "outputs")
        if not _repro_ok:
            # gate 是粗筛 advisory, 不是 veto. v21 死在 veto: agent 写
            # "rel err < 2×10⁻¹⁶" (机器精度) 被当 claim, outputs/ 没匹配
            # → gate fail → score=0, 但 agent 产出了完整 pipeline + ALL PASS.
            # gate 对 sci-notation 以外的 claim (整数序列、MAE、分类标签)
            # 本就抓不到, 作为硬 veto 会系统性误杀非能量类论文.
            # 注入 note 让 LLM critique 判断, 不跳过.
            print(f"[Step3] repro gate advisory: {_repro_note}", flush=True)
            checklist = (
                checklist
                + f"\n\n## 复现性门禁 (advisory, 非否决)\n{_repro_note}\n请结合 outputs/ 产物完整性综合判断, 勿仅凭门禁结果打分.\n"
            )
    except Exception as _e:
        print(f"[Step3] repro gate skipped: {_e}", flush=True)
    return checklist


def _inject_step2_signals(ws: Path, checklist: str) -> str:
    """A2/A3 信号注入 — 让 critic 看到 Step 2 结束时的 audit 结果.

    ponytail: 纯文本拼接, 不改 adversarial_critique 签名. critic 是 LLM,
    看到 "BLOCKER: outputs/ 无真实 metrics" 自然会降权 Results claim.
    """
    try:
        _audit_path = ws / HUGINN_DIR_NAME / "step2_audit.json"
        _gate_path = ws / HUGINN_DIR_NAME / "step2_outputs_gate.json"
        if _audit_path.exists():
            _a = json.loads(_audit_path.read_text(encoding="utf-8"))
            _unresolved = _a.get("unresolved", [])
            if _unresolved:
                _blk = "\n\n## A3 Substitution Audit (Step 2 结束机械比对)\n"
                _blk += "以下 [EXACT] 组件无实现痕迹且未声明 METHOD SUBSTITUTE:\n"
                for it in _unresolved:
                    _blk += f"- {it.get('component', '?')} (失败 {it.get('failures', 0)} 次)\n"
                _blk += "Results 中关于这些组件的 claim 应判未执行 / 0 分.\n"
                checklist = checklist + _blk
        if _gate_path.exists():
            _g = json.loads(_gate_path.read_text(encoding="utf-8"))
            if _g.get("blocker"):
                checklist = checklist + (
                    "\n\n## A2 Outputs Gate (Step 2 结束产物门控)\n"
                    "BLOCKER: outputs/ 无真实 metrics 文件. "
                    "report.md 中所有 Results 数值 claim 缺乏产物支撑, "
                    "应判 fabricated / 0 分. Discussion 应显式声明 "
                    "'EXECUTION BLOCKER' 而非叙述实验结果.\n"
                )
    except Exception as _e:
        print(f"[Step3] A2/A3 signal inject skipped: {_e}", flush=True)
    return checklist


def _inject_uncertainty_signals(evals_history: list, checklist: str) -> str:
    """P2 check: 误差建模缺失信号注入 critique."""
    try:
        from huginn.metacog.step_evaluator import check_uncertainty_propagation

        _unc_issues = check_uncertainty_propagation(evals_history)
        if _unc_issues:
            _unc_block = "\n\n## 误差建模缺失 (P2 check)\n"
            for iss in _unc_issues:
                _unc_block += (
                    f"- step {iss['step_id']}: {iss['issue']} — {iss['detail']}\n"
                )
            checklist = checklist + _unc_block
            print(
                f"[P2: {len(_unc_issues)} uncertainty issue(s) injected to critique]",
                flush=True,
            )
    except Exception as _ue:
        print(f"[P2 uncertainty check skipped: {_ue}]", flush=True)
    return checklist


def _apply_metric_recompute(object_verdict: dict, report_text: str, ws: Path) -> dict:
    """G28: report 数值 claim 与 outputs/ 复算值交叉校验, 不一致 → red flag."""
    try:
        recomputed = _recompute_report_metrics(report_text, ws)
        if recomputed:
            object_verdict.setdefault("recomputed_red_flags", []).extend(recomputed)
            if object_verdict.get("overall_verdict") == "pass":
                object_verdict["overall_verdict"] = "fix_needed"
            print(
                f"[G28: {len(recomputed)} metric claim(s) mismatch recomputed values]",
                flush=True,
            )
    except Exception as e:
        print(f"[G28: recompute skipped: {e}]", flush=True)
    return object_verdict


def _trigger_meta_feedback(
    agent: Any, object_verdict: dict | None, external_critique_block: str
) -> None:
    """Layer 2 — meta mode: 触发 CSM 进 S6_FEEDBACK → S7_SELF_MODIFY."""
    try:
        from huginn.cognitive_engine import CognitiveState, TransitionSignal

        csm = getattr(agent, "_csm", None)
        if csm is None or object_verdict is None:
            return
        verdict_flag = object_verdict.get("overall_verdict", "fix_needed")
        sig = "tool_failure" if verdict_flag != "pass" else "tool_success"
        new_state = csm.transition(
            TransitionSignal(
                sig,
                {
                    "objective": "step3_critique",
                    "result_summary": f"object_verdict={verdict_flag}",
                },
            )
        )
        if new_state == CognitiveState.S6_FEEDBACK and verdict_flag != "pass":
            csm.transition(
                TransitionSignal(
                    "gap_found",
                    {
                        "gap": external_critique_block[:200]
                        or "step3 object critique red flags",
                    },
                )
            )
    except Exception:
        logger.debug("Step 3 CSM S6/S7 trigger failed", exc_info=True)


_STEP3_BASE_PROMPT = (
    "ADVERSARIAL SELF-CRITIQUE. You are now a SKEPTICAL REVIEWER who wants to score this report LOW. "
    "Do NOT be lenient with yourself.\n\n"
    "## A. Sanity Check (do this FIRST — catches fabricated/impossible results)\n"
    "Read your report/report.md. Extract EVERY quantitative claim (MAE, R², accuracy, loss, etc.).\n"
    "RECOMPUTE each claim via an independent computation path — do NOT trust your own previous text. "
    "Call validate_tool (numerical cross-check) or symbolic_math_tool (closed-form re-derivation) "
    "or code_tool (re-run the metric on outputs/ artifacts) for each claimed number. "
    "P1-B3 hard rule: a claim that you cannot recompute independently is FABRICATED, treat it as 0. "
    "Build a table: | Metric | Claimed | Recomputed (independent path) | Tool used | Match? |\n"
    "If Claimed != Recomputed (>1% deviation), that is a RED FLAG — fix the number in report.md. "
    "If ANY recomputed metric is BETTER than the paper's baseline, that is also a RED FLAG.\n"
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
    "## D. Visual Self-Check (if report has figures)\n"
    "For each figure in report/images/, call image_analysis tool with action='compare_to_target' "
    "and parameters={'target_path': <corresponding paper target image>, 'candidate_paths': [<your figure>]}. "
    "The target image is in the task's target_study/images/ directory. "
    "If CV similarity < 40, regenerate the figure. If 40-70, improve it. If > 70, keep it.\n"
    "Do NOT blindly regenerate figures that already match the paper.\n\n"
    "## E. Fix & Rewrite (SURGICAL, not full rewrite)\n"
    "For each gap found in A/B/C:\n"
    "  - Missing metric → compute it now (run code_tool)\n"
    "  - Missing [EXACT] component → implement it (push through, try ≥2 approaches before giving up)\n"
    "  - Implausible result → fix the bug or document honestly why it's off\n"
    "CRITICAL RULE: Only modify the SPECIFIC sections/numbers that critique flagged as problematic. "
    "DO NOT touch numbers, tables, or sections that were NOT flagged. "
    "DO NOT regenerate figures that passed visual self-check (D). "
    "Preserve all correct results. A surgical edit beats a full rewrite — full rewrites risk "
    "breaking numbers that were already correct.\n"
    "Update report/report.md with: surgical fixes to flagged gaps + baseline comparison table + "
    "honest Limitations section (only for items where you tried ≥2 approaches and genuinely failed).\n"
    "Use file_write_tool for the update."
)


def _build_step3_prompt(external_critique_block: str) -> str:
    """构造对抗自检 prompt (sanity / substitution / coverage / visual / fix 五段)."""
    step3_prompt = _STEP3_BASE_PROMPT
    if external_critique_block:
        step3_prompt = (
            external_critique_block
            + "\n\n## Now act on the critique above:\n"
            + step3_prompt
        )
    return step3_prompt


def _prepare_cross_retry_path(ws: Path) -> Path:
    """cross_retry.jsonl 路径初始化 (清上次的, 避免跨 task 污染)."""
    # cross-retry memory: 记录每次 retry 的 gap + report diff, 下次 retry 注入.
    # 之前 retry1/2 互不通信, retry2 重蹈 retry1 失败覆辙. 现在 retry1 改了啥
    # 写 cross_retry.jsonl, retry2 prompt 注入 "retry1 试了 X 方案仍 fix_needed".
    _xretry_path = ws / HUGINN_DIR_NAME / "cross_retry.jsonl"
    try:
        _xretry_path.parent.mkdir(parents=True, exist_ok=True)
        _xretry_path.unlink(missing_ok=True)
    except Exception:
        logger.debug("cross_retry path setup skipped", exc_info=True)
    return _xretry_path


def _step3_deadline_expired() -> bool:
    """deadline 检查: 剩余 < 600s 时跳过 retry → 直接评分.

    retry 很重 (agent 回 execute 模式重跑), 没时间预算感知会一直跑到外部
    asyncio.wait_for 超时杀进程, 连评分都跑不到 (v20 死在这).
    留 600s 给后续评分+manifest.
    """
    _rcb_deadline = os.environ.get("HUGINN_RCB_DEADLINE")
    if not _rcb_deadline:
        return False
    try:
        _remaining = float(_rcb_deadline) - time.time()
    except (ValueError, TypeError):
        logger.debug("best-effort op failed", exc_info=True)
        return False
    if _remaining < 600:
        print(
            f"[step3_retry: deadline in {_remaining:.0f}s, skip retry → score",
            flush=True,
        )
        return True
    return False


async def _run_step3_retry_loop(
    *,
    ws: Path,
    model: Any,
    checklist: str,
    report_path: Path,
    stream_chat_fn,
    persona: Any,
    kb: Any,
    mem_mgr: Any,
    evals_history: list,
    xretry_path: Path,
    initial_verdict: str | None,
) -> str | None:
    """Step3→Step2 回退循环 (硬上限 2 次), 返回最新 verdict."""
    _retry_count = 0
    _final_verdict = initial_verdict
    _trace_task_id_s3 = _infer_task_id_from_workspace(ws.name)
    _prompt_kwargs = {
        "persona": persona,
        "kb": kb,
        "mem_mgr": mem_mgr,
        "evals_history": evals_history,
        "xretry_path": xretry_path,
        "trace_task_id": _trace_task_id_s3,
    }
    while True:
        if not report_path.exists() or not checklist:
            break
        if _step3_deadline_expired():
            break
        _recritique = await _recritique_step3(model, checklist, report_path)
        if _recritique is None:
            break
        _retry_report, _verdict_dict = _recritique
        _verdict, _retry_count, _done = await _step3_retry_round(
            ws,
            stream_chat_fn,
            report_path,
            _prompt_kwargs,
            _retry_count,
            _retry_report,
            _verdict_dict,
        )
        _final_verdict = _verdict
        if _done:
            break
    return _final_verdict


async def _recritique_step3(
    model: Any, checklist: str, report_path: Path
) -> tuple[str, dict] | None:
    """重审 report; 失败打印并返回 None."""
    try:
        _retry_report = report_path.read_text(encoding="utf-8")
        _verdict_dict = await adversarial_critique(
            model,
            _retry_report,
            checklist,
            mode="object",
        )
    except Exception as _e:
        print(f"[step3_retry: re-critique failed: {_e}]", flush=True)
        return None
    return _retry_report, _verdict_dict


async def _step3_retry_round(
    ws: Path,
    stream_chat_fn: Any,
    report_path: Path,
    prompt_kwargs: dict[str, Any],
    retry_count: int,
    retry_report: str,
    verdict_dict: dict,
) -> tuple[str, int, bool]:
    """判定 + 执行一次回退; 返回 (verdict, retry_count, 循环是否结束)."""
    _retry_verdict = verdict_dict.get("overall_verdict", "fix_needed")
    _retry_gap = _derive_gap_type(verdict_dict)
    _retry_beta_1 = _infer_beta_1_simple(ws)

    if not _should_retry_execute(_retry_verdict, _retry_beta_1, _retry_gap):
        print(
            f"[step3_retry: verdict={_retry_verdict}, gap={_retry_gap}, no retry]",
            flush=True,
        )
        return _retry_verdict, retry_count, True

    if retry_count >= 2:
        await _finalize_after_retry_limit(
            ws=ws,
            retry_count=retry_count,
            gap=_retry_gap,
            verdict=_retry_verdict,
            stream_chat_fn=stream_chat_fn,
        )
        return _retry_verdict, retry_count, True

    retry_count += 1
    _critique_summary = format_critique_for_agent(verdict_dict)[:500]
    _ok = await _attempt_step3_retry(
        ws,
        stream_chat_fn,
        report_path,
        prompt_kwargs,
        retry_count,
        retry_report,
        _retry_verdict,
        _retry_gap,
        _critique_summary,
        _retry_beta_1,
    )
    return _retry_verdict, retry_count, not _ok


async def _attempt_step3_retry(
    ws: Path,
    stream_chat_fn: Any,
    report_path: Path,
    prompt_kwargs: dict[str, Any],
    retry_count: int,
    retry_report: str,
    verdict: str,
    gap: str,
    critique_summary: str,
    beta_1: int,
) -> bool:
    """构造 retry prompt + 写 trace + 执行 + 记 cross-retry; 返回是否成功."""
    _retry_execute_prompt = _build_step3_retry_prompt(
        ws, prompt_kwargs, verdict, gap, critique_summary, retry_report, retry_count
    )
    _write_retry_trace(
        ws,
        trace_task_id=prompt_kwargs["trace_task_id"],
        retry_count=retry_count,
        critique_summary=critique_summary,
        verdict=verdict,
        gap=gap,
        beta_1=beta_1,
    )
    print(f"[step3_retry: attempt {retry_count}/2, gap={gap}]", flush=True)
    if not await _stream_step3_retry(
        ws,
        stream_chat_fn,
        _retry_execute_prompt,
        retry_count,
    ):
        return False
    _log_cross_retry(
        prompt_kwargs["xretry_path"],
        retry_count=retry_count,
        gap=gap,
        critique_summary=critique_summary,
        report_before=retry_report,
        report_path=report_path,
    )
    return True


def _build_step3_retry_prompt(
    ws: Path,
    prompt_kwargs: dict[str, Any],
    verdict: str,
    gap: str,
    critique_summary: str,
    retry_report: str,
    retry_count: int,
) -> str:
    """组装 step3 retry 的执行 prompt (注入 PMK/KB/state/critique)."""
    return _build_retry_execute_prompt(
        ws=ws,
        verdict=verdict,
        gap=gap,
        critique_summary=critique_summary,
        report=retry_report,
        retry_count=retry_count,
        persona=prompt_kwargs["persona"],
        kb=prompt_kwargs["kb"],
        mem_mgr=prompt_kwargs["mem_mgr"],
        evals_history=prompt_kwargs["evals_history"],
        xretry_path=prompt_kwargs["xretry_path"],
    )


async def _stream_step3_retry(
    ws: Path, stream_chat_fn: Any, retry_execute_prompt: str, retry_count: int
) -> bool:
    """fresh thread 跑一次回退执行; 失败返回 False."""
    # fresh thread per retry — 避免同 _step3_tid 累积消息超限.
    # 每次重试都是独立 thread + 结构化 state injection (上方构造).
    # fresh_history=True: 不拉 ConversationTree 历史, 避免 Step 2 的
    # 1M+ tokens 累积. retry prompt 已注入 PMK/KB/state/critique 全部 state.
    _retry_tid = f"rcb_{ws.name}_step3_retry{retry_count}"
    try:
        # A4: Step-3 verdict≠pass 追加专用 50 次预算 — 路线图 P1-A4 / 05 报告 R4.
        # retry 本就是 fix_needed 时回退执行, 但默认用 agent 全局 max_tool_calls
        # (150/300) — Step 2 已烧光, retry 没预算等于空跑. 专用 50 次预算池
        # 让 agent 真有资源修 gap. ponytail: BudgetSpec 在 _stream_chat 内构造,
        # 失败不影响 agent 全局配置.
        await stream_chat_fn(
            retry_execute_prompt,
            "step3_retry",
            tid=_retry_tid,
            fresh_history=True,
            extra_budget=50,
        )
    except Exception as _e:
        print(f"[step3_retry {retry_count}] stream_chat_fn failed: {_e}", flush=True)
        return False
    return True


async def _finalize_after_retry_limit(
    *,
    ws: Path,
    retry_count: int,
    gap: str,
    verdict: str,
    stream_chat_fn,
) -> None:
    """spec §"回退次数上限" — 写 rejection + 强制 finalize."""
    _write_directive_rejection(ws, gap, verdict, retry_count)
    _report_text = ""
    try:
        _rp = ws / "report" / "report.md"
        if _rp.exists():
            _report_text = _rp.read_text(encoding="utf-8")
    except Exception:
        logger.debug("report.md read for finalize skipped", exc_info=True)
    _finalize_prompt = (
        f"Retry limit reached ({retry_count}/2). Critique still finds gap "
        f"(type={gap}, verdict={verdict}).\n"
        f"Add to report/report.md Limitations section: "
        f"'Attempted {retry_count} retries, could not fix gap ({gap}).'\n"
        f"UPDATE report/report.md — keep all existing content, only ADD the "
        f"Limitations note above. Do NOT overwrite or remove existing sections.\n"
    )
    if _report_text:
        _finalize_prompt += (
            f"\n## Current report content (PRESERVE THIS):\n{_report_text[:8000]}"
        )
    # fresh thread 避免累积 — finalize 只需当前 report + critique verdict.
    # 之前用 _step3_tid 会累积 step3 + step3_retry 历史 → 1.6M tokens 超限.
    _finalize_tid = f"rcb_{ws.name}_step3_finalize"
    try:
        await stream_chat_fn(
            _finalize_prompt, "step3_finalize", tid=_finalize_tid, fresh_history=True
        )
    except Exception as _e:
        print(f"[step3_finalize] stream_chat_fn failed: {_e}", flush=True)


def _read_instructions_excerpt(ws: Path) -> str:
    """读 INSTRUCTIONS.md 前 2000 字符 (fresh thread 的 task gradient)."""
    _inst_text = ""
    try:
        _inst_path = ws / "INSTRUCTIONS.md"
        if _inst_path.exists():
            _inst_text = _inst_path.read_text(encoding="utf-8")[:2000]
    except Exception:
        logger.debug("INSTRUCTIONS.md read skipped", exc_info=True)
    return _inst_text


def _build_pmk_block(
    persona: Any,
    evals_history: list,
    kb: Any,
    mem_mgr: Any,
    retry_gap: str,
) -> tuple[str, str]:
    """PMK 三路立场 + KB 专业召回 (按 gap 关键词二次召回)."""
    # 之前 retry 是 PMK 孤岛: persona/kb 都没传进来, 现在 build_pmk_state
    # 抽 Persona/Memory/KB 三路立场, KB 再用 gap 关键词二次召回专业背景.
    # ponytail: KB.query 是字符串匹配, 不上 LLM rerank; 升级路径接 Ising rerank.
    _pmk_block = ""
    _kb_recall = ""
    try:
        from huginn.autoloop.cognitive_loop import build_pmk_state

        _last_eval = evals_history[-1] if evals_history else None
        _pmk = build_pmk_state(persona, _last_eval, kb, mem_mgr=mem_mgr)
        if _pmk:
            _pmk_block = (
                f"### PMK 三路立场 (Step 2 末态)\n"
                f"- Persona: {_pmk.get('persona', '') or '(无)'}\n"
                f"- Memory:  {_pmk.get('memory', '') or '(无)'}\n"
                f"- KB:      {_pmk.get('kb', '') or '(无)'}\n\n"
            )
        if kb is not None and retry_gap:
            _hits = kb.query(retry_gap, top_k=2)
            if _hits:
                _kb_recall = (
                    "### KB 专业召回 (按 gap 关键词)\n"
                    + " ".join(
                        str(h.get("content", "") if isinstance(h, dict) else h)[:300]
                        for h in _hits[:2]
                    )
                    + "\n\n"
                )
    except Exception as _e:
        print(f"[step3_retry PMK/KB injection skipped: {_e}]", flush=True)
    return _pmk_block, _kb_recall


def _scan_state_checkpoint(ws: Path) -> str:
    """扫 outputs/code/images/report 文件清单 + 大小 (retry 可复用产物)."""
    # retry agent 知道 "已有 gp_model.pkl 可以复用", 不用从头跑. 避免 retry
    # 重复 Step 2 的 EDA/数据加载/模型训练, 直接复用已有 artifacts.
    # ponytail: 纯文件系统扫描, 无 LLM; 升级路径: LLM 抽取 artifact 语义.
    _state_ckpt = ""
    try:
        _ckpt_files = []
        for _sub in ("outputs", "code", "images", "report"):
            _sp = ws / _sub
            if not _sp.exists():
                continue
            for _fp in _sp.rglob("*"):
                if _fp.is_file() and _fp.stat().st_size > 0:
                    _rel = _fp.relative_to(ws).as_posix()
                    _sz = _fp.stat().st_size
                    _unit = "B" if _sz < 1024 else ("KB" if _sz < 1048576 else "MB")
                    _sz_str = (
                        f"{_sz / 1024:.1f}KB"
                        if _unit == "KB"
                        else (f"{_sz / 1048576:.1f}MB" if _unit == "MB" else f"{_sz}B")
                    )
                    _ckpt_files.append(f"  - {_rel} ({_sz_str})")
        if _ckpt_files:
            _state_ckpt = (
                "### State Checkpoint (Step 2 已生成, 可复用 — 不要从头跑)\n"
                + "\n".join(_ckpt_files[:25])
                + "\n\n"
            )
    except Exception:
        logger.debug("state checkpoint scan skipped", exc_info=True)
    return _state_ckpt


def _build_cross_retry_block(xretry_path: Path) -> str:
    """Cross-retry memory — 上次 retry 的 gap + report diff 摘要."""
    # retry1 改了 X 行但仍 fix_needed, retry2 知道后试不同策略.
    # ponytail: difflib 算 added/removed lines; 升级路径: LLM summarize diff.
    _xretry_block = ""
    try:
        if xretry_path.exists():
            _xretry_lines = xretry_path.read_text(encoding="utf-8").strip().split("\n")
            _xretry_entries = []
            for _xl in _xretry_lines:
                if _xl.strip():
                    with contextlib.suppress(Exception):
                        _xretry_entries.append(json.loads(_xl))
            if _xretry_entries:
                _xretry_block = "### Cross-Retry Memory (避免重蹈覆辙)\n"
                for _xr in _xretry_entries:
                    _xretry_block += (
                        f"- Retry {_xr.get('retry', '?')}: gap={_xr.get('gap', '?')}, "
                        f"changed +{_xr.get('added_lines', 0)}/-{_xr.get('removed_lines', 0)} lines, "
                        f"verdict_after=still_fix_needed\n"
                    )
                _xretry_block += (
                    "\nDO NOT repeat the same approach. Try a DIFFERENT strategy "
                    "(e.g. if retry1 added text, retry2 should RECOMPUTE data).\n\n"
                )
    except Exception:
        logger.debug("cross-retry memory read skipped", exc_info=True)
    return _xretry_block


def _build_retry_execute_prompt(
    *,
    ws: Path,
    verdict: str,
    gap: str,
    critique_summary: str,
    report: str,
    retry_count: int,
    persona: Any,
    kb: Any,
    mem_mgr: Any,
    evals_history: list,
    xretry_path: Path,
) -> str:
    """构造 retry prompt: fresh thread + 结构化 state injection."""
    # 结构化 state injection: fresh thread 注入 INSTRUCTIONS + 前次 verdict,
    # 不截断不累积. 比 simple truncation 智能: 保留 task gradient + critique context.
    # ponytail: 升级路径 — LLM summarization 替代 fresh thread (保留中间过程),
    # 但 Step 3 是 critique report, 不需要 Step 2 中间过程, fresh thread 已够.
    _inst_text = _read_instructions_excerpt(ws)
    _pmk_block, _kb_recall = _build_pmk_block(persona, evals_history, kb, mem_mgr, gap)
    _state_ckpt = _scan_state_checkpoint(ws)
    _xretry_block = _build_cross_retry_block(xretry_path)
    return (
        f"## Task Context (fresh thread — previous history not carried)\n"
        f"### INSTRUCTIONS.md (excerpt):\n{_inst_text}\n\n"
        f"{_pmk_block}"
        f"{_kb_recall}"
        f"{_state_ckpt}"
        f"{_xretry_block}"
        f"### Previous Critique Verdict: {verdict}\n"
        f"### Gap Type: {gap}\n"
        f"### Critique Summary:\n{critique_summary}\n\n"
        f"## Current Report (full content below — DO NOT lose existing sections):\n"
        f"{report}\n\n"
        f"## Task: Return to EXECUTE mode. Re-run code_tool to fix the gap.\n"
        f"UPDATE report/report.md after fix — keep all existing valid sections, "
        f"only fix the gap identified above. DO NOT overwrite with a skeleton.\n"
        f"Retry attempt {retry_count}/2."
    )


def _write_retry_trace(
    ws: Path,
    *,
    trace_task_id: str,
    retry_count: int,
    critique_summary: str,
    verdict: str,
    gap: str,
    beta_1: int,
) -> None:
    """写 trace entry 标记回退事件 (cochain_type="curl", role="step3_retry")."""
    try:
        _trace_path_s3 = ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
        _trace_path_s3.parent.mkdir(parents=True, exist_ok=True)
        _retry_entry = {
            "iteration": -1,
            "ts": time.time(),
            "role": "step3_retry",
            "attempted": critique_summary,
            "found": "retry triggered",
            "evidence": f"verdict={verdict}, gap_type={gap}, beta_1={beta_1}",
            "limitations": "",
            "artifacts": [],
            "next_hint": "re-execute and fix gap",
            "darwin_score": 0.3,
            "supported_ratio": 0.0,
            "simplex_id": _make_simplex_id(trace_task_id, retry_count, "step3_retry"),
            "cochain_type": "curl",
            "domain": _infer_domain(trace_task_id),
            "task_id": trace_task_id,
            "model_version": _MODEL_VERSION,
        }
        with _trace_path_s3.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(_retry_entry, ensure_ascii=False) + "\n")
    except Exception as _e:
        print(f"[step3_retry trace write failed: {_e}]", flush=True)


def _log_cross_retry(
    xretry_path: Path,
    *,
    retry_count: int,
    gap: str,
    critique_summary: str,
    report_before: str,
    report_path: Path,
) -> None:
    """记录 cross-retry memory: 算 report diff, 下次 retry 注入."""
    # retry1 改了 +X/-Y 行, retry2 看到后试不同策略 (不重蹈覆辙).
    try:
        import difflib as _dl

        _report_after_retry = (
            report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        )
        _diff = _dl.unified_diff(
            report_before.splitlines(),
            _report_after_retry.splitlines(),
            lineterm="",
        )
        _added = sum(
            1 for _l in _diff if _l.startswith("+") and not _l.startswith("+++")
        )
        _removed = sum(
            1 for _l in _diff if _l.startswith("-") and not _l.startswith("---")
        )
        _xentry = {
            "retry": retry_count,
            "gap": gap,
            "critique_before": critique_summary[:200],
            "added_lines": _added,
            "removed_lines": _removed,
            "report_chars_before": len(report_before),
            "report_chars_after": len(_report_after_retry),
        }
        with xretry_path.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(_xentry, ensure_ascii=False) + "\n")
    except Exception as _e:
        print(f"[step3_retry cross_retry log failed: {_e}]", flush=True)


def _write_evidence_manifest(ws: Path) -> None:
    """生成 evidence manifest 用于复现性审计."""
    # ponytail: generate_evidence_manifest 只返 dict 不落盘, 这里写 manifest.json.
    try:
        from huginn.bench.evidence_manifest import generate_evidence_manifest

        _manifest = generate_evidence_manifest(ws)
        _manifest_path = ws / "outputs" / "manifest.json"
        _manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _manifest_path.write_text(
            json.dumps(_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[Step3] manifest.json generated at {_manifest_path}", flush=True)
    except Exception as _e:
        print(f"[Step3] manifest generation failed: {_e}", flush=True)


def _run_learn_from_rcb(
    *,
    ws: Path,
    checklist: str,
    final_verdict: str | None,
    mem_mgr: Any,
    persona_name: str,
    task_id: str,
    cross_task_store: Any,
) -> None:
    """P4 Task 24: 接 autoloop _learn — 写 long-term memory + cross_task_store."""
    try:
        from huginn.autoloop.cognitive_loop import learn_from_rcb

        _learn_hyp = ""
        _report_path_final = ws / "report" / "report.md"
        if _report_path_final.exists():
            _learn_hyp = _report_path_final.read_text(encoding="utf-8")[:400]
        if not _learn_hyp:
            _learn_hyp = (checklist or "")[:200]
        _learn_validation = {
            "tests_passed": final_verdict == "pass",
            "darwin_score": 0.6 if final_verdict == "pass" else 0.3,
            "found": final_verdict or "unknown",
        }
        _learn_domain = _infer_domain(_infer_task_id_from_workspace(task_id))
        _learn_res = learn_from_rcb(
            mem_mgr=mem_mgr,
            hypothesis=_learn_hyp,
            validation=_learn_validation,
            persona_name=persona_name,
            run_id=task_id or ws.name,
            cross_task_store=cross_task_store,
            domain=_learn_domain,
        )
        print(f"[Step3] learn_from_rcb: {_learn_res['summary']}", flush=True)
    except Exception as _e:
        print(f"[Step3] learn_from_rcb skipped: {_e}", flush=True)


async def _run_blind_reconstruct_verify(
    ws: Path, final_verdict: str | None, stream_chat_fn: Any
) -> bool:
    """P4 Task 25: 盲重构验证 — 返回 False 表示 verdict=pass 但盲推不同意."""
    # ponytail: AutoloopEngine._blind_reconstruct_verify 依赖 hypothesis_graph /
    #   _agent_factory / memory.session 等 engine 状态, RCB 不实例化 engine.
    #   这里用 stream_chat_fn 调 LLM 独立推导作最小可用版本, 不接 SubagentDispatch.
    #   升级路径: RCB 实例化轻量 engine 后调原 _blind_reconstruct_verify.
    try:
        _br_report_path = ws / "report" / "report.md"
        _br_statement = ""
        if _br_report_path.exists():
            _br_statement = _br_report_path.read_text(encoding="utf-8")
        # 只对 pass 的 verdict 做盲重构 (fix_needed 已知有问题, 不必再验)
        if not (
            _br_statement and final_verdict == "pass" and stream_chat_fn is not None
        ):
            return True
        _br_prompt = (
            "Independently derive whether this report's conclusions hold, "
            "from first principles. Do NOT assume any prior proof. "
            'Output JSON: {"holds": true/false, "confidence": 0.0-1.0, '
            '"derivation": "..."}\n\n'
            f"Report:\n{_br_statement[:2000]}"
        )
        _br_response = await stream_chat_fn(_br_prompt, "blind_reconstruct")
        _br_holds = False
        if _br_response:
            import json as _br_json

            try:
                _br_parsed = _br_json.loads(_br_response)
                _br_holds = bool(_br_parsed.get("holds", False))
            except Exception:
                _br_holds = "true" in _br_response.lower()
        if not _br_holds:
            print(
                "[Step3] blind_reconstruct_verify FAILED "
                "(verdict was pass but blind disagrees), "
                "not recording final score",
                flush=True,
            )
            return False
        print("[Step3] blind_reconstruct_verify passed", flush=True)
        return True
    except Exception as _e:
        print(f"[Step3] blind_reconstruct_verify skipped: {_e}", flush=True)
        return True
