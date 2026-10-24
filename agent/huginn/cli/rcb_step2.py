"""RCB Step 2 执行 — 依赖 rcb_utils / rcb_cognition / rcb_audit."""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huginn.cli.rcb.audit import _rcb_drift_check
from huginn.cli.rcb.prompt_builders import _legacy_build_step2_prompt
from huginn.cli.rcb_audit import (
    _checklist_item_parser,
    _derivation_chain_audit,
    _llm_coverage_audit,
    _rcb_effort_floor,
    _report_coverage_compass,
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
from huginn.cli.rcb_fork_merge import (
    _FCM_PERSPECTIVES,
    anneal_fork_count,
    judge_fork_reports,
)
from huginn.cli.rcb_utils import (
    _MODEL_VERSION,
    _cross_task_store,
    _detect_file_rewrite_stagnation,
    _infer_domain,
    _infer_task_id_from_workspace,
    _make_simplex_id,
    _save_manifold,
)
from huginn.feature_flags import FeatureFlags
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
    # P5 挂钟预算守卫: run() 创建 goal 后传入, _step2_execute 每轮查.
    # None 表示未开 P5, 守卫总返 False (不阻断).
    wall_expired_fn: Any = None
    # persona_name: run() 从 ws.name 推断后传入, _step2_execute 写 iteration_result 用.
    persona_name: str = "default"
    # P3-7: run 开始的 ISO 时间戳, 传给 build_pmk_state 的 since 参数,
    # 限定 KB query / Memory recall 只看本 run 之后的新知识.
    run_start_iso: str | None = None
    # 执行迭代上限 (原函数内读 HUGINN_RCB_MAX_EXEC_ITERS); 供子阶段日志/度量复用.
    max_exec_iters: int = 1


def _announce_hypothesis_cycle(history: list[str]) -> None:
    """P4 Task 23: cycle_detect 顶层函数接入 — hypothesis id 序列卡顿检测."""
    # ponytail: 不扩展 VF2 cycle 检测到更长周期轨道, 保持现状 (Floyd 算法).
    # 升级路径: 接 trajectory_pattern.VF2 做语义级 cycle 检测.
    try:
        from huginn.runtime.cycle_detect import is_stuck as _cycle_is_stuck

        if _cycle_is_stuck(history, min_cycle_len=2, min_repeats=2):
            print(f"[P4] hypothesis cycle detected: {history[-6:]}", flush=True)
    except Exception:
        logger.debug("hypothesis cycle detection skipped", exc_info=True)


def _register_imagined_hypothesis(manifold: Any, new_h: Any, manifold_path: Path) -> None:
    """把想象出的假设挂到 manifold 上; 重复 h_id 视为正常跳过."""
    try:
        manifold.add(new_h)
        _save_manifold(manifold, manifold_path)
    except ValueError:
        logger.debug("best-effort op failed", exc_info=True)  # duplicate h_id, 跳过


def _record_imagination_entry(
    *,
    iter_n: int,
    transform: str,
    parent_h: Any,
    new_h: Any,
    task_id: str,
    trace_path: Path,
) -> None:
    """v15 Phase 4 Task 9.2: harmonic trace entry (升级 + 落盘 + 打印)."""
    import json as _json
    import time as _time

    entry = {
        "iteration": iter_n + 1,
        "ts": _time.time(),
        "role": "imagination",
        "attempted": f"imagine ({transform}) on {parent_h.h_id}",
        "found": f"new_h={new_h.h_id} desc={new_h.description[:80]}",
        "evidence": [
            f"transform={transform}",
            f"n_params={new_h.n_params}",
            f"predictions={list(new_h.predictions.keys())}",
        ],
        "limitations": [],
        "artifacts": [],
        "next_hint": f"test {new_h.h_id} predictions",
        "darwin_score": 0.0,
        "supported_ratio": 0.0,
        "simplex_id": _make_simplex_id(task_id, iter_n + 1, f"imagination_{transform}"),
        "cochain_type": "harmonic",
        "domain": _infer_domain(task_id),
        "task_id": task_id,
        "model_version": _MODEL_VERSION,
        "hypothesis_id": new_h.h_id,
        "log_posterior": 0.0,
        "fisher_info": 0.0,
        "imagination_parent": parent_h.h_id,
    }
    try:
        from huginn.metacog.trace_topology import upgrade_entry as _upgrade_entry

        _upgrade_entry(entry)
    except Exception:
        logger.debug("imagination trace entry upgrade skipped", exc_info=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[v15] imagination: {parent_h.h_id} -> {new_h.h_id} ({transform})", flush=True)


def _run_stagnation_imagination(
    history: list[str],
    best_h_id: str,
    manifold: Any,
    model: Any,
    *,
    iter_n: int,
    task_id: str,
    log_path: Path,
    manifold_path: Path,
    trace_path: Path,
) -> None:
    """检测 hypothesis 卡顿, 轮试三族变换生成新假设; 全链路 best-effort."""
    try:
        from huginn.metacog.imagination import detect_stagnation as _detect_stagnation
        from huginn.metacog.imagination import (
            imagine_with_checks as _imagine_with_checks,
        )

        if not _detect_stagnation(history, N=3):
            return
        print(
            f"[v15] stagnation detected (best h={best_h_id} for 3 iters), triggering imagination",
            flush=True,
        )
        parent_h = manifold._hyp.get(best_h_id)
        if parent_h is None:
            return
        # 轮试三族变换, 第一个成功就用
        for t_type in ("algebraic", "topological", "order"):
            new_h = _imagine_with_checks(
                parent_h,
                t_type,
                manifold,
                model=model,
                sigma=0.5,
                log_path=log_path,
            )
            if new_h is None:
                continue
            _register_imagined_hypothesis(manifold, new_h, manifold_path)
            _record_imagination_entry(
                iter_n=iter_n,
                transform=t_type,
                parent_h=parent_h,
                new_h=new_h,
                task_id=task_id,
                trace_path=trace_path,
            )
            break
    except Exception as _e:
        print(f"[v15] imagination skipped: {_e}", flush=True)


def _continuation_base_prompt(iter_n: int, max_iters: int) -> str:
    """iter>0 的续跑提示: 让 agent 自己找下一个 gap 并覆写 report."""
    return (
        f"Continue execution. Iteration {iter_n + 1}/{max_iters}.\n"
        f"Review the Research Trace section above for what you've already tried.\n"
        f"Identify the NEXT gap from your checklist (missing component, weak metric, "
        f"untested claim) and address it.\n"
        f"OVERWRITE report/report.md with updated results as you make progress.\n"
        f"If the report is complete and covers ALL checklist items, respond with "
        f"'TASK COMPLETE' followed by a one-paragraph summary. No tool call needed."
    )


async def _coverage_and_derivation(
    ws: Path, checklist: str, model: Any, iter_n: int, evals_history: list
) -> tuple[str, str]:
    """M2Flow 覆盖度 compass + v16 derivation chain audit (全链路 best-effort)."""
    compass = ""
    derivation_audit = ""
    try:
        compass = _report_coverage_compass(ws, checklist) or ""
        # v8: 每 5 轮做一次 LLM 语义深度审计 (规则版每轮都跑, LLM 版成本高)
        if iter_n > 0 and iter_n % 5 == 0 and model and compass:
            llm_compass = await _llm_coverage_audit(model, ws, checklist, compass)
            if llm_compass:
                compass = llm_compass
        # v16: 触发: iter>=2 每 3 轮一次, 或 drift fire 时强制触发.
        drift_fire = False
        with contextlib.suppress(Exception):
            drift_fire, _ = _rcb_drift_check(evals_history)
        scheduled = iter_n >= 2 and iter_n % 3 == 0
        if not (model and compass and (scheduled or drift_fire)):
            return compass, derivation_audit
        derivation_audit = await _derivation_chain_audit(model, ws, checklist, compass) or ""
        if not derivation_audit:
            return compass, derivation_audit
        print(
            f"[v16] derivation audit iter {iter_n}: "
            f"{'drift-fire' if drift_fire else 'scheduled'}",
            flush=True,
        )
        # 抽 derivation 百分比打印
        import re as _re_d

        dm = _re_d.search(r"DERIVATION:\s*(\d+)%", derivation_audit)
        if dm:
            print(f"[v16] derivation: {dm.group(1)}%", flush=True)
    except Exception as e:
        print(f"[v16] derivation audit skipped: {e}", flush=True)
    return compass, derivation_audit


def _kb_gap_query(last_step_eval: Any, checklist: str) -> str:
    """上一轮 attempted/gap 优先, 退化到 checklist 头部."""
    gap = ""
    if last_step_eval is not None:
        gap = (
            getattr(last_step_eval, "attempted", "")
            or getattr(last_step_eval, "gap", "")
            or ""
        )[:200]
    return gap or checklist[:200]


def _append_ucv_chunks(
    chunks: list[str], gap_query: str, domain: str, task_id: str, kg: Any
) -> bool:
    """UnifiedComplexView 双源融合 (KG + Meta-Trace); 返回是否命中."""
    try:
        from huginn.metacog.unified_complex import UnifiedComplexView

        view = UnifiedComplexView(cross_task_store=_cross_task_store, kg=kg)
        vertices = view.query(domain=domain, task_id=task_id, keyword=gap_query, top_k=2)
        for v in vertices[:2]:
            txt = getattr(v, "content", "") or ""
            if txt:
                chunks.append(txt[:400])
        if not chunks:
            return False
        print(
            f"[Step 2] UnifiedComplexView: "
            f"{len(vertices)} vertices (kg={kg is not None}, "
            f"store={_cross_task_store is not None})",
            flush=True,
        )
        return True
    except Exception as ucv_e:
        print(f"[UnifiedComplexView skipped: {ucv_e}]", flush=True)
        return False


def _retrieve_kb_chunks(
    kb: Any, last_step_eval: Any, checklist: str, *, domain: str, task_id: str, kg: Any
) -> str:
    """F5: 每轮基于上一轮 attempted 做 KB 检索, top-2 chunk 注入 prompt."""
    if kb is None:
        return ""
    try:
        gap_query = _kb_gap_query(last_step_eval, checklist)
        chunks: list[str] = []
        if not _append_ucv_chunks(chunks, gap_query, domain, task_id, kg):
            for hit in (kb.query(gap_query, top_k=2) or [])[:2]:
                txt = hit.get("content", "") if isinstance(hit, dict) else str(hit)
                if txt:
                    chunks.append(txt[:400])
        if not chunks:
            return ""
        return "\n\n## Domain Knowledge (KB retrieval, top-2)\n" + "\n---\n".join(chunks)
    except Exception:
        logger.debug("kb gap retrieval skipped", exc_info=True)
        return ""


def _posterior_guided_hint(
    manifold: Any, observations: list, history: list, mcmc_engine: Any
) -> str:
    """v15 Phase 2 Task 4: 构造 posterior-guided hint (核心 + 探索)."""
    if manifold is None or not observations:
        return ""
    try:
        from huginn.agent.hint_coordinator import (
            _build_posterior_guided_hint as _build_pg_hint,
        )

        return _build_pg_hint(
            manifold=manifold,
            observations=observations,
            history_entries=history,
            mcmc_current=(
                mcmc_engine._mcmc_current if mcmc_engine is not None else None
            ),
        )
    except Exception as pe:
        print(f"[v15] posterior hint build skipped: {pe}", flush=True)
        return ""


async def _build_continuation_prompt(
    ctx: _RCBStep2Ctx,
    *,
    iter_n: int,
    max_iters: int,
    evals_history: list,
    last_step_eval: Any,
    trace_domain: str,
    trace_task_id: str,
    hint_coord_enabled: bool,
    hint_coord: Any,
    drift_info: Any,
    manifold: Any,
    iter_observations: list,
    trace_history: list,
    mcmc_engine: Any,
    step2_prompt_base: str,
    cross_task_prior: Any,
) -> tuple[str, str]:
    """构造 iter>0 的完整 prompt; 返回 (prompt, derivation_audit)."""
    iter_base = _continuation_base_prompt(iter_n, max_iters)
    compass, derivation_audit = await _coverage_and_derivation(
        ctx.ws, ctx.checklist, ctx.model, iter_n, evals_history
    )
    # F4: FCM winner_plan 每轮提醒, 防 compaction 后漂移.
    fcm_winner = (ctx.fcm.get("winner_plan") or "").strip() if ctx.fcm else ""
    fcm_reminder = (
        "\n\n## Selected Execution Plan (reminder — Step 1.7 FCM winner)\n"
        + fcm_winner[:1200]
    ) if fcm_winner else ""
    kb_text = _retrieve_kb_chunks(
        ctx.kb,
        last_step_eval,
        ctx.checklist,
        domain=trace_domain,
        task_id=trace_task_id,
        kg=ctx.kg,
    )
    # v14 Task 6: HintCoordinator 接 gradient/curl/harmonic, retrieval 族直接拼后面.
    if not hint_coord_enabled:
        prompt = iter_base
        if compass:
            prompt += "\n\n" + compass
        if fcm_reminder:
            prompt += fcm_reminder
        if kb_text:
            prompt += kb_text
        return prompt, derivation_audit
    drift_text = None
    if isinstance(drift_info, tuple) and drift_info and drift_info[0]:
        drift_text = str(drift_info[1]) if len(drift_info) > 1 else None
    posterior_hint = _posterior_guided_hint(
        manifold, iter_observations, trace_history, mcmc_engine
    )
    prompt, _hint_trace_events = hint_coord.coordinate(
        iter_n=iter_n,
        csm_state="S4_CONSTRUCT",
        beta=(1, 0),
        last_verdict=None,
        fcm_winner=fcm_winner or None,
        scan_text=None,
        step2_prompt=step2_prompt_base,
        iter_prompt=iter_base,
        compass=compass or None,
        step_eval=None,
        drift_info=drift_text,
        imagination=None,
        meta_agent=None,
        cross_task_prior=cross_task_prior,
        manifold=manifold,
        posterior_hint=posterior_hint or None,
    )
    if kb_text:
        prompt += kb_text
    return prompt, derivation_audit


def _imagery_spec(
    manifold: Any, best_h_id: str | None, checklist: str, iter_prompt: str
) -> str:
    """spec 从上一轮 best hypothesis 提取, 没有就用 checklist 兜底."""
    spec = ""
    if manifold is not None and best_h_id is not None:
        prev_h = manifold._hyp.get(best_h_id)
        if prev_h is not None:
            spec = getattr(prev_h, "statement", "") or ""
    return spec or (checklist or iter_prompt or "")[:200]


def _sketch_to_kb(
    iter_n: int,
    manifold: Any,
    best_h_id: str | None,
    checklist: str,
    iter_prompt: str,
    kb: Any,
) -> None:
    """P3 Task 15: mental_imagery sketch→verify 闭环, 草图入 RAG KB (失败不阻塞)."""
    try:
        from huginn.metacog import mental_imagery as _mi

        spec = _imagery_spec(manifold, best_h_id, checklist, iter_prompt)
        img_bytes = _mi.sketch(spec)
        if not img_bytes:
            return
        verify_res = _mi.verify(img_bytes, {"kind": "unknown"})
        # 草图作为 visual primitive 注入 RAG KB (G4 auto-ingest)
        if kb is not None:
            with contextlib.suppress(Exception):
                kb.add_text(
                    f"[mental_imagery sketch] spec={spec[:80]} "
                    f"verified={verify_res.get('verified', False)} "
                    f"n_regions={verify_res.get('n_regions_detected', 0)}",
                    filename=f"mental_imagery_iter{iter_n}.txt",
                    metadata={
                        "source": "mental_imagery",
                        "iter": iter_n,
                        "verified": str(verify_res.get("verified", False)),
                    },
                )
        print(
            f"[Step 2] mental_imagery: verified="
            f"{verify_res.get('verified', False)} "
            f"n_regions={verify_res.get('n_regions_detected', 0)} "
            f"at iter {iter_n}",
            flush=True,
        )
    except Exception as mie:
        print(f"[mental_imagery skipped: {mie}]", flush=True)


def _maybe_inject_imagery(
    heat_engine: Any,
    iter_n: int,
    manifold: Any,
    best_h_id: str | None,
    checklist: str,
    iter_prompt: str,
    kb: Any,
    imagination_block: str,
) -> str:
    """P1 想象力机制 — should_imaginate 为真时注入 imagination block + 草图闭环."""
    try:
        if not heat_engine.should_imaginate(iter_n):
            return iter_prompt
    except Exception as ie:
        print(f"[should_imaginate skipped: {ie}]", flush=True)
        return iter_prompt
    print(f"[Step 2] imagination triggered at iter {iter_n}", flush=True)
    prompt = iter_prompt + imagination_block
    _sketch_to_kb(iter_n, manifold, best_h_id, checklist, prompt, kb)
    return prompt


def _snapshot_fork_report(ws: Path, persp: str) -> str:
    """把 fork 写出的主 report.md 快照成 report_fork_X.md, 返回正文."""
    main_rp = ws / "report" / "report.md"
    if not main_rp.exists():
        return ""
    try:
        text = main_rp.read_text(encoding="utf-8")
        (ws / "report" / f"report_fork_{persp}.md").write_text(text, encoding="utf-8")
        return text
    except Exception as e:
        print(f"[tfm snapshot {persp} skipped: {e}]", flush=True)
        return ""


def _snapshot_fork_outputs(ws: Path, persp: str) -> None:
    """fork 的计算产物也快照 — 复现门禁的对照集 (顺序 fork 共享 outputs/)."""
    out_src = ws / "outputs"
    if not out_src.is_dir():
        return
    try:
        shutil.copytree(out_src, ws / "report" / f"outputs_fork_{persp}", dirs_exist_ok=True)
    except Exception as e:
        print(f"[tfm outputs snap {persp} skipped: {e}]", flush=True)


async def _run_one_fork(
    ctx: _RCBStep2Ctx,
    tree: Any,
    branch_point: str,
    persp: str,
    bias: str,
    iter_prompt: str,
    iter_n: int,
    thread_id: str,
) -> tuple[str, str]:
    """单条 fork: 换 leaf → stream → 快照 report/outputs; 返回 (leaf, report)."""
    tree.set_active_leaf(branch_point)
    fork_tid = f"{thread_id}_f{iter_n}_{persp}"
    # ponytail: 不让 fork 改写 report_fork_X.md — system prompt 里 "写 report.md"
    # 的先验太强, 顺着先验让它写, runner 结束后自己快照供 verifier 评审.
    fork_prompt = f"[Trajectory fork — '{persp}' bias] {bias}\n\n{iter_prompt}"
    await ctx.stream_chat_fn(fork_prompt, f"fork_{persp}", tid=fork_tid)
    leaf = tree.active_leaf_id or branch_point
    report = _snapshot_fork_report(ctx.ws, persp)
    _snapshot_fork_outputs(ctx.ws, persp)
    return leaf, report


def _apply_fork_verdict(
    ctx: _RCBStep2Ctx,
    tree: Any,
    branch_point: str,
    verdict: dict,
    fork_reports: dict[str, str],
    fork_leaves: dict[str, str],
    iter_n: int,
    thread_id: str,
    backup_path: Path,
    backup_text: str | None,
) -> tuple[str, str, str]:
    """winner 入主路径 / 无 winner 恢复备份; 返回 (ai_text, merge_hint, thread_id)."""
    winner = verdict["winner"]
    print(f"[tfm: winner={winner} scores={verdict['scores']}]", flush=True)
    if verdict.get("gate"):
        print(f"[tfm gate: {verdict['gate']}]\n", flush=True)
    if winner and winner in fork_leaves:
        # 合并: winner 报告入主路径, 轨迹沿 winner 分支继续
        report_dir = ctx.ws / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.md").write_text(fork_reports[winner], encoding="utf-8")
        tree.set_active_leaf(fork_leaves[winner])
        merge_hint = ""
        if verdict["merge_notes"]:
            notes = "\n".join(f"- {x}" for x in verdict["merge_notes"])
            merge_hint = (
                "\n\n## Merge notes from rejected forks (trajectory fork-merge)\n"
                f"{notes}\nSalvage these into report/report.md where applicable."
            )
        return (
            f"[tfm: merged fork '{winner}' into main trajectory]",
            merge_hint,
            f"{thread_id}_f{iter_n}_{winner}",
        )
    tree.set_active_leaf(branch_point)
    # winner=None 时 fork 已破坏主 report.md, 从备份恢复.
    if backup_text is not None:
        try:
            backup_path.write_text(backup_text, encoding="utf-8")
        except Exception as e:
            print(f"[tfm restore main report failed: {e}]", flush=True)
    return "[tfm: all forks produced empty reports, main trajectory unchanged]", "", thread_id


def _record_fork_trace(
    *,
    iter_n: int,
    k: int,
    t_hot: float,
    winner: str | None,
    verdict: dict,
    fork_reports: dict[str, str],
    task_id: str,
    trace_path: Path,
) -> None:
    """Meta-Trace: 分叉轮也留痕 (role=trajectory_fork_merge)."""
    import json as _json
    import time as _time

    try:
        entry = {
            "iteration": iter_n + 1,
            "ts": _time.time(),
            "role": "trajectory_fork_merge",
            "attempted": f"{k}-fork trajectory exploration (T_hot={t_hot:.2f})",
            "found": f"winner={winner} scores={verdict.get('scores')} gate={verdict.get('gate')}",
            "evidence": [fork_reports.get(winner, "")[:150]] if winner else [],
            "limitations": ["sequential forks share workspace; outputs snapshotted per fork"],
            "artifacts": [f"report/report_fork_{p}.md" for p in fork_reports],
            "next_hint": "salvage merge_notes" if verdict.get("merge_notes")
                         else "continue winner trajectory",
            "darwin_score": 0.0,
            "supported_ratio": 0.0,
            # v14 Task 1: trajectory_fork_merge 跟 FCM winner 同族 (gradient).
            "simplex_id": _make_simplex_id(task_id, iter_n + 1, "trajectory_fork_merge"),
            "cochain_type": "gradient",
            "domain": _infer_domain(task_id),
            "task_id": task_id,
            "model_version": _MODEL_VERSION,
        }
        # v15: upgrade_entry 补默认字段, 保证 schema 一致.
        try:
            from huginn.metacog.trace_topology import upgrade_entry as _upgrade_entry

            _upgrade_entry(entry)
        except Exception:
            logger.debug("tfm trace entry upgrade skipped", exc_info=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[tfm trace skipped: {e}]", flush=True)


async def _run_fork_iteration(
    ctx: _RCBStep2Ctx,
    *,
    iter_n: int,
    max_iters: int,
    k: int,
    t_hot: float,
    iter_prompt: str,
    thread_id: str,
    branch_point: str,
    trace_task_id: str,
    trace_path: Path,
) -> tuple[str, str, str]:
    """TFM 分叉执行 + judge + 合并; 返回 (ai_text, merge_hint, thread_id)."""
    print(
        f"\n--- Step 2 iter {iter_n + 1}/{max_iters} "
        f"[TFM: T_hot={t_hot:.2f} → {k} forks] ---\n",
        flush=True,
    )
    # 顺序 fork 共享 workspace, 后序会覆盖主 report.md → 先备份, winner=None 时恢复.
    backup_path = ctx.ws / "report" / "report.md"
    backup_text = backup_path.read_text(encoding="utf-8") if backup_path.exists() else None

    tree = ctx.agent._conversation_tree
    fork_reports: dict[str, str] = {}
    fork_leaves: dict[str, str] = {}
    for persp, bias in _FCM_PERSPECTIVES[:k]:
        leaf, report = await _run_one_fork(
            ctx, tree, branch_point, persp, bias, iter_prompt, iter_n, thread_id
        )
        fork_leaves[persp] = leaf
        fork_reports[persp] = report

    verdict = await judge_fork_reports(
        fork_reports,
        ctx.checklist,
        ctx.model,
        artifact_dirs={p: ctx.ws / "report" / f"outputs_fork_{p}" for p in fork_reports},
    )
    ai_text, merge_hint, thread_id = _apply_fork_verdict(
        ctx,
        tree,
        branch_point,
        verdict,
        fork_reports,
        fork_leaves,
        iter_n,
        thread_id,
        backup_path,
        backup_text,
    )
    _record_fork_trace(
        iter_n=iter_n,
        k=k,
        t_hot=t_hot,
        winner=verdict["winner"],
        verdict=verdict,
        fork_reports=fork_reports,
        task_id=trace_task_id,
        trace_path=trace_path,
    )
    return ai_text, merge_hint, thread_id


def _effort_reject_prompt(iter_n: int, max_iters: int, reason: str) -> str:
    """TASK COMPLETE 被驳回时覆盖下一轮 prompt, agent 必须先补缺再声称完成."""
    return (
        f"Continue execution. Iteration {iter_n + 2}/{max_iters}.\n"
        f"Previous TASK COMPLETE was rejected by effort floor: "
        f"{reason}. Address the MISSING items and re-claim "
        f"TASK COMPLETE only when report.md covers them.\n\n"
        f"Review the Research Trace section and Coverage Compass above."
    )


def _metacog_reject_prompt(iter_n: int, max_iters: int) -> str:
    """被反完成审计阻断时覆盖下一轮 prompt."""
    return (
        f"Continue execution. Iteration {iter_n + 2}/{max_iters}.\n"
        f"Previous TASK COMPLETE blocked by metacog audit "
        f"(completion/topology). Address the gaps and re-claim "
        f"TASK COMPLETE when done.\n\n"
        f"Review the Research Trace section and Coverage Compass above."
    )


def _metacog_block(
    ws: Path, report_text: str, report_path: Path, iter_observations: list | None
) -> tuple[bool, Any]:
    """Task 5+10 反完成审计: 4 层完成度 + 拓扑坍缩 + 选择偏差; 返回 (blocked, completion)."""
    completion = None
    try:
        from huginn.autoloop.cognitive_loop import (
            metacog_check_completion,
            metacog_check_selection_bias,
            metacog_check_topology_collapse,
        )

        # _report_text 在 iter 头部算, 可能空; 兜底重读文件.
        rep_md = report_text or (
            report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        )
        completion = metacog_check_completion(report_md=rep_md, outputs_dir=ws / "outputs")
        blocked = not completion.get("passed", True)
        if blocked:
            print(
                f"[TaskComplete] blocked by metacog_check: "
                f"{completion.get('block_reasons', [])}",
                flush=True,
            )
        # 拓扑层反完成审计: hypothesis space 多样性检测.
        topo = metacog_check_topology_collapse(hypothesis_graph=None, hypothesis_list=None)
        if topo.get("collapsed", False):
            print(f"[TaskComplete] blocked by topology collapse: {topo.get('reason')}", flush=True)
            blocked = True
        # 选择偏差反完成审计: 样本系统性缺一类 (幸存者偏差) 时阻断.
        selection_bias = metacog_check_selection_bias(observations=iter_observations)
        if selection_bias.get("biased", False):
            print(f"[TaskComplete] blocked by selection bias: {selection_bias.get('reason')}", flush=True)
            blocked = True
        return blocked, completion
    except Exception as e:
        print(f"[TaskComplete] metacog audit failed: {e}", flush=True)
        return False, completion


async def _handle_task_complete(
    ctx: _RCBStep2Ctx,
    *,
    iter_n: int,
    max_iters: int,
    derivation_audit: str,
    report_text: str,
    report_path: Path,
    iter_observations: list | None,
    entry: Any,
    pmk_state: Any,
    manifold: Any,
    heat_engine: Any,
    bandit: Any,
    completion_audit: Any,
    mcmc_current_info: Any,
    anomaly_info: Any,
    rejections: int,
    max_rejections: int,
) -> tuple[int, bool, str | None]:
    """TASK COMPLETE 收尾: effort floor + 反完成审计.

    返回 (新的连续驳回数, 是否接受, 下一轮 prompt 覆盖).
    """
    # 连续驳回计数: agent 反复 'TASK COMPLETE' 不调工具时, 第 N+1 次接受.
    rejections += 1
    force_accept = rejections > max_rejections
    if force_accept:
        print(
            f"[effort floor] 接受 TASK COMPLETE: 连续驳回 "
            f"{rejections - 1} 次达上限 "
            f"{max_rejections}. agent 已尽力, 强制收尾.",
            flush=True,
        )
    final_derivation = derivation_audit
    if not force_accept and ctx.model and not final_derivation:
        try:
            final_derivation = await _derivation_chain_audit(
                ctx.model,
                ctx.ws,
                ctx.checklist,
                _report_coverage_compass(ctx.ws, ctx.checklist) or "",
            ) or ""
        except Exception as e:
            print(f"[v16] final derivation audit skipped: {e}", flush=True)
    if force_accept:
        eff_ok, eff_reason = True, "force_accept skipped"
    else:
        eff_ok, eff_reason = _rcb_effort_floor(
            ctx.ws, ctx.checklist, derivation_audit=final_derivation or None
        )
    if not eff_ok:
        print(
            f"[effort floor] TASK COMPLETE 被驳回: {eff_reason}. 继续迭代补缺.",
            flush=True,
        )
        return rejections, False, _effort_reject_prompt(iter_n, max_iters, eff_reason)

    if force_accept:
        blocked, completion = False, completion_audit
    else:
        blocked, completion = _metacog_block(
            ctx.ws, report_text, report_path, iter_observations
        )
    if blocked:
        return rejections, False, _metacog_reject_prompt(iter_n, max_iters)

    print("[agent signalled TASK COMPLETE, breaking]", flush=True)
    # P0-C: TASK COMPLETE 时写 final cognitive evidence snapshot.
    _write_cognitive_evidence(
        ctx.ws,
        iter_n,
        entry=entry,
        pmk_state=pmk_state,
        hypo_manifold=manifold,
        heat_engine=heat_engine,
        bandit_controller=bandit,
        completion_audit=completion,
        mcmc_info=mcmc_current_info,
        anomaly_info=anomaly_info,
        is_final=True,
    )
    return rejections, True, None


@dataclass
class _StepEvalOutcome:
    """StepEvaluator 阶段的可变产物 (drift / task_metrics 跨轮滚动)."""

    step_eval: Any = None
    drift_info: Any = None
    task_metrics: Any = None
    merge_hint: str = ""


def _resolve_audit_path() -> Any:
    try:
        from huginn.events.audit_log import _resolve_audit_path as _rap

        return _rap()
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return None


def _update_self_model_from_audit(self_model: Any, audit_path: Any, iter_n: int) -> None:
    """v15 Phase 5 Task 10.2: 从 audit_log 抓工具调用记录, 更新 SelfModel."""
    if self_model is None or audit_path is None:
        return
    try:
        from huginn.metacog.self_model import (
            extract_step_result_from_audit as _extract_step,
        )

        step_records = _extract_step(audit_path, step_id=iter_n + 1)
        if step_records:
            self_model.update_from_step(step_records)
    except Exception as se:
        print(f"[v15] self_model update skipped: {se}", flush=True)


def _remember_prospective(mem_mgr: Any, step_eval: Any, iter_n: int) -> None:
    """AV6: on_track=false/unsure 时记一条 intention, 下一轮 recall_prospective 触发."""
    if mem_mgr is None or step_eval.on_track not in ("false", "unsure"):
        return
    try:
        from huginn.memory.prospective import _new_intention_id

        mem_mgr.remember_prospective({
            "intention_id": _new_intention_id(),
            "description": (
                f"上一步脱轨需复核: attempted={step_eval.attempted[:80]}"
                f"; deviation={step_eval.deviation[:80]}"
            ),
            "trigger_type": "dependency",
            "trigger_payload": {"depends_on_step": iter_n},
            "priority": 5,
            "created_at": _now(),
            "source_step": iter_n,
        })
    except Exception as pe:
        print(f"[prospective store skipped: {pe}]", flush=True)


def _now() -> float:
    import time as _time

    return _time.time()


def _update_heat_after_step(
    heat_engine: Any,
    step_eval: Any,
    iter_n: int,
    step2_prompt: str,
    iter_prompt: str,
    target_chains: list,
) -> None:
    """P1+AV8+AV4: heat_engine 闭环 — 调 cognitive_loop 共享函数."""
    if heat_engine is None:
        return
    from huginn.autoloop.cognitive_loop import update_heat_engine_after_step

    idea_count = sum(
        len(getattr(tc, "completed_results", set()) or set()) for tc in target_chains
    )
    sp_len = len(step2_prompt) if iter_n == 0 else len(iter_prompt)
    update_heat_engine_after_step(heat_engine, step_eval, sp_len, idea_count)


def _update_drift_and_metrics(
    evals_history: list,
    step_eval: Any,
    task_metrics: Any,
    task_state: Any,
    ws: Path,
    task_id: str,
    max_iters: int,
    target_chains: list,
) -> tuple[Any, Any]:
    """G62+G70: detect_drift + TaskMetrics 滚动更新; 返回 (drift_info, task_metrics)."""
    try:
        from huginn.autoloop.cognitive_loop import update_drift_and_metrics

        # target_chain_progress 取所有 chain 的平均 progress — 整体任务完成度.
        tc_prog = (
            sum(getattr(tc, "progress", 0.0) for tc in target_chains) / len(target_chains)
        ) if target_chains else None
        drift_info, task_metrics = update_drift_and_metrics(
            evals_history=evals_history,
            step_eval=step_eval,
            task_metrics=task_metrics,
            task_state=task_state,
            workspace=ws,
            run_id=task_id,
            max_iterations=max_iters,
            target_chain_progress=tc_prog,
        )
        if drift_info and drift_info[0]:
            print(f"[Step 2] drift detected: {drift_info[1]}", flush=True)
        return drift_info, task_metrics
    except Exception as de:
        print(f"[Step 2] drift/metrics update skipped: {de}", flush=True)
        return None, task_metrics


def _append_step_eval_trace(
    iter_n: int, step_eval: Any, task_id: str, trace_path: Path
) -> None:
    """step_evaluation 的 meta-trace 留痕 (cochain_type=curl)."""
    import json as _json

    entry = {
        "iteration": iter_n + 1,
        "ts": _now(),
        "role": "step_evaluation",
        "attempted": f"evaluate step {iter_n + 1}",
        "found": f"on_track={step_eval.on_track} evidence={step_eval.evidence_quality}",
        "evidence": [],
        "limitations": [],
        "artifacts": [],
        "next_hint": "",
        "darwin_score": 0.0,
        "supported_ratio": 0.0,
        # v14 Task 1: step_eval = curl (critique-driven).
        "simplex_id": _make_simplex_id(task_id, iter_n + 1, "step_evaluation"),
        "cochain_type": "curl",
        "domain": _infer_domain(task_id),
        "task_id": task_id,
        "model_version": _MODEL_VERSION,
    }
    # v15: upgrade_entry 补默认字段 (step_eval entry 不填 abduction 结果)
    try:
        from huginn.metacog.trace_topology import upgrade_entry as _upgrade_entry

        _upgrade_entry(entry)
    except Exception:
        logger.debug("step_eval trace entry upgrade skipped", exc_info=True)
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(_json.dumps(entry, ensure_ascii=False) + "\n")


def _should_continue_hint(
    evals_history: list, step_eval: Any, merge_hint: str
) -> str:
    """should_continue=False 时把重定向 + Reflector 建议拼进下一轮 merge_hint."""
    from huginn.metacog.step_evaluator import should_continue

    cont, msg = should_continue(evals_history)
    if cont:
        return merge_hint
    print(f"[Step 2] should_continue=False: {msg}", flush=True)
    # 不 break, 重定向提示拼到 merge_hint, 下一轮 prompt 注入
    merge_hint = (merge_hint or "") + f"\n\n[REDIRECT] {msg}\n"
    # 25.4: 工具异常时让 Reflector 介入, 文本拼到 merge_hint.
    if "Reflector" not in msg and "工具调用异常" not in msg:
        return merge_hint
    try:
        from huginn.metacog.reflector import format_reflector_text, reflect

        actions = reflect(
            tool_call_health=getattr(step_eval, "tool_call_health", None),
            last_step_evaluations=evals_history,
        )
        refl_text = format_reflector_text(actions)
        if refl_text:
            merge_hint = merge_hint + f"\n\n{refl_text}\n"
    except Exception as re:
        print(f"[Step 2] reflector warning: {re}", flush=True)
    return merge_hint


async def _evaluate_step_phase(
    ctx: _RCBStep2Ctx,
    *,
    iter_n: int,
    entry: Any,
    target_chains: list,
    self_model: Any,
    heat_engine: Any,
    evals_history: list,
    task_metrics: Any,
    task_state_for_metrics: Any,
    step2_prompt: str,
    iter_prompt: str,
    merge_hint: str,
    trace_task_id: str,
    trace_path: Path,
) -> _StepEvalOutcome:
    """StepEvaluator 评估 + SelfModel/Prospective/heat/drift 闭环 + trace 留痕."""
    outcome = _StepEvalOutcome(task_metrics=task_metrics, merge_hint=merge_hint)
    try:
        from huginn.metacog.step_evaluator import evaluate_step
        from huginn.metacog.target_chain import update_progress

        # scan_text 是 Step 1.5 的纯文本输出, 不是 list[dict] → 传 None 走 LLM 兜底.
        audit_path = _resolve_audit_path()
        # ponytail: prev_step_id 用上一轮的 iteration; 首轮 None, 不写 dep edge.
        prev_sid = iter_n if iter_n > 0 else None
        step_eval = evaluate_step(
            meta_trace_entry=entry,
            target_chains=target_chains,
            verification_signals=None,
            memory=ctx.mem_mgr,
            kb=ctx.kb,
            persona=ctx.persona,
            model=ctx.model,
            tool_call_health=None,
            kg=ctx.kg,
            prev_step_id=prev_sid,
            audit_log_path=audit_path,
        )
        outcome.step_eval = step_eval
        evals_history.append(step_eval)
        for tc in target_chains:
            update_progress(tc, step_eval.found)
        _update_self_model_from_audit(self_model, audit_path, iter_n)
        _remember_prospective(ctx.mem_mgr, step_eval, iter_n)
        _update_heat_after_step(
            heat_engine, step_eval, iter_n, step2_prompt, iter_prompt, target_chains
        )
        outcome.drift_info, outcome.task_metrics = _update_drift_and_metrics(
            evals_history,
            step_eval,
            outcome.task_metrics,
            task_state_for_metrics,
            ctx.ws,
            ctx.task_id,
            ctx.max_exec_iters,
            target_chains,
        )
        _append_step_eval_trace(iter_n, step_eval, trace_task_id, trace_path)
        outcome.merge_hint = _should_continue_hint(evals_history, step_eval, outcome.merge_hint)
    except Exception as e:
        print(f"[Step 2] step evaluator warning: {e}", flush=True)
    return outcome


def _entry_get(entry: dict | None, key: str, default: Any) -> Any:
    """从 meta_trace entry 取字段; entry 未生成 (None) 时回落 default."""
    if not isinstance(entry, dict):
        return default
    return entry.get(key, default)


def _inject_context_blocks(
    iter_prompt: str,
    *,
    agent: Any,
    mem_mgr: Any,
    target_chains: list,
    iter_n: int,
    last_step_eval: Any,
    kg: Any,
    drift_info: Any,
    persona: Any,
    kb: Any,
) -> tuple[str, list]:
    """Task 18 / G66: 注入 prospective / target_chain / step_eval / 元 Agent 文本."""
    fired: list = []
    try:
        ctx_b = getattr(agent, "_ctx_builder", None)
        if mem_mgr is not None:
            try:
                fired = mem_mgr.recall_prospective({"current_step": iter_n})
            except Exception as e:
                print(f"[prospective recall skipped: {e}]", flush=True)
        if ctx_b is not None:
            tc_text = ctx_b.build_target_chain_text(target_chains, iter_n) or ""
            pro_text = ctx_b.build_prospective_text(fired) or ""
            se_text = ctx_b.build_step_eval_text(last_step_eval) or ""
            eh_text = ctx_b.build_episode_history_text(kg, iter_n) if kg else ""
            ma_text = ""
            try:
                ma_text = ctx_b.build_meta_agent_text(
                    target_chains=target_chains,
                    last_step_evaluation=last_step_eval,
                    tool_call_health=getattr(last_step_eval, "tool_call_health", None),
                    drift_info=drift_info,
                ) or ""
            except Exception as e:
                print(f"[meta_agent_text skipped: {e}]", flush=True)
            pmk_text = ""
            try:
                pmk_text = ctx_b.build_pmk_text(
                    persona=persona,
                    memory=mem_mgr,
                    kb=kb,
                    last_step_evaluation=last_step_eval,
                ) or ""
            except Exception as e:
                print(f"[pmk_text skipped: {e}]", flush=True)
            inject = tc_text + pro_text + se_text + eh_text + ma_text + pmk_text
            if inject:
                iter_prompt += "\n\n" + inject
    except Exception as e:
        print(f"[ctx inject skipped: {e}]", flush=True)
    return iter_prompt, fired


def _inject_curiosity_hint(iter_prompt: str, mem_mgr: Any) -> str:
    """跨任务 curiosity hint — 注入历史弱 persona 簇."""
    if not (FeatureFlags.shared().is_enabled("curiosity_hint") and mem_mgr is not None):
        return iter_prompt
    try:
        sm = mem_mgr.longterm.get_self_model()
        if sm:
            weak = [
                f"- {v.get('dimension', '?')}/{v.get('hyp_type', '?')}: "
                f"rate={v.get('rate', 0):.2f} (n={v.get('success', 0) + v.get('failure', 0)})"
                for v in sm.values()
                if isinstance(v.get("rate"), int | float)
                and v["rate"] < 0.4
                and v.get("success", 0) + v.get("failure", 0) >= 3
            ]
            if weak:
                iter_prompt += (
                    "\n\n[CURIOSITY] 历史预测不准的簇 (主动探索方向):\n"
                    + "\n".join(weak[:5]) + "\n"
                )
                print(f"[curiosity] injected {len(weak)} weak personas", flush=True)
    except Exception as e:
        print(f"[curiosity skipped: {e}]", flush=True)
    return iter_prompt


def _inject_deliverable_hint(iter_prompt: str, ws: Path, iter_n: int) -> str:
    """Deliverable 物理检查 — checklist 管内容覆盖, 这里管文件存在性."""
    if iter_n <= 0:
        return iter_prompt
    try:
        from huginn.bench.orchestrator import RCB_DELIVERABLES, _triage_prompt

        missing = RCB_DELIVERABLES.missing(ws)
        if missing:
            iter_prompt += "\n\n" + _triage_prompt(missing)
            print(f"[deliverable] missing: {missing}", flush=True)
    except Exception as e:
        print(f"[deliverable check skipped: {e}]", flush=True)
    return iter_prompt


def _collect_iter_observations_and_abduction(
    *,
    manifold: Any,
    ai_text: str,
    ws: Path,
    checklist: str,
    obs_path: Path,
    manifold_path: Path,
    trace_path: Path,
    trace_task_id: str,
    iter_n: int,
) -> tuple[list, str | None, float, float, dict | None]:
    """v15 Phase 2: 收集 observations + abduction; 失败降级空 observations."""
    observations: list = []
    best_h_id: str | None = None
    log_post = 0.0
    fisher_info = 0.0
    abduction: dict | None = None
    try:
        report_text = ""
        report_path = ws / "report" / "report.md"
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
        observations = _collect_observations(
            step_result=ai_text or "",
            report_text=report_text,
            checklist=checklist or "",
        )
        if observations:
            best_h_id, log_post, fisher_info = _compute_v15_fields(manifold, observations)
            _append_observations_log(observations, obs_path, iteration=iter_n + 1)
            _record_abduction(
                manifold,
                observations,
                trace_path=trace_path,
                task_id=trace_task_id,
                iteration=iter_n + 1,
                ts=time.time(),
            )
            # 持久化 manifold (predictions 可能被外部更新, 此处保状态)
            _save_manifold(manifold, manifold_path)
            abduction = {
                "best_h_id": best_h_id,
                "log_posterior": log_post,
                "fisher_info": fisher_info,
                "n_observations": len(observations),
                "iteration": iter_n + 1,
            }
    except Exception as e:
        print(f"[v15] observation/abduction skipped: {e}", flush=True)
    return observations, best_h_id, log_post, fisher_info, abduction


def _run_blind_spot_imagination(
    self_model: Any,
    manifold: Any,
    model: Any,
    *,
    iter_n: int,
    log_path: Path,
    manifold_path: Path,
) -> None:
    """v15 Phase 5: blind_spot 触发 imagination (每 3 轮一次, 失败不阻塞)."""
    if not (
        self_model is not None and manifold is not None and iter_n > 0 and iter_n % 3 == 0
    ):
        return
    try:
        from huginn.metacog.blind_spot_mapper import (
            infer_blind_spots,
            pick_imagination_seed,
        )
        from huginn.metacog.imagination import imagine_from_blind_spot

        bs_list = infer_blind_spots(self_model)
        seed = pick_imagination_seed(bs_list)
        if seed is None:
            return
        new_h = imagine_from_blind_spot(
            seed, manifold, model=model, log_path=log_path, sigma=0.5,
        )
        if new_h is None:
            return
        try:
            manifold.add(new_h)
            _save_manifold(manifold, manifold_path)
        except ValueError:
            logger.debug("best-effort op failed", exc_info=True)  # duplicate h_id, 跳过
        # imagination 成功 → uncertain 升 capable (乐观反馈, 新 h 进 manifold 即算 success).
        with contextlib.suppress(Exception):
            self_model.feedback_from_imagination(seed.skill, success=True)
        print(
            f"[v15] blind_spot imagination: {seed.skill} -> {new_h.h_id}",
            flush=True,
        )
    except Exception as e:
        print(f"[v15] blind_spot imagination skipped: {e}", flush=True)


def _write_meta_trace_entry(
    *,
    ws: Path,
    ai_text: str | None,
    iter_prompt: str,
    iter_n: int,
    max_iters: int,
    trace_history: list,
    last_step_eval: Any,
    best_h_id: str | None,
    log_post: float,
    fisher_info: float,
    trace_task_id: str,
    shard_writer: Any,
    shard: Any,
    betti_path: Path,
    compute_betti: Any,
) -> tuple[str, Path, dict | None]:
    """写本轮 Meta-Trace entry + betti; 失败降级为空 report / None entry."""
    report_text = ""
    report_path = ws / "report" / "report.md"
    entry: dict | None = None
    try:
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
        # v14 Task 3: supported_ratio = 命中数(overlap>0.7) / max(历史总数, 1).
        # 控制消息 ([tfm:...] / TASK COMPLETE) 与 evidence TF-IDF 必然失配, fallback
        # 到 report.md 前 200 字符, 让 supported_ratio 反映 report.md 自我一致性.
        ai_text_stripped = (ai_text or "").strip()
        is_control_msg = (
            not ai_text_stripped
            or ai_text_stripped.startswith("[tfm:")
            or ai_text_stripped.upper().startswith("TASK COMPLETE")
            or len(ai_text_stripped) < 50
        )
        if is_control_msg and report_text:
            attempted_text = report_text[:200].replace("\n", " ")
        else:
            attempted_text = (
                ai_text[:200] if ai_text else iter_prompt[:200]
            ).replace("\n", " ")
        supported_hits = 0
        if trace_history:
            try:
                from huginn.context_builder import _compute_semantic_overlap

                for hist in trace_history:
                    hev = hist.get("evidence") or []
                    hev_text = " ".join(hev) if isinstance(hev, list) else str(hev)
                    if _compute_semantic_overlap(attempted_text, hev_text) > 0.7:
                        supported_hits += 1
            except Exception as e:
                print(f"[supported_ratio skipped: {e}]", flush=True)
        supported_ratio = supported_hits / max(len(trace_history), 1)
        entry = {
            "iteration": iter_n + 1,
            "ts": time.time(),
            "role": "rcb_exec",
            "attempted": attempted_text,
            "found": (ai_text or "")[:300],
            "evidence": [report_text[:150]] if report_text else [],
            "limitations": [],
            "artifacts": ["report/report.md"] if report_path.exists() else [],
            "next_hint": (
                "continue execution" if iter_n < max_iters - 1 else "step3 critique"
            ),
            # v14 Task 2: darwin = 1 - gap_severity, 首轮 _last_step_eval=None → 0.5.
            "darwin_score": _compute_darwin_score(last_step_eval),
            "supported_ratio": supported_ratio,
            "simplex_id": _make_simplex_id(trace_task_id, iter_n + 1, "rcb_exec"),
            "cochain_type": "gradient",
            "domain": _infer_domain(trace_task_id),
            "task_id": trace_task_id,
            "model_version": _MODEL_VERSION,
        }
        # v15 Phase 2 Task 3.4: upgrade_entry 补 v15 默认值 + 填本轮 abduction 结果.
        try:
            from huginn.metacog.trace_topology import upgrade_entry

            upgrade_entry(entry)
            entry["hypothesis_id"] = best_h_id
            entry["log_posterior"] = log_post
            entry["fisher_info"] = fisher_info
        except Exception:
            logger.debug("meta trace entry upgrade skipped", exc_info=True)
        # v26 Task 26.11: 走分片写入, 跨 shard 边界自动 gzip 归档老分片.
        shard_writer(shard, entry, entry.get("iteration"))
        # v14 Task 4: 累积 entry 到 _trace_history, 算 betti 写 jsonl.
        trace_history.append(entry)
        if compute_betti is not None:
            try:
                b0, b1 = compute_betti(trace_history)
                with betti_path.open("a", encoding="utf-8") as bf:
                    bf.write(json.dumps({
                        "ts": time.time(),
                        "iteration": iter_n + 1,
                        "beta_0": b0,
                        "beta_1": b1,
                        "n_entries": len(trace_history),
                    }, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[betti write skipped: {e}]", flush=True)
    except Exception as e:
        print(f"[meta_trace write skipped: {e}]", flush=True)
    return report_text, report_path, entry


def _apply_pmk_reverse_edges(
    *,
    pmk_state: dict | None,
    entry: dict | None,
    mem_mgr: Any,
    kb: Any,
    iter_n: int,
    task_id: str,
    prev_darwin: float | None,
) -> float | None:
    """P1-B: PMK 三路反向边 (prospective / KB); 失败只 warn, 返回新 prev_darwin."""
    if pmk_state is None:
        return prev_darwin
    try:
        cur_darwin = float(_entry_get(entry, "darwin_score", 0.5))
        # 反向边 1 (persona.adaptive_layer) 原实现依赖未定义的 `_pm`, 恒为死分支, 已移除.
        # 反向边 2: PMK memory → prospective memory
        if mem_mgr is not None and pmk_state.get("memory"):
            from huginn.memory.prospective import _new_intention_id

            mem_mgr.remember_prospective({
                "intention_id": _new_intention_id(),
                "description": f"PMK memory reverse edge: {pmk_state['memory'][:160]}",
                "trigger_type": "dependency",
                "trigger_payload": {"depends_on_step": iter_n},
                "priority": 3,
                "created_at": time.time(),
                "source_step": iter_n,
            })
        # 反向边 3: PMK → KB (darwin 上升时入库, 下降/持平不入库).
        if (
            kb is not None
            and prev_darwin is not None
            and cur_darwin > prev_darwin
            and isinstance(entry, dict)
        ):
            kb.add_text(
                f"[iter {iter_n + 1} darwin={cur_darwin:.3f}] "
                f"{(entry.get('attempted') or '')[:200]}",
                metadata={
                    "source": "pmk_reverse_edge",
                    "iter": iter_n + 1,
                    "darwin_score": cur_darwin,
                    "task_id": task_id,
                },
            )
            print(
                f"[PMK reverse edge] KB add_text "
                f"(darwin {prev_darwin:.3f}→{cur_darwin:.3f})",
                flush=True,
            )
        return cur_darwin
    except Exception as e:
        print(f"[PMK reverse edge] skipped: {e}", flush=True)
        return prev_darwin


def _handle_pause_decision(
    *,
    pause_reason: str,
    pause_opts: list,
    iter_n: int,
    task_id: str,
    trace_task_id: str,
    ws: Path,
    trace_path: Path,
    evals_history: list,
    kb: Any,
) -> None:
    """Task 29: 人机协同 pause — RCB 测试环境自动选 A resume."""
    from huginn.runtime.task_lifecycle import (
        DecisionRequest,
        TaskLifecycle,
        TaskState,
        load_task_lifecycle,
        save_task_lifecycle,
    )

    pause_step = iter_n + 1
    request = DecisionRequest(
        step_id=pause_step,
        question=pause_reason,
        options=pause_opts,
        context_summary=(
            f"iter={pause_step}, "
            f"evals={len(evals_history)}, "
            f"kb={'empty' if kb is None else 'available'}"
        ),
    )
    lifecycle = load_task_lifecycle(task_id, ws) or TaskLifecycle(task_id=task_id)
    if lifecycle.state != TaskState.RUNNING:
        try:
            lifecycle.transition(TaskState.RUNNING)
        except ValueError:
            # 终态/暂停态 → 新建一个干净的 lifecycle
            lifecycle = TaskLifecycle(task_id=task_id)
            lifecycle.transition(TaskState.RUNNING)
    lifecycle.pause_for_decision(request)
    save_task_lifecycle(lifecycle, ws)
    print(
        f"\n[human-in-loop] PAUSE step={pause_step}: {pause_reason}", flush=True)
    for opt in pause_opts:
        print(
            f"  {opt.get('id', '?')}: {opt.get('label', '')} "
            f"(pros: {opt.get('pros', '')}, cons: {opt.get('cons', '')})",
            flush=True,
        )
    # RCB 测试环境: 默认选 A 自动 resume, 不等真人
    auto_choice = pause_opts[0].get("id", "A") if pause_opts else "A"
    lifecycle.resume(answer=auto_choice)
    save_task_lifecycle(lifecycle, ws)
    print(
        f"[human-in-loop] auto-resume with {auto_choice}: "
        f"{pause_opts[0].get('label', '') if pause_opts else ''}",
        flush=True,
    )
    try:
        hd_entry = {
            "iteration": pause_step,
            "ts": time.time(),
            "role": "human_decision",
            "attempted": f"pause: {pause_reason}",
            "found": (
                f"auto-resume {auto_choice}: "
                f"{pause_opts[0].get('label', '') if pause_opts else ''}"
            ),
            "evidence": [],
            "limitations": ["RCB auto-resume, no real human in loop"],
            "artifacts": [],
            "next_hint": "continue after decision",
            "darwin_score": 0.0,
            "supported_ratio": 0.0,
            # v14 Task 1: human_decision 不在 spec 三族主映射里 → legacy.
            "simplex_id": _make_simplex_id(trace_task_id, iter_n + 1, "human_decision"),
            "cochain_type": "legacy",
            "domain": _infer_domain(trace_task_id),
            "task_id": trace_task_id,
            "model_version": _MODEL_VERSION,
        }
        try:
            from huginn.metacog.trace_topology import upgrade_entry

            upgrade_entry(hd_entry)
        except Exception:
            logger.debug("human_decision trace entry upgrade skipped", exc_info=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(hd_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[human-in-loop] meta_trace write skipped: {e}", flush=True)


def _maybe_pause_for_human(
    *,
    evals_history: list,
    target_chains: list,
    kb: Any,
    fired: list,
    persona: Any,
    last_step_eval: Any,
    run_start_iso: str | None,
    mem_mgr: Any,
    entry: dict | None,
    prev_darwin: float | None,
    iter_n: int,
    task_id: str,
    trace_task_id: str,
    ws: Path,
    trace_path: Path,
) -> tuple[dict | None, float | None]:
    """AV4 + P1-B: 构建 PMK 状态, 跑反向边, 判定是否 pause."""
    pmk_state = None
    try:
        from huginn.autoloop.cognitive_loop import build_pmk_state, check_pause_decision

        pmk_state = build_pmk_state(
            persona, last_step_eval, kb, since=run_start_iso, mem_mgr=mem_mgr,
        )
        prev_darwin = _apply_pmk_reverse_edges(
            pmk_state=pmk_state,
            entry=entry,
            mem_mgr=mem_mgr,
            kb=kb,
            iter_n=iter_n,
            task_id=task_id,
            prev_darwin=prev_darwin,
        )
        pause, pause_reason, pause_opts = check_pause_decision(
            evals_history, target_chains, kb, fired, pmk_state,
        )
        if pause:
            _handle_pause_decision(
                pause_reason=pause_reason,
                pause_opts=pause_opts,
                iter_n=iter_n,
                task_id=task_id,
                trace_task_id=trace_task_id,
                ws=ws,
                trace_path=trace_path,
                evals_history=evals_history,
                kb=kb,
            )
    except Exception as e:
        print(f"[Step 2] human-in-loop pause warning: {e}", flush=True)
    return pmk_state, prev_darwin


def _save_loop_checkpoint(
    *,
    entry: dict | None,
    ai_text: str | None,
    iter_n: int,
    target_chains: list,
    prospective_mem: Any,
    task_id: str,
    ws: Path,
) -> None:
    """G59: 每轮 checkpoint 落盘 (含 engine_state_digest 防 drift); 失败只 warn."""
    try:
        from huginn.runtime.checkpoint import save_checkpoint

        tc_progress = {tc.target_id: tc.progress for tc in target_chains}
        pending = (
            [i.intention_id for i in prospective_mem.list_pending()]
            if prospective_mem is not None else []
        )
        darwin_cp = float(_entry_get(entry, "darwin_score", 0.5))
        sup_ratio_cp = float(_entry_get(entry, "supported_ratio", 0.0))
        es_digest = hashlib.md5(
            f"{darwin_cp:.6f}|{sup_ratio_cp:.6f}".encode(),
            usedforsecurity=False,
        ).hexdigest()
        save_checkpoint(
            task_id=task_id,
            step_id=iter_n + 1,
            phase="execute",
            workspace=ws,
            context_digest=hashlib.md5(
                (ai_text or "").encode(), usedforsecurity=False).hexdigest(),
            memory_cursor=None,
            target_chain_progress=tc_progress,
            prospective_queue=pending,
            engine_state_digest=es_digest,
        )
    except Exception as e:
        print(f"[Step 2] checkpoint save warning: {e}", flush=True)


def _record_cross_task_iteration(
    *,
    mem_mgr: Any,
    entry: dict | None,
    iter_n: int,
    ws_name: str,
    persona_name: str,
) -> None:
    """跨任务 iteration_result 写入 — curiosity hint 的数据源."""
    if mem_mgr is None or os.environ.get("HUGINN_RCB_CROSS_TASK", "1") != "1":
        return
    try:
        darwin = float(_entry_get(entry, "darwin_score", 0.5))
        sup_ratio = float(_entry_get(entry, "supported_ratio", 0.0))
        typed_status = "supported" if darwin >= 0.5 else "refuted"
        mem_mgr.remember_typed(
            content=(
                f"RCB iter {iter_n + 1} task={ws_name} darwin={darwin:.2f} "
                f"supported_ratio={sup_ratio:.2f}"
            ),
            memory_type="iteration_result",
            run_id=ws_name,
            persona_id=persona_name,
            status=typed_status,
            importance=darwin,
            tier="mid",
            tags=["rcb", f"task:{ws_name}", f"iter:{iter_n + 1}"],
            source="rcb_runner",
        )
    except Exception as e:
        print(f"[Memory] iteration_result write skipped: {e}", flush=True)


def _update_stagnation_heat(
    *,
    report_text: str,
    prev_hash: str | None,
    stagnation_count: int,
    t_hot: float,
) -> tuple[str | None, int, float]:
    """报告 hash 不变则计停滞并升温; 返回 (新 hash, 计数, T_hot)."""
    curr_hash = (
        hashlib.md5(report_text.encode(), usedforsecurity=False).hexdigest()
        if report_text else None
    )
    if curr_hash == prev_hash and curr_hash is not None:
        stagnation_count += 1
        t_hot = min(1.0, t_hot + 0.5)
        print(
            f"[stagnation: report.md unchanged for {stagnation_count} iters, heating]",
            flush=True,
        )
    else:
        stagnation_count = 0
    return curr_hash, stagnation_count, t_hot


def _push_score_history(score_history: Any, entry: dict | None, iter_n: int) -> None:
    """P4 Task 21: score_history 滑动窗口 — 只监测与日志, 不早退."""
    if score_history is None:
        return
    try:
        darwin = _entry_get(entry, "darwin_score", 0.5)
        score_history.push(darwin)
        temp = score_history.get_temperature()
        stats = score_history.stats()
        if stats["n_samples"] >= 2:
            print(
                f"[score_history] iter {iter_n}: temp={temp:.2f} "
                f"mean={stats['mean']:.3f} var={stats['variance']:.3f} "
                f"streak={stats['monotonic_streak']}",
                flush=True,
            )
        if score_history.should_terminate():
            print(
                f"[score_history] monotonic decrease "
                f"{stats['monotonic_streak']} iters (advisory, not stopping)",
                flush=True,
            )
    except Exception as e:
        print(f"[score_history push skipped: {e}]", flush=True)


def _run_cumulative_audit(
    auditor: Any, ws: Path, task_type: str, iter_n: int
) -> bool:
    """P4 Task 22: 累积产出审计; 返回 True 表示需阻断主循环."""
    if auditor is None:
        return False
    try:
        res = auditor.audit_step(ws / "outputs", task_type=task_type)
        if res.get("blocked"):
            print(
                f"[cumulative_audit] BLOCKED at iter {iter_n}: {res.get('reason')}",
                flush=True,
            )
            return True
    except Exception as e:
        print(f"[cumulative_audit skipped: {e}]", flush=True)
    return False


def _run_darwin_ratchet(
    *, entry: dict | None, stagnation_count: int, iter_n: int
) -> tuple[str | None, int]:
    """认知反馈: darwin ratchet 分类 stall; 返回 (action, stagnation_count)."""
    try:
        from huginn.autoloop.cognitive_loop import classify_stall, darwin_ratchet_check

        darwin = float(_entry_get(entry, "darwin_score", 0.5))
        ratio = float(_entry_get(entry, "supported_ratio", 0.0))
        action, state = darwin_ratchet_check(
            darwin_score=darwin,
            supported_ratio=ratio,
            stagnation_count=stagnation_count,
        )
        if action == "stop":
            print(
                f"[Step2] darwin_ratchet: stop at iter {iter_n}, score={darwin:.3f}",
                flush=True,
            )
        elif action in ("pivot", "counterexample"):
            stall_type = classify_stall([darwin], [ratio])
            print(
                f"[Step2] darwin_ratchet: {action} "
                f"(stall_type={stall_type}) at iter {iter_n}",
                flush=True,
            )
        return action, state.get("stagnation_count", stagnation_count)
    except Exception as e:
        print(f"[Step2] darwin_ratchet_check failed: {e}", flush=True)
        return None, stagnation_count


def _run_mcmc_step(
    *,
    manifold: Any,
    observations: list,
    iter_n: int,
    best_h_id: str | None,
    engine: Any,
    cached_log_p: float | None,
    interval: int,
    checkpoint_interval: int,
    haptic_enabled: bool,
    haptic_temperature: float,
    alignment_enabled: bool,
    alignment_temperature: float,
    task_id: str,
    ws: Path,
) -> tuple[float | None, dict | None]:
    """P1-A: 周期性 mcmc_step; 返回 (新 cached_log_p, current_info)."""
    info: dict | None = None
    if manifold is None or not observations or iter_n % interval != 0:
        return cached_log_p, info
    try:
        # 初始化 current: 用本轮 best_h_id (posterior 最高的)
        if engine._mcmc_current is None and best_h_id is not None:
            engine._mcmc_current = best_h_id
        if engine._mcmc_current is None or len(manifold._hyp) < 2:
            return cached_log_p, info
        prev = engine._mcmc_current
        # 增量路径 — cached_log_p 跨步复用, 不再调 log_posterior 全量.
        kwargs = {
            "rng": engine._mcmc_rng,
            "cached_log_p_current": cached_log_p,
            # Gramian 谱预条件 (沿高可控/高信息主轴提议); 数据不足自动退化.
            "gramian_enabled": os.environ.get("HUGINN_MCMC_GRAMIAN", "1") == "1",
            "gramian_k": int(os.environ.get("HUGINN_MCMC_GRAMIAN_K", "1")),
        }
        if haptic_enabled:
            kwargs["haptic_enabled"] = True
            kwargs["haptic_temperature"] = haptic_temperature
        if alignment_enabled:
            kwargs["alignment_enabled"] = True
            kwargs["alignment_temperature"] = alignment_temperature
        next_h, next_log_p = manifold.mcmc_step(
            observations, engine._mcmc_current, **kwargs)
        engine._mcmc_current = next_h
        cached_log_p = next_log_p
        accepted = engine._mcmc_current != prev
        if accepted:
            engine._mcmc_accept_count += 1
        engine._mcmc_step_count += 1
        engine._iteration = iter_n
        # 接受率注入 bandit 作探索/利用信号; 失败静默, 不阻塞 MCMC.
        try:
            from huginn.agent.bandit_controller import EffortBandit

            EffortBandit.get_instance().update_mcmc_acceptance(
                1.0 if accepted else 0.0)
        except Exception:
            logger.debug("bandit mcmc acceptance update skipped", exc_info=True)
        print(
            f"[mcmc] iter {iter_n}: "
            f"{'accepted' if accepted else 'rejected'} "
            f"h={engine._mcmc_current} llh={next_log_p:.3f}",
            flush=True,
        )
        info = {
            "accepted": accepted,
            "h_id": engine._mcmc_current,
            "llh": next_log_p,
        }
        # 周期 checkpoint — 每 HUGINN_MCMC_CHECKPOINT_INTERVAL 步落盘
        if (
            checkpoint_interval > 0
            and engine._mcmc_step_count % checkpoint_interval == 0
        ):
            try:
                from huginn.runtime.engine_state import save_engine_state

                engine._mcmc_rng_state = engine._mcmc_rng.getstate()
                save_engine_state(engine, task_id, ws)
            except Exception:
                print("[mcmc] checkpoint save failed (non-fatal)", flush=True)
    except Exception as e:
        print(f"[mcmc] skipped: {e}", flush=True)
        info = None
    return cached_log_p, info


def _run_completion_audit(
    *,
    ws: Path,
    report_text: str,
    report_path: Path,
    iter_n: int,
    interval: int,
    prev_hint: str,
) -> tuple[dict | None, str]:
    """P1-C: 周期性完成度审计; 返回 (audit, 下轮 gap hint)."""
    audit = None
    if iter_n <= 0 or iter_n % interval != 0:
        return audit, prev_hint
    try:
        from huginn.autoloop.cognitive_loop import metacog_check_completion

        rep_md = report_text or (
            report_path.read_text(encoding="utf-8") if report_path.exists() else ""
        )
        audit = metacog_check_completion(
            report_md=rep_md, outputs_dir=ws / "outputs")
        if not audit.get("passed", True):
            gaps = audit.get("block_reasons", [])
            print(f"[completion audit iter {iter_n}] NOT passed: {gaps}", flush=True)
            hint = (
                f"\n\n## Completion Audit Advisory (iter {iter_n})\n"
                f"Auto-audit found gaps: {gaps}. "
                f"Address these before claiming TASK COMPLETE."
            )
            return audit, hint
        print(f"[completion audit iter {iter_n}] passed", flush=True)
        return audit, ""
    except Exception as e:
        print(f"[completion audit skipped: {e}]", flush=True)
        return audit, prev_hint


async def _detect_anomalies(
    manifold: Any, model: Any, iter_n: int, interval: int
) -> dict | None:
    """触觉层: 同构不同性异常检测 (每 interval 步一次); 失败返回 None."""
    if manifold is None or iter_n <= 0 or iter_n % interval != 0:
        return None
    try:
        pairs = manifold.detect_isomorphic_anomaly()
        generated: list[str] = []
        if pairs:
            generated = await _trigger_anomaly_hypothesis(pairs, model)
        if pairs or generated:
            print(
                f"[anomaly] iter {iter_n}: {len(pairs)} pair(s), "
                f"{len(generated)} new hypothesis",
                flush=True,
            )
            return {"pairs": pairs, "generated": generated}
    except Exception as e:
        print(f"[anomaly] detection skipped: {e}", flush=True)
    return None


@dataclass
class _S2State:
    """Step 2 执行循环状态 — 跨阶段显式传递, 承载原函数局部变量.

    字段赋值/读取顺序与原 `_step2_execute` 完全一致; 未赋值字段用 None/"" 兜底,
    与原先 `"_x" in dir()` 守卫的语义等价.
    """
    # --- 只读上下文 (来自 _RCBStep2Ctx) ---
    ws: Path
    model: Any
    agent: Any
    kb: Any
    mem_mgr: Any
    persona: Any
    kg: Any
    thread_id: str
    task_id: str
    trace_task_id: str
    trace_domain: Any
    cross_task_prior: list
    resume_from_iter: int
    extreme: bool
    checklist: str
    scan_text: str
    fcm: dict
    target_chains: list
    instructions: Any
    persona_name: str = "default"
    # --- iter 0 prompt ---
    scan_hint: str = ""
    fcm_hint: str = ""
    step2_prompt_base: str = ""
    step2_prompt: str = ""
    hint_coord_enabled: bool = False
    hint_coord: Any = None
    # --- trace / 迭代上限 ---
    trace_path: Path | None = None
    meta_trace_shard: Any = None
    meta_trace_writer: Any = None
    max_exec_iters: int = 0
    # --- 循环计时/审计器 ---
    prev_report_hash: str | None = None
    stagnation_count: int = 0
    prev_darwin: float | None = None
    score_history: Any = None
    cumulative_auditor: Any = None
    audit_task_type: str = "repro"
    # --- 退火 / 分叉 ---
    t_hot: float = 1.0
    fork_k_max: int = 3
    fork_enabled: bool = True
    merge_hint: str = ""
    # --- StepEvaluator / 拓扑 ---
    evals_history: list | None = None
    trace_history: list | None = None
    compute_betti: Any = None
    betti_path: Path | None = None
    prospective_mem: Any = None
    last_step_eval: Any = None
    # --- metrics / drift / heat ---
    run_started_at: float = 0.0
    task_metrics: Any = None
    task_state_for_metrics: Any = None
    metrics_ok: bool = False
    drift_info: Any = None
    heat_engine: Any = None
    imagination_block: str = ""
    # --- manifold / abduction ---
    hypo_manifold: Any = None
    hypo_manifold_path: Path | None = None
    hypo_obs_path: Path | None = None
    last_abduction_result: Any = None
    iter_observations: list | None = None
    # --- MCMC ---
    mcmc_engine: Any = None
    mcmc_cached_log_p: float | None = None
    mcmc_interval: int = 5
    mcmc_ckpt_interval: int = 10000
    mcmc_haptic_enabled: bool = False
    mcmc_haptic_temperature: float = 1.0
    mcmc_alignment_enabled: bool = False
    mcmc_alignment_temperature: float = 1.0
    # --- 完成度审计 / 停滞 ---
    completion_interval: int = 10
    prev_completion_hint: str = ""
    stagnation_history: list | None = None
    imagination_log_path: Path | None = None
    # --- 认知层 ---
    self_model: Any = None
    budget_items: Any = None
    bandit: Any = None
    # --- 循环变量 ---
    consecutive_complete_rejections: int = 0
    iter_best_h_id: str | None = None
    entry: Any = None
    pmk_state: Any = None
    mcmc_current_info: Any = None
    report_text: str = ""
    report_path_iter: Path | None = None
    max_complete_rejections: int = 3
    derivation_audit: str = ""
    iter_prompt: str = ""
    ai_text: Any = None
    fired: Any = None
    iter_log_post: float = 0.0
    iter_fisher_info: float = 0.0
    completion_audit: Any = None
    anomaly_info: Any = None


def _s2_build_scan_hint(scan_text: str) -> str:
    """Step 1.5 scan 结果 → execution guidance hint (空文本返空串)."""
    if scan_text and scan_text.strip():
        return (
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
    return ""


def _s2_build_fcm_hint(fcm: dict) -> str:
    """Step 1.7 fork-critique-merge 选中计划的 hint (无 winner 返空串)."""
    if not fcm["winner_plan"]:
        return ""
    _insights = "\n".join(f"- {x}" for x in fcm["merge_insights"]) or "- (none)"
    return (
        f"\n\n## Selected Execution Plan (Step 1.7 fork-critique-merge, "
        f"winner perspective: {fcm['winner_perspective']})\n{fcm['winner_plan']}\n\n"
        f"## Merge insights from rejected candidates\n{_insights}\n\n"
        f"Follow this plan unless execution proves it infeasible — "
        f"if you deviate, explain why in report.md."
    )


def _s2_step2_prompt_base() -> str:
    """Step 2 基础 prompt — 方法论 checklist 执行 + METHOD SUBSTITUTE 纪律."""
    return (
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


def _s2_build_step2_prompt(st: _S2State) -> None:
    """iter 0 prompt: HintCoordinator Hodge 合并, 或旧拼接路径兜底."""
    st.step2_prompt_base = _s2_step2_prompt_base()
    st.hint_coord_enabled = os.environ.get(
        "HUGINN_HINT_COORDINATOR", "1").lower() not in ("0", "false", "no")
    if st.hint_coord_enabled:
        from huginn.agent.hint_coordinator import HintCoordinator
        st.hint_coord = HintCoordinator()
        st.step2_prompt, _hint_trace_events = st.hint_coord.coordinate(
            iter_n=0,
            # ponytail: 字符串硬编码, Task 2 CSM 完善后接 csm.current_state.
            csm_state="S4_CONSTRUCT",
            # ponytail: Task 4 未完, 占位 (1, 0). 升级路径: 接 betti 计算.
            beta=(1, 0),
            last_verdict=None,
            fcm_winner=st.fcm.get("winner_plan") or None,
            scan_text=st.scan_text,
            step2_prompt=st.step2_prompt_base,
            iter_prompt=None,
            compass=None,
            step_eval=None,
            drift_info=None,
            imagination=None,
            meta_agent=None,
            cross_task_prior=st.cross_task_prior,
            # v15 Task 4: iter 0 不传 manifold — _hypo_manifold 在下方才 init,
            # 且 iter 0 无 observations 不需要 posterior lift. ponytail: 默认 None
            # 走 v14 keyword overlap, 不破坏 v14 行为.
        )
    else:
        st.step2_prompt = _legacy_build_step2_prompt(
            st.step2_prompt_base, st.scan_hint, st.fcm_hint)


def _s2_init_trace(st: _S2State) -> None:
    """meta_trace 路径 / 分片状态 / 迭代上限."""
    st.trace_path = st.ws / HUGINN_DIR_NAME / "meta_trace.jsonl"
    st.trace_path.parent.mkdir(parents=True, exist_ok=True)
    # v26 Task 26.11: meta_trace 也走分片. RCB 跑 700 万 call 时单 jsonl 撑爆磁盘,
    # 每 N iter (HUGINN_TRACE_SHARD_INTERVAL, 默认 100) 切一个文件, 老分片 gzip 到
    # .huginn/archive/<task_id>/. 跟 audit_log 复用 _ShardState / write_sharded_jsonl,
    # 不引新依赖. ponytail: task_id 取 _trace_task_id (短 id, 已 strip 时间戳).
    # 升级路径: 整套换 Postgres/Cassandra, 文件分片只在单机 dev 场景留作 fallback.
    from huginn.events.audit_log import _ShardState as _MetaTraceShard
    from huginn.events.audit_log import write_sharded_jsonl as _mt_write_shard
    st.meta_trace_shard = _MetaTraceShard(
        base_dir=st.trace_path.parent,
        default_path=st.trace_path,
        shard_interval=int(os.environ.get("HUGINN_TRACE_SHARD_INTERVAL", "100")) or 100,
        filename_prefix="meta_trace",
        task_id=st.trace_task_id,
    )
    st.meta_trace_writer = _mt_write_shard
    st.max_exec_iters = int(os.environ.get(
        "HUGINN_RCB_EXEC_ITERS",
        "20" if st.extreme else "2",
    ))


def _s2_init_loop_timers(st: _S2State) -> None:
    """停滞 / darwin 对比计时器 + score_history 滑窗."""
    st.prev_report_hash = None
    st.stagnation_count = 0
    # P1-B: PMK 闭环反向边用 — 记上一轮 darwin_score, 本轮对比判断升/降.
    st.prev_darwin = None
    # P4 Task 21: score_history 滑动窗口 — 监测 darwin_score 振荡/单调下降.
    st.score_history = None
    try:
        from huginn.runtime.score_history import ScoreHistory
        st.score_history = ScoreHistory()
    except Exception as _she:
        print(f"[score_history init skipped: {_she}]", flush=True)


def _s2_init_auditors(st: _S2State) -> None:
    """P4 Task 22: cumulative output audit 初始化 + task_type 推断."""
    st.cumulative_auditor = None
    st.audit_task_type = "repro"
    try:
        from huginn.security.cumulative_audit import CumulativeAuditor
        st.cumulative_auditor = CumulativeAuditor()
        _ck_lower = (st.checklist or "").lower()
        if "optim" in _ck_lower and "repro" not in _ck_lower:
            st.audit_task_type = "optim"
        elif "repro" not in _ck_lower and "reproduction" not in _ck_lower:
            st.audit_task_type = "default"
    except Exception as _cae:
        print(f"[cumulative_audit init skipped: {_cae}]", flush=True)


def _s2_init_fork_state(st: _S2State) -> None:
    """认知退火初始温度 + 分叉开关 + merge hint 缓存."""
    # 任务开始 = 1.0 (最热, 多轨迹探索), 每轮几何降温 ×0.5, 停滞重热 +0.5.
    # ponytail: 不接 belief_entropy 测量 — RCB mini-loop 没有 hypothesis graph,
    #   熵无来源. 用模拟退火 schedule + 停滞信号做代理. 升级路径: 7-phase
    #   autoloop 接入后换 belief_entropy 驱动 (cognitive_heat_engine 已有接口).
    st.t_hot = 1.0
    st.fork_k_max = int(os.environ.get("HUGINN_RCB_FORK_K_MAX", "3"))
    st.fork_enabled = os.environ.get(
        "HUGINN_RCB_FORK_ENABLED", "1").lower() not in ("0", "false", "no")
    st.merge_hint = ""


def _s2_init_eval_state(st: _S2State) -> None:
    """StepEvaluator 历史 / trace 历史 / betti / prospective memory."""
    st.evals_history = []
    # v14 Task 4: trace entry 累积, 每轮 append 主 entry 供 compute_betti 用.
    st.trace_history = []
    # v14 Task 4: betti 计算 lazy import — 失败不阻塞主循环, betti jsonl 跳过.
    st.compute_betti = None
    try:
        from huginn.metacog.trace_topology import compute_betti as _compute_betti
        st.compute_betti = _compute_betti
    except Exception as _tte:
        print(f"[betti init skipped: {_tte}]", flush=True)
    st.betti_path = st.ws / HUGINN_DIR_NAME / "meta_trace_betti.jsonl"
    st.prospective_mem = None
    try:
        from huginn.memory.prospective import ProspectiveMemory
        st.prospective_mem = ProspectiveMemory(workspace=st.ws)
    except Exception as _e:
        print(f"[Prospective] init warning: {_e}", flush=True)
    # Task 18: 上一轮 StepEvaluation, 首轮 None. 每轮结束更新, 下一轮注入 prompt.
    st.last_step_eval = None


def _s2_init_task_metrics(st: _S2State) -> None:
    """G70: TaskMetrics 初始化 (resume 时从盘上 load) + domain_label 推断."""
    # total_steps 用 _max_exec_iters 估上界; task_state 用 SimpleNamespace(created_at=start_ts)
    # 给 update_metrics 算 estimated_remaining 用.
    st.run_started_at = time.time()
    try:
        from types import SimpleNamespace as _NS  # noqa: N814

        from huginn.runtime.task_metrics import (
            TaskMetrics,
            load_metrics,
        )
        st.task_metrics = load_metrics(st.task_id, st.ws) or TaskMetrics(
            task_id=st.task_id, total_steps=st.max_exec_iters)
        # 跨领域: 用 suggest_domain 推断 domain_label (材料/物理/化学/医学/数学)
        # ponytail: keyword 匹配, 不上分类器. 失败默认 unknown.
        try:
            from huginn.personas import PersonaManager
            _pm_for_domain = PersonaManager()
            _domain = _pm_for_domain.suggest_domain(
                (st.checklist or "") + " " + (st.instructions or ""))
            st.task_metrics.domain_label = _domain
        except Exception:
            logger.debug("task domain label inference skipped", exc_info=True)
        st.task_state_for_metrics = _NS(created_at=st.run_started_at)
        st.metrics_ok = True
    except Exception as _e:
        print(f"[task_metrics init skipped: {_e}]", flush=True)
        st.task_metrics = None
        st.task_state_for_metrics = None
        st.metrics_ok = False


def _s2_imagination_block() -> str:
    """P1: 想象力 directive 文本块 (复制自 autoloop 的 _IMAGINATION_PROMPT_BLOCK)."""
    return """
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


def _s2_init_heat(st: _S2State) -> None:
    """drift 缓存 + heat_engine + imagination 文本块."""
    # G62: 首轮 None; build_meta_agent_text(drift_info=None) 走空路径不报错.
    st.drift_info = None
    # P1: 想象力机制耦合 — 复用 cognitive_heat_engine.
    st.heat_engine = None
    try:
        from huginn.metacog.cognitive_heat_engine import get_heat_engine
        st.heat_engine = get_heat_engine()
    except Exception as _he:
        print(f"[heat_engine init skipped: {_he}]", flush=True)
    st.imagination_block = _s2_imagination_block()


def _s2_init_manifold_state(st: _S2State) -> None:
    """v15 HypothesisManifold 路径 + abduction / observations 缓存."""
    st.hypo_manifold = None
    st.hypo_manifold_path = st.ws / HUGINN_DIR_NAME / "hypothesis_manifold.jsonl"
    st.hypo_obs_path = st.ws / HUGINN_DIR_NAME / "observations.jsonl"
    st.last_abduction_result = None
    # v15 Task 4: 上一轮 observations, 给 _build_posterior_guided_hint 用.
    st.iter_observations = []


def _s2_init_mcmc(st: _S2State) -> None:
    """P1-A: MCMC holder + 间隔 / 触觉层 / alignment env 配置."""
    # ponytail: MCMC 是 advisory only, 不强制 agent 用 MCMC 选的 h.
    # rcb_runner 不接 AutoloopEngine, 用 SimpleNamespace 当 holder 让 save_engine_state 能拉到字段
    import random as _mcmc_random
    import types as _mcmc_types
    st.mcmc_engine = _mcmc_types.SimpleNamespace(
        _mcmc_current=None,
        _mcmc_rng=_mcmc_random.Random(
            int(os.environ.get("HUGINN_MCMC_SEED", "42"))),
        _mcmc_rng_state=None,
        _mcmc_accept_count=0,
        _mcmc_step_count=0,
        _mcmc_chains={},
        _iteration=0,
        workspace=st.ws,
        hypothesis_graph=None,
    )
    # Step 2: 跨步缓存 current 的 log_posterior, 拒绝时复用避免重算
    st.mcmc_cached_log_p = None
    st.mcmc_interval = int(os.environ.get("HUGINN_MCMC_INTERVAL", "5"))
    st.mcmc_ckpt_interval = int(os.environ.get(
        "HUGINN_MCMC_CHECKPOINT_INTERVAL", "10000"))
    # 触觉层 env var: extreme 默认开, 非 extreme 默认关 (行为不变).
    # ponytail: haptic_layers.json 不存在时 _haptic_proposal 全返 None, 自动退化 fisher.
    st.mcmc_haptic_enabled = os.environ.get(
        "HUGINN_MCMC_HAPTIC", "1" if st.extreme else "0") == "1"
    st.mcmc_haptic_temperature = float(
        os.environ.get("HUGINN_MCMC_HAPTIC_TEMPERATURE", "1.0"))
    st.mcmc_alignment_enabled = os.environ.get("HUGINN_MCMC_ALIGNMENT", "0") == "1"
    st.mcmc_alignment_temperature = float(
        os.environ.get("HUGINN_MCMC_ALIGNMENT_TEMPERATURE", "1.0"))


def _s2_init_completion_state(st: _S2State) -> None:
    """P1-C 完成度审计间隔 + v15 stagnation 历史 / imagination 日志路径."""
    st.completion_interval = int(os.environ.get("HUGINN_COMPLETION_CHECK_INTERVAL", "10"))
    st.prev_completion_hint = ""
    st.stagnation_history = []
    st.imagination_log_path = st.ws / HUGINN_DIR_NAME / "imagination_log.jsonl"


def _s2_init_self_model(st: _S2State) -> None:
    """v15 Phase 5 Task 10: SelfModel 接入 (失败降级 None)."""
    st.self_model = None
    _self_model_path = st.ws / HUGINN_DIR_NAME / "self_model.json"
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
            _self_model_cross_path = st.ws / HUGINN_DIR_NAME / "self_model_cross_task.json"
        from huginn.metacog.self_model import SelfModel
        st.self_model = SelfModel(
            task_local_path=_self_model_path,
            cross_task_path=_self_model_cross_path,
            model=st.model,
        )
        print(
            f"[v15] SelfModel init: {len(st.self_model._skills)} skills "
            f"(cross-task={_self_model_cross_path})",
            flush=True,
        )
    except Exception as _e:
        print(f"[v15] SelfModel init skipped: {_e}", flush=True)
        st.self_model = None


def _s2_init_hypothesis_manifold(st: _S2State) -> None:
    """v15 Phase 2 Task 3.1: HypothesisManifold init (失败降级 None)."""
    try:
        st.hypo_manifold = _init_hypothesis_manifold(
            ws=st.ws,
            task_id=st.trace_task_id,
            checklist=st.checklist or "",
            instructions=st.instructions,
            scan_text=st.scan_text or "",
            model=st.model,
            task_ctx=st.checklist or "",
        )
        if st.hypo_manifold is not None:
            print(
                f"[v15] HypothesisManifold init: "
                f"{len(st.hypo_manifold._hyp)} hypotheses",
                flush=True,
            )
    except Exception as _e:
        print(f"[v15] HypothesisManifold init skipped: {_e}", flush=True)
        st.hypo_manifold = None


def _s2_init_bandit(st: _S2State) -> None:
    """v18: bandit effort controller — register checklist items."""
    # ponytail: 不再 time-slice (v17 dead code), bandit 通过 reward 学习何时 switch.
    st.budget_items = None
    try:
        st.budget_items = _checklist_item_parser(st.checklist or "")
        if st.budget_items and len(st.budget_items) >= 2:
            from huginn.agent.bandit_controller import EffortBandit
            _bandit = EffortBandit.get_instance()
            _bandit.set_items(st.budget_items)
            _bandit.switch_item(0)
            print(f"[v18] bandit registered {len(st.budget_items)} items, "
                  f"Q-table size={len(_bandit._Q)}", flush=True)
        else:
            st.budget_items = None
    except Exception as _e:
        print(f"[v18] bandit init skipped (fallback to no-bandit): {_e}", flush=True)
        st.budget_items = None


def _s2_init_loop_counters(st: _S2State) -> None:
    """循环计数器 + 条件赋值变量的显式 None 初始化 (替代原 dir() 守卫)."""
    # Task 3: 从 resume 的 iter 开始, 不重跑已 checkpoint 的轮次
    # 连续驳回计数: agent 已尽力但 gates 太严时, 第 3 次接受 TASK COMPLETE.
    st.consecutive_complete_rejections = 0
    # 上一轮 best hypothesis id; iter 0 无值 (None) → 草图 spec 走 checklist 兜底.
    st.iter_best_h_id = None
    st.entry = None
    st.pmk_state = None
    st.bandit = None
    st.mcmc_current_info = None
    st.report_text = ""
    st.report_path_iter = st.ws / "report" / "report.md"
    st.max_complete_rejections = int(
        os.environ.get("HUGINN_RCB_MAX_COMPLETE_REJECTIONS", "3"))


def _s2_make_state(ctx: _RCBStep2Ctx) -> _S2State:
    """上下文解包 + trace task/domain 推断 + 跨 task darwin prior."""
    # v14 Task 1: trace 里存短 task_id (Astronomy_000), 不是 ws.name (带时间戳).
    _trace_task_id = _infer_task_id_from_workspace(ctx.task_id)
    # v14 Task 15: 跨 task darwin prior — 取本 domain 历史 high darwin entry,
    # 传给 HintCoordinator boost 当前 hint. 跨 domain 隔离在 CrossTaskStore 层做.
    _trace_domain = _infer_domain(_trace_task_id)
    _cross_task_prior = (
        _cross_task_store.query_high_darwin(domain=_trace_domain, top_k=5)
        if _cross_task_store is not None else []
    )
    return _S2State(
        ws=ctx.ws,
        model=ctx.model,
        agent=ctx.agent,
        kb=ctx.kb,
        mem_mgr=ctx.mem_mgr,
        persona=ctx.persona,
        kg=ctx.kg,
        thread_id=ctx.thread_id,
        task_id=ctx.task_id,
        trace_task_id=_trace_task_id,
        trace_domain=_trace_domain,
        cross_task_prior=_cross_task_prior,
        resume_from_iter=ctx.resume_from_iter,
        extreme=ctx.extreme,
        checklist=ctx.checklist,
        scan_text=ctx.scan_text,
        fcm=ctx.fcm,
        target_chains=ctx.target_chains,
        instructions=ctx.instructions,
        persona_name=ctx.persona_name,
    )


def _s2_prepare(ctx: _RCBStep2Ctx) -> _S2State:
    """Step 2 setup: prompt 组装 + 循环/认知层初始化 (顺序同原实现)."""
    st = _s2_make_state(ctx)
    print("\n=== Step 2: Execution (iterative) ===\n", flush=True)
    ctx.rcb_csm_advance_fn("user_confirmed", {"plan": "execute methodology checklist"})
    st.scan_hint = _s2_build_scan_hint(st.scan_text)
    st.fcm_hint = _s2_build_fcm_hint(st.fcm)
    _s2_build_step2_prompt(st)
    _s2_init_trace(st)
    _s2_init_loop_timers(st)
    _s2_init_auditors(st)
    _s2_init_fork_state(st)
    _s2_init_eval_state(st)
    _s2_init_task_metrics(st)
    _s2_init_heat(st)
    _s2_init_manifold_state(st)
    _s2_init_mcmc(st)
    _s2_init_completion_state(st)
    _s2_init_self_model(st)
    _s2_init_hypothesis_manifold(st)
    _s2_init_bandit(st)
    _s2_init_loop_counters(st)
    return st


def _s2_begin_iteration(st: _S2State, iter_n: int) -> None:
    """每轮开头: derivation_audit 归零 + 更新 LLM likelihood iter counter."""
    # ponytail: v16 引入 _derivation_audit 只在 else 分支赋值, iter 0
    # 走 if 分支跳过赋值, 引用时 UnboundLocalError 崩溃.
    # 在 for 开头初始化, iter 0 默认空串不触发 derivation gate.
    st.derivation_audit = ""
    # v15 Phase 3: 更新 LLM likelihood 的 iter counter (决定本轮是否调 LLM)
    _llm_lik_handle = getattr(st.hypo_manifold, "_llm_likelihood", None)
    if _llm_lik_handle is not None:
        _llm_lik_handle.iter_n = iter_n


def _s2_wall_expired(ctx: _RCBStep2Ctx, iter_n: int) -> bool:
    """P5 wall_clock 守卫: 挂钟预算耗尽就停, 优先级高于其他早停条件."""
    if ctx.wall_expired_fn and ctx.wall_expired_fn():
        print(f"[P5] wall_clock budget expired at iter {iter_n}, stopping.", flush=True)
        return True
    return False


async def _s2_build_iter_prompt(ctx: _RCBStep2Ctx, st: _S2State, iter_n: int) -> None:
    """本轮 prompt: iter 0 用 step2_prompt, 否则 HintCoordinator 续写 + merge hint."""
    if iter_n == 0:
        st.iter_prompt = st.step2_prompt
    else:
        st.iter_prompt, st.derivation_audit = await _build_continuation_prompt(
            ctx,
            iter_n=iter_n,
            max_iters=st.max_exec_iters,
            evals_history=st.evals_history,
            last_step_eval=st.last_step_eval,
            trace_domain=st.trace_domain,
            trace_task_id=st.trace_task_id,
            hint_coord_enabled=st.hint_coord_enabled,
            hint_coord=st.hint_coord,
            drift_info=st.drift_info,
            manifold=st.hypo_manifold,
            iter_observations=st.iter_observations,
            trace_history=st.trace_history,
            mcmc_engine=st.mcmc_engine,
            step2_prompt_base=st.step2_prompt_base,
            cross_task_prior=st.cross_task_prior,
        )
    st.iter_prompt += st.merge_hint
    st.merge_hint = ""


def _s2_inject_stagnation_hints(st: _S2State, iter_n: int) -> None:
    """P6 文件反复重写 pivot 提示 + P1-C 上轮完成度 gap 提示."""
    # ponytail: 不阻断主流程, 只注入提示. 升级路径: 配合 darwin_score 无提升
    #   才触发, 当前 iter 边界 darwin_score 可能未算完, 先纯文件名触发.
    if iter_n >= 1:
        _fr_stuck, _fr_msg = _detect_file_rewrite_stagnation(st.ws / "code")
        if _fr_stuck:
            st.iter_prompt += "\n\n" + _fr_msg + "\n"
            print(f"[iter {iter_n}] {_fr_msg}", flush=True)
    # P1-C: 上轮完成度审计的 gap 提示注入 — advisory, 不阻断.
    if st.prev_completion_hint:
        st.iter_prompt += st.prev_completion_hint


def _s2_inject_imagery_and_blindspot(st: _S2State, iter_n: int) -> None:
    """P1 想象力 directive + v15 Phase 5 Task 11 blind_spot hint (均失败降级)."""
    # 触发条件: Re_cog > Re_crit (概念湍流) 或 T_hot > 0.7 (高熵).
    # P0-B: 加 HUGINN_USE_MENTAL_IMAGERY=1 守卫, 非 extreme 模式默认关.
    if st.heat_engine is not None and os.environ.get("HUGINN_USE_MENTAL_IMAGERY", "0") == "1":
        st.iter_prompt = _maybe_inject_imagery(
            st.heat_engine,
            iter_n,
            st.hypo_manifold,
            st.iter_best_h_id,
            st.checklist,
            st.iter_prompt,
            st.kb,
            st.imagination_block,
        )
    # 失败降级空串, 不阻塞. iter 0 跳过 (SelfModel 还没工具调用记录).
    if iter_n > 0 and st.self_model is not None:
        try:
            from huginn.metacog.blind_spot_mapper import (
                infer_blind_spots as _infer_bs,
            )
            from huginn.metacog.blind_spot_mapper import (
                map_blind_spots_to_hint as _map_bs_hint,
            )
            _bs_list = _infer_bs(st.self_model)
            _bs_hint = _map_bs_hint(_bs_list, max_n=3)
            if _bs_hint:
                st.iter_prompt += "\n\n" + _bs_hint
                _n_high = sum(1 for b in _bs_list if b.priority == "high")
                print(
                    f"[v15] blind_spot hint injected: "
                    f"{len(_bs_list)} spots ({_n_high} high)",
                    flush=True,
                )
        except Exception as _be:
            print(f"[v15] blind_spot hint skipped: {_be}", flush=True)


def _s2_inject_context(st: _S2State, iter_n: int) -> None:
    """Task 18 / G66 上下文块 + curiosity + deliverable + v16 derivation 审计注入."""
    # ponytail: 直接拼到 _iter_prompt 末尾 — fork / 主路径都吃同一份 prompt.
    st.iter_prompt, st.fired = _inject_context_blocks(
        st.iter_prompt,
        agent=st.agent,
        mem_mgr=st.mem_mgr,
        target_chains=st.target_chains,
        iter_n=iter_n,
        last_step_eval=st.last_step_eval,
        kg=st.kg,
        drift_info=st.drift_info,
        persona=st.persona,
        kb=st.kb,
    )
    st.iter_prompt = _inject_curiosity_hint(st.iter_prompt, st.mem_mgr)
    st.iter_prompt = _inject_deliverable_hint(st.iter_prompt, st.ws, iter_n)
    # v16: 有 DISCUSSED_ONLY 段时强制注入 MISSING DERIVATION 提示.
    if st.derivation_audit and "DISCUSSED_ONLY:" in st.derivation_audit:
        st.iter_prompt += (
            "\n\n## DERIVATION CHAIN AUDIT (v16 gate — agent repeatedly "
            "discusses checklist items in report without executing them)\n"
            + st.derivation_audit
            + "\n\n→ Do NOT just rewrite report sections. PRODUCE the missing "
            "artifacts in outputs/ first (e.g. outputs/imo_results.json with "
            "your own run results, NOT report citing paper values). "
            "TASK COMPLETE will be BLOCKED until artifacts exist."
        )


async def _s2_run_iteration_llm(
    ctx: _RCBStep2Ctx, st: _S2State, iter_n: int
) -> None:
    """TFM 单轨迹 / k 路分叉执行 + 退火降温 + 连续驳回计数清零."""
    _k = anneal_fork_count(st.t_hot, st.fork_k_max) if st.fork_enabled else 1
    _tree = st.agent._conversation_tree
    _branch_point = _tree.active_leaf_id if _tree is not None else None
    if _branch_point is None:
        _k = 1  # 树是空的没法分叉 (Step 1 都失败了)

    if _k > 1:
        st.ai_text, st.merge_hint, st.thread_id = await _run_fork_iteration(
            ctx,
            iter_n=iter_n,
            max_iters=st.max_exec_iters,
            k=_k,
            t_hot=st.t_hot,
            iter_prompt=st.iter_prompt,
            thread_id=st.thread_id,
            branch_point=_branch_point,
            trace_task_id=st.trace_task_id,
            trace_path=st.trace_path,
        )
    else:
        print(f"\n--- Step 2 iter {iter_n + 1}/{st.max_exec_iters} ---\n", flush=True)
        st.ai_text = await ctx.stream_chat_fn(
            st.iter_prompt, f"step2_iter{iter_n + 1}")

    # 退火降温 (在停滞重热之前, 停滞信号下一轮生效)
    st.t_hot = max(0.0, st.t_hot * 0.5)
    # agent 这轮没说 TASK COMPLETE → 连续驳回计数清零. agent 做了实际工作,
    # 之前的驳回不算"陷入循环". ponytail: 字符串匹配, 简单粗暴.
    if not (st.ai_text and "TASK COMPLETE" in st.ai_text.upper()):
        st.consecutive_complete_rejections = 0


def _s2_collect_observations(st: _S2State, iter_n: int) -> None:
    """v15 Phase 2 Task 3.2+3.3: 收集 observations + abductive inference."""
    # 失败降级到空 observations + 不写 abduction entry, 不阻塞主循环.
    st.iter_observations = []
    st.iter_best_h_id = None
    st.iter_log_post = 0.0
    st.iter_fisher_info = 0.0
    if st.hypo_manifold is not None:
        (
            st.iter_observations,
            st.iter_best_h_id,
            st.iter_log_post,
            st.iter_fisher_info,
            st.last_abduction_result,
        ) = _collect_iter_observations_and_abduction(
            manifold=st.hypo_manifold,
            ai_text=st.ai_text or "",
            ws=st.ws,
            checklist=st.checklist or "",
            obs_path=st.hypo_obs_path,
            manifold_path=st.hypo_manifold_path,
            trace_path=st.trace_path,
            trace_task_id=st.trace_task_id,
            iter_n=iter_n,
        )


def _s2_run_imagination(st: _S2State, iter_n: int) -> None:
    """v15 Phase 4 Task 8 stagnation 触发 + Phase 5 Task 12.2 blind_spot 触发."""
    # best h_id 连续 3 轮不变 -> 在 manifold 上做 structure-preserving transform
    if st.iter_best_h_id is not None and st.hypo_manifold is not None:
        st.stagnation_history.append(st.iter_best_h_id)
        # P4 Task 23: cycle_detect 顶层函数接入 — hypothesis id 序列卡顿检测.
        _announce_hypothesis_cycle(st.stagnation_history)
        _run_stagnation_imagination(
            st.stagnation_history,
            st.iter_best_h_id,
            st.hypo_manifold,
            st.model,
            iter_n=iter_n,
            task_id=st.trace_task_id,
            log_path=st.imagination_log_path,
            manifold_path=st.hypo_manifold_path,
            trace_path=st.trace_path,
        )
    # 选 high priority blind spot 作种子, 每 3 轮试一次, 避免 LLM 调用 spam.
    _run_blind_spot_imagination(
        st.self_model,
        st.hypo_manifold,
        st.model,
        iter_n=iter_n,
        log_path=st.imagination_log_path,
        manifold_path=st.hypo_manifold_path,
    )


def _s2_write_trace(st: _S2State, iter_n: int) -> None:
    """写 Meta-Trace entry (P1 下一轮 build_meta_trace_text 会读到) + bandit iter-end."""
    # ponytail: 字段从 self/agent 状态抽, 不调 LLM. RCB mini-loop 不跑 darwin
    #   ratchet, darwin_score 用 _compute_darwin_score, supported_ratio 见下方.
    st.report_text, st.report_path_iter, st.entry = _write_meta_trace_entry(
        ws=st.ws,
        ai_text=st.ai_text,
        iter_prompt=st.iter_prompt,
        iter_n=iter_n,
        max_iters=st.max_exec_iters,
        trace_history=st.trace_history,
        last_step_eval=st.last_step_eval,
        best_h_id=st.iter_best_h_id,
        log_post=st.iter_log_post,
        fisher_info=st.iter_fisher_info,
        trace_task_id=st.trace_task_id,
        shard_writer=st.meta_trace_writer,
        shard=st.meta_trace_shard,
        betti_path=st.betti_path,
        compute_betti=st.compute_betti,
    )
    # v18: bandit iter-end — reward_slow = β * Δdarwin_score.
    # ponytail: 失败静默, bandit 内部 catch. _entry 可能未定义, 用 get 兜.
    try:
        if st.budget_items:
            from huginn.agent.bandit_controller import EffortBandit

            EffortBandit.get_instance().update_iter_end(
                float(_entry_get(st.entry, "darwin_score", 0.5))
            )
    except Exception:
        logger.debug("bandit iter-end update skipped", exc_info=True)


async def _s2_evaluate(
    ctx: _RCBStep2Ctx, st: _S2State, iter_n: int
) -> None:
    """StepEvaluator 评估 + 回填 last_step_eval / drift / metrics / merge hint."""
    _eval_outcome = await _evaluate_step_phase(
        ctx,
        iter_n=iter_n,
        entry=st.entry,
        target_chains=st.target_chains,
        self_model=st.self_model,
        heat_engine=st.heat_engine,
        evals_history=st.evals_history,
        task_metrics=st.task_metrics,
        task_state_for_metrics=st.task_state_for_metrics,
        step2_prompt=st.step2_prompt,
        iter_prompt=st.iter_prompt,
        merge_hint=st.merge_hint,
        trace_task_id=st.trace_task_id,
        trace_path=st.trace_path,
    )
    if _eval_outcome.step_eval is not None:
        st.last_step_eval = _eval_outcome.step_eval  # Task 18: 供下一轮 prompt 注入
    st.drift_info = _eval_outcome.drift_info
    st.task_metrics = _eval_outcome.task_metrics
    st.merge_hint = _eval_outcome.merge_hint


def _s2_pause(ctx: _RCBStep2Ctx, st: _S2State, iter_n: int) -> None:
    """Task 29 人机协同 pause 检查 (G71); 默认自动 resume."""
    # ponytail: RCB 是自动化测试环境, 真人在环是 v8 升级路径.
    st.pmk_state, st.prev_darwin = _maybe_pause_for_human(
        evals_history=st.evals_history,
        target_chains=st.target_chains,
        kb=st.kb,
        fired=st.fired,
        persona=st.persona,
        last_step_eval=st.last_step_eval,
        run_start_iso=ctx.run_start_iso,
        mem_mgr=st.mem_mgr,
        entry=st.entry,
        prev_darwin=st.prev_darwin,
        iter_n=iter_n,
        task_id=st.task_id,
        trace_task_id=st.trace_task_id,
        ws=st.ws,
        trace_path=st.trace_path,
    )


def _s2_save_checkpoint(st: _S2State, iter_n: int) -> None:
    """Checkpoint 落盘 (G59) + 跨任务 iteration_result 写入."""
    # P4 Task 26.6: 补 engine_state_digest (darwin_score + supported_ratio hash),
    # resume 时 resume_from_checkpoint 会校验防 drift.
    _save_loop_checkpoint(
        entry=st.entry,
        ai_text=st.ai_text,
        iter_n=iter_n,
        target_chains=st.target_chains,
        prospective_mem=st.prospective_mem,
        task_id=st.task_id,
        ws=st.ws,
    )
    # status 由 darwin_score 映射: >=0.5 supported, <0.5 refuted.
    # ponytail: 只在 cross-task db 开时写, 单任务模式跳过 (无读者).
    _record_cross_task_iteration(
        mem_mgr=st.mem_mgr,
        entry=st.entry,
        iter_n=iter_n,
        ws_name=st.ws.name,
        persona_name=st.persona_name,
    )


def _s2_update_heat(st: _S2State, iter_n: int) -> None:
    """停滞重热 + score_history 滑窗登记 (均只监测, 不早退)."""
    # C7: 删 stagnation early-stop — wall_clock 守卫唯一控制退出.
    st.prev_report_hash, st.stagnation_count, st.t_hot = _update_stagnation_heat(
        report_text=st.report_text,
        prev_hash=st.prev_report_hash,
        stagnation_count=st.stagnation_count,
        t_hot=st.t_hot,
    )
    # C7: 早退已删, 仅记 stats + 日志.
    _push_score_history(st.score_history, st.entry, iter_n)


def _s2_run_gates(st: _S2State, iter_n: int) -> bool:
    """早停门: cumulative audit 越权 / darwin ratchet action=stop. True 表示 break."""
    if _run_cumulative_audit(
        st.cumulative_auditor, st.ws, st.audit_task_type, iter_n
    ):
        return True
    # advisory only — pivot/counterexample 不 break, 只有 action=stop 才 break.
    _action, st.stagnation_count = _run_darwin_ratchet(
        entry=st.entry,
        stagnation_count=st.stagnation_count,
        iter_n=iter_n,
    )
    return _action == "stop"


async def _s2_run_audits(st: _S2State, iter_n: int) -> None:
    """P1-A MCMC + P1-C 完成度审计 + 触觉层同构异常检测."""
    # ponytail: MCMC 是 advisory only, 不强制 agent 用 MCMC 选的 h.
    st.mcmc_cached_log_p, st.mcmc_current_info = _run_mcmc_step(
        manifold=st.hypo_manifold,
        observations=st.iter_observations,
        iter_n=iter_n,
        best_h_id=st.iter_best_h_id,
        engine=st.mcmc_engine,
        cached_log_p=st.mcmc_cached_log_p,
        interval=st.mcmc_interval,
        checkpoint_interval=st.mcmc_ckpt_interval,
        haptic_enabled=st.mcmc_haptic_enabled,
        haptic_temperature=st.mcmc_haptic_temperature,
        alignment_enabled=st.mcmc_alignment_enabled,
        alignment_temperature=st.mcmc_alignment_temperature,
        task_id=st.task_id,
        ws=st.ws,
    )
    # P1-C: 每 _completion_interval 步强制跑一次, gap 写 cognitive_evidence.md.
    st.completion_audit, st.prev_completion_hint = _run_completion_audit(
        ws=st.ws,
        report_text=st.report_text,
        report_path=st.report_path_iter,
        iter_n=iter_n,
        interval=st.completion_interval,
        prev_hint=st.prev_completion_hint,
    )
    # 触觉层: 同构不同性异常检测 (advisory only, 失败只 warn 不阻塞).
    st.anomaly_info = await _detect_anomalies(
        st.hypo_manifold, st.model, iter_n, st.completion_interval
    )


def _s2_write_evidence(st: _S2State, iter_n: int) -> None:
    """P0-C: 每 iter 结束写 cognitive_evidence.md (追加写, 失败只 warn)."""
    _write_cognitive_evidence(
        st.ws,
        iter_n,
        entry=st.entry,
        pmk_state=st.pmk_state,
        hypo_manifold=st.hypo_manifold,
        heat_engine=st.heat_engine,
        bandit_controller=st.bandit,
        mcmc_info=st.mcmc_current_info,
        completion_audit=st.completion_audit,
        anomaly_info=st.anomaly_info,
    )


async def _s2_handle_complete(
    ctx: _RCBStep2Ctx, st: _S2State, iter_n: int
) -> bool:
    """TASK COMPLETE 收尾 (effort floor + 反完成审计). True 表示 break."""
    # P5 守卫: wall_clock 未耗尽才允许 TASK COMPLETE break.
    if not (st.ai_text and "TASK COMPLETE" in st.ai_text.upper()):
        return False
    if ctx.wall_expired_fn and ctx.wall_expired_fn():
        return False
    # v16: TASK COMPLETE 时强制重跑 derivation audit (最后一次放行机会).
    (
        st.consecutive_complete_rejections,
        _accepted,
        _override,
    ) = await _handle_task_complete(
        ctx,
        iter_n=iter_n,
        max_iters=st.max_exec_iters,
        derivation_audit=st.derivation_audit,
        report_text=st.report_text,
        report_path=st.report_path_iter,
        iter_observations=st.iter_observations,
        entry=st.entry,
        pmk_state=st.pmk_state,
        manifold=st.hypo_manifold,
        heat_engine=st.heat_engine,
        bandit=st.bandit,
        completion_audit=st.completion_audit,
        mcmc_current_info=st.mcmc_current_info,
        anomaly_info=st.anomaly_info,
        rejections=st.consecutive_complete_rejections,
        max_rejections=st.max_complete_rejections,
    )
    if _override is not None:
        st.iter_prompt = _override
    return _accepted


async def _s2_iteration(
    ctx: _RCBStep2Ctx, st: _S2State, iter_n: int
) -> bool:
    """单轮 Step 2: prompt → LLM → 观测 → 评估 → 审计. True 表示停止循环."""
    _s2_begin_iteration(st, iter_n)
    if _s2_wall_expired(ctx, iter_n):
        return True
    await _s2_build_iter_prompt(ctx, st, iter_n)
    _s2_inject_stagnation_hints(st, iter_n)
    _s2_inject_imagery_and_blindspot(st, iter_n)
    _s2_inject_context(st, iter_n)
    await _s2_run_iteration_llm(ctx, st, iter_n)
    _s2_collect_observations(st, iter_n)
    _s2_run_imagination(st, iter_n)
    _s2_write_trace(st, iter_n)
    await _s2_evaluate(ctx, st, iter_n)
    _s2_pause(ctx, st, iter_n)
    _s2_save_checkpoint(st, iter_n)
    _s2_update_heat(st, iter_n)
    if _s2_run_gates(st, iter_n):
        return True
    await _s2_run_audits(st, iter_n)
    _s2_write_evidence(st, iter_n)
    return await _s2_handle_complete(ctx, st, iter_n)


async def _step2_execute(ctx: _RCBStep2Ctx) -> list:
    """Step 2: 执行任务 — setup + 迭代执行循环.

    从 run() 抽出, 逻辑保持不变. 闭包 _stream_chat / _rcb_csm_advance 通过 ctx
    传入; 循环状态收拢在 _S2State, 各阶段函数按原顺序调用.
    返回 _evals_history 供 Step 3 的 _step3_adversarial 使用.
    """
    # 对标 Oxelra 206 步: 单次 chat() 已能跑 150-300 tool calls, 但会因 context
    # 溢出或 agent 主动 emit text-only 提前终止. mini-loop 让 agent 在多次 chat()
    # 间累积进展, 每轮间写 Meta-Trace entry 供下一轮 prompt 读回.
    #
    # ponytail: 不接 AutoloopEngine (它用 CoderRunner/WorkflowEngine, 不写
    #   report/report.md, 会破坏 RCBench 评分). 用 mini-loop + 手写 trace.
    st = _s2_prepare(ctx)
    for _iter_n in range(st.resume_from_iter, st.max_exec_iters):
        if await _s2_iteration(ctx, st, _iter_n):
            break
    return st.evals_history
