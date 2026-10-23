"""认知环路的共享元认知护航原语 (AV4 路径 B).

从 ``cognitive_loop.py`` (198KB) 拆出的段 2: 4 个无状态共享函数, 被 rcb_runner
mini-loop 和 autoloop run_cognitive 两条路径共用. 拆出后 cognitive_loop 通过
import 再导出, 对外契约不变 (``from huginn.autoloop.cognitive_loop import build_pmk_state``
仍可用).

拆出原则 (同段 1):
- 只搬"给定参数 → 返回结果 / 就地更新"的无状态函数, 无 self / 无 engine 状态读写.
- 只做 duck-typing (step_eval / heat_engine / kb 等), 不引入 CognitiveLoop 子类, 不绑定 LLM.
- 文件 I/O / 记忆 / 指标等有副作用者留在 cognitive_loop (段 3+ 再处理).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from huginn.utils.runtime import get_runtime_home

logger = logging.getLogger(__name__)


def update_heat_engine_after_step(
    heat_engine: Any,
    step_eval: Any,
    prompt_len: int,
    idea_count: int,
    *,
    stable_principles_count: int = 1,
) -> None:
    """AV4: heat_engine 闭环共享函数 — StepEvaluation → T_hot/T_cold/kinematics.

    rcb_runner (AV8) 和 autoloop run_cognitive 都调这个, 避免 4 档映射逻辑两边各写一份.
    T_hot: evidence_quality 熵代理 (low=0.8 / medium=0.5 / high=0.2 / unknown=0.5).
    T_cold: on_track 秩序代理 (true=0.7 / false=0.2 / unsure=0.4).
    ponytail: 4 档离散映射, 天花板: 跳变不连续; 升级路径接连续 evidence_score.
    """
    if heat_engine is None:
        return
    try:
        _eq = (getattr(step_eval, "evidence_quality", "unknown") or "unknown").lower().strip()
        _t_hot_proxy = {"low": 0.8, "medium": 0.5, "high": 0.2}.get(_eq, 0.5)
        _ot = (getattr(step_eval, "on_track", "unsure") or "unsure").lower().strip()
        _t_cold_proxy = {"true": 0.7, "false": 0.2}.get(_ot, 0.4)
        heat_engine.update_T_hot(_t_hot_proxy)
        heat_engine.update_T_cold(_t_cold_proxy, darwin_score=0.0)
        heat_engine.update_kinematics(
            idea_count=idea_count,
            stable_principles_count=stable_principles_count,
            system_prompt_len=prompt_len,
        )
    except Exception as exc:
        logger.debug("update_heat_engine_after_step failed: %s", exc)


def update_drift_and_metrics(
    evals_history: list,
    step_eval: Any,
    task_metrics: Any,
    task_state: Any,
    workspace: Any,
    run_id: str,
    max_iterations: int,
    target_chain_progress: float | None = None,
) -> tuple[tuple | None, Any]:
    """AV4: detect_drift + TaskMetrics 滚动更新共享函数.

    返回 (drift_info, task_metrics). 失败时 drift_info=None, task_metrics 不变.
    rcb_runner (G62+G70) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: duck typing — step_eval 可以是 StepEvaluation 或 SimpleNamespace.
    target_chain_progress: RCB 路径算的 chain 平均进度, 透传给 update_metrics.
      autoloop 不传 (默认 None), 行为不变.
    """
    drift_info: tuple | None = None
    try:
        from huginn.metacog.target_chain import detect_drift as _detect_drift
        drift_info = _detect_drift(evals_history, window=3)
    except Exception as exc:
        logger.debug("AV4 detect_drift failed: %s", exc)

    try:
        from huginn.runtime.task_metrics import (
            TaskMetrics,
        )
        from huginn.runtime.task_metrics import (
            load_metrics as _lm,
        )
        from huginn.runtime.task_metrics import (
            save_metrics as _sm,
        )
        from huginn.runtime.task_metrics import (
            update_metrics as _um,
        )
        if task_metrics is None:
            import time
            task_metrics = _lm(run_id, workspace) or TaskMetrics(
                task_id=run_id, total_steps=max_iterations * 6,
            )
            if task_state is None:
                from types import SimpleNamespace
                task_state = SimpleNamespace(created_at=time.time())
        # target_chain_progress 透传 — update_metrics 接受 None (autoloop 路径).
        task_metrics = _um(
            task_metrics, step_eval,
            task_state=task_state,
            target_chain_progress=target_chain_progress,
        )
        _sm(task_metrics, workspace)
    except Exception as exc:
        logger.debug("AV4 TaskMetrics update failed: %s", exc)

    return drift_info, task_metrics


def build_pmk_state(
    persona: Any,
    last_step_eval: Any,
    kb: Any,
    *,
    top_k: int = 2,
    since: str | None = None,
    mem_mgr: Any = None,
    timeseries_ctx: str | None = None,
) -> dict[str, str] | None:
    """AV4: PMK 三路立场状态构建共享函数.

    persona: PersonaManager 返回的对象或 dict, 取 description.
    last_step_eval: 上一步 StepEvaluation, 取 pmk_feedback 中 Memory 段.
    kb: 知识库, 用 last_step_eval.attempted 查 top_k hits 取前 200 字.
    since: 时序窗口 — KB query 和 Memory recall 都限定到该时间之后,
        让 PMK 聚焦本轮/本会话的新知识, 避免旧文献值跟新实验冲突被误判.
    mem_mgr: MemoryManager, 传入时补一次 recall(since=...) 作为 M 路时序补充.
    timeseries_ctx: P3 物理时序摘要文本, 传入时作为第四路 "timeseries" 加入.
        让 PMK 一致性检查能感知系统动力学趋势 (比如 VACF 衰减但 persona
        说束缚态 → 冲突). ponytail: 纯文本, 跟其他路同格式.
    返回 {"persona","memory","kb"[,"timeseries"]} 或 None (全空时).
    rcb_runner (P0-A) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: 三/四段文本拼接, 不上 LLM 抽取; 升级路径接 LLM 立场抽取.
    """
    try:
        persona_text = _pmk_persona_text(persona)
        mem_text = _pmk_enrich_memory(
            _pmk_memory_text(last_step_eval), mem_mgr, last_step_eval, since
        )
        kb_text = _pmk_enrich_kb(_pmk_kb_text(kb, last_step_eval, top_k, since))
        return _pmk_assemble(persona_text, mem_text, kb_text, timeseries_ctx)
    except Exception as exc:
        logger.debug("AV4 build_pmk_state failed: %s", exc)
    return None


def _pmk_persona_text(persona: Any) -> str:
    """Extract the persona description as the P stance."""
    if persona is None:
        return ""
    return str(
        getattr(persona, "description", None)
        or (persona.get("description") if isinstance(persona, dict) else "")
        or ""
    )


def _pmk_memory_text(last_step_eval: Any) -> str:
    """Extract the "memory:" segment of the previous step's pmk_feedback."""
    if last_step_eval is None:
        return ""
    pmk_fb = getattr(last_step_eval, "pmk_feedback", "") or ""
    for seg in pmk_fb.split(";"):
        seg = seg.strip()
        if seg.lower().startswith("memory:"):
            return seg[len("memory:"):].strip()
    return ""


def _pmk_enrich_memory(
    mem_text: str, mem_mgr: Any, last_step_eval: Any, since: str | None
) -> str:
    """Append recent / failed-direction / procedural recalls to the M stance."""
    if mem_mgr is None:
        return mem_text
    mem_text = _pmk_recall_recent(mem_text, mem_mgr, last_step_eval, since)
    mem_text = _pmk_recall_failed(mem_text, mem_mgr)
    return _pmk_recall_procedural(mem_text, mem_mgr, last_step_eval)


def _pmk_recall_recent(
    mem_text: str, mem_mgr: Any, last_step_eval: Any, since: str | None
) -> str:
    """时序结合: M 路补 recall(since=...) — 让 PMK 看到本会话新写入的记忆.

    不只是 last_step_eval 的 pmk_feedback. ponytail: 失败静默, 不阻塞 PMK.
    """
    if not since:
        return mem_text
    try:
        ts_mem = mem_mgr.recall_for_prompt(
            getattr(last_step_eval, "attempted", "") or "",
            max_entries=2,
            since=since,
        )
        if not ts_mem:
            return mem_text
        return (mem_text + " | recent: " + ts_mem[:200]).strip()
    except Exception:  # 防御: 记忆召回失败跳过
        logger.debug("memory recall_for_prompt skipped", exc_info=True)
        return mem_text


def _pmk_recall_failed(mem_text: str, mem_mgr: Any) -> str:
    """桥 H: M 路补 failed_directions — 感知"persona 推荐方向 vs 已知失败方向".

    顺带接通 recall_procedural 死代码 (procedural memory = stable_principles
    关键词召回).
    """
    try:
        failed = mem_mgr.recall_failed_directions(limit=3)
        if not failed:
            return mem_text
        lines = [f"{h[:60]} ({r[:40]})" for h, r, _ in failed if h]
        return (mem_text + " | failed_directions: " + " ; ".join(lines)).strip()
    except Exception:  # 防御: 失败方向召回跳过
        logger.debug("failed_directions recall skipped", exc_info=True)
        return mem_text


def _pmk_recall_procedural(
    mem_text: str, mem_mgr: Any, last_step_eval: Any
) -> str:
    """M 路补 procedural — 接通 recall_procedural 死代码."""
    try:
        attempted = getattr(last_step_eval, "attempted", "") or ""
        proc = mem_mgr.recall_procedural(attempted, top_k=2)
        if not proc:
            return mem_text
        return (
            mem_text + " | procedural: " + " ; ".join(p[:60] for p in proc)
        ).strip()
    except Exception:  # 防御: 程序性召回跳过
        logger.debug("procedural recall skipped", exc_info=True)
        return mem_text


def _pmk_kb_text(
    kb: Any, last_step_eval: Any, top_k: int, since: str | None
) -> str:
    """KB retrieval via the dedup path when available, first 200 chars."""
    if kb is None or last_step_eval is None:
        return ""
    try:
        query = getattr(last_step_eval, "attempted", "") or ""
        # 桥 RAG: 优先走去重的检索 (query_with_dedup), 吃到 RAG 升级后的
        # BM25 中文分词 + 近似去重, 跟 _build_kb_text 同一检索路径.
        # 无该方法时回退旧 query, 不破坏既有调用方.
        hits = (
            kb.query_with_dedup(query, top_k=top_k, since=since)
            if hasattr(kb, "query_with_dedup")
            else kb.query(query, top_k=top_k, since=since)
        )
        if not hits:
            return ""
        return " ".join(
            str(h.get("content", "") if isinstance(h, dict) else h)
            for h in hits[:top_k]
        )[:200]
    except Exception:  # 防御: 知识库查询失败跳过
        logger.debug("kb query skipped", exc_info=True)
        return ""


def _prepend_pmk_block(base: str, tag: str, body: str) -> str:
    """Prefix a labelled block to the KB stance when the body is non-empty."""
    if not body:
        return base
    return (f"{tag} " + body + " | " + base).strip(" |")


def _evolution_action_desc(rule: dict) -> str:
    """action 在 EvolutionRule 里是 str, 但历史数据可能是 dict. 两种都处理."""
    action = rule.get("action", "")
    if isinstance(action, dict):
        return str(action.get("description", ""))[:80]
    return str(action)[:80]


def _pmk_learned_lessons() -> str:
    """桥 C: top-3 high-confidence evolution rules as a "lesson" string.

    让 PMK 一致性检查能看到历史教训, 不只 ChromaDB 检索.
    路径对齐 context_builder.build_evolution_rules.
    """
    try:
        rules_path = Path(str(get_runtime_home())) / "logs" / "evolution_rules.json"
        if not rules_path.exists():
            return ""
        with rules_path.open("r", encoding="utf-8") as rf:
            rules = json.load(rf)
        if not isinstance(rules, list):
            return ""
        top_rules = sorted(
            rules, key=lambda r: r.get("confidence", 0), reverse=True
        )[:3]
        lessons = [
            f"{r.get('rule_id','')}: {_evolution_action_desc(r)}"
            for r in top_rules
            if r.get("confidence", 0) > 0.3
        ]
        return " | ".join(lessons)
    except Exception:  # 防御: 演化规则加载失败跳过
        logger.debug("evolution_rules load skipped", exc_info=True)
        return ""


def _pmk_stable_principles() -> str:
    """桥 D: distilled long-term principles as a KB stance."""
    try:
        from huginn.memory.longterm import load_stable_principles

        principles = load_stable_principles()
        if not principles:
            return ""
        return " | ".join(str(p)[:80] for p in principles[:3] if p)
    except Exception:  # 防御: 原则加载失败跳过
        logger.debug("stable_principles load skipped", exc_info=True)
        return ""


def _pmk_enrich_kb(kb_text: str) -> str:
    """Bridge C/D: prepend learned lessons then stable principles to KB."""
    kb_text = _prepend_pmk_block(
        kb_text, "[learned_lessons]", _pmk_learned_lessons()
    )
    return _prepend_pmk_block(
        kb_text, "[stable_principles]", _pmk_stable_principles()
    )


def _pmk_assemble(
    persona_text: str,
    mem_text: str,
    kb_text: str,
    timeseries_ctx: str | None,
) -> dict[str, str] | None:
    """Assemble the non-empty stances; None when everything is empty."""
    result: dict[str, str] = {}
    if persona_text:
        result["persona"] = persona_text
    if mem_text:
        result["memory"] = mem_text
    if kb_text:
        result["kb"] = kb_text
    # P3 结合: 物理时序作为第四路 (PMKT) — 让一致性检查感知系统动力学.
    if timeseries_ctx:
        result["timeseries"] = timeseries_ctx
    return result or None




def check_pause_decision(
    evals_history: list,
    target_chains: list,
    kb: Any,
    fired_intentions: list | None,
    pmk_state: dict[str, str] | None,
    grill_state: dict | None = None,
    iteration_history: list[dict] | None = None,
) -> tuple[bool, str, list]:
    """AV4: should_pause_for_decision 共享包装.

    返回 (pause, reason, opts). 失败时 (False, "", []).
    rcb_runner (G71) 和 autoloop reflect_fn (AV2) 都调这个.
    ponytail: 只调判定, 不做 lifecycle/resume — 那些动作两条路径不同, 留在 caller.

    grill_state (P0 grill-me) 由 caller 在 plan_check 阶段构造:
    {"has_grilled": bool, "ambiguity_score": float, "tier": str,
     "scene_tag": str, "plan_is_empty": bool}.

    iteration_history (P0 时序结合): 传入时检查 PMK 冲突是否在最近几轮重复
    出现 — 重复冲突升级为 pause, 避免 agent 重复陷入同样的 PMK 障碍.
    ponytail: 只看 redirect 标记, 不解析 advice 文本. 升级路径: advice 相似度.
    """
    try:
        from huginn.runtime.task_lifecycle import (
            should_pause_for_decision as _spd,
        )
        _pause, _reason, _opts = _spd(
            evals_history, target_chains,
            kb_recall_empty=(kb is None),
            fired_intentions=fired_intentions or [],
            pmk_state=pmk_state,
            grill_state=grill_state,
        )
        # P0 时序结合: 如果本轮 PMK 冲突触发 pause, 检查 iteration_history 里
        # 最近 5 轮是否有重复 redirect — 重复说明 agent 陷入循环, 升级 reason.
        if _pause and iteration_history:
            _recent = iteration_history[-5:]
            _redirect_count = sum(1 for h in _recent if h.get("redirect"))
            if _redirect_count >= 3:
                _reason = (
                    f"{_reason} | repeated PMK conflict ({_redirect_count}/5 "
                    f"recent iterations redirected) — likely stuck in loop"
                )
        return bool(_pause), str(_reason or ""), list(_opts or [])
    except Exception as exc:
        logger.debug("AV4 check_pause_decision failed: %s", exc)
        return False, "", []
