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
        _persona_text = ""
        if persona is not None:
            _persona_text = str(
                getattr(persona, "description", None)
                or (persona.get("description") if isinstance(persona, dict) else "")
                or ""
            )
        _mem_text = ""
        if last_step_eval is not None:
            _pmk_fb = getattr(last_step_eval, "pmk_feedback", "") or ""
            for _seg in _pmk_fb.split(";"):
                _seg = _seg.strip()
                if _seg.lower().startswith("memory:"):
                    _mem_text = _seg[len("memory:"):].strip()
                    break
        # 时序结合: M 路补 recall(since=...) — 让 PMK 看到本会话新写入的记忆,
        # 不只是 last_step_eval 的 pmk_feedback. ponytail: 失败静默, 不阻塞 PMK.
        if mem_mgr is not None and since:
            try:
                _ts_mem = mem_mgr.recall_for_prompt(
                    getattr(last_step_eval, "attempted", "") or "",
                    max_entries=2, since=since,
                )
                if _ts_mem:
                    _mem_text = (_mem_text + " | recent: " + _ts_mem[:200]).strip()
            except Exception:
                logger.debug("memory recall_for_prompt skipped", exc_info=True)
        # 桥 H: M 路补 failed_directions — 让 PMK 一致性检查感知"persona 推荐
        # 的方向 vs 已知失败方向"的冲突. 顺带接通 recall_procedural 死代码
        # (procedural memory = stable_principles 关键词召回).
        if mem_mgr is not None:
            try:
                _failed = mem_mgr.recall_failed_directions(limit=3)
                if _failed:
                    _f_lines = [
                        f"{h[:60]} ({r[:40]})" for h, r, _ in _failed if h
                    ]
                    _mem_text = (
                        _mem_text + " | failed_directions: "
                        + " ; ".join(_f_lines)
                    ).strip()
            except Exception:
                logger.debug("failed_directions recall skipped", exc_info=True)
            try:
                _attempted = getattr(last_step_eval, "attempted", "") or ""
                _proc = mem_mgr.recall_procedural(_attempted, top_k=2)
                if _proc:
                    _mem_text = (
                        _mem_text + " | procedural: "
                        + " ; ".join(p[:60] for p in _proc)
                    ).strip()
            except Exception:
                logger.debug("procedural recall skipped", exc_info=True)
        _kb_text = ""
        if kb is not None and last_step_eval is not None:
            try:
                _kb_query = getattr(last_step_eval, "attempted", "") or ""
                # 桥 RAG: 优先走去重的检索 (query_with_dedup), 吃到 RAG 升级后的
                # BM25 中文分词 + 近似去重, 跟 _build_kb_text 同一检索路径.
                # 无该方法时回退旧 query, 不破坏既有调用方.
                _kb_hits = (
                    kb.query_with_dedup(_kb_query, top_k=top_k, since=since)
                    if hasattr(kb, "query_with_dedup")
                    else kb.query(_kb_query, top_k=top_k, since=since)
                )
                if _kb_hits:
                    _kb_text = " ".join(
                        str(h.get("content", "") if isinstance(h, dict) else h)
                        for h in _kb_hits[:top_k]
                    )[:200]
            except Exception:
                logger.debug("kb query skipped", exc_info=True)
        # 桥 C: evolution_rules 作为 Knowledge 路 — 让 PMK 一致性检查能看到
        # 历史教训, 不只 ChromaDB 检索. 路径对齐 context_builder.build_evolution_rules.
        try:
            _base = str(get_runtime_home())
            _rules_path = Path(_base) / "logs" / "evolution_rules.json"
            if _rules_path.exists():
                with _rules_path.open("r", encoding="utf-8") as _rf:
                    _rules = json.load(_rf)
                if isinstance(_rules, list):
                    _top_rules = sorted(
                        _rules, key=lambda r: r.get("confidence", 0), reverse=True
                    )[:3]
                    _lessons = []
                    for r in _top_rules:
                        if r.get("confidence", 0) <= 0.3:
                            continue
                        _act = r.get("action", "")
                        # action 在 EvolutionRule 里是 str, 但历史数据可能是 dict.
                        # 两种都处理, 不假设格式.
                        if isinstance(_act, dict):
                            _desc = str(_act.get("description", ""))[:80]
                        else:
                            _desc = str(_act)[:80]
                        _lessons.append(f"{r.get('rule_id','')}: {_desc}")
                    if _lessons:
                        _kb_text = (
                            "[learned_lessons] "
                            + " | ".join(_lessons)
                            + " | "
                            + _kb_text
                        ).strip(" |")
        except Exception:
            logger.debug("evolution_rules load skipped", exc_info=True)
        # 桥 D: stable_principles 作为 Knowledge 路 — 蒸馏出的原则直接作为 KB 立场,
        # 让 PMK 能检测 persona/memory 与长期原则的冲突. load_stable_principles 返回 list[str].
        try:
            from huginn.memory.longterm import load_stable_principles
            _principles = load_stable_principles()
            if _principles:
                _p_texts = [str(p)[:80] for p in _principles[:3] if p]
                if _p_texts:
                    _kb_text = (
                        "[stable_principles] "
                        + " | ".join(_p_texts)
                        + " | "
                        + _kb_text
                    ).strip(" |")
        except Exception:
            logger.debug("stable_principles load skipped", exc_info=True)
        _result: dict[str, str] = {}
        if _persona_text:
            _result["persona"] = _persona_text
        if _mem_text:
            _result["memory"] = _mem_text
        if _kb_text:
            _result["kb"] = _kb_text
        # P3 结合: 物理时序作为第四路 (PMKT) — 让一致性检查感知系统动力学.
        if timeseries_ctx:
            _result["timeseries"] = timeseries_ctx
        if _result:
            return _result
    except Exception as exc:
        logger.debug("AV4 build_pmk_state failed: %s", exc)
    return None


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