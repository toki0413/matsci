"""认知环路的共享副作用持久化原语 (段 3: run_context schema 收敛).

从 ``cognitive_loop.py`` 拆出的段 3: 只收敛"两条路径共用同一 schema"的副作用
持久化. 目前唯一满足该条件的是 run_context 探索摘要 —

- autoloop run 结束: ``CognitiveLoop._persist_run_context`` (框出 cog/state 后写入)
- RCB runner 收尾: ``learn_from_rcb`` 内联写入 (框出 hypothesis/validation 后写入)

两处写的是同一 schema (run_id/objective/hypothesis/plan_mode/outcome/iterations/
recent_steps/inconclusive), 下 run 由 ``_load_prev_run_context`` 读回. 之前两处
各写一遍, 一旦字段漂移读者就读不到. 这里抽共享 writer+reader, 让 schema 单一来源.

拆出原则 (同段 1/2):
- 只搬"给定参数 → 副作用"的模块级函数, 无 self / 无 engine 状态依赖.
- memory 用 duck-typing (有 remember/recall 即可), 不绑定 MemoryManager 类型.
- 仅收敛"多路径共用"的副作用; autoloop 独有持久化 (failure_pattern 等) 留在原地,
  不因"能抽"而抽 (ponytail: 只抽共享逻辑).
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# run_context 的 schema 契约 (writer + reader 共用, 单一来源).
_RUN_CONTEXT_KEYS: tuple[str, ...] = (
    "run_id",
    "objective",
    "hypothesis",
    "plan_mode",
    "outcome",
    "iterations",
    "recent_steps",
    "inconclusive",
)


def persist_run_context(
    memory: Any,
    *,
    run_id: str,
    objective: str,
    hypothesis: str,
    plan_mode: str,
    outcome: str,
    iterations: int,
    recent_steps: list,
    inconclusive: list,
    source: str | None = None,
) -> None:
    """P2: run 结束时存探索摘要 (category="run_context"), 供下 run 加载.

    autoloop run 结束与 learn_from_rcb (RCB 收尾) 两条路径共用此函数, 保证写侧
    schema 单一来源. 失败静默 (best-effort), 不阻塞收尾流程.

    memory: 有 remember(content=..., category=..., tags=..., importance=..., tier=...)
        方法的任意对象 (MemoryManager), None 时跳过.
    source: 额外标记写入来源 (如 "learn_from_rcb"), 可选, 不影响 reader 读取.
    """
    if memory is None:
        return
    snapshot: dict[str, Any] = {
        "run_id": run_id,
        "objective": (objective or "")[:200],
        "hypothesis": (hypothesis or "")[:300],
        "plan_mode": plan_mode,
        "outcome": outcome,
        "iterations": iterations,
        "recent_steps": recent_steps,
        "inconclusive": inconclusive,
    }
    if source:
        snapshot["source"] = source
    try:
        content = json.dumps(snapshot, ensure_ascii=False)
        memory.remember(
            content=content,
            category="run_context",
            tags=["run_context", run_id],
            importance=0.5,
            tier="mid",
        )
    except Exception:
        logger.debug("run_context store failed (non-fatal)", exc_info=True)


def load_run_context(memory: Any) -> str:
    """P2: 加载最近一条 run_context, 返回人类可读摘要供 decider prompt 注入.

    返回空串表示无历史 / 加载失败 / 无有效 hypothesis. reader 与 persist_run_context
    共用同一 schema, 避免读侧漂移. memory: 有 recall(query=..., category=..., top_k=...) 即可.
    """
    if memory is None:
        return ""
    try:
        results = memory.recall(
            query="",
            category="run_context",
            top_k=1,
        )
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        return ""
    if not results:
        return ""
    entry = results[0] if isinstance(results, list) else results
    content = entry.get("content", "") if isinstance(entry, dict) else str(entry)
    if not content:
        return ""
    try:
        snap = json.loads(content)
    except (ValueError, TypeError):
        logger.debug("best-effort op failed", exc_info=True)
        return ""
    parts = [
        f"hypothesis: {(snap.get('hypothesis') or '')[:120]}",
        f"plan: {snap.get('plan_mode', 'none')}",
        f"outcome: {snap.get('outcome', 'unknown')}",
        f"iters: {snap.get('iterations', '?')}",
    ]
    _incon = snap.get("inconclusive") or []
    if _incon:
        parts.append(
            "inconclusive: " + "; ".join(
                f"iter{i.get('iter')}:{(i.get('advice') or '')[:50]}"
                for i in _incon[:2]
            )
        )
    return " | ".join(parts)