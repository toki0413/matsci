"""FailedDirectionStore — 失败方向统一存储.

P14 spec: 把 4 处分散的失败记录合并 (HypothesisNode.status / distilled_knowledge
memory / ToolBelief / _plan_check_patterns). 借鉴 EvoScientist EMA.

存储复用 P12 typed memory (memory_type="failed_direction"). P12 不可用时
降级到 category="failed_direction" 字符串.

ponytail: 单文件, 不新建表, 复用 SQLite memories.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FailedDirectionRecord:
    """单条失败方向记录."""

    hypothesis_text: str
    reason: str
    run_id: str
    persona_id: str | None = None
    math_concept: str = ""
    strategy_name: str = ""
    status: str = "refuted"  # refuted / superseded / strategy_failed


class FailedDirectionStore:
    """失败方向存储. 透传到 P12 typed memory, 不持有自有状态."""

    def __init__(self, memory_manager) -> None:
        self._mm = memory_manager

    def record(
        self,
        hypothesis_text: str,
        reason: str,
        run_id: str,
        persona_id: str | None = None,
        math_concept: str = "",
        strategy_name: str = "",
        status: str = "refuted",
    ) -> str:
        """记一条失败方向. 返回 entry_id (失败时 "")."""
        # P12 typed API 优先 — record_failed_direction 不接 strategy_name,
        # 我们把 strategy 拼进 content + tag, recall 端能解析回来.
        if hasattr(self._mm, "record_failed_direction"):
            content = (
                f"[Failed Direction] hypothesis: {hypothesis_text}\n"
                f"reason: {reason}"
            )
            if math_concept:
                content += f"\nmath_concept: {math_concept}"
            if strategy_name:
                content += f"\nstrategy: {strategy_name}"
            tags: list[str] = []
            if math_concept:
                tags.append(f"math_concept:{math_concept}")
            if strategy_name:
                tags.append(f"strategy:{strategy_name}")
            try:
                from huginn.memory.typing import remember_typed

                return remember_typed(
                    self._mm,
                    content=content,
                    memory_type="failed_direction",
                    run_id=run_id or None,
                    persona_id=persona_id,
                    status=status,
                    tags=tags,
                    importance=0.7,
                    tier="long",
                    source="evolution:failed_direction_store",
                )
            except Exception:
                logger.warning(
                    "remember_typed failed_direction failed, fallback to legacy",
                    exc_info=True,
                )

        # 降级: P12 未就位 / typed 写入抛错 → 走 legacy remember
        try:
            content = f"[Failed Direction] {hypothesis_text}: {reason}"
            if strategy_name:
                content += f" [strategy={strategy_name}]"
            return self._mm.remember(
                content=content,
                category="failed_direction",
                importance=0.7,
                tier="mid",
                tags=["failed_direction"],
            )
        except Exception:
            # mm 也不行 — 静默, 不能让 evolution 阻断主循环
            logger.warning("failed_direction record failed (no path)", exc_info=True)
            return ""

    def query(
        self,
        limit: int = 5,
        persona_id: str | None = None,
        math_concept: str | None = None,
    ) -> list[FailedDirectionRecord]:
        """查最近失败方向. P12 不可用时返空 list (不抛).

        优先用 recall_typed 拿全字段 (persona_id / run_id 都能恢复);
        recall_typed 不可用时降级到 recall_failed_directions 三元组.
        """
        records: list[FailedDirectionRecord] = []

        # 优先 recall_typed — 拿全字段
        if hasattr(self._mm, "recall_typed"):
            try:
                from huginn.memory.typing import recall_typed

                rows = recall_typed(
                    self._mm,
                    memory_type="failed_direction",
                    persona_id=persona_id,
                    limit=limit,
                )
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    content = row.get("content", "") or ""
                    hyp = ""
                    reason = ""
                    mc = ""
                    strat = ""
                    for line in content.split("\n"):
                        if line.startswith("[Failed Direction] hypothesis: "):
                            hyp = line[len("[Failed Direction] hypothesis: "):]
                        elif line.startswith("reason: "):
                            reason = line[len("reason: "):]
                        elif line.startswith("math_concept: "):
                            mc = line[len("math_concept: "):]
                        elif line.startswith("strategy: "):
                            strat = line[len("strategy: "):]
                    if math_concept and mc != math_concept:
                        continue
                    records.append(
                        FailedDirectionRecord(
                            hypothesis_text=hyp,
                            reason=reason,
                            run_id=row.get("run_id") or "",
                            persona_id=row.get("persona_id"),
                            math_concept=mc,
                            strategy_name=strat,
                            status=row.get("status") or "refuted",
                        )
                    )
                return records
            except Exception:
                logger.warning("recall_typed failed_direction failed", exc_info=True)

        # 降级: recall_failed_directions 三元组 (不带 persona_id/run_id)
        if not hasattr(self._mm, "recall_failed_directions"):
            return []
        try:
            triples = self._mm.recall_failed_directions(
                limit=limit, persona_id=persona_id
            )
        except Exception:
            logger.warning("recall_failed_directions failed", exc_info=True)
            return []

        for t in triples:
            hyp = t[0] if len(t) > 0 else ""
            reason = t[1] if len(t) > 1 else ""
            mc = t[2] if len(t) > 2 else ""
            if math_concept and mc != math_concept:
                continue
            records.append(
                FailedDirectionRecord(
                    hypothesis_text=hyp,
                    reason=reason,
                    run_id="",
                    persona_id=persona_id,
                    math_concept=mc,
                )
            )
        return records


# ── selfcheck ─────────────────────────────────────────────────────────────
# 2 场景: record + query round-trip / P12 不可用降级. 用真 in-memory SQLite,
# 跟 memory/typing.py 的 selfcheck 风格一致.

def _selfcheck() -> None:
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    _agent_root = Path(__file__).resolve().parents[2]
    if str(_agent_root) not in sys.path:
        sys.path.insert(0, str(_agent_root))

    from huginn.memory.longterm import LongTermMemory
    from huginn.memory.manager import MemoryManager

    tmpdir = tempfile.mkdtemp(prefix="huginn_fds_selfcheck_")
    db_path = Path(tmpdir) / "memory.db"
    ltm = LongTermMemory(db_path=str(db_path), enable_semantic=False)
    mm = MemoryManager(longterm=ltm)
    store = FailedDirectionStore(mm)

    # 场景 1: record + query round-trip (走 P12 typed API)
    eid = store.record(
        hypothesis_text="GaN band gap > 4 eV with LDA",
        reason="LDA underestimates gap, experimental ~3.4 eV",
        run_id="run_001",
        persona_id="dft_expert",
        math_concept="DFT-PZ LDA band gap underestimation",
        strategy_name="lda_direct_gap",
        status="refuted",
    )
    assert eid, "record should return entry_id"
    records = store.query(limit=5)
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    r = records[0]
    assert "GaN band gap" in r.hypothesis_text, r.hypothesis_text
    assert "LDA underestimates" in r.reason, r.reason
    assert r.math_concept == "DFT-PZ LDA band gap underestimation", r.math_concept
    assert r.persona_id == "dft_expert", r.persona_id
    print("1. record + query round-trip OK")

    # 场景 2: P12 不可用降级 — 给个 mock mm 没有 typed API, 只 remember
    class _LegacyMM:
        def __init__(self):
            self.calls = []

        def remember(self, content, category=None, **kw):
            self.calls.append((content, category))
            return "legacy_id"

    legacy = _LegacyMM()
    legacy_store = FailedDirectionStore(legacy)
    # recall_failed_directions 不存在 → query 返空
    assert legacy_store.query(limit=5) == []
    # record 走 legacy remember
    rid = legacy_store.record(
        hypothesis_text="legacy hyp",
        reason="legacy reason",
        run_id="r2",
    )
    assert rid == "legacy_id", rid
    assert len(legacy.calls) == 1
    assert "legacy hyp" in legacy.calls[0][0]
    assert legacy.calls[0][1] == "failed_direction"
    print("2. legacy degradation OK")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("failed_direction_store selfcheck OK (2 scenarios)")


if __name__ == "__main__":
    _selfcheck()
