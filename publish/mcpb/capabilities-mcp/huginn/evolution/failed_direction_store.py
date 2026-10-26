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
    # Physical RSI 归因: environment_gap (世界模型无法复现失败因果, 需更新认知)
    # / skill_gap (可复现, 需换策略) / unknown (未归因).
    gap_type: str = "unknown"


def _build_typed_payload(
    hypothesis_text: str,
    reason: str,
    math_concept: str,
    strategy_name: str,
    gap_type: str,
) -> tuple[str, list[str]]:
    """构造 P12 typed memory 的 content + tags."""
    content = f"[Failed Direction] hypothesis: {hypothesis_text}\nreason: {reason}"
    if math_concept:
        content += f"\nmath_concept: {math_concept}"
    if strategy_name:
        content += f"\nstrategy: {strategy_name}"
    if gap_type and gap_type != "unknown":
        content += f"\ngap_type: {gap_type}"
    tags: list[str] = []
    if math_concept:
        tags.append(f"math_concept:{math_concept}")
    if strategy_name:
        tags.append(f"strategy:{strategy_name}")
    if gap_type and gap_type != "unknown":
        tags.append(f"gap_type:{gap_type}")
    return content, tags


def _build_legacy_content(
    hypothesis_text: str,
    reason: str,
    strategy_name: str,
    gap_type: str,
) -> str:
    """构造 legacy remember 的 content."""
    content = f"[Failed Direction] {hypothesis_text}: {reason}"
    if strategy_name:
        content += f" [strategy={strategy_name}]"
    if gap_type and gap_type != "unknown":
        content += f" [gap_type={gap_type}]"
    return content


def _parse_typed_content(content: str) -> tuple[str, str, str, str, str]:
    """从 typed memory content 解析出各字段."""
    hyp = ""
    reason = ""
    mc = ""
    strat = ""
    gap = "unknown"
    for line in content.split("\n"):
        if line.startswith("[Failed Direction] hypothesis: "):
            hyp = line[len("[Failed Direction] hypothesis: ") :]
        elif line.startswith("reason: "):
            reason = line[len("reason: ") :]
        elif line.startswith("math_concept: "):
            mc = line[len("math_concept: ") :]
        elif line.startswith("strategy: "):
            strat = line[len("strategy: ") :]
        elif line.startswith("gap_type: "):
            gap = line[len("gap_type: ") :]
    return hyp, reason, mc, strat, gap


def _record_from_typed_row(
    row: dict,
    math_concept: str | None,
    gap_type: str | None,
) -> FailedDirectionRecord | None:
    """把一行 recall_typed 结果转成记录; 被过滤时返回 None."""
    content = row.get("content", "") or ""
    hyp, reason, mc, strat, gap = _parse_typed_content(content)
    if math_concept and mc != math_concept:
        return None
    if gap_type and gap != gap_type:
        return None
    return FailedDirectionRecord(
        hypothesis_text=hyp,
        reason=reason,
        run_id=row.get("run_id") or "",
        persona_id=row.get("persona_id"),
        math_concept=mc,
        strategy_name=strat,
        status=row.get("status") or "refuted",
        gap_type=gap,
    )


def _query_typed(
    mm,
    limit: int,
    persona_id: str | None,
    math_concept: str | None,
    gap_type: str | None,
) -> list[FailedDirectionRecord] | None:
    """优先 recall_typed — 拿全字段. 返回 None 表示应降级到三元组."""
    if not hasattr(mm, "recall_typed"):
        return None
    try:
        from huginn.memory.typing import recall_typed

        rows = recall_typed(
            mm,
            memory_type="failed_direction",
            persona_id=persona_id,
            limit=limit,
        )
        records: list[FailedDirectionRecord] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            record = _record_from_typed_row(row, math_concept, gap_type)
            if record is not None:
                records.append(record)
        return records
    except Exception:
        logger.warning("recall_typed failed_direction failed", exc_info=True)
        return None


def _query_legacy(
    mm,
    limit: int,
    persona_id: str | None,
    math_concept: str | None,
    gap_type: str | None,
) -> list[FailedDirectionRecord]:
    """降级: recall_failed_directions 三元组 (不带 persona_id/run_id)."""
    if not hasattr(mm, "recall_failed_directions"):
        return []
    try:
        triples = mm.recall_failed_directions(limit=limit, persona_id=persona_id)
    except Exception:
        logger.warning("recall_failed_directions failed", exc_info=True)
        return []

    records: list[FailedDirectionRecord] = []
    for t in triples:
        hyp = t[0] if len(t) > 0 else ""
        reason = t[1] if len(t) > 1 else ""
        mc = t[2] if len(t) > 2 else ""
        if math_concept and mc != math_concept:
            continue
        # legacy 三元组不带 gap_type (视为 unknown); 明确过滤非 unknown 时跳过
        if gap_type and gap_type != "unknown":
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


def _record_typed(
    mm,
    hypothesis_text: str,
    reason: str,
    run_id: str,
    persona_id: str | None,
    math_concept: str,
    strategy_name: str,
    status: str,
    gap_type: str,
) -> tuple[bool, str]:
    """P12 typed API 写入. 返回 (是否已处理, entry_id); 未处理时走 legacy."""
    # P12 typed API 优先 — record_failed_direction 不接 strategy_name,
    # 我们把 strategy 拼进 content + tag, recall 端能解析回来.
    if not hasattr(mm, "record_failed_direction"):
        return False, ""
    content, tags = _build_typed_payload(
        hypothesis_text, reason, math_concept, strategy_name, gap_type
    )
    try:
        from huginn.memory.typing import remember_typed

        return True, remember_typed(
            mm,
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
        return False, ""


def _record_legacy(
    mm,
    hypothesis_text: str,
    reason: str,
    strategy_name: str,
    gap_type: str,
) -> str:
    """降级: P12 未就位 / typed 写入抛错 → 走 legacy remember."""
    try:
        content = _build_legacy_content(
            hypothesis_text, reason, strategy_name, gap_type
        )
        return mm.remember(
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
        gap_type: str = "unknown",
    ) -> str:
        """记一条失败方向. 返回 entry_id (失败时 "")."""
        handled, entry_id = _record_typed(
            self._mm,
            hypothesis_text,
            reason,
            run_id,
            persona_id,
            math_concept,
            strategy_name,
            status,
            gap_type,
        )
        if handled:
            return entry_id

        return _record_legacy(
            self._mm, hypothesis_text, reason, strategy_name, gap_type
        )

    def query(
        self,
        limit: int = 5,
        persona_id: str | None = None,
        math_concept: str | None = None,
        gap_type: str | None = None,
    ) -> list[FailedDirectionRecord]:
        """查最近失败方向. P12 不可用时返空 list (不抛).

        优先用 recall_typed 拿全字段 (persona_id / run_id 都能恢复);
        recall_typed 不可用时降级到 recall_failed_directions 三元组.
        gap_type 为 None 时不过滤.
        """
        records = _query_typed(self._mm, limit, persona_id, math_concept, gap_type)
        if records is not None:
            return records

        return _query_legacy(self._mm, limit, persona_id, math_concept, gap_type)


# ── selfcheck ─────────────────────────────────────────────────────────────
# 场景: record + query round-trip (含 gap_type 归因) / P12 不可用降级. 用真
# in-memory SQLite, 跟 memory/typing.py 的 selfcheck 风格一致.


def _selfcheck_typed_roundtrip(store: FailedDirectionStore) -> None:
    """场景 1: record + query round-trip (走 P12 typed API)."""
    eid = store.record(
        hypothesis_text="GaN band gap > 4 eV with LDA",
        reason="LDA underestimates gap, experimental ~3.4 eV",
        run_id="run_001",
        persona_id="dft_expert",
        math_concept="DFT-PZ LDA band gap underestimation",
        strategy_name="lda_direct_gap",
        status="refuted",
        gap_type="environment_gap",
    )
    assert eid, "record should return entry_id"
    records = store.query(limit=5)
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    r = records[0]
    assert "GaN band gap" in r.hypothesis_text, r.hypothesis_text
    assert "LDA underestimates" in r.reason, r.reason
    assert r.math_concept == "DFT-PZ LDA band gap underestimation", r.math_concept
    assert r.persona_id == "dft_expert", r.persona_id
    assert r.gap_type == "environment_gap", r.gap_type
    # gap_type 过滤
    assert store.query(limit=5, gap_type="environment_gap") == records
    assert store.query(limit=5, gap_type="skill_gap") == []
    print("1. record + query round-trip (gap_type) OK")


def _selfcheck_legacy_degradation() -> None:
    """场景 2: P12 不可用降级 — 给个 mock mm 没有 typed API, 只 remember."""

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

    _selfcheck_typed_roundtrip(store)
    _selfcheck_legacy_degradation()

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("failed_direction_store selfcheck OK (2 scenarios)")


if __name__ == "__main__":
    _selfcheck()
