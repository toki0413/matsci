"""Typed memory API — 结构化 memory_type/run_id/persona_id/status.

P12 spec: 替代 category 字符串 + tags JSON hack. 借鉴 InternAgent 3 类
(Strategy-Procedural / Task-Episodic / Semantic-Knowledge) + EvoScientist 2 类
(ideation / experimentation, 含失败方向记录).

MemoryType enum 扩展到 10 值:
- 现有 5 (USER/FEEDBACK/PROJECT/REFERENCE/CALCULATION, 来自 types.py)
- 新增 5: ITERATION_RESULT / PERSONA_HISTORY / FAILED_DIRECTION /
  CROSS_DOMAIN_TRANSFER / STABLE_PRINCIPLE

API:
- remember_typed(mm, content, memory_type, run_id=None, persona_id=None, status=None, **kwargs)
- recall_typed(mm, memory_type, persona_id=None, run_id=None, status=None, limit=10)
- record_failed_direction(mm, hypothesis_text, reason, run_id, persona_id, math_concept="")
- recall_failed_directions(mm, limit=5, persona_id=None)

ponytail: 复用 SQLite memories 表扩列, 不新建表. 单文件, 不引入新依赖.
"""
from __future__ import annotations

import os
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    """扩展到 10 值. 继承 str 让枚举值直接 JSON/SQL 友好."""

    # 现有 5 (跟 types.py 保持值一致, 旧文件存储路径不破)
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"
    CALCULATION = "calculation"
    # P12 新增 5
    ITERATION_RESULT = "iteration_result"
    PERSONA_HISTORY = "persona_history"
    FAILED_DIRECTION = "failed_direction"
    CROSS_DOMAIN_TRANSFER = "cross_domain_transfer"
    STABLE_PRINCIPLE = "stable_principle"


def _use_typing() -> bool:
    """env flag HUGINN_USE_MEMORY_TYPING 控制 engine 是否调 typed API.

    默认 OFF — 行为 100% 不变, 跟 P12 之前一致. schema migration v2 仍然跑
    (加列 NULL 不影响旧行为), flag 翻转不需要再迁移. 显式设 =1 时 engine
    的 _learn / _pick_hypothesis_persona / _recent_failed_hypotheses 改走
    typed 路径, 失败时降级回原路径.
    """
    return os.environ.get("HUGINN_USE_MEMORY_TYPING", "0") in ("1", "true", "True")


def _memory_type_to_category(memory_type: str) -> str:
    """memory_type → 旧 category 映射, 保持向后兼容旧 reader.

    旧 reader (lint / recall_for_prompt / RAG) 按 category 字符串过滤, 不识别
    memory_type. typed 写入时同时填 category, 旧 reader 继续工作.
    """
    mapping = {
        "user": "conversation",
        "feedback": "conversation",
        "project": "fact",
        "reference": "fact",
        "calculation": "calculation",
        "iteration_result": "iteration",
        "persona_history": "episode",
        "failed_direction": "distilled_knowledge",
        "cross_domain_transfer": "distilled_knowledge",
        "stable_principle": "distilled_knowledge",
    }
    return mapping.get(memory_type, "fact")


def remember_typed(
    mm: Any,
    content: str,
    memory_type: str,
    *,
    run_id: str | None = None,
    persona_id: str | None = None,
    status: str | None = None,
    importance: float = 0.5,
    tier: str = "mid",
    tags: list[str] | None = None,
    source: str = "",
    **extra: Any,
) -> str:
    """写 typed memory. 复用 longterm.store + 扩展字段.

    返回 entry_id. mm 是 MemoryManager 实例.
    直接走 longterm.store 而不是 mm.remember, 因为 mm.remember 把 source
    硬编码成 "session:{sid}", typed memory 需要自定义 source (如 run_id).
    """
    # category 字段保持向后兼容旧 reader — 用 memory_type 映射
    category = _memory_type_to_category(memory_type)
    entry_id = mm.longterm.store(
        content=content,
        category=category,
        tags=tags or [],
        source=source or f"typed:{memory_type}",
        importance=importance,
        tier=tier,
    )
    # 扩展字段单独 UPDATE (longterm.store 不接 typed 列)
    mm._update_typed_fields(
        entry_id,
        memory_type=memory_type,
        run_id=run_id,
        persona_id=persona_id,
        status=status,
    )
    return entry_id


def recall_typed(
    mm: Any,
    memory_type: str,
    *,
    persona_id: str | None = None,
    run_id: str | None = None,
    status: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """查 typed memory. 严格匹配 memory_type, 旧行 (NULL) 不参与.

    返回 list[dict], 每条含 memories 表全部列. 按 last_accessed DESC 排序.
    """
    return mm._recall_typed(
        memory_type=memory_type,
        persona_id=persona_id,
        run_id=run_id,
        status=status,
        limit=limit,
    )


def record_failed_direction(
    mm: Any,
    hypothesis_text: str,
    reason: str,
    run_id: str,
    persona_id: str | None = None,
    math_concept: str = "",
) -> str:
    """记录失败方向. status 默认 "refuted".

    content 格式固定, recall_failed_directions 靠这个格式反解三元组.
    math_concept 留给 P13 CrossDomain 用 — 关联数学概念让跨域类比能查到.
    """
    content = f"[Failed Direction] hypothesis: {hypothesis_text}\nreason: {reason}"
    if math_concept:
        content += f"\nmath_concept: {math_concept}"
    tags = [f"math_concept:{math_concept}"] if math_concept else []
    return remember_typed(
        mm,
        content=content,
        memory_type=MemoryType.FAILED_DIRECTION.value,
        run_id=run_id,
        persona_id=persona_id,
        status="refuted",
        tags=tags,
        importance=0.7,
        tier="long",  # 失败方向要长期保留, 跨 session 可恢复
    )


def recall_failed_directions(
    mm: Any,
    limit: int = 5,
    persona_id: str | None = None,
) -> list[tuple[str, str, str]]:
    """查最近失败方向. 返回 [(hypothesis_text, reason, math_concept)] 三元组.

    跨 session 可恢复 — 数据在 SQLite, 不依赖 hypothesis_graph 内存状态.
    """
    records = recall_typed(
        mm,
        memory_type=MemoryType.FAILED_DIRECTION.value,
        persona_id=persona_id,
        status="refuted",
        limit=limit,
    )
    results: list[tuple[str, str, str]] = []
    for r in records:
        content = r.get("content", "") if isinstance(r, dict) else str(r)
        hyp = ""
        reason = ""
        math_concept = ""
        for line in content.split("\n"):
            if line.startswith("[Failed Direction] hypothesis: "):
                hyp = line[len("[Failed Direction] hypothesis: "):]
            elif line.startswith("reason: "):
                reason = line[len("reason: "):]
            elif line.startswith("math_concept: "):
                math_concept = line[len("math_concept: "):]
        results.append((hyp, reason, math_concept))
    return results


# ── selfcheck ─────────────────────────────────────────────────────────────
# 4 场景: 用 in-memory SQLite + 真 LongTermMemory, 不调 LLM. 跟 cluster.py
# 风格一致 — 真 DB 跑 migration v2, 真 store/retrieve, 只 mock LLM 层.

def _selfcheck() -> None:
    import sqlite3
    import sys
    import tempfile
    from pathlib import Path

    # 让 huginn.* 可导入 (跟 manager.py / cluster.py 一致做法)
    _agent_root = Path(__file__).resolve().parents[2]
    if str(_agent_root) not in sys.path:
        sys.path.insert(0, str(_agent_root))

    from huginn.memory.longterm import LongTermMemory
    from huginn.memory.manager import MemoryManager

    # 在临时目录建真 SQLite DB, migration v2 自动跑
    tmpdir = tempfile.mkdtemp(prefix="huginn_typing_selfcheck_")
    db_path = Path(tmpdir) / "memory.db"
    ltm = LongTermMemory(db_path=str(db_path), enable_semantic=False)
    mm = MemoryManager(longterm=ltm)

    # 场景 1: remember_typed + recall_typed round-trip
    eid = remember_typed(
        mm,
        content="persona dft_expert r_phys=0.78 on GaN band gap",
        memory_type=MemoryType.PERSONA_HISTORY.value,
        run_id="run_001",
        persona_id="dft_expert",
        status="supported",
        importance=0.8,
        tier="long",
    )
    assert eid, "remember_typed should return entry_id"
    rows = recall_typed(
        mm,
        memory_type=MemoryType.PERSONA_HISTORY.value,
        persona_id="dft_expert",
        limit=5,
    )
    assert len(rows) == 1, f"expected 1 row, got {len(rows)}"
    row = rows[0]
    assert row["memory_type"] == "persona_history"
    assert row["run_id"] == "run_001"
    assert row["persona_id"] == "dft_expert"
    assert row["status"] == "supported"
    # category 兼容字段也填了
    assert row["category"] == "episode"
    print("1. remember_typed + recall_typed round-trip OK")

    # 场景 2: record_failed_direction + recall_failed_directions
    fid = record_failed_direction(
        mm,
        hypothesis_text="GaN band gap > 4 eV with LDA",
        reason="LDA underestimates gap, experimental ~3.4 eV",
        run_id="run_002",
        persona_id="dft_expert",
        math_concept="DFT-PZ LDA band gap underestimation",
    )
    assert fid, "record_failed_direction should return entry_id"
    failed = recall_failed_directions(mm, limit=5)
    assert len(failed) == 1, f"expected 1 failed, got {len(failed)}"
    hyp, reason, mc = failed[0]
    assert "GaN band gap > 4 eV" in hyp
    assert "LDA underestimates" in reason
    assert mc == "DFT-PZ LDA band gap underestimation"
    print("2. record_failed_direction + recall_failed_directions OK")

    # 场景 3: 跨 session 模拟 (新 MemoryManager 实例 + 同 DB)
    # 关键: 不依赖内存 hypothesis_graph, 数据在 SQLite 里持久化
    ltm2 = LongTermMemory(db_path=str(db_path), enable_semantic=False)
    mm2 = MemoryManager(longterm=ltm2)
    failed2 = recall_failed_directions(mm2, limit=5)
    assert len(failed2) == 1, (
        f"cross-session recall should find 1 failed, got {len(failed2)}"
    )
    assert failed2[0][0].startswith("GaN band gap"), failed2[0]
    persona_rows = recall_typed(
        mm2,
        memory_type=MemoryType.PERSONA_HISTORY.value,
        persona_id="dft_expert",
        limit=5,
    )
    assert len(persona_rows) == 1, "cross-session persona_history recall"
    print("3. cross-session recovery OK")

    # 场景 4: NULL 兼容路径 — 旧行 memory_type IS NULL 不参与 typed 查询
    # 先写一条 legacy 行 (走 mm.remember, 不调 typed API)
    legacy_id = mm2.remember(
        content="legacy memory without typed fields",
        category="fact",
        tags=["legacy"],
        importance=0.5,
        tier="mid",
    )
    assert legacy_id
    # typed 查询不应该返回 legacy 行
    typed_rows = recall_typed(
        mm2,
        memory_type=MemoryType.PERSONA_HISTORY.value,
        limit=10,
    )
    assert len(typed_rows) == 1, (
        f"typed query should not return legacy NULL rows, got {len(typed_rows)}"
    )
    for tr in typed_rows:
        assert tr["memory_type"] == "persona_history", (
            f"strict match failed: {tr.get('memory_type')}"
        )
    print("4. NULL compatibility (legacy rows excluded from typed query) OK")

    # 清理临时目录
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("memory.typing selfcheck OK (4 scenarios)")


if __name__ == "__main__":
    _selfcheck()
