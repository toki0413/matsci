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

    C4 后默认 ON — typed memory 是分层 memory 的基础, 平常该用.
    显式设 =0 可关 (回退到 grep/正则 fallback 路径).
    schema migration v2 无条件跑, 旧 DB 升级不需要额外操作.
    """
    return os.environ.get("HUGINN_USE_MEMORY_TYPING", "1") in ("1", "true", "True")


# tags/source/category → memory_type 反推规则 (lazy migrate 用)
# 旧行 memory_type IS NULL, recall 时按这些规则临时推断 type
_INFER_RULES: list[tuple[str, str, str]] = [
    # (predicate, matcher, memory_type)
    # predicate 在 tags 任一元素上 prefix-match, 或 source/category 精确匹配
    ("tag_prefix", "autoloop", "iteration_result"),
    ("tag_prefix", "math_concept:", "failed_direction"),
    ("tag_prefix", "strategy:", "failed_direction"),
    ("source_prefix", "cross_domain:", "cross_domain_transfer"),
    ("source_prefix", "evolution:failed_direction_store", "failed_direction"),
    ("source_prefix", "evolution:distill", "stable_principle"),
    ("source_prefix", "typed:failed_direction", "failed_direction"),
    ("source_prefix", "typed:stable_principle", "stable_principle"),
    ("source_prefix", "typed:cross_domain_transfer", "cross_domain_transfer"),
    ("source_prefix", "typed:iteration_result", "iteration_result"),
    ("source_prefix", "typed:persona_history", "persona_history"),
    ("category_eq", "episode", "persona_history"),
    ("category_eq", "iteration", "iteration_result"),
]


def _infer_memory_type_from_tags(
    tags: list[str],
    source: str,
    category: str,
) -> str | None:
    """从 tags/source/category 反推 memory_type (lazy migrate 用).

    旧行 memory_type IS NULL, recall 时按这些规则推断. 命中返回 memory_type,
    否则 None. ponytail: 规则表驱动, 不引入 ML 分类器.
    """
    for pred, matcher, mtype in _INFER_RULES:
        if pred == "tag_prefix":
            if any(t.startswith(matcher) for t in tags):
                return mtype
        elif pred == "source_prefix":
            if source and source.startswith(matcher):
                return mtype
        elif pred == "category_eq":
            if category == matcher:
                return mtype
    return None


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
    # typed 查询不应该返回 legacy 行 (category=fact 无反推规则)
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

    # 场景 5: lazy migrate — legacy 行 tags 含 "autoloop" + "persona:reviewer"
    # 应被反推为 iteration_result (write-on-read), typed 查询能命中
    legacy_autoloop_id = mm2.remember(
        content="Persona: reviewer, r_phys: 0.85 on GaN band gap",
        category="autoloop_iteration",
        tags=["autoloop", "persona:reviewer", "r_phys:0.85", "surprise:0.3"],
        importance=0.8,
        tier="mid",
    )
    assert legacy_autoloop_id
    # 第一次查 iteration_result — strict 返空, 触发 lazy migrate
    iter_rows_1 = recall_typed(
        mm2,
        memory_type=MemoryType.ITERATION_RESULT.value,
        persona_id="reviewer",  # 但 legacy 行没 persona_id 列, 这个过滤会失败
        limit=10,
    )
    # persona_id 列在 legacy 行也是 NULL, 加了 persona_id 过滤就查不到
    # 不加 persona_id 过滤再查一次
    iter_rows_2 = recall_typed(
        mm2,
        memory_type=MemoryType.ITERATION_RESULT.value,
        limit=10,
    )
    assert len(iter_rows_2) >= 1, (
        f"lazy migrate should make legacy autoloop row appear as iteration_result, "
        f"got {len(iter_rows_2)}"
    )
    # 验证 memory_type 已回填 (write-on-read)
    assert iter_rows_2[0]["memory_type"] == "iteration_result", (
        f"lazy migrate should backfill memory_type, got {iter_rows_2[0].get('memory_type')}"
    )
    print("5. lazy migrate (legacy autoloop → iteration_result) OK")

    # 场景 6: 反推规则覆盖 — _infer_memory_type_from_tags 各路径
    assert _infer_memory_type_from_tags(["autoloop"], "", "") == "iteration_result"
    assert _infer_memory_type_from_tags(["math_concept:X"], "", "") == "failed_direction"
    assert _infer_memory_type_from_tags(["strategy:Y"], "", "") == "failed_direction"
    assert _infer_memory_type_from_tags([], "cross_domain:foo", "") == "cross_domain_transfer"
    assert _infer_memory_type_from_tags([], "evolution:distill", "") == "stable_principle"
    assert _infer_memory_type_from_tags([], "", "episode") == "persona_history"
    assert _infer_memory_type_from_tags([], "", "iteration") == "iteration_result"
    assert _infer_memory_type_from_tags(["legacy"], "", "fact") is None
    print("6. _infer_memory_type_from_tags rule coverage OK")

    # 清理临时目录
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("memory.typing selfcheck OK (6 scenarios)")


if __name__ == "__main__":
    _selfcheck()
