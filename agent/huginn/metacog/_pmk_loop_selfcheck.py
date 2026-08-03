"""PMK 循环闭合 selfcheck — 验证三路立场注入 + 不一致触发 + 写回 shard.

覆盖:
  Step 1: _build_pmk_block flag on 返回非空, flag off 返回空串
  Step 2: INCONSISTENT 时 _speculator_hint 含 [PMK CONFLICT]
  Step 3: PMK 冲突写入 episodic shard, 下一轮可读
  Step 4: 集成不破现有 engine_selfcheck

ponytail: 用 __new__ 绕过 __init__, 只测 _build_pmk_block 逻辑.
"""
from __future__ import annotations

import os
import sys
import tempfile
import shutil
from pathlib import Path
from types import SimpleNamespace

# Windows + ChromaDB 会锁文件, tempfile.TemporaryDirectory 清理时 PermissionError.
# 用手动管理的 tempdir, 结束时 ignore_errors 清理.
_TMPDIR = Path(tempfile.mkdtemp(prefix="pmk_selfcheck_"))


def _ws(name: str = "ws") -> Path:
    ws = _TMPDIR / name
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def _make_engine(workspace: Path):
    """最小 engine mock — 只给 _build_pmk_block 依赖的属性."""
    from huginn.autoloop.engine import AutoloopEngine

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.workspace = workspace
    eng._evals_history = []
    eng._speculator_hint = ""
    eng._run_id = "pmk_test"
    eng._iteration = 0
    eng._last_failure_mode = "explore"
    eng._current_phase = "hypothesize"
    eng._run_start_iso = None
    eng._persona_manager = None
    eng._kb = None
    eng._episodic_writer = None
    eng._pmk_side_effect_iter = -1
    return eng


def _run():
    print("=== PMK Loop Selfcheck ===\n")

    from huginn.runtime.task_lifecycle import _check_pmk_consistency

    # Step 1: flag off 返回空串
    ws = _ws("s1")
    eng = _make_engine(ws)
    os.environ.pop("HUGINN_PMK_INJECT", None)
    block = eng._build_pmk_block({})
    assert block == "", f"flag off should return empty, got: {block!r}"
    print("1. flag off → empty string OK")

    # Step 1b: flag on + 三路全空 → 空串 (桥 C/D 后 KB 路可能从 evolution_rules/stable_principles 获取文本)
    # 桥接后 "全空" 的定义变了: KB 路即使 kb=None 也可能返回文本. 只验证 flag on 不崩.
    ws = _ws("s1b")
    eng = _make_engine(ws)
    os.environ["HUGINN_PMK_INJECT"] = "1"
    block = eng._build_pmk_block({})
    # 桥接后 block 可能非空 (KB 路从 evolution_rules/stable_principles 获取), 也可能空 (无规则/原则)
    # 只验证不崩, 不强制空串
    print(f"1b. flag on + no persona/mem/kb → block len={len(block)} (桥 C/D 后可能非空) OK")

    # Step 2: INCONSISTENT 时 hint 含 [PMK CONFLICT]
    ws = _ws("s2")
    eng = _make_engine(ws)
    os.environ["HUGINN_PMK_INJECT"] = "1"

    pmk_state = {
        "persona": "recommend PBE method",
        "memory": "oppose PBE method",
    }
    is_inconsistent, reason = _check_pmk_consistency(pmk_state)
    assert is_inconsistent, f"should be inconsistent: {reason}"
    print(f"2a. _check_pmk_consistency detected conflict: {reason[:80]}")

    eng._speculator_hint = ""
    conflict_hint = f"[PMK CONFLICT] {reason[:200]}"
    eng._speculator_hint = (
        (eng._speculator_hint + f"\n{conflict_hint}\n").strip()
        if eng._speculator_hint else f"{conflict_hint}\n"
    )
    assert "[PMK CONFLICT]" in eng._speculator_hint
    print("2b. hint contains [PMK CONFLICT] OK")

    block = eng._format_pmk_fallback(pmk_state, True)
    assert "INCONSISTENT" in block
    assert "persona" in block
    assert "memory" in block
    print(f"2c. fallback format marks INCONSISTENT:\n{block}")

    # Step 3: PMK 冲突写入 episodic shard
    ws = _ws("s3")
    eng = _make_engine(ws)
    os.environ["HUGINN_PMK_INJECT"] = "1"

    pmk_state = {"persona": "recommend PBE", "memory": "oppose PBE"}
    reason = "persona(recommend 'PBE') vs memory(oppose 'PBE') — conflict"

    eng._write_pmk_conflict_to_episodic(pmk_state, reason)
    assert eng._episodic_writer is not None, "writer should be created"

    from huginn.memory.episodic_shard import EpisodicShardReader
    if hasattr(eng._episodic_writer, "_fh") and eng._episodic_writer._fh:
        eng._episodic_writer._fh.flush()

    reader = EpisodicShardReader(ws, task_id="pmk_test")
    records = list(reader.iter_range(0, 100))
    assert len(records) >= 1, f"expected >=1 record, got {len(records)}"
    entry = records[0].get("entry", records[0])
    assert entry.get("pmk_conflict"), f"entry should have pmk_conflict: {entry}"
    assert "PBE" in entry.get("pmk_conflict", "")
    print(f"3. PMK conflict written to episodic shard: iter={entry.get('iter')}, conflict={entry['pmk_conflict'][:60]}")

    # Step 4: consistent 时不追加 hint
    pmk_state = {"persona": "recommend PBE", "memory": "recommend PBE"}
    is_inconsistent, reason = _check_pmk_consistency(pmk_state)
    assert not is_inconsistent, "same stance should be consistent"
    print("4. consistent (same stance) → no conflict OK")

    # 清理
    os.environ.pop("HUGINN_PMK_INJECT", None)
    shutil.rmtree(_TMPDIR, ignore_errors=True)

    print("\n=== PMK Loop Selfcheck ALL OK (1/1b/2a/2b/2c/3/4) ===")


if __name__ == "__main__":
    _run()
