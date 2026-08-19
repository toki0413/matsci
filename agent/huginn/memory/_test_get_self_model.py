"""路径 A 自检: get_self_model 跨任务聚合 + curiosity block 闭环.

不引框架, 全用 assert. 验证:
  1. 空库 → {}
  2. 单 persona 单 status → rate=1.0 或 0.0
  3. 多 persona 混合 status → 按 persona 聚合, rate 正确
  4. 跨 db 实例 (模拟跨任务) → 新实例能读旧数据 (共享 db 路径)
  5. 弱 persona (rate<0.4, n>=3) 触发 curiosity block 格式

运行: python -m huginn.memory._test_get_self_model
"""
import sys
import tempfile
from pathlib import Path

# 加 agent 到 path
_AGENT = Path(__file__).resolve().parents[2]
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))


def _make_ltm(db_path: Path):
    """建一个 LongTermMemory 实例, 跳过 vector_store."""
    from huginn.memory.longterm import LongTermMemory
    return LongTermMemory(db_path=str(db_path), vector_store=None, enable_semantic=False)


def _write_iteration(mm, persona: str, status: str, content: str = "test"):
    """模拟 engine._learn 写 iteration_result."""
    from huginn.memory.typing import MemoryType, remember_typed
    remember_typed(
        mm,
        content=content,
        memory_type=MemoryType.ITERATION_RESULT.value,
        persona_id=persona,
        status=status,
        importance=0.5,
    )


def test_empty_db():
    """空库 get_self_model 返回 {}."""
    with tempfile.TemporaryDirectory() as td:
        ltm = _make_ltm(Path(td) / "mem.db")
        sm = ltm.get_self_model()
        assert sm == {}, f"empty db should give {{}}, got {sm}"
    print("[OK] empty db → {}")


def test_single_persona_supported():
    """单 persona 全 supported → rate=1.0."""
    with tempfile.TemporaryDirectory() as td:
        ltm = _make_ltm(Path(td) / "mem.db")
        # 直接走 longterm.store + _update_typed_fields (绕过 MemoryManager)
        _eid = ltm.store(
            content="test", category="iteration_result",
            tags=[], source="test", importance=0.5, tier="mid",
        )
        # 手动 UPDATE typed 字段 (模拟 remember_typed 的 UPDATE)
        with ltm._connect() as conn:
            conn.execute(
                "UPDATE memories SET memory_type='iteration_result', persona_id='phys', status='supported' WHERE id=?",
                (_eid,),
            )
            conn.commit()
        sm = ltm.get_self_model()
        assert "phys" in sm, f"persona phys missing: {sm}"
        assert sm["phys"]["rate"] == 1.0, f"rate should be 1.0, got {sm['phys']['rate']}"
        assert sm["phys"]["success"] == 1
        assert sm["phys"]["failure"] == 0
    print("[OK] single persona supported → rate=1.0")


def test_mixed_persona_status():
    """多 persona 混合 status → 按 persona 聚合, rate 正确."""
    with tempfile.TemporaryDirectory() as td:
        ltm = _make_ltm(Path(td) / "mem.db")
        # persona A: 2 supported, 3 refuted → rate=0.4
        # persona B: 1 supported, 0 refuted → rate=1.0
        rows = [
            ("A", "supported"), ("A", "supported"),
            ("A", "refuted"), ("A", "refuted"), ("A", "refuted"),
            ("B", "supported"),
        ]
        for i, (p, s) in enumerate(rows):
            _eid = ltm.store(
                content=f"test{i}", category="iteration_result",
                tags=[], source="test", importance=0.5, tier="mid",
            )
            with ltm._connect() as conn:
                conn.execute(
                    "UPDATE memories SET memory_type='iteration_result', persona_id=?, status=? WHERE id=?",
                    (p, s, _eid),
                )
                conn.commit()
        sm = ltm.get_self_model()
        assert "A" in sm and "B" in sm, f"both personas missing: {sm}"
        assert sm["A"]["rate"] == 0.4, f"A rate should be 0.4, got {sm['A']['rate']}"
        assert sm["A"]["success"] == 2 and sm["A"]["failure"] == 3
        assert sm["B"]["rate"] == 1.0, f"B rate should be 1.0, got {sm['B']['rate']}"
    print("[OK] mixed persona → A rate=0.4, B rate=1.0")


def test_cross_instance_shared_db():
    """跨 db 实例 (模拟跨任务) → 新实例能读旧数据."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "shared.db"
        # 第一次任务: 写 persona A
        ltm1 = _make_ltm(db_path)
        _eid = ltm1.store(
            content="task1", category="iteration_result",
            tags=[], source="test", importance=0.5, tier="mid",
        )
        with ltm1._connect() as conn:
            conn.execute(
                "UPDATE memories SET memory_type='iteration_result', persona_id='A', status='refuted' WHERE id=?",
                (_eid,),
            )
            conn.commit()
        # 第二次任务: 新实例, 同 db
        ltm2 = _make_ltm(db_path)
        sm = ltm2.get_self_model()
        assert "A" in sm, f"cross-instance should see persona A: {sm}"
        assert sm["A"]["failure"] == 1
    print("[OK] cross-instance shared db → persona A visible")


def test_curiosity_block_format():
    """弱 persona (rate<0.4, n>=3) 触发 curiosity block 格式."""
    with tempfile.TemporaryDirectory() as td:
        ltm = _make_ltm(Path(td) / "mem.db")
        # persona weak: 1 supported, 4 refuted → rate=0.2, n=5
        rows = [("weak", "supported")] + [("weak", "refuted")] * 4
        for i, (p, s) in enumerate(rows):
            _eid = ltm.store(
                content=f"t{i}", category="iteration_result",
                tags=[], source="test", importance=0.5, tier="mid",
            )
            with ltm._connect() as conn:
                conn.execute(
                    "UPDATE memories SET memory_type='iteration_result', persona_id=?, status=? WHERE id=?",
                    (p, s, _eid),
                )
                conn.commit()
        sm = ltm.get_self_model()
        # 模拟 rcb_runner 的 curiosity 注入逻辑
        weak = [
            f"- {v.get('dimension', '?')}/{v.get('hyp_type', '?')}: "
            f"rate={v.get('rate', 0):.2f} (n={v.get('success', 0) + v.get('failure', 0)})"
            for v in sm.values()
            if isinstance(v.get("rate"), (int, float))
            and v["rate"] < 0.4
            and v.get("success", 0) + v.get("failure", 0) >= 3
        ]
        assert len(weak) == 1, f"should have 1 weak persona, got {weak}"
        assert "weak/unknown" in weak[0], f"format wrong: {weak[0]}"
        assert "rate=0.20" in weak[0], f"rate format wrong: {weak[0]}"
        assert "(n=5)" in weak[0], f"sample size wrong: {weak[0]}"
    print("[OK] curiosity block format: weak/unknown rate=0.20 (n=5)")


def test_archived_excluded():
    """archived=1 的条目不参与 get_self_model."""
    with tempfile.TemporaryDirectory() as td:
        ltm = _make_ltm(Path(td) / "mem.db")
        _eid = ltm.store(
            content="archived_test", category="iteration_result",
            tags=[], source="test", importance=0.5, tier="mid",
        )
        with ltm._connect() as conn:
            conn.execute(
                "UPDATE memories SET memory_type='iteration_result', persona_id='X', status='supported', archived=1 WHERE id=?",
                (_eid,),
            )
            conn.commit()
        sm = ltm.get_self_model()
        assert "X" not in sm, f"archived should be excluded: {sm}"
    print("[OK] archived excluded from self_model")


if __name__ == "__main__":
    test_empty_db()
    test_single_persona_supported()
    test_mixed_persona_status()
    test_cross_instance_shared_db()
    test_curiosity_block_format()
    test_archived_excluded()
    print("\n所有自检通过 — get_self_model 跨任务聚合 + curiosity block 闭环 OK")
