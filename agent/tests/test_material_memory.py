"""Phase 5a 材料长期记忆测试.

5 测:
  1. store_material 正常 (GaN property)
  2. store_material 异常 category (ValueError)
  3. recall material_filter formula 命中
  4. recall material_filter formula 不命中
  5. FTS5 formula 检索
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from huginn.memory.longterm import LongTermMemory, MATERIAL_CATEGORIES
from huginn.memory.manager import MemoryManager


@pytest.fixture
def tmp_memory():
    """临时 SQLite, 不开 semantic (没 vector_store)."""
    with tempfile.TemporaryDirectory() as d:
        mem = LongTermMemory(db_path=str(Path(d) / "test_mem.db"), enable_semantic=False)
        yield mem


class TestMaterialMemory:
    """store_material + recall with material_filter."""

    def test_store_material_normal(self, tmp_memory: LongTermMemory) -> None:
        eid = tmp_memory.store_material(
            formula="GaN",
            category="property",
            payload={"band_gap": 3.4, "unit": "eV"},
        )
        assert eid.startswith("mem_")
        # 验证 formula 列写入
        row = tmp_memory.get_by_id(eid)
        assert row is not None
        assert row["formula"] == "GaN"
        assert row["category"] == "material_property"
        assert "GaN" in row["tags"]

    def test_store_material_invalid_category(self, tmp_memory: LongTermMemory) -> None:
        with pytest.raises(ValueError, match="Invalid material category"):
            tmp_memory.store_material(
                formula="Si", category="bogus", payload={"x": 1}
            )

    def test_recall_material_filter_hit(self, tmp_memory: LongTermMemory) -> None:
        tmp_memory.store_material("GaN", "property", {"band_gap": 3.4})
        tmp_memory.store_material("Si", "property", {"band_gap": 1.1})
        results = tmp_memory.retrieve(
            query="band gap",
            formula="GaN",
            top_k=10,
            semantic=False,
        )
        assert len(results) >= 1
        assert all(r["formula"] == "GaN" for r in results)

    def test_recall_material_filter_miss(self, tmp_memory: LongTermMemory) -> None:
        tmp_memory.store_material("GaN", "property", {"band_gap": 3.4})
        results = tmp_memory.retrieve(
            query="band gap",
            formula="Si",
            top_k=10,
            semantic=False,
        )
        assert len(results) == 0

    def test_fts5_formula_search(self, tmp_memory: LongTermMemory) -> None:
        """FTS5 应该能通过 formula token 搜到材料记忆."""
        tmp_memory.store_material("GaN", "structure", {"lattice": "wurtzite"})
        results = tmp_memory.retrieve(query="GaN", top_k=10, semantic=False)
        assert len(results) >= 1
        assert any(r["formula"] == "GaN" for r in results)


class TestManagerMaterialFilter:
    """manager.recall_for_prompt 带 material_filter."""

    def test_manager_recall_with_material_filter(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            mgr = MemoryManager(
                longterm=LongTermMemory(
                    db_path=str(Path(d) / "mgr.db"), enable_semantic=False
                )
            )
            mgr.longterm.store_material("GaN", "property", {"band_gap": 3.4})
            mgr.longterm.store_material("Si", "property", {"band_gap": 1.1})
            text = mgr.recall_for_prompt(
                "band gap",
                material_filter={"formula": "GaN", "category": "property"},
            )
            assert "GaN" in text or "3.4" in text
            assert "1.1" not in text
