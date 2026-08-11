"""Tests for memory modules."""

import tempfile
from pathlib import Path

from huginn.core_types import AgentMessage, ToolResult
from huginn.memory.longterm import LongTermMemory
from huginn.memory.manager import MemoryConfig, MemoryManager
from huginn.memory.session import SessionContext, ToolCallRecord


class TestSessionContext:
    def test_add_message_and_compact(self):
        ctx = SessionContext(max_messages=5)
        for i in range(10):
            ctx.add_message(AgentMessage(role="user", content=f"msg{i}"))
        assert len(ctx.messages) <= 5
        assert ctx.messages[-1].content == "msg9"

    def test_tool_calls(self):
        ctx = SessionContext()
        ctx.add_tool_call(
            ToolCallRecord(tool_name="vasp_tool", input_args={"action": "relax"})
        )
        assert len(ctx.tool_calls) == 1
        assert ctx.tool_calls[0].tool_name == "vasp_tool"

    def test_to_dict(self):
        ctx = SessionContext()
        ctx.add_message(AgentMessage(role="user", content="hello"))
        d = ctx.to_dict()
        assert "session_id" in d
        assert d["message_count"] == 1


class TestLongTermMemory:
    def test_store_and_retrieve(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            entry_id = mem.store(
                "Ti has hcp structure", category="fact", tags=["Ti", "structure"]
            )
            assert entry_id.startswith("mem_")

            results = mem.retrieve("hcp structure")
            assert len(results) > 0
            assert "hcp" in results[0]["content"]

    def test_get_by_id_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            eid = mem.store("old content", importance=0.3)
            mem.update(eid, content="new content", importance=0.8)
            retrieved = mem.get_by_id(eid)
            assert retrieved["content"] == "new content"
            assert retrieved["importance"] == 0.8

    def test_delete_and_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            eid = mem.store("temp", importance=0.1)
            assert mem.delete(eid)
            assert mem.get_by_id(eid) is None

    def test_list_by_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            mem.store("fact1", category="fact")
            mem.store("fact2", category="fact")
            mem.store("insight1", category="insight")
            assert len(mem.list_by_category("fact")) == 2
            assert len(mem.list_by_category("insight")) == 1


class TestMemoryManager:
    def test_promote_tool_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            mgr = MemoryManager(
                longterm=mem, config=MemoryConfig(auto_promote_to_longterm=True)
            )
            mgr.add_tool_call(
                "vasp_tool",
                {"action": "relax"},
                result=ToolResult(data={"energy": -10.5}, success=True),
            )
            # Should have promoted to long-term
            facts = mem.list_by_category("calculation")
            assert len(facts) >= 1

    def test_recall_for_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = LongTermMemory(db_path=Path(tmp) / "memory.db")
            mgr = MemoryManager(longterm=mem)
            mgr.remember(
                "Si band gap is 1.1 eV", category="fact", tags=["Si", "band_gap"]
            )
            prompt = mgr.recall_for_prompt("band gap")
            assert "Si band gap" in prompt

    # ── 护栏: 蒸馏知识的 verification 闭环 ────────────────────────
    # 成功工具调用后, _verify_distilled_for_tool 必须真的把对应蒸馏知识
    # (source="distiller:{kid}") 升级为 confirmed, 否则它永远卡在 unverified,
    # 达不到 auto_ingest_to_kb 的准入门槛, 蒸馏→KB 一环形同虚设。

    def test_verify_distilled_promotes_to_confirmed(self, monkeypatch):
        from unittest.mock import MagicMock

        mgr = MemoryManager(longterm=MagicMock())
        mgr.longterm.retrieve.return_value = [
            {
                "id": "e1",
                "content": "vasp_tool convergence rule",
                "source": "distiller:kid_vasp",
            }
        ]
        # mock KnowledgeDistiller 类, 让方法内局部 import 拿到假的实例
        import huginn.evolution.knowledge_distiller as kd_mod

        mock_cls = MagicMock()
        mock_dist = mock_cls.return_value
        monkeypatch.setattr(kd_mod, "KnowledgeDistiller", mock_cls)

        mgr._verify_distilled_for_tool("vasp_tool", "...")

        # 蒸馏条目被 touch + verify_knowledge 升级为 confirmed
        mgr.longterm.touch.assert_called_once_with("e1")
        mock_dist.verify_knowledge.assert_called_once_with("kid_vasp", "confirmed")

    def test_verify_distilled_skips_non_distilled_source(self, monkeypatch):
        from unittest.mock import MagicMock

        mgr = MemoryManager(longterm=MagicMock())
        # source 不是 "distiller:" 前缀 (如普通 fact) → 不升级, 只 touch
        mgr.longterm.retrieve.return_value = [
            {"id": "e2", "content": "vasp_tool rule", "source": "session:s1"}
        ]
        import huginn.evolution.knowledge_distiller as kd_mod

        mock_cls = MagicMock()
        mock_dist = mock_cls.return_value
        monkeypatch.setattr(kd_mod, "KnowledgeDistiller", mock_cls)

        mgr._verify_distilled_for_tool("vasp_tool", "...")

        mgr.longterm.touch.assert_called_once_with("e2")
        mock_dist.verify_knowledge.assert_not_called()
