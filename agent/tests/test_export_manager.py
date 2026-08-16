"""Tests for the export manager."""

from __future__ import annotations

import csv
import io
import json

from huginn.export_manager import ExportManager


class TestExportManager:
    def test_export_audit_to_json(self, tmp_path):
        audit = tmp_path / "huginn_audit.jsonl"
        audit.write_text(
            json.dumps({"event_type": "tool_call", "actor": "user"}) + "\n",
            encoding="utf-8",
        )

        manager = ExportManager(tmp_path)
        result = manager.export("audit", tmp_path / "audit.json", fmt="json")

        assert result.record_count == 1
        assert result.output_path.exists()
        data = json.loads(result.output_path.read_text(encoding="utf-8"))
        assert data[0]["event_type"] == "tool_call"

    def test_export_audit_to_markdown(self, tmp_path):
        audit = tmp_path / "huginn_audit.jsonl"
        audit.write_text(
            json.dumps({"event_type": "tool_call", "actor": "user"}) + "\n",
            encoding="utf-8",
        )

        manager = ExportManager(tmp_path)
        result = manager.export("audit", tmp_path / "audit.md", fmt="markdown")

        assert result.record_count == 1
        text = result.output_path.read_text(encoding="utf-8")
        assert "# Huginn Export: audit" in text

    def test_export_checkpoints(self, tmp_path):
        checkpoint_dir = tmp_path / ".huginn_checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "cp1.json").write_text("{}", encoding="utf-8")

        manager = ExportManager(tmp_path)
        result = manager.export("checkpoints", tmp_path / "checkpoints.json")

        assert result.record_count == 1
        data = json.loads(result.output_path.read_text(encoding="utf-8"))
        assert data[0]["filename"] == "cp1.json"

    def test_export_remote_jobs_empty(self, tmp_path):
        manager = ExportManager(tmp_path)
        result = manager.export("remote_jobs", tmp_path / "jobs.json")
        assert result.record_count == 0

    def test_unknown_source_raises(self, tmp_path):
        manager = ExportManager(tmp_path)
        try:
            manager.export("unknown", tmp_path / "out.json")
        except ValueError as exc:
            assert "Unknown export source" in str(exc)


class _FakeKbCollection:
    def get(self, include=None):
        return {
            "ids": ["c1", "c2", "c3"],
            "documents": ["文本A-1", "文本A-2", "文本B"],
            "metadatas": [
                {"doc_id": "doc_a", "filename": "paper_a.pdf",
                 "domain": "materials", "domain_tags": '["合金"]', "chunk": 0},
                {"doc_id": "doc_a", "filename": "paper_a.pdf",
                 "domain": "materials", "domain_tags": '["合金"]', "chunk": 1},
                {"doc_id": "doc_b", "filename": "autoloop_iter_3.md",
                 "domain": "ml", "domain_tags": "[]", "chunk": 0},
            ],
        }


class _FakeKb:
    collection = _FakeKbCollection()
    _hit_counts = {"doc_a": 5, "doc_b": 2}


class TestKnowledgeFullExport:
    def _manager(self, tmp_path) -> ExportManager:
        import huginn.knowledge.store as store
        store.KnowledgeBase = lambda root: _FakeKb()
        return ExportManager(tmp_path)

    def test_knowledge_json_includes_summary_and_records(self, tmp_path):
        manager = self._manager(tmp_path)
        result = manager.export("knowledge", tmp_path / "kb.json", fmt="json")
        data = json.loads(result.output_path.read_text(encoding="utf-8"))
        assert data["summary"]["total_documents"] == 2
        assert data["summary"]["total_chunks"] == 3
        assert data["summary"]["by_domain"] == {"materials": 2, "ml": 1}
        assert data["summary"]["by_source_type"] == {"pdf": 2, "autoloop": 1}
        assert data["summary"]["top_retrieved"][0]["doc_id"] == "doc_a"
        assert len(data["records"]) == 3
        assert data["records"][0]["text"] == "文本A-1"
        assert data["records"][2]["hit_count"] == 2

    def test_knowledge_markdown_has_summary_table(self, tmp_path):
        manager = self._manager(tmp_path)
        result = manager.export("knowledge", tmp_path / "kb.md", fmt="markdown")
        text = result.output_path.read_text(encoding="utf-8")
        assert "## Summary" in text
        assert "| Domain | Chunks |" in text
        assert "paper_a.pdf" in text

    def test_knowledge_csv_has_comment_and_header(self, tmp_path):
        manager = self._manager(tmp_path)
        result = manager.export("knowledge", tmp_path / "kb.csv", fmt="csv")
        text = result.output_path.read_text(encoding="utf-8")
        assert text.startswith("# Huginn knowledge summary: 2 docs")
        rows = list(csv.reader(io.StringIO(text)))
        # 过滤注释行后: 表头 + 3 chunk
        data_rows = [r for r in rows if r and not r[0].startswith("#")]
        assert data_rows[0][0] == "chunk_id"
        assert len(data_rows) == 4
        assert data_rows[1][9] == "文本A-1"

    def test_knowledge_xlsx_has_two_sheets(self, tmp_path):
        import pytest
        openpyxl = pytest.importorskip("openpyxl")
        manager = self._manager(tmp_path)
        result = manager.export("knowledge", tmp_path / "kb.xlsx", fmt="xlsx")
        wb = openpyxl.load_workbook(io.BytesIO(result.output_path.read_bytes()))
        assert wb.sheetnames == ["Summary", "Records"]
        assert wb["Records"].max_row == 4

    def test_non_knowledge_csv_raises_clear_error(self, tmp_path):
        manager = self._manager(tmp_path)
        try:
            manager.export("audit", tmp_path / "a.csv", fmt="csv")
        except ValueError as exc:
            assert "仅支持 knowledge" in str(exc)
