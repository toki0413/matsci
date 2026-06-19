"""Tests for the export manager."""

from __future__ import annotations

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
