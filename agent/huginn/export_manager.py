"""Export manager for Huginn records and data.

Allows users to export audit logs, remote job records, knowledge-base entries,
and workflow checkpoints to JSON, Markdown, or HTML so they can be shared,
archived, or inspected outside the agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExportResult:
    """Result of an export operation."""

    output_path: Path
    format: str
    source: str
    record_count: int


class ExportManager:
    """Collect Huginn records and serialize them to portable formats."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()

    def export(
        self,
        source: str,
        output_path: str | Path,
        fmt: str = "json",
        **kwargs: Any,
    ) -> ExportResult:
        """Export a data source to the requested format.

        Supported sources:
            - ``audit``: ``huginn_audit.jsonl`` in the workspace
            - ``remote_jobs``: persisted remote HPC job records
            - ``knowledge``: knowledge-base document list
            - ``checkpoints``: workflow checkpoint files

        Supported formats: ``json``, ``markdown``, ``html``.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        records = self._collect(source, **kwargs)
        rendered = self._render(records, fmt, source)
        output.write_text(rendered, encoding="utf-8")

        return ExportResult(
            output_path=output,
            format=fmt,
            source=source,
            record_count=len(records),
        )

    def list_sources(self) -> list[str]:
        """Return available export sources."""
        return ["audit", "remote_jobs", "knowledge", "checkpoints"]

    def _collect(self, source: str, **kwargs: Any) -> list[dict[str, Any]]:
        if source == "audit":
            return self._collect_audit(**kwargs)
        if source == "remote_jobs":
            return self._collect_remote_jobs(**kwargs)
        if source == "knowledge":
            return self._collect_knowledge(**kwargs)
        if source == "checkpoints":
            return self._collect_checkpoints(**kwargs)
        raise ValueError(f"Unknown export source: {source}")

    def _collect_audit(
        self, log_path: str | Path | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        path = Path(log_path) if log_path else self.workspace / "huginn_audit.jsonl"
        if not path.exists():
            return []
        records = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def _collect_remote_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        from huginn.execution.remote_job_store import RemoteJobStore

        store = RemoteJobStore(workspace=self.workspace)
        return [self._job_to_dict(r) for r in store.load()]

    def _collect_knowledge(self, **kwargs: Any) -> list[dict[str, Any]]:
        try:
            from huginn.knowledge.store import KnowledgeBase
        except Exception:
            return []
        try:
            kb = KnowledgeBase(self.workspace / ".huginn_kb")
            return kb.list_documents()
        except Exception:
            return []

    def _collect_checkpoints(self, **kwargs: Any) -> list[dict[str, Any]]:
        pattern = kwargs.get("pattern", "*.json")
        checkpoint_dir = self.workspace / ".huginn_checkpoints"
        if not checkpoint_dir.exists():
            return []
        records = []
        for path in sorted(checkpoint_dir.glob(pattern)):
            try:
                records.append(
                    {
                        "filename": path.name,
                        "modified": path.stat().st_mtime,
                        "size_bytes": path.stat().st_size,
                    }
                )
            except Exception:
                continue
        return records

    @staticmethod
    def _job_to_dict(job: Any) -> dict[str, Any]:
        """Serialize a RemoteJobRecord to a plain dict."""
        from dataclasses import asdict

        try:
            return asdict(job)
        except Exception:
            return {"raw": str(job)}

    def _render(self, records: list[dict[str, Any]], fmt: str, source: str) -> str:
        if fmt == "json":
            return json.dumps(records, indent=2, ensure_ascii=False, default=str)
        if fmt == "markdown":
            return self._render_markdown(records, source)
        if fmt == "html":
            return self._render_html(records, source)
        raise ValueError(f"Unknown export format: {fmt}")

    def _render_markdown(self, records: list[dict[str, Any]], source: str) -> str:
        lines = [f"# Huginn Export: {source}", ""]
        for i, record in enumerate(records, 1):
            lines.append(f"## Record {i}")
            for key, value in record.items():
                if isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                lines.append(f"- **{key}**: {value}")
            lines.append("")
        return "\n".join(lines)

    def _render_html(self, records: list[dict[str, Any]], source: str) -> str:
        lines = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>Huginn Export: {source}</title>",
            "</head><body>",
            f"<h1>Huginn Export: {source}</h1>",
        ]
        for i, record in enumerate(records, 1):
            lines.append(f"<h2>Record {i}</h2><ul>")
            for key, value in record.items():
                if isinstance(value, dict):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                lines.append(
                    f"<li><strong>{key}:</strong> {self._escape_html(str(value))}</li>"
                )
            lines.append("</ul>")
        lines.append("</body></html>")
        return "\n".join(lines)

    @staticmethod
    def _escape_html(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
