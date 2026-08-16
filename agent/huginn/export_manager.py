"""Export manager for Huginn records and data.

Allows users to export audit logs, remote job records, knowledge-base entries,
and workflow checkpoints to JSON, Markdown, or HTML so they can be shared,
archived, or inspected outside the agent.
"""

from __future__ import annotations

import csv
import io
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

        Supported formats: ``json``, ``markdown``, ``html``, ``csv``, ``xlsx``.
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        if source == "knowledge":
            # 知识库完整导出: chunk 级记录 + 汇总统计表.
            # xlsx 为二进制, 其余为文本.
            records, summary = self._collect_knowledge_full(**kwargs)
            rendered = self._render_knowledge(records, summary, fmt)
        else:
            records = self._collect(source, **kwargs)
            rendered = self._render(records, fmt, source)
        if isinstance(rendered, bytes):
            output.write_bytes(rendered)
        else:
            output.write_text(rendered, encoding="utf-8")

        return ExportResult(
            output_path=output,
            format=fmt,
            source=source,
            record_count=len(records),
        )

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
        # 兼容入口: 仅文档清单 (doc 级). 完整导出走 _collect_knowledge_full.
        try:
            from huginn.knowledge.store import KnowledgeBase
        except Exception:
            return []
        try:
            kb = KnowledgeBase(self.workspace / ".huginn_kb")
            return kb.list_documents()
        except Exception:
            return []

    def _collect_knowledge_full(
        self, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """知识库完整采集: chunk 级记录 + 汇总统计表.

        records 每条 = 一个 chunk (含正文/元数据/检索命中); summary 为
        文档/chunk 计数、按领域与来源类型分布、检索命中 Top 等.
        """
        from datetime import datetime

        try:
            from huginn.knowledge.store import KnowledgeBase, EMBED_MODEL
        except Exception:
            return [], self._empty_summary()

        try:
            kb = KnowledgeBase(self.workspace / ".huginn_kb")
        except Exception:
            return [], self._empty_summary()

        hit_counts: dict[str, int] = {}
        try:
            hit_counts = dict(getattr(kb, "_hit_counts", {}) or {})
        except Exception:
            pass

        try:
            data = kb.collection.get(include=["documents", "metadatas"])
        except Exception:
            return [], self._empty_summary()

        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []

        records: list[dict[str, Any]] = []
        by_domain: dict[str, int] = {}
        by_source: dict[str, int] = {}
        seen_docs: set[str] = set()
        for i, cid in enumerate(ids):
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            doc_id = str(meta.get("doc_id", ""))
            filename = str(meta.get("filename", "unknown"))
            domain = str(meta.get("domain", "") or "未分类")
            tags = meta.get("domain_tags", "")
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            if not isinstance(tags, list):
                tags = []
            by_domain[domain] = by_domain.get(domain, 0) + 1
            src = self._source_type(filename)
            by_source[src] = by_source.get(src, 0) + 1
            if doc_id:
                seen_docs.add(doc_id)
            records.append(
                {
                    "chunk_id": str(cid),
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": meta.get("chunk", i),
                    "section": meta.get("section", ""),
                    "domain": domain,
                    "domain_tags": tags,
                    "created_at": meta.get("created_at", ""),
                    "hit_count": hit_counts.get(doc_id, 0),
                    "text": docs[i] if i < len(docs) else "",
                }
            )

        # 检索命中 Top (按 doc 聚合)
        top_retrieved = sorted(
            ({"doc_id": d, "hits": h} for d, h in hit_counts.items()),
            key=lambda x: -x["hits"],
        )[:5]

        summary = {
            "total_documents": len(seen_docs),
            "total_chunks": len(ids),
            "by_domain": by_domain,
            "by_source_type": by_source,
            "total_retrieval_hits": sum(hit_counts.values()),
            "top_retrieved": top_retrieved,
            "embedding_model": EMBED_MODEL,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
        }
        return records, summary

    @staticmethod
    def _empty_summary() -> dict[str, Any]:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "by_domain": {},
            "by_source_type": {},
            "total_retrieval_hits": 0,
            "top_retrieved": [],
            "embedding_model": "",
            "exported_at": "",
        }

    @staticmethod
    def _source_type(filename: str) -> str:
        """按文件类型归类来源 (pdf/md/结构化/自动迭代/其他)."""
        low = filename.lower()
        if low.startswith("autoloop_iter_"):
            return "autoloop"
        if low.endswith((".cif", ".poscar", ".contcar")):
            return "structure"
        for ext, label in (
            (".pdf", "pdf"),
            (".md", "markdown"),
            (".docx", "docx"),
            (".xlsx", "xlsx"),
            (".txt", "text"),
        ):
            if low.endswith(ext):
                return label
        return "other"

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
        if fmt in ("csv", "xlsx"):
            raise ValueError(
                f"csv/xlsx 格式仅支持 knowledge source, 当前为 {source}"
            )
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

    def _render_knowledge(
        self,
        records: list[dict[str, Any]],
        summary: dict[str, Any],
        fmt: str,
    ) -> str | bytes:
        if fmt == "json":
            return json.dumps(
                {"summary": summary, "records": records},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        if fmt == "markdown":
            return self._render_knowledge_markdown(records, summary)
        if fmt == "html":
            return self._render_knowledge_html(records, summary)
        if fmt == "csv":
            return self._render_knowledge_csv(records, summary)
        if fmt == "xlsx":
            return self._render_knowledge_xlsx(records, summary)
        raise ValueError(f"Unknown export format: {fmt}")

    def _render_knowledge_markdown(
        self, records: list[dict[str, Any]], summary: dict[str, Any]
    ) -> str:
        lines = ["# Huginn Export: knowledge", "", "## Summary"]
        lines.append(f"- **Documents**: {summary['total_documents']}")
        lines.append(f"- **Chunks**: {summary['total_chunks']}")
        lines.append(f"- **Retrieval hits**: {summary['total_retrieval_hits']}")
        lines.append(f"- **Embedding model**: {summary['embedding_model']}")
        if summary["by_domain"]:
            lines.append("")
            lines.append("| Domain | Chunks |")
            lines.append("| --- | --- |")
            for dom, cnt in sorted(summary["by_domain"].items(), key=lambda x: -x[1]):
                lines.append(f"| {dom} | {cnt} |")
        if summary["top_retrieved"]:
            lines.append("")
            lines.append("### Top retrieved documents")
            for r in summary["top_retrieved"]:
                lines.append(f"- `{r['doc_id']}`: {r['hits']} hits")
        lines.append("")
        for i, rec in enumerate(records, 1):
            lines.append(f"## Chunk {i} ({rec['chunk_id']})")
            for key, value in rec.items():
                if key == "text":
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                lines.append(f"- **{key}**: {value}")
            lines.append("")
            lines.append("```text")
            lines.append(str(rec.get("text", "")))
            lines.append("```")
            lines.append("")
        return "\n".join(lines)

    def _render_knowledge_html(
        self, records: list[dict[str, Any]], summary: dict[str, Any]
    ) -> str:
        lines = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<meta charset='utf-8'>",
            "<title>Huginn Export: knowledge</title>",
            "</head><body>",
            "<h1>Huginn Export: knowledge</h1>",
            "<h2>Summary</h2><ul>",
        ]
        lines.append(f"<li><strong>Documents:</strong> {summary['total_documents']}</li>")
        lines.append(f"<li><strong>Chunks:</strong> {summary['total_chunks']}</li>")
        lines.append(
            f"<li><strong>Retrieval hits:</strong> {summary['total_retrieval_hits']}</li>"
        )
        lines.append(
            f"<li><strong>Embedding model:</strong> {summary['embedding_model']}</li>"
        )
        lines.append("</ul>")
        for dom, cnt in sorted(summary["by_domain"].items(), key=lambda x: -x[1]):
            lines.append(
                f"<p><strong>{self._escape_html(str(dom))}:</strong> {cnt} chunks</p>"
            )
        for i, rec in enumerate(records, 1):
            lines.append(f"<h2>Chunk {i} ({rec['chunk_id']})</h2><ul>")
            for key, value in rec.items():
                if key == "text":
                    continue
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                lines.append(
                    f"<li><strong>{key}:</strong> {self._escape_html(str(value))}</li>"
                )
            lines.append("</ul><pre>")
            lines.append(self._escape_html(str(rec.get("text", ""))))
            lines.append("</pre>")
        lines.append("</body></html>")
        return "\n".join(lines)

    def _render_knowledge_csv(
        self, records: list[dict[str, Any]], summary: dict[str, Any]
    ) -> str:
        """CSV: 汇总行作注释, 之后是 chunk 明细表."""
        buf = io.StringIO()
        buf.write(
            f"# Huginn knowledge summary: {summary['total_documents']} docs, "
            f"{summary['total_chunks']} chunks, "
            f"{summary['total_retrieval_hits']} hits\n"
        )
        if summary["by_domain"]:
            buf.write(
                "# domain chunks: "
                + ", ".join(f"{k}={v}" for k, v in summary["by_domain"].items())
                + "\n"
            )
        writer = csv.writer(buf)
        headers = [
            "chunk_id", "doc_id", "filename", "chunk_index", "section",
            "domain", "domain_tags", "created_at", "hit_count", "text",
        ]
        writer.writerow(headers)
        for rec in records:
            writer.writerow([
                rec["chunk_id"], rec["doc_id"], rec["filename"],
                rec["chunk_index"], rec["section"], rec["domain"],
                json.dumps(rec["domain_tags"], ensure_ascii=False),
                rec["created_at"], rec["hit_count"], rec["text"],
            ])
        return buf.getvalue()

    def _render_knowledge_xlsx(
        self, records: list[dict[str, Any]], summary: dict[str, Any]
    ) -> bytes:
        """XLSX (二进制): 两个 sheet — Summary 汇总表 + Records chunk 明细."""
        from io import BytesIO

        try:
            from openpyxl import Workbook
        except Exception:
            # 无 openpyxl 时降级为 CSV 文本
            return self._render_knowledge_csv(records, summary).encode("utf-8")

        wb = Workbook()
        ws = wb.active
        ws.title = "Summary"
        summary_rows = [
            ("Documents", summary["total_documents"]),
            ("Chunks", summary["total_chunks"]),
            ("Retrieval hits", summary["total_retrieval_hits"]),
            ("Embedding model", summary["embedding_model"]),
            ("Exported at", summary["exported_at"]),
        ]
        for k, v in summary_rows:
            ws.append([k, v])
        ws.append([])
        ws.append(["Domain", "Chunks"])
        for dom, cnt in sorted(summary["by_domain"].items(), key=lambda x: -x[1]):
            ws.append([dom, cnt])

        ws2 = wb.create_sheet("Records")
        ws2.append([
            "chunk_id", "doc_id", "filename", "chunk_index", "section",
            "domain", "domain_tags", "created_at", "hit_count", "text",
        ])
        for rec in records:
            ws2.append([
                rec["chunk_id"], rec["doc_id"], rec["filename"],
                rec["chunk_index"], rec["section"], rec["domain"],
                json.dumps(rec["domain_tags"], ensure_ascii=False),
                rec["created_at"], rec["hit_count"], rec["text"],
            ])

        buf = BytesIO()
        wb.save(buf)
        return buf.getvalue()

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
