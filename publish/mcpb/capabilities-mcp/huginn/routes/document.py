"""Document understanding pipeline endpoints (M7).

Exposes the DocGraph pipeline as HTTP endpoints so a client can upload a
PDF and pull back structured information packages, the full graph, and
summary stats.

Pipeline run on upload:
  M1  PDFElementExtractor    parse PDF into typed elements
  M2  FigureDataExtractor    run plot_extract on chart figures
  M3  DocumentGraph          build the heterogeneous graph
  M4  RelationPredictor      inject predicted REFERENCES edges
  M5  CrossModalAdapter      extract CLAIM nodes + SUPPORTS/CONTRADICTS
  M6  InfoPackAssembler      bundle into InformationPackage units
  RAG RAGBridge              (optional) push packages into the KB

Endpoints:
  POST /document/parse                  upload + run full pipeline
  GET  /document/{document_id}/packages fetch assembled packages
  GET  /document/{document_id}/graph    fetch nodes + edges
  GET  /document/{document_id}/stats    fetch summary stats
  GET  /document/list                   list parsed documents
  POST /document/{document_id}/ingest   push packages into the KB

Parsed results live in an in-process dict -- fine for dev / single-user
setups. Swap for a real store when going multi-tenant.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from huginn.perception.rag_bridge import RAGBridge
from huginn.server_core import get_context

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["document"])

# 100 MB cap -- matches the knowledge upload limit. Papers with embedded
# high-res figures can get bulky, but anything past this is suspicious.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024

# In-memory store for parsed documents. The shape is deliberately simple:
# document_id -> {filename, graph, packages, stats}. Production would back
# this with a DB; the dict is just enough for a single-node dev box.
_document_store: dict[str, dict[str, Any]] = {}


def _is_pdf(content: bytes) -> bool:
    """Sniff the magic bytes -- every PDF starts with %PDF-."""
    return content[:5] == b"%PDF-"


def _try_ingest(record: dict[str, Any]) -> tuple[int, bool]:
    """Best-effort push the doc's packages into the KB.

    Returns (count_ingested, kb_configured). kb_configured=False means
    no KB is wired up -- callers can surface a clearer message in that
    case instead of a misleading zero. Ingest failures are logged and
    swallowed: parsing is the primary contract, KB is a side-effect.
    """
    try:
        from huginn.server_context import _server_context
        kb = getattr(_server_context, "kb", None)
    except Exception:
        logger.debug("best-effort op failed", exc_info=True)
        kb = None
    if kb is None:
        return 0, False
    bridge = RAGBridge(kb=kb)
    try:
        n = bridge.ingest(
            record["packages"],
            document_id=record["document_id"],
            filename=record["filename"],
        )
        return n, True
    except Exception as exc:
        logger.warning("KB ingest failed for %s: %s", record["document_id"], exc)
        return 0, True


@router.post("/parse")
async def parse_document(file: UploadFile = File(...)) -> StreamingResponse:
    """Upload a PDF, stream the DocGraph pipeline.

    Responds with Server-Sent Events. Every stage yields ``{type: stage, pct,
    message}``; the final event is ``{type: result, result}`` (or ``{type:
    error, error}`` on failure).

    The heavy M1-M6 + KB ingest runs in a *worker subprocess*
    (huginn.perception.doc_parse_worker). A native crash in a C extension
    (e.g. the VCRUNTIME140 access-violation we hit on real papers) then kills
    only the worker instead of the whole backend.
    """
    content = await file.read()

    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"file too large ({len(content)} bytes, "
                f"max {_MAX_UPLOAD_BYTES})"
            ),
        )
    if not _is_pdf(content):
        raise HTTPException(
            status_code=415,
            detail="only PDF files are accepted",
        )

    tmp_pdf = Path(tempfile.gettempdir()) / f"doc_{uuid.uuid4().hex}.pdf"
    out_json = Path(tempfile.gettempdir()) / f"doc_{uuid.uuid4().hex}.json"
    tmp_pdf.write_bytes(content)
    filename = file.filename or "uploaded.pdf"
    # 把 KB 存储位置传给 worker, 让自动入库也在子进程里做 (连 embedding 一起隔离).
    workspace = _get_workspace()

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "huginn.perception.doc_parse_worker",
        str(tmp_pdf),
        str(out_json),
        filename,
        workspace,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            **os.environ,
            # OpenBLAS 多线程栈分配失败会直接带崩进程 (我们实测到
            # "Memory allocation still failed"→VCRUNTIME140 崩). worker 是隔离的
            # 轻活, 单线程足够, 也降低那种原生崩的触发率.
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )

    async def event_stream():
        assert proc.stdout is not None and proc.stderr is not None
        try:
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip("\n")
                if text.startswith("PROGRESS "):
                    yield f"data: {text[len('PROGRESS '):]}\n\n"

            rc = await proc.wait()
            if rc != 0:
                err = (await proc.stderr.read()).decode("utf-8", errors="replace")
                detail = err.strip() or f"解析进程异常退出 (code {rc})"
                yield f"data: {json.dumps({'type': 'error', 'error': detail})}\n\n"
            else:
                record = json.loads(out_json.read_text(encoding="utf-8"))
                _document_store[record["document_id"]] = record
                result = {
                    "document_id": record["document_id"],
                    "filename": record["filename"],
                    "stats": record["stats"],
                    "n_packages": len(record["packages"]),
                    "packages": record["packages"],
                    "auto_ingested": record.get("auto_ingested", 0),
                }
                yield f"data: {json.dumps({'type': 'result', 'result': result}, ensure_ascii=False)}\n\n"
        finally:
            tmp_pdf.unlink(missing_ok=True)
            out_json.unlink(missing_ok=True)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _get_workspace() -> str:
    """后端单行 workspace 路径, 传给 worker 做 KB 入库; 没有就空串."""
    try:
        ws = get_context().config.workspace
    except Exception:
        return ""
    return str(ws) if ws else ""


@router.get("/{document_id}/packages")
async def get_packages(document_id: str) -> dict[str, Any]:
    """Get all information packages for a parsed document."""
    record = _document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    packages = record["packages"]
    return {
        "document_id": document_id,
        "n_packages": len(packages),
        "packages": packages,
    }


@router.get("/{document_id}/graph")
async def get_graph(document_id: str) -> dict[str, Any]:
    """Get the document graph (nodes + edges)."""
    record = _document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return record["graph"]


@router.get("/{document_id}/stats")
async def get_stats(document_id: str) -> dict[str, Any]:
    """Get document statistics."""
    record = _document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return record["stats"]


@router.get("/list")
async def list_documents() -> dict[str, Any]:
    """List all parsed documents."""
    docs = [
        {
            "document_id": doc_id,
            "filename": rec["filename"],
            "stats": rec["stats"],
        }
        for doc_id, rec in _document_store.items()
    ]
    return {"documents": docs, "count": len(docs)}


@router.post("/{document_id}/ingest")
async def ingest_to_kb(document_id: str) -> dict[str, Any]:
    """Push a document's info packages into the knowledge base.

    Requires a KnowledgeBase to be available in the server context.
    Returns the number of packages actually ingested. Note that /parse
    already auto-ingests when a KB is configured -- this endpoint is a
    manual re-trigger for cases where the auto path failed or KB came
    up after the parse.
    """
    record = _document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")

    n, kb_on = _try_ingest(record)
    if not kb_on:
        return {
            "document_id": document_id,
            "ingested": 0,
            "message": "no knowledge base configured",
        }
    return {
        "document_id": document_id,
        "ingested": n,
        "total_packages": len(record["packages"]),
    }


def _selfcheck() -> None:
    """Smoke-check _try_ingest's two branches without spinning up a real KB.

    Run with: python -c "from huginn.routes.document import _selfcheck; _selfcheck()"
    """
    from types import SimpleNamespace

    from huginn import server_context as sc

    # _try_ingest only reads sc._server_context.kb via getattr, so a
    # SimpleNamespace is enough -- no need to construct a full ServerContext.
    orig = sc._server_context
    record = {"document_id": "t", "filename": "t.pdf", "packages": []}

    try:
        # Case 1: no KB wired -> (0, False)
        sc._server_context = SimpleNamespace(kb=None)
        n, on = _try_ingest(record)
        assert (n, on) == (0, False), f"no-KB branch: expected (0, False), got ({n}, {on})"

        # Case 2: KB wired, empty packages -> RAGBridge short-circuits to (0, True)
        class _DummyKB:
            def add_document(self, *a, **kw): pass
        sc._server_context = SimpleNamespace(kb=_DummyKB())
        n, on = _try_ingest(record)
        assert (n, on) == (0, True), f"empty-packages branch: expected (0, True), got ({n}, {on})"
    finally:
        sc._server_context = orig

    print("document._try_ingest selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
