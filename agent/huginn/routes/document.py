"""Document understanding pipeline endpoints (M7).

Exposes the DocGraph pipeline as HTTP endpoints so a client can upload a
PDF and pull back structured information packages, the full graph, and
summary stats.

Pipeline run on upload:
  M1  PDFElementExtractor    parse PDF into typed elements
  M3  DocumentGraph          build the heterogeneous graph
  M4  RelationPredictor      inject predicted REFERENCES edges
  M5  CrossModalAdapter      extract CLAIM nodes + SUPPORTS/CONTRADICTS
  M6  InfoPackAssembler      bundle into InformationPackage units

Endpoints:
  POST /document/parse                  upload + run full pipeline
  GET  /document/{document_id}/packages fetch assembled packages
  GET  /document/{document_id}/graph    fetch nodes + edges
  GET  /document/{document_id}/stats    fetch summary stats
  GET  /document/list                   list parsed documents

Parsed results live in an in-process dict -- fine for dev / single-user
setups. Swap for a real store when going multi-tenant.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from huginn.perception.cross_validator import CrossModalAdapter
from huginn.perception.document_graph import DocumentGraph
from huginn.perception.info_pack import InfoPackAssembler
from huginn.perception.pdf_parser import PDFElementExtractor
from huginn.perception.relation_predictor import RelationPredictor

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


@router.post("/parse")
async def parse_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a PDF and run the full DocGraph pipeline.

    Returns the document_id along with a summary of the extracted
    information packages.
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

    # spill to a temp file so PyMuPDF can mmap it rather than holding the
    # whole thing in memory twice.
    tmp_path = Path(tempfile.gettempdir()) / f"doc_{uuid.uuid4().hex}.pdf"
    try:
        tmp_path.write_bytes(content)

        # M1: parse PDF into elements
        extractor = PDFElementExtractor()
        elements = extractor.extract(tmp_path)

        # M3: build the heterogeneous graph (structural edges only)
        graph = DocumentGraph(elements)

        # M4: predict REFERENCES edges (mention -> figure/table)
        RelationPredictor().predict(graph)

        # M5: cross-modal validation (injects CLAIM nodes + verdict edges)
        CrossModalAdapter().process(graph)

        # M6: assemble information packages
        packages = InfoPackAssembler().assemble(graph)
    finally:
        # always clean up the temp file, even if the pipeline blew up
        tmp_path.unlink(missing_ok=True)

    document_id = uuid.uuid4().hex
    record: dict[str, Any] = {
        "document_id": document_id,
        "filename": file.filename or "uploaded.pdf",
        "graph": graph,
        "packages": packages,
        "stats": graph.stats(),
    }
    _document_store[document_id] = record

    return {
        "document_id": document_id,
        "filename": record["filename"],
        "stats": record["stats"],
        "n_packages": len(packages),
        "packages": [p.to_dict() for p in packages],
    }


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
        "packages": [p.to_dict() for p in packages],
    }


@router.get("/{document_id}/graph")
async def get_graph(document_id: str) -> dict[str, Any]:
    """Get the document graph (nodes + edges)."""
    record = _document_store.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return record["graph"].to_dict()


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
