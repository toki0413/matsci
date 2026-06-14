#!/usr/bin/env python3
"""
Ingest Sobko cleaned database into Huginn's RAG system.

This script reads the cleaned Sobko_MCP_project database and imports
chunks into the Huginn vector store for local knowledge retrieval.

Usage:
    cd agent && python ../scripts/ingest_sobko_to_rag.py --sobko-path ../../Sobko_MCP_project --persist-dir ./sobko_rag_db

Requirements:
    - Huginn package installed (pip install -e ./agent)
    - chromadb or the vector store backend configured in huginn.rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Ingest Sobko database into Huginn RAG")
    parser.add_argument(
        "--sobko-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "Sobko_MCP_project",
        help="Path to Sobko_MCP_project root",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("./sobko_rag_db"),
        help="Directory to persist the vector store",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5000,
        help="Maximum number of chunks to ingest (default: 5000)",
    )
    parser.add_argument(
        "--authority-priority",
        action="store_true",
        default=True,
        help="Prioritize authority-A sources first",
    )
    args = parser.parse_args()

    cleaned_dir = args.sobko_path / "cleaned"
    if not cleaned_dir.exists():
        print(f"ERROR: Cleaned database not found at {cleaned_dir}")
        print("Run Sobko_MCP_project/scripts/clean_database.py first.")
        sys.exit(1)

    # Load data
    print(f"Loading Sobko data from {cleaned_dir} ...")
    sources = load_jsonl(cleaned_dir / "source_registry.jsonl")
    chunks = load_jsonl(cleaned_dir / "chunks.jsonl")
    print(f"  Sources: {len(sources)}")
    print(f"  Chunks: {len(chunks)}")

    source_by_id = {s["source_id"]: s for s in sources}

    # Sort chunks: authority A first, then by text richness
    if args.authority_priority:
        chunks.sort(
            key=lambda c: (
                0 if c.get("authority_level") == "A" else 1,
                -len(c.get("text", "")),
            )
        )

    # Prepare documents for ingestion
    documents = []
    metadatas = []
    ids = []

    for i, chunk in enumerate(chunks[: args.max_chunks]):
        chunk_id = chunk.get("chunk_id", f"chunk_{i}")
        source_id = chunk.get("source_id", "unknown")
        source = source_by_id.get(source_id, {})

        text = chunk.get("text", "").strip()
        if not text:
            continue

        # Enrich text with metadata for better retrieval
        enriched = f"[{source.get('title', 'Unknown')}]\n"
        section = " > ".join(chunk.get("section_path", [])[1:])
        if section:
            enriched += f"Section: {section}\n"
        enriched += f"Tags: {', '.join(chunk.get('software_tags', []))} | {', '.join(chunk.get('topic_tags', []))}\n"
        enriched += f"---\n{text}"

        documents.append(enriched)
        metadatas.append(
            {
                "source_id": source_id,
                "chunk_id": chunk_id,
                "title": source.get("title", ""),
                "authority": chunk.get("authority_level", ""),
                "software_tags": chunk.get("software_tags", []),
                "topic_tags": chunk.get("topic_tags", []),
                "method_tags": chunk.get("method_tags", []),
                "url": source.get("canonical_url", ""),
            }
        )
        ids.append(chunk_id)

    print(f"Prepared {len(documents)} documents for ingestion")

    # Import huginn RAG
    try:
        from huginn.rag.vector_store import VectorStore
    except ImportError as e:
        print(f"ERROR: Cannot import Huginn RAG: {e}")
        print("Make sure huginn is installed: pip install -e ./agent")
        sys.exit(1)

    # Ingest
    print(f"Ingesting into vector store at {args.persist_dir} ...")
    store = VectorStore(persist_dir=str(args.persist_dir))
    store.ingest(documents, metadatas=metadatas, ids=ids)

    print(f"Done! Total documents in store: {store.count()}")
    print(f"\nYou can now use rag_tool with source_filter='sobko' to search this knowledge base.")

    # Print sample queries
    print("\nSample queries to test:")
    print('  rag_tool(action="search", query="福井函数 亲电反应位点", top_k=5)')
    print('  rag_tool(action="search", query="IGMH 弱相互作用可视化", top_k=5)')
    print('  rag_tool(action="search", query="RESP 电荷拟合步骤", top_k=5)')


if __name__ == "__main__":
    main()
