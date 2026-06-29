"""RAG Tool for Agent — semantic search and document ingestion.

Supports both plaintext and encrypted storage modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.rag.vector_store import EncryptedVectorStore, VectorStore
from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult


class RAGToolInput(BaseModel):
    action: Literal["search", "ingest", "list", "delete", "count", "get"] = Field(...)
    query: str = Field(default="", description="Search query (for search action)")
    file_path: str | None = Field(
        default=None, description="File to ingest (for ingest action)"
    )
    document: str | None = Field(
        default=None, description="Raw text to ingest (for ingest action)"
    )
    doc_id: str | None = Field(
        default=None, description="Document ID to get (for get action)"
    )
    doc_ids: list[str] = Field(
        default_factory=list, description="Document IDs to delete"
    )
    top_k: int = Field(default=5, ge=1, le=20)
    source_filter: str | None = Field(
        default=None, description="Filter by source (e.g., 'vasp_manual')"
    )


class RAGTool(HuginnTool):
    """Retrieve material science knowledge from local vector database."""

    name = "rag_tool"
    category = "search"
    profile = ToolProfile(cost_tier="light")
    description = "Search and ingest material science documents into a local knowledge base for RAG"
    input_schema = RAGToolInput

    def __init__(
        self,
        persist_dir: str | None = None,
        vault: Any | None = None,
        encrypt_documents: bool = True,
        encrypt_metadata: bool = True,
    ):
        super().__init__()
        if vault is not None and vault.is_unlocked():
            self.store = EncryptedVectorStore(
                vault=vault,
                persist_dir=persist_dir,
                encrypt_documents=encrypt_documents,
                encrypt_metadata=encrypt_metadata,
            )
        else:
            self.store = VectorStore(persist_dir=persist_dir)

        # 层次化路由检索：识别 VASP/LAMMPS/Gaussian 等14种软件 + 17种方法关键词
        # 命中路由时在对应子索引里优先搜，提升精度；没命中就退回普通语义搜索
        self._hierarchical: Any | None = None
        try:
            from huginn.rag.router_retriever import HierarchicalRetriever

            self._hierarchical = HierarchicalRetriever(self.store)
        except Exception as e:
            # 路由检索不可用就走原来的路径，不影响主流程
            print(f"[rag_tool] hierarchical retriever unavailable: {e}")

    def is_read_only(self, args: RAGToolInput) -> bool:
        return args.action in ["search", "list", "count", "get"]

    def is_destructive(self, args: RAGToolInput) -> bool:
        return args.action == "delete"

    async def call(self, args: RAGToolInput, context: ToolContext) -> ToolResult:
        if args.action == "search":
            return self._search(args)
        elif args.action == "ingest":
            return self._ingest(args)
        elif args.action == "list":
            return self._list_docs()
        elif args.action == "delete":
            return self._delete(args)
        elif args.action == "count":
            return self._count()
        elif args.action == "get":
            return self._get(args)

        return ToolResult(
            data=None, success=False, error=f"Unknown action: {args.action}"
        )

    def _search(self, args: RAGToolInput) -> ToolResult:
        if not args.query:
            return ToolResult(
                data=None, success=False, error="query is required for search"
            )

        filter_dict = None
        if args.source_filter:
            filter_dict = {"source": args.source_filter}

        # 优先走层次化路由检索（带软件/方法识别），失败再退回普通语义搜索
        if self._hierarchical is not None:
            try:
                routed = self._hierarchical.search(
                    query=args.query,
                    top_k=args.top_k,
                    filter_dict=filter_dict,
                )
                results = routed.get("results", [])
                return ToolResult(
                    data={
                        "query": args.query,
                        "results_count": len(results),
                        "results": results,
                        "route": routed.get("route"),
                        "routing_reason": routed.get("routing_reason"),
                        "targeted_count": routed.get("targeted_count", 0),
                        "general_count": routed.get("general_count", 0),
                    },
                    success=True,
                )
            except Exception as e:
                # 路由检索挂了就 fallback，不能让用户拿不到结果
                print(f"[rag_tool] hierarchical search failed, fallback: {e}")

        try:
            results = self.store.search(
                query=args.query,
                top_k=args.top_k,
                filter_dict=filter_dict,
            )

            return ToolResult(
                data={
                    "query": args.query,
                    "results_count": len(results),
                    "results": results,
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Search failed: {e}")

    def _ingest(self, args: RAGToolInput) -> ToolResult:
        if args.file_path:
            path = Path(args.file_path)
            if not path.exists():
                return ToolResult(
                    data=None, success=False, error=f"File not found: {path}"
                )

            try:
                ids = self.store.ingest_file(str(path))
                return ToolResult(
                    data={
                        "ingested": len(ids),
                        "ids": ids[:10],
                        "source": str(path),
                    },
                    success=True,
                )
            except Exception as e:
                return ToolResult(data=None, success=False, error=f"Ingest failed: {e}")

        elif args.document:
            try:
                ids = self.store.ingest(
                    [args.document], metadatas=[{"source": "manual"}]
                )
                return ToolResult(
                    data={"ingested": len(ids), "ids": ids},
                    success=True,
                )
            except Exception as e:
                return ToolResult(data=None, success=False, error=f"Ingest failed: {e}")

        return ToolResult(
            data=None, success=False, error="file_path or document required for ingest"
        )

    def _list_docs(self) -> ToolResult:
        try:
            docs = self.store.list_documents(limit=100)
            return ToolResult(
                data={"total": self.store.count(), "documents": docs},
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"List failed: {e}")

    def _get(self, args: RAGToolInput) -> ToolResult:
        if not args.doc_id:
            return ToolResult(
                data=None, success=False, error="doc_id is required for get"
            )
        try:
            doc = self.store.get_document(args.doc_id)
            if doc is None:
                return ToolResult(
                    data=None, success=False, error=f"Document not found: {args.doc_id}"
                )
            return ToolResult(data=doc, success=True)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Get failed: {e}")

    def _delete(self, args: RAGToolInput) -> ToolResult:
        if not args.doc_ids:
            return ToolResult(
                data=None, success=False, error="doc_ids required for delete"
            )

        try:
            self.store.delete(args.doc_ids)
            return ToolResult(
                data={"deleted": len(args.doc_ids)},
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Delete failed: {e}")

    def _count(self) -> ToolResult:
        try:
            count = self.store.count()
            return ToolResult(data={"total_documents": count}, success=True)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Count failed: {e}")
