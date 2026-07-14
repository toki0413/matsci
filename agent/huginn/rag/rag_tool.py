"""RAG Tool for Agent — semantic search and document ingestion.

Supports both plaintext and encrypted storage modes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.rag.vector_store import EncryptedVectorStore, VectorStore
from huginn.security.prompt_security import wrap_rag_chunks
from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult
import logging
logger = logging.getLogger(__name__)



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
        kb: Any | None = None,
    ):
        super().__init__()
        # 共享 KnowledgeBase 模式：传入 KB 实例后直接复用它的 ChromaDB
        # collection，不再单独建 VectorStore。这样 agent 的 RAG 和 REST
        # 上传接口读写同一份数据，两边互通。
        self._kb = kb
        if kb is not None:
            self.store = None
        elif vault is not None and vault.is_unlocked():
            self.store = EncryptedVectorStore(
                vault=vault,
                persist_dir=persist_dir,
                encrypt_documents=encrypt_documents,
                encrypt_metadata=encrypt_metadata,
            )
        else:
            # ponytail: 用独立 collection, 不和 LongTermMemory 的 huginn_knowledge
            # 混在一起. 否则 KB 不可用时文档块会污染 agent 记忆检索结果.
            # 生产路径下 KB 总会注入, 这条 fallback 只在 KB 初始化失败/CLI
            # 单跑时触发, 但一旦触发不能写错地方.
            self.store = VectorStore(
                persist_dir=persist_dir, collection_name="huginn_rag"
            )

        # 层次化路由检索：识别 VASP/LAMMPS/Gaussian 等14种软件 + 17种方法关键词
        # 命中路由时在对应子索引里优先搜，提升精度；没命中就退回普通语义搜索
        # 只在 VectorStore 模式下可用；共享 KB 模式有自己的分块/嵌入流程
        self._hierarchical: Any | None = None
        if self.store is not None:
            try:
                from huginn.rag.router_retriever import HierarchicalRetriever

                self._hierarchical = HierarchicalRetriever(self.store)
            except Exception as e:
                # 路由检索不可用就走原来的路径，不影响主流程
                print(f"[rag_tool] hierarchical retriever unavailable: {e}")

    def _resolve_kb(self) -> Any | None:
        """返回当前共享的 KnowledgeBase，必要时延迟绑定。

        KB 在 server lifespan 阶段才创建——晚于工具注册。如果注册时
        没有拿到 KB，就在第一次实际调用时从 server context 里取，
        保证 agent 搜索和前端上传落在同一个 collection 上。
        """
        if self._kb is not None:
            return self._kb
        try:
            from huginn.server_core import get_context

            kb = get_context().kb
            if kb is not None:
                self._kb = kb
                return kb
        except Exception:
            logger.debug("resolve kb failed", exc_info=True)
        return None

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

        # 共享 KB 路径：跳过层次化路由，直接走 KB 的 query
        kb = self._resolve_kb()
        if kb is not None:
            try:
                chunks = kb.query(args.query, top_k=args.top_k)
                # KB 返回 {chunk_id, text, metadata, distance}，
                # 映射成 VectorStore.search 的格式让调用方无感知
                results = [
                    {
                        "id": c["chunk_id"],
                        "document": c["text"],
                        "metadata": c["metadata"],
                        "distance": c["distance"],
                    }
                    for c in chunks
                ]
                # 视觉压缩页 chunk 的 image_ref 是裸文件系统路径, 前端拿不到.
                # 同时塞一份 image_url 让前端按 URL 直接 fetch (GET /knowledge/image).
                # 之前 context_builder 只把路径塞进 prompt 文本, 前端永远看不到图.
                from urllib.parse import quote
                for r in results:
                    img_ref = (r.get("metadata") or {}).get("image_ref")
                    if img_ref:
                        r["image_url"] = f"/knowledge/image?path={quote(str(img_ref))}"
                # ponytail: 外部检索到的 chunk 是不可信内容, LLM 可能被注入
                # "ignore previous instructions" 之类的文本. 用显式标记包起来,
                # 让模型把这段当数据而非指令 (Odysseus untrusted_context_message 模式)
                wrap_rag_chunks(results)
                return ToolResult(
                    data={
                        "query": args.query,
                        "results_count": len(results),
                        "results": results,
                    },
                    success=True,
                )
            except Exception as e:
                return ToolResult(
                    data=None, success=False, error=f"Search failed: {e}"
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
                wrap_rag_chunks(results)
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
            wrap_rag_chunks(results)
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
        kb = self._resolve_kb()
        if kb is not None:
            if args.file_path:
                path = Path(args.file_path)
                if not path.exists():
                    return ToolResult(
                        data=None, success=False, error=f"File not found: {path}"
                    )
                try:
                    info = kb.add_document(path.name, path.read_bytes())
                    return ToolResult(
                        data={
                            "ingested": info["chunks"],
                            "ids": [info["doc_id"]],
                            "source": str(path),
                        },
                        success=True,
                    )
                except Exception as e:
                    return ToolResult(
                        data=None, success=False, error=f"Ingest failed: {e}"
                    )
            elif args.document:
                try:
                    info = kb.add_document(
                        "manual.txt", args.document.encode("utf-8")
                    )
                    return ToolResult(
                        data={
                            "ingested": info["chunks"],
                            "ids": [info["doc_id"]],
                        },
                        success=True,
                    )
                except Exception as e:
                    return ToolResult(
                        data=None, success=False, error=f"Ingest failed: {e}"
                    )
            return ToolResult(
                data=None,
                success=False,
                error="file_path or document required for ingest",
            )

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
        kb = self._resolve_kb()
        if kb is not None:
            try:
                docs = kb.list_documents()
                # KB 返回 [{doc_id, filename}]，补成 VectorStore.list_documents
                # 的字段结构，调用方拿到的 schema 保持一致
                formatted = [
                    {"id": d["doc_id"], "metadata": {"filename": d["filename"]}}
                    for d in docs
                ]
                return ToolResult(
                    data={"total": kb.count(), "documents": formatted},
                    success=True,
                )
            except Exception as e:
                return ToolResult(data=None, success=False, error=f"List failed: {e}")

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

        kb = self._resolve_kb()
        if kb is not None:
            try:
                # KB 没有单文档查询接口，直接拿底层 ChromaDB collection
                result = kb.collection.get(
                    ids=[args.doc_id],
                    include=["documents", "metadatas"],
                )
                if not result["ids"]:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"Document not found: {args.doc_id}",
                    )
                return ToolResult(
                    data={
                        "id": result["ids"][0],
                        "document": result["documents"][0],
                        "metadata": result["metadatas"][0],
                    },
                    success=True,
                )
            except Exception as e:
                return ToolResult(data=None, success=False, error=f"Get failed: {e}")

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

        kb = self._resolve_kb()
        if kb is not None:
            try:
                # KB.delete_document 只接受单个 doc_id，逐个调
                deleted = sum(
                    1 for doc_id in args.doc_ids if kb.delete_document(doc_id)
                )
                return ToolResult(data={"deleted": deleted}, success=True)
            except Exception as e:
                return ToolResult(data=None, success=False, error=f"Delete failed: {e}")

        try:
            self.store.delete(args.doc_ids)
            return ToolResult(
                data={"deleted": len(args.doc_ids)},
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Delete failed: {e}")

    def _count(self) -> ToolResult:
        kb = self._resolve_kb()
        if kb is not None:
            try:
                return ToolResult(data={"total_documents": kb.count()}, success=True)
            except Exception as e:
                return ToolResult(data=None, success=False, error=f"Count failed: {e}")

        try:
            count = self.store.count()
            return ToolResult(data={"total_documents": count}, success=True)
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Count failed: {e}")
