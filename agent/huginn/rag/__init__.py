"""RAG (Retrieval-Augmented Generation) module.

Local vector database for material science knowledge retrieval.
"""

from huginn.rag.vector_store import VectorStore
from huginn.rag.rag_tool import RAGTool

__all__ = ["VectorStore", "RAGTool"]
