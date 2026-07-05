"""RAG (Retrieval-Augmented Generation) module.

Local vector database for material science knowledge retrieval.
"""

from huginn.rag.rag_tool import RAGTool
from huginn.rag.vector_store import VectorStore, ZvecBackend, create_vector_store

__all__ = ["VectorStore", "ZvecBackend", "create_vector_store", "RAGTool"]
