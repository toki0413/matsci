"""RAG (Retrieval-Augmented Generation) module.

Local vector database for material science knowledge retrieval.
"""

from matsci_agent.rag.vector_store import VectorStore
from matsci_agent.rag.rag_tool import RAGTool

__all__ = ["VectorStore", "RAGTool"]
