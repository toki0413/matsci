"""Memory backend abstraction for long-term memory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryBackend(ABC):
    """Abstract backend for long-term memory storage and retrieval."""

    @abstractmethod
    def store(
        self,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        source: str = "",
        importance: float = 0.5,
        tier: str = "mid",
        ttl_hours: float | None = None,
    ) -> str:
        """Store a memory entry and return its ID."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        category: str | None = None,
        tier: str | None = None,
        top_k: int = 5,
        semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """Retrieve memories matching the query."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry by ID."""
        raise NotImplementedError


class SQLiteMemoryBackend(MemoryBackend):
    """SQLite-backed memory backend.

    This is a thin wrapper that instantiates ``LongTermMemory`` so we can keep
    the existing SQLite implementation while exposing it through the backend
    port.
    """

    def __init__(
        self,
        db_path: str | None = None,
        vector_store: Any | None = None,
        enable_semantic: bool = True,
    ) -> None:
        from huginn.memory.longterm import LongTermMemory

        self._impl = LongTermMemory(
            db_path=db_path,
            vector_store=vector_store,
            enable_semantic=enable_semantic,
        )

    def store(
        self,
        content: str,
        category: str = "fact",
        tags: list[str] | None = None,
        source: str = "",
        importance: float = 0.5,
        tier: str = "mid",
        ttl_hours: float | None = None,
    ) -> str:
        return self._impl.store(
            content=content,
            category=category,
            tags=tags,
            source=source,
            importance=importance,
            tier=tier,
            ttl_hours=ttl_hours,
        )

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        tier: str | None = None,
        top_k: int = 5,
        semantic: bool = True,
    ) -> list[dict[str, Any]]:
        return self._impl.retrieve(
            query=query,
            category=category,
            tier=tier,
            top_k=top_k,
            semantic=semantic,
        )

    def delete(self, memory_id: str) -> bool:
        return self._impl.delete(memory_id)
