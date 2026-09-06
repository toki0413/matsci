"""Small bounded time-to-live cache utility.

Used to improve cache hit rates for embeddings, RAG queries, and read-only
tool calls without unbounded memory growth.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class TimedLRUCache(Generic[T]):
    """Bounded LRU cache with per-entry TTL (seconds).

    Thread-safe: all operations are protected by an internal lock so
    the cache can be shared across threads and asyncio callbacks
    without corruption.
    """

    def __init__(self, max_size: int = 128, ttl: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl
        self._data: OrderedDict[Any, tuple[T, float]] = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self) -> None:
        now = time.time()
        expired = [k for k, (_, exp) in self._data.items() if now > exp]
        for k in expired:
            del self._data[k]

    def get(self, key: Any) -> T | None:
        with self._lock:
            self._expire()
            if key in self._data:
                value, _ = self._data.pop(key)
                self._data[key] = (value, time.time() + self.ttl)
                return value
            return None

    def set(self, key: Any, value: T) -> None:
        with self._lock:
            self._expire()
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (value, time.time() + self.ttl)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            self._expire()
            return len(self._data)


# 共享 embedding 向量缓存 — 合并 knowledge.store._EmbeddingModel 与
# rag.vector_store.VectorStore 两套重复缓存为单一实例 (此前各自维护一份,
# 缓存同一批向量). 存 list[list[float]] (纯 Python, 无 numpy 依赖),
# 超过 max_size LRU 淘汰 + 1h TTL. 两端按需在 ndarray ↔ list 间转换.
embedding_cache: TimedLRUCache[list[list[float]]] = TimedLRUCache(
    max_size=2048, ttl=3600.0
)
