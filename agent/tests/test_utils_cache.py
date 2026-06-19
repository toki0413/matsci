"""Tests for TimedLRUCache utility."""

from __future__ import annotations

import time

from huginn.utils.cache import TimedLRUCache


class TestTimedLRUCache:
    def test_basic_get_set(self):
        cache = TimedLRUCache[str](max_size=3, ttl=10.0)
        cache.set("a", "alpha")
        assert cache.get("a") == "alpha"
        assert cache.get("missing") is None

    def test_lru_eviction(self):
        cache = TimedLRUCache[int](max_size=2, ttl=10.0)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_ttl_expiration(self):
        cache = TimedLRUCache[str](max_size=2, ttl=0.05)
        cache.set("a", "alpha")
        assert cache.get("a") == "alpha"
        time.sleep(0.06)
        assert cache.get("a") is None
