"""Background memory maintenance — runs decay, prune, and dedupe periodically.

Runs in a daemon thread, wakes up every hour, applies MemoryDecayPolicy
to long-term memories, prunes low-importance entries, and deduplicates
near-identical records. Zero LLM token consumption — pure housekeeping.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Run maintenance every N seconds (default: 1 hour)
_MAINTENANCE_INTERVAL = 3600


class MemoryMaintainer:
    """Daemon thread that runs memory maintenance periodically.

    Usage:
        maintainer = MemoryMaintainer(memory_manager=mm)
        maintainer.start()
        maintainer.stop()

    P5: cluster=True + llm_chat_fn 注入时, maintenance 会跑 semantic cluster.
    HUGINN_MEMORY_CLUSTER=1 env 控制. llm_chat_fn 由 caller 提供 (通常用
    verification_model 的 ainvoke), None 时跳过 cluster step.
    """

    def __init__(
        self,
        memory_manager: Any = None,
        interval: float = _MAINTENANCE_INTERVAL,
        cluster: bool = False,
        llm_chat_fn: Any = None,
    ):
        self._mm = memory_manager
        self._interval = interval
        # P5: cluster 配置透传给 maintenance(). 默认关, HUGINN_MEMORY_CLUSTER=1
        # env 才生效. llm_chat_fn 为 None 时即使 cluster=True 也跳过 (没 LLM 没法 summarize).
        self._cluster = cluster
        self._llm_chat_fn = llm_chat_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="memory-maintainer")
        self._thread.start()
        logger.info("memory maintainer started (interval=%ds)", self._interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("memory maintainer stopped")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Wait for interval or stop signal
            if self._stop_event.wait(self._interval):
                break
            try:
                self._run_maintenance()
            except Exception as e:
                logger.warning("memory maintenance error: %s", e)

    def _run_maintenance(self) -> None:
        """Run one maintenance cycle."""
        if self._mm is None:
            return
        result = self._mm.maintenance(
            decay_per_day=0.97,
            prune_threshold=0.15,
            deduplicate=True,
            cluster=self._cluster,
            llm_chat_fn=self._llm_chat_fn,
        )
        pruned = result.get("pruned", 0)
        deduped = result.get("deduplicated", 0)
        clustered = result.get("clustered", 0)
        archived = result.get("archived", 0)
        if pruned > 0 or deduped > 0 or clustered > 0:
            logger.info(
                "memory maintenance: pruned %d, deduped %d, clustered %d, archived %d",
                pruned, deduped, clustered, archived,
            )

    def run_once(self) -> dict[str, int]:
        """Run maintenance once (for testing or manual trigger)."""
        if self._mm is None:
            return {"pruned": 0, "deduplicated": 0, "expired": 0}
        return self._mm.maintenance(
            cluster=self._cluster,
            llm_chat_fn=self._llm_chat_fn,
        )
