"""Background memory maintenance — runs decay, prune, and dedupe periodically.

Runs in a daemon thread, wakes up every hour, applies MemoryDecayPolicy
to long-term memories, prunes low-importance entries, and deduplicates
near-identical records. Zero LLM token consumption — pure housekeeping.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
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
        dream_enabled: bool | None = None,
        persona_id: str = "default",
        sleep_clock: Any = None,
        dream_store: Any = None,
        dream_memories_fn: Any = None,
        now_fn: Any = None,
    ):
        self._mm = memory_manager
        self._interval = interval
        # P5: cluster 配置透传给 maintenance(). 默认关, HUGINN_MEMORY_CLUSTER=1
        # env 才生效. llm_chat_fn 为 None 时即使 cluster=True 也跳过 (没 LLM 没法 summarize).
        self._cluster = cluster
        self._llm_chat_fn = llm_chat_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        from huginn.env_access import env_bool

        self._dream_enabled = (
            env_bool("HUGINN_DREAM_ENABLED")
            if dream_enabled is None
            else bool(dream_enabled)
        )
        self._persona_id = persona_id
        self._sleep_clock = sleep_clock or self._build_sleep_clock(persona_id)
        self._dream_store = dream_store
        self._dream_memories_fn = dream_memories_fn or self._default_dream_memories
        self._now_fn = now_fn or (lambda: datetime.now(UTC).isoformat())

    @staticmethod
    def _build_sleep_clock(persona_id: str) -> Any:
        """从环境构造人格作息；无窗口 → active（梦境默认休眠，不伪造作息）."""
        from huginn.env_access import env_str
        from huginn.memory.dream import SleepClock

        windows: tuple[tuple[str, str, str], ...] = ()
        raw = env_str("HUGINN_DREAM_WINDOWS").strip()
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    windows = tuple(
                        tuple(item)
                        for item in data
                        if isinstance(item, (list, tuple)) and len(item) == 3
                    )
            except (ValueError, TypeError):
                logger.warning("invalid HUGINN_DREAM_WINDOWS; dream stays inactive")
        return SleepClock(
            persona_id=persona_id,
            timezone=env_str("HUGINN_DREAM_TIMEZONE") or "UTC",
            weekly_windows=windows,
        )

    def _default_dream_memories(self) -> list[dict[str, Any]]:
        longterm = getattr(self._mm, "longterm", None) if self._mm else None
        snapshot = getattr(longterm, "snapshot_for_dream", None)
        if callable(snapshot):
            return snapshot()
        return []

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

    def _run_dream_cycle(self) -> dict[str, Any]:
        """睡眠窗口内跑梦境周期；活跃状态直接跳过（不伪造作息）."""
        if not self._dream_enabled or self._sleep_clock is None:
            return {"dreamed": 0}
        try:
            import uuid

            from huginn.memory.dream import run_dream_cycle, tick

            now = self._now_fn()
            clock = tick(self._sleep_clock, now)
            self._sleep_clock = clock
            if clock.state != "sleep":
                return {"dreamed": 0, "state": clock.state}
            memories = self._dream_memories_fn() if self._dream_memories_fn else []
            if not memories:
                return {"dreamed": 0, "state": "sleep", "reason": "no_memories"}
            result = run_dream_cycle(
                clock=clock,
                dream_id=f"dream-{uuid.uuid4().hex[:10]}",
                memories=memories,
                now=now,
            )
            if result.status == "completed" and result.dream_object is not None:
                from huginn.memory.dream_store import get_dream_store

                store = self._dream_store or get_dream_store()
                store.save(result.dream_object, list(result.repairs))
                return {
                    "dreamed": 1,
                    "findings": len(result.findings),
                    "repairs": len(result.repairs),
                    "dream_id": result.dream_object.dream_id,
                }
        except Exception:  # 防御: 梦境失败不影响常规维护
            logger.warning("dream cycle failed (non-fatal)", exc_info=True)
        return {"dreamed": 0}

    def _run_maintenance(self) -> None:
        """Run one maintenance cycle.

        阈值 (decay_per_day / prune_threshold / deduplicate) 由
        plugins.memory_maintenance_policy 策略注册表提供, 不在守护线程里硬编码.
        """
        if self._mm is None:
            return
        result = self._mm.maintenance(
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
        dream = self._run_dream_cycle()
        if dream.get("dreamed"):
            logger.info(
                "dream cycle: id=%s findings=%s repairs=%s",
                dream.get("dream_id"),
                dream.get("findings"),
                dream.get("repairs"),
            )

    def run_once(self) -> dict[str, Any]:
        """Run maintenance once (for testing or manual trigger)."""
        if self._mm is None:
            return {"pruned": 0, "deduplicated": 0, "expired": 0, "dreamed": 0}
        result = self._mm.maintenance(
            cluster=self._cluster,
            llm_chat_fn=self._llm_chat_fn,
        )
        result["dream"] = self._run_dream_cycle()
        return result
