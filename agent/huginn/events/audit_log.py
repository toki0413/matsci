"""Audit log subscriber — writes every event to ~/.huginn/events/audit.jsonl.

This is the provenance trail: append-only, one JSON object per line.
Inspired by Codex's rollout.jsonl — if something went wrong, you can
replay the audit log to reconstruct the full session timeline.

Usage:
    from huginn.events.audit_log import install_audit_subscriber
    install_audit_subscriber()  # call once at startup

After that, every published event lands in audit.jsonl automatically.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from huginn.events.event_bus import AgentEvent, EventBus
from huginn.events.event_types import ALL

logger = logging.getLogger(__name__)

# 缓冲写入: 攒够 _FLUSH_BATCH 条或 _FLUSH_INTERVAL 秒后一次性落盘.
# 避免 Windows AV 对每次 open+write+close 都扫一遍.
_FLUSH_BATCH = 20
_FLUSH_INTERVAL = 2.0  # seconds

_unsubscribe: Any = None  # stored so we can detach if needed


def _resolve_audit_path() -> Path:
    """Figure out where audit.jsonl lives.

    Tries the runtime home (HUGINN_CACHE_DIR or ~/.huginn), falls back
    to ~/.huginn directly. Never raises — worst case we write to the
    home directory.
    """
    try:
        from huginn.utils.runtime import get_runtime_home
        base = get_runtime_home()
    except Exception:
        base = Path.home() / ".huginn"
    events_dir = base / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    return events_dir / "audit.jsonl"


class _BufferedAuditWriter:
    """缓冲写入器: 攒一批再 flush, 减少 open/close 次数.

    线程安全. flush 在后台 daemon 线程跑, 主线程只往 deque 里塞.
    进程退出时 daemon 线程自动终止, 最多丢 _FLUSH_BATCH 条未落盘事件.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._buffer: deque[str] = deque()
        self._last_flush = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._flush_loop, name="audit-writer", daemon=True,
        )
        self._thread.start()

    def append(self, event: AgentEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        with self._lock:
            self._buffer.append(line)
            should_flush = len(self._buffer) >= _FLUSH_BATCH
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            lines = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.monotonic()
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in lines)
        except Exception:
            logger.debug("audit log flush failed (%d lines lost)", len(lines), exc_info=True)

    def _flush_loop(self) -> None:
        while not self._stop.wait(timeout=_FLUSH_INTERVAL):
            # 定时 flush, 即使没攒够 batch 也落盘
            if time.monotonic() - self._last_flush >= _FLUSH_INTERVAL:
                self.flush()

    def stop(self) -> None:
        self._stop.set()
        self.flush()
        self._thread.join(timeout=1.0)


# 全局 writer 实例 (install 时创建, uninstall 时停止)
_writer: _BufferedAuditWriter | None = None


def _make_subscriber(path: Path):
    """Create a subscriber closure bound to a buffered writer."""
    global _writer
    _writer = _BufferedAuditWriter(path)

    def _on_event(event: AgentEvent) -> None:
        _writer.append(event)

    return _on_event


def install_audit_subscriber(
    bus: EventBus | None = None,
    path: Path | None = None,
) -> Any:
    """Attach the audit log subscriber to the event bus.

    Call this once at startup. Returns an unsubscribe function — calling
    it stops the audit log from receiving further events.

    Args:
        bus: EventBus to subscribe to. Defaults to the shared singleton.
        path: Override for the audit file path. Defaults to
              ~/.huginn/events/audit.jsonl (or $HUGINN_CACHE_DIR/events/).
    """
    global _unsubscribe
    bus = bus or EventBus.shared()
    audit_path = path or _resolve_audit_path()
    subscriber = _make_subscriber(audit_path)
    _unsubscribe = bus.subscribe(ALL, subscriber)
    logger.info("audit subscriber installed: %s", audit_path)
    return _unsubscribe


def uninstall_audit_subscriber() -> None:
    """Detach the audit subscriber and flush remaining buffer."""
    global _unsubscribe, _writer
    if _unsubscribe is not None:
        _unsubscribe()
        _unsubscribe = None
    if _writer is not None:
        _writer.stop()
        _writer = None
