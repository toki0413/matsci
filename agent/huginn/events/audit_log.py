"""Audit log subscriber — writes every event to ~/.huginn/events/audit.jsonl.

This is the provenance trail: append-only, one JSON object per line.
Inspired by Codex's rollout.jsonl — if something went wrong, you can
replay the audit log to reconstruct the full session timeline.

Tamper-evident: each record carries a SHA-256 hash of its content chained
to the previous record's hash. Any modification breaks the chain.
Design borrowed from OpenParallax's audit module.

Usage:
    from huginn.events.audit_log import install_audit_subscriber
    install_audit_subscriber()  # call once at startup

After that, every published event lands in audit.jsonl automatically.
"""

from __future__ import annotations

import hashlib
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


def _compute_hash(payload: dict[str, Any], prev_hash: str) -> str:
    """SHA-256 over canonical JSON of (payload_without_hash_fields, prev_hash).

    We strip _hash and _prev_hash from the payload before hashing so the
    hash only covers the event content, not the chain metadata itself.
    """
    content = {k: v for k, v in payload.items() if k not in ("_hash", "_prev_hash")}
    blob = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(f"{blob}{prev_hash}".encode("utf-8")).hexdigest()


class _BufferedAuditWriter:
    """缓冲写入器: 攒一批再 flush, 减少 open/close 次数.

    线程安全. flush 在后台 daemon 线程跑, 主线程只往 deque 里塞.
    进程退出时 daemon 线程自动终止, 最多丢 _FLUSH_BATCH 条未落盘事件.

    Hash chain: 每个 record 的 _hash = SHA-256(content + _prev_hash).
    链头 _prev_hash = "0" * 64 (genesis). 任何篡改都会断链.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._buffer: deque[dict[str, Any]] = deque()
        self._last_flush = time.monotonic()
        self._stop = threading.Event()
        self._prev_hash = self._load_last_hash()
        self._thread = threading.Thread(
            target=self._flush_loop, name="audit-writer", daemon=True,
        )
        self._thread.start()

    def _load_last_hash(self) -> str:
        """Read the last _hash from an existing audit file to resume the chain.

        Returns "0"*64 if the file is missing, empty, or corrupted —
        a fresh chain is always valid.
        """
        genesis = "0" * 64
        try:
            if not self._path.exists():
                return genesis
            last_hash = genesis
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        h = rec.get("_hash")
                        if h:
                            last_hash = h
                    except json.JSONDecodeError:
                        continue
            return last_hash
        except Exception:
            return genesis

    def append(self, event: AgentEvent) -> None:
        with self._lock:
            self._buffer.append(event.to_dict())
            should_flush = len(self._buffer) >= _FLUSH_BATCH
        if should_flush:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            events = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.monotonic()

        # Build hash-chained records outside the lock — hashing is CPU-only
        lines: list[str] = []
        prev = self._prev_hash
        for ev in events:
            record = dict(ev)
            record["_prev_hash"] = prev
            record["_hash"] = _compute_hash(record, prev)
            prev = record["_hash"]
            lines.append(json.dumps(record, ensure_ascii=False, default=str))

        # Persist the last hash so the next flush continues the chain
        self._prev_hash = prev

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


# ── campaign/quality 业务订阅者 ─────────────────────────────────
# 之前 campaign.* / quality.check 事件只落 audit.jsonl, 无业务消费方 (emit→log→dead).
# 这里挂一个轻量订阅器: 把 iteration/hypothesis/retry/suspect/refine/quality.check
# 事件提到 logger.info + 可选 telemetry, 让用户/telemetry 能感知研究循环的实时进度.
# ponytail: 只 log, 不做状态更新. 升级: 接 progress 面板 / WebSocket push.
_CAMPAIGN_UNSUBSCRIBES: list[Any] = []


def install_campaign_subscriber(bus: EventBus | None = None) -> None:
    """订阅 campaign.* / quality.check 事件, 转 logger.info + telemetry.

    之前这些事件 emit 后只落 audit.jsonl, 业务侧无感知. 这里挂一个轻量订阅器,
    让 research loop 的关键节点 (iteration/hypothesis/retry/suspect/refine/quality)
    至少在 log 和 telemetry 里可见.
    ponytail: 只 log + Counter, 不做 UI push. 升级: WebSocket / progress 面板.
    """
    global _CAMPAIGN_UNSUBSCRIBES
    bus = bus or EventBus.shared()
    events = (
        "campaign.iteration", "campaign.hypothesis",
        "campaign.retry", "campaign.suspect",
        "campaign.refine", "quality.check",
    )

    def _on_campaign(event: AgentEvent) -> None:
        try:
            logger.info(
                "campaign event: %s iter=%s payload_keys=%s",
                event.type,
                event.data.get("iteration", "?"),
                list(event.data.keys()),
            )
        except Exception:
            pass

    for evt_type in events:
        try:
            unsub = bus.subscribe(evt_type, _on_campaign)
            _CAMPAIGN_UNSUBSCRIBES.append(unsub)
        except Exception:
            logger.debug("campaign subscribe failed for %s", evt_type, exc_info=True)


def uninstall_campaign_subscriber() -> None:
    """Detach all campaign/quality subscribers."""
    global _CAMPAIGN_UNSUBSCRIBES
    for unsub in _CAMPAIGN_UNSUBSCRIBES:
        try:
            unsub()
        except Exception:
            pass
    _CAMPAIGN_UNSUBSCRIBES = []


def verify_audit_chain(path: Path | None = None) -> bool:
    """Verify the integrity of the audit log hash chain.

    Returns True if every record's _hash matches the recomputed value
    and the chain is unbroken. Returns True for an empty/missing file.
    Logs the first broken record if verification fails.
    """
    audit_path = path or _resolve_audit_path()
    if not audit_path.exists():
        return True
    prev_hash = "0" * 64
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("audit chain broken at line %d: invalid JSON", lineno)
                    return False
                stored_hash = rec.get("_hash", "")
                stored_prev = rec.get("_prev_hash", "")
                if stored_prev != prev_hash:
                    logger.warning(
                        "audit chain broken at line %d: prev_hash mismatch "
                        "(expected %s, got %s)",
                        lineno, prev_hash[:16], stored_prev[:16],
                    )
                    return False
                recomputed = _compute_hash(rec, prev_hash)
                if recomputed != stored_hash:
                    logger.warning(
                        "audit chain broken at line %d: hash mismatch "
                        "(expected %s, got %s)",
                        lineno, recomputed[:16], stored_hash[:16],
                    )
                    return False
                prev_hash = stored_hash
        return True
    except Exception:
        logger.debug("verify_audit_chain failed", exc_info=True)
        return False
