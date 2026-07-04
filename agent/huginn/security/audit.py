"""Structured audit logging for Huginn.

Provides append-only JSONL logs with tamper-evident hash chaining,
HMAC-SHA256 digital signatures, automatic log rotation, and a structured
query interface.

Phase 4 additions:
- ``sign_event()`` — HMAC-SHA256 signature per event using a dedicated key
- ``AuditLogRotator`` — size/time-based rotation with compressed archives
- ``AuditQuery`` — filter events by type, actor, action, time range
"""

from __future__ import annotations

import calendar
import gzip
import hashlib
import hmac as _hmac
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditEvent:
    """A single audit event."""

    timestamp: str
    event_type: str  # e.g. "tool_call", "subprocess_exec", "llm_invoke", "config_load"
    actor: str  # e.g. "user", "agent", "skill"
    action: str  # e.g. "vasp_relaxation", "lake_build"
    details: dict[str, Any] = field(default_factory=dict)
    input_hash: str | None = None
    output_hash: str | None = None
    prev_hash: str = ""  # hash chain for tamper evidence
    signature: str | None = None  # HMAC-SHA256 signature


class AuditLogger:
    """Thread-safe append-only audit logger with optional HMAC signing.

    Usage::

        audit = AuditLogger("audit.jsonl", signing_key=b"secret")
        audit.log("tool_call", "user", "vasp_relaxation",
                  details={"incar": "..."}, input_data="raw_input")
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        signing_key: str | bytes | None = None,
    ) -> None:
        if log_path is None:
            # 落在 runtime home 下, 不散落在 CWD
            try:
                from huginn.utils.runtime import get_runtime_home
                log_path = get_runtime_home() / "audit.jsonl"
            except Exception:
                log_path = Path("huginn_audit.jsonl")
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._last_hash = ""

        # Signing key (optional — enables tamper-proof signatures)
        if isinstance(signing_key, str):
            signing_key = signing_key.encode("utf-8")
        self._signing_key: bytes | None = signing_key

        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Replay chain state from existing log
        if self.log_path.exists():
            self._replay_chain()

    def _replay_chain(self) -> None:
        """Re-read the log file to restore the last hash for chain continuity."""
        last_hash = ""
        try:
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()[:32]
        except OSError:
            pass
        self._last_hash = last_hash

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _sign_line(self, line: str) -> str:
        """Produce HMAC-SHA256 signature for a JSON line."""
        if self._signing_key is None:
            return ""
        return _hmac.new(
            self._signing_key, line.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:32]

    def _redact_details(self, details: dict[str, Any]) -> dict[str, Any]:
        """Scan dict values for secrets and redact them before logging."""
        try:
            from huginn.privacy.scanner import SecretScanner
            scanner = SecretScanner()
        except Exception as exc:
            # 扫描器不可用时必须警告 — 不能静默放过未脱敏数据
            logging.getLogger(__name__).warning(
                "SecretScanner unavailable (%s), details NOT redacted", exc,
            )
            return details

        redacted = {}
        for key, value in details.items():
            if isinstance(value, str):
                cleaned = scanner.redact(value)
                if cleaned != value:
                    redacted[key] = cleaned
                else:
                    redacted[key] = value
            elif isinstance(value, dict):
                redacted[key] = self._redact_details(value)
            elif isinstance(value, list):
                redacted[key] = [
                    self._redact_details({"_": item}).get("_", item) if isinstance(item, dict)
                    else scanner.redact(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                redacted[key] = value
        return redacted

    def log(
        self,
        event_type: str,
        actor: str,
        action: str,
        details: dict[str, Any] | None = None,
        input_data: str | bytes | None = None,
        output_data: str | bytes | None = None,
    ) -> AuditEvent:
        """Record an audit event."""
        details = self._redact_details(details or {})

        input_hash = None
        if input_data is not None:
            input_hash = self._hash_data(input_data)

        output_hash = None
        if output_data is not None:
            output_hash = self._hash_data(output_data)

        # Build the event dict WITHOUT the signature field for signing
        event_dict = {
            "timestamp": self._now(),
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "details": details,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "prev_hash": self._last_hash,
        }

        # Sign the canonical JSON (no signature field)
        canonical = json.dumps(event_dict, sort_keys=True, default=str)
        sig = self._sign_line(canonical)
        if sig:
            event_dict["signature"] = sig

        event = AuditEvent(
            timestamp=event_dict["timestamp"],
            event_type=event_type,
            actor=actor,
            action=action,
            details=details,
            input_hash=input_hash,
            output_hash=output_hash,
            prev_hash=self._last_hash,
            signature=sig or None,
        )

        line = json.dumps(event_dict, sort_keys=True, default=str)

        with self._lock:
            self._last_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()[:32]
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

        return event

    def record(
        self,
        event_type: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEvent:
        """简化版记录接口：只传事件类型和详情 dict。

        内部还是走 log()，actor 默认 system，action 从 details.tool 取，
        没有就用 event_type。调试时随手记一条用这个最省事。
        """
        details = details or {}
        action = details.get("tool") or details.get("action") or event_type
        return self.log(event_type, "system", action, details=details)

    @staticmethod
    def _hash_data(data: str | bytes) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()[:16]

    # -- verification ---------------------------------------------------

    def verify_chain(self) -> list[tuple[int, str, str]]:
        """Verify the integrity of the audit log hash chain.

        Each event's ``prev_hash`` must equal the SHA-256 hash of the previous
        event's JSON line. Returns a list of (line_number, expected_hash,
        actual_hash) for mismatches.
        """
        mismatches: list[tuple[int, str, str]] = []
        if not self.log_path.exists():
            return mismatches

        prev_line_hash = ""
        with open(self.log_path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    mismatches.append((i, "valid_json", f"parse_error:{exc}"))
                    break

                stored_prev = record.get("prev_hash", "")
                if i > 1 and stored_prev != prev_line_hash:
                    mismatches.append((i, prev_line_hash, stored_prev))

                prev_line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()[:32]

        return mismatches

    def verify_signatures(self) -> list[tuple[int, str]]:
        """Verify HMAC signatures on all events.

        Returns list of (line_number, issue) for unsigned or invalid events.
        """
        issues: list[tuple[int, str]] = []
        if self._signing_key is None:
            return issues  # Cannot verify without key

        if not self.log_path.exists():
            return issues

        with open(self.log_path, encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    issues.append((i, "invalid_json"))
                    continue

                stored_sig = record.pop("signature", None)
                if not stored_sig:
                    issues.append((i, "missing_signature"))
                    continue

                # Re-compute signature on the record WITHOUT the signature field
                reconstructed = json.dumps(record, sort_keys=True, default=str)
                expected = _hmac.new(
                    self._signing_key,
                    reconstructed.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()[:32]

                if not _hmac.compare_digest(stored_sig, expected):
                    issues.append((i, "signature_mismatch"))

        return issues

    # -- query interface ------------------------------------------------

    @staticmethod
    def _ts_to_unix(ts: float | str | None) -> float | None:
        """把 ISO 字符串或 unix float 统一成 unix timestamp。

        None / 解析失败返回 None。unix float 原样返回。
        """
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return float(ts)
        try:
            # 时间戳格式 "2026-06-12T16:22:13Z", Z = UTC
            return float(calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")))
        except (ValueError, TypeError):
            return None

    def _iter_records(self):
        """读全部记录并 yield dict，跳过空行和坏行。"""
        if not self.log_path.exists():
            return
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def query(
        self,
        *,
        event_type: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        session_id: str | None = None,
        tool: str | None = None,
        since: float | str | None = None,
        until: float | str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Query audit events with filters.

        Parameters
        ----------
        event_type : filter by event type (exact match)
        actor : filter by actor (exact match)
        action : filter by action (substring match)
        session_id : 按 session 过滤，从 details.session_id 取
        tool : 按 tool 过滤，从 details.tool 取
        since : 下界 (inclusive)，unix float 或 ISO 字符串都行
        until : 上界 (inclusive)，unix float 或 ISO 字符串都行
        limit : max results (default 1000)
        """
        since_unix = self._ts_to_unix(since)
        until_unix = self._ts_to_unix(until)
        results: list[dict[str, Any]] = []

        for record in self._iter_records():
            if event_type and record.get("event_type") != event_type:
                continue
            if actor and record.get("actor") != actor:
                continue
            if action and action.lower() not in record.get("action", "").lower():
                continue

            details = record.get("details", {})
            if session_id is not None:
                rec_session = details.get("session_id") or record.get("session_id")
                if rec_session != session_id:
                    continue
            if tool is not None and details.get("tool") != tool:
                continue

            if since_unix is not None:
                rec_unix = self._ts_to_unix(record.get("timestamp", ""))
                if rec_unix is None or rec_unix < since_unix:
                    continue
            if until_unix is not None:
                rec_unix = self._ts_to_unix(record.get("timestamp", ""))
                if rec_unix is None or rec_unix > until_unix:
                    continue

            results.append(record)
            if len(results) >= limit:
                break

        return results

    def aggregate(
        self,
        group_by: str = "tool",
        session_id: str | None = None,
        since: float | None = None,
    ) -> list[dict[str, Any]]:
        """按维度分组统计。

        group_by 取 "tool" / "event_type" / "session"。
        返回每组的 count / success_count / fail_count / success_rate
        以及首末事件时间，按 count 降序排。

        success 从 details.success 取，True 算成功，False 算失败，
        没有这个字段的不计入 success/fail。
        """
        since_unix = self._ts_to_unix(since) if since is not None else None
        groups: dict[str, list[dict[str, Any]]] = {}

        for record in self._iter_records():
            details = record.get("details", {})
            # session_id 过滤
            if session_id is not None:
                rec_session = details.get("session_id") or record.get("session_id")
                if rec_session != session_id:
                    continue
            # since 过滤
            if since_unix is not None:
                rec_unix = self._ts_to_unix(record.get("timestamp", ""))
                if rec_unix is None or rec_unix < since_unix:
                    continue

            # 分组 key
            if group_by == "tool":
                key = details.get("tool", "unknown")
            elif group_by == "event_type":
                key = record.get("event_type", "unknown")
            elif group_by == "session":
                key = (
                    details.get("session_id")
                    or record.get("session_id")
                    or "unknown"
                )
            else:
                key = "unknown"

            groups.setdefault(key, []).append(record)

        results: list[dict[str, Any]] = []
        for key, recs in groups.items():
            count = len(recs)
            success_count = sum(
                1 for r in recs if r.get("details", {}).get("success") is True
            )
            fail_count = sum(
                1 for r in recs if r.get("details", {}).get("success") is False
            )
            success_rate = (success_count / count) if count else 0.0
            timestamps = [
                r.get("timestamp", "") for r in recs if r.get("timestamp")
            ]
            results.append({
                group_by: key,
                "count": count,
                "success_count": success_count,
                "fail_count": fail_count,
                "success_rate": success_rate,
                "first_event": min(timestamps) if timestamps else None,
                "last_event": max(timestamps) if timestamps else None,
            })

        # 按 count 降序，方便先看调用最多的
        results.sort(key=lambda x: x["count"], reverse=True)
        return results

    def summary(self, since: float | None = None) -> dict[str, Any]:
        """总体统计：事件数 / 工具数 / 会话数 / 时间范围 / 按类型计数。"""
        since_unix = self._ts_to_unix(since) if since is not None else None
        tools: set[str] = set()
        sessions: set[str] = set()
        event_types: dict[str, int] = {}
        timestamps: list[str] = []
        total = 0

        for record in self._iter_records():
            if since_unix is not None:
                rec_unix = self._ts_to_unix(record.get("timestamp", ""))
                if rec_unix is None or rec_unix < since_unix:
                    continue
            total += 1
            details = record.get("details", {})
            if details.get("tool"):
                tools.add(details["tool"])
            rec_session = details.get("session_id") or record.get("session_id")
            if rec_session:
                sessions.add(rec_session)
            et = record.get("event_type", "unknown")
            event_types[et] = event_types.get(et, 0) + 1
            ts = record.get("timestamp")
            if ts:
                timestamps.append(ts)

        return {
            "total_events": total,
            "num_tools": len(tools),
            "num_sessions": len(sessions),
            "event_types": event_types,
            "tools": sorted(tools),
            "time_range": (
                [min(timestamps), max(timestamps)] if timestamps else [None, None]
            ),
        }


# ---------------------------------------------------------------------------
# Log rotation
# ---------------------------------------------------------------------------

class AuditLogRotator:
    """Size- and time-based log rotation with optional gzip compression.

    Usage::

        rotator = AuditLogRotator(logger, max_size=10_000_000, max_age_days=30)
        rotator.rotate_if_needed()
    """

    def __init__(
        self,
        logger: AuditLogger,
        max_size: int = 10_000_000,       # 10 MB
        max_age_days: float = 30.0,
        compress: bool = True,
        max_archives: int = 10,
    ) -> None:
        self.logger = logger
        self.max_size = max_size
        self.max_age_days = max_age_days
        self.compress = compress
        self.max_archives = max_archives

    def should_rotate(self) -> bool:
        path = self.logger.log_path
        if not path.exists():
            return False

        # Size check
        if path.stat().st_size >= self.max_size:
            return True

        # Age check
        mtime = path.stat().st_mtime
        age_days = (time.time() - mtime) / 86400.0
        if age_days >= self.max_age_days:
            return True

        return False

    def rotate(self) -> Path | None:
        """Rotate the current log file.

        Returns the path to the archived file, or None if nothing to rotate.
        """
        path = self.logger.log_path
        if not path.exists() or path.stat().st_size == 0:
            return None

        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        archive_name = f"{path.stem}_{ts}{path.suffix}"
        archive_path = path.parent / archive_name

        # Move current log to archive
        with self.logger._lock:
            shutil.move(str(path), str(archive_path))
            self.logger._last_hash = ""  # Reset chain for new file

        # Compress if requested
        if self.compress:
            gz_path = Path(str(archive_path) + ".gz")
            with open(archive_path, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            archive_path.unlink()
            archive_path = gz_path

        # Prune old archives
        self._prune_archives()

        return archive_path

    def rotate_if_needed(self) -> Path | None:
        if self.should_rotate():
            return self.rotate()
        return None

    def _prune_archives(self) -> None:
        """Remove oldest archives beyond max_archives."""
        parent = self.logger.log_path.parent
        stem = self.logger.log_path.stem
        archives = sorted(
            parent.glob(f"{stem}_*.jsonl*"),
            key=lambda p: p.stat().st_mtime,
        )
        while len(archives) > self.max_archives:
            oldest = archives.pop(0)
            oldest.unlink()

    def list_archives(self) -> list[Path]:
        parent = self.logger.log_path.parent
        stem = self.logger.log_path.stem
        return sorted(
            parent.glob(f"{stem}_*.jsonl*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
