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

import gzip
import hashlib
import hmac as _hmac
import json
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
        details = details or {}

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

    def query(
        self,
        *,
        event_type: str | None = None,
        actor: str | None = None,
        action: str | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events with filters.

        Parameters
        ----------
        event_type : filter by event type (exact match)
        actor : filter by actor (exact match)
        action : filter by action (substring match)
        since : ISO timestamp lower bound (inclusive)
        until : ISO timestamp upper bound (inclusive)
        limit : max results (default 100)
        """
        results: list[dict[str, Any]] = []
        if not self.log_path.exists():
            return results

        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event_type and record.get("event_type") != event_type:
                    continue
                if actor and record.get("actor") != actor:
                    continue
                if action and action.lower() not in record.get("action", "").lower():
                    continue
                ts = record.get("timestamp", "")
                if since and ts < since:
                    continue
                if until and ts > until:
                    continue

                results.append(record)
                if len(results) >= limit:
                    break

        return results


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
