"""Structured audit logging for Huginn.

Provides append-only JSONL logs with tamper-evident hash chaining.
"""

from __future__ import annotations

import hashlib
import json
import os
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


class AuditLogger:
    """Thread-safe append-only audit logger.

    Usage:
        audit = AuditLogger("audit.jsonl")
        audit.log("tool_call", "user", "vasp_relaxation",
                  details={"incar": "..."}, input_data="raw_input")
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        if log_path is None:
            # Default to workspace-relative audit log
            log_path = Path("huginn_audit.jsonl")
        self.log_path = Path(log_path)
        self._lock = threading.Lock()
        self._last_hash = ""
        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

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

        event = AuditEvent(
            timestamp=self._now(),
            event_type=event_type,
            actor=actor,
            action=action,
            details=details,
            input_hash=input_hash,
            output_hash=output_hash,
            prev_hash=self._last_hash,
        )

        line = json.dumps(asdict(event), sort_keys=True, default=str)
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
