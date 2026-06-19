"""Tests for audit log hash-chain verification."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from huginn.security.audit import AuditLogger


class TestAuditVerifyChain:
    def test_empty_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = AuditLogger(Path(tmp) / "audit.jsonl")
            assert logger.verify_chain() == []

    def test_valid_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            logger = AuditLogger(log_path)
            logger.log("tool_call", "agent", "test_tool", input_data="in")
            logger.log("tool_call", "agent", "test_tool2", input_data="in2")
            assert logger.verify_chain() == []

    def test_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "audit.jsonl"
            logger = AuditLogger(log_path)
            logger.log("tool_call", "agent", "test_tool", input_data="in")
            logger.log("tool_call", "agent", "test_tool2", input_data="in2")

            lines = log_path.read_text(encoding="utf-8").strip().split("\n")
            record = json.loads(lines[0])
            record["actor"] = "attacker"
            lines[0] = json.dumps(record)
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            mismatches = logger.verify_chain()
            assert len(mismatches) == 1
            line_no, expected, actual = mismatches[0]
            assert line_no == 2
            assert actual == json.loads(lines[1]).get("prev_hash", "")
            assert expected == hashlib.sha256(lines[0].encode("utf-8")).hexdigest()[:32]
