"""Tests for privacy deepening: audit redaction, PII scanning, secure cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.crypto import EncryptedDatabase
from huginn.privacy.scanner import SecretScanner
from huginn.security.audit import AuditLogger


# Build a key that matches the anthropic-api-key pattern:
# sk-ant-api03-<93 chars>AA
_ANTHROPIC_KEY = "sk-ant-api03-" + "a" * 93 + "AA"
_GITHUB_PAT = "ghp_" + "a" * 36


class TestAuditRedaction:
    def test_audit_redacts_api_key_in_details(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)

        logger.log(
            "tool_call",
            "agent",
            "api_request",
            details={"api_key": _ANTHROPIC_KEY},
        )

        content = log_path.read_text(encoding="utf-8")
        assert "[REDACTED]" in content
        assert _ANTHROPIC_KEY not in content

    def test_audit_redacts_nested_dict(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)

        details = {
            "repository": "materials-project",
            "auth": {"token": _GITHUB_PAT, "user": "alice"},
        }
        logger.log("tool_call", "agent", "git_clone", details=details)

        content = log_path.read_text(encoding="utf-8")
        assert "[REDACTED]" in content
        assert _GITHUB_PAT not in content
        # Non-secret nested fields should survive
        assert "materials-project" in content

    def test_audit_preserves_non_secret_details(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path)

        details = {
            "incar": "ENCUT=520",
            "kpoints": [4, 4, 4],
            "system": "Si",
        }
        logger.log("tool_call", "agent", "vasp_run", details=details)

        content = log_path.read_text(encoding="utf-8")
        assert "ENCUT=520" in content
        assert "Si" in content
        assert "[REDACTED]" not in content


class TestPIIScanning:
    def test_scanner_detects_email(self):
        scanner = SecretScanner()
        matches = scanner.scan_pii("contact: user@example.com")
        rule_ids = [m.rule_id for m in matches]
        assert "pii-email" in rule_ids

    def test_scanner_detects_orcid(self):
        scanner = SecretScanner()
        matches = scanner.scan_pii("ORCID: 0000-0001-2345-6789")
        rule_ids = [m.rule_id for m in matches]
        assert "pii-orcid" in rule_ids

    def test_scanner_redacts_pii(self):
        scanner = SecretScanner()
        text = "Contact user@example.com for details"
        result = scanner.redact_pii(text)
        assert "[PII_REDACTED]" in result
        assert "user@example.com" not in result

    def test_scanner_secrets_unchanged(self):
        scanner = SecretScanner()

        # Existing secret detection should still work
        matches = scanner.scan(_GITHUB_PAT)
        rule_ids = [m.rule_id for m in matches]
        assert "github-pat" in rule_ids

        # scan() must not flag PII — that's scan_pii()'s job
        email_matches = scanner.scan("user@example.com")
        assert len(email_matches) == 0


class TestEncryptedDBSecureCleanup:
    def test_encrypted_db_secure_cleanup(self, tmp_path):
        # Mock the vault so we don't need a real encryption key
        vault = MagicMock()
        enc_path = tmp_path / "test_db.enc"
        db = EncryptedDatabase(vault, enc_path)

        db.mount()
        temp_dir = db._temp_dir
        assert temp_dir is not None
        assert temp_dir.exists()

        # Drop a couple files in the temp dir to simulate real db usage
        db.plaintext_path.write_text("plaintext database content")
        extra_file = temp_dir / "wal.log"
        extra_file.write_text("write-ahead log data")

        db.unmount()

        # Everything should be gone after unmount
        assert not temp_dir.exists()
        assert not extra_file.exists()
        assert db._temp_dir is None
        assert db._plaintext_path is None
