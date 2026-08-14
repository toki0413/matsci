"""Tests for the privacy / secret scanner."""

from __future__ import annotations

from unittest.mock import MagicMock

from huginn.crypto import EncryptedDatabase
from huginn.privacy import redact_secrets, scan_for_secrets
from huginn.privacy.scanner import SecretScanner
from huginn.security.audit import AuditLogger


def _dummy_openai_key() -> str:
    return "sk-" + "a" * 20 + "T3BlbkFJ" + "b" * 20


def test_scan_detects_openai_key() -> None:
    key = _dummy_openai_key()
    text = f"My key is {key} for OpenAI."
    matches = scan_for_secrets(text)
    assert any(m.rule_id == "openai-api-key" for m in matches)


def test_scan_detects_aws_key() -> None:
    text = "AKIAIOSFODNN7EXAMPLE"
    matches = scan_for_secrets(text)
    assert any(m.rule_id == "aws-access-token" for m in matches)


def test_scan_detects_private_key() -> None:
    body = "x" * 80
    text = f"-----BEGIN OPENSSH PRIVATE KEY-----\n{body}\n-----END OPENSSH PRIVATE KEY-----"
    matches = scan_for_secrets(text)
    assert any(m.rule_id == "private-key" for m in matches)


def test_scan_no_false_positive() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    matches = scan_for_secrets(text)
    assert len(matches) == 0


def test_redact_replaces_secret() -> None:
    openai_key = _dummy_openai_key()
    text = f"key=AKIAIOSFODNN7EXAMPLE and token={openai_key}"
    redacted = redact_secrets(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert openai_key not in redacted
    assert "[REDACTED]" in redacted


def test_redact_preserves_context() -> None:
    text = "AWS=AKIAIOSFODNN7EXAMPLE; rest of sentence."
    redacted = redact_secrets(text)
    assert redacted.startswith("AWS=[REDACTED]")
    assert "rest of sentence" in redacted


# ── 深化扩展 (原 test_privacy_deepening.py) ───────────────────────────────

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
