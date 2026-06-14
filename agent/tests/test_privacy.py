"""Tests for the privacy / secret scanner."""

from __future__ import annotations

from matsci_agent.privacy import scan_for_secrets, redact_secrets


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
