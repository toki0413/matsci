"""Tests for the CredentialStore <-> ModelRegistry bridge.

Covers:
  * ModelConfig.credential_id field exists and defaults to None
  * ModelRegistry.get() falls back to CredentialStore when api_key is empty
  * Graceful degradation when credential_id points to a missing credential
  * CredentialStore.import_from_config() batch import from HuginnConfig
  * Skipping already-imported credentials on repeated import
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch

from cryptography.fernet import Fernet

from huginn.config import HuginnConfig, ModelConfig
from huginn.models.registry import ModelRegistry
from huginn.security.credential_store import CRED_KIND_LLM, CredentialStore


# ── helpers ────────────────────────────────────────────────────────────

def _make_store(tmp_path) -> CredentialStore:
    """Isolated CredentialStore backed by a temp DB + throwaway Fernet key."""
    fernet = Fernet(Fernet.generate_key())
    return CredentialStore(tmp_path / "creds.sqlite", fernet=fernet)


# ── ModelConfig.credential_id ──────────────────────────────────────────

def test_credential_id_field_exists():
    """ModelConfig should expose credential_id, defaulting to None."""
    field_names = {f.name for f in fields(ModelConfig)}
    assert "credential_id" in field_names

    cfg = ModelConfig(alias="test", provider="openai")
    assert cfg.credential_id is None


# ── ModelRegistry.get() fallback ───────────────────────────────────────
#
# All registry tests mock create_langchain_model so we never hit a real
# LLM provider. Provider is set to "default" which has no env-var mapping,
# so resolve_provider_key() always returns None when api_key is empty.

def test_registry_fallback_to_credential_store():
    """When api_key is empty but credential_id is set, the registry pulls
    the key from CredentialStore."""
    cfg = ModelConfig(
        alias="bridged",
        provider="default",
        model="gpt-4o",
        api_key=None,
        credential_id="cred-abc",
    )
    registry = ModelRegistry(models=[cfg])

    fake_store = MagicMock()
    fake_store.to_llm_info.return_value = {
        "api_key": "sk-from-store",
        "model": "",
        "base_url": None,
    }

    with patch(
        "huginn.security.credential_store.get_credential_store",
        return_value=fake_store,
    ), patch("huginn.models.registry.create_langchain_model") as mock_create:
        registry.get("bridged")

    _, kwargs = mock_create.call_args
    assert kwargs["api_key"] == "sk-from-store"
    fake_store.to_llm_info.assert_called_once_with("cred-abc")


def test_registry_no_credential_id():
    """No api_key and no credential_id — key stays None, no store lookup."""
    cfg = ModelConfig(
        alias="bare",
        provider="default",
        model="gpt-4o",
        api_key=None,
        credential_id=None,
    )
    registry = ModelRegistry(models=[cfg])

    with patch("huginn.models.registry.create_langchain_model") as mock_create:
        registry.get("bare")

    _, kwargs = mock_create.call_args
    assert kwargs["api_key"] is None


def test_registry_credential_not_found():
    """credential_id is set but the credential doesn't exist — fall back to
    None without raising."""
    cfg = ModelConfig(
        alias="ghost",
        provider="default",
        model="gpt-4o",
        api_key=None,
        credential_id="does-not-exist",
    )
    registry = ModelRegistry(models=[cfg])

    fake_store = MagicMock()
    fake_store.to_llm_info.return_value = None  # credential not found

    with patch(
        "huginn.security.credential_store.get_credential_store",
        return_value=fake_store,
    ), patch("huginn.models.registry.create_langchain_model") as mock_create:
        registry.get("ghost")

    _, kwargs = mock_create.call_args
    assert kwargs["api_key"] is None


# ── CredentialStore.import_from_config ─────────────────────────────────

def test_import_from_config(tmp_path):
    """Only models with a plain-text api_key get imported; env:/keyring:
    references are skipped."""
    store = _make_store(tmp_path)

    config = HuginnConfig(models=[
        ModelConfig(
            alias="plain-key",
            provider="openai",
            model="gpt-4o",
            api_key="sk-abc123",
        ),
        ModelConfig(
            alias="env-ref",
            provider="deepseek",
            model="deepseek-chat",
            api_key="env:DEEPSEEK_API_KEY",
        ),
    ])

    result = store.import_from_config(config)

    # Only the plain-key model should be imported
    assert "plain-key" in result
    assert "env-ref" not in result
    assert len(result) == 1

    # Verify it landed in the DB with the right secret
    creds = store.list(CRED_KIND_LLM)
    assert len(creds) == 1
    assert creds[0]["name"] == "plain-key"

    cid = result["plain-key"]
    assert store.get_secret(cid) == "sk-abc123"


def test_import_from_config_skip_existing(tmp_path):
    """Importing twice should not create duplicate credentials."""
    store = _make_store(tmp_path)

    config = HuginnConfig(models=[
        ModelConfig(
            alias="plain-key",
            provider="openai",
            model="gpt-4o",
            api_key="sk-abc123",
        ),
    ])

    first = store.import_from_config(config)
    assert len(first) == 1

    second = store.import_from_config(config)
    assert len(second) == 0  # nothing new — already exists

    # Still only one entry in the store
    creds = store.list(CRED_KIND_LLM)
    assert len(creds) == 1
