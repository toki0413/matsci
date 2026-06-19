"""Tests for encrypted config persistence and keyring resolution."""

from __future__ import annotations

import pytest
from cryptography.fernet import InvalidToken

from huginn.config import HuginnConfig, ModelConfig


def test_encrypted_config_roundtrip(tmp_path):
    """Saving with encrypt_config should produce an unreadable but loadable file."""
    cfg = HuginnConfig(
        provider="openai",
        api_key="sk-secret",
        encrypt_config=True,
        encryption_password="super-pass",
        models=[
            ModelConfig(alias="gpt4", provider="openai", api_key="sk-model-secret")
        ],
    )
    path = tmp_path / "config.enc"
    cfg.save(path)

    # File should not contain the plaintext secret
    raw = path.read_bytes()
    assert b"sk-secret" not in raw
    assert b"sk-model-secret" not in raw

    loaded = HuginnConfig.load(path, password="super-pass")
    assert loaded.api_key == "sk-secret"
    assert loaded.models[0].api_key == "sk-model-secret"


def test_encrypted_config_wrong_password(tmp_path):
    cfg = HuginnConfig(
        api_key="secret",
        encrypt_config=True,
        encryption_password="right",
    )
    path = tmp_path / "config.enc"
    cfg.save(path)

    with pytest.raises(InvalidToken):
        HuginnConfig.load(path, password="wrong")


def test_plain_config_save_load(tmp_path):
    cfg = HuginnConfig(api_key="plain-secret")
    path = tmp_path / "config.toml"
    cfg.save(path)

    loaded = HuginnConfig.load(path)
    assert loaded.api_key == "plain-secret"


def test_to_dict_masks_password_and_api_key():
    cfg = HuginnConfig(
        api_key="secret",
        encryption_password="pass",
    )
    data = cfg.to_dict(mask_key=True)
    assert data["api_key"] == "***"
    assert data["encryption_password"] == "***"
