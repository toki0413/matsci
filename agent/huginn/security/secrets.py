"""Secret backend abstraction for Huginn.

The default backend reads secrets from environment variables. Additional
backends (e.g. HashiCorp Vault, AWS Secrets Manager, Azure Key Vault) can be
implemented by subclassing ``SecretBackend`` and registering them via
``HUGINN_SECRET_BACKEND``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class SecretBackend(ABC):
    """Abstract secret storage backend."""

    @abstractmethod
    def get(self, name: str) -> str | None:
        """Return the secret value for ``name`` or None if not found."""
        raise NotImplementedError

    @abstractmethod
    def set(self, name: str, value: str) -> None:
        """Store ``value`` under ``name``."""
        raise NotImplementedError


class EnvSecretBackend(SecretBackend):
    """Read secrets from environment variables."""

    def get(self, name: str) -> str | None:
        return os.environ.get(name)

    def set(self, name: str, value: str) -> None:
        os.environ[name] = value


class VaultSecretBackend(SecretBackend):
    """HashiCorp Vault backend (placeholder).

    Requires the ``hvac`` package and ``HUGINN_VAULT_ADDR`` /
    ``HUGINN_VAULT_TOKEN`` environment variables.
    """

    def __init__(self, addr: str | None = None, token: str | None = None) -> None:
        self.addr = addr or os.environ.get("HUGINN_VAULT_ADDR", "")
        self.token = token or os.environ.get("HUGINN_VAULT_TOKEN", "")

    def get(self, name: str) -> str | None:
        try:
            import hvac  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError("pip install hvac to use VaultSecretBackend") from err

        client = hvac.Client(url=self.addr, token=self.token)
        path, key = name.split(":", 1) if ":" in name else (name, "value")
        response = client.secrets.kv.v2.read_secret_version(path=path)
        return response["data"]["data"].get(key)

    def set(self, name: str, value: str) -> None:
        raise NotImplementedError("VaultSecretBackend.set is not implemented")


def get_secret_backend() -> SecretBackend:
    """Return the configured secret backend."""
    backend_name = os.environ.get("HUGINN_SECRET_BACKEND", "env").lower()
    if backend_name == "vault":
        return VaultSecretBackend()
    return EnvSecretBackend()
