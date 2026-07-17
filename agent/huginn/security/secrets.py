"""Secret backend abstraction for Huginn.

The default backend reads secrets from environment variables. Additional
backends (e.g. HashiCorp Vault, AWS Secrets Manager) can be implemented
by subclassing ``SecretBackend`` and registering them via
``HUGINN_SECRET_BACKEND``.

Phase 4 additions:
- ``VaultSecretBackend.set()`` — write secrets to Vault KV v2
- ``AWSSecretsManagerBackend`` — read/write via AWS Secrets Manager
- ``SecretRotationHook`` — automatic rotation callbacks
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

logger = logging.getLogger(__name__)


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

    def delete(self, name: str) -> bool:
        """Delete a secret. Returns True on success."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support deletion"
        )

    def list_keys(self, prefix: str = "") -> list[str]:
        """List secret names matching a prefix. Optional for backends."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support listing"
        )

    def exists(self, name: str) -> bool:
        """Check if a secret exists."""
        return self.get(name) is not None


class EnvSecretBackend(SecretBackend):
    """Read/write secrets as environment variables.

    Supports an optional file-backed overlay for persistence across restarts.
    """

    def __init__(self, persist_file: str | None = None) -> None:
        self._persist_file = persist_file
        self._store: dict[str, str] = {}
        if persist_file:
            self._load_persisted()

    def _load_persisted(self) -> None:
        import os
        try:
            with open(self._persist_file, "rb") as f:
                raw = f.read()
            # Try encrypted format first, fall back to legacy plaintext
            try:
                from cryptography.fernet import Fernet
                key_path = os.path.join(
                    os.path.dirname(self._persist_file), ".secret_key"
                )
                with open(key_path, "rb") as kf:
                    key = kf.read()
                fernet = Fernet(key)
                decrypted = fernet.decrypt(raw)
                self._store = json.loads(decrypted)
            except Exception:
                # Legacy plaintext — migrate on next save
                self._store = json.loads(raw.decode("utf-8"))
            # Merge into env so other code sees them
            for k, v in self._store.items():
                os.environ.setdefault(k, v)
        except (FileNotFoundError, json.JSONDecodeError):
            self._store = {}

    def _save_persisted(self) -> None:
        if not self._persist_file:
            return
        # Encrypt at rest — never write secrets as plaintext JSON.
        from cryptography.fernet import Fernet
        import tempfile, os
        key_path = os.path.join(os.path.dirname(self._persist_file), ".secret_key")
        try:
            if os.path.exists(key_path):
                with open(key_path, "rb") as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                with open(key_path, "wb") as f:
                    f.write(key)
                os.chmod(key_path, 0o600)
            fernet = Fernet(key)
            payload = json.dumps(self._store).encode("utf-8")
            encrypted = fernet.encrypt(payload)
            # Atomic write
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self._persist_file))
            try:
                os.write(fd, encrypted)
                os.close(fd)
                os.replace(tmp, self._persist_file)
            except Exception:
                os.close(fd) if not os.get_inheritable(fd) else None
                os.unlink(tmp)
                raise
        except Exception as e:
            # 加密失败必须向上抛，否则调用方会误以为保存成功
            logger.error(
                "Failed to encrypt persisted secrets: %s", e, exc_info=True
            )
            raise

    def get(self, name: str) -> str | None:
        return os.environ.get(name)

    def set(self, name: str, value: str) -> None:
        os.environ[name] = value
        self._store[name] = value
        self._save_persisted()

    def delete(self, name: str) -> bool:
        removed = os.environ.pop(name, None) is not None
        self._store.pop(name, None)
        self._save_persisted()
        return removed

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = set(os.environ.keys()) | set(self._store.keys())
        return sorted(k for k in keys if k.startswith(prefix))


class VaultSecretBackend(SecretBackend):
    """HashiCorp Vault backend using KV v2 engine.

    Requires the ``hvac`` package and ``HUGINN_VAULT_ADDR`` /
    ``HUGINN_VAULT_TOKEN`` environment variables.

    Secret names use the format ``path:key`` where ``path`` is the KV
    secret path and ``key`` is the field within the secret data.
    """

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        mount_point: str = "secret",
    ) -> None:
        self.addr = addr or os.environ.get("HUGINN_VAULT_ADDR", "")
        self.token = token or os.environ.get("HUGINN_VAULT_TOKEN", "")
        self.mount_point = mount_point

    def _client(self):
        try:
            import hvac  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError("pip install hvac to use VaultSecretBackend") from err
        return hvac.Client(url=self.addr, token=self.token)

    @staticmethod
    def _parse_name(name: str) -> tuple[str, str]:
        """Parse ``path:key`` into (path, key).  Defaults key to 'value'."""
        if ":" in name:
            path, key = name.split(":", 1)
        else:
            path, key = name, "value"
        return path, key

    def get(self, name: str) -> str | None:
        client = self._client()
        path, key = self._parse_name(name)
        try:
            response = client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self.mount_point
            )
            return response["data"]["data"].get(key)
        except Exception as exc:
            logger.warning("Vault get('%s') failed: %s", name, exc)
            return None

    def set(self, name: str, value: str) -> None:
        """Write a secret to Vault KV v2.

        If the secret path already exists, the new key is merged with existing
        data.  Otherwise a new secret is created.
        """
        client = self._client()
        path, key = self._parse_name(name)

        # Read existing data to merge
        existing: dict[str, str] = {}
        try:
            response = client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self.mount_point
            )
            existing = response.get("data", {}).get("data", {})
        except Exception as exc:
            logger.debug("Vault path '%s' doesn't exist yet: %s", path, exc)

        existing[key] = value
        client.secrets.kv.v2.create_or_update_secret(
            path=path,
            secret=existing,
            mount_point=self.mount_point,
        )

    def delete(self, name: str) -> bool:
        client = self._client()
        path, key = self._parse_name(name)
        try:
            # Read existing, remove the specific key, write back
            response = client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self.mount_point
            )
            data = response.get("data", {}).get("data", {})
            if key in data:
                del data[key]
                if data:
                    client.secrets.kv.v2.create_or_update_secret(
                        path=path, secret=data, mount_point=self.mount_point
                    )
                else:
                    client.secrets.kv.v2.delete_metadata_and_all_versions(
                        path=path, mount_point=self.mount_point
                    )
                return True
            return False
        except Exception:
            return False

    def exists(self, name: str) -> bool:
        return self.get(name) is not None


class AWSSecretsManagerBackend(SecretBackend):
    """AWS Secrets Manager backend.

    Requires the ``boto3`` package and valid AWS credentials (via env vars,
    instance profile, or ``AWS_PROFILE``).

    Secret names map directly to AWS Secrets Manager secret IDs.
    """

    def __init__(
        self,
        region_name: str | None = None,
        prefix: str = "",
    ) -> None:
        self.region_name = region_name or os.environ.get(
            "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        self.prefix = prefix  # e.g. "huginn/prod/"

    def _client(self):
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "pip install boto3 to use AWSSecretsManagerBackend"
            ) from err
        return boto3.client("secretsmanager", region_name=self.region_name)

    def _full_name(self, name: str) -> str:
        return f"{self.prefix}{name}" if self.prefix else name

    def get(self, name: str) -> str | None:
        client = self._client()
        try:
            response = client.get_secret_value(SecretId=self._full_name(name))
            return response.get("SecretString")
        except client.exceptions.ResourceNotFoundException:
            return None
        except Exception:
            return None

    def set(self, name: str, value: str) -> None:
        client = self._client()
        full_name = self._full_name(name)
        try:
            client.put_secret_value(
                SecretId=full_name,
                SecretString=value,
            )
        except client.exceptions.ResourceNotFoundException:
            client.create_secret(
                Name=full_name,
                SecretString=value,
            )

    def delete(self, name: str) -> bool:
        client = self._client()
        try:
            client.delete_secret(
                SecretId=self._full_name(name),
                ForceDeleteWithoutRecovery=True,
            )
            return True
        except Exception:
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        client = self._client()
        try:
            paginator = client.get_paginator("list_secrets")
            results = []
            full_prefix = self._full_name(prefix)
            for page in paginator.paginate():
                for secret in page.get("SecretList", []):
                    name = secret["Name"]
                    if name.startswith(full_prefix):
                        # Strip the backend prefix for the caller
                        if self.prefix and name.startswith(self.prefix):
                            name = name[len(self.prefix):]
                        results.append(name)
            return sorted(results)
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Secret rotation hooks
# ---------------------------------------------------------------------------

class SecretRotationHook:
    """Register callbacks for automatic secret rotation.

    A rotation callback receives the old secret value and must return the
    new secret value.  The hook then writes the new value back to the
    backend.

    Usage::

        hooks = SecretRotationHook(backend)
        hooks.register("MP_API_KEY", rotate_mp_key)
        hooks.rotate("MP_API_KEY")
        hooks.rotate_all()  # rotate everything registered
    """

    def __init__(self, backend: SecretBackend) -> None:
        self.backend = backend
        self._callbacks: dict[str, Callable[[str | None], str]] = {}
        self._last_rotated: dict[str, float] = {}

    def register(
        self,
        name: str,
        callback: Callable[[str | None], str],
    ) -> None:
        """Register a rotation callback for a named secret."""
        self._callbacks[name] = callback

    def unregister(self, name: str) -> None:
        self._callbacks.pop(name, None)

    def rotate(self, name: str) -> str:
        """Rotate a single secret.  Returns the new value."""
        if name not in self._callbacks:
            raise KeyError(f"No rotation callback registered for '{name}'")

        old_value = self.backend.get(name)
        new_value = self._callbacks[name](old_value)
        self.backend.set(name, new_value)
        self._last_rotated[name] = time.time()
        logger.info("Rotated secret '%s'", name)
        return new_value

    def rotate_all(self) -> dict[str, bool]:
        """Rotate all registered secrets.  Returns {name: success}."""
        results: dict[str, bool] = {}
        for name in self._callbacks:
            try:
                self.rotate(name)
                results[name] = True
            except Exception as exc:
                logger.error("Failed to rotate '%s': %s", name, exc)
                results[name] = False
        return results

    def needs_rotation(
        self, name: str, max_age_seconds: float = 86400.0
    ) -> bool:
        """Check if a secret has exceeded its max age since last rotation."""
        last = self._last_rotated.get(name)
        if last is None:
            return True
        return (time.time() - last) > max_age_seconds

    def last_rotated(self, name: str) -> float | None:
        return self._last_rotated.get(name)

    @property
    def registered_names(self) -> list[str]:
        return sorted(self._callbacks.keys())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type[SecretBackend]] = {
    "env": EnvSecretBackend,
    "vault": VaultSecretBackend,
    "aws": AWSSecretsManagerBackend,
}


def register_backend(name: str, cls: type[SecretBackend]) -> None:
    """Register a custom secret backend type."""
    _BACKEND_REGISTRY[name.lower()] = cls


def get_secret_backend(name: str | None = None) -> SecretBackend:
    """Return the configured secret backend.

    Reads ``HUGINN_SECRET_BACKEND`` env var if *name* is not provided.
    Supported values: ``env``, ``vault``, ``aws``, or any registered name.
    """
    backend_name = (name or os.environ.get("HUGINN_SECRET_BACKEND", "env")).lower()
    cls = _BACKEND_REGISTRY.get(backend_name)
    if cls is None:
        raise ValueError(
            f"Unknown secret backend '{backend_name}'. "
            f"Available: {sorted(_BACKEND_REGISTRY)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# 便捷函数 (G24): 集中 API key 读取入口
# 新代码优先用 get() / get_or_none() 而非散落 os.environ.get(), 方便审计
# ---------------------------------------------------------------------------

class SecretsError(RuntimeError):
    """缺失必需 secret 时抛出."""


def get(name: str) -> str:
    """读 secret, 缺失 raise SecretsError. 用于必需 key."""
    val = get_secret_backend().get(name)
    if not val:
        raise SecretsError(f"Missing required secret: {name}")
    return val


def get_or_none(name: str) -> str | None:
    """读 secret, 缺失返回 None. 用于可选 key."""
    return get_secret_backend().get(name)


def validate_required(*names: str) -> None:
    """启动时校验一组 key 非空, 缺失任一 raise SecretsError.

    调用方按 provider 决定校验哪些:
        validate_required("DEEPSEEK_API_KEY")  # deepseek
        validate_required("OPENAI_API_KEY")    # openai
    """
    backend = get_secret_backend()
    missing = [n for n in names if not backend.get(n)]
    if missing:
        raise SecretsError(f"Missing required secrets: {missing}")
