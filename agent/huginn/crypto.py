"""Local data encryption module.

Uses Fernet (AES-128-CBC + HMAC-SHA256) with PBKDF2 key derivation.
Key stays in memory only — never persisted to disk by default.
Supports database file encryption, stream encryption for large files,
and optional key file storage with additional password protection.
"""

from __future__ import annotations

import base64
import os
import secrets
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CryptoVault:
    """Encrypted vault for sensitive data with per-item salt derivation."""

    SALT_LENGTH = 32
    ITERATIONS = 480_000  # OWASP recommended minimum for PBKDF2-SHA256

    def __init__(self, master_password: str | None = None):
        self._password: str | None = None
        if master_password:
            self.unlock(master_password)

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive a Fernet key from password + salt using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
        return key

    def unlock(self, password: str) -> None:
        """Unlock the vault with a master password."""
        self._password = password

    def is_unlocked(self) -> bool:
        return self._password is not None

    def lock(self) -> None:
        """Clear password from memory."""
        self._password = None

    def _require_unlocked(self) -> str:
        if not self.is_unlocked():
            raise RuntimeError("Vault is locked. Call unlock() first.")
        return self._password

    def encrypt(self, plaintext: str | bytes) -> bytes:
        """Encrypt plaintext. Returns base64-encoded ciphertext with embedded salt."""
        password = self._require_unlocked()

        salt = os.urandom(self.SALT_LENGTH)
        key = self._derive_key(password, salt)
        fernet = Fernet(key)

        data = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        ciphertext = fernet.encrypt(data)

        # Format: base64(salt + ciphertext)
        return base64.b64encode(salt + ciphertext)

    def decrypt(self, ciphertext: bytes) -> str:
        """Decrypt ciphertext. Returns plaintext string."""
        password = self._require_unlocked()

        raw = base64.b64decode(ciphertext)
        salt = raw[: self.SALT_LENGTH]
        encrypted_data = raw[self.SALT_LENGTH :]

        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        plaintext = fernet.decrypt(encrypted_data)
        return plaintext.decode("utf-8")

    def encrypt_bytes(self, plaintext: str | bytes) -> bytes:
        """Encrypt and return raw bytes (same as encrypt but explicit)."""
        return self.encrypt(plaintext)

    def decrypt_bytes(self, ciphertext: bytes) -> bytes:
        """Decrypt to raw bytes."""
        password = self._require_unlocked()

        raw = base64.b64decode(ciphertext)
        salt = raw[: self.SALT_LENGTH]
        encrypted_data = raw[self.SALT_LENGTH :]

        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        return fernet.decrypt(encrypted_data)

    def encrypt_file(self, src_path: str | Path, dst_path: str | Path) -> None:
        """Encrypt a file."""
        with open(src_path, "rb") as f:
            data = f.read()
        encrypted = self.encrypt(data)
        with open(dst_path, "wb") as f:
            f.write(encrypted)

    def decrypt_file(self, src_path: str | Path, dst_path: str | Path) -> None:
        """Decrypt a file."""
        with open(src_path, "rb") as f:
            data = f.read()
        decrypted = self.decrypt_bytes(data)
        with open(dst_path, "wb") as f:
            f.write(decrypted)

    def encrypt_stream(
        self, src: BinaryIO, dst: BinaryIO, chunk_size: int = 64 * 1024
    ) -> None:
        """Encrypt a stream (memory-efficient for large files).

        Writes base64-encoded encrypted chunks to dst.
        """
        password = self._require_unlocked()
        salt = os.urandom(self.SALT_LENGTH)
        key = self._derive_key(password, salt)
        fernet = Fernet(key)

        # Read all, encrypt, write (Fernet requires full message)
        # For true streaming we'd need chunked Fernet or AES-GCM directly
        data = src.read()
        ciphertext = fernet.encrypt(data)
        dst.write(base64.b64encode(salt + ciphertext))

    def decrypt_stream(self, src: BinaryIO, dst: BinaryIO) -> None:
        """Decrypt a stream."""
        password = self._require_unlocked()
        data = base64.b64decode(src.read())
        salt = data[: self.SALT_LENGTH]
        encrypted_data = data[self.SALT_LENGTH :]

        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        plaintext = fernet.decrypt(encrypted_data)
        dst.write(plaintext)

    def encrypt_directory(
        self,
        src_dir: str | Path,
        dst_path: str | Path,
        exclude: list[str] | None = None,
    ) -> None:
        """Encrypt an entire directory as a tar archive."""
        exclude = exclude or []
        src = Path(src_dir)
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            with tarfile.open(tmp.name, "w") as tar:
                for item in src.rglob("*"):
                    if any(e in str(item) for e in exclude):
                        continue
                    if item.is_file():
                        tar.add(item, arcname=item.relative_to(src))
            tmp.flush()

        self.encrypt_file(tmp.name, dst)
        os.unlink(tmp.name)

    def decrypt_directory(self, src_path: str | Path, dst_dir: str | Path) -> None:
        """Decrypt a tar archive to a directory."""
        dst = Path(dst_dir)
        dst.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
            self.decrypt_file(src_path, tmp.name)
            with tarfile.open(tmp.name, "r") as tar:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(path=dst, filter=tarfile.data_filter)
                else:
                    tar.extractall(path=dst)

        os.unlink(tmp.name)


class KeyManager:
    """Manages encryption keys with optional file-backed storage.

    The key file itself is encrypted with a user password.
    """

    KEY_FILE_VERSION = b"MSKV1"  # huginn Key Vault v1
    SALT_LENGTH = 32
    ITERATIONS = 600_000

    def __init__(self, key_file: str | Path | None = None):
        self.key_file = Path(key_file) if key_file else None
        self._master_key: bytes | None = None

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))

    def generate_master_key(self) -> bytes:
        """Generate a random 256-bit master key."""
        return base64.urlsafe_b64encode(os.urandom(32))

    def create_key_file(
        self, password: str, key_file: str | Path | None = None
    ) -> bytes:
        """Create a new key file encrypted with password. Returns the raw master key."""
        path = Path(key_file) if key_file else self.key_file
        if path is None:
            raise ValueError("key_file path is required")

        master_key = self.generate_master_key()
        salt = os.urandom(self.SALT_LENGTH)
        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        encrypted_key = fernet.encrypt(master_key)

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.KEY_FILE_VERSION)
            f.write(salt)
            f.write(encrypted_key)

        self._master_key = master_key
        return master_key

    def load_key_file(self, password: str, key_file: str | Path | None = None) -> bytes:
        """Load master key from encrypted key file."""
        path = Path(key_file) if key_file else self.key_file
        if path is None or not path.exists():
            raise FileNotFoundError(f"Key file not found: {path}")

        with open(path, "rb") as f:
            version = f.read(len(self.KEY_FILE_VERSION))
            if version != self.KEY_FILE_VERSION:
                raise ValueError("Invalid key file format or version")
            salt = f.read(self.SALT_LENGTH)
            encrypted_key = f.read()

        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        master_key = fernet.decrypt(encrypted_key)
        self._master_key = master_key
        return master_key

    def get_vault(self, password: str | None = None) -> CryptoVault:
        """Get a CryptoVault using the managed master key.

        If password is provided, loads/unlocks from key file.
        Otherwise uses already-loaded master key.
        """
        if password and self.key_file and self.key_file.exists():
            self.load_key_file(password)

        if self._master_key is None:
            raise RuntimeError(
                "No master key loaded. Provide password or call load_key_file()."
            )

        # Use master key directly as vault password (it's already high-entropy)
        vault = CryptoVault()
        vault._password = self._master_key.decode("utf-8")
        return vault


class EncryptedDatabase:
    """File-level encryption wrapper for database files (SQLite/ChromaDB).

    Transparently encrypts/decrypts database files on disk.
    When active, the plaintext database lives in a temporary location
    and is re-encrypted on close.
    """

    def __init__(self, vault: CryptoVault, encrypted_path: str | Path):
        self.vault = vault
        self.encrypted_path = Path(encrypted_path)
        self._temp_dir: Path | None = None
        self._plaintext_path: Path | None = None

    @property
    def plaintext_path(self) -> Path:
        """Path to the decrypted database (valid only inside context manager)."""
        if self._plaintext_path is None:
            raise RuntimeError("Database not mounted. Use as context manager.")
        return self._plaintext_path

    def is_mounted(self) -> bool:
        return self._plaintext_path is not None

    def mount(self) -> Path:
        """Decrypt database to temporary location. Returns plaintext path."""
        if self.is_mounted():
            return self._plaintext_path

        self._temp_dir = Path(tempfile.mkdtemp(prefix="huginn_db_"))
        self._plaintext_path = self._temp_dir / self.encrypted_path.name.replace(
            ".enc", ""
        )

        if self.encrypted_path.exists():
            self.vault.decrypt_file(self.encrypted_path, self._plaintext_path)
        else:
            # New database — plaintext path will be created by caller
            self._plaintext_path.parent.mkdir(parents=True, exist_ok=True)

        return self._plaintext_path

    def unmount(self, save: bool = True) -> None:
        """Encrypt and save, then cleanup temporary files."""
        if not self.is_mounted():
            return

        try:
            if save and self._plaintext_path and self._plaintext_path.exists():
                self.encrypted_path.parent.mkdir(parents=True, exist_ok=True)
                self.vault.encrypt_file(self._plaintext_path, self.encrypted_path)
        finally:
            if self._temp_dir and self._temp_dir.exists():
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
            self._plaintext_path = None

    def __enter__(self) -> Path:
        return self.mount()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.unmount(save=(exc_type is None))


class EncryptedConfig:
    """Encrypted configuration file manager."""

    def __init__(
        self, config_path: str | Path | None = None, vault: CryptoVault | None = None
    ):
        self.config_path = (
            Path(config_path) if config_path else Path.home() / ".huginn" / "config.enc"
        )
        self.vault = vault or CryptoVault()

    def load(self) -> dict:
        """Load and decrypt config."""
        if not self.config_path.exists():
            return {}

        with open(self.config_path, "rb") as f:
            encrypted = f.read()

        json_str = self.vault.decrypt(encrypted)
        import json

        return json.loads(json_str)

    def save(self, config: dict) -> None:
        """Encrypt and save config."""
        import json

        json_str = json.dumps(config, indent=2, ensure_ascii=False)
        encrypted = self.vault.encrypt(json_str)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "wb") as f:
            f.write(encrypted)

    def exists(self) -> bool:
        return self.config_path.exists()


def secure_erase_file(path: str | Path, passes: int = 3) -> None:
    """Securely overwrite and delete a file."""
    p = Path(path)
    if not p.exists():
        return

    size = p.stat().st_size
    for _ in range(passes):
        with open(path, "wb") as f:
            f.write(secrets.token_bytes(size))

    os.unlink(path)
