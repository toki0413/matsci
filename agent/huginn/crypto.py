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
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import logging
logger = logging.getLogger(__name__)


# 流式加密格式 magic + version, 用来和 encrypt() 的 base64 密文区分.
# 旧 encrypt_stream 一把读 src 进内存 (假流式), 对几 GB 轨迹文件会 OOM.
# 新格式走 chunked AES-GCM, 每块独立 nonce + 16 字节 tag, 真流式 + 完整性.
_STREAM_MAGIC = b"HGS1"
# Vault 主路径密文 magic (新格式). 没有 magic 前缀的视为旧 base64(salt+ct) 格式,
# decrypt 时走 _decrypt_legacy. 这样已有数据不丢, 新数据走快路径.
_VAULT_MAGIC = b"HGV2"


def _write_chunk(dst: BinaryIO, aesgcm: AESGCM, nonce: bytes, plaintext: bytes) -> None:
    """写一个 AES-GCM chunk: nonce(12) + ct_len(4 BE) + ciphertext."""
    ct = aesgcm.encrypt(nonce, plaintext, None)
    dst.write(nonce)
    dst.write(len(ct).to_bytes(4, "big"))
    dst.write(ct)



class CryptoVault:
    """Encrypted vault for sensitive data.

    Key derivation: PBKDF2-SHA256 (480k iterations) runs ONCE at unlock()
    with a vault-scope salt; encrypt/decrypt reuse the cached Fernet key.
    Fernet's per-message random IV already gives distinct ciphertexts for
    identical plaintexts, so the old per-item salt design was just burning
    ~200ms of KDF on every call — brutal for RAG indexing (thousands of
    chunks). Old base64(salt+ct) ciphertexts still decrypt via the legacy
    path, so existing data keeps working.
    """

    SALT_LENGTH = 32
    ITERATIONS = 480_000  # OWASP recommended minimum for PBKDF2-SHA256
    # Vault-scope salt for the one-time master KDF. Randomness comes from
    # the password itself; this just needs to be a fixed, distinct value.
    _MASTER_SALT = b"huginn-vault-master-v1"

    def __init__(self, master_password: str | None = None):
        self._password: str | None = None
        self._master_key: bytes | None = None  # base64-encoded Fernet key
        if master_password:
            self.unlock(master_password)

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive a Fernet key (base64-encoded 32 bytes) from password + salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))

    def unlock(self, password: str) -> None:
        """Unlock the vault with a master password (runs PBKDF2 once)."""
        self._password = password
        self._master_key = self._derive_key(password, self._MASTER_SALT)

    def is_unlocked(self) -> bool:
        return self._master_key is not None

    def lock(self) -> None:
        """Clear key material from memory."""
        self._password = None
        self._master_key = None

    def _require_unlocked(self) -> bytes:
        if self._master_key is None:
            raise RuntimeError("Vault is locked. Call unlock() first.")
        return self._master_key

    def encrypt(self, plaintext: str | bytes) -> bytes:
        """Encrypt plaintext. Returns ``HGV2``-prefixed Fernet ciphertext."""
        key = self._require_unlocked()
        fernet = Fernet(key)
        data = plaintext.encode("utf-8") if isinstance(plaintext, str) else plaintext
        # 新格式: magic + fernet_ct (raw). Fernet 自带 per-message IV + HMAC,
        # 相同明文不会得到相同密文, 不需要 per-item salt.
        return _VAULT_MAGIC + fernet.encrypt(data)

    def decrypt(self, ciphertext: bytes) -> str:
        """Decrypt ciphertext (new ``HGV2`` format or legacy base64 format)."""
        key = self._require_unlocked()
        if ciphertext.startswith(_VAULT_MAGIC):
            fernet = Fernet(key)
            return fernet.decrypt(ciphertext[len(_VAULT_MAGIC) :]).decode("utf-8")
        # 旧格式: base64(salt + fernet_ct), 用 embedded salt 重新 KDF.
        return self._decrypt_legacy(ciphertext).decode("utf-8")

    def encrypt_bytes(self, plaintext: str | bytes) -> bytes:
        """Encrypt and return raw bytes (same as encrypt but explicit)."""
        return self.encrypt(plaintext)

    def decrypt_bytes(self, ciphertext: bytes) -> bytes:
        """Decrypt to raw bytes."""
        key = self._require_unlocked()
        if ciphertext.startswith(_VAULT_MAGIC):
            fernet = Fernet(key)
            return fernet.decrypt(ciphertext[len(_VAULT_MAGIC) :])
        return self._decrypt_legacy(ciphertext)

    def _decrypt_legacy(self, ciphertext: bytes) -> bytes:
        """解旧格式密文 base64(salt + fernet_ct). 需要 _password 重新 KDF."""
        if self._password is None:
            raise RuntimeError("Vault is locked. Call unlock() first.")
        raw = base64.b64decode(ciphertext)
        salt = raw[: self.SALT_LENGTH]
        encrypted_data = raw[self.SALT_LENGTH :]
        legacy_key = self._derive_key(self._password, salt)
        return Fernet(legacy_key).decrypt(encrypted_data)

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
        """Encrypt a stream chunk-by-chunk (memory-bounded for large files).

        Format: ``HGS1`` magic + 32-byte salt + 4-byte chunk_size (BE),
        then per chunk: 12-byte nonce + 4-byte ct_len (BE, includes 16-byte
        GCM tag) + ciphertext. A final empty plaintext chunk marks EOF so
        truncation is detectable on decrypt.

        Per-stream salt + PBKDF2 here is intentional: streams may outlive a
        single vault session (e.g. archived trajectory dumps), so deriving
        a stream-scoped key from the master password + fresh salt keeps
        each stream self-contained. The KDF runs once per stream, not per
        chunk.
        """
        password = self._password
        if password is None:
            raise RuntimeError("Vault is locked. Call unlock() first.")
        salt = os.urandom(self.SALT_LENGTH)
        # PBKDF2 raw 32 bytes 喂 AESGCM (Fernet 要 base64 包装, AESGCM 要 raw).
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        aesgcm = AESGCM(kdf.derive(password.encode("utf-8")))

        dst.write(_STREAM_MAGIC)
        dst.write(salt)
        dst.write(chunk_size.to_bytes(4, "big"))

        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                # 空明文 chunk 当 EOF marker, 即便源文件恰好是 chunk_size 整数倍
                # 也能让 decrypt 干净退出.
                _write_chunk(dst, aesgcm, os.urandom(12), b"")
                break
            _write_chunk(dst, aesgcm, os.urandom(12), chunk)

    def decrypt_stream(self, src: BinaryIO, dst: BinaryIO) -> None:
        """Decrypt a chunked AES-GCM stream produced by ``encrypt_stream``."""
        password = self._password
        if password is None:
            raise RuntimeError("Vault is locked. Call unlock() first.")
        magic = src.read(4)
        if magic != _STREAM_MAGIC:
            raise ValueError(
                f"invalid stream magic: expected {_STREAM_MAGIC!r}, got {magic!r}"
            )
        salt = src.read(self.SALT_LENGTH)
        if len(salt) != self.SALT_LENGTH:
            raise ValueError("truncated stream header (salt)")
        # chunk_size 在 header 里只是回放, 解密侧按 ct_len 字段读, 不依赖它.
        _len_bytes = src.read(4)
        if len(_len_bytes) != 4:
            raise ValueError("truncated stream header (chunk_size)")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.ITERATIONS,
        )
        aesgcm = AESGCM(kdf.derive(password.encode("utf-8")))

        while True:
            nonce = src.read(12)
            if not nonce:
                break  # 干净 EOF
            if len(nonce) < 12:
                raise ValueError("truncated stream (partial nonce)")
            len_bytes = src.read(4)
            if len(len_bytes) < 4:
                raise ValueError("truncated stream (partial ct_len)")
            ct_len = int.from_bytes(len_bytes, "big")
            ct = src.read(ct_len)
            if len(ct) < ct_len:
                raise ValueError("truncated stream (partial ciphertext)")
            # GCM tag 校验失败会抛 InvalidTag, 调用方拿到异常就知道数据被篡改.
            pt = aesgcm.decrypt(nonce, ct, None)
            if not pt:
                break  # EOF marker chunk
            dst.write(pt)

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

        # Use master key directly as vault password (it's already high-entropy).
        # 走 unlock() 而不是直接塞 _password, 这样 CryptoVault 内部的 master_key
        # 缓存被正确填上 (新版 encrypt/decrypt 复用这个缓存, 不再每次 KDF).
        vault = CryptoVault()
        vault.unlock(self._master_key.decode("utf-8"))
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
        """Encrypt and save, then securely cleanup temporary files."""
        if not self.is_mounted():
            return

        try:
            if save and self._plaintext_path and self._plaintext_path.exists():
                self.encrypted_path.parent.mkdir(parents=True, exist_ok=True)
                self.vault.encrypt_file(self._plaintext_path, self.encrypted_path)
        finally:
            # Securely erase all files in temp dir before removing it
            if self._temp_dir and self._temp_dir.exists():
                for child in self._temp_dir.rglob("*"):
                    if child.is_file():
                        try:
                            secure_erase_file(child)
                        except Exception:
                            logger.debug("secure erase file failed", exc_info=True)
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
