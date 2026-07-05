"""High-level encrypted RAG manager for Huginn.

Integrates VectorStore, CryptoVault, and KeyManager into a single
interface with database-at-rest encryption support.
"""

from __future__ import annotations

import datetime
import shutil
import time
from pathlib import Path
from typing import Any

from huginn.crypto import CryptoVault, EncryptedDatabase, KeyManager
from huginn.rag.vector_store import EncryptedVectorStore, VectorStore


class EncryptedRAGManager:
    """Manages encrypted local vector databases for material science RAG.

    Provides transparent encryption at two levels:
    1. **Document-level**: Individual documents and metadata encrypted before storage.
    2. **Database-level**: The entire ChromaDB SQLite file encrypted at rest.

    Usage:
        mgr = EncryptedRAGManager(password="my_password")
        mgr.ingest_file("paper.pdf")
        results = mgr.search("band gap of silicon")
        mgr.lock()
    """

    def __init__(
        self,
        password: str | None = None,
        key_file: str | Path | None = None,
        persist_dir: str | None = None,
        db_encryption: bool = False,
        doc_encryption: bool = True,
        meta_encryption: bool = True,
    ):
        """Initialize the encrypted RAG manager.

        Args:
            password: Master password for encryption. If None, must call unlock().
            key_file: Path to encrypted key file. If provided, password unlocks it.
            persist_dir: Directory for ChromaDB persistence.
            db_encryption: If True, encrypt the entire database file at rest.
            doc_encryption: If True, encrypt document contents.
            meta_encryption: If True, encrypt sensitive metadata fields.
        """
        self._vault: CryptoVault | None = None
        self._key_manager: KeyManager | None = None
        self._db_encryption = db_encryption
        self._doc_encryption = doc_encryption
        self._meta_encryption = meta_encryption
        self._persist_dir = persist_dir
        self._db_wrapper: EncryptedDatabase | None = None
        self._store: VectorStore | EncryptedVectorStore | None = None

        if key_file:
            self._key_manager = KeyManager(key_file)
            if password:
                self._vault = self._key_manager.get_vault(password)
        elif password:
            self._vault = CryptoVault(password)

        self._init_store()

    def _init_store(self) -> None:
        """Initialize the underlying vector store."""
        if self._vault is not None and self._vault.is_unlocked():
            self._store = EncryptedVectorStore(
                vault=self._vault,
                persist_dir=self._persist_dir,
                encrypt_documents=self._doc_encryption,
                encrypt_metadata=self._meta_encryption,
            )
        else:
            self._store = VectorStore(persist_dir=self._persist_dir)

    @property
    def is_unlocked(self) -> bool:
        return self._vault is not None and self._vault.is_unlocked()

    @property
    def store(self) -> VectorStore | EncryptedVectorStore:
        if self._store is None:
            self._init_store()
        return self._store

    def unlock(self, password: str) -> None:
        """Unlock with password."""
        if self._key_manager is not None:
            self._vault = self._key_manager.get_vault(password)
        elif self._vault is not None:
            self._vault.unlock(password)
        else:
            self._vault = CryptoVault(password)
        self._init_store()

    def lock(self) -> None:
        """Lock vault and clear decrypted data from memory."""
        if self._vault is not None:
            self._vault.lock()
        self._store = None

    def create_key_file(self, password: str, path: str | Path) -> None:
        """Create an encrypted key file for future unlocks."""
        km = KeyManager(path)
        km.create_key_file(password)
        self._key_manager = km

    # --- Delegated store operations ---

    def ingest(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        return self.store.ingest(documents, metadatas=metadatas, ids=ids)

    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> list[str]:
        return self.store.ingest_file(
            file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.store.search(query, top_k=top_k, filter_dict=filter_dict)

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.store.get_document(doc_id)

    def update_document(
        self,
        doc_id: str,
        document: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.update_document(doc_id, document=document, metadata=metadata)

    def delete(self, ids: list[str]) -> None:
        self.store.delete(ids)

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.store.list_documents(limit=limit)

    def count(self) -> int:
        return self.store.count()

    # --- Database-level encryption ---

    def _get_db_file(self) -> Path:
        """Infer the ChromaDB SQLite file path."""
        base = Path(self._persist_dir or Path.home() / ".huginn" / "rag")
        # ChromaDB creates a sqlite3 file in a subdir
        db_path = base / "chroma.sqlite3"
        return db_path

    def encrypt_database_file(self, dst: str | Path | None = None) -> Path:
        """Encrypt the entire ChromaDB database file.

        This creates an encrypted copy of the database. The original remains
        in plaintext until decrypt_database_file() is called.
        """
        if not self.is_unlocked:
            raise RuntimeError("Vault must be unlocked to encrypt database.")

        db_path = self._get_db_file()
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        enc_path = Path(dst) if dst else db_path.with_suffix(".sqlite3.enc")
        self._vault.encrypt_file(db_path, enc_path)
        return enc_path

    def decrypt_database_file(self, src: str | Path, overwrite: bool = False) -> Path:
        """Decrypt a database file back to plaintext."""
        if not self.is_unlocked:
            raise RuntimeError("Vault must be unlocked to decrypt database.")

        src_path = Path(src)
        db_path = self._get_db_file()

        if db_path.exists() and not overwrite:
            raise FileExistsError(
                f"Plaintext database already exists: {db_path}. Use overwrite=True."
            )

        self._vault.decrypt_file(src_path, db_path)
        return db_path

    def backup_encrypted(self, backup_dir: str | Path) -> Path:
        """Create an encrypted backup of the entire RAG directory."""
        if not self.is_unlocked:
            raise RuntimeError("Vault must be unlocked to create backup.")

        persist = Path(self._persist_dir or Path.home() / ".huginn" / "rag")
        backup_path = (
            Path(backup_dir)
            / f"rag_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.enc"
        )
        backup_path.parent.mkdir(parents=True, exist_ok=True)

        self._vault.encrypt_directory(
            persist, backup_path, exclude=["__pycache__", ".pytest_cache"]
        )
        return backup_path

    def restore_encrypted(self, backup_path: str | Path) -> None:
        """Restore RAG directory from an encrypted backup."""
        if not self.is_unlocked:
            raise RuntimeError("Vault must be unlocked to restore backup.")

        persist = Path(self._persist_dir or Path.home() / ".huginn" / "rag")
        if persist.exists():
            # Move existing to temp for safety
            temp_backup = persist.with_suffix(f".bak_{time.time()}")
            shutil.move(str(persist), str(temp_backup))

        self._vault.decrypt_directory(backup_path, persist)
