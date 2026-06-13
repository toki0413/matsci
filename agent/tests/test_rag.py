"""Tests for RAG and encrypted RAG modules."""

import tempfile
from pathlib import Path

import pytest

from matsci_agent.crypto import CryptoVault
from matsci_agent.rag.vector_store import VectorStore, EncryptedVectorStore
from matsci_agent.rag.encrypted_rag import EncryptedRAGManager


class TestVectorStore:
    def test_ingest_and_search(self):
        # Use a persistent test dir to avoid Windows file lock issues with ChromaDB
        tmp = Path(tempfile.gettempdir()) / "matsci_test_rag"
        tmp.mkdir(exist_ok=True)
        store = VectorStore(persist_dir=str(tmp), collection_name="test")
        # Clear any previous data
        for doc in store.list_documents(limit=1000):
            store.delete([doc["id"]])

        ids = store.ingest(
            ["Silicon has a diamond cubic structure", "Band gap of Si is 1.1 eV"],
            metadatas=[{"source": "wiki"}, {"source": "paper"}],
        )
        assert len(ids) == 2
        assert store.count() == 2

        # Keyword fallback search
        results = store.search("silicon structure", top_k=2)
        assert len(results) > 0
        assert "silicon" in results[0]["document"].lower()

    def test_ingest_file_and_chunking(self):
        tmp = Path(tempfile.gettempdir()) / "matsci_test_rag_file"
        tmp.mkdir(exist_ok=True)
        txt = tmp / "doc.txt"
        txt.write_text("Chapter 1\n" + "x" * 1000 + "\nChapter 2\n" + "y" * 1000)
        store = VectorStore(persist_dir=str(tmp) + "_db", collection_name="test_file")
        ids = store.ingest_file(str(txt), chunk_size=200, chunk_overlap=20)
        assert len(ids) >= 2

    def test_get_update_delete_document(self):
        tmp = Path(tempfile.gettempdir()) / "matsci_test_rag_crud"
        tmp.mkdir(exist_ok=True)
        store = VectorStore(persist_dir=str(tmp), collection_name="test_crud")
        ids = store.ingest(["original text"], metadatas=[{"tag": "v1"}])
        doc_id = ids[0]

        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["document"] == "original text"

        store.update_document(doc_id, document="updated text", metadata={"tag": "v2"})
        doc = store.get_document(doc_id)
        assert doc["document"] == "updated text"

        store.delete([doc_id])
        assert store.count() == 0


class TestEncryptedVectorStore:
    def test_encrypted_ingest_and_search(self):
        vault = CryptoVault("rag_password")
        tmp = Path(tempfile.gettempdir()) / "matsci_test_enc_rag"
        tmp.mkdir(exist_ok=True)
        store = EncryptedVectorStore(
            vault=vault,
            persist_dir=str(tmp),
            collection_name="test_enc",
            encrypt_documents=True,
            encrypt_metadata=True,
        )
        ids = store.ingest(
            ["Secret formula for superalloy"],
            metadatas=[{"source": "lab_notebook", "author": "Dr. X"}],
        )
        assert len(ids) == 1

        # Search should return decrypted content
        results = store.search("superalloy", top_k=1)
        assert len(results) == 1
        assert "Secret formula" in results[0]["document"]
        assert results[0]["metadata"]["source"] == "lab_notebook"

    def test_encrypted_vs_plaintext_store(self):
        vault = CryptoVault("test")
        tmp = Path(tempfile.gettempdir()) / "matsci_test_enc_plain"
        tmp.mkdir(exist_ok=True)
        enc_store = EncryptedVectorStore(vault=vault, persist_dir=str(tmp / "enc"))
        plain_store = VectorStore(persist_dir=str(tmp / "plain"))

        enc_store.ingest(["secret"])
        plain_store.ingest(["public"])

        # Encrypted store's internal documents should be encrypted markers
        raw_doc = enc_store._store.get_document(enc_store._store.list_documents(limit=1)[0]["id"])
        assert raw_doc["document"].startswith("__ENC__")


class TestEncryptedRAGManager:
    def test_manager_unlock_lock(self):
        tmp = Path(tempfile.gettempdir()) / "matsci_test_mgr"
        tmp.mkdir(exist_ok=True)
        mgr = EncryptedRAGManager(password="mgr_pass", persist_dir=str(tmp))
        assert mgr.is_unlocked
        mgr.ingest(["test document"])
        assert mgr.count() == 1

        mgr.lock()
        assert not mgr.is_unlocked

    def test_backup_and_restore(self):
        vault = CryptoVault("backup_pass")
        tmp = Path(tempfile.gettempdir()) / "matsci_test_backup"
        tmp.mkdir(exist_ok=True)
        persist = tmp / "rag"
        backup_dir = tmp / "backups"
        mgr = EncryptedRAGManager(password="backup_pass", persist_dir=str(persist))
        mgr.ingest(["backup me"])

        backup_path = mgr.backup_encrypted(str(backup_dir))
        assert backup_path.exists()

        # Restore to new location
        restore_dir = tmp / "restored_rag"
        mgr2 = EncryptedRAGManager(password="backup_pass", persist_dir=str(restore_dir))
        mgr2.restore_encrypted(backup_path)
        assert mgr2.count() == 1
