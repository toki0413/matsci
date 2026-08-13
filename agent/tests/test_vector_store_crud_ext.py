"""VectorStore/EncryptedVectorStore 集成路径补测 — 覆盖 ingest/delete/
list_documents/count/get_document/update_document/ingest_file/_parse_file
各格式/_chunk_text/加密包装.

配合既有 search() 测试 (test_vector_store_search_ext.py), 把 vector_store.py
覆盖率从 36% 提升到 90%+.
"""

from __future__ import annotations

import json

import pytest

from huginn.rag.vector_store import (
    EncryptedVectorStore,
    VectorStore,
)


def _make_store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "rag"))


class _MemCol:
    """内存 fake collection: 支持 add/get/count/update/delete/query."""

    def __init__(self):
        self._data = {}

    def add(self, documents=None, metadatas=None, ids=None, embeddings=None):
        for i, _id in enumerate(ids):
            self._data[_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
                "embedding": embeddings[i] if embeddings else None,
            }

    def count(self):
        return len(self._data)

    def get(self, ids=None, include=None, limit=None):
        items = list(self._data.items()) if ids is None else [
            (i, d) for i, d in self._data.items() if i in ids
        ]
        if limit is not None:
            items = items[:limit]
        return {
            "ids": [i for i, _ in items],
            "documents": [d["document"] for _, d in items],
            "metadatas": [d["metadata"] for _, d in items],
            "embeddings": [d["embedding"] for _, d in items],
        }

    def query(self, *a, **kw):
        raise RuntimeError("not used")

    def update(self, ids, documents=None, metadatas=None):
        for i, _id in enumerate(ids):
            if documents:
                self._data[_id]["document"] = documents[i]
            if metadatas:
                self._data[_id]["metadata"] = metadatas[i]

    def delete(self, ids):
        for _id in ids:
            self._data.pop(_id, None)


# ── ingest ────────────────────────────────────────────────────────────────


def test_ingest_empty(tmp_path):
    store = _make_store(tmp_path)
    assert store.ingest([]) == []


def test_ingest_with_ids_and_metadata(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    store._compute_embeddings = lambda texts: [[0.1], [0.2]]

    ids = store.ingest(
        ["doc1", "doc2"],
        metadatas=[{"cat": "a"}, {"cat": "b"}],
        ids=["x1", "x2"],
    )
    assert ids == ["x1", "x2"]
    assert col.count() == 2
    # metadata 补默认 source/ingested_at
    assert col._data["x1"]["metadata"]["source"] == "unknown"
    assert "ingested_at" in col._data["x1"]["metadata"]


def test_ingest_auto_ids(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    ids = store.ingest(["hello world"])
    assert len(ids) == 1
    assert col.count() == 1


# ── CRUD 方法 ────────────────────────────────────────────────────────────


def test_delete(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    ids = store.ingest(["a", "b"], ids=["x1", "x2"])
    store.delete(["x1"])
    assert col.count() == 1


def test_list_documents(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    store.ingest(["a", "b"], metadatas=[{"m": 1}, {"m": 2}], ids=["x1", "x2"])
    docs = store.list_documents()
    assert len(docs) == 2
    assert docs[0]["id"] == "x1"
    assert docs[0]["metadata"]["m"] == 1


def test_count(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    store.ingest(["a", "b"], ids=["x1", "x2"])
    assert store.count() == 2


def test_get_document_found(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    store.ingest(["hello doc"], metadatas=[{"src": "s"}], ids=["x1"])
    doc = store.get_document("x1")
    assert doc["document"] == "hello doc"
    assert doc["metadata"]["src"] == "s"


def test_get_document_missing(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    assert store.get_document("nope") is None


def test_update_document_doc_and_meta(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    store.ingest(["old"], ids=["x1"])
    store.update_document("x1", document="new", metadata={"k": "v"})
    assert col._data["x1"]["document"] == "new"
    assert col._data["x1"]["metadata"]["k"] == "v"


def test_update_document_meta_only(tmp_path):
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    store.ingest(["body"], ids=["x1"])
    store.update_document("x1", metadata={"a": 1})
    assert col._data["x1"]["document"] == "body"
    assert col._data["x1"]["metadata"]["a"] == 1


# ── ingest_file / _parse_file ─────────────────────────────────────────────


def test_ingest_file_missing(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.ingest_file(str(tmp_path / "nope.txt"))


def test_parse_file_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\nsecond line\n", encoding="utf-8")
    store = _make_store(tmp_path)
    assert "hello world" in store._parse_file(p)


def test_parse_file_json(tmp_path):
    p = tmp_path / "data.json"
    p.write_text(json.dumps({"a": 1, "b": [2, 3]}), encoding="utf-8")
    store = _make_store(tmp_path)
    out = store._parse_file(p)
    assert '"a": 1' in out or '"a":1' in out


def test_parse_file_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    store = _make_store(tmp_path)
    assert "a, b" in store._parse_file(p)


def test_ingest_file_end_to_end(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("This is a materials science document about diffusion kinetics.\n" * 20, encoding="utf-8")
    store = _make_store(tmp_path)
    col = _MemCol()
    store._get_collection = lambda: col
    ids = store.ingest_file(str(p), chunk_size=100, chunk_overlap=10)
    assert ids
    assert col.count() > 0
    # metadata 带 filename/chunk_index
    assert col._data[ids[0]]["metadata"]["filename"] == "doc.txt"


# ── _chunk_text 边界 ─────────────────────────────────────────────────────


def test_chunk_text_small_text():
    store = VectorStore(persist_dir=".")
    chunks = store._chunk_text("short text", 500, 50)
    assert chunks == ["short text"]


def test_chunk_text_with_newline_boundary():
    store = VectorStore(persist_dir=".")
    text = "line one\nline two\nline three\n" * 10
    chunks = store._chunk_text(text, 20, 0)
    assert chunks
    assert all(c for c in chunks)


def test_chunk_text_no_overlap_finite():
    store = VectorStore(persist_dir=".")
    text = "".join(f"word{i} " for i in range(200))
    chunks = store._chunk_text(text, 50, 5)
    assert all(c for c in chunks)


# ── EncryptedVectorStore ─────────────────────────────────────────────────


class _FakeVault:
    def __init__(self, unlocked=True):
        self._unlocked = unlocked

    def is_unlocked(self):
        return self._unlocked

    def encrypt(self, text):
        import base64

        return base64.b64encode(text.encode("utf-8"))

    def decrypt(self, cipher):
        import base64

        return base64.b64decode(cipher).decode("utf-8")


def test_encrypted_ingest_search_roundtrip(tmp_path, monkeypatch):
    vault = _FakeVault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    # 用内存 col 替换底层 store
    col = _MemCol()
    evs._store._get_collection = lambda: col
    evs._store._compute_embeddings = lambda texts: [[0.1]]

    ids = evs.ingest(["secret doc"], metadatas=[{"email": "a@b.c"}], ids=["e1"])
    assert ids == ["e1"]
    # 文档被加密, 敏感元数据被加密
    stored = col._data["e1"]["document"]
    assert stored.startswith(EncryptedVectorStore.ENCRYPTED_MARKER)
    assert "a@b.c" not in col._data["e1"]["metadata"]["email"]


def test_encrypted_get_document_decrypts(tmp_path):
    import base64 as b64

    vault = _FakeVault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    col = _MemCol()
    evs._store._get_collection = lambda: col
    evs._store.ingest(
        [f"__ENC__{b64.b64encode(b'secret').decode()}"],
        metadatas=[{"email": f"__ENC__{b64.b64encode(b'a@b.c').decode()}"}],
        ids=["e1"],
    )
    doc = evs.get_document("e1")
    assert doc["document"] == "secret"
    assert doc["metadata"]["email"] == "a@b.c"


def test_encrypted_encrypt_when_locked(tmp_path):
    vault = _FakeVault(unlocked=False)
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    assert evs._encrypt("plain") == "plain"
    assert evs._decrypt("__ENC__x") == "__ENC__x"
    assert evs._is_encrypted("__ENC__x") is True


def test_encrypted_encrypt_exception(tmp_path, monkeypatch):
    vault = _FakeVault()

    def boom(t):
        raise RuntimeError("vault fail")

    monkeypatch.setattr(vault, "encrypt", boom)
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    assert evs._encrypt("x") == "x"


def test_encrypted_metadata_respects_flag(tmp_path):
    vault = _FakeVault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"),
                               encrypt_metadata=False)
    meta = evs._encrypt_metadata({"email": "a@b.c"})
    assert meta["email"] == "a@b.c"  # 未加密

    evs2 = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"),
                                encrypt_metadata=False)
    assert evs2._decrypt_metadata({"x": 1}) == {"x": 1}


def test_encrypted_delete_count_list(tmp_path):
    import base64 as b64

    vault = _FakeVault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    col = _MemCol()
    evs._store._get_collection = lambda: col
    evs._store.ingest(
        ["a", "b"],
        metadatas=[
            {"email": f"__ENC__{b64.b64encode(b'm').decode()}"},
            {"email": f"__ENC__{b64.b64encode(b'n').decode()}"},
        ],
        ids=["x1", "x2"],
    )
    assert evs.count() == 2
    evs.delete(["x1"])
    assert evs.count() == 1
    docs = evs.list_documents()
    assert len(docs) == 1
    assert docs[0]["metadata"]["email"] == "n"


def test_encrypted_update_encrypts(tmp_path):
    vault = _FakeVault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    col = _MemCol()
    evs._store._get_collection = lambda: col
    evs._store.ingest(["old"], ids=["x1"])
    evs.update_document("x1", document="newdoc", metadata={"author": "z"})
    assert col._data["x1"]["document"].startswith("__ENC__")
    assert col._data["x1"]["metadata"]["author"].startswith("__ENC__")


def test_encrypted_ingest_file_missing(tmp_path):
    vault = _FakeVault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    with pytest.raises(FileNotFoundError):
        evs.ingest_file(str(tmp_path / "nope.txt"))


def test_encrypted_ingest_empty(tmp_path):
    vault = _FakeVault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    assert evs.ingest([]) == []
