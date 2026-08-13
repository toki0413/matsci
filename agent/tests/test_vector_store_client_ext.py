"""VectorStore 客户端/embedding 初始化与 PDF/加密边界补测.

覆盖 _get_client/_check_embedding/_get_collection/_compute_embeddings 缓存、
_parse_file 的 PDF(pymupdf/OCR) 分支、EncryptedVectorStore 剩余边界.
"""

from __future__ import annotations

import sys
import types

import pytest

from huginn.rag.vector_store import VectorStore, EncryptedVectorStore


# ── 注入 fake chromadb ───────────────────────────────────────────────────


def _install_fake_chromadb(monkeypatch, persistent_client=None,
                           default_ef=None, raise_import=False):
    if raise_import:
        monkeypatch.setitem(sys.modules, "chromadb", None)
        return
    chromadb = types.ModuleType("chromadb")
    chromadb.PersistentClient = persistent_client
    utils = types.ModuleType("chromadb.utils")
    ef_mod = types.ModuleType("chromadb.utils.embedding_functions")
    ef_mod.DefaultEmbeddingFunction = default_ef
    utils.embedding_functions = ef_mod
    chromadb.utils = utils
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    monkeypatch.setitem(sys.modules, "chromadb.utils", utils)
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", ef_mod)


def _make_store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "rag"))


class _FakeEF:
    def __init__(self):
        pass

    def __call__(self, texts):
        return [[0.1] * len(texts) for _ in texts]


class _FakeClient:
    def __init__(self):
        self.collection = None

    def get_or_create_collection(self, name=None, embedding_function=None,
                                 metadata=None):
        self.collection = _FakeCollection(name, embedding_function, metadata)
        return self.collection


class _FakeCollection:
    def __init__(self, name, ef, metadata):
        self.name, self.ef, self.metadata = name, ef, metadata


def test_get_client_initializes_once(monkeypatch, tmp_path):
    client = _FakeClient()
    calls = []
    def pclient(**kw):
        calls.append(kw)
        return client
    _install_fake_chromadb(monkeypatch, persistent_client=pclient)
    store = _make_store(tmp_path)
    c1 = store._get_client()
    c2 = store._get_client()
    assert c1 is client and c2 is client
    assert len(calls) == 1  # 只初始化一次


def test_check_embedding_no_model(monkeypatch, tmp_path):
    _install_fake_chromadb(monkeypatch, default_ef=None)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: False)
    assert store._check_embedding() is False
    # 第二次直接返回缓存结果
    assert store._check_embedding() is False


def test_check_embedding_success(monkeypatch, tmp_path):
    client = _FakeClient()
    _install_fake_chromadb(monkeypatch, persistent_client=lambda **kw: client,
                           default_ef=_FakeEF)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: True)
    assert store._check_embedding() is True
    assert store._embedding_fn is not None


def test_check_embedding_failure(monkeypatch, tmp_path):
    _install_fake_chromadb(monkeypatch, default_ef=_FakeEF)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: True)

    class _BoomEF(_FakeEF):
        def __call__(self, texts):
            raise RuntimeError("embed fail")

    monkeypatch.setattr(sys.modules["chromadb.utils.embedding_functions"],
                        "DefaultEmbeddingFunction", _BoomEF)
    assert store._check_embedding() is False


def test_get_collection_with_ef(monkeypatch, tmp_path):
    client = _FakeClient()
    _install_fake_chromadb(monkeypatch, persistent_client=lambda **kw: client,
                           default_ef=_FakeEF)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: True)
    col = store._get_collection()
    assert col.ef is not None
    assert col.metadata == {"hnsw:space": "cosine"}


def test_get_collection_without_ef(monkeypatch, tmp_path):
    client = _FakeClient()
    _install_fake_chromadb(monkeypatch, persistent_client=lambda **kw: client,
                           default_ef=None)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: False)
    col = store._get_collection()
    assert col.ef is None


def test_compute_embeddings_caches(monkeypatch, tmp_path):
    _install_fake_chromadb(monkeypatch, default_ef=_FakeEF)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: True)
    e1 = store._compute_embeddings(["hello"])
    e2 = store._compute_embeddings(["hello"])
    assert e1 == e2  # 命中缓存
    assert store._embed_cache


def test_compute_embeddings_no_ef(monkeypatch, tmp_path):
    _install_fake_chromadb(monkeypatch, default_ef=None)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: False)
    assert store._compute_embeddings(["x"]) is None


def test_compute_embeddings_exception(monkeypatch, tmp_path):
    _install_fake_chromadb(monkeypatch, default_ef=_FakeEF)
    store = _make_store(tmp_path)
    monkeypatch.setattr("huginn.rag.vector_store._embedding_model_cached",
                        lambda: True)

    class _BoomEF(_FakeEF):
        def __call__(self, texts):
            raise RuntimeError("boom")

    monkeypatch.setattr(sys.modules["chromadb.utils.embedding_functions"],
                        "DefaultEmbeddingFunction", _BoomEF)
    assert store._compute_embeddings(["x"]) is None


# ── _parse_file PDF 分支 ─────────────────────────────────────────────────


def test_parse_pdf_with_text(monkeypatch, tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF fake")
    fake_fitz = types.ModuleType("fitz")
    class _Doc:
        def __iter__(self):
            return iter([types.SimpleNamespace(get_text=lambda: "page text")])
    fake_fitz.open = lambda **kw: _Doc()
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    store = _make_store(tmp_path)
    assert "page text" in store._parse_file(p)


def test_parse_pdf_requires_pymupdf(monkeypatch, tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF")
    monkeypatch.setitem(sys.modules, "fitz", None)
    store = _make_store(tmp_path)
    with pytest.raises(RuntimeError, match="pymupdf"):
        store._parse_file(p)


def test_parse_pdf_falls_back_ocr(monkeypatch, tmp_path):
    p = tmp_path / "scan.pdf"
    p.write_bytes(b"%PDF")
    fake_fitz = types.ModuleType("fitz")
    class _Doc:
        def __iter__(self):
            return iter([types.SimpleNamespace(get_text=lambda: "  ")])  # 空白
    fake_fitz.open = lambda **kw: _Doc()
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    from huginn.knowledge import ocr_loader as ocr
    monkeypatch.setattr(ocr, "extract_text_with_ocr", lambda name, data: "OCR text")
    monkeypatch.setattr(ocr, "is_image_file", lambda name: False)
    store = _make_store(tmp_path)
    assert "OCR text" in store._parse_file(p)


# ── EncryptedVectorStore 边界 ────────────────────────────────────────────


class _Vault:
    def __init__(self, unlocked=True):
        self._u = unlocked

    def is_unlocked(self):
        return self._u

    def encrypt(self, t):
        import base64
        return base64.b64encode(t.encode())

    def decrypt(self, c):
        import base64
        return base64.b64decode(c).decode()


def test_encrypted_search_returns_none_doc(tmp_path):
    vault = _Vault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    evs._store.get_document = lambda doc_id: None
    assert evs.get_document("x") is None


def test_encrypted_ingest_file_end_to_end(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("some encrypted content here\n" * 20, encoding="utf-8")
    vault = _Vault()
    evs = EncryptedVectorStore(vault, persist_dir=str(tmp_path / "rag"))
    col = _MemCol()
    evs._store._get_collection = lambda: col
    ids = evs.ingest_file(str(p), chunk_size=100, chunk_overlap=10)
    assert ids
    assert col.count() > 0


class _MemCol:
    def __init__(self):
        self._data = {}

    def add(self, documents=None, metadatas=None, ids=None, embeddings=None):
        for i, _id in enumerate(ids):
            self._data[_id] = {"document": documents[i], "metadata": metadatas[i]}

    def count(self):
        return len(self._data)

    def get(self, ids=None, include=None, limit=None):
        items = list(self._data.items())
        return {
            "ids": [i for i, _ in items],
            "documents": [d["document"] for _, d in items],
            "metadatas": [d["metadata"] for _, d in items],
        }

    def query(self, *a, **kw):
        raise RuntimeError("n/a")

    def update(self, ids, documents=None, metadatas=None):
        for i, _id in enumerate(ids):
            if documents:
                self._data[_id]["document"] = documents[i]
            if metadatas:
                self._data[_id]["metadata"] = metadatas[i]

    def delete(self, ids):
        for _id in ids:
            self._data.pop(_id, None)