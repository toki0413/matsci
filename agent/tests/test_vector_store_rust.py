"""VectorStore 的 Rust `top_k` 桥接测试.

`_rust_top_k` 优先调用编译扩展 `huginn_ext.top_k`, 失败/不可用降级返回 None,
由调用方走 Python 打分. chromadb 未安装也能测: __init__ 不触 chromadb,
`_rust_top_k` 本身自包含.
"""

from __future__ import annotations

import sys
import types

from huginn.rag.vector_store import VectorStore


def _make_store(tmp_path):
    return VectorStore(persist_dir=str(tmp_path / "rag"))


def _install_fake_huginn_ext(monkeypatch, top_k=None, raise_import=False):
    if raise_import:
        monkeypatch.setitem(sys.modules, "huginn_ext", None)
        return
    ext = types.ModuleType("huginn_ext")
    if top_k is not None:
        ext.top_k = top_k
    monkeypatch.setitem(sys.modules, "huginn_ext", ext)


def test_rust_top_k_import_missing_returns_none(monkeypatch, tmp_path):
    store = _make_store(tmp_path)
    _install_fake_huginn_ext(monkeypatch, raise_import=True)
    assert store._rust_top_k([1.0], [[1.0]], 1) is None


def test_rust_top_k_empty_embeddings_returns_none(monkeypatch, tmp_path):
    store = _make_store(tmp_path)

    def top_k(q, e, k):
        raise AssertionError("should not be called")

    _install_fake_huginn_ext(monkeypatch, top_k=top_k)
    assert store._rust_top_k([1.0], [], 3) is None


def test_rust_top_k_success(monkeypatch, tmp_path):
    store = _make_store(tmp_path)
    expected = [(0, 0.9), (1, 0.5)]

    def top_k(q, e, k):
        return expected

    _install_fake_huginn_ext(monkeypatch, top_k=top_k)
    assert store._rust_top_k([1.0], [[1.0], [0.5]], 2) is expected


def test_rust_top_k_raises_returns_none(monkeypatch, tmp_path):
    store = _make_store(tmp_path)

    def top_k(q, e, k):
        raise RuntimeError("rust top_k crash")

    _install_fake_huginn_ext(monkeypatch, top_k=top_k)
    assert store._rust_top_k([1.0], [[1.0], [0.5]], 2) is None
