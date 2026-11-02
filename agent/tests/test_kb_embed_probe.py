"""KB embedding 探测必须有界且不得在启动阶段联网.

chromadb 的 ``DefaultEmbeddingFunction`` 首次调用会联网下载 ONNX 权重, 且
它自带的下载器不读 ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``. 下载期间
chromadb 内部持锁 —— 实测把 TestClient 启动卡死在 ``validate_metadata``,
全量套件停在 1%. 本门锁住两条结构约束:

1. 本地未缓存 ``onnx/model.onnx`` → 启动阶段**不得**触发探测/下载, 直接降级;
2. 已缓存但加载卡住 → 有界超时后降级, 不永久阻塞.
"""

from __future__ import annotations

import threading

from huginn.knowledge import store as kb_store


def _patch(monkeypatch, *, cached: bool, ef_class=None):
    import chromadb.utils.embedding_functions as ef_mod

    monkeypatch.setattr(kb_store, "_chroma_onnx_cached", lambda: cached)
    monkeypatch.setattr(kb_store._EmbeddingModel, "_onnx_degraded", False)
    if ef_class is not None:
        monkeypatch.setattr(ef_mod, "DefaultEmbeddingFunction", ef_class)


def test_probe_degrades_when_onnx_download_hangs(monkeypatch):
    """权重已缓存但加载卡住时, 探测必须在超时内返回 None 并降级."""
    release = threading.Event()

    class _HangingEF:
        def __init__(self) -> None:
            pass

        def __call__(self, texts):
            release.wait(10)
            return [[0.0]]

    _patch(monkeypatch, cached=True, ef_class=_HangingEF)
    monkeypatch.setattr(kb_store, "_EMBED_PROBE_TIMEOUT_SECONDS", 0.2)

    try:
        assert kb_store._EmbeddingModel._probe_chroma_ef() is None
    finally:
        release.set()

    assert kb_store._EmbeddingModel._onnx_degraded is True


def test_probe_never_runs_when_weights_not_cached(monkeypatch):
    """未缓存时不得启动探测线程 (否则下载持锁会卡死启动)."""
    calls = {"n": 0}

    class _BoomEF:
        def __init__(self) -> None:
            calls["n"] += 1

        def __call__(self, texts):
            calls["n"] += 1
            return [[0.0]]

    _patch(monkeypatch, cached=False, ef_class=_BoomEF)

    assert kb_store._EmbeddingModel._probe_chroma_ef() is None
    assert calls["n"] == 0, "未缓存仍构造/调用了 embedder, 启动会被联网下载拖死"
    assert kb_store._EmbeddingModel._onnx_degraded is True


def test_probe_uses_cached_embedder_without_degrading(monkeypatch):
    """本地缓存完整且加载正常时不误降级."""

    class _FastEF:
        def __init__(self) -> None:
            pass

        def __call__(self, texts):
            return [[0.0, 1.0]]

    _patch(monkeypatch, cached=True, ef_class=_FastEF)

    ef = kb_store._EmbeddingModel._probe_chroma_ef()
    assert ef is not None
    assert kb_store._EmbeddingModel._onnx_degraded is False
