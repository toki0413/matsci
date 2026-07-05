"""Tests for the ImageIndex add / search / persist / degrade paths.

Backed by the deterministic fake encoder so cosine similarity is meaningful
without a real model. The graceful-degradation case forces an unavailable
encoder and checks the index still stores paths but skips similarity scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from huginn.perception import visual_encoder as ve_module
from huginn.perception.image_index import ImageIndex
from huginn.perception.visual_encoder import VisualEncoder


def test_add_and_search(
    fake_encoder,
    generate_synthetic_sem_image,
    generate_synthetic_tem_image,
    generate_synthetic_eds_image,
    generate_synthetic_particle_image,
    generate_synthetic_defect_image,
):
    paths = [
        generate_synthetic_sem_image,
        generate_synthetic_tem_image,
        generate_synthetic_eds_image,
        generate_synthetic_particle_image,
        generate_synthetic_defect_image,
    ]
    idx = ImageIndex(encoder=fake_encoder)
    for i, p in enumerate(paths):
        idx.add_image(p, metadata={"idx": i})

    assert len(idx) == 5

    # Searching with one of the indexed images should surface it at rank 1.
    results = idx.search(generate_synthetic_tem_image, top_k=3)
    assert results, "expected non-empty search results"
    assert results[0]["path"] == generate_synthetic_tem_image
    assert results[0]["similarity"] > 0.99  # self-similarity ~ 1.0
    assert results[0]["metadata"]["idx"] == 1


def test_persist_and_load(
    fake_encoder,
    tmp_path,
    generate_synthetic_sem_image,
    generate_synthetic_tem_image,
):
    store = tmp_path / "idx.json"
    idx = ImageIndex(store_path=str(store), encoder=fake_encoder)
    idx.add_image(generate_synthetic_sem_image)
    idx.add_image(generate_synthetic_tem_image)
    idx.save()

    # Fresh instance reads the JSON back off disk.
    loaded = ImageIndex(store_path=str(store), encoder=fake_encoder)
    assert len(loaded) == 2
    results = loaded.search(generate_synthetic_sem_image, top_k=2)
    assert results
    assert results[0]["path"] == generate_synthetic_sem_image
    assert results[0]["similarity"] > 0.99


def test_empty_index(fake_encoder, generate_synthetic_sem_image):
    idx = ImageIndex(encoder=fake_encoder)
    assert len(idx) == 0
    assert idx.search(generate_synthetic_sem_image) == []
    stats = idx.stats()
    assert stats["total_images"] == 0


def test_graceful_degradation(monkeypatch, tmp_path, generate_synthetic_sem_image):
    """Unavailable encoder: index stores paths, but search is a no-op."""

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("no backend")

    for name in ("_IJEPABackend", "_CLIPBackend", "_ResNetBackend"):
        monkeypatch.setattr(ve_module, name, _Boom)

    enc = VisualEncoder()
    assert enc.available is False  # the premise of this test

    idx = ImageIndex(store_path=str(tmp_path / "idx.json"), encoder=enc)
    idx.add_image(generate_synthetic_sem_image, metadata={"k": "v"})

    # Path is still recorded...
    assert len(idx) == 1
    assert idx.stats()["total_images"] == 1
    assert idx.stats()["encoder_available"] is False
    # ...but similarity scoring is skipped.
    assert idx.search(generate_synthetic_sem_image) == []
