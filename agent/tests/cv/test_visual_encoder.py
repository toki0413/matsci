"""Tests for the VisualEncoder.

The encoder is exercised for real (not mocked), but its backend is swapped for
the deterministic _FakeBackend from conftest so no torch / model download is
needed. The fallback-chain test forces every real backend to fail and checks
that the encoder degrades to "no backend" instead of raising.
"""

from __future__ import annotations

import numpy as np
import pytest

from huginn.perception import visual_encoder as ve_module
from huginn.perception.visual_encoder import (
    VisualEncoder,
    get_encoder,
    reset_encoder,
)


def test_encode_returns_vector(fake_encoder, generate_synthetic_sem_image):
    vec = fake_encoder.encode_image(generate_synthetic_sem_image)

    assert vec is not None
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == fake_encoder.dim
    assert np.issubdtype(vec.dtype, np.floating)


def test_encode_l2_normalized(fake_encoder, generate_synthetic_sem_image):
    vec = fake_encoder.encode_image(generate_synthetic_sem_image)
    # Backends are contract-bound to hand back unit vectors.
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_encode_singleton(fake_encoder, generate_synthetic_sem_image):
    # Module-level cache: get_encoder() hands out the same instance.
    first = get_encoder()
    second = get_encoder()
    assert first is second

    # reset_encoder() drops the cache, so the next call builds a new one.
    reset_encoder()
    third = get_encoder()
    assert third is not first

    # And the fake backend is deterministic - same image -> same vector.
    v1 = fake_encoder.encode_image(generate_synthetic_sem_image)
    v2 = fake_encoder.encode_image(generate_synthetic_sem_image)
    np.testing.assert_allclose(v1, v2)


def test_fallback_chain(monkeypatch, generate_synthetic_sem_image):
    """Every backend fails -> encoder reports unavailable, never raises."""

    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("simulated torch missing")

    for name in ("_IJEPABackend", "_CLIPBackend", "_ResNetBackend"):
        monkeypatch.setattr(ve_module, name, _Boom)

    enc = VisualEncoder()

    assert enc.available is False
    assert enc.backend_name is None
    assert enc.dim == 0
    assert isinstance(enc.init_error, str) and enc.init_error
    # The all-important contract: encode degrades to None instead of blowing up.
    assert enc.encode_image(generate_synthetic_sem_image) is None
