"""HTTP tests for the viewer3d + visual perception endpoints.

The real route modules are loaded standalone (importlib from file) so we skip
``huginn.routes.__init__``'s eager import of every router - keeps the surface
tiny and avoids pulling the whole agent stack. ``server_core`` is stubbed for
the visual router's import; the handlers themselves are the real ones, wired
to the deterministic fake encoder / an in-memory ImageIndex.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from huginn.perception.image_index import ImageIndex

_AGENT_ROOT = Path(__file__).resolve().parents[2]
_VISUAL_PATH = _AGENT_ROOT / "huginn" / "routes" / "visual.py"
_VIEWER3D_PATH = _AGENT_ROOT / "huginn" / "routes" / "viewer3d.py"


def _load_standalone(name: str, path: Path):
    """Import a .py file as an isolated module (no parent package __init__)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def visual_app(monkeypatch, tmp_path, fake_encoder):
    # visual.py does `from huginn.server_core import ...` at import time. We
    # don't want the real server_core (it drags in the whole agent stack), so
    # park a stub there just long enough for the import to resolve.
    stub = types.ModuleType("huginn.server_core")
    stub.get_image_index = lambda: None
    stub.get_visual_encoder = lambda: None
    saved = sys.modules.get("huginn.server_core")
    sys.modules["huginn.server_core"] = stub
    try:
        vis = _load_standalone("huginn.routes._visual_ep_test", _VISUAL_PATH)
        v3d = _load_standalone("huginn.routes._viewer3d_ep_test", _VIEWER3D_PATH)
    finally:
        if saved is not None:
            sys.modules["huginn.server_core"] = saved
        else:
            sys.modules.pop("huginn.server_core", None)

    # Wire the real handlers to test doubles via their module globals.
    idx = ImageIndex(store_path=str(tmp_path / "vis_index.json"), encoder=fake_encoder)
    monkeypatch.setattr(vis, "get_image_index", lambda: idx)
    monkeypatch.setattr(vis, "get_visual_encoder", lambda: fake_encoder)

    app = FastAPI()
    app.include_router(v3d.router)
    app.include_router(vis.router)
    return app, idx


def test_get_elements(visual_app):
    app, _ = visual_app
    client = TestClient(app)

    r = client.get("/viewer3d/elements")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["elements"], list) and len(body["elements"]) > 0
    first = body["elements"][0]
    assert {"symbol", "color", "covalent_radius"} <= set(first)
    assert "default_color" in body
    assert "default_radius" in body
    assert "bond_tolerance" in body


def test_visual_encode(visual_app, generate_synthetic_sem_image):
    app, _ = visual_app
    client = TestClient(app)

    with open(generate_synthetic_sem_image, "rb") as fh:
        r = client.post(
            "/visual/encode",
            files={"file": ("sem.png", fh, "image/png")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["backend"] == "fake"
    assert body["dim"] == 16
    assert isinstance(body["vector"], list)
    assert len(body["vector"]) == 16


def test_visual_search(
    visual_app,
    generate_synthetic_sem_image,
    generate_synthetic_tem_image,
):
    app, idx = visual_app
    # Seed the index with two known images so search has something to rank.
    idx.add_image(generate_synthetic_sem_image, metadata={"name": "sem"})
    idx.add_image(generate_synthetic_tem_image, metadata={"name": "tem"})

    client = TestClient(app)
    with open(generate_synthetic_sem_image, "rb") as fh:
        r = client.post(
            "/visual/search",
            files={"file": ("sem.png", fh, "image/png")},
            data={"top_k": "5"},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["count"] >= 1
    top = body["results"][0]
    assert top["path"] == generate_synthetic_sem_image
    assert top["similarity"] > 0.99
