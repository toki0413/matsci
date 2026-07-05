"""Shared fixtures for the computer-vision test suite.

Synthesises tiny (128x128) SEM / TEM / EDS / particle / defect / chart images on
the fly so ImageAnalysisTool can be exercised without real microscopy data, and
wires up a deterministic fake encoder backend so the visual-encoder and image
index paths can be tested without torch or any downloaded model weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make `huginn` importable whether pytest is launched as `python -m pytest`
# (which already prepends cwd) or a bare `pytest` (which does not). Resolves to
# <agent>/ so the package is always on sys.path.
_AGENT_ROOT = Path(__file__).resolve().parents[2]
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from PIL import Image  # noqa: E402  (import after the sys.path tweak)

from huginn.perception.visual_encoder import (  # noqa: E402
    VisualEncoder,
    _Backend,
    reset_encoder,
)
from huginn.tools.image_analysis.tool import ImageAnalysisTool  # noqa: E402
from huginn.types import ToolContext  # noqa: E402

# Small images keep the suite fast; 128 px is plenty for the stats we check.
_IMG_SIZE = 128


# ── tiny drawing helpers (numpy only, no cv2 dependency) ──────────────────


def _draw_disk(buf: np.ndarray, cy: int, cx: int, radius: int, value) -> None:
    """Fill a solid disk into ``buf`` in-place."""
    yy, xx = np.ogrid[: buf.shape[0], : buf.shape[1]]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    buf[mask] = value


def _save_png(buf: np.ndarray, path: Path) -> str:
    """Save a uint8 array as PNG and return the path string."""
    mode = "RGB" if buf.ndim == 3 else "L"
    Image.fromarray(buf.astype(np.uint8), mode=mode).save(path)
    return str(path)


# ── synthetic image fixtures ─────────────────────────────────────────────


@pytest.fixture
def generate_synthetic_sem_image(tmp_path):
    """Grayscale SEM-ish image: noisy texture plus a handful of bright spots."""
    rng = np.random.default_rng(42)
    img = rng.normal(120, 18, (_IMG_SIZE, _IMG_SIZE)).clip(0, 255)
    for _ in range(10):
        cy, cx = rng.integers(8, _IMG_SIZE - 8, 2)
        _draw_disk(img, int(cy), int(cx), int(rng.integers(2, 5)), 240)
    return _save_png(img, tmp_path / "sem.png")


@pytest.fixture
def generate_synthetic_tem_image(tmp_path):
    """Grayscale TEM-ish image: a 2D sine lattice with an 8 px period.

    The periodicity gives the FFT a clean peak at radius 128/8 = 16, which the
    lattice analyser turns into a d-spacing.
    """
    n = _IMG_SIZE
    y, x = np.indices((n, n))
    img = 128 + 70 * np.sin(2 * np.pi * x / 8) * np.sin(2 * np.pi * y / 8)
    img += np.random.default_rng(7).normal(0, 3, img.shape)  # break bit-exactness
    return _save_png(img, tmp_path / "tem.png")


@pytest.fixture
def generate_synthetic_eds_image(tmp_path):
    """RGB EDS-ish map: four solid colour quadrants for element-matching."""
    n = _IMG_SIZE
    h = n // 2
    img = np.zeros((n, n, 3), dtype=np.uint8)
    img[:h, :h] = (255, 0, 0)      # red
    img[:h, h:] = (0, 200, 0)      # green
    img[h:, :h] = (0, 0, 255)      # blue
    img[h:, h:] = (255, 215, 0)    # yellow
    return _save_png(img, tmp_path / "eds.png")


@pytest.fixture
def generate_synthetic_particle_image(tmp_path):
    """Bright-field image with dark circular particles of mixed sizes.

    Particles are dark on a bright background so the default (invert=False)
    threshold path treats them as foreground and yields a spread of sizes for
    D10 / D50 / D90.  Gaussian noise is added everywhere — without it the image
    is perfectly bimodal (two values) and Otsu's threshold lands right on the
    dark value's bin edge, causing `arr < threshold` to miss every pixel.
    """
    n = _IMG_SIZE
    img = np.full((n, n), 210, dtype=np.float64)
    rng = np.random.default_rng(11)
    for r in [4, 6, 8, 10, 5, 7, 9, 6, 5, 8]:
        cy, cx = rng.integers(r + 2, n - r - 2, 2)
        _draw_disk(img, int(cy), int(cx), r, 40)
    # Spread the values so Otsu picks a threshold between the two modes
    img += rng.normal(0, 5, img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    return _save_png(img, tmp_path / "particles.png")


@pytest.fixture
def generate_synthetic_defect_image(tmp_path):
    """Bright matrix with scattered dark blobs - targets the 'pore' detector.

    Same noise trick as the particle fixture: without it Otsu's threshold sits
    on the dark value's bin edge and the pore detector finds nothing.
    """
    n = _IMG_SIZE
    img = np.full((n, n), 205, dtype=np.float64)
    rng = np.random.default_rng(13)
    for _ in range(7):
        r = int(rng.integers(4, 9))
        cy, cx = rng.integers(r + 2, n - r - 2, 2)
        _draw_disk(img, int(cy), int(cx), r, 25)
    img += rng.normal(0, 5, img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    return _save_png(img, tmp_path / "defect.png")


@pytest.fixture
def generate_synthetic_chart_image(tmp_path):
    """A simple matplotlib line chart for the plot_extract action.

    Returns the PNG path plus the axis params the extractor needs (data ranges,
    curve colour and the axis box in pixel coords). The axis box is derived
    from the fixed axes position so the test doesn't have to guess it.
    """
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt

    left, bottom, w, h = 0.15, 0.15, 0.75, 0.75
    fig = plt.figure(figsize=(3, 3), dpi=50)  # 150x150 px
    ax = fig.add_axes([left, bottom, w, h])
    xs = np.linspace(0, 10, 120)
    ys = 0.5 + 0.4 * np.sin(xs)
    ax.plot(xs, ys, color="blue", linewidth=3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)

    path = tmp_path / "chart.png"
    fig.savefig(str(path), dpi=50)
    plt.close(fig)

    width = height = 150
    axis_box = [
        int(left * width),                  # x_left
        int((1 - (bottom + h)) * height),   # y_top (data y_max)
        int((left + w) * width),            # x_right
        int((1 - bottom) * height),         # y_bottom (data y_min)
    ]
    return {
        "path": str(path),
        "params": {
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "curve_color": "blue",
            "axis_box": axis_box,
        },
    }


# ── deterministic encoder backend (no torch, no model downloads) ───────────


class _FakeBackend(_Backend):
    """Hash-driven encoder backend for tests.

    Hashes the RGB pixel bytes to seed a fixed RNG, then draws a 16-d vector.
    Same image always maps to the same vector and different images map to
    different vectors, so cosine similarity is meaningful without any real
    model. Returned vectors are L2-normalised to mirror the real backends.
    """

    name = "fake"
    dim = 16

    def encode(self, pil_img):
        import hashlib

        arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        seed = int(hashlib.md5(arr.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


@pytest.fixture
def fake_encoder():
    """A real VisualEncoder wired to _FakeBackend (available=True, no torch).

    Built via __new__ so _build_backend never runs - no torch probe, no network.
    """
    enc = VisualEncoder.__new__(VisualEncoder)
    enc._backend = _FakeBackend()
    enc._backend_name = "fake"
    enc._init_error = None
    return enc


# ── shared tool / context fixtures ─────────────────────────────────────────


@pytest.fixture
def cv_tool():
    return ImageAnalysisTool()


@pytest.fixture
def tool_context(tmp_path):
    return ToolContext(session_id="cv-test", workspace=str(tmp_path))


@pytest.fixture(autouse=True)
def _reset_encoder_singleton():
    # Drop the module-level encoder cache between tests so state never leaks.
    reset_encoder()
    yield
    reset_encoder()
