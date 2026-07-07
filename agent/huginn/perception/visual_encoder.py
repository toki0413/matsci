"""Visual encoder using I-JEPA frozen representations.

Provides image embeddings for materials science images (SEM/TEM/XRD/etc.)
that can be used for similarity search, clustering, and retrieval —
giving text-only LLMs a form of visual perception.

The encoder is intentionally dependency-light: torch / transformers / timm /
PIL are all imported lazily inside the backend loaders so that importing
this module never fails on a machine without the ML stack installed. When no
backend can be built, ``encode_image`` simply returns ``None`` and callers
are expected to degrade gracefully (e.g. the image index keeps the entry but
skips it for similarity search).
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import logging
logger = logging.getLogger(__name__)


# ImageNet stats — works fine for ResNet, CLIP and I-JEPA pretraining.
# Keeping them as module constants avoids re-allocating on every call.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ── image loading ───────────────────────────────────────────────────


def _load_pil_image(image: str | Path | bytes) -> Any:
    """Load a PIL image from a path or raw bytes.

    Returns the image converted to RGB. Raises if PIL isn't installed —
    callers catch and treat it as "no backend available".
    """
    from PIL import Image  # lazy import

    if isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    else:
        img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ── backend interface ───────────────────────────────────────────────


class _Backend:
    """Minimal interface every encoder backend implements."""

    name: str = "base"
    dim: int = 0

    def encode(self, pil_img: Any) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        """Release heavyweight resources (model + tensors)."""
        # default no-op; backends override when needed
        return None


# ── I-JEPA backend ──────────────────────────────────────────────────
#
# I-JEPA (github.com/facebookresearch/ijepa) ships a ViT target/context
# encoder whose state_dict is a plain torch checkpoint. We rebuild the
# architecture with timm and load the weights with strict=False so the
# encoder is usable as a frozen feature extractor without depending on
# the upstream repo at runtime.


class _IJEPABackend(_Backend):
    name = "ijepa"

    def __init__(self) -> None:
        import torch  # noqa: F401
        import timm

        ckpt_path = os.environ.get("IJEPA_CHECKPOINT", "").strip()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise RuntimeError("I-JEPA checkpoint not found (set IJEPA_CHECKPOINT)")

        arch = os.environ.get("IJEPA_ARCH", "vit_huge_patch16_224")
        img_size = int(os.environ.get("IJEPA_IMG_SIZE", "224"))
        device = os.environ.get("IJEPA_DEVICE", "cuda" if _torch_cuda_available() else "cpu")

        # num_classes=0 drops the classification head — we want raw features.
        self._model = timm.create_model(
            arch,
            pretrained=False,
            num_classes=0,
            img_size=img_size,
        )
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, dict) and "model" in state:
            state = state["model"]
        # I-JEPA prefixes encoder weights with "module." / "target_encoder."
        # in some checkpoints. Strip common prefixes so timm can match.
        cleaned: dict[str, Any] = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "target_encoder.", "encoder."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        self._model.load_state_dict(cleaned, strict=False)
        self._model.eval()
        self._device = torch.device(device)
        self._model.to(self._device)

        self._arch = arch
        self._img_size = img_size
        # feature dim is only known after building the model — read it back.
        self.dim = int(self._model.num_features)

        # Precompute the preprocessing transform once.
        from torchvision import transforms

        self._tfm = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def encode(self, pil_img: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = self._tfm(pil_img).unsqueeze(0).to(self._device)
            feats = self._model.forward_head(self._model.forward_features(x), pre_logits=True)
            vec = feats.squeeze(0).float().cpu().numpy()
        return _l2_normalize(vec)

    def close(self) -> None:
        try:
            del self._model
        except Exception:
            logger.debug("close failed", exc_info=True)
        self._model = None  # type: ignore[assignment]


# ── CLIP backend (transformers) ─────────────────────────────────────


class _CLIPBackend(_Backend):
    name = "clip"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = os.environ.get("CLIP_DEVICE", "cuda" if _torch_cuda_available() else "cpu")
        self._device = torch.device(device)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self._device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        # CLIP vision feature dim — base patch32 is 512.
        self.dim = int(self._model.config.projection_dim)

    def encode(self, pil_img: Any) -> np.ndarray:
        import torch

        inputs = self._processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(pixel_values=pixel_values)
            vec = feats.squeeze(0).float().cpu().numpy()
        return _l2_normalize(vec)

    def close(self) -> None:
        try:
            del self._model
        except Exception:
            logger.debug("close failed", exc_info=True)
        self._model = None  # type: ignore[assignment]


# ── ResNet50 backend (torchvision) ──────────────────────────────────


class _ResNetBackend(_Backend):
    name = "resnet50"

    def __init__(self) -> None:
        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        # IMAGENET1K_V2 weights give a better representation than V1 and
        # are still just a download; fall back to V1 if V2 is unavailable.
        weights = None
        for w in ("IMAGENET1K_V2", "IMAGENET1K_V1", "DEFAULT"):
            try:
                weights = getattr(models.ResNet50_Weights, w)
                break
            except AttributeError:
                continue
        if weights is None:
            raise RuntimeError("no ResNet50 weights available")

        resnet = models.resnet50(weights=weights)
        # Drop the 1000-way classifier — we want the 2048-d pooled features.
        resnet.fc = nn.Identity()
        resnet.eval()

        device = os.environ.get("RESNET_DEVICE", "cuda" if _torch_cuda_available() else "cpu")
        self._device = torch.device(device)
        resnet.to(self._device)
        self._model = resnet
        self.dim = 2048  # avgpool output width, fixed for ResNet50

        self._tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def encode(self, pil_img: Any) -> np.ndarray:
        import torch

        with torch.no_grad():
            x = self._tfm(pil_img).unsqueeze(0).to(self._device)
            vec = self._model(x).squeeze(0).float().cpu().numpy()
        return _l2_normalize(vec)

    def close(self) -> None:
        try:
            del self._model
        except Exception:
            logger.debug("close failed", exc_info=True)
        self._model = None  # type: ignore[assignment]


# ── helpers ─────────────────────────────────────────────────────────


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Return a unit-norm copy of ``vec`` so cosine similarity == dot product."""
    v = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v


# ── public API ──────────────────────────────────────────────────────


class VisualEncoder:
    """Frozen-image feature encoder with a cascading backend fallback.

    Backend priority (first that builds wins):
      1. I-JEPA   — needs IJEPA_CHECKPOINT + torch + timm
      2. CLIP     — needs transformers + torch
      3. ResNet50 — needs torchvision + torch
      4. None     — every backend failed; encode_image returns None

    A single shared instance is normally obtained via :func:`get_encoder`
    so the model is only loaded once per process.
    """

    def __init__(self) -> None:
        self._backend: _Backend | None = None
        self._backend_name: str | None = None
        self._init_error: str | None = None
        self._build_backend()

    def _build_backend(self) -> None:
        """Try each backend in order, keep the first that constructs."""
        attempts: list[tuple[str, type[_Backend]]] = [
            ("ijepa", _IJEPABackend),
            ("clip", _CLIPBackend),
            ("resnet50", _ResNetBackend),
        ]
        errors: list[str] = []
        for label, cls in attempts:
            try:
                self._backend = cls()
                self._backend_name = label
                return
            except Exception as exc:  # noqa: BLE001 - intentional broad fallback
                errors.append(f"{label}: {exc}")
        # Nothing loaded — record why so callers can surface a reason.
        self._backend = None
        self._backend_name = None
        self._init_error = "; ".join(errors) if errors else "no backends attempted"

    # ── properties ──

    @property
    def available(self) -> bool:
        return self._backend is not None

    @property
    def backend_name(self) -> str | None:
        return self._backend_name

    @property
    def dim(self) -> int:
        return self._backend.dim if self._backend is not None else 0

    @property
    def init_error(self) -> str | None:
        return self._init_error

    # ── encoding ──

    def encode_image(self, image: str | Path | bytes) -> np.ndarray | None:
        """Encode a single image into a fixed-dim L2-normalized vector.

        ``image`` may be a filesystem path or raw bytes. Returns ``None``
        when no backend is available or the image cannot be decoded.
        """
        if self._backend is None:
            return None
        try:
            pil_img = _load_pil_image(image)
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"image load failed: {exc}"
            return None
        try:
            return self._backend.encode(pil_img)
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"encode failed: {exc}"
            return None

    def encode_batch(self, image_paths: Iterable[str | Path]) -> list[np.ndarray | None]:
        """Encode several images. ``None`` entries are kept in order so the
        result aligns 1:1 with the input paths."""
        results: list[np.ndarray | None] = []
        for path in image_paths:
            results.append(self.encode_image(path))
        return results

    # ── similarity ──

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity. Vectors are assumed pre-normalized (which
        ``encode_image`` guarantees), but we guard against un-normalized
        inputs by dividing by the norms."""
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))


# ── module-level singleton cache ────────────────────────────────────
#
# Models are expensive to load (hundreds of MB). We cache a single
# process-wide instance and hand it back on every get_encoder() call.
# Tests that need isolation can call reset_encoder() to drop the cache.

_ENCODER: VisualEncoder | None = None
_ENCODER_LOCK = threading.Lock()


def get_encoder() -> VisualEncoder | None:
    """Return the shared :class:`VisualEncoder`, building it on first use.

    Returns ``None`` only if construction itself blew up so badly that we
    can't even produce a degraded encoder object — normally you get a
    VisualEncoder whose ``available`` flag is False.
    """
    global _ENCODER
    if _ENCODER is None:
        with _ENCODER_LOCK:
            if _ENCODER is None:
                try:
                    _ENCODER = VisualEncoder()
                except Exception:  # noqa: BLE001 - never let importers crash
                    _ENCODER = None
    return _ENCODER


def reset_encoder() -> None:
    """Drop the cached encoder (mainly for tests)."""
    global _ENCODER
    enc = _ENCODER
    _ENCODER = None
    if enc is not None and enc._backend is not None:  # noqa: SLF001
        try:
            enc._backend.close()  # noqa: SLF001
        except Exception:
            logger.debug("close failed", exc_info=True)
