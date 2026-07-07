"""Vision routing — detect images in user messages and route to appropriate handler.

Fallback chain:
1. Vision LLM (GPT-4o/Claude/Gemini) — direct multimodal
2. Visual encoder + image index — "visual memory" for text LLM
3. Image analysis tool — structured numerical analysis (SEM/TEM/XRD)

When a vision LLM is available, BOTH paths fire: the LLM gets the raw image
for semantic understanding, and CV pre-analysis runs in parallel to provide
quantitative hints (image type, rough metrics) injected into the text prompt.
This avoids the old either/or split where vision LLMs never got structured
measurements and CV tools never got semantic context.
"""

from __future__ import annotations

import base64
import mimetypes
import re
from enum import Enum
from pathlib import Path
from typing import Any

from huginn.models.registry import get_model_capabilities

# Image extensions we recognise as user-supplied image paths.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

# Matches a filesystem-ish path ending in a known image extension.
_IMAGE_PATH_RE = re.compile(
    r'(?:[\w./\\-]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif))',
    re.IGNORECASE,
)


class VisionRoute(Enum):
    """Where an image-bearing message should be sent."""

    NATIVE_LLM = "native_llm"
    CV_TOOLS = "cv_tools"
    BOTH = "both"
    TEXT_ONLY = "text_only"


def detect_image_in_message(message: str | dict | list) -> bool:
    """Return True if *message* references an image path or carries raw bytes.

    Handles three shapes the agent might pass in:
    - plain string: looks for a path-like token ending in an image ext
    - dict with ``image_path`` / ``image_bytes`` keys
    - list of content blocks (LangChain multimodal) containing an image_url type
    """
    if message is None:
        return False
    if isinstance(message, str):
        return bool(_IMAGE_PATH_RE.search(message))
    if isinstance(message, dict):
        return bool(message.get("image_path") or message.get("image_bytes"))
    if isinstance(message, list):
        for block in message:
            if isinstance(block, dict) and block.get("type") in ("image_url", "image"):
                return True
            if isinstance(block, dict) and block.get("image_path"):
                return True
    return False


def route_vision(model_name: str | None, has_image: bool) -> VisionRoute:
    """Decide how to handle an image based on the active model's capabilities.

    - No image at all -> TEXT_ONLY (the common case, zero overhead).
    - Image + vision-capable model -> BOTH (LLM sees image + CV pre-analysis
      runs in parallel to inject quantitative hints).
    - Image + text-only model -> CV_TOOLS (fall back to encoder + analysis tool).
    """
    if not has_image:
        return VisionRoute.TEXT_ONLY
    caps = get_model_capabilities(model_name or "")
    if caps.vision:
        return VisionRoute.BOTH
    return VisionRoute.CV_TOOLS


def _guess_mime(path: str | Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/png"


def build_multimodal_content(message: str, image_path: str | Path | bytes) -> list[dict]:
    """Build a LangChain/OpenAI multimodal content list from text + image.

    Returns a list of content blocks:
    ``[{"type": "text", ...}, {"type": "image_url", ...}]``

    If the image can't be read (missing file, bad bytes) we degrade to
    text-only so the LLM still gets the user's question.
    """
    blocks: list[dict] = [{"type": "text", "text": message}]

    if isinstance(image_path, (bytes, bytearray)):
        b64 = base64.b64encode(image_path).decode("ascii")
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
        return blocks

    p = Path(image_path)
    if not p.is_file():
        return blocks

    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    mime = _guess_mime(p)
    blocks.append({
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    })
    return blocks


def _cv_pre_analyze(image_path: str | Path | bytes) -> str:
    """Fast CV pre-analysis: image type guess + basic stats.

    Runs numpy-only feature extraction (~50ms, no LLM cost) to give the
    vision LLM quantitative hints alongside the raw image. Falls back
    gracefully if the image can't be loaded or numpy is unavailable.
    """
    if isinstance(image_path, (bytes, bytearray)):
        return "[CV pre-analysis skipped: raw bytes, no path]"

    p = Path(image_path)
    if not p.is_file():
        return f"[CV pre-analysis skipped: file not found: {image_path}]"

    try:
        import numpy as np
        from huginn.tools.image_analysis._utils import load_gray
    except ImportError:
        return "[CV pre-analysis unavailable: numpy/Pillow not installed]"

    try:
        arr = load_gray(str(p))
        mean_i = float(arr.mean())
        std_i = float(arr.std())
        p5, p50, p95 = np.percentile(arr, [5, 50, 95])

        # Edge density — rough indicator of "busy" vs "smooth" image
        edges = 0
        try:
            from scipy.ndimage import sobel
            sx = sobel(arr, axis=0)
            sy = sobel(arr, axis=1)
            edges = int((np.hypot(sx, sy) > 50).sum())
            total = arr.shape[0] * arr.shape[1]
            edge_density = edges / total if total > 0 else 0.0
        except Exception:
            edge_density = -1.0

        # Guess image type from extension + stats
        ext = p.suffix.lower()
        name_hint = p.stem.lower()
        img_type = "unknown"
        if any(kw in name_hint for kw in ("sem", "sem_photo", "fesem")):
            img_type = "SEM"
        elif any(kw in name_hint for kw in ("tem", "hrtem", "stem")):
            img_type = "TEM"
        elif any(kw in name_hint for kw in ("xrd", "diffraction")):
            img_type = "XRD_plot"
        elif any(kw in name_hint for kw in ("eds", "mapping")):
            img_type = "EDS_map"
        elif ext in (".csv", ".txt"):
            img_type = "tabular_data"
        elif edge_density > 0 and edge_density < 0.05:
            img_type = "likely_microscopy_smooth"
        elif edge_density > 0.15:
            img_type = "likely_microscopy_busy_or_plot"

        lines = [
            f"[CV pre-analysis] image_type_guess={img_type}",
            f"  shape={arr.shape[1]}x{arr.shape[0]}, mean={mean_i:.1f}, "
            f"std={std_i:.1f}, percentiles(5/50/95)={p5:.0f}/{p50:.0f}/{p95:.0f}",
        ]
        if edge_density >= 0:
            lines.append(f"  edge_density={edge_density:.4f} "
                         f"({'busy' if edge_density > 0.1 else 'smooth'})")
        lines.append(
            "  Use image_analysis_tool for detailed SEM/TEM/EDS/particle metrics."
        )
        return "\n".join(lines)
    except Exception as exc:
        return f"[CV pre-analysis failed: {exc}]"


def build_cv_context(
    image_path: str | Path | bytes,
    visual_encoder: Any | None = None,
    image_index: Any | None = None,
) -> str:
    """Build a text description of an image for text-only LLMs.

    Tries two things in order, both best-effort:
    1. Encode the image with *visual_encoder* (if available) and search
       *image_index* for similar indexed images — gives the LLM a
       "visual memory" hit: "this looks like SEM image X you saw before".
    2. Falls back to a bare path annotation so the LLM at least knows
       an image was attached and can call image_analysis_tool itself.
    """
    parts: list[str] = []

    # ── CV pre-analysis (fast, always runs) ──
    cv_hints = _cv_pre_analyze(image_path)
    if cv_hints:
        parts.append(cv_hints)

    # ── visual memory: similar-image search ──
    if visual_encoder is not None and image_index is not None:
        try:
            results = image_index.search(image_path, top_k=5)
        except Exception:
            results = []

        if results:
            paths = [r.get("path", "?") for r in results]
            parts.append(f"Found {len(results)} similar indexed images: {', '.join(paths)}")
            best = results[0]
            meta = best.get("metadata", {})
            sim = best.get("similarity", 0.0)
            meta_str = ", ".join(f"{k}={v}" for k, v in meta.items()) if meta else "no metadata"
            parts.append(
                f"Closest match: {best.get('path', '?')} "
                f"(similarity={sim:.3f}, {meta_str})"
            )
        else:
            parts.append("No similar indexed images found in visual memory.")

    # ── encoder health check ──
    if visual_encoder is not None:
        avail = getattr(visual_encoder, "available", False)
        backend = getattr(visual_encoder, "backend_name", None)
        if avail:
            parts.append(f"Visual encoder active (backend={backend}).")
        else:
            parts.append("Visual encoder unavailable — image_analysis_tool recommended.")

    # ── always: tell the LLM an image was attached ──
    label = "<bytes>" if isinstance(image_path, (bytes, bytearray)) else str(image_path)
    parts.append(
        f"User attached an image ({label}). "
        "If you need structured analysis (SEM/TEM/XRD), call image_analysis_tool."
    )

    return "\n".join(parts)


class VisionRouter:
    """Stateful wrapper that an agent holds to route images per-turn.

    The encoder and image index are optional — if either is missing the
    router degrades gracefully (NATIVE_LLM still works, CV_TOOLS falls
    back to a path-annotation only).
    """

    def __init__(
        self,
        visual_encoder: Any | None = None,
        image_index: Any | None = None,
    ) -> None:
        self.visual_encoder = visual_encoder
        self.image_index = image_index

    def route(self, model_name: str | None, message: str | dict | list) -> VisionRoute:
        has_image = detect_image_in_message(message)
        return route_vision(model_name, has_image)

    def build_context(
        self, image_path: str | Path | bytes
    ) -> str:
        return build_cv_context(image_path, self.visual_encoder, self.image_index)

    def build_content(
        self, message: str, image_path: str | Path | bytes
    ) -> list[dict]:
        return build_multimodal_content(message, image_path)

    def coordinate(
        self,
        message: str,
        image_path: str | Path | bytes,
        model_name: str | None = None,
    ) -> tuple[list[dict], str]:
        """Run BOTH paths: multimodal content for LLM + CV pre-analysis text.

        Returns (multimodal_content, cv_hints_text). The caller should inject
        cv_hints_text as a SystemMessage alongside the multimodal content so
        the vision LLM sees both the raw image and quantitative hints.
        """
        content = self.build_content(message, image_path)
        cv_hints = _cv_pre_analyze(image_path)
        return content, cv_hints
