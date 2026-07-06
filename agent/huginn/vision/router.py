"""Vision routing — detect images in user messages and route to appropriate handler.

Fallback chain:
1. Vision LLM (GPT-4o/Claude/Gemini) — direct multimodal
2. Visual encoder + image index — "visual memory" for text LLM
3. Image analysis tool — structured numerical analysis (SEM/TEM/XRD)
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any

from huginn.models.registry import get_model_capabilities

# Image extensions we recognise as user-supplied image paths.
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

# Matches a filesystem-ish path ending in a known image extension.
# Loose on purpose — we'd rather over-detect and let the caller verify
# the file exists than miss a path the user pasted inline.
_IMAGE_PATH_RE = re.compile(
    r'(?:[\w./\\-]+\.(?:png|jpg|jpeg|gif|webp|bmp|tiff|tif))',
    re.IGNORECASE,
)


class VisionRoute(Enum):
    """Where an image-bearing message should be sent."""

    NATIVE_LLM = "native_llm"
    CV_TOOLS = "cv_tools"
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

    - No image at all → TEXT_ONLY (the common case, zero overhead).
    - Image + vision-capable model → NATIVE_LLM (let the LLM see it directly).
    - Image + text-only model → CV_TOOLS (fall back to encoder + analysis tool).
    """
    if not has_image:
        return VisionRoute.TEXT_ONLY
    caps = get_model_capabilities(model_name or "")
    if caps.vision:
        return VisionRoute.NATIVE_LLM
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
