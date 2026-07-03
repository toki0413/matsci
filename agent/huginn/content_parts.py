"""Structured content parts for multimodal messages — AstrBot inspired.

AstrBot uses a ContentPart base class with __init_subclass__ auto-registration.
Subclasses (TextPart, ImageURLPart, AudioURLPart) are automatically registered
and can be deserialized by type. This module brings the same to Huginn.

Key difference from AstrBot: we add StructurePart and PlotPart for
materials science specific content (crystal structures, matplotlib plots).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class ContentPart:
    """Base class for a piece of message content.
    
    Subclasses register themselves via __init_subclass__ and can be
    deserialized from a dict by 'type' field.
    """
    type: str = "text"
    _no_save: bool = False  # If True, don't persist (temporary content like inline images)
    
    _registry: ClassVar[dict[str, type["ContentPart"]]] = {}
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Register by the type field's default value
        for base in cls.__mro__:
            if hasattr(base, "__annotations__") and "type" in base.__annotations__:
                # Get default from class dict or annotations
                type_val = cls.__dict__.get("type", None)
                if type_val:
                    ContentPart._registry[type_val] = cls
                    break
    
    def mark_as_temp(self) -> "ContentPart":
        """Mark this content as temporary (not persisted in history)."""
        self._no_save = True
        return self
    
    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI message content format."""
        return {"type": self.type}
    
    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic message content format."""
        return {"type": self.type}
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentPart":
        """Deserialize a content part from a dict (dispatches by type)."""
        type_val = data.get("type", "text")
        target_cls = cls._registry.get(type_val, TextPart)
        return target_cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "ContentPart":
        raise NotImplementedError


@dataclass
class TextPart(ContentPart):
    """A text content part."""
    type: str = "text"
    text: str = ""
    
    def to_openai_format(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}
    
    def to_anthropic_format(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}
    
    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "TextPart":
        return cls(text=data.get("text", ""))


@dataclass
class ImageURLPart(ContentPart):
    """An image content part (URL or base64 data URI)."""
    type: str = "image_url"
    image_url: str = ""  # URL or data:image/png;base64,...
    detail: str = "auto"  # "auto" | "low" | "high"
    
    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "image_url",
            "image_url": {"url": self.image_url, "detail": self.detail},
        }
    
    def to_anthropic_format(self) -> dict[str, Any]:
        # Anthropic uses base64 source, not URL
        if self.image_url.startswith("data:"):
            # Parse data URI
            header, _, b64data = self.image_url.partition(",")
            media_type = "image/png"
            if "image/jpeg" in header:
                media_type = "image/jpeg"
            elif "image/webp" in header:
                media_type = "image/webp"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64data,
                },
            }
        return {"type": "image", "source": {"type": "url", "url": self.image_url}}
    
    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "ImageURLPart":
        url_data = data.get("image_url", {})
        if isinstance(url_data, dict):
            return cls(image_url=url_data.get("url", ""), detail=url_data.get("detail", "auto"))
        return cls(image_url=str(url_data))
    
    @classmethod
    def from_file(cls, path: str | Path, detail: str = "auto") -> "ImageURLPart":
        """Create an ImageURLPart from a local image file (base64 encoded)."""
        path = Path(path)
        ext = path.suffix.lower().lstrip(".")
        media_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
        media_type = media_map.get(ext, "image/png")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return cls(image_url=f"data:{media_type};base64,{b64}", detail=detail)


@dataclass
class StructurePart(ContentPart):
    """A crystallographic structure content part (CIF/POSCAR).
    
    Materials science specific — carries the structure text plus
    metadata (space group, formula) for the LLM to reason about.
    """
    type: str = "structure"
    format: str = "cif"  # "cif" | "poscar" | "xyz"
    content: str = ""  # Raw structure text
    formula: str = ""  # Chemical formula (optional)
    space_group: str = ""  # Space group (optional)
    
    def to_openai_format(self) -> dict[str, Any]:
        return {"type": "text", "text": f"```{self.format}\n{self.content}\n```"}
    
    def to_anthropic_format(self) -> dict[str, Any]:
        return {"type": "text", "text": f"```{self.format}\n{self.content}\n```"}
    
    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "StructurePart":
        return cls(
            format=data.get("format", "cif"),
            content=data.get("content", ""),
            formula=data.get("formula", ""),
            space_group=data.get("space_group", ""),
        )


@dataclass
class PlotPart(ContentPart):
    """A matplotlib plot content part.
    
    Carries a base64-encoded PNG of a plot, plus optional metadata
    about what the plot shows (axis labels, title, data range).
    """
    type: str = "plot"
    image_url: str = ""  # base64 data URI
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    
    def to_openai_format(self) -> dict[str, Any]:
        # Send as image_url with text description
        return {
            "type": "image_url",
            "image_url": {"url": self.image_url, "detail": "high"},
        }
    
    def to_anthropic_format(self) -> dict[str, Any]:
        if self.image_url.startswith("data:"):
            header, _, b64data = self.image_url.partition(",")
            media_type = "image/png"
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": b64data},
            }
        return {"type": "image", "source": {"type": "url", "url": self.image_url}}
    
    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "PlotPart":
        return cls(
            image_url=data.get("image_url", ""),
            title=data.get("title", ""),
            x_label=data.get("x_label", ""),
            y_label=data.get("y_label", ""),
        )
    
    @classmethod
    def from_file(cls, path: str | Path, title: str = "", x_label: str = "", y_label: str = "") -> "PlotPart":
        """Create a PlotPart from a PNG file."""
        path = Path(path)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return cls(
            image_url=f"data:image/png;base64,{b64}",
            title=title, x_label=x_label, y_label=y_label,
        )


def content_to_message(content: str | list[ContentPart], provider: str = "openai") -> str | list[dict]:
    """Convert content (str or list of ContentPart) to provider message format.
    
    If content is a string, returns it as-is.
    If content is a list of ContentParts, converts each to the provider format.
    """
    if isinstance(content, str):
        return content
    
    if provider == "anthropic":
        return [part.to_anthropic_format() for part in content]
    else:
        return [part.to_openai_format() for part in content]


__all__ = [
    "ContentPart",
    "TextPart",
    "ImageURLPart",
    "StructurePart",
    "PlotPart",
    "content_to_message",
]
