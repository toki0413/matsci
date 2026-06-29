"""Token-count estimation utilities.

Primary path uses tiktoken for OpenAI-compatible tokenization.
Fallback is a conservative character-length heuristic inspired by
Claude Code's tokenEstimation.ts.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cache of tiktoken encoders keyed by encoding name. Building an Encoding
# is relatively expensive, so we keep them around for the lifetime of the
# process. tiktoken is an optional dep, so values are typed as Any.
_encoder_cache: dict[str, Any] = {}
# Encodings we already attempted to load, so we don't retry on every call.
_tried_encodings: set[str] = set()


def _select_encoder_for_model(model_name: str | None) -> str:
    """Pick the tiktoken encoding name that best matches a given model.

    Falls back to cl100k_base for anything we don't recognise — it's a
    reasonable approximation for budget estimation across most models.
    """
    if not model_name:
        return "cl100k_base"
    name = model_name.lower()
    # GPT-4o family and o1/o3 reasoning models ship the newer o200k vocab.
    if "gpt-4o" in name or "o1" in name or "o3" in name:
        return "o200k_base"
    # GPT-4, GPT-3.5 and text-davinci use cl100k_base. Anthropic (Claude),
    # Moonshot (Kimi) and Deepseek don't publish tokenizers, but cl100k_base
    # tracks them closely enough for token-budget purposes.
    return "cl100k_base"


def _get_tiktoken_encoder(encoding_name: str = "cl100k_base") -> Any | None:
    """Return a cached tiktoken encoder for the given encoding, or None."""
    if encoding_name in _tried_encodings:
        return _encoder_cache.get(encoding_name)
    _tried_encodings.add(encoding_name)
    try:
        import tiktoken

        _encoder_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        logger.debug("tiktoken encoder loaded for %s", encoding_name)
    except ImportError:
        logger.debug("tiktoken not installed, falling back to heuristic estimation")
    except Exception as exc:
        logger.debug("tiktoken init failed for %s: %s", encoding_name, exc)
    return _encoder_cache.get(encoding_name)


def bytes_per_token_for_extension(ext: str | None) -> float:
    """Return a heuristic bytes-per-token ratio for dense formats."""
    if ext in ("json", "jsonl", "jsonc"):
        return 2.0
    return 4.0


def rough_token_count(text: str, bytes_per_token: float = 4.0) -> int:
    """Estimate token count from character length."""
    if not text:
        return 0
    return max(1, round(len(text) / bytes_per_token))


def rough_token_count_for_text(text: str, file_extension: str | None = None) -> int:
    """Estimate token count using a format-aware ratio."""
    return rough_token_count(text, bytes_per_token_for_extension(file_extension))


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Count tokens accurately with tiktoken, falling back to heuristic.

    The encoder is chosen based on ``model_name``. When omitted (or when
    the model isn't recognised) we default to cl100k_base, which is
    GPT-4/Claude compatible. For CJK text the heuristic overestimates by
    ~2x, so tiktoken is strongly preferred when available.
    """
    if not text:
        return 0
    encoding_name = _select_encoder_for_model(model_name)
    encoder = _get_tiktoken_encoder(encoding_name)
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return rough_token_count_for_text(text)


def count_message_tokens(
    content: str | list[Any] | None,
    role: str = "user",
    model_name: str | None = None,
) -> int:
    """Count tokens for a single message, including role overhead.

    Each message has ~4 tokens of structural overhead (role tags, separators).
    ``model_name`` is forwarded to :func:`count_tokens` so the right encoder
    is used for the message body.
    """
    if content is None:
        return 4
    if isinstance(content, list):
        # Multi-block content (e.g. text + image)
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                text_parts.append(str(block.get("text", "")))
        text = "\n".join(text_parts)
    else:
        text = str(content)
    return count_tokens(text, model_name=model_name) + 4  # +4 for role/separators
