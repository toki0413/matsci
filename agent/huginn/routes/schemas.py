"""Pydantic v2 request schemas for chat, WebSocket, and thread endpoints.

Centralizes input validation so every entry point enforces consistent
type, length, and format constraints instead of trusting raw dicts.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

# Restrict thread IDs to alphanumeric, underscore, and hyphen.
# Anything else risks path traversal or injection when the ID is used
# in checkpoint paths, SQLite keys, or log lines downstream.
_THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class ChatRequest(BaseModel):
    """Body schema for POST /agents/{id}/chat and the SSE chat/stream variant."""

    content: str = Field(..., max_length=50000, description="User message text")
    thread_id: str = Field("default", max_length=128, pattern=_THREAD_ID_PATTERN)
    thinking: Optional[str] = None
    max_tokens: Optional[int] = Field(None, gt=0, le=100000)
    persona: Optional[str] = Field(None, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _accept_message_field(cls, values: Any) -> Any:
        # Some older clients still send "message" instead of "content".
        # Normalize it before validation so the length cap applies either way.
        if isinstance(values, dict):
            if "content" not in values and "message" in values:
                values = {**values, "content": values["message"]}
        return values


class WSMessage(BaseModel):
    """Inbound WebSocket message envelope, validated right after json.loads."""

    type: str = Field("user_input", max_length=64)
    content: str = Field("", max_length=50000)
    thread_id: str = Field("default", max_length=128, pattern=_THREAD_ID_PATTERN)
    thinking: Optional[Any] = None
    max_tokens: Optional[Any] = None
    persona: Optional[str] = Field(None, max_length=64)


class CreateThreadRequest(BaseModel):
    """Body schema for POST /threads."""

    title: Optional[str] = Field(None, max_length=256)
    metadata: Optional[dict[str, Any]] = None
