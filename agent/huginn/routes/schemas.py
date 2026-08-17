"""Pydantic v2 request schemas for chat, WebSocket, and thread endpoints.

Centralizes input validation so every entry point enforces consistent
type, length, and format constraints instead of trusting raw dicts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

# Restrict thread IDs to alphanumeric, underscore, and hyphen.
# Anything else risks path traversal or injection when the ID is used
# in checkpoint paths, SQLite keys, or log lines downstream.
_THREAD_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class ChatRequest(BaseModel):
    """Body schema for POST /agents/{id}/chat and the SSE chat/stream variant."""

    content: str = Field(..., max_length=50000, description="User message text")
    thread_id: str = Field("default", max_length=128, pattern=_THREAD_ID_PATTERN)
    thinking: str | None = None
    max_tokens: int | None = Field(None, gt=0, le=100000)
    persona: str | None = Field(None, max_length=64)

    @model_validator(mode="before")
    @classmethod
    def _accept_message_field(cls, values: Any) -> Any:
        # Some older clients still send "message" instead of "content".
        # Normalize it before validation so the length cap applies either way.
        if isinstance(values, dict) and "content" not in values and "message" in values:
            values = {**values, "content": values["message"]}
        return values


class WSMessage(BaseModel):
    """Inbound WebSocket message envelope, validated right after json.loads."""

    type: str = Field("user_input", max_length=64)
    content: str = Field("", max_length=50000)
    thread_id: str = Field("default", max_length=128, pattern=_THREAD_ID_PATTERN)
    thinking: Any | None = None
    max_tokens: Any | None = None
    persona: str | None = Field(None, max_length=64)

    # Plan / approval / clarification / suggest flows. ws.py reads these off
    # the validated WSMessage instead of the raw dict, so every field a
    # handler consumes must be declared here to stay in sync.
    plan_id: str | None = None
    confirmed: bool | None = None
    edited_plan: Any | None = None
    question_id: str | None = None
    answer: str | None = None
    request_id: str | None = None
    approved: bool | None = None
    enabled: bool | None = None
    action: str | None = None
    edited_code: str | None = None
    config: dict[str, Any] | None = Field(None, description="explore_start config")

    # decision_response (cost & pruning participation, contract §3.2)
    decision_point_id: str | None = None
    decision: str | None = None
    option: str | None = None


class CreateThreadRequest(BaseModel):
    """Body schema for POST /threads."""

    title: str | None = Field(None, max_length=256)
    metadata: dict[str, Any] | None = None
