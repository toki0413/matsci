"""Session memory — short-term context for active conversations.

Stores messages, tool calls, and reasoning traces for the current session.
Automatically compacts when context grows too large.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from huginn.types import AgentMessage, ToolResult


@dataclass
class ToolCallRecord:
    """Record of a single tool invocation."""
    tool_name: str
    input_args: dict[str, Any]
    result: ToolResult | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    call_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class SessionContext:
    """Mutable context for the current agent session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    messages: list[AgentMessage] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    reasoning_trace: list[str] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    user_preferences: dict[str, Any] = field(default_factory=dict)

    # Context compaction settings
    max_messages: int = 100
    max_tool_calls: int = 50
    max_reasoning_lines: int = 200

    def add_message(self, message: AgentMessage | str, content: str | None = None) -> None:
        if isinstance(message, str) and content is not None:
            message = AgentMessage(role=message, content=content)
        self.messages.append(message)
        self._compact_if_needed()

    def add_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)
        if len(self.tool_calls) > self.max_tool_calls:
            # Keep most recent, archive oldest
            self.tool_calls = self.tool_calls[-self.max_tool_calls :]

    def add_reasoning(self, text: str) -> None:
        self.reasoning_trace.append(text)
        if len(self.reasoning_trace) > self.max_reasoning_lines:
            # Summarize oldest entries
            self.reasoning_trace = self.reasoning_trace[-self.max_reasoning_lines :]

    def set_working_memory(self, key: str, value: Any) -> None:
        self.working_memory[key] = value

    def get_working_memory(self, key: str, default: Any = None) -> Any:
        return self.working_memory.get(key, default)

    def get_recent_messages(self, n: int = 10) -> list[AgentMessage]:
        return self.messages[-n:]

    def get_recent_tool_calls(self, n: int = 5) -> list[ToolCallRecord]:
        return self.tool_calls[-n:]

    def _compact_if_needed(self) -> None:
        if len(self.messages) > self.max_messages:
            # Strategy: keep first system message, then most recent
            system_msgs = [m for m in self.messages if m.role == "system"]
            recent = self.messages[-(self.max_messages - len(system_msgs)) :]
            self.messages = system_msgs + recent

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_calls),
            "working_memory_keys": list(self.working_memory.keys()),
            "user_preferences": self.user_preferences,
        }

    def export_full(self) -> dict[str, Any]:
        """Export complete session for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "messages": [
                {
                    "role": m.role,
                    "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.messages
            ],
            "tool_calls": [
                {
                    "tool_name": t.tool_name,
                    "input_args": t.input_args,
                    "success": t.result.success if t.result else None,
                    "timestamp": t.timestamp.isoformat(),
                    "call_id": t.call_id,
                }
                for t in self.tool_calls
            ],
            "working_memory": self.working_memory,
        }
