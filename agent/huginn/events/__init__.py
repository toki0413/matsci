"""Unified event bus + SSE observability for Huginn.

Inspired by OpenCode's bus/ + event-v2-bridge pattern. This package
provides a typed, async event bus that any component can publish to
and subscribe from. Subscribers can push events to SSE clients, write
to an audit log, or trigger reactive hooks.

Importing this package must never break the agent. If anything fails
to import, the integration helpers degrade to no-ops silently.
"""

from huginn.events.event_bus import AgentEvent, EventBus
from huginn.events.event_types import (
    ALL,
    CAMPAIGN_HYPOTHESIS,
    CAMPAIGN_ITERATION,
    CAMPAIGN_REFINE,
    COMPACT_END,
    COMPACT_START,
    CONTEXT_OVERFLOW,
    PIPELINE_STAGE_CHANGE,
    PIPELINE_SUGGEST,
    SESSION_END,
    SESSION_START,
    SNAPSHOT_REVERT,
    SNAPSHOT_TAKE,
    STEP_RETRY,
    TOOL_BLOCKED,
    TOOL_CALL,
    TOOL_ERROR,
    TOOL_RESULT,
)
from huginn.events.projection import (
    AutoloopStateProjection,
    MessagePathProjection,
    ProjectionDefinition,
    ProjectionEngine,
    RuntimeStateProjection,
    UiBlock,
    UiProjection,
)

# T-BCSE-01/02/04: session event-sourcing core (append-only log + projections).
# See huginn/events/session_log.py and huginn/events/projection.py.
from huginn.events.session_log import (
    EVENT_AUTOLOOP_PHASE,
    EVENT_BRANCH_SUMMARY,
    EVENT_COMPACTION,
    EVENT_CUSTOM,
    EVENT_MESSAGE,
    EVENT_RESET_BOUNDARY,
    CompactionEntry,
    SessionEvent,
    SessionEventLog,
)
from huginn.events.transcript import TranscriptStore

__all__ = [
    "AgentEvent",
    "EventBus",
    "TranscriptStore",
    # Event types
    "ALL",
    "TOOL_CALL",
    "TOOL_RESULT",
    "TOOL_ERROR",
    "TOOL_BLOCKED",
    "COMPACT_START",
    "COMPACT_END",
    "CONTEXT_OVERFLOW",
    "PIPELINE_SUGGEST",
    "PIPELINE_STAGE_CHANGE",
    "CAMPAIGN_ITERATION",
    "CAMPAIGN_REFINE",
    "CAMPAIGN_HYPOTHESIS",
    "SNAPSHOT_TAKE",
    "SNAPSHOT_REVERT",
    "SESSION_START",
    "SESSION_END",
    "STEP_RETRY",
    # Session event-sourcing core (T-BCSE-01/02/04)
    "SessionEvent",
    "SessionEventLog",
    "CompactionEntry",
    "EVENT_MESSAGE",
    "EVENT_COMPACTION",
    "EVENT_BRANCH_SUMMARY",
    "EVENT_RESET_BOUNDARY",
    "EVENT_CUSTOM",
    "EVENT_AUTOLOOP_PHASE",
    "ProjectionDefinition",
    "ProjectionEngine",
    "RuntimeStateProjection",
    "MessagePathProjection",
    "AutoloopStateProjection",
    "UiBlock",
    "UiProjection",
]
