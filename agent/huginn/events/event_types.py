"""Standard event type strings for the unified event bus.

These are dotted strings (not enums) so external subscribers — SSE clients,
audit log readers, shell scripts — can match on prefixes without importing
Python code. Adding a new type is just adding a constant here.

Groups:
  - tool.*    : tool call lifecycle
  - compact.* : context compaction
  - pipeline.*: workflow pipeline transitions
  - campaign.*: autonomous loop events
  - snapshot.*: state snapshot take/revert
  - session.*: session start/end
"""

# ── Tool lifecycle ──────────────────────────────────────────────────
TOOL_CALL = "tool.call"
TOOL_RESULT = "tool.result"
TOOL_ERROR = "tool.error"
TOOL_BLOCKED = "tool.blocked"

# ── Context management ───────────────────────────────────────────────
COMPACT_START = "compact.start"
COMPACT_END = "compact.end"
CONTEXT_OVERFLOW = "context.overflow"

# ── Pipeline ────────────────────────────────────────────────────────
PIPELINE_SUGGEST = "pipeline.suggest"
PIPELINE_STAGE_CHANGE = "pipeline.stage_change"

# ── Campaign (autonomous loop) ───────────────────────────────────────
CAMPAIGN_ITERATION = "campaign.iteration"
CAMPAIGN_REFINE = "campaign.refine"
CAMPAIGN_HYPOTHESIS = "campaign.hypothesis"

# ── Snapshot ─────────────────────────────────────────────────────────
SNAPSHOT_TAKE = "snapshot.take"
SNAPSHOT_REVERT = "snapshot.revert"

# ── Quality ────────────────────────────────────────────────────────
QUALITY_CHECK = "quality.check"

# ── Cognitive heat engine (v7 G59) ────────────────────────────────
# 每轮 darwin_ratchet 后推送热机健康状态: Re_cog / η_cog / status / warnings.
# 前端订阅 /tasks/stream 的 'heat_engine.health' event 实时展示.
HEAT_ENGINE_HEALTH = "heat_engine.health"

# ── Session ─────────────────────────────────────────────────────────
SESSION_START = "session.start"
SESSION_END = "session.end"

# Wildcard — subscribe to this to receive everything.
ALL = "*"

# All known types, for validation / docs. Not exhaustive — callers can
# publish arbitrary type strings, this just helps catch typos in code.
ALL_TYPES = frozenset({
    TOOL_CALL, TOOL_RESULT, TOOL_ERROR, TOOL_BLOCKED,
    COMPACT_START, COMPACT_END, CONTEXT_OVERFLOW,
    PIPELINE_SUGGEST, PIPELINE_STAGE_CHANGE,
    CAMPAIGN_ITERATION, CAMPAIGN_REFINE, CAMPAIGN_HYPOTHESIS,
    SNAPSHOT_TAKE, SNAPSHOT_REVERT,
    QUALITY_CHECK,
    HEAT_ENGINE_HEALTH,
    SESSION_START, SESSION_END,
})
