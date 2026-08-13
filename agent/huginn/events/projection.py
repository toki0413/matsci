"""Projection engine over a :class:`~huginn.events.session_log.SessionEventLog`.

Models a session's state as a set of *projections* — each is a pure function
over the event path. This is the dsh ``session-projection`` pattern
(``ProjectionDefinition{init/apply/view/stateVersion}``) mapped onto Huginn's
``SessionEventLog``.

Why projections:
  - State is *derived*, never stored redundantly — replay is the only way to
    (re)compute it, so it is always consistent with the log.
  - Different consumers (LLM context, runtime phase machine, frontend blocks)
    each subscribe to their own projection; one event fan-outs to all.
  - ``stateVersion`` bumps when a projection contract changes; the engine
    rebuilds that projection instead of letting stale cells drift.

Cells are cached with a ``weakref.WeakKeyDictionary`` keyed by the log
object, so a session that is no longer referenced is collected automatically.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from huginn.events.session_log import SessionEvent, SessionEventLog

logger = logging.getLogger(__name__)

S = TypeVar("S")  # internal projection state
V = TypeVar("V")  # view value


class ProjectionDefinition:  # noqa: D101
    """A pure-function projection. Subclass and override ``apply``/``view``.

    ``apply(state, event) -> state`` must be a pure function: same event path
    always yields the same state. ``view(state) -> V`` is the read-model
    exposed to consumers. ``init()`` returns the empty state.
    """

    key: str
    stateVersion: int = 1  # noqa: N815 - dsh/projection naming (deliberate)

    def init(self) -> S:  # pragma: no cover - overridden
        raise NotImplementedError

    def apply(self, state: S, event: SessionEvent) -> S:  # pragma: no cover
        raise NotImplementedError

    def view(self, state: S) -> V:  # pragma: no cover
        raise NotImplementedError


@dataclass(eq=False)
class _Cell:
    """Mutable per-(log, projection) slot: current state + last observed seq."""

    state: Any
    observed_seq: int = -1
    version: int = 1


class ProjectionEngine:
    """Registry + driver for projections over session logs.

    Usage::

        engine = ProjectionEngine()
        engine.register(RuntimeStateProjection())
        engine.drive(log, event)          # push one event to all projections
        value = engine.build(log, "runtime")  # ensure + return view
        unsub = engine.subscribe(log, "runtime", listener)
    """

    def __init__(self) -> None:
        self._defs: dict[str, ProjectionDefinition] = {}
        # log object -> {def_key: _Cell}; weak to auto-collect dead sessions.
        self._cells: weakref.WeakKeyDictionary[SessionEventLog, dict[str, _Cell]] = (
            weakref.WeakKeyDictionary()
        )
        # (log, def_key) -> [listeners]; weak so we don't keep listeners alive.
        self._listeners: weakref.WeakKeyDictionary[
            SessionEventLog, dict[str, list[Any]]
        ] = weakref.WeakKeyDictionary()

    # ── registration ───────────────────────────────────────────────
    def register(self, definition: ProjectionDefinition) -> None:
        """Register a projection definition. Re-registering resets its cells."""
        self._defs[definition.key] = definition
        # version bump handled lazily in build() (def registry is authoritative).

    def unregister(self, key: str) -> None:
        self._defs.pop(key, None)

    def definitions(self) -> dict[str, ProjectionDefinition]:
        return dict(self._defs)

    # ── building / driving ─────────────────────────────────────────
    def _cells_for(self, log: SessionEventLog) -> dict[str, _Cell]:
        cells = self._cells.get(log)
        if cells is None:
            cells = {}
            self._cells[log] = cells
        return cells

    def _rebuild_cell_from(
        self,
        log: SessionEventLog,
        def_key: str,
        prefix_events: list[SessionEvent],
    ) -> _Cell:
        """Apply ``prefix_events`` in order to a fresh cell (init + apply)."""
        definition = self._defs[def_key]
        state = definition.init()
        observed = -1
        for ev in prefix_events:
            state = definition.apply(state, ev)
            observed = ev.seq
        cell = _Cell(state=state, observed_seq=observed, version=definition.stateVersion)
        self._cells_for(log)[def_key] = cell
        return cell

    def drive(self, log: SessionEventLog, event: SessionEvent) -> None:
        """Apply one event to every registered projection, fanning out to
        listeners when a projection's view changed.

        The cell is built from the events *strictly before* ``event`` on first
        touch, then ``event`` is applied exactly once — so driving a path in
        order never double-applies.
        """
        path = log.events_on_path()
        for def_key, definition in self._defs.items():
            cells = self._cells_for(log)
            cell = cells.get(def_key)
            if cell is None or cell.version != definition.stateVersion:
                prefix = [ev for ev in path if ev.seq < event.seq]
                cell = self._rebuild_cell_from(log, def_key, prefix)
            elif event.seq <= cell.observed_seq:
                continue  # stale / duplicate delivery — already applied
            next_state = definition.apply(cell.state, event)
            changed = next_state is not cell.state
            cell.state = next_state
            cell.observed_seq = event.seq
            if changed:
                self._notify(log, def_key, definition.view(cell.state), event.seq)

    def build(self, log: SessionEventLog, def_key: str) -> Any:
        """Ensure the projection is fully built from the log's current path
        and return its view value."""
        definition = self._defs[def_key]
        cells = self._cells_for(log)
        cell = cells.get(def_key)
        if cell is None or cell.version != definition.stateVersion:
            cell = self._rebuild_cell_from(log, def_key, log.events_on_path())
        return definition.view(cell.state)

    # ── subscriptions ──────────────────────────────────────────────
    def _notify(self, log: SessionEventLog, def_key: str, value: Any, seq: int) -> None:
        listeners = self._listeners.get(log, {}).get(def_key)
        if not listeners:
            return
        for listener in list(listeners):
            try:
                listener(log, def_key, value, seq)
            except Exception:
                logger.exception(
                    "projection listener failed for %s/%s", log.session_id, def_key
                )

    def subscribe(
        self,
        log: SessionEventLog,
        def_key: str,
        listener: Any,
    ) -> Any:
        """Listen for projection view changes. Returns an unsubscribe fn."""
        listeners = self._listeners.setdefault(log, {})
        listeners.setdefault(def_key, []).append(listener)

        def _unsubscribe() -> None:
            ll = self._listeners.get(log, {}).get(def_key)
            if ll is not None:
                with _suppress(ValueError):
                    ll.remove(listener)

        return _unsubscribe


def _suppress(exc: type[Exception]) -> Any:
    import contextlib

    return contextlib.suppress(exc)


# ── built-in projections ────────────────────────────────────────────
class RuntimeStateProjection(ProjectionDefinition):
    """Replays the runtime state that survives compaction (mode/phase/plan/
    model). Mirrors ``UnifiedSessionState`` persistent fields."""

    key = "runtime"
    stateVersion = 1  # noqa: N815

    def __init__(self) -> None:
        self._default: dict[str, Any] = {
            "cognitive_mode": "discover",
            "phase": "explore",
            "l1_coordinates": "",
            "active_plan_id": None,
            "active_plan_objective": "",
            "active_plan_step_index": 0,
            "model": None,
            "turns_count": 0,
        }

    def init(self) -> dict[str, Any]:
        return dict(self._default)

    def apply(self, state: dict[str, Any], event: SessionEvent) -> dict[str, Any]:
        # Return the *same* state object when nothing changed, so the engine's
        # identity-based change detection stays meaningful (no spurious fires).
        if event.kind == "cognitive_mode_change":
            v = event.payload.get("cognitive_mode")
            if v is not None and v != state["cognitive_mode"]:
                return {**state, "cognitive_mode": v}
        elif event.kind == "phase_change":
            v = event.payload.get("phase")
            if v is not None and v != state["phase"]:
                return {**state, "phase": v}
        elif event.kind == "model_change":
            if event.payload.get("model") != state["model"]:
                return {**state, "model": event.payload.get("model")}
        elif event.kind == "message" and event.payload.get("role") == "user":
            return {**state, "turns_count": state["turns_count"] + 1}
        return state

    def view(self, state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)


class MessagePathProjection(ProjectionDefinition):
    """Collects the LLM-visible message path (messages + custom messages).

    ``compaction`` events are *not* dropped — they are kept as markers so a
    later consumer (or the frontend) can render ``── compacted ──`` dividers
    instead of losing history.
    """

    key = "messages"
    stateVersion = 1  # noqa: N815

    def init(self) -> list[dict[str, Any]]:
        return []

    def apply(self, state: list[dict[str, Any]], event: SessionEvent) -> list[dict[str, Any]]:
        if event.kind == "message":
            state = [*state, {"role": event.payload.get("role"), "content": event.payload.get("content")}]
        elif event.kind == "compaction":
            state = [*state, {"role": "compaction", "summary": event.payload.get("summary", "")}]
        elif event.kind == "branch_summary":
            state = [*state, {"role": "branch_summary", "summary": event.payload.get("summary", "")}]
        return state

    def view(self, state: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(state)


# Block-kind constants for the UI projection (aligned with frontend UiProjection).
BlockKind = str
BLOCK_TEXT = "text"
BLOCK_COMPACTION = "compaction"
BLOCK_TOOL = "tool"


class UiBlock:
    """An immutable render block for one message.

    ``frozen`` marks a block that is final and must not re-render; a streaming
    tail block flips ``frozen`` from ``False`` to ``True`` exactly once when
    its message completes. ``rev`` increments on each mutation so the frontend
    can diff cheaply.
    """

    __slots__ = ("kind", "text", "frozen", "rev")

    def __init__(self, kind: BlockKind, text: str = "", frozen: bool = False, rev: int = 0) -> None:
        self.kind = kind
        self.text = text
        self.frozen = frozen
        self.rev = rev

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "frozen": self.frozen, "rev": self.rev}


class UiProjection(ProjectionDefinition):
    """Block-structured session view for the incremental frontend engine.

    Each *message* becomes one ``UiBlock``: ``frozen`` is ``False`` while its
    text may still grow (streaming), ``True`` once sealed. ``compaction`` and
    ``branch_summary`` events become their own frozen divider blocks so the
    frontend renders ``── compacted ──`` without losing prior history.

    The projection intentionally returns *new lists/new blocks* on every
    mutation and bumps ``rev`` — the engine's identity-based change detection
    then drives listeners (e.g. a WS delta) on exactly the changed tail.
    """

    key = "ui"
    stateVersion = 1  # noqa: N815

    def init(self) -> list[UiBlock]:
        return []

    def apply(self, state: list[UiBlock], event: SessionEvent) -> list[UiBlock]:
        if event.kind == "message":
            role = event.payload.get("role")
            content = event.payload.get("content", "")
            text = f"**{role}:** {content}" if role else str(content)
            block = UiBlock(BLOCK_TEXT, text, frozen=True, rev=1)
            return [*state, block]
        elif event.kind == "compaction":
            block = UiBlock(BLOCK_COMPACTION, event.payload.get("summary", ""), frozen=True, rev=1)
            return [*state, block]
        elif event.kind == "branch_summary":
            block = UiBlock(BLOCK_TOOL, event.payload.get("summary", ""), frozen=True, rev=1)
            return [*state, block]
        return state  # metadata events don't affect the UI

    def view(self, state: list[UiBlock]) -> list[dict[str, Any]]:
        return [b.to_dict() for b in state]


def _selfcheck() -> None:
    import tempfile

    from huginn.events.session_log import (
        EVENT_COMPACTION,
        EVENT_MESSAGE,
        SessionEventLog,
    )

    print("Running projection selfcheck...")

    with tempfile.TemporaryDirectory() as tmp:
        log = SessionEventLog("s1", Path(tmp) / "s1.jsonl", load=False)
        engine = ProjectionEngine()
        engine.register(RuntimeStateProjection())
        engine.register(MessagePathProjection())

        # ── drive events → projections update ──
        log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        log.append(EVENT_MESSAGE, {"role": "assistant", "content": "ok"})
        log.append("cognitive_mode_change", {"cognitive_mode": "construct"})
        log.append(EVENT_COMPACTION, {"summary": "compacted early"})

        for ev in log:
            engine.drive(log, ev)

        runtime = engine.build(log, "runtime")
        assert runtime["cognitive_mode"] == "construct"
        assert runtime["turns_count"] == 1, (
            f"turns_count should be 1, got {runtime['turns_count']}"
        )
        msgs = engine.build(log, "messages")
        assert msgs[-1]["role"] == "compaction", "compaction kept as marker"
        assert len(msgs) == 3, f"expected 3 (user+assistant+compaction), got {len(msgs)}"
        print("  [OK] drive → both projections updated; compaction kept as marker")

        # ── build is idempotent / replay-safe from a cold engine ──
        engine2 = ProjectionEngine()
        engine2.register(RuntimeStateProjection())
        runtime2 = engine2.build(log, "runtime")
        assert runtime2 == runtime, "cold rebuild from path must equal driven state"
        print("  [OK] cold rebuild from event path == driven state")

        # ── listener fires on change only ──
        calls: list[tuple[str, str, int]] = []
        def _listener(log_, key, value, seq):
            calls.append((key, value["cognitive_mode"], seq))
        engine.subscribe(log, "runtime", _listener)
        log.append("cognitive_mode_change", {"cognitive_mode": "construct"})  # same value
        engine.drive(log, log.get(log.leaf_id))
        log.append("cognitive_mode_change", {"cognitive_mode": "discover"})  # change
        engine.drive(log, log.get(log.leaf_id))
        assert len(calls) == 1, f"listener should fire once, got {len(calls)}"
        assert calls[0][1] == "discover"  # value at fire time (the real change)
        print("  [OK] listener fires only when view changed")

        # ── stateVersion bump forces rebuild ──
        RuntimeStateProjection.stateVersion = 2
        try:
            engine.build(log, "runtime")  # rebuild with new version
            assert engine._cells_for(log)["runtime"].version == 2
            print("  [OK] stateVersion bump rebuilds cell")
        finally:
            RuntimeStateProjection.stateVersion = 1  # noqa: N815

    # ── UiProjection: block structure for the incremental frontend ──
    with tempfile.TemporaryDirectory() as tmp:
        from huginn.events.session_log import EVENT_BRANCH_SUMMARY
        log2 = SessionEventLog("s2", Path(tmp) / "s2.jsonl", load=False)
        eng2 = ProjectionEngine()
        eng2.register(UiProjection())
        log2.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        log2.append(EVENT_COMPACTION, {"summary": "compacted"})
        log2.append(EVENT_BRANCH_SUMMARY, {"summary": "abandoned path"})
        for ev in log2:
            eng2.drive(log2, ev)
        blocks = eng2.build(log2, "ui")
        assert [b["kind"] for b in blocks] == [BLOCK_TEXT, BLOCK_COMPACTION, BLOCK_TOOL]
        assert all(b["frozen"] for b in blocks), "all blocks frozen"
        assert all(b["rev"] == 1 for b in blocks)
        assert blocks[1]["kind"] == BLOCK_COMPACTION
        print("  [OK] UiProjection emits frozen blocks + compaction/tool dividers")

    print("projection selfcheck passed.")


if __name__ == "__main__":
    _selfcheck()
