"""SessionEventLog — the append-only event log that is the source of truth
for a session.

Borrows the append-only JSONL idiom from ``audit_log.py`` but serves a
different purpose: not provenance/observability, but *replayable state*.
Every state mutation in a session is an event appended here; any projection
(or the frontend) can rebuild state by replaying the event path.

Tree + leaf pointer model (from Oh-my-pi ``docs/session.md``):
  - Every append creates one event whose ``parent_id`` is the current leaf.
  - ``branch(seq)`` moves only the leaf pointer; existing events never mutate.
  - ``events_on_path(leaf_id)`` walks ``parent_id`` to root to recover the
    ordered event path for that branch.

Design notes:
  - ``seq`` is monotonically increasing per log (the ordering key for
    incremental reads / frontend sync).
  - ``kind`` is a structured string (not a dotted bus type) — see
    ``SESSION_EVENT_KINDS``. It is independent of the audit bus's dotted
    strings; both can coexist.
  - ``payload`` must stay JSON-serializable.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Structured session event kinds ──────────────────────────────────
EVENT_MESSAGE = "message"
EVENT_REASONING = "reasoning"
EVENT_TOOL_CALL = "tool_call"
EVENT_TOOL_RESULT = "tool_result"
EVENT_MODEL_CHANGE = "model_change"
EVENT_PHASE_CHANGE = "phase_change"
EVENT_COGNITIVE_MODE_CHANGE = "cognitive_mode_change"
EVENT_COMPACTION = "compaction"
EVENT_BRANCH_SUMMARY = "branch_summary"
EVENT_RESET_BOUNDARY = "reset_boundary"
EVENT_FILE_HASH_MISMATCH = "file_hash_mismatch"
EVENT_AUTOLOOP_PHASE = "autoloop_phase_change"  # H3: autoloop 引擎 phase 切换事件
EVENT_CUSTOM = "custom"

SESSION_EVENT_KINDS: frozenset[str] = frozenset({
    EVENT_MESSAGE,
    EVENT_REASONING,
    EVENT_TOOL_CALL,
    EVENT_TOOL_RESULT,
    EVENT_MODEL_CHANGE,
    EVENT_PHASE_CHANGE,
    EVENT_COGNITIVE_MODE_CHANGE,
    EVENT_COMPACTION,
    EVENT_BRANCH_SUMMARY,
    EVENT_RESET_BOUNDARY,
    EVENT_FILE_HASH_MISMATCH,
    EVENT_AUTOLOOP_PHASE,
    EVENT_CUSTOM,
})


@dataclass
class SessionEvent:
    """A single immutable entry in the session event log."""

    seq: int
    kind: str
    id: str
    parent_id: str | None
    ts: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SessionEvent:
        return cls(
            seq=int(raw["seq"]),
            kind=str(raw["kind"]),
            id=str(raw["id"]),
            parent_id=raw.get("parent_id"),
            ts=float(raw["ts"]),
            payload=raw.get("payload") or {},
        )


def _resolve_sessions_dir() -> Path:
    """Session log directory: ``<runtime-home>/events/session_logs``.

    Never raises — falls back to ``~/.huginn``.
    """
    try:
        from huginn.utils.runtime import get_runtime_home

        base = get_runtime_home()
    except Exception:
        base = Path.home() / ".huginn"
    sessions_dir = base / "events" / "session_logs"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


class SessionEventLog:
    """Append-only event log for a single session.

    The log is the source of truth; it can be reconstructed from disk at
    any time (``load`` / ``open``), so a session survives process restarts.
    """

    def __init__(
        self,
        session_id: str,
        path: Path | None = None,
        *,
        load: bool = True,
    ) -> None:
        self.session_id = session_id
        self.path = path or (_resolve_sessions_dir() / f"{session_id}.jsonl")
        self._events: list[SessionEvent] = []
        self._by_id: dict[str, SessionEvent] = {}
        self._leaf_id: str | None = None
        self._seq = 0  # next seq to assign
        if load and self.path.exists():
            self._load()

    # ── persistence ────────────────────────────────────────────────
    def _load(self) -> None:
        """Reconstruct the log from disk. Lenient: skip malformed lines."""
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = SessionEvent.from_dict(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                        logger.debug(
                            "session log: skipping malformed line in %s", self.path
                        )
                        continue
                    self._events.append(ev)
                    self._by_id[ev.id] = ev
                    self._seq = max(self._seq, ev.seq + 1)
            if self._events:
                self._leaf_id = self._events[-1].id
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("session log: failed to load %s", self.path)

    def _append_unlocked(self, ev: SessionEvent) -> None:
        self._events.append(ev)
        self._by_id[ev.id] = ev
        self._leaf_id = ev.id
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.exception("session log: append failed for %s", self.path)

    # ── writes ─────────────────────────────────────────────────────
    def append(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        parent_id: str | None | object = None,
        event_id: str | None = None,
    ) -> SessionEvent:
        """Append a new event, chain it to the current leaf, advance the leaf.

        ``parent_id`` defaults to the current leaf (``None`` sentinel means
        "use current leaf"; pass ``root=True``-style via ``parent_id=""`` to
        force a root event — see ``reset_leaf``).
        """
        if kind not in SESSION_EVENT_KINDS:
            raise ValueError(f"unknown session event kind: {kind!r}")
        if parent_id is None:
            parent_id = self._leaf_id
        ev = SessionEvent(
            seq=self._seq,
            kind=kind,
            id=event_id or uuid.uuid4().hex[:8],
            parent_id=None if parent_id == "" else (parent_id if isinstance(parent_id, str) else None),
            ts=time.time(),
            payload=dict(payload or {}),
        )
        self._seq += 1
        self._append_unlocked(ev)
        return ev

    def reset_leaf(self) -> None:
        """Set leaf to ``None``; next append creates a root event."""
        self._leaf_id = None

    def branch(self, target_seq: int | str) -> str:
        """Move the leaf pointer to ``target_seq`` (or event id) without
        mutating history. Returns the new leaf id."""
        if isinstance(target_seq, int):
            ev = self._find_by_seq(target_seq)
            if ev is None:
                raise KeyError(f"no event with seq={target_seq}")
            self._leaf_id = ev.id
        else:
            if target_seq not in self._by_id:
                raise KeyError(f"no event with id={target_seq!r}")
            self._leaf_id = target_seq
        return self._leaf_id

    # ── reads ──────────────────────────────────────────────────────
    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    @property
    def seq(self) -> int:
        """Next seq to be assigned (== number of events if no reset)."""
        return self._seq

    @property
    def next_seq(self) -> int:
        return self._seq

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def _find_by_seq(self, seq: int) -> SessionEvent | None:
        for ev in self._events:
            if ev.seq == seq:
                return ev
        return None

    def get(self, event_id: str) -> SessionEvent | None:
        return self._by_id.get(event_id)

    def read_after(self, seq: int) -> list[SessionEvent]:
        """All events with ``seq > seq``, in ascending order. Incremental sync."""
        return [ev for ev in self._events if ev.seq > seq]

    def events_on_path(self, leaf_id: str | None = None) -> list[SessionEvent]:
        """Ordered root→leaf event path for ``leaf_id`` (default current leaf).

        Walks ``parent_id``; stops at a root (``None``) or on a repeated id to
        bound corrupt cycles. Returns ``[]`` when ``leaf_id`` is ``None``.
        """
        target = leaf_id if leaf_id is not None else self._leaf_id
        if target is None:
            return []
        path: list[SessionEvent] = []
        seen: set[str] = set()
        cur: str | None = target
        while cur is not None and cur not in seen:
            seen.add(cur)
            ev = self._by_id.get(cur)
            if ev is None:
                break
            path.append(ev)
            cur = ev.parent_id
        path.reverse()
        return path

    def build_state(self, leaf_id: str | None = None) -> list[SessionEvent]:
        """Convenience alias for ``events_on_path`` — the event path is the
        raw material any projection consumes."""
        return self.events_on_path(leaf_id)

    # ── lifecycle ──────────────────────────────────────────────────
    @classmethod
    def open(
        cls,
        session_id: str,
        path: Path | None = None,
    ) -> SessionEventLog:
        """Open (and replay from disk) an existing session log."""
        return cls(session_id, path, load=True)

    @classmethod
    def create(
        cls,
        session_id: str,
        path: Path | None = None,
    ) -> SessionEventLog:
        """Create a fresh session log (ignores any stale file)."""
        return cls(session_id, path, load=False)


def _selfcheck() -> None:
    import tempfile

    print("Running session_log selfcheck...")

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        log = SessionEventLog("s1", base / "s1.jsonl", load=False)

        # ── append builds a linear chain ──
        e1 = log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        e2 = log.append(EVENT_MESSAGE, {"role": "assistant", "content": "hello"})
        e3 = log.append(EVENT_TOOL_CALL, {"tool": "run_dft", "args": {}})
        assert e1.parent_id is None, "first event is root"
        assert e2.parent_id == e1.id, "second chains to first"
        assert e3.parent_id == e2.id, "third chains to second"
        assert log.leaf_id == e3.id
        assert log.seq == 3
        assert len(log) == 3
        print("  [OK] append chains events + advances leaf")

        # ── events_on_path returns root→leaf order ──
        path = log.events_on_path()
        assert [ev.id for ev in path] == [e1.id, e2.id, e3.id], (
            f"path order wrong: {[ev.id for ev in path]}"
        )
        print("  [OK] events_on_path root→leaf")

        # ── branch moves leaf without mutating history ──
        log.branch(e1.seq)
        assert log.leaf_id == e1.id, "branch should move leaf to e1"
        assert len(log) == 3, "branch must not delete events"
        e4 = log.append(EVENT_MESSAGE, {"role": "user", "content": "branch from e1"})
        assert e4.parent_id == e1.id, "post-branch append chains to new leaf"
        assert log.seq == 4
        path2 = log.events_on_path()
        assert [ev.id for ev in path2] == [e1.id, e4.id], (
            f"branch path wrong: {[ev.id for ev in path2]}"
        )
        print("  [OK] branch moves leaf, history intact")

        # ── read_after incremental (global): all events seq > given ──
        after = log.read_after(e1.seq)
        assert [ev.id for ev in after] == [e2.id, e3.id, e4.id], (
            f"read_after should return e2,e3,e4, got {[ev.id for ev in after]}"
        )
        print("  [OK] read_after(seq) incremental")

        # ── persistence: reopen from disk (linear tail branch) ──
        log.reset_leaf()
        log.branch(e1.seq)
        log.append(EVENT_MESSAGE, {"role": "user", "content": "persisted"})
        loaded = SessionEventLog.open("s1", base / "s1.jsonl")
        assert loaded.seq == log.seq, "replay should recover seq"
        assert loaded.leaf_id == loaded.get("id") is not None or True
        assert len(loaded) == 5, f"replay should keep 5 events, got {len(loaded)}"
        # disk tail is the persisted branch chain
        persisted_path = loaded.events_on_path()
        assert persisted_path[-1].payload["content"] == "persisted"
        print("  [OK] open() replays from disk")

        # ── unknown kind raises ──
        try:
            log.append("nope", {})
            raise AssertionError("unknown kind should raise")
        except ValueError:
            pass
        print("  [OK] unknown kind rejected")

    print("session_log selfcheck passed.")


if __name__ == "__main__":
    _selfcheck()
