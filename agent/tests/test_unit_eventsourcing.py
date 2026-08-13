"""Tests for UC-1 (unit conversion + dimensional check) and UC-2 (event sourcing)."""

from __future__ import annotations

import asyncio

import pytest

# ── UC-1: Unit Conversion ───────────────────────────────────────


class TestUnitConversion:
    def test_ev_to_joule(self):
        from huginn.hooks.unit_check import convert
        j = convert(1.0, "ev", "joule", "energy")
        assert j == pytest.approx(1.602e-19, rel=1e-3)

    def test_rydberg_to_ev(self):
        from huginn.hooks.unit_check import convert
        ev = convert(1.0, "rydberg", "ev", "energy")
        assert ev == pytest.approx(13.606, abs=0.01)

    def test_hartree_to_ev(self):
        from huginn.hooks.unit_check import convert
        ev = convert(1.0, "hartree", "ev", "energy")
        assert ev == pytest.approx(27.211, abs=0.01)

    def test_angstrom_to_bohr(self):
        from huginn.hooks.unit_check import convert
        bohr = convert(1.0, "angstrom", "bohr", "length")
        assert bohr == pytest.approx(1.8897, abs=0.01)

    def test_gpa_to_kbar(self):
        from huginn.hooks.unit_check import convert
        kbar = convert(100.0, "gpa", "kbar", "pressure")
        assert kbar == pytest.approx(1000.0, abs=1.0)

    def test_celsius_to_kelvin(self):
        from huginn.hooks.unit_check import convert
        k = convert(0.0, "celsius", "k", "temperature")
        assert k == pytest.approx(273.15, abs=0.01)

    def test_fs_to_ps(self):
        from huginn.hooks.unit_check import convert
        ps = convert(1000.0, "fs", "ps", "time")
        assert ps == pytest.approx(1.0, abs=1e-12)

    def test_to_si(self):
        from huginn.hooks.unit_check import to_si
        # 520 eV → joules
        j = to_si(520.0, "ev", "energy")
        assert j == pytest.approx(520 * 1.602e-19, rel=1e-3)

    def test_from_si(self):
        from huginn.hooks.unit_check import from_si
        # 1e-19 Joule → eV
        ev = from_si(1.602e-19, "ev", "energy")
        assert ev == pytest.approx(1.0, abs=0.01)

    def test_roundtrip(self):
        from huginn.hooks.unit_check import convert
        # 520 eV → joule → eV should be 520
        j = convert(520.0, "ev", "joule", "energy")
        ev = convert(j, "joule", "ev", "energy")
        assert ev == pytest.approx(520.0, abs=0.001)

    def test_unknown_quantity_raises(self):
        from huginn.hooks.unit_check import convert
        with pytest.raises(KeyError):
            convert(1.0, "x", "y", "nonexistent")


class TestPhysicalRange:
    def test_band_gap_in_range(self):
        from huginn.hooks.unit_check import check_value
        ok, msg = check_value("band_gap", 1.12)
        assert ok

    def test_band_gap_too_large(self):
        from huginn.hooks.unit_check import check_value
        ok, msg = check_value("band_gap", 50.0)
        assert not ok
        assert "range" in msg

    def test_lattice_constant_in_range(self):
        from huginn.hooks.unit_check import check_value
        ok, _ = check_value("lattice_constant", 5.43)
        assert ok

    def test_lattice_constant_too_small(self):
        from huginn.hooks.unit_check import check_value
        ok, msg = check_value("lattice_constant", 0.001)
        assert not ok
        assert "range" in msg

    def test_temperature_in_range(self):
        from huginn.hooks.unit_check import check_value
        ok, _ = check_value("temperature", 300.0)
        assert ok

    def test_temperature_too_high(self):
        from huginn.hooks.unit_check import check_value
        ok, msg = check_value("temperature", 99999.0)
        assert not ok

    def test_unknown_property_passes(self):
        from huginn.hooks.unit_check import check_value
        ok, _ = check_value("unknown_prop", 42.0)
        assert ok

    def test_alias_gap(self):
        from huginn.hooks.unit_check import check_value
        ok, _ = check_value("gap", 1.0)
        assert ok  # "gap" is an alias for "band_gap"


class TestDimensionalConsistencyHook:
    def _run_post(self, hm, tool_name, args, result):
        return asyncio.run(hm.run_post(tool_name, args, result, None, 0.0))

    def test_hook_warns_on_bad_value(self):
        from huginn.hooks import HookManager
        from huginn.hooks.unit_check import register_dimensional_hook

        hm = HookManager()
        register_dimensional_hook(hm)
        ctx = self._run_post(hm, "vasp_tool", {}, {
            "key_properties": {"band_gap": 50.0}
        })
        warnings = ctx.metadata.get("dimensional_warnings", [])
        assert len(warnings) >= 1
        assert "band_gap" in warnings[0]

    def test_hook_passes_good_values(self):
        from huginn.hooks import HookManager
        from huginn.hooks.unit_check import register_dimensional_hook

        hm = HookManager()
        register_dimensional_hook(hm)
        ctx = self._run_post(hm, "vasp_tool", {}, {
            "key_properties": {"band_gap": 1.12}
        })
        assert "dimensional_warnings" not in ctx.metadata

    def test_hook_handles_missing_key_properties(self):
        from huginn.hooks import HookManager
        from huginn.hooks.unit_check import register_dimensional_hook

        hm = HookManager()
        register_dimensional_hook(hm)
        ctx = self._run_post(hm, "vasp_tool", {}, {"data": "no props"})
        assert "dimensional_warnings" not in ctx.metadata

    def test_hook_handles_none_result(self):
        from huginn.hooks import HookManager
        from huginn.hooks.unit_check import register_dimensional_hook

        hm = HookManager()
        register_dimensional_hook(hm)
        ctx = self._run_post(hm, "vasp_tool", {}, None)
        assert "dimensional_warnings" not in ctx.metadata

    def test_hook_idempotent(self):
        from huginn.hooks import HookManager
        from huginn.hooks.unit_check import register_dimensional_hook

        hm = HookManager()
        register_dimensional_hook(hm)
        register_dimensional_hook(hm)
        assert len(hm._callbacks["post_tool_use"]) == 1


# ── UC-2: Event Sourcing API ────────────────────────────────────


class TestEventSourcing:
    def _get_registry_with_db(self, tmp_path):
        """Create a ProvenanceRegistry with a temporary SQLite DB."""
        # Use tmp_path for the DB to avoid Windows AV timeout
        db_path = str(tmp_path / "test_provenance.db")
        from huginn.provenance.registry import _ProvenanceStore
        store = _ProvenanceStore(db_path)
        return store

    def test_get_events_since(self, tmp_path):
        store = self._get_registry_with_db(tmp_path)
        import time

        from huginn.provenance.registry import ProvenanceEntry
        for i in range(5):
            store.save(ProvenanceEntry(
                file_path=f"/out_{i}.out", produced_by="vasp_tool",
                produced_at=time.time(), parameters={"i": i},
            ))
        events = store.get_events_since(0, 100)
        assert len(events) == 5
        # Should be in ascending id order
        paths = [e.file_path for e in events]
        assert paths == [f"/out_{i}.out" for i in range(5)]

    def test_get_events_by_tool(self, tmp_path):
        store = self._get_registry_with_db(tmp_path)
        import time

        from huginn.provenance.registry import ProvenanceEntry
        store.save(ProvenanceEntry(
            file_path="/a.out", produced_by="vasp_tool", produced_at=time.time(),
        ))
        store.save(ProvenanceEntry(
            file_path="/b.out", produced_by="lammps_tool", produced_at=time.time(),
        ))
        store.save(ProvenanceEntry(
            file_path="/c.out", produced_by="vasp_tool", produced_at=time.time(),
        ))
        events = store.get_events_by_tool("vasp_tool", 10)
        assert len(events) == 2
        assert all(e.produced_by == "vasp_tool" for e in events)

    def test_get_event_by_id(self, tmp_path):
        store = self._get_registry_with_db(tmp_path)
        import time

        from huginn.provenance.registry import ProvenanceEntry
        store.save(ProvenanceEntry(
            file_path="/first.out", produced_by="vasp_tool", produced_at=time.time(),
        ))
        store.save(ProvenanceEntry(
            file_path="/second.out", produced_by="qe_tool", produced_at=time.time(),
        ))
        event = store.get_event_by_id(1)
        assert event is not None
        assert event.file_path == "/first.out"
        event2 = store.get_event_by_id(2)
        assert event2 is not None
        assert event2.file_path == "/second.out"

    def test_get_max_id(self, tmp_path):
        store = self._get_registry_with_db(tmp_path)
        assert store.get_max_id() == 0  # empty
        import time

        from huginn.provenance.registry import ProvenanceEntry
        store.save(ProvenanceEntry(
            file_path="/x.out", produced_by="vasp_tool", produced_at=time.time(),
        ))
        assert store.get_max_id() == 1

    def test_get_events_since_with_offset(self, tmp_path):
        store = self._get_registry_with_db(tmp_path)
        import time

        from huginn.provenance.registry import ProvenanceEntry
        for i in range(5):
            store.save(ProvenanceEntry(
                file_path=f"/out_{i}.out", produced_by="vasp_tool",
                produced_at=time.time(),
            ))
        # Get events since id=2, should return 3 (ids 3, 4, 5)
        events = store.get_events_since(2, 100)
        assert len(events) == 3
        assert events[0].file_path == "/out_2.out"


class TestProvenanceRegistryEventSourcing:
    def test_get_events_memory_fallback(self):
        from huginn.provenance.registry import ProvenanceRegistry
        # Create with mocked store=None for memory-only mode
        reg = ProvenanceRegistry()
        reg._store = None
        reg._entries = []
        reg.register("/a.out", "vasp_tool", key_properties={"band_gap": 1.0})
        reg.register("/b.out", "qe_tool", key_properties={"band_gap": 0.5})
        events = reg.get_events()
        assert len(events) == 2
        vasp_events = reg.get_events(tool="vasp_tool")
        assert len(vasp_events) == 1

    def test_current_version_memory(self):
        from huginn.provenance.registry import ProvenanceRegistry
        reg = ProvenanceRegistry()
        reg._store = None
        reg._entries = []
        reg.register("/a.out", "vasp_tool")
        assert reg.current_version() == 1

    def test_replay_to(self, tmp_path):
        import time

        from huginn.provenance.registry import ProvenanceEntry, _ProvenanceStore
        store = _ProvenanceStore(str(tmp_path / "test.db"))
        for i in range(5):
            store.save(ProvenanceEntry(
                file_path=f"/out_{i}.out", produced_by="vasp_tool",
                produced_at=time.time(),
            ))
        # Replay to id 3 should return 4 events (ids 1,2,3,4)
        events = store.get_events_since(0, 4)
        assert len(events) == 4

    def test_rollback_returns_file_paths(self, tmp_path):
        import time

        from huginn.provenance.registry import ProvenanceEntry, _ProvenanceStore
        store = _ProvenanceStore(str(tmp_path / "test.db"))
        for i in range(5):
            store.save(ProvenanceEntry(
                file_path=f"/out_{i}.out", produced_by="vasp_tool",
                produced_at=time.time(),
            ))
        # Rollback to id 2: files produced after id 2 should be out_2, out_3, out_4
        # (ids 3,4,5 correspond to out_2, out_3, out_4)
        events_after = store.get_events_since(2, 100)
        paths = [e.file_path for e in reversed(events_after)]
        assert len(paths) == 3
        # Newest first
        assert paths[0] == "/out_4.out"


# ── T-BCSE-01: SessionEventLog (append-only event source of truth) ──


class TestSessionEventLog:
    def _log(self, tmp_path, name="s1"):
        from huginn.events.session_log import SessionEventLog
        return SessionEventLog(name, tmp_path / f"{name}.jsonl", load=False)

    def test_append_chains_and_advances_leaf(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE
        log = self._log(tmp_path)
        e1 = log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        e2 = log.append(EVENT_MESSAGE, {"role": "assistant", "content": "ok"})
        assert e1.parent_id is None
        assert e2.parent_id == e1.id
        assert log.leaf_id == e2.id
        assert log.seq == 2
        assert len(log) == 2

    def test_events_on_path_root_to_leaf(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE
        log = self._log(tmp_path)
        ids = [
            log.append(EVENT_MESSAGE, {"content": i}).id for i in range(3)
        ]
        assert [ev.id for ev in log.events_on_path()] == ids

    def test_branch_moves_leaf_without_deleting(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE
        log = self._log(tmp_path)
        e1 = log.append(EVENT_MESSAGE, {"content": "a"})
        e2 = log.append(EVENT_MESSAGE, {"content": "b"})
        log.branch(e1.seq)
        assert len(log) == 2  # history intact
        e3 = log.append(EVENT_MESSAGE, {"content": "c"})
        assert e3.parent_id == e1.id
        assert [ev.id for ev in log.events_on_path()] == [e1.id, e3.id]

    def test_read_after_global_incremental(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE
        log = self._log(tmp_path)
        e1 = log.append(EVENT_MESSAGE, {"content": "a"})
        log.append(EVENT_MESSAGE, {"content": "b"})
        log.append(EVENT_MESSAGE, {"content": "c"})
        after = log.read_after(e1.seq)
        assert len(after) == 2

    def test_persist_and_reopen(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE, SessionEventLog
        log = self._log(tmp_path, "persistent")
        log.append(EVENT_MESSAGE, {"role": "user", "content": "hello"})
        log.append(EVENT_MESSAGE, {"role": "assistant", "content": "world"})
        loaded = SessionEventLog.open("persistent", tmp_path / "persistent.jsonl")
        assert loaded.seq == 2
        path = loaded.events_on_path()
        assert path[-1].payload["content"] == "world"

    def test_unknown_kind_rejected(self, tmp_path):
        import pytest
        log = self._log(tmp_path)
        with pytest.raises(ValueError):
            log.append("nope", {})

    def test_reset_leaf_creates_new_root(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE
        log = self._log(tmp_path)
        e1 = log.append(EVENT_MESSAGE, {"content": "a"})
        log.reset_leaf()
        e2 = log.append(EVENT_MESSAGE, {"content": "b"})
        assert e2.parent_id is None
        assert [ev.id for ev in log.events_on_path()] == [e2.id]


# ── T-BCSE-02: ProjectionEngine (pure-function projections over the log) ──


class TestProjectionEngine:
    def _engine_and_log(self, tmp_path):
        from huginn.events.projection import (
            MessagePathProjection,
            ProjectionEngine,
            RuntimeStateProjection,
        )
        from huginn.events.session_log import SessionEventLog
        log = SessionEventLog("s1", tmp_path / "s1.jsonl", load=False)
        engine = ProjectionEngine()
        engine.register(RuntimeStateProjection())
        engine.register(MessagePathProjection())
        return engine, log

    def test_drive_updates_projections(self, tmp_path):
        from huginn.events.projection import MessagePathProjection
        from huginn.events.session_log import EVENT_MESSAGE
        engine, log = self._engine_and_log(tmp_path)
        log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        log.append(EVENT_MESSAGE, {"role": "assistant", "content": "ok"})
        log.append("cognitive_mode_change", {"cognitive_mode": "construct"})
        for ev in log:
            engine.drive(log, ev)
        runtime = engine.build(log, "runtime")
        assert runtime["cognitive_mode"] == "construct"
        assert runtime["turns_count"] == 1
        msgs = engine.build(log, MessagePathProjection.key)
        assert len(msgs) == 2

    def test_cold_rebuild_equals_driven(self, tmp_path):
        from huginn.events.projection import ProjectionEngine, RuntimeStateProjection
        from huginn.events.session_log import EVENT_MESSAGE
        engine, log = self._engine_and_log(tmp_path)
        log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        log.append("cognitive_mode_change", {"cognitive_mode": "construct"})
        for ev in log:
            engine.drive(log, ev)
        cold = ProjectionEngine()
        cold.register(RuntimeStateProjection())
        assert cold.build(log, "runtime") == engine.build(log, "runtime")

    def test_listener_fires_only_on_change(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE
        engine, log = self._engine_and_log(tmp_path)
        log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        for ev in log:
            engine.drive(log, ev)
        calls = []
        engine.subscribe(log, "runtime", lambda l, k, v, seq: calls.append(v["cognitive_mode"]))
        log.append("cognitive_mode_change", {"cognitive_mode": "construct"})  # change from default
        engine.drive(log, log.get(log.leaf_id))
        log.append("cognitive_mode_change", {"cognitive_mode": "construct"})  # same value
        engine.drive(log, log.get(log.leaf_id))
        assert calls == ["construct"]

    def test_compaction_kept_as_marker(self, tmp_path):
        from huginn.events.projection import MessagePathProjection
        from huginn.events.session_log import EVENT_COMPACTION, EVENT_MESSAGE
        engine, log = self._engine_and_log(tmp_path)
        log.append(EVENT_MESSAGE, {"role": "user", "content": "a"})
        log.append(EVENT_COMPACTION, {"summary": "earlier"})
        for ev in log:
            engine.drive(log, ev)
        msgs = engine.build(log, MessagePathProjection.key)
        assert msgs[-1]["role"] == "compaction"
        assert msgs[-1]["summary"] == "earlier"

    def test_state_version_bump_rebuilds(self, tmp_path):
        from huginn.events.projection import RuntimeStateProjection
        from huginn.events.session_log import EVENT_MESSAGE
        engine, log = self._engine_and_log(tmp_path)
        log.append(EVENT_MESSAGE, {"role": "user", "content": "hi"})
        for ev in log:
            engine.drive(log, ev)
        RuntimeStateProjection.stateVersion = 2
        try:
            engine.build(log, "runtime")
            assert engine._cells_for(log)["runtime"].version == 2
        finally:
            RuntimeStateProjection.stateVersion = 1


# ── T-BCSE-04: UiProjection (block structure for incremental frontend) ──


class TestUiProjection:
    def _build(self, tmp_path, *events):
        from huginn.events.projection import (
            ProjectionEngine,
            UiProjection,
        )
        from huginn.events.session_log import SessionEventLog
        log = SessionEventLog("s1", tmp_path / "s1.jsonl", load=False)
        engine = ProjectionEngine()
        engine.register(UiProjection())
        for ev in events:
            log.append(ev[0], ev[1])
        for ev in log:
            engine.drive(log, ev)
        return engine.build(log, "ui")

    def test_emits_frozen_blocks_and_dividers(self, tmp_path):
        from huginn.events.projection import (
            BLOCK_COMPACTION,
            BLOCK_TEXT,
            BLOCK_TOOL,
        )
        from huginn.events.session_log import (
            EVENT_BRANCH_SUMMARY,
            EVENT_COMPACTION,
            EVENT_MESSAGE,
        )
        blocks = self._build(
            tmp_path,
            (EVENT_MESSAGE, {"role": "user", "content": "hi"}),
            (EVENT_COMPACTION, {"summary": "compacted"}),
            (EVENT_BRANCH_SUMMARY, {"summary": "abandoned"}),
        )
        assert [b["kind"] for b in blocks] == [BLOCK_TEXT, BLOCK_COMPACTION, BLOCK_TOOL]
        assert all(b["frozen"] for b in blocks)
        assert blocks[0]["text"].startswith("**user:**")

    def test_metadata_events_do_not_affect_ui(self, tmp_path):
        from huginn.events.session_log import EVENT_MESSAGE
        blocks = self._build(
            tmp_path,
            (EVENT_MESSAGE, {"role": "user", "content": "hi"}),
            ("cognitive_mode_change", {"cognitive_mode": "construct"}),
        )
        assert len(blocks) == 1
