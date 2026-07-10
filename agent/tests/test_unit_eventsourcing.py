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
        import os, tempfile
        # Use tmp_path for the DB to avoid Windows AV timeout
        db_path = str(tmp_path / "test_provenance.db")
        from huginn.provenance.registry import _ProvenanceStore
        store = _ProvenanceStore(db_path)
        return store

    def test_get_events_since(self, tmp_path):
        store = self._get_registry_with_db(tmp_path)
        from huginn.provenance.registry import ProvenanceEntry
        import time
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
        from huginn.provenance.registry import ProvenanceEntry
        import time
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
        from huginn.provenance.registry import ProvenanceEntry
        import time
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
        from huginn.provenance.registry import ProvenanceEntry
        import time
        store.save(ProvenanceEntry(
            file_path="/x.out", produced_by="vasp_tool", produced_at=time.time(),
        ))
        assert store.get_max_id() == 1

    def test_get_events_since_with_offset(self, tmp_path):
        store = self._get_registry_with_db(tmp_path)
        from huginn.provenance.registry import ProvenanceEntry
        import time
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
        from huginn.provenance.registry import ProvenanceEntry, ProvenanceRegistry
        # Create with mocked store=None for memory-only mode
        reg = ProvenanceRegistry()
        reg._store = None
        reg._entries = []
        import time
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
        from huginn.provenance.registry import _ProvenanceStore, ProvenanceEntry
        import time
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
        from huginn.provenance.registry import _ProvenanceStore, ProvenanceEntry
        import time
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
