"""P0 tests for physical pre-check hooks — band/SCF, elastic/relax, MD timestep."""

from __future__ import annotations

from types import SimpleNamespace

from huginn.hooks import HookContext
from huginn.hooks.physical_precheck import (
    band_before_scf_hook,
    elastic_without_relax_hook,
    md_timestep_hook,
)


# ── provenance stub ─────────────────────────────────────────────────


class _FakeRegistry:
    """Minimal provenance registry stub — find_by_tool returns canned entries."""

    def __init__(self, entries_by_tool: dict | None = None) -> None:
        self._entries = entries_by_tool or {}

    def find_by_tool(self, tool_name: str) -> list:
        return self._entries.get(tool_name, [])


def _patch_provenance(monkeypatch, entries_by_tool: dict | None = None) -> None:
    """Swap ProvenanceRegistry.shared() so hooks see our fake entries."""
    fake = _FakeRegistry(entries_by_tool)
    # ponytail: plain lambda replaces the classmethod; when accessed on the
    # class Python returns the function itself, so shared() -> fake
    monkeypatch.setattr(
        "huginn.provenance.registry.ProvenanceRegistry.shared",
        lambda: fake,
    )


# ── band_before_scf ─────────────────────────────────────────────────


class TestBandBeforeScf:
    async def test_blocked_without_prior_scf(self, monkeypatch):
        _patch_provenance(monkeypatch, {})  # no prior SCF anywhere
        ctx = HookContext(tool_name="vasp_tool", args={"action": "band"})
        result = await band_before_scf_hook(ctx)
        assert result is not None
        assert result.blocked is True
        assert "force_proceed" in result.metadata["block_reason"]

    async def test_passes_with_prior_scf(self, monkeypatch):
        scf_entry = SimpleNamespace(parameters={"action": "static"})
        _patch_provenance(monkeypatch, {"vasp_tool": [scf_entry]})
        ctx = HookContext(tool_name="vasp_tool", args={"action": "band"})
        result = await band_before_scf_hook(ctx)
        assert result is None

    async def test_dos_action_also_checked(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(tool_name="qe_tool", args={"action": "dos"})
        result = await band_before_scf_hook(ctx)
        assert result is not None
        assert result.blocked is True

    async def test_relax_action_not_checked(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(tool_name="vasp_tool", args={"action": "relax"})
        result = await band_before_scf_hook(ctx)
        assert result is None

    async def test_force_proceed_bypasses(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(
            tool_name="vasp_tool",
            args={"action": "band", "force_proceed": True},
        )
        result = await band_before_scf_hook(ctx)
        assert result is None


# ── elastic_without_relax ───────────────────────────────────────────


class TestElasticWithoutRelax:
    async def test_blocked_without_relax(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(
            tool_name="mechanical_tool",
            args={"action": "elastic_constants"},
        )
        result = await elastic_without_relax_hook(ctx)
        assert result is not None
        assert result.blocked is True

    async def test_passes_with_prior_relax(self, monkeypatch):
        relax_entry = SimpleNamespace(parameters={"action": "relax"})
        _patch_provenance(monkeypatch, {"vasp_tool": [relax_entry]})
        ctx = HookContext(
            tool_name="mechanical_tool",
            args={"action": "elastic_constants"},
        )
        result = await elastic_without_relax_hook(ctx)
        assert result is None

    async def test_non_elastic_action_not_checked(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(
            tool_name="mechanical_tool",
            args={"action": "hardness"},
        )
        result = await elastic_without_relax_hook(ctx)
        assert result is None

    async def test_force_proceed_bypasses(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(
            tool_name="mechanical_tool",
            args={"action": "elastic", "force_proceed": True},
        )
        result = await elastic_without_relax_hook(ctx)
        assert result is None


# ── md_timestep ─────────────────────────────────────────────────────


class TestMdTimestep:
    async def test_blocked_for_large_timestep(self):
        ctx = HookContext(
            tool_name="lammps_tool",
            args={"timestep": 10.0},
        )
        result = await md_timestep_hook(ctx)
        assert result is not None
        assert result.blocked is True

    async def test_passes_for_reasonable_timestep(self):
        ctx = HookContext(
            tool_name="lammps_tool",
            args={"timestep": 1.0},
        )
        result = await md_timestep_hook(ctx)
        assert result is None

    async def test_blocked_with_dt_alias(self):
        ctx = HookContext(
            tool_name="gromacs_tool",
            args={"dt": 8.0},
        )
        result = await md_timestep_hook(ctx)
        assert result is not None
        assert result.blocked is True

    async def test_boundary_exactly_5fs_passes(self):
        ctx = HookContext(
            tool_name="lammps_tool",
            args={"timestep": 5.0},
        )
        result = await md_timestep_hook(ctx)
        assert result is None

    async def test_no_timestep_arg_passes(self):
        ctx = HookContext(
            tool_name="lammps_tool",
            args={"action": "minimize"},
        )
        result = await md_timestep_hook(ctx)
        assert result is None

    async def test_force_proceed_bypasses(self):
        ctx = HookContext(
            tool_name="lammps_tool",
            args={"timestep": 50.0, "force_proceed": True},
        )
        result = await md_timestep_hook(ctx)
        assert result is None


# ── non-sim tools are never intercepted ─────────────────────────────


class TestNonSimToolsNotIntercepted:
    async def test_web_search_not_blocked_by_any_hook(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(tool_name="web_search", args={"action": "search"})
        assert await band_before_scf_hook(ctx) is None
        assert await elastic_without_relax_hook(ctx) is None
        assert await md_timestep_hook(ctx) is None

    async def test_literature_tool_not_blocked(self, monkeypatch):
        _patch_provenance(monkeypatch, {})
        ctx = HookContext(tool_name="literature_tool", args={})
        assert await band_before_scf_hook(ctx) is None
        assert await elastic_without_relax_hook(ctx) is None
        assert await md_timestep_hook(ctx) is None
