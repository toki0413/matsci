"""Regression tests for ToolProfile metadata backfill (Commit A1).

Locks down behavior equivalence between the old hardcoded dispatch dicts
(HEAVY_TOOLS / LIGHT_TOOLS / PHASE_TOOLS / _TOOL_CONSTRAINT_SCOPES /
LIGHT_ALTERNATIVES) and the new declarative ToolProfile metadata on each
tool class. These snapshots must stay green as we migrate the dispatch
sites to read from tool metadata in Commits A2-A4.
"""

from __future__ import annotations

import pytest

from huginn.phases import (
    _CORE_TOOLS,
    PhaseManager,
    ResearchPhase,
)
from huginn.tools.registry import ToolRegistry


# ── Frozen snapshots of pre-migration dispatch values ──────────────────
# These capture the old hardcoded PHASE_TOOLS dict so the regression test
# compares metadata-derived values against the historical baseline, not
# against the live (now also derived) PHASE_TOOLS dict.
_FROZEN_PHASE_TOOLS: dict[ResearchPhase, set[str] | None] = {
    ResearchPhase.LITERATURE: _CORE_TOOLS | {
        "browser_tool",
        "database_tool",
        "materials_database_tool",
        "extract_tool",
        "symbolic_math_tool",
        "agentic_search_tool",
    },
    ResearchPhase.HYPOTHESIS: _CORE_TOOLS | {
        "symbolic_math_tool",
        "symbolic_regression_tool",
        "descriptor_tool",
        "database_tool",
        "materials_database_tool",
        "autodiff_tool",
    },
    ResearchPhase.PLANNING: _CORE_TOOLS | {
        "structure_tool",
        "symmetry_tool",
        "descriptor_tool",
        "symbolic_math_tool",
        "parameters",
        "packing_tool",
        "tda",
        "multi_fidelity_tool",
    },
    ResearchPhase.EXECUTION: _CORE_TOOLS | {
        "vasp_tool",
        "qe_tool",
        "cp2k_tool",
        "lammps_tool",
        "openfoam_tool",
        "comsol_tool",
        "abaqus_tool",
        "fenics_tool",
        "elmer_tool",
        "gromacs_tool",
        "job_tool",
        "orchestrate",
        "high_throughput_tool",
        "ml_potential_tool",
        "structure_tool",
        "packing_tool",
        "workflow_tool",
    },
    ResearchPhase.VALIDATION: _CORE_TOOLS | {
        "validate_tool",
        "uq_tool",
        "gp_tool",
        "symbolic_math_tool",
        "autodiff_tool",
        "diagnose_tool",
        "system_diagnostic_tool",
        "characterization_tool",
        "experimental_data_tool",
        "diff_tool",
        "active_learning_tool",
        "symbolic_regression_tool",
        "symmetry_tool",
        "evidence_fusion_tool",
        "tda",
        "xrd_sim_tool",
        "multi_fidelity_tool",
    },
    ResearchPhase.REPORTING: _CORE_TOOLS | {
        "report_tool",
        "visualize_tool",
        "diff_tool",
        "symbolic_math_tool",
        "extract_tool",
        "xrd_sim_tool",
    },
    ResearchPhase.OPEN: None,
}

_FROZEN_HEAVY_TOOLS: set[str] = {
    "vasp_tool",
    "qe_tool",
    "cp2k_tool",
    "lammps_tool",
    "abaqus_tool",
    "comsol_tool",
    "openfoam_tool",
    "ml_potential_tool",
    "gromacs_tool",
}

_FROZEN_LIGHT_TOOLS: set[str] = {
    "kb_tool",
    "web_search_tool",
    "rag_tool",
    "materials_database_tool",
    "structure_tool",
    "symbolic_math_tool",
    "numerical_tool",
    "symbolic_regression_tool",
    "local_structure_db",
    "xrd_sim_tool",
    "system_diagnostic_tool",
    "git_tool",
    "github_tool",
    "multi_edit_tool",
    "onboarding_tool",
    "phase_tool",
    "workflow_tool",
    "multi_fidelity_tool",
}

_FROZEN_LIGHT_ALTERNATIVES: dict[str, list[str]] = {
    "vasp_tool": [
        "materials_database_tool",
        "local_structure_db",
        "symbolic_math_tool",
        "numerical_tool",
    ],
    "qe_tool": [
        "materials_database_tool",
        "local_structure_db",
        "symbolic_math_tool",
    ],
    "cp2k_tool": [
        "materials_database_tool",
        "local_structure_db",
        "symbolic_math_tool",
    ],
    "lammps_tool": [
        "symbolic_math_tool",
        "numerical_tool",
    ],
    "abaqus_tool": [
        "symbolic_math_tool",
        "numerical_tool",
    ],
    "comsol_tool": [
        "symbolic_math_tool",
        "numerical_tool",
    ],
    "openfoam_tool": [
        "symbolic_math_tool",
        "numerical_tool",
    ],
    "ml_potential_tool": [
        "materials_database_tool",
        "numerical_tool",
    ],
    "gromacs_tool": [
        "symbolic_math_tool",
        "numerical_tool",
    ],
}

_FROZEN_CONSTRAINT_SCOPES: dict[str, str] = {
    "vasp_tool": "dft",
    "qe_tool": "dft",
    "cp2k_tool": "dft",
    "lammps_tool": "md",
    "openfoam_tool": "cfd",
    "comsol_tool": "fea",
    "abaqus_tool": "fea",
    "structural_analytical_tool": "fea",
    "specialty_analysis_tool": "fea",
    "fem_tool": "fea",
    "fenics_tool": "fem",
    "elmer_tool": "fem",
    "gromacs_tool": "md",
}


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def _registered_tools():
    """Ensure all tools are registered before tests run."""
    from huginn.tools import register_all_tools

    if not ToolRegistry.list_tools():
        register_all_tools()
    yield


def _registered(name: str) -> bool:
    """True if a tool with *name* is currently registered (deps present)."""
    return ToolRegistry.get(name) is not None


# ── Heavy / light cost tier ────────────────────────────────────────────


class TestCostTier:
    """cost_tier metadata must match old HEAVY_TOOLS / LIGHT_TOOLS."""

    def test_heavy_tools_match(self):
        old_heavy = {n for n in _FROZEN_HEAVY_TOOLS if _registered(n)}
        derived_heavy = {
            t.name for t in ToolRegistry._tools.values() if t.cost_tier == "heavy"
        }
        assert derived_heavy == old_heavy, (
            f"heavy mismatch: extra={derived_heavy - old_heavy}, "
            f"missing={old_heavy - derived_heavy}"
        )

    def test_light_tools_match(self):
        old_light = {n for n in _FROZEN_LIGHT_TOOLS if _registered(n)}
        derived_light = {
            t.name for t in ToolRegistry._tools.values() if t.cost_tier == "light"
        }
        assert derived_light == old_light, (
            f"light mismatch: extra={derived_light - old_light}, "
            f"missing={old_light - derived_light}"
        )

    def test_mlp_heavy_actions(self):
        """ml_potential_tool should gate only train/fit/training actions."""
        tool = ToolRegistry.get("ml_potential_tool")
        if tool is None:
            pytest.skip("ml_potential_tool not registered")
        assert tool.heavy_actions == frozenset({"train", "fit", "training"})


# ── Constraint scope ───────────────────────────────────────────────────


class TestConstraintScope:
    """constraint_scope metadata must match old _TOOL_CONSTRAINT_SCOPES."""

    def test_scopes_match(self):
        old_scopes = {
            n: s for n, s in _FROZEN_CONSTRAINT_SCOPES.items() if _registered(n)
        }
        derived_scopes = {
            t.name: t.constraint_scope
            for t in ToolRegistry._tools.values()
            if t.constraint_scope is not None
        }
        assert derived_scopes == old_scopes, (
            f"scope mismatch: extra={set(derived_scopes) - set(old_scopes)}, "
            f"missing={set(old_scopes) - set(derived_scopes)}"
        )


# ── Light alternatives ─────────────────────────────────────────────────


class TestLightAlternatives:
    """light_alternatives must match old LIGHT_ALTERNATIVES."""

    def test_alternatives_match(self):
        for heavy_name, expected_alts in _FROZEN_LIGHT_ALTERNATIVES.items():
            tool = ToolRegistry.get(heavy_name)
            if tool is None:
                continue  # skip dead references (e.g. cp2k_tool)
            assert tuple(expected_alts) == tool.light_alternatives, (
                f"{heavy_name}: expected {expected_alts}, "
                f"got {tool.light_alternatives}"
            )


# ── Phase derivation ───────────────────────────────────────────────────


class TestPhaseDerivation:
    """Derived phase tool sets must match old hardcoded PHASE_TOOLS."""

    @staticmethod
    def _derived_for_phase(phase: ResearchPhase) -> set[str]:
        """Compute the tool set for *phase* from ToolProfile metadata."""
        derived = set()
        for tool in ToolRegistry._tools.values():
            phases = tool.phases
            if phases is None or phase in phases:
                derived.add(tool.name)
        return _CORE_TOOLS | derived

    @pytest.mark.parametrize(
        "phase",
        [
            ResearchPhase.LITERATURE,
            ResearchPhase.HYPOTHESIS,
            ResearchPhase.PLANNING,
            ResearchPhase.EXECUTION,
            ResearchPhase.VALIDATION,
            ResearchPhase.REPORTING,
        ],
    )
    def test_phase_tools_match(self, phase: ResearchPhase):
        old_set = _FROZEN_PHASE_TOOLS[phase]
        if old_set is None:
            pytest.skip("OPEN phase has no filter")
        derived = self._derived_for_phase(phase)
        # Old set may contain dead references (e.g. "parameters") that don't
        # correspond to any registered tool. Filter those out before comparing.
        old_live = {n for n in old_set if _registered(n) or n in _CORE_TOOLS}
        assert derived == old_live, (
            f"{phase.value}: extra={derived - old_live}, "
            f"missing={old_live - derived}"
        )

    def test_open_phase_returns_none(self):
        """OPEN phase must return None (all tools) from tool_filter."""
        mgr = PhaseManager(initial=ResearchPhase.OPEN)
        assert mgr.tool_filter() is None
