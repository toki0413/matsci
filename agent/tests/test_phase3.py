"""Phase 3 tests — merged from test_phase3_compat / database_tool / diff_tool / features.

Covers:
- compat.py analysis and visualization endpoints (crystal system, ASCII plot/phase)
- database_tool (real API calls and mock fallback)
- diff_tool (MathDiffer integration and deep diff fallback)
- phase3 features (report comparison, symbolic regression compare,
  workflow custom validation, and exploration query)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from huginn.core_types import ToolContext
from huginn.exploration.core import Branch, BranchStatus, Decision, ExplorationSpace
from huginn.routes.compat import (
    _ascii_phase,
    _ascii_plot,
    _crystal_system,
    _guess_crystal_system,
)
from huginn.tools.database_tool import DatabaseTool, DatabaseToolInput
from huginn.tools.diff_tool import DiffTool, DiffToolInput, _deep_diff
from huginn.tools.report_tool import ReportComparator, ReportTool, ReportToolInput
from huginn.tools.sci.symbolic_regression_tool import (
    SymbolicRegressionInput,
    SymbolicRegressionTool,
)
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.stages import ValidationRule


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


# ═══════════════════════════════════════════════════════════════════
# compat — Crystal system helpers
# ═══════════════════════════════════════════════════════════════════


class TestCrystalSystem:
    def test_triclinic(self):
        assert _crystal_system(1) == "triclinic"
        assert _crystal_system(2) == "triclinic"

    def test_monoclinic(self):
        assert _crystal_system(3) == "monoclinic"
        assert _crystal_system(15) == "monoclinic"

    def test_orthorhombic(self):
        assert _crystal_system(16) == "orthorhombic"
        assert _crystal_system(74) == "orthorhombic"

    def test_tetragonal(self):
        assert _crystal_system(75) == "tetragonal"
        assert _crystal_system(142) == "tetragonal"

    def test_trigonal(self):
        assert _crystal_system(143) == "trigonal"
        assert _crystal_system(167) == "trigonal"

    def test_hexagonal(self):
        assert _crystal_system(168) == "hexagonal"
        assert _crystal_system(194) == "hexagonal"

    def test_cubic(self):
        assert _crystal_system(195) == "cubic"
        assert _crystal_system(230) == "cubic"


class TestGuessCrystalSystem:
    def test_cubic(self):
        norms = np.array([4.0, 4.0, 4.0])
        angles = [90.0, 90.0, 90.0]
        assert _guess_crystal_system(norms, angles) == "cubic"

    def test_tetragonal(self):
        norms = np.array([4.0, 4.0, 6.0])
        angles = [90.0, 90.0, 90.0]
        assert _guess_crystal_system(norms, angles) == "tetragonal"

    def test_orthorhombic(self):
        norms = np.array([3.0, 4.0, 5.0])
        angles = [90.0, 90.0, 90.0]
        assert _guess_crystal_system(norms, angles) == "orthorhombic"

    def test_hexagonal(self):
        norms = np.array([3.0, 3.0, 5.0])
        angles = [90.0, 90.0, 120.0]
        assert _guess_crystal_system(norms, angles) == "hexagonal"

    def test_triclinic_fallback(self):
        norms = np.array([3.0, 4.0, 5.0])
        angles = [80.0, 85.0, 95.0]
        assert _guess_crystal_system(norms, angles) == "triclinic"


# ═══════════════════════════════════════════════════════════════════
# compat — ASCII plot helpers
# ═══════════════════════════════════════════════════════════════════


class TestAsciiPlot:
    def test_basic_plot(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        result = _ascii_plot(x, y, "sin")
        assert "sin" in result
        assert len(result) > 0

    def test_empty_data(self):
        result = _ascii_plot(np.array([]), np.array([]), "test")
        assert result == "(empty data)"


class TestAsciiPhase:
    def test_basic_phase(self):
        t = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)
        result = _ascii_phase(x, y)
        assert "." in result  # should have some points

    def test_empty_phase(self):
        result = _ascii_phase(np.array([]), np.array([]))
        assert result == "(empty data)"


# ═══════════════════════════════════════════════════════════════════
# compat — Compat API endpoint tests (via TestClient)
# ═══════════════════════════════════════════════════════════════════


class TestCompatEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from huginn.routes.compat import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_firewall_status(self, client):
        resp = client.get("/firewall/status")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    # ── Symmetry ──

    def test_symmetry_missing_params(self, client):
        resp = client.post("/analyze/symmetry", json={})
        assert "error" in resp.json()

    def test_symmetry_fallback(self, client):
        # Cubic cell: a=b=c=4, all 90°
        lattice = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
        positions = [[0, 0, 0]]
        numbers = [14]
        resp = client.post("/analyze/symmetry", json={
            "lattice": lattice,
            "positions": positions,
            "numbers": numbers,
        })
        data = resp.json()
        assert "crystal_system" in data
        assert data["crystal_system"] == "cubic"

    # ── Spectral ──

    def test_spectral_missing_data(self, client):
        resp = client.post("/analyze/spectral", json={})
        assert "error" in resp.json()

    def test_spectral_basic(self, client):
        # Sine wave at 5 Hz
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t).tolist()
        resp = client.post("/analyze/spectral", json={
            "data": signal,
            "sample_rate": 100.0,
        })
        data = resp.json()
        assert data["n_samples"] == 100
        assert len(data["peaks"]) > 0
        # Dominant frequency should be ~5 Hz
        assert abs(data["dominant_frequency"] - 5.0) < 2.0

    # ── Dynamics ──

    def test_dynamics_missing_positions(self, client):
        resp = client.post("/analyze/dynamics", json={})
        assert "error" in resp.json()

    def test_dynamics_basic(self, client):
        # 3 frames, 2 atoms, 3D
        positions = [
            [[0, 0, 0], [1, 0, 0]],
            [[0.1, 0, 0], [1.1, 0, 0]],
            [[0.2, 0, 0], [1.2, 0, 0]],
        ]
        resp = client.post("/analyze/dynamics", json={
            "positions": positions,
            "timestep": 1.0,
        })
        data = resp.json()
        assert data["n_frames"] == 3
        assert data["n_atoms"] == 2
        assert "msd" in data
        assert len(data["msd"]) == 3
        assert data["msd"][0] == 0.0  # first frame = no displacement

    # ── TDA ──

    def test_tda_missing_data(self, client):
        resp = client.post("/analyze/tda", json={})
        assert "error" in resp.json()

    def test_tda_point_cloud(self, client):
        # Random point cloud
        np.random.seed(42)
        points = np.random.randn(20, 3).tolist()
        resp = client.post("/analyze/tda", json={"data": points})
        data = resp.json()
        assert data["n_points"] == 20
        assert data["embedding_dimension"] == 3

    # ── SINDy ──

    def test_sindy_missing_data(self, client):
        resp = client.post("/analyze/sindy", json={})
        assert "error" in resp.json()

    def test_sindy_linear_system(self, client):
        # dx/dt = -x (exponential decay)
        t = np.linspace(0, 5, 100)
        x = np.exp(-t)
        data = x.reshape(-1, 1).tolist()
        resp = client.post("/analyze/sindy", json={
            "data": data,
            "time": t.tolist(),
            "threshold": 0.01,
            "library": "polynomial",
        })
        result = resp.json()
        assert result["n_samples"] == 100
        assert result["n_variables"] == 1
        assert len(result["discovered_equations"]) == 1

    # ── Visualization ──

    def test_viz_dos_no_data(self, client):
        resp = client.post("/viz/dos", json={})
        assert resp.json()["fallback"] is True

    def test_viz_dos_with_data(self, client):
        energies = np.linspace(-10, 10, 50).tolist()
        dos = np.exp(-np.linspace(-10, 10, 50) ** 2).tolist()
        resp = client.post("/viz/dos", json={
            "energies": energies,
            "dos": dos,
            "fermi_level": 0.0,
        })
        data = resp.json()
        assert data["fallback"] is False
        assert "total_states" in data

    def test_viz_phase_no_data(self, client):
        resp = client.post("/viz/phase", json={})
        assert resp.json()["fallback"] is True

    def test_viz_phase_with_data(self, client):
        t = np.linspace(0, 2 * np.pi, 50)
        resp = client.post("/viz/phase", json={
            "x": np.cos(t).tolist(),
            "y": np.sin(t).tolist(),
        })
        data = resp.json()
        assert data["fallback"] is False
        assert data["n_points"] == 50

    def test_viz_persistence_no_data(self, client):
        resp = client.post("/viz/persistence", json={})
        assert resp.json()["fallback"] is True

    def test_viz_persistence_with_data(self, client):
        diagram = [
            {"dimension": 0, "birth": 0.0, "death": 0.5},
            {"dimension": 0, "birth": 0.1, "death": 0.8},
            {"dimension": 1, "birth": 0.3, "death": 1.0},
        ]
        resp = client.post("/viz/persistence", json={"diagram": diagram})
        data = resp.json()
        assert data["fallback"] is False
        assert data["n_features"] == 3

    def test_viz_sindy_no_data(self, client):
        resp = client.post("/viz/sindy", json={})
        assert resp.json()["fallback"] is True

    def test_viz_sindy_with_data(self, client):
        resp = client.post("/viz/sindy", json={
            "equations": ["-1.0000*x0", "0.5000*x1"],
        })
        data = resp.json()
        assert data["fallback"] is False
        assert data["n_equations"] == 2


# ═══════════════════════════════════════════════════════════════════
# database_tool — mock fallback when no API keys
# ═══════════════════════════════════════════════════════════════════


class TestDatabaseToolMockFallback:
    """Test mock fallback when no API keys are set."""

    def setup_method(self):
        self.tool = DatabaseTool()
        self.ctx = _ctx()

    @pytest.mark.asyncio
    async def test_mp_search_no_key(self):
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="SiO2",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["mock"] is True
        assert "MP_API_KEY" in result.data["reason"]

    @pytest.mark.asyncio
    async def test_mp_search_missing_formula(self):
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
        )
        # Without API key, returns mock (mock doesn't check formula)
        # But with API key, should fail
        args_with_key = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            api_key="test_key",
        )
        result = await self.tool.call(args_with_key, self.ctx)
        assert not result.success
        assert "formula" in result.error

    @pytest.mark.asyncio
    async def test_mp_get_structure_no_key(self):
        args = DatabaseToolInput(
            database="materials_project",
            query_type="get_structure",
            material_id="mp-149",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["mock"] is True

    @pytest.mark.asyncio
    async def test_mp_get_properties_no_key(self):
        args = DatabaseToolInput(
            database="materials_project",
            query_type="get_properties",
            material_id="mp-149",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["mock"] is True
        assert "properties" in result.data

    @pytest.mark.asyncio
    async def test_oqmd_search_no_key(self):
        args = DatabaseToolInput(
            database="oqmd",
            query_type="search",
            formula="Fe2O3",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["mock"] is True

    @pytest.mark.asyncio
    async def test_aflow_no_key(self):
        # AFLOW is a public API — no key needed. Mock aiohttp so we test
        # the real path without hitting the network.
        args = DatabaseToolInput(
            database="aflow",
            query_type="search",
            formula="Si",
        )
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[
            {"compound": "Si", "auid": "aflow:1", "Egap": 1.1}
        ])
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["database"] == "aflow"

    @pytest.mark.asyncio
    async def test_nomad_no_key(self):
        # NOMAD public data — no key needed. Mock aiohttp for offline test.
        args = DatabaseToolInput(
            database="nomad",
            query_type="search",
            formula="TiO2",
        )
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "data": [{
                "entry_id": "nomad-1",
                "results": {
                    "material": [{"chemical_formula_descriptive": "TiO2"}],
                    "properties": {},
                },
            }]
        })
        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))
        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["database"] == "nomad"

    @pytest.mark.asyncio
    async def test_compare_no_keys(self):
        args = DatabaseToolInput(
            database="materials_project",
            query_type="compare",
            formula="SiO2",
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "No database API keys" in result.error

    @pytest.mark.asyncio
    async def test_compare_missing_formula(self):
        args = DatabaseToolInput(
            database="materials_project",
            query_type="compare",
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "formula or material_id" in result.error

    def test_read_only(self):
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="Si",
        )
        assert self.tool.is_read_only(args) is True


class TestDatabaseToolApiKeyResolution:
    def test_env_fallback(self):
        tool = DatabaseTool()
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="Si",
        )
        with patch.dict("os.environ", {"MP_API_KEY": "env_key_123"}):
            key = tool._get_api_key(args)
            assert key == "env_key_123"

    def test_explicit_key_override(self):
        tool = DatabaseTool()
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="Si",
            api_key="explicit_key",
        )
        with patch.dict("os.environ", {"MP_API_KEY": "env_key"}):
            key = tool._get_api_key(args)
            assert key == "explicit_key"

    def test_no_key_returns_none(self):
        tool = DatabaseTool()
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="Si",
        )
        with patch.dict("os.environ", {}, clear=True):
            key = tool._get_api_key(args)
            assert key is None


class TestDatabaseToolWithMockedHttp:
    """Test real API call paths with mocked aiohttp."""

    @pytest.mark.asyncio
    async def test_mp_search_with_key(self):
        tool = DatabaseTool()
        ctx = _ctx()
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="SiO2",
            api_key="test_key",
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={
            "data": [{
                "material_id": "mp-149",
                "formula_pretty": "SiO2",
                "energy_per_atom": -5.0,
                "band_gap": 9.0,
                "symmetry": {"symbol": "P3_221"},
            }],
        })

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await tool.call(args, ctx)
            assert result.success
            assert result.data["count"] == 1
            assert result.data["records"][0]["material_id"] == "mp-149"

    @pytest.mark.asyncio
    async def test_mp_api_error(self):
        tool = DatabaseTool()
        ctx = _ctx()
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="SiO2",
            api_key="bad_key",
        )

        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.text = AsyncMock(return_value="Forbidden")

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await tool.call(args, ctx)
            assert not result.success
            assert "403" in result.error

    @pytest.mark.asyncio
    async def test_mp_material_id_search(self):
        tool = DatabaseTool()
        ctx = _ctx()
        args = DatabaseToolInput(
            database="materials_project",
            query_type="search",
            formula="mp-149",
            api_key="test_key",
        )

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"data": []})

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        )):
            result = await tool.call(args, ctx)
            assert result.success


# ═══════════════════════════════════════════════════════════════════
# diff_tool — _deep_diff unit tests
# ═══════════════════════════════════════════════════════════════════


class TestDeepDiff:
    def test_identical_dicts(self):
        assert _deep_diff({"a": 1}, {"a": 1}) == []

    def test_value_changed(self):
        changes = _deep_diff({"a": 1}, {"a": 2})
        assert len(changes) == 1
        assert changes[0]["type"] == "value_changed"
        assert changes[0]["old_value"] == 1
        assert changes[0]["new_value"] == 2

    def test_key_added(self):
        changes = _deep_diff({}, {"a": 1})
        assert len(changes) == 1
        assert changes[0]["type"] == "added"

    def test_key_removed(self):
        changes = _deep_diff({"a": 1}, {})
        assert len(changes) == 1
        assert changes[0]["type"] == "removed"
        assert changes[0]["severity"] == "warning"

    def test_nested_diff(self):
        old = {"params": {"encut": 400}}
        new = {"params": {"encut": 520}}
        changes = _deep_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["path"] == "params.encut"

    def test_list_length_change(self):
        changes = _deep_diff([1, 2], [1, 2, 3])
        assert any(c["type"] == "list_length_changed" for c in changes)

    def test_list_element_change(self):
        changes = _deep_diff([1, 2, 3], [1, 9, 3])
        assert any(
            c["type"] == "value_changed" and "[1]" in c["path"]
            for c in changes
        )

    def test_critical_severity_for_equations(self):
        changes = _deep_diff(
            {"equation": "x+y"},
            {"equation": "x+z"},
        )
        assert changes[0]["severity"] == "critical"

    def test_critical_severity_for_boundary(self):
        changes = _deep_diff(
            {"boundary_conditions": "periodic"},
            {"boundary_conditions": "fixed"},
        )
        assert changes[0]["severity"] == "critical"

    def test_critical_severity_for_symmetry(self):
        changes = _deep_diff({"symmetry": "Fm-3m"}, {"symmetry": "Pm-3m"})
        assert changes[0]["severity"] == "critical"


# ── diff_tool — DiffTool tests ──────────────────────────────────────


class TestDiffTool:
    def setup_method(self):
        self.tool = DiffTool()
        self.ctx = _ctx()

    @pytest.mark.asyncio
    async def test_inline_full_diff(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="full",
            inline_a={"encut": 400, "kpoints": [4, 4, 4]},
            inline_b={"encut": 520, "kpoints": [6, 6, 6]},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["summary"]["total_changes"] > 0
        assert result.data["engine"] in ("math_differ", "builtin_deep_diff")

    @pytest.mark.asyncio
    async def test_identical_inline(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="full",
            inline_a={"x": 1},
            inline_b={"x": 1},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["summary"]["total_changes"] == 0
        assert "No differences" in result.data["semantic_summary"]

    @pytest.mark.asyncio
    async def test_parameters_filter(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="parameters",
            inline_a={"encut": 400, "energy": -100},
            inline_b={"encut": 520, "energy": -102},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        # Should only include parameter changes (encut matches "cutoff")
        paths = [c["path"] for c in result.data["changes"]]
        assert any("encut" in p for p in paths)

    @pytest.mark.asyncio
    async def test_results_filter(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="results",
            inline_a={"encut": 400, "energy": -100},
            inline_b={"encut": 520, "energy": -102},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        paths = [c["path"] for c in result.data["changes"]]
        assert any("energy" in p for p in paths)

    @pytest.mark.asyncio
    async def test_file_path_loading(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"param": 1}, f)
            f.flush()
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"param": 2}, f)
            f.flush()
            path_b = f.name

        args = DiffToolInput(calc_a=path_a, calc_b=path_b, comparison_type="full")
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["summary"]["total_changes"] > 0

        Path(path_a).unlink()
        Path(path_b).unlink()

    @pytest.mark.asyncio
    async def test_plain_identifier_fallback(self):
        args = DiffToolInput(
            calc_a="run_001", calc_b="run_002",
            comparison_type="full",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_read_only(self):
        args = DiffToolInput(calc_a="a", calc_b="b", comparison_type="full")
        assert self.tool.is_read_only(args) is True

    @pytest.mark.asyncio
    async def test_critical_summary(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="full",
            inline_a={"equation": "x+y"},
            inline_b={"equation": "x+z"},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert "critical" in result.data["semantic_summary"].lower()


# ═══════════════════════════════════════════════════════════════════
# features — Report Comparison Tests
# ═══════════════════════════════════════════════════════════════════


class TestReportComparator:
    def test_two_dataset_comparison(self):
        comp = ReportComparator(["run_A", "run_B"])
        datasets = {
            "run_A": {
                "methods": {"functional": "PBE", "encut": 400},
                "results": {"energy": -100.0},
            },
            "run_B": {
                "methods": {"functional": "PBE", "encut": 520},
                "results": {"energy": -102.5},
            },
        }
        report = comp.generate(datasets)
        assert "Comparison Report" in report
        assert "Methods Comparison" in report
        assert "Results Comparison" in report
        assert "run_A" in report
        assert "run_B" in report

    def test_three_dataset_comparison(self):
        comp = ReportComparator(["A", "B", "C"])
        datasets = {
            "A": {"methods": {"encut": 300}, "results": {}},
            "B": {"methods": {"encut": 400}, "results": {}},
            "C": {"methods": {"encut": 500}, "results": {}},
        }
        report = comp.generate(datasets)
        assert "A" in report and "B" in report and "C" in report

    def test_convergence_comparison(self):
        comp = ReportComparator(["run1", "run2"])
        datasets = {
            "run1": {
                "convergence": {"energy": -100, "n_iterations": 20, "converged": True},
                "results": {},
            },
            "run2": {
                "convergence": {"energy": -102, "n_iterations": 25, "converged": True},
                "results": {},
            },
        }
        report = comp.generate(datasets)
        assert "Convergence Comparison" in report


class TestReportToolCompare:
    def setup_method(self):
        self.tool = ReportTool()
        self.ctx = _ctx()

    @pytest.mark.asyncio
    async def test_compare_no_datasets(self):
        args = ReportToolInput(action="compare")
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "comparison_datasets" in result.error

    @pytest.mark.asyncio
    async def test_compare_one_dataset(self):
        args = ReportToolInput(
            action="compare",
            comparison_datasets={"only_one": {"methods": {}}},
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "2 datasets" in result.error

    @pytest.mark.asyncio
    async def test_compare_two_datasets(self):
        args = ReportToolInput(
            action="compare",
            comparison_datasets={
                "calc_A": {"methods": {"encut": 400}, "results": {"energy": -100}},
                "calc_B": {"methods": {"encut": 520}, "results": {"energy": -102}},
            },
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert "report" in result.data

    @pytest.mark.asyncio
    async def test_compare_from_workflow_results(self):
        args = ReportToolInput(
            action="compare",
            workflow_results={
                "comparison_datasets": {
                    "A": {"methods": {}, "results": {}},
                    "B": {"methods": {}, "results": {}},
                }
            },
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success


# ═══════════════════════════════════════════════════════════════════
# features — Symbolic Regression Compare Tests
# ═══════════════════════════════════════════════════════════════════


class TestSymRegCompare:
    def setup_method(self):
        self.tool = SymbolicRegressionTool()
        self.ctx = _ctx()

    @pytest.mark.asyncio
    async def test_compare_no_expression(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1, 2, 3], "y": [1, 4, 9]},
            target_column="y",
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "probe_expression" in result.error

    @pytest.mark.asyncio
    async def test_compare_single_expression(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1, 2, 3, 4, 5], "y": [1, 4, 9, 16, 25]},
            target_column="y",
            probe_expression="x**2",
        )
        result = await self.tool.call(args, self.ctx)
        assert not result.success
        assert "2 expressions" in result.error

    @pytest.mark.asyncio
    async def test_compare_two_expressions(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 4.0, 9.0, 16.0, 25.0]},
            target_column="y",
            probe_expression="x**2; x**2 + x",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["expressions_evaluated"] == 2
        assert result.data["best_expression"] is not None
        # x**2 should have perfect R² for y = x²
        assert result.data["best_r2"] > 0.99

    @pytest.mark.asyncio
    async def test_compare_ranking(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]},
            target_column="y",
            probe_expression="2*x; x; x**2",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        # 2*x should be rank 1 (perfect for y=2x)
        comparison = result.data["comparison"]
        rank1 = [c for c in comparison if c.get("rank") == 1][0]
        assert rank1["expression"] == "2*x"

    @pytest.mark.asyncio
    async def test_compare_with_error(self):
        args = SymbolicRegressionInput(
            action="compare",
            data_json={"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]},
            target_column="y",
            probe_expression="x; forbidden_func(x)",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        comparison = result.data["comparison"]
        error_entries = [c for c in comparison if "error" in c]
        assert len(error_entries) == 1


# ═══════════════════════════════════════════════════════════════════
# features — Workflow Custom Validation Tests
# ═══════════════════════════════════════════════════════════════════


class TestWorkflowCustomValidation:
    def setup_method(self):
        self.engine = WorkflowEngine(tool_registry=None)

    def test_builtin_convergence_pass(self):
        rule = ValidationRule(check="convergence")
        assert self.engine._validate({"converged": True}, rule) is True

    def test_builtin_convergence_fail(self):
        rule = ValidationRule(check="convergence")
        assert self.engine._validate({"converged": False}, rule) is False

    def test_builtin_energy_sign(self):
        rule = ValidationRule(check="energy_sign")
        assert self.engine._validate({"energy": -100}, rule) is True
        assert self.engine._validate({"energy": 50}, rule) is False

    def test_builtin_force_threshold(self):
        rule = ValidationRule(check="force_threshold", threshold=0.05)
        assert self.engine._validate({"max_force": 0.01}, rule) is True
        assert self.engine._validate({"max_force": 0.1}, rule) is False

    def test_custom_registered_validator(self):
        def check_positive(data):
            return data.get("value", 0) > 0

        self.engine.register_validator("check_positive", check_positive)
        rule = ValidationRule(check="custom", custom_fn="check_positive")
        assert self.engine._validate({"value": 5}, rule) is True
        assert self.engine._validate({"value": -1}, rule) is False

    def test_custom_with_threshold(self):
        def check_range(data, threshold=1.0):
            return abs(data.get("error", 999)) < threshold

        self.engine.register_validator("check_range", check_range)
        rule = ValidationRule(check="custom", custom_fn="check_range", threshold=0.1)
        assert self.engine._validate({"error": 0.05}, rule) is True
        assert self.engine._validate({"error": 0.5}, rule) is False

    def test_custom_unknown_validator_fails_closed(self):
        rule = ValidationRule(check="custom", custom_fn="nonexistent_fn")
        assert self.engine._validate({"data": 1}, rule) is False

    def test_custom_exception_returns_false(self):
        def bad_fn(data):
            raise ValueError("boom")

        self.engine.register_validator("bad_fn", bad_fn)
        rule = ValidationRule(check="custom", custom_fn="bad_fn")
        assert self.engine._validate({}, rule) is False

    def test_resolve_dotted_path(self):
        # Test that _resolve_custom_fn handles non-existent modules gracefully
        result = WorkflowEngine._resolve_custom_fn("nonexistent.module.fn")
        assert result is None

    def test_resolve_simple_name(self):
        result = WorkflowEngine._resolve_custom_fn("simple_name")
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# features — Exploration Query Tests
# ═══════════════════════════════════════════════════════════════════


def _make_space() -> ExplorationSpace:
    """Create a test exploration space with branches."""
    space = ExplorationSpace(
        id="test-space",
        name="Test Space",
        objective="Find best material",
    )
    space.objectives_config = {"energy": "minimize", "band_gap": "maximize"}

    b1 = Branch(id="b1", name="PBE-Si", hypothesis="PBE functional for Si")
    b1.status = BranchStatus.COMPLETED
    b1.objectives = {"energy": -100.0, "band_gap": 1.1}
    b1.decisions = [
        Decision(
            id="d1", description="Choose functional",
            decision_type="categorical",
            chosen_option="PBE", available_options=["PBE", "HSE06"],
            rationale="Fast and reliable",
        ),
    ]

    b2 = Branch(id="b2", name="HSE06-Si", hypothesis="HSE06 for Si")
    b2.status = BranchStatus.COMPLETED
    b2.objectives = {"energy": -102.0, "band_gap": 1.5}
    b2.decisions = [
        Decision(
            id="d2", description="Choose functional",
            decision_type="categorical",
            chosen_option="HSE06", available_options=["PBE", "HSE06"],
            rationale="More accurate band gap",
        ),
    ]

    b3 = Branch(id="b3", name="LDA-test", hypothesis="LDA test")
    b3.status = BranchStatus.PRUNED
    b3.prune_reason = "Energy too high, dominated by b1 and b2"

    space.add_branch(b1)
    space.add_branch(b2)
    space.add_branch(b3)
    space.mark_pruned("b3", "Energy too high")

    space.update_pareto_front()
    return space


class TestExplorationQuery:
    def setup_method(self):
        self.space = _make_space()

    def test_pareto_query(self):
        result = self.space.query("What are the best branches?")
        assert result["type"] == "pareto_front"
        assert result["count"] >= 1

    def test_pareto_explicit(self):
        result = self.space.query("show pareto front")
        assert result["type"] == "pareto_front"

    def test_pruned_query(self):
        result = self.space.query("Why were branches pruned?")
        assert result["type"] == "pruned"
        assert result["count"] == 1
        assert result["branches"][0]["id"] == "b3"

    def test_pruned_with_filter(self):
        result = self.space.query("Why were LDA branches rejected?")
        assert result["type"] == "pruned"
        # Should filter for "lda"
        if result["count"] > 0:
            assert "lda" in result["branches"][0]["name"].lower()

    def test_path_query(self):
        result = self.space.query("What is the decision path for b1?")
        assert result["type"] == "path"
        assert result["branch_id"] == "b1"
        assert len(result["decisions"]) == 1
        assert result["decisions"][0]["chosen"] == "PBE"

    def test_path_query_fallback(self):
        result = self.space.query("Show me the decision path")
        assert result["type"] == "path"
        # Should fall back to first completed branch
        assert "branch_id" in result

    def test_status_query(self):
        result = self.space.query("What is the exploration status?")
        assert result["type"] == "status"
        assert result["total_branches"] == 3
        assert result["pruned"] == 1

    def test_list_all_query(self):
        result = self.space.query("List all branches")
        assert result["type"] == "all_branches"
        assert result["count"] == 3

    def test_unrecognized_query(self):
        result = self.space.query("xyzzy foobar baz")
        assert result["type"] == "unrecognized"
        assert "hint" in result
        assert "available_query_types" in result

    def test_optimal_keyword(self):
        result = self.space.query("optimal solutions")
        assert result["type"] == "pareto_front"

    def test_discard_keyword(self):
        result = self.space.query("discarded options")
        assert result["type"] == "pruned"
