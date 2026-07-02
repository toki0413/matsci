"""Tests for Phase 3 database_tool — real API calls and mock fallback."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from huginn.tools.database_tool import DatabaseTool, DatabaseToolInput
from huginn.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


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
