"""Tests for MaterialsDatabaseTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.tools.materials_database_tool import MaterialsDatabaseTool
from huginn.types import ToolContext


class _FakeSession:
    """Minimal async context manager to keep aiohttp imports optional in tests."""

    def __init__(self, responses):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, **kwargs):
        for prefix, data in self._responses:
            if url.startswith(prefix):
                return _FakeResponse(200, data)
        return _FakeResponse(404, {})


class _FakeResponse:
    def __init__(self, status, data):
        self.status = status
        self._data = data

    async def text(self):
        import json

        return json.dumps(self._data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.fixture
def tool(tmp_path):
    return MaterialsDatabaseTool(mp_api_key="test-mp-key", oqmd_api_key="test-oqmd-key")


@pytest.fixture
def context(tmp_path):
    return ToolContext(session_id="test", workspace=str(tmp_path))


@pytest.fixture(autouse=True)
def _clear_tool_cache():
    # @cacheable decorator uses a process-wide shared cache; prior tests
    # may have cached local_db results that bypass our monkeypatches.
    from huginn.tools.tool_cache import ToolCache
    ToolCache.shared().clear()
    yield


@pytest.mark.asyncio
async def test_mp_summary(tool, context, monkeypatch):
    async def _fake_get_json(session, url):
        return {
            "data": [
                {
                    "material_id": "mp-149",
                    "formula_pretty": "Si",
                    "energy_per_atom": -5.0,
                    "band_gap": 1.1,
                    "symmetry": {"symbol": "Fd-3m"},
                }
            ],
            "meta": {"total": 1},
        }

    monkeypatch.setattr(tool, "_get_json", _fake_get_json)
    # Bypass local structure cache so the mocked API path is exercised
    monkeypatch.setattr(tool, "_local_lookup", lambda *a, **kw: None)

    result = await tool.call(
        tool.input_schema(action="mp_summary", query="Si", limit=1), context
    )
    assert result.success is True
    data = result.data
    assert data["source"] == "materials_project"
    assert data["count"] == 1
    assert data["records"][0]["id"] == "mp-149"


@pytest.mark.asyncio
async def test_mp_structure_saves_json(tool, context, monkeypatch):
    async def _fake_get_json(session, url):
        return {
            "data": [
                {
                    "material_id": "mp-149",
                    "formula_pretty": "Si",
                    "structure": {"lattice": {}, "sites": []},
                }
            ]
        }

    monkeypatch.setattr(tool, "_get_json", _fake_get_json)
    monkeypatch.setattr(tool, "_local_lookup", lambda *a, **kw: None)

    result = await tool.call(
        tool.input_schema(action="mp_structure", query="mp-149", output_format="json"),
        context,
    )
    assert result.success is True
    saved = result.data["saved_file"]
    assert saved is not None
    assert Path(saved).exists()


@pytest.mark.asyncio
async def test_oqmd_query(tool, context, monkeypatch):
    async def _fake_get_json(session, url):
        return {
            "data": [
                {"entry_id": "12345", "name": "Si", "delta_e": -5.0, "band_gap": 1.0}
            ]
        }

    monkeypatch.setattr(tool, "_get_json", _fake_get_json)

    result = await tool.call(
        tool.input_schema(action="oqmd_query", query="Si", limit=1), context
    )
    assert result.success is True
    assert result.data["source"] == "oqmd"
    assert result.data["records"][0]["formula"] == "Si"


@pytest.mark.asyncio
async def test_mp_key_from_env(context, monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    tool = MaterialsDatabaseTool()
    # Bypass local cache so the missing-key error path is exercised
    monkeypatch.setattr(tool, "_local_lookup", lambda *a, **kw: None)
    result = await tool.call(
        tool.input_schema(action="mp_summary", query="Si"), context
    )
    assert result.success is False
    assert "MP_API_KEY" in result.error


def test_mp_key_resolution(tool):
    assert tool._mp_key("override") == "override"


def test_oqmd_key_optional(tool):
    # OQMD key can be None; tool should still work (open endpoint).
    assert tool._oqmd_key(None) is None or tool._oqmd_key(None) == "test-oqmd-key"


@pytest.mark.asyncio
async def test_http_error(tool, context, monkeypatch):
    async def _fake_get_json(session, url):
        raise RuntimeError("Database request failed (500): boom")

    monkeypatch.setattr(tool, "_get_json", _fake_get_json)
    monkeypatch.setattr(tool, "_local_lookup", lambda *a, **kw: None)

    result = await tool.call(
        tool.input_schema(action="mp_summary", query="Si"), context
    )
    assert result.success is False
    assert "500" in result.error or "boom" in result.error
