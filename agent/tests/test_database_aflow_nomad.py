"""Tests for AFLOW + NOMAD real API integration in database_tool."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from huginn.tools.database_tool import (
    DatabaseTool,
    DatabaseToolInput,
    MaterialsDatabaseTool,
    _formula_to_elements,
)


class TestFormulaToElements:
    def test_simple_binary(self):
        assert _formula_to_elements("SiO2") == ["Si", "O"]

    def test_ternary(self):
        assert _formula_to_elements("BaTiO3") == ["Ba", "Ti", "O"]


class TestNormalizeAflow:
    def test_normalize_json_list(self):
        tool = MaterialsDatabaseTool()
        raw = [
            {"compound": "SiO2", "auid": "aflow:123", "sg": "P1", "Egap": 5.2, "enthalpy_atom": -9.0},
        ]
        records = tool._normalize_aflow(raw)
        assert len(records) == 1
        assert records[0]["formula"] == "SiO2"
        assert records[0]["entry_id"] == "aflow:123"
        assert records[0]["band_gap"] == 5.2
        assert records[0]["source"] == "aflow"

    def test_normalize_text_block(self):
        tool = MaterialsDatabaseTool()
        raw = ">>>compound=SiO2\nauid=aflow:456\nEgap=3.1\n>>>\ncompound=Fe2O3\nauid=aflow:789"
        records = tool._normalize_aflow(raw)
        assert len(records) == 2
        assert records[0]["formula"] == "SiO2"
        assert records[1]["formula"] == "Fe2O3"

    def test_normalize_dict_with_data_key(self):
        tool = MaterialsDatabaseTool()
        raw = {"data": [{"compound": "GaAs", "auid": "aflow:abc", "Egap": 1.4}]}
        records = tool._normalize_aflow(raw)
        assert records[0]["formula"] == "GaAs"
        assert records[0]["band_gap"] == 1.4

    def test_normalize_caps_at_50(self):
        tool = MaterialsDatabaseTool()
        raw = [{"compound": f"C{i}", "auid": f"aflow:{i}"} for i in range(100)]
        records = tool._normalize_aflow(raw)
        assert len(records) == 50


class TestNormalizeNomad:
    def test_normalize_nomad_response(self):
        tool = MaterialsDatabaseTool()
        raw = {
            "data": [
                {
                    "entry_id": "nomad-xyz",
                    "results": {
                        "material": [{
                            "chemical_formula_descriptive": "SiO2",
                            "structure": {"space_group": "P1"},
                        }],
                        "properties": {
                            "electronic": {"band_gap": 5.0},
                            "energetic": {"total_energy": -100.0},
                        },
                    },
                }
            ]
        }
        records = tool._normalize_nomad(raw)
        assert len(records) == 1
        assert records[0]["entry_id"] == "nomad-xyz"
        assert records[0]["formula"] == "SiO2"
        assert records[0]["band_gap"] == 5.0
        assert records[0]["energy"] == -100.0
        assert records[0]["source"] == "nomad"


class TestAflowQuery:
    def test_aflow_requires_formula_or_material_id(self):
        tool = DatabaseTool()
        args = DatabaseToolInput(database="aflow", query_type="search", formula=None, material_id=None)
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.success is False
        assert "required" in result.error.lower()

    def test_aflow_success_parses_records(self, monkeypatch):
        tool = DatabaseTool()
        args = DatabaseToolInput(database="aflow", query_type="search", formula="SiO2")

        # mock aiohttp
        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.json = AsyncMock(return_value=[{"compound": "SiO2", "auid": "aflow:1", "Egap": 5.2}])

        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=None)
        fake_session.get = MagicMock(return_value=fake_resp)
        fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
        fake_resp.__aexit__ = AsyncMock(return_value=None)

        fake_aiohttp = MagicMock()
        fake_aiohttp.ClientTimeout = MagicMock()
        fake_aiohttp.ClientSession = MagicMock(return_value=fake_session)

        monkeypatch.setitem(__import__("sys").modules, "aiohttp", fake_aiohttp)
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.success is True
        assert result.data["database"] == "aflow"
        assert result.data["count"] == 1
        assert result.data["records"][0]["formula"] == "SiO2"

    def test_aflow_timeout_degrades_gracefully(self, monkeypatch):
        tool = DatabaseTool()
        # 用 Fe2O3 避开 success 测试的 SiO2 缓存命中 (@cacheable TTL=7d)
        args = DatabaseToolInput(database="aflow", query_type="search", formula="Fe2O3")

        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=None)
        fake_session.get = MagicMock(side_effect=asyncio.TimeoutError())

        fake_aiohttp = MagicMock()
        fake_aiohttp.ClientTimeout = MagicMock()
        fake_aiohttp.ClientSession = MagicMock(return_value=fake_session)

        monkeypatch.setitem(__import__("sys").modules, "aiohttp", fake_aiohttp)
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.success is False
        assert "timed out" in result.error.lower()


class TestNomadQuery:
    def test_nomad_success(self, monkeypatch):
        tool = DatabaseTool()
        args = DatabaseToolInput(database="nomad", query_type="search", formula="SiO2")

        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.json = AsyncMock(return_value={
            "data": [{
                "entry_id": "nomad-1",
                "results": {
                    "material": [{"chemical_formula_descriptive": "SiO2"}],
                    "properties": {},
                },
            }]
        })

        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=None)
        fake_session.post = MagicMock(return_value=fake_resp)
        fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
        fake_resp.__aexit__ = AsyncMock(return_value=None)

        fake_aiohttp = MagicMock()
        fake_aiohttp.ClientTimeout = MagicMock()
        fake_aiohttp.ClientSession = MagicMock(return_value=fake_session)

        monkeypatch.setitem(__import__("sys").modules, "aiohttp", fake_aiohttp)
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.success is True
        assert result.data["database"] == "nomad"
        assert result.data["records"][0]["formula"] == "SiO2"

    def test_nomad_api_error_returns_failure(self, monkeypatch):
        tool = DatabaseTool()
        # 用 Al2O3 避开 success 测试的 SiO2 缓存命中 (@cacheable TTL=7d)
        args = DatabaseToolInput(database="nomad", query_type="search", formula="Al2O3")

        fake_resp = MagicMock()
        fake_resp.status = 500
        fake_resp.text = AsyncMock(return_value="internal error")

        fake_session = MagicMock()
        fake_session.__aenter__ = AsyncMock(return_value=fake_session)
        fake_session.__aexit__ = AsyncMock(return_value=None)
        fake_session.post = MagicMock(return_value=fake_resp)
        fake_resp.__aenter__ = AsyncMock(return_value=fake_resp)
        fake_resp.__aexit__ = AsyncMock(return_value=None)

        fake_aiohttp = MagicMock()
        fake_aiohttp.ClientTimeout = MagicMock()
        fake_aiohttp.ClientSession = MagicMock(return_value=fake_session)

        monkeypatch.setitem(__import__("sys").modules, "aiohttp", fake_aiohttp)
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.success is False
        assert "500" in result.error
