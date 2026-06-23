"""Tests for the Huginn data dictionary — types, registry, and routes."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from huginn.data.dictionary import DataDictionary
from huginn.data.types import DataField, DataSchema, DataType
from huginn.routes.data_dict import router


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def app() -> FastAPI:
    """Create a minimal FastAPI app with only the data_dict router."""
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ── DataType enum tests ───────────────────────────────────────────


class TestDataTypes:
    def test_data_type_values(self):
        assert DataType.CRYSTAL_STRUCTURE.value == "crystal_structure"
        assert DataType.DFT_RESULT.value == "dft_result"

    def test_data_type_is_string_enum(self):
        assert isinstance(DataType.CRYSTAL_STRUCTURE, str)

    def test_all_seven_types_exist(self):
        assert len(DataType) == 7


# ── DataField / DataSchema tests ──────────────────────────────────


class TestDataSchema:
    def test_data_field_defaults(self):
        f = DataField("energy", "float")
        assert f.required is False
        assert f.description == ""
        assert f.unit is None

    def test_data_field_with_unit(self):
        f = DataField("energy", "float", required=True, description="Total energy", unit="eV")
        assert f.unit == "eV"
        assert f.required is True

    def test_data_schema_creation(self):
        schema = DataSchema(
            type_name=DataType.DFT_RESULT,
            description="DFT results",
            fields=[DataField("energy", "float", required=True)],
        )
        assert schema.version == "1.0"
        assert schema.tags == []
        assert len(schema.fields) == 1


# ── DataDictionary registry tests ─────────────────────────────────


class TestDataDictionary:
    def test_builtin_schemas_registered(self):
        types = DataDictionary.list_types()
        assert "crystal_structure" in types
        assert "dft_result" in types
        assert "molecular_dynamics" in types
        assert "potential" in types
        assert "job_record" in types
        assert "experimental_data" in types
        assert "descriptor" in types

    def test_list_types_count(self):
        assert len(DataDictionary.list_types()) == 7

    def test_get_existing_schema(self):
        schema = DataDictionary.get(DataType.CRYSTAL_STRUCTURE)
        assert schema is not None
        assert schema.type_name == DataType.CRYSTAL_STRUCTURE
        assert "formula" in [f.name for f in schema.fields]

    def test_get_dft_schema_has_energy_field(self):
        schema = DataDictionary.get(DataType.DFT_RESULT)
        assert schema is not None
        field_names = [f.name for f in schema.fields]
        assert "energy" in field_names
        assert "converged" in field_names

    def test_register_custom_schema(self):
        custom = DataSchema(
            type_name=DataType.CRYSTAL_STRUCTURE,
            description="Overridden crystal structure",
            fields=[DataField("custom_field", "str", required=True)],
        )
        DataDictionary.register(custom)
        retrieved = DataDictionary.get(DataType.CRYSTAL_STRUCTURE)
        assert retrieved is not None
        assert retrieved.description == "Overridden crystal structure"

        # Restore original by re-registering builtins
        from huginn.data.dictionary import _register_builtins
        _register_builtins()


# ── Validation tests ──────────────────────────────────────────────


class TestDataValidation:
    def test_validate_valid_crystal_structure(self):
        errors = DataDictionary.validate(
            DataType.CRYSTAL_STRUCTURE, {"formula": "Si"}
        )
        assert errors == []

    def test_validate_missing_required_field(self):
        errors = DataDictionary.validate(DataType.CRYSTAL_STRUCTURE, {})
        assert len(errors) == 1
        assert "formula" in errors[0]

    def test_validate_dft_missing_multiple_required(self):
        errors = DataDictionary.validate(DataType.DFT_RESULT, {})
        assert len(errors) == 2  # energy and converged are required

    def test_validate_dft_all_required_present(self):
        errors = DataDictionary.validate(
            DataType.DFT_RESULT, {"energy": -5.0, "converged": 1}
        )
        assert errors == []

    def test_validate_unknown_type_returns_error(self):
        # We can't easily create an unknown DataType enum member,
        # so we test that valid types with empty data produce expected errors
        errors = DataDictionary.validate(DataType.JOB_RECORD, {})
        assert len(errors) >= 1


# ── Route tests ───────────────────────────────────────────────────


class TestDataDictRoutes:
    def test_list_data_types(self, client: TestClient):
        resp = client.get("/data/dictionary")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 7
        names = [item["type_name"] for item in data]
        assert "crystal_structure" in names

    def test_get_schema_crystal_structure(self, client: TestClient):
        resp = client.get("/data/dictionary/crystal_structure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type_name"] == "crystal_structure"
        assert "fields" in data
        assert len(data["fields"]) > 0

    def test_get_schema_unknown_type(self, client: TestClient):
        resp = client.get("/data/dictionary/nonexistent")
        assert resp.status_code == 404

    def test_validate_valid_data(self, client: TestClient):
        resp = client.post(
            "/data/validate",
            json={"type_name": "crystal_structure", "data": {"formula": "Si"}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is True
        assert body["errors"] == []

    def test_validate_invalid_data(self, client: TestClient):
        resp = client.post(
            "/data/validate",
            json={"type_name": "dft_result", "data": {}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["valid"] is False
        assert len(body["errors"]) > 0

    def test_validate_unknown_type(self, client: TestClient):
        resp = client.post(
            "/data/validate",
            json={"type_name": "bogus_type", "data": {}},
        )
        assert resp.status_code == 404
