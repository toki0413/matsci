"""Tests for tunable parameters (ParameterRegistry + routes)."""

from __future__ import annotations

import pytest

from huginn.tools.parameters import (
    Parameter,
    ParameterRegistry,
    ParamType,
    _register_defaults,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset all parameters to defaults before each test."""
    ParameterRegistry.reset()
    yield
    ParameterRegistry.reset()


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------


class TestParameter:
    def test_default_current(self) -> None:
        p = Parameter(name="t", param_type=ParamType.INT, default=5)
        assert p.current == 5

    def test_validate_int_valid(self) -> None:
        p = Parameter(
            name="t",
            param_type=ParamType.INT,
            default=5,
            min_value=0,
            max_value=10,
        )
        ok, msg = p.validate(7)
        assert ok is True

    def test_validate_int_below_min(self) -> None:
        p = Parameter(
            name="t",
            param_type=ParamType.INT,
            default=5,
            min_value=0,
            max_value=10,
        )
        ok, msg = p.validate(-1)
        assert ok is False
        assert "below minimum" in msg

    def test_validate_int_above_max(self) -> None:
        p = Parameter(
            name="t",
            param_type=ParamType.INT,
            default=5,
            min_value=0,
            max_value=10,
        )
        ok, msg = p.validate(11)
        assert ok is False
        assert "above maximum" in msg

    def test_validate_int_bad_type(self) -> None:
        p = Parameter(name="t", param_type=ParamType.INT, default=5)
        ok, msg = p.validate("abc")
        assert ok is False
        assert "Expected integer" in msg

    def test_validate_float_valid(self) -> None:
        p = Parameter(
            name="t",
            param_type=ParamType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        ok, _ = p.validate(1.5)
        assert ok is True

    def test_validate_float_out_of_range(self) -> None:
        p = Parameter(
            name="t",
            param_type=ParamType.FLOAT,
            default=1.0,
            min_value=0.0,
            max_value=2.0,
        )
        ok, msg = p.validate(3.0)
        assert ok is False

    def test_validate_float_bad_type(self) -> None:
        p = Parameter(name="t", param_type=ParamType.FLOAT, default=1.0)
        ok, msg = p.validate("nope")
        assert ok is False
        assert "Expected float" in msg

    def test_validate_bool_valid(self) -> None:
        p = Parameter(name="t", param_type=ParamType.BOOL, default=True)
        ok, _ = p.validate(False)
        assert ok is True

    def test_validate_bool_bad_type(self) -> None:
        p = Parameter(name="t", param_type=ParamType.BOOL, default=True)
        ok, msg = p.validate(1)
        assert ok is False
        assert "Expected bool" in msg

    def test_validate_string_valid(self) -> None:
        p = Parameter(name="t", param_type=ParamType.STRING, default="hi")
        ok, _ = p.validate("hello")
        assert ok is True

    def test_validate_string_bad_type(self) -> None:
        p = Parameter(name="t", param_type=ParamType.STRING, default="hi")
        ok, msg = p.validate(123)
        assert ok is False
        assert "Expected string" in msg

    def test_validate_choice_valid(self) -> None:
        p = Parameter(
            name="t",
            param_type=ParamType.CHOICE,
            default="a",
            choices=["a", "b", "c"],
        )
        ok, _ = p.validate("b")
        assert ok is True

    def test_validate_choice_invalid(self) -> None:
        p = Parameter(
            name="t",
            param_type=ParamType.CHOICE,
            default="a",
            choices=["a", "b", "c"],
        )
        ok, msg = p.validate("d")
        assert ok is False
        assert "must be one of" in msg


# ---------------------------------------------------------------------------
# ParameterRegistry
# ---------------------------------------------------------------------------


class TestParameterRegistry:
    def test_register_and_get(self) -> None:
        p = Parameter(name="test_reg", param_type=ParamType.INT, default=10)
        ParameterRegistry.register(p)
        assert ParameterRegistry.get("test_reg") is p

    def test_get_unknown(self) -> None:
        assert ParameterRegistry.get("nonexistent_xyz") is None

    def test_set_value_valid(self) -> None:
        p = Parameter(
            name="test_set",
            param_type=ParamType.INT,
            default=10,
            min_value=0,
            max_value=100,
        )
        ParameterRegistry.register(p)
        ok, _ = ParameterRegistry.set_value("test_set", 50)
        assert ok is True
        assert ParameterRegistry.get_value("test_set") == 50

    def test_set_value_invalid(self) -> None:
        p = Parameter(
            name="test_set_inv",
            param_type=ParamType.INT,
            default=10,
            min_value=0,
            max_value=100,
        )
        ParameterRegistry.register(p)
        ok, msg = ParameterRegistry.set_value("test_set_inv", 200)
        assert ok is False
        assert ParameterRegistry.get_value("test_set_inv") == 10

    def test_set_value_unknown(self) -> None:
        ok, msg = ParameterRegistry.set_value("nonexistent_xyz", 1)
        assert ok is False
        assert "Unknown parameter" in msg

    def test_get_value_default(self) -> None:
        val = ParameterRegistry.get_value("nonexistent_xyz", default=42)
        assert val == 42

    def test_reset_single(self) -> None:
        p = Parameter(name="test_rst", param_type=ParamType.INT, default=5)
        ParameterRegistry.register(p)
        ParameterRegistry.set_value("test_rst", 99)
        assert ParameterRegistry.get_value("test_rst") == 99
        ParameterRegistry.reset("test_rst")
        assert ParameterRegistry.get_value("test_rst") == 5

    def test_reset_all(self) -> None:
        p = Parameter(name="test_rst_all", param_type=ParamType.INT, default=7)
        ParameterRegistry.register(p)
        ParameterRegistry.set_value("test_rst_all", 99)
        ParameterRegistry.reset()
        assert ParameterRegistry.get_value("test_rst_all") == 7

    def test_list_params_all(self) -> None:
        params = ParameterRegistry.list_params()
        assert len(params) > 0
        names = [p["name"] for p in params]
        assert "max_walltime_hours" in names

    def test_list_params_by_category(self) -> None:
        hpc_params = ParameterRegistry.list_params(category="hpc")
        assert len(hpc_params) > 0
        for p in hpc_params:
            assert p["category"] == "hpc"

    def test_list_params_empty_category(self) -> None:
        params = ParameterRegistry.list_params(category="nonexistent_cat")
        assert params == []

    def test_categories(self) -> None:
        cats = ParameterRegistry.categories()
        assert "hpc" in cats
        assert "llm" in cats
        assert "execution" in cats

    def test_on_change_callback(self) -> None:
        changes: list = []
        p = Parameter(
            name="test_cb",
            param_type=ParamType.INT,
            default=0,
            on_change=lambda v: changes.append(v),
        )
        ParameterRegistry.register(p)
        ParameterRegistry.set_value("test_cb", 42)
        assert changes == [42]

    def test_on_change_not_called_same_value(self) -> None:
        changes: list = []
        p = Parameter(
            name="test_cb_same",
            param_type=ParamType.INT,
            default=5,
            on_change=lambda v: changes.append(v),
        )
        ParameterRegistry.register(p)
        ParameterRegistry.set_value("test_cb_same", 5)
        assert changes == []


# ---------------------------------------------------------------------------
# Built-in parameters
# ---------------------------------------------------------------------------


class TestBuiltinParameters:
    def test_builtin_params_registered(self) -> None:
        expected = [
            "max_walltime_hours",
            "max_parallel_jobs",
            "default_queue",
            "auto_validate_physics",
            "mock_mode",
            "cache_ttl_seconds",
            "max_script_timeout",
            "llm_temperature",
            "llm_max_tokens",
        ]
        for name in expected:
            assert ParameterRegistry.get(name) is not None, (
                f"Built-in parameter '{name}' not registered"
            )

    def test_max_walltime_bounds(self) -> None:
        p = ParameterRegistry.get("max_walltime_hours")
        assert p.min_value == 1
        assert p.max_value == 720

    def test_default_queue_choices(self) -> None:
        p = ParameterRegistry.get("default_queue")
        assert "normal" in p.choices
        assert "gpu" in p.choices


# ---------------------------------------------------------------------------
# Route endpoints
# ---------------------------------------------------------------------------


class TestParameterRoutes:
    """Test the /parameters/* FastAPI endpoints."""

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from huginn.routes.parameters import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_list_parameters(self, client) -> None:
        resp = client.get("/parameters/")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        names = [p["name"] for p in data]
        assert "max_walltime_hours" in names

    def test_list_parameters_by_category(self, client) -> None:
        resp = client.get("/parameters/?category=hpc")
        assert resp.status_code == 200
        data = resp.json()
        assert all(p["category"] == "hpc" for p in data)

    def test_categories(self, client) -> None:
        resp = client.get("/parameters/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert "categories" in data
        assert "hpc" in data["categories"]

    def test_set_parameter_valid(self, client) -> None:
        resp = client.post(
            "/parameters/set",
            json={"name": "max_parallel_jobs", "value": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["value"] == 10

    def test_set_parameter_invalid(self, client) -> None:
        resp = client.post(
            "/parameters/set",
            json={"name": "max_parallel_jobs", "value": 999},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "error" in data

    def test_set_parameter_unknown(self, client) -> None:
        resp = client.post(
            "/parameters/set",
            json={"name": "nonexistent_xyz", "value": 1},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_get_parameter(self, client) -> None:
        resp = client.get("/parameters/llm_temperature")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "llm_temperature"
        assert data["type"] == "float"
        assert "current" in data

    def test_get_parameter_unknown(self, client) -> None:
        resp = client.get("/parameters/nonexistent_xyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_reset_parameters(self, client) -> None:
        # First change a value
        client.post(
            "/parameters/set",
            json={"name": "max_parallel_jobs", "value": 10},
        )
        # Reset it
        resp = client.post("/parameters/reset?name=max_parallel_jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        # Verify reset
        resp2 = client.get("/parameters/max_parallel_jobs")
        assert resp2.json()["current"] == 5

    def test_reset_all(self, client) -> None:
        resp = client.post("/parameters/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["reset"] == "all"
