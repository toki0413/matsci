"""Tests for huginn.env_defaults — typed env accessors and registry."""
from __future__ import annotations

import pytest
from pathlib import Path

from huginn.env_defaults import (
    ENV_REGISTRY,
    EnvCategory,
    get_bool,
    get_float,
    get_int,
    get_path,
    get_str,
)

# ── get_str ────────────────────────────────────────────────────────


class TestGetStr:
    def test_simple(self, monkeypatch):
        monkeypatch.setenv("TEST_STR", "hello")
        assert get_str("TEST_STR") == "hello"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("TEST_STR", "  hello  ")
        assert get_str("TEST_STR") == "hello"

    def test_unset_returns_default(self):
        assert get_str("TEST_STR_UNSET", default="fb") == "fb"

    def test_unset_returns_empty_string(self):
        assert get_str("TEST_STR_UNSET") == ""


# ── get_bool ───────────────────────────────────────────────────────


class TestGetBool:
    @pytest.mark.parametrize("val", ["1", "true", "TRUE", "True", "yes", "Yes", "on", "ON"])
    def test_truthy_values(self, monkeypatch, val):
        monkeypatch.setenv("TEST_BOOL", val)
        assert get_bool("TEST_BOOL") is True

    @pytest.mark.parametrize("val", ["0", "false", "FALSE", "no", "No", "off", "OFF", "", "none"])
    def test_falsy_values(self, monkeypatch, val):
        monkeypatch.setenv("TEST_BOOL", val)
        assert get_bool("TEST_BOOL") is False

    def test_unset_returns_default_true(self):
        assert get_bool("TEST_BOOL_UNSET", default=True) is True

    def test_unset_returns_default_false(self):
        assert get_bool("TEST_BOOL_UNSET", default=False) is False

    def test_unrecognized_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_BOOL", "maybe")
        assert get_bool("TEST_BOOL", default=True) is True
        assert get_bool("TEST_BOOL", default=False) is False


# ── get_int ────────────────────────────────────────────────────────


class TestGetInt:
    def test_simple(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "42")
        assert get_int("TEST_INT") == 42

    def test_negative(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "-7")
        assert get_int("TEST_INT") == -7

    def test_unset_returns_default(self):
        assert get_int("TEST_INT_UNSET", default=99) == 99

    def test_unset_returns_zero(self):
        assert get_int("TEST_INT_UNSET") == 0

    def test_unparseable_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "not_a_number")
        assert get_int("TEST_INT", default=7) == 7

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_INT", "")
        assert get_int("TEST_INT", default=5) == 5


# ── get_float ──────────────────────────────────────────────────────


class TestGetFloat:
    def test_simple(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert get_float("TEST_FLOAT") == pytest.approx(3.14)

    def test_integer_string(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "5")
        assert get_float("TEST_FLOAT") == 5.0

    def test_unset_returns_default(self):
        assert get_float("TEST_FLOAT_UNSET", default=1.5) == 1.5

    def test_unparseable_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_FLOAT", "nope")
        assert get_float("TEST_FLOAT", default=2.5) == 2.5


# ── get_path ───────────────────────────────────────────────────────


class TestGetPath:
    def test_simple(self, monkeypatch):
        monkeypatch.setenv("TEST_PATH", "/tmp/foo")
        p = get_path("TEST_PATH")
        assert p is not None
        assert p == Path("/tmp/foo")

    def test_expands_user(self, monkeypatch):
        monkeypatch.setenv("TEST_PATH", "~/foo")
        p = get_path("TEST_PATH")
        assert p is not None
        assert "~" not in str(p)

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("TEST_PATH", "  /tmp/foo  ")
        p = get_path("TEST_PATH")
        assert p is not None
        assert p == Path("/tmp/foo")

    def test_unset_returns_default(self):
        assert get_path("TEST_PATH_UNSET") is None
        assert get_path("TEST_PATH_UNSET", default=None) is None

    def test_empty_string_returns_default(self, monkeypatch):
        monkeypatch.setenv("TEST_PATH", "")
        assert get_path("TEST_PATH") is None


# ── Registry ───────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_not_empty(self):
        assert len(ENV_REGISTRY) > 30

    def test_each_entry_has_required_fields(self):
        required = {"category", "type", "default", "description", "consumer"}
        for name, entry in ENV_REGISTRY.items():
            missing = required - set(entry.keys())
            assert not missing, f"{name} missing fields: {missing}"

    def test_all_keys_prefixed_with_huginn(self):
        for name in ENV_REGISTRY:
            assert name.startswith("HUGINN_"), f"{name} lacks HUGINN_ prefix"

    def test_categories_are_valid(self):
        valid_cats = {
            EnvCategory.RUNTIME, EnvCategory.AUTH, EnvCategory.ENCRYPTION,
            EnvCategory.SANDBOX, EnvCategory.RATE_LIMIT, EnvCategory.LLM,
            EnvCategory.AGENT, EnvCategory.MEMORY, EnvCategory.FEATURE_FLAG,
            EnvCategory.METACOG, EnvCategory.HPC, EnvCategory.LITERATURE,
            EnvCategory.MIDDLEWARE, EnvCategory.LOGGING, EnvCategory.GOVERNANCE,
            EnvCategory.RCB,
        }
        for name, entry in ENV_REGISTRY.items():
            assert entry["category"] in valid_cats, (
                f"{name} has unknown category: {entry['category']}"
            )

    def test_types_are_valid(self):
        valid_types = {"str", "bool", "int", "float", "path", "json"}
        for name, entry in ENV_REGISTRY.items():
            assert entry["type"] in valid_types, (
                f"{name} has unknown type: {entry['type']}"
            )

    def test_known_vars_present(self):
        """Spot-check that critical vars are registered."""
        must_have = [
            "HUGINN_CACHE_DIR",
            "HUGINN_API_KEY",
            "HUGINN_DEV_MODE",
            "HUGINN_CODEACT_MEM_CAP",
            "HUGINN_RATE_LIMIT_ENABLED",
            "HUGINN_LOG_LEVEL",
        ]
        for var in must_have:
            assert var in ENV_REGISTRY, f"{var} not in registry"
