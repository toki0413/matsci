"""Tests for the HuginnSystem object — consolidated runtime state."""
from __future__ import annotations

import pytest

from huginn.system import HuginnSystem, get_system, set_system


class TestHuginnSystemCreation:
    """Tests for creating HuginnSystem instances."""

    def test_default_creation(self):
        sys = HuginnSystem()
        assert sys.config is None
        assert sys.tool_registry is None
        assert sys.agent is None

    def test_creation_with_config(self):
        sys = HuginnSystem(config={"provider": "ollama"})
        assert sys.config == {"provider": "ollama"}

    def test_creation_with_multiple_fields(self):
        sys = HuginnSystem(
            config="mock_config",
            tool_registry="mock_registry",
            agent="mock_agent",
        )
        assert sys.config == "mock_config"
        assert sys.tool_registry == "mock_registry"
        assert sys.agent == "mock_agent"

    def test_default_active_threads_is_empty_dict(self):
        sys = HuginnSystem()
        assert sys.active_threads == {}
        assert isinstance(sys.active_threads, dict)

    def test_default_edit_tools_is_empty_set(self):
        sys = HuginnSystem()
        assert sys.edit_tools == set()
        assert isinstance(sys.edit_tools, set)


class TestHuginnSystemProperties:
    """Tests for HuginnSystem properties and methods."""

    def test_is_configured_false_when_no_config(self):
        sys = HuginnSystem()
        assert sys.is_configured is False

    def test_is_configured_true_with_config(self):
        sys = HuginnSystem(config="something")
        assert sys.is_configured is True

    def test_get_component_existing(self):
        sys = HuginnSystem(config="my_config")
        assert sys.get_component("config") == "my_config"

    def test_get_component_none_when_unset(self):
        sys = HuginnSystem()
        assert sys.get_component("config") is None

    def test_get_component_unknown_name(self):
        sys = HuginnSystem()
        assert sys.get_component("nonexistent_component") is None

    def test_list_components_all_none(self):
        sys = HuginnSystem()
        components = sys.list_components()
        assert isinstance(components, dict)
        assert all(v is False for v in components.values())
        assert "config" in components
        assert "agent" in components

    def test_list_components_partial(self):
        sys = HuginnSystem(config="cfg", tool_registry="tr")
        components = sys.list_components()
        assert components["config"] is True
        assert components["tool_registry"] is True
        assert components["agent"] is False

    def test_list_components_contains_all_expected_keys(self):
        sys = HuginnSystem()
        components = sys.list_components()
        expected_keys = {
            "config", "tool_registry", "skill_registry", "audit_logger",
            "memory_backend", "checkpointer_backend", "remote_job_backend",
            "agent_factory", "orchestrator", "memory_manager", "kb",
            "codebase", "agent", "planner_agent", "mcp_manager", "plan_store",
        }
        assert set(components.keys()) == expected_keys


class TestHuginnSystemTracking:
    """Tests for thread and edit_tools tracking."""

    def test_active_threads_mutation(self):
        sys = HuginnSystem()
        sys.active_threads["t1"] = {"status": "running"}
        assert "t1" in sys.active_threads
        assert sys.active_threads["t1"]["status"] == "running"

    def test_edit_tools_mutation(self):
        sys = HuginnSystem()
        sys.edit_tools.add("file_write_tool")
        assert "file_write_tool" in sys.edit_tools

    def test_independent_instances(self):
        """Each HuginnSystem instance has its own mutable state."""
        s1 = HuginnSystem()
        s2 = HuginnSystem()
        s1.active_threads["x"] = 1
        s1.edit_tools.add("a")
        assert "x" not in s2.active_threads
        assert "a" not in s2.edit_tools


class TestSystemSingleton:
    """Tests for module-level get_system / set_system."""

    def test_get_system_returns_instance(self):
        # Reset the global singleton
        import huginn.system as mod
        mod._system = None

        sys = get_system()
        assert isinstance(sys, HuginnSystem)

    def test_get_system_is_singleton(self):
        import huginn.system as mod
        mod._system = None

        s1 = get_system()
        s2 = get_system()
        assert s1 is s2

    def test_set_system_replaces(self):
        import huginn.system as mod
        mod._system = None

        custom = HuginnSystem(config="custom_config")
        set_system(custom)
        assert get_system() is custom
        assert get_system().config == "custom_config"

        # Clean up
        mod._system = None


class TestFromServerContext:
    """Tests for creating HuginnSystem from ServerContext-like fields."""

    def test_from_context_fields(self):
        """HuginnSystem can hold the same fields as ServerContext."""
        sys = HuginnSystem(
            config="cfg",
            tool_registry="tr",
            skill_registry="sr",
            audit_logger="al",
            memory_backend="mb",
            checkpointer_backend="cb",
            remote_job_backend="rjb",
            agent_factory="af",
            orchestrator="orch",
            memory_manager="mm",
            kb="kb",
            codebase="code",
            agent="agent",
            planner_agent="planner",
            mcp_manager="mcp",
            plan_store="ps",
            permission_config="perm",
        )
        components = sys.list_components()
        assert all(v is True for v in components.values())
        assert sys.permission_config == "perm"
        assert sys.plan_store == "ps"
