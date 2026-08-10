"""Tests for the HuginnSystem object — consolidated runtime state."""
from __future__ import annotations

from huginn.system import HuginnSystem, get_system, set_system


class TestHuginnSystemCreation:
    """Tests for creating HuginnSystem instances."""

    def test_default_creation(self):
        """v23 状态树合并后, HuginnSystem 继承 ServerContext 默认值.

        config / agent 等可选组件仍默认 None; 但 tool_registry /
        skill_registry / permission_config / audit_logger 等基础设施
        默认会实例化 (与 ServerContext 一致), 不再是 None.
        """
        sys = HuginnSystem()
        assert sys.config is None
        assert sys.agent is None
        # 基础设施默认实例化 (v23 unification 后跟 ServerContext 对齐)
        assert sys.tool_registry is not None
        assert sys.skill_registry is not None

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

    def test_list_components_defaults(self):
        """v23 unification: 空 HuginnSystem 的组件状态混合 (None + 默认实例).

        config / agent / orchestrator 等可选组件为 None (False);
        tool_registry / skill_registry 等基础设施默认实例化 (True).
        """
        sys = HuginnSystem()
        components = sys.list_components()
        assert isinstance(components, dict)
        # 可选组件默认 None
        assert components["config"] is False
        assert components["agent"] is False
        assert components["orchestrator"] is False
        # 基础设施默认实例化 (v23 跟 ServerContext 对齐)
        assert components["tool_registry"] is True
        assert components["skill_registry"] is True

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


class TestStateTreeUnification:
    """v23 状态树合并 — HuginnSystem 是 ServerContext 的子类."""

    def test_huginn_system_is_server_context_subclass(self):
        from huginn.server_context import ServerContext

        assert issubclass(HuginnSystem, ServerContext)

    def test_huginn_system_inherits_active_threads(self):
        sys = HuginnSystem()
        assert hasattr(sys, "active_threads")
        assert sys.active_threads == {}

    def test_huginn_system_inherits_edit_tools(self):
        sys = HuginnSystem()
        assert hasattr(sys, "edit_tools")
        assert sys.edit_tools == set()

    def test_huginn_system_inherits_is_configured(self):
        sys = HuginnSystem()
        assert sys.is_configured is False
        sys.config = "x"
        assert sys.is_configured is True

    def test_huginn_system_inherits_get_component(self):
        sys = HuginnSystem(config="cfg")
        assert sys.get_component("config") == "cfg"
        assert sys.get_component("nonexistent") is None

    def test_set_system_accepts_server_context(self):
        """set_system 接受 ServerContext 实例 (自动转型为 HuginnSystem)."""
        import huginn.system as mod
        from huginn.server_context import ServerContext

        mod._system = None
        ctx = ServerContext(config="from_ctx")
        set_system(ctx)
        sys = get_system()
        assert isinstance(sys, HuginnSystem)
        assert sys.config == "from_ctx"
        mod._system = None

    def test_huginn_system_has_encrypted_rag_field(self):
        """v23 unification: encrypted_rag 字段从 ServerContext 继承."""
        sys = HuginnSystem()
        assert hasattr(sys, "encrypted_rag")
        assert sys.encrypted_rag is None

    def test_huginn_system_has_state_registry_field(self):
        """v23 unification: state_registry 字段从 ServerContext 继承."""
        sys = HuginnSystem()
        assert hasattr(sys, "state_registry")
        # 默认实例化 (StateRegistry.shared())
        assert sys.state_registry is not None
