"""ToolSchedulingPolicy 门面测试 —— 验证 facade 值 == ToolProfile 元数据."""

from __future__ import annotations

import pytest

from huginn.tools.registry import ToolRegistry
from huginn.tools.scheduling_policy import ToolSchedulingPolicy


@pytest.fixture(scope="module", autouse=True)
def _registered():
    """Ensure all tools are registered before tests run."""
    from huginn.tools import register_all_tools

    if not ToolRegistry.list_tools():
        register_all_tools()
    yield


class TestFacadeMatchesMetadata:
    """每个 facade 查询必须和 ToolProfile 上的字段完全一致."""

    def test_is_heavy(self):
        for t in ToolRegistry._tools.values():
            assert ToolSchedulingPolicy.is_heavy(t.name) == (t.cost_tier == "heavy")

    def test_is_light(self):
        for t in ToolRegistry._tools.values():
            assert ToolSchedulingPolicy.is_light(t.name) == (t.cost_tier == "light")

    def test_scope_of(self):
        for t in ToolRegistry._tools.values():
            assert ToolSchedulingPolicy.scope_of(t.name) == t.constraint_scope

    def test_phases_of(self):
        for t in ToolRegistry._tools.values():
            assert ToolSchedulingPolicy.phases_of(t.name) == t.phases

    def test_alternatives_for(self):
        for t in ToolRegistry._tools.values():
            assert ToolSchedulingPolicy.alternatives_for(t.name) == t.light_alternatives

    def test_heavy_actions_of(self):
        for t in ToolRegistry._tools.values():
            expected = t.heavy_actions if t.heavy_actions else frozenset()
            assert ToolSchedulingPolicy.heavy_actions_of(t.name) == expected

    def test_heavy_tool_names_matches_registry(self):
        expected = {
            t.name for t in ToolRegistry._tools.values() if t.cost_tier == "heavy"
        }
        assert set(ToolSchedulingPolicy.heavy_tool_names()) == expected

    def test_light_tool_names_matches_registry(self):
        expected = {
            t.name for t in ToolRegistry._tools.values() if t.cost_tier == "light"
        }
        assert set(ToolSchedulingPolicy.light_tool_names()) == expected

    def test_unknown_tool_returns_safe_defaults(self):
        """查不存在的工具名时返回安全默认值, 不抛."""
        assert ToolSchedulingPolicy.is_heavy("nonexistent_tool") is False
        assert ToolSchedulingPolicy.is_light("nonexistent_tool") is False
        assert ToolSchedulingPolicy.scope_of("nonexistent_tool") is None
        assert ToolSchedulingPolicy.phases_of("nonexistent_tool") is None
        assert ToolSchedulingPolicy.alternatives_for("nonexistent_tool") == ()
        assert ToolSchedulingPolicy.heavy_actions_of("nonexistent_tool") == frozenset()
