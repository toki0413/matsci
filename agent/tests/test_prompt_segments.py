"""Tests for prompt segment pluginization (Everything is a Plugin, 形态 B).

Phase 1 diff 验证: 段插件组装产出与硬编码拼接一致, 保证切换无回归.
"""

from __future__ import annotations

import pytest

from huginn.agent.prompt_builder import (
    metacog_segment,
    mode_segment,
    persona_segment,
    phase_segment,
    safety_segment,
    tools_segment,
)
from huginn.plugins.prompt_segments import (
    assemble_prompt_segments,
    clear_registry,
    register_prompt_segment,
    unregister_prompt_segment,
)


def _hardcoded_reference(mode, phase, metacog, system_prompt=None):
    """切换前 build_prompt 的硬编码六段拼接 (不含 flag-gated thinking)."""
    segments = [
        persona_segment(system_prompt),
        mode_segment(mode),
        phase_segment(phase),
        metacog_segment(metacog),
        tools_segment(mode, phase, metacog),
        safety_segment(),
    ]
    return "\n\n".join(s for s in segments if s)


@pytest.fixture(autouse=True)
def _reset_registry_between_tests():
    yield
    # 每个测试后恢复内置六段 + thinking, 防止 clear/override 污染后续用例.
    from huginn.agent.prompt_builder import _register_builtin_segments

    clear_registry()
    _register_builtin_segments()


class TestPluginMatchesHardcoded:
    @pytest.mark.parametrize(
        "mode,phase,metacog",
        [
            ("research", "execute", "s4_construct"),
            ("chat", "perceive", "unknown_state"),
            ("research", "validate", "s7_self_modify"),
            ("code", "hypothesis", "s0_blank"),
            ("fusion", "report", "s5_consolidate"),
        ],
    )
    def test_assembly_equals_hardcoded(self, mode, phase, metacog):
        # external_thinking flag 默认关 → thinking 段为空, 插件组装应与六段一致.
        got = assemble_prompt_segments(mode, phase, metacog)
        want = _hardcoded_reference(mode, phase, metacog)
        assert got == want

    def test_persona_override_received(self):
        # system_prompt 传入时 persona 段应使用 runtime persona.
        got = assemble_prompt_segments("chat", "execute", "s0_blank", "My custom persona.")
        assert "My custom persona." in got

    def test_build_prompt_smoke(self):
        from huginn.agent.prompt_builder import build_prompt

        p = build_prompt("research", "execute", "s4_construct")
        assert p and len(p) > 50


class TestPluginOverride:
    def test_register_new_segment_appears(self):
        register_prompt_segment("custom", lambda m, p, s, sp: "## CUSTOM\nhello")
        got = assemble_prompt_segments("chat", "execute", "s0_blank")
        assert "## CUSTOM\nhello" in got
        unregister_prompt_segment("custom")

    def test_override_builtin_segment(self):
        # 同名注册覆盖内置 tools 段.
        register_prompt_segment("tools", lambda m, p, s, sp: "## TOOLS\ncustom-tools")
        got = assemble_prompt_segments("chat", "execute", "s0_blank")
        assert "custom-tools" in got
        assert "## TOOLS\nTools available" not in got
        unregister_prompt_segment("tools")

    def test_segment_exception_isolated(self):
        def boom(m, p, s, sp):
            raise RuntimeError("segment blew up")

        register_prompt_segment("boom", boom)
        # 异常段被跳过, 其余段正常拼接, 不抛异常.
        got = assemble_prompt_segments("chat", "execute", "s0_blank")
        assert "boom" not in got
        assert "## SAFETY" in got
        unregister_prompt_segment("boom")

    def test_clear_registry_empty_returns_empty(self):
        clear_registry()
        assert assemble_prompt_segments("chat", "execute", "s0_blank") == ""