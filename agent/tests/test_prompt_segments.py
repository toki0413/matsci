"""Tests for prompt segment pluginization (Everything is a Plugin, 形态 B).

Phase 1 diff 验证: 段插件组装产出与硬编码拼接一致, 保证切换无回归.
"""

from __future__ import annotations

import pytest

from huginn.agent.prompt_builder import (
    metacog_segment,
    mode_segment,
    multimodal_segment,
    persona_segment,
    phase_segment,
    safety_segment,
    tools_segment,
    writing_segment,
)
from huginn.plugins.prompt_segments import (
    assemble_prompt_segments,
    clear_registry,
    register_prompt_segment,
    unregister_prompt_segment,
)


def _hardcoded_reference(mode, phase, metacog, system_prompt=None):
    """切换前 build_prompt 的硬编码拼接 (单段含 multimodal/writing, 不含 flag-gated thinking)."""
    segments = [
        persona_segment(system_prompt),
        mode_segment(mode),
        phase_segment(phase),
        metacog_segment(metacog),
        tools_segment(mode, phase, metacog),
        multimodal_segment(),
        writing_segment(),
        safety_segment(),
    ]
    return "\n\n".join(s for s in segments if s)


@pytest.fixture(autouse=True)
def _reset_registry_between_tests():
    # external_thinking flag 是进程级单例, 会被 model_tier 的 set_tier
    # (M4 联动) 或其它测试开启, 必须在每次用例前恢复默认关, 才能保证
    # "默认关 → thinking 段为空" 的组装断言不被污染.
    from huginn.feature_flags import FeatureFlags

    FeatureFlags.shared().reset("external_thinking")
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


class TestPluginOnlyAssembly:
    """只组装插件段 (排除框架骨架段) —— autoloop 直接 LLM 路径的注入源."""

    def test_plugin_segment_included_framework_excluded(self):
        from huginn.agent.prompt_builder import _register_builtin_segments
        from huginn.plugins.prompt_segments import assemble_plugin_prompt_segments

        clear_registry()
        _register_builtin_segments()
        register_prompt_segment("custom", lambda m, p, s, sp: "## CUSTOM")
        got = assemble_plugin_prompt_segments("chat", "execute", "s0_blank")
        assert "## CUSTOM" in got
        # 框架骨架段 (persona/mode/phase/tools/safety) 不属插件贡献, 被排除.
        assert "## SAFETY" not in got
        assert "## TOOLS" not in got

    def test_empty_when_only_framework_segments(self):
        from huginn.agent.prompt_builder import _register_builtin_segments
        from huginn.plugins.prompt_segments import assemble_plugin_prompt_segments

        clear_registry()
        _register_builtin_segments()
        assert assemble_plugin_prompt_segments("chat", "execute", "s0_blank") == ""


class TestAutoloopConsumesPluginSegments:
    """autoloop 的 _llm_chat 绕过 build_prompt, 必须显式消费插件段."""

    async def test_llm_chat_appends_plugin_segment(self, monkeypatch):
        from huginn.autoloop.engine_act import EngineAct
        from huginn.plugins.asd_ste100.main import (
            SEGMENT_NAME,
            ste_prompt_segment,
        )
        from huginn.plugins.prompt_segments import (
            register_prompt_segment,
            unregister_prompt_segment,
        )

        monkeypatch.setenv("HUGINN_STE_MODE", "agents")
        captured: dict = {}

        class _FakeLLM:
            async def ainvoke(self, messages):  # noqa: ANN001
                captured["messages"] = messages

                class _R:
                    content = "ok"
                    usage_metadata = None

                return _R()

        class _StubEngine:
            _current_phase = "hypothesize"
            _grill_active = False
            model_router = None

            def __init__(self) -> None:
                self.model = _FakeLLM()

            def _persona_system_prompt(self, name):  # noqa: ANN001
                return "PERSONA-CORE"

            async def _track_llm_usage(self, usage):  # noqa: ANN001
                return None

        register_prompt_segment(SEGMENT_NAME, ste_prompt_segment, priority=65)
        try:
            await EngineAct(_StubEngine())._llm_chat("hi", persona_name="default")
        finally:
            unregister_prompt_segment(SEGMENT_NAME)

        sys_msg = captured["messages"][0]
        assert type(sys_msg).__name__ == "SystemMessage"
        assert "PERSONA-CORE" in sys_msg.content
        assert "ASD-STE100" in sys_msg.content
