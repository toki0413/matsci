"""Tests for the ponytail behavior-constraint plugin."""

import asyncio
from unittest.mock import MagicMock

import pytest

from huginn.api.context import PluginContext
from huginn.api.event import Event, EventType, LLMRequestEvent, MessageEvent
from huginn.api.star import Star
from huginn.plugins.ponytail.main import PonytailStar, PonytailState, _audit_python_tree
from huginn.plugins.skill_loader import parse_skill_file


class TestPonytailStar:
    """Plugin instantiation and metadata tests."""

    def test_metadata_fields(self):
        star = PonytailStar()
        assert star.name == "ponytail"
        assert star.version == "1.0.0"
        assert star.priority == 50

    def test_default_mode_is_full(self):
        star = PonytailStar()
        assert star._state.mode == "full"

    def test_collect_handlers_returns_expected_events(self):
        star = PonytailStar()
        handlers = star.collect_handlers()
        # Should register for ON_LLM_REQUEST + ON_MESSAGE_RECEIVED (commands)
        event_types = {h.event_type for h in handlers}
        assert EventType.ON_LLM_REQUEST in event_types
        assert EventType.ON_MESSAGE_RECEIVED in event_types

    def test_is_star_subclass(self):
        assert issubclass(PonytailStar, Star)


class TestLLMInjection:
    """Verify ponytail rules get injected into the system prompt."""

    @pytest.mark.asyncio
    async def test_full_mode_appends_rules(self):
        star = PonytailStar()
        star._state.mode = "full"
        event = LLMRequestEvent(
            type=EventType.ON_LLM_REQUEST,
            system_prompt="You are a helpful assistant.",
        )
        await star.inject_rules(event)
        assert "Ponytail" in event.system_prompt
        assert "ladder" in event.system_prompt.lower()
        assert event.context.get("ponytail_mode") == "full"

    @pytest.mark.asyncio
    async def test_lite_mode_appends_brief_rules(self):
        star = PonytailStar()
        star._state.mode = "lite"
        event = LLMRequestEvent(
            type=EventType.ON_LLM_REQUEST,
            system_prompt="Base prompt.",
        )
        await star.inject_rules(event)
        assert "lite" in event.system_prompt.lower()
        assert event.context.get("ponytail_mode") == "lite"

    @pytest.mark.asyncio
    async def test_ultra_mode_includes_yagni_extremist(self):
        star = PonytailStar()
        star._state.mode = "ultra"
        event = LLMRequestEvent(
            type=EventType.ON_LLM_REQUEST,
            system_prompt="Base.",
        )
        await star.inject_rules(event)
        assert "YAGNI" in event.system_prompt
        assert "extremist" in event.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_off_mode_does_not_inject(self):
        star = PonytailStar()
        star._state.mode = "off"
        event = LLMRequestEvent(
            type=EventType.ON_LLM_REQUEST,
            system_prompt="Original prompt.",
        )
        await star.inject_rules(event)
        assert event.system_prompt == "Original prompt."
        assert "ponytail_mode" not in event.context

    @pytest.mark.asyncio
    async def test_empty_system_prompt_gets_rules(self):
        star = PonytailStar()
        star._state.mode = "full"
        event = LLMRequestEvent(
            type=EventType.ON_LLM_REQUEST,
            system_prompt="",
        )
        await star.inject_rules(event)
        assert "Ponytail" in event.system_prompt
        assert len(event.system_prompt) > 50


class TestModeSwitching:
    """Verify /ponytail command handles mode switching."""

    @pytest.mark.asyncio
    async def test_switch_to_ultra(self):
        star = PonytailStar()
        star._state.mode = "full"
        event = MessageEvent(type=EventType.ON_MESSAGE_RECEIVED, text="/ponytail ultra")
        await star.handle_ponytail(event)
        assert star._state.mode == "ultra"
        assert "ultra" in event.extra.get("reply", "")

    @pytest.mark.asyncio
    async def test_switch_to_off(self):
        star = PonytailStar()
        star._state.mode = "full"
        event = MessageEvent(type=EventType.ON_MESSAGE_RECEIVED, text="/ponytail off")
        await star.handle_ponytail(event)
        assert star._state.mode == "off"

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_error(self):
        star = PonytailStar()
        event = MessageEvent(type=EventType.ON_MESSAGE_RECEIVED, text="/ponytail extreme")
        await star.handle_ponytail(event)
        assert "Unknown" in event.extra.get("reply", "")
        assert star._state.mode == "full"  # unchanged

    @pytest.mark.asyncio
    async def test_no_arg_shows_status(self):
        star = PonytailStar()
        star._state.mode = "full"
        star._state.skipped_count = 5
        star._state.lines_saved = 42
        event = MessageEvent(type=EventType.ON_MESSAGE_RECEIVED, text="/ponytail")
        await star.handle_ponytail(event)
        reply = event.extra.get("reply", "")
        assert "full" in reply
        assert "5" in reply
        assert "42" in reply


class TestOverEngineeringAudit:
    """Verify AST-based audit detects over-engineering patterns."""

    def test_detects_abc_with_single_impl(self):
        code = """
from abc import ABC

class DataStore(ABC):
    def get(self, key): ...
    def set(self, key, val): ...

class SqliteStore(DataStore):
    def get(self, key): return "value"
    def set(self, key, val): pass
"""
        import ast
        tree = ast.parse(code)
        findings = _audit_python_tree(tree, "test.py")
        # SqliteStore is the only concrete impl → should flag
        assert any("DataStore" in f for f in findings)

    def test_no_false_positive_on_multiple_impls(self):
        code = """
from abc import ABC

class DataStore(ABC):
    def get(self, key): ...

class SqliteStore(DataStore):
    def get(self, key): return "v"

class RedisStore(DataStore):
    def get(self, key): return "v"
"""
        import ast
        tree = ast.parse(code)
        findings = _audit_python_tree(tree, "test.py")
        # Two implementations → should NOT flag
        assert not any("DataStore" in f for f in findings)

    def test_detects_factory_with_single_product(self):
        code = """
def create_tool_factory():
    return Tool(name="only_one")
"""
        import ast
        tree = ast.parse(code)
        findings = _audit_python_tree(tree, "test.py")
        assert any("factory" in f.lower() for f in findings)


class TestSkillMdParsing:
    """Verify the SKILL.md file is parseable by skill_loader."""

    def test_parse_skill_file(self, tmp_path):
        skill_file = tmp_path / "ponytail" / "SKILL.md"
        skill_file.parent.mkdir()
        skill_file.write_text(
            "---\n"
            "name: ponytail\n"
            "description: Lazy engineer ladder\n"
            "---\n\n"
            "# Ponytail\n"
            "Stop at the first rung.\n",
            encoding="utf-8",
        )
        result = parse_skill_file(skill_file)
        assert result["name"] == "ponytail"
        assert "Lazy" in result["description"]
        assert "Ponytail" in result["content"]


class TestPluginLoading:
    """Verify the plugin can be loaded by the plugin loader."""

    def test_metadata_yaml_exists(self):
        from pathlib import Path
        meta = Path(__file__).resolve().parent.parent / "huginn" / "plugins" / "ponytail" / "metadata.yaml"
        assert meta.exists()

    def test_conf_schema_exists(self):
        from pathlib import Path
        schema = Path(__file__).resolve().parent.parent / "huginn" / "plugins" / "ponytail" / "_conf_schema.json"
        assert schema.exists()

    def test_skill_md_exists(self):
        from pathlib import Path
        skill = Path(__file__).resolve().parent.parent / "huginn" / "plugins" / "ponytail" / "SKILL.md"
        assert skill.exists()
