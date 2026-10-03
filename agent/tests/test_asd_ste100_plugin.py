"""Tests for the ASD-STE100 controlled-language plugin.

Covers four live surfaces: the deterministic linter, the prompt segment, the
``ste_lint`` tool, and the plugin lifecycle (mount/unmount + loader).
"""

from pathlib import Path

import pytest

from huginn.core_types import ToolContext
from huginn.plugins.asd_ste100.main import (
    AsdSte100Star,
    SEGMENT_NAME,
    TOOL_NAME,
    SteLintInput,
    SteLintTool,
    current_mode,
    ste_prompt_segment,
)
from huginn.plugins.asd_ste100.ste_lint import lint, summarize
from huginn.plugins.prompt_segments import (
    assemble_prompt_segments,
    registered_prompt_segments,
    unregister_prompt_segment,
)
from huginn.tools.registry import ToolRegistry

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "huginn" / "plugins" / "asd_ste100"


def _ctx() -> ToolContext:
    return ToolContext(session_id="t", workspace="/tmp")


class TestLinter:
    def test_hard_violations_detected(self):
        findings, _ = lint("The panel is removed; spin up the job.")
        rules = {f["rule"] for f in findings}
        assert "semicolon" in rules
        assert "phrasal-verb" in rules
        assert "passive-voice" in rules

    def test_hedges_never_flagged(self):
        findings, _ = lint(
            "The request may have failed. It could be a timeout. The disk might have filled."
        )
        assert findings == []

    def test_code_fence_skipped(self):
        findings, _ = lint("```\nx = a; y = b\n```")
        assert findings == []

    def test_synonym_rotation(self):
        findings, _ = lint("Check the config. Then verify the output.")
        rot = [f for f in findings if f["rule"] == "synonym-rotation"]
        assert len(rot) == 1
        assert "'verify' and 'check'" in rot[0]["message"]

    def test_summarize_ok_semantics(self):
        result = summarize("Analyze the log.")
        assert result["ok"] is True
        assert result["hard_count"] == 0

    def test_summarize_hard_count(self):
        result = summarize("The panel is removed; spin up the job.")
        assert result["ok"] is False
        assert result["hard_count"] > 0

    def test_selftest_runs(self):
        from huginn.plugins.asd_ste100.ste_lint import selftest

        selftest()


class TestPromptSegment:
    def test_agents_mode_block(self, monkeypatch):
        monkeypatch.setenv("HUGINN_STE_MODE", "agents")
        text = ste_prompt_segment("normal", "execution", "none", None)
        assert "ASD-STE100" in text
        assert "active voice" in text

    def test_strict_mode_block(self, monkeypatch):
        monkeypatch.setenv("HUGINN_STE_MODE", "strict")
        text = ste_prompt_segment("normal", "execution", "none", None)
        assert "Strict" in text

    def test_flavored_mode_block(self, monkeypatch):
        monkeypatch.setenv("HUGINN_STE_MODE", "flavored")
        text = ste_prompt_segment("normal", "execution", "none", None)
        assert "flavored" in text

    def test_off_mode_returns_empty(self, monkeypatch):
        monkeypatch.setenv("HUGINN_STE_MODE", "off")
        assert ste_prompt_segment("normal", "execution", "none", None) == ""

    def test_invalid_mode_falls_back_to_agents(self, monkeypatch):
        monkeypatch.setenv("HUGINN_STE_MODE", "bogus")
        assert current_mode() == "agents"

    def test_segment_reaches_assembled_prompt(self, monkeypatch):
        monkeypatch.setenv("HUGINN_STE_MODE", "agents")
        from huginn.plugins.prompt_segments import register_prompt_segment

        register_prompt_segment(SEGMENT_NAME, ste_prompt_segment, priority=65)
        try:
            assembled = assemble_prompt_segments("normal", "execution", "none", None)
            assert "ASD-STE100" in assembled
        finally:
            unregister_prompt_segment(SEGMENT_NAME)


class TestTool:
    @pytest.mark.asyncio
    async def test_flags_hard_violation(self):
        tool = SteLintTool()
        result = await tool.call(SteLintInput(text="The panel is removed; spin up the job."), _ctx())
        assert result.success is True
        assert result.data["hard_count"] > 0

    @pytest.mark.asyncio
    async def test_clean_text_passes(self):
        tool = SteLintTool()
        result = await tool.call(SteLintInput(text="Analyze the log."), _ctx())
        assert result.success is True
        assert result.data["hard_count"] == 0
        assert "no structural violations" in result.data["summary"]

    @pytest.mark.asyncio
    async def test_path_input(self, tmp_path):
        tool = SteLintTool()
        target = tmp_path / "draft.md"
        target.write_text("The panel is removed; spin up the job.", encoding="utf-8")
        result = await tool.call(SteLintInput(path=str(target)), _ctx())
        assert result.success is True
        assert result.data["hard_count"] > 0

    @pytest.mark.asyncio
    async def test_missing_input_errors(self):
        tool = SteLintTool()
        result = await tool.call(SteLintInput(), _ctx())
        assert result.success is False
        assert result.error

    @pytest.mark.asyncio
    async def test_unreadable_path_errors(self):
        tool = SteLintTool()
        result = await tool.call(SteLintInput(path="/no/such/file.md"), _ctx())
        assert result.success is False
        assert "cannot read" in result.error


class TestLifecycle:
    def test_metadata_fields(self):
        star = AsdSte100Star()
        assert star.name == SEGMENT_NAME == "asd_ste100"
        assert star.version == "1.0.0"

    def test_is_star_subclass(self):
        from huginn.api.star import Star

        assert issubclass(AsdSte100Star, Star)

    @pytest.mark.asyncio
    async def test_mount_and_unmount(self):
        star = AsdSte100Star()
        await star.on_load()
        try:
            assert SEGMENT_NAME in registered_prompt_segments()
            assert ToolRegistry.get(TOOL_NAME) is not None
        finally:
            await star.on_unload()
        assert SEGMENT_NAME not in registered_prompt_segments()
        assert ToolRegistry.get(TOOL_NAME) is None


class TestPluginLoading:
    def test_required_files_exist(self):
        assert (PLUGIN_DIR / "metadata.yaml").is_file()
        assert (PLUGIN_DIR / "main.py").is_file()
        assert (PLUGIN_DIR / "SKILL.md").is_file()
        assert (PLUGIN_DIR / "ste_lint.py").is_file()

    @pytest.mark.asyncio
    async def test_loader_mounts_plugin(self):
        import asyncio

        from huginn.plugins.loader import PluginLoader

        loader = PluginLoader(plugins_dir=str(PLUGIN_DIR.parent))
        try:
            meta = await loader.load_one_async(PLUGIN_DIR)
            assert meta is not None
            assert meta.name == "asd_ste100"
            assert "asd_ste100" in loader.list_loaded()
            assert ToolRegistry.get(TOOL_NAME) is not None
            assert SEGMENT_NAME in registered_prompt_segments()
        finally:
            loader.unload("asd_ste100")
            # Inside a running loop the loader schedules on_unload fire-and-forget
            # (ensure_future, not awaited); yield so its cleanup runs before we
            # assert.
            await asyncio.sleep(0)
        assert ToolRegistry.get(TOOL_NAME) is None
        assert SEGMENT_NAME not in registered_prompt_segments()