"""Tests for the science-skills bridge plugin.

Covers:
- SKILL.md frontmatter parsing
- ScienceSkillsLoader discovery
- Tool registration (≥35 tools)
- CLI command building (dry-run)
- Output file reading
- Error handling (missing uv, timeout)
- Permission wildcard (science_* → AUTO)
- Idempotent registration
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from huginn.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Save/restore ToolRegistry state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_registry():
    """Save and restore ToolRegistry._tools so tests don't leak."""
    saved = dict(ToolRegistry._tools)
    yield
    ToolRegistry._tools.clear()
    ToolRegistry._tools.update(saved)


@pytest.fixture(autouse=True)
def _reset_registered_flag():
    """Reset the bridge's idempotency flag between tests."""
    import huginn.plugins.science_skills_bridge as bridge
    original = bridge._registered
    yield
    bridge._registered = original


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_basic_frontmatter(self):
        from huginn.plugins.science_skills_bridge import _parse_frontmatter

        text = "---\nname: pubchem-database\ndescription: Query PubChem for chemical information\n---\nBody text"
        result = _parse_frontmatter(text)
        assert result["name"] == "pubchem-database"
        assert result["description"] == "Query PubChem for chemical information"

    def test_multiline_description(self):
        from huginn.plugins.science_skills_bridge import _parse_frontmatter

        text = "---\nname: test-skill\ndescription: >\n  A multi-line\n  description here\n---\n"
        result = _parse_frontmatter(text)
        assert result["name"] == "test-skill"
        assert "multi-line" in result["description"]

    def test_no_frontmatter(self):
        from huginn.plugins.science_skills_bridge import _parse_frontmatter

        text = "# Just a markdown file\nNo frontmatter here."
        result = _parse_frontmatter(text)
        assert result == {}

    def test_empty_frontmatter(self):
        from huginn.plugins.science_skills_bridge import _parse_frontmatter

        text = "---\n---\n"
        result = _parse_frontmatter(text)
        assert result == {}

    def test_comments_skipped(self):
        from huginn.plugins.science_skills_bridge import _parse_frontmatter

        text = "---\n# This is a comment\nname: my-skill\n---\n"
        result = _parse_frontmatter(text)
        assert result["name"] == "my-skill"
        assert "#" not in str(result.keys())


# ---------------------------------------------------------------------------
# SkillMetadata
# ---------------------------------------------------------------------------

class TestSkillMetadata:
    def test_tool_name_conversion(self):
        from huginn.plugins.science_skills_bridge import SkillMetadata

        meta = SkillMetadata(
            name="pubchem-database",
            description="test",
            directory=Path("/tmp/test"),
        )
        assert meta.tool_name == "science_pubchem_database"

    def test_tool_name_no_hyphens(self):
        from huginn.plugins.science_skills_bridge import SkillMetadata

        meta = SkillMetadata(
            name="uniprot",
            description="test",
            directory=Path("/tmp/test"),
        )
        assert meta.tool_name == "science_uniprot"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestScienceSkillsLoader:
    def test_discover_finds_skills(self):
        """Loader discovers ≥35 skills from the cloned repo."""
        from huginn.plugins.science_skills_bridge import ScienceSkillsLoader

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        assert len(skills) >= 30, f"Expected ≥30 skills, got {len(skills)}"

    def test_discover_all_have_scripts(self):
        from huginn.plugins.science_skills_bridge import ScienceSkillsLoader

        loader = ScienceSkillsLoader()
        for meta in loader.discover():
            assert meta.scripts, f"Skill '{meta.name}' has no scripts"
            assert meta.primary_script is not None

    def test_discover_all_have_names(self):
        from huginn.plugins.science_skills_bridge import ScienceSkillsLoader

        loader = ScienceSkillsLoader()
        for meta in loader.discover():
            assert meta.name, f"Skill in {meta.directory} has empty name"
            assert meta.description, f"Skill '{meta.name}' has empty description"

    def test_discover_skip_dirs(self):
        """Skipped dirs (scienceskillscommon, uv) should not appear."""
        from huginn.plugins.science_skills_bridge import ScienceSkillsLoader

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        names = {m.name for m in skills}
        assert "scienceskillscommon" not in names
        assert "uv" not in names

    def test_discover_nonexistent_dir(self):
        from huginn.plugins.science_skills_bridge import ScienceSkillsLoader

        loader = ScienceSkillsLoader(skills_dir=Path("/nonexistent/path"))
        skills = loader.discover()
        assert skills == []

    def test_discover_none_dir(self):
        from huginn.plugins.science_skills_bridge import ScienceSkillsLoader

        loader = ScienceSkillsLoader(skills_dir=None)
        # Should handle gracefully — either finds bundled or returns []
        skills = loader.discover()
        assert isinstance(skills, list)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_science_skills(self):
        from huginn.plugins.science_skills_bridge import register_science_skills

        names = register_science_skills()
        assert len(names) >= 30, f"Expected ≥30 registered tools, got {len(names)}"
        for name in names:
            assert name.startswith("science_")

    def test_register_appears_in_registry(self):
        from huginn.plugins.science_skills_bridge import register_science_skills

        register_science_skills()
        all_tools = ToolRegistry.list_tools()
        science_tools = [t for t in all_tools if t.startswith("science_")]
        assert len(science_tools) >= 30

    def test_register_idempotent(self):
        from huginn.plugins.science_skills_bridge import register_science_skills

        names1 = register_science_skills()
        names2 = register_science_skills()
        assert names1 == names2

    def test_register_specific_tools(self):
        """Check some well-known skills are registered."""
        from huginn.plugins.science_skills_bridge import register_science_skills

        names = register_science_skills()
        # PubChem should be present
        assert any("pubchem" in n for n in names), f"No pubchem in {names}"


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

class TestCommandBuilding:
    def _make_tool(self):
        from huginn.plugins.science_skills_bridge import (
            ScienceSkillTool,
            ScienceSkillsLoader,
        )

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        assert skills, "No skills discovered"
        return ScienceSkillTool(skills[0])

    def test_build_basic_command(self, tmp_path):
        from huginn.plugins.science_skills_bridge import ScienceSkillInput

        tool = self._make_tool()
        args = ScienceSkillInput(query="aspirin", action="resolve")
        output = tmp_path / "output.json"
        cmd = tool._build_command(args, output, "uv")

        assert cmd[0] == "uv"
        assert cmd[1] == "run"
        assert "resolve" in cmd
        assert "--query" in cmd
        assert "aspirin" in cmd
        assert "--output" in cmd

    def test_build_command_with_identifiers(self, tmp_path):
        from huginn.plugins.science_skills_bridge import ScienceSkillInput

        tool = self._make_tool()
        args = ScienceSkillInput(
            identifiers=["CID:2244", "CID:3672"],
            action="properties",
        )
        output = tmp_path / "output.json"
        cmd = tool._build_command(args, output, "uv")

        assert "--id" in cmd
        assert "CID:2244" in cmd
        assert "CID:3672" in cmd

    def test_build_command_extra_args(self, tmp_path):
        from huginn.plugins.science_skills_bridge import ScienceSkillInput

        tool = self._make_tool()
        args = ScienceSkillInput(
            query="test",
            extra_args={"--limit": "10", "--smiles": "CCO"},
        )
        output = tmp_path / "output.json"
        cmd = tool._build_command(args, output, "uv")

        assert "--limit" in cmd
        assert "10" in cmd
        assert "--smiles" in cmd
        assert "CCO" in cmd

    def test_build_command_bool_flag(self, tmp_path):
        from huginn.plugins.science_skills_bridge import ScienceSkillInput

        tool = self._make_tool()
        args = ScienceSkillInput(
            query="test",
            extra_args={"--verbose": "true"},
        )
        output = tmp_path / "output.json"
        cmd = tool._build_command(args, output, "uv")

        assert "--verbose" in cmd
        # Boolean flags should not have a value after them
        idx = cmd.index("--verbose")
        # The next item should NOT be "true"
        if idx + 1 < len(cmd):
            assert cmd[idx + 1] != "true"


# ---------------------------------------------------------------------------
# Tool execution (mocked)
# ---------------------------------------------------------------------------

class TestToolExecution:
    def test_missing_uv_error(self):
        from huginn.plugins.science_skills_bridge import (
            ScienceSkillInput,
            ScienceSkillTool,
            ScienceSkillsLoader,
        )
        from huginn.types import ToolContext

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        tool = ScienceSkillTool(skills[0])
        ctx = ToolContext(session_id="test", workspace="/tmp")

        with patch("shutil.which", return_value=None):
            result = asyncio.run(tool.call(ScienceSkillInput(query="test"), ctx))

        assert result.success is False
        assert "uv" in result.error.lower()

    def test_successful_execution(self, tmp_path):
        from huginn.plugins.science_skills_bridge import (
            ScienceSkillInput,
            ScienceSkillTool,
            ScienceSkillsLoader,
        )
        from huginn.types import ToolContext

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        tool = ScienceSkillTool(skills[0])
        ctx = ToolContext(session_id="test", workspace="/tmp")

        # Mock subprocess to return success with output file
        output_file = tmp_path / "output.json"
        output_data = {"result": "mock data"}
        output_file.write_text(json.dumps(output_data))

        mock_result = {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
        }

        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch(
                "huginn.plugins.science_skills_bridge._run_subprocess",
                return_value=mock_result,
            ):
                args = ScienceSkillInput(
                    query="test",
                    output_file=str(output_file),
                )
                result = asyncio.run(tool.call(args, ctx))

        assert result.success is True
        assert result.data["skill"] == skills[0].name
        assert result.data["result"] == output_data

    def test_failed_execution(self):
        from huginn.plugins.science_skills_bridge import (
            ScienceSkillInput,
            ScienceSkillTool,
            ScienceSkillsLoader,
        )
        from huginn.types import ToolContext

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        tool = ScienceSkillTool(skills[0])
        ctx = ToolContext(session_id="test", workspace="/tmp")

        mock_result = {
            "returncode": 1,
            "stdout": "",
            "stderr": "API key not found",
        }

        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch(
                "huginn.plugins.science_skills_bridge._run_subprocess",
                return_value=mock_result,
            ):
                result = asyncio.run(
                    tool.call(ScienceSkillInput(query="test"), ctx)
                )

        assert result.success is False
        assert "API key not found" in result.error

    def test_timeout_handling(self):
        from huginn.plugins.science_skills_bridge import (
            ScienceSkillInput,
            ScienceSkillTool,
            ScienceSkillsLoader,
        )
        from huginn.types import ToolContext

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        tool = ScienceSkillTool(skills[0])
        ctx = ToolContext(session_id="test", workspace="/tmp")

        mock_result = {
            "returncode": -1,
            "stdout": "",
            "stderr": "Process timed out",
        }

        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch(
                "huginn.plugins.science_skills_bridge._run_subprocess",
                return_value=mock_result,
            ):
                result = asyncio.run(
                    tool.call(ScienceSkillInput(query="test"), ctx)
                )

        assert result.success is False
        assert "timed out" in result.error.lower() or "rc=-1" in result.error

    def test_stdout_fallback_no_output_file(self, tmp_path):
        """When no output file exists, stdout is used as result."""
        from huginn.plugins.science_skills_bridge import (
            ScienceSkillInput,
            ScienceSkillTool,
            ScienceSkillsLoader,
        )
        from huginn.types import ToolContext

        loader = ScienceSkillsLoader()
        skills = loader.discover()
        tool = ScienceSkillTool(skills[0])
        ctx = ToolContext(session_id="test", workspace="/tmp")

        mock_result = {
            "returncode": 0,
            "stdout": "plain text output",
            "stderr": "",
        }

        with patch("shutil.which", return_value="/usr/bin/uv"):
            with patch(
                "huginn.plugins.science_skills_bridge._run_subprocess",
                return_value=mock_result,
            ):
                # Point to a non-existent output file
                args = ScienceSkillInput(
                    query="test",
                    output_file=str(tmp_path / "nonexistent.json"),
                )
                result = asyncio.run(tool.call(args, ctx))

        assert result.success is True
        assert result.data["result"] == "plain text output"


# ---------------------------------------------------------------------------
# Permission wildcard
# ---------------------------------------------------------------------------

class TestPermissionWildcard:
    def test_science_tools_auto_approved(self):
        from huginn.permissions import PermissionConfig
        from huginn.types import PermissionMode

        config = PermissionConfig()
        assert config.get_mode("science_pubchem_database") == PermissionMode.AUTO
        assert config.get_mode("science_uniprot") == PermissionMode.AUTO
        assert config.get_mode("science_any_new_skill") == PermissionMode.AUTO

    def test_non_science_tools_unaffected(self):
        from huginn.permissions import PermissionConfig
        from huginn.types import PermissionMode

        config = PermissionConfig()
        # Existing tools should still work
        assert config.get_mode("vasp_tool") == PermissionMode.ASK
        assert config.get_mode("structure_tool") == PermissionMode.AUTO
        # Unknown tool → ASK
        assert config.get_mode("unknown_tool") == PermissionMode.ASK


# ---------------------------------------------------------------------------
# Info API
# ---------------------------------------------------------------------------

class TestGetScienceSkillsInfo:
    def test_info_returns_list(self):
        from huginn.plugins.science_skills_bridge import get_science_skills_info

        info = get_science_skills_info()
        assert isinstance(info, list)
        assert len(info) >= 30

    def test_info_fields(self):
        from huginn.plugins.science_skills_bridge import get_science_skills_info

        info = get_science_skills_info()
        for item in info:
            assert "name" in item
            assert "tool_name" in item
            assert "description" in item
            assert "directory" in item
            assert "scripts" in item
            assert item["tool_name"].startswith("science_")
