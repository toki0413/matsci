"""Tests for the external .md description loader on HuginnTool.

Covers: auto-load when description is empty/placeholder, leave-hardcoded-
description-alone (backward compat), per-name caching, instance-attr shadowing,
and the missing-file no-op path.
"""

from __future__ import annotations

import pytest

from huginn.tools.base import HuginnTool


@pytest.fixture(autouse=True)
def _clear_desc_cache():
    # _description_cache is class-level and would otherwise leak between tests.
    HuginnTool._description_cache.clear()
    yield
    HuginnTool._description_cache.clear()


class _NoDescTool(HuginnTool):
    name = "vasp_tool"  # has a sibling descriptions/vasp_tool.md

    async def _execute(self, args, context):  # noqa: ANN001
        from huginn.types import ToolResult

        return ToolResult(data=None, success=True)


class _ExplicitTool(HuginnTool):
    name = "lammps_tool"  # also has a .md, but we set our own below
    description = "my own text"

    async def _execute(self, args, context):  # noqa: ANN001
        from huginn.types import ToolResult

        return ToolResult(data=None, success=True)


class _PlaceholderTool(HuginnTool):
    name = "fep_tool"  # has a sibling descriptions/fep_tool.md
    description = "TODO"

    async def _execute(self, args, context):  # noqa: ANN001
        from huginn.types import ToolResult

        return ToolResult(data=None, success=True)


class _MissingMdTool(HuginnTool):
    name = "no_such_tool_anywhere"

    async def _execute(self, args, context):  # noqa: ANN001
        from huginn.types import ToolResult

        return ToolResult(data=None, success=True)


class TestDescriptionLoader:
    def test_empty_description_loads_from_md(self):
        tool = _NoDescTool()
        assert "VASP DFT" in tool.description

    def test_class_level_description_unchanged(self):
        # only the instance attr is set; the class default stays empty
        _NoDescTool()
        assert _NoDescTool.description == ""

    def test_explicit_description_not_overwritten(self):
        tool = _ExplicitTool()
        assert tool.description == "my own text"

    def test_placeholder_triggers_load(self):
        tool = _PlaceholderTool()
        assert "free energy" in tool.description.lower()

    def test_cache_populated_and_reused(self):
        first = _NoDescTool()
        assert "vasp_tool" in HuginnTool._description_cache
        second = _NoDescTool()
        assert second.description == first.description

    def test_missing_md_stays_empty_and_caches_none(self):
        tool = _MissingMdTool()
        assert tool.description == ""
        assert HuginnTool._description_cache["no_such_tool_anywhere"] is None
