"""Tests for Phase 3 diff_tool — MathDiffer integration and deep diff fallback."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huginn.tools.diff_tool import DiffTool, DiffToolInput, _deep_diff
from huginn.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="test", workspace=".")


# ── _deep_diff unit tests ───────────────────────────────────────────


class TestDeepDiff:
    def test_identical_dicts(self):
        assert _deep_diff({"a": 1}, {"a": 1}) == []

    def test_value_changed(self):
        changes = _deep_diff({"a": 1}, {"a": 2})
        assert len(changes) == 1
        assert changes[0]["type"] == "value_changed"
        assert changes[0]["old_value"] == 1
        assert changes[0]["new_value"] == 2

    def test_key_added(self):
        changes = _deep_diff({}, {"a": 1})
        assert len(changes) == 1
        assert changes[0]["type"] == "added"

    def test_key_removed(self):
        changes = _deep_diff({"a": 1}, {})
        assert len(changes) == 1
        assert changes[0]["type"] == "removed"
        assert changes[0]["severity"] == "warning"

    def test_nested_diff(self):
        old = {"params": {"encut": 400}}
        new = {"params": {"encut": 520}}
        changes = _deep_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["path"] == "params.encut"

    def test_list_length_change(self):
        changes = _deep_diff([1, 2], [1, 2, 3])
        assert any(c["type"] == "list_length_changed" for c in changes)

    def test_list_element_change(self):
        changes = _deep_diff([1, 2, 3], [1, 9, 3])
        assert any(
            c["type"] == "value_changed" and "[1]" in c["path"]
            for c in changes
        )

    def test_critical_severity_for_equations(self):
        changes = _deep_diff(
            {"equation": "x+y"},
            {"equation": "x+z"},
        )
        assert changes[0]["severity"] == "critical"

    def test_critical_severity_for_boundary(self):
        changes = _deep_diff(
            {"boundary_conditions": "periodic"},
            {"boundary_conditions": "fixed"},
        )
        assert changes[0]["severity"] == "critical"

    def test_critical_severity_for_symmetry(self):
        changes = _deep_diff({"symmetry": "Fm-3m"}, {"symmetry": "Pm-3m"})
        assert changes[0]["severity"] == "critical"


# ── DiffTool tests ──────────────────────────────────────────────────


class TestDiffTool:
    def setup_method(self):
        self.tool = DiffTool()
        self.ctx = _ctx()

    @pytest.mark.asyncio
    async def test_inline_full_diff(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="full",
            inline_a={"encut": 400, "kpoints": [4, 4, 4]},
            inline_b={"encut": 520, "kpoints": [6, 6, 6]},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["summary"]["total_changes"] > 0
        assert result.data["engine"] in ("math_differ", "builtin_deep_diff")

    @pytest.mark.asyncio
    async def test_identical_inline(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="full",
            inline_a={"x": 1},
            inline_b={"x": 1},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["summary"]["total_changes"] == 0
        assert "No differences" in result.data["semantic_summary"]

    @pytest.mark.asyncio
    async def test_parameters_filter(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="parameters",
            inline_a={"encut": 400, "energy": -100},
            inline_b={"encut": 520, "energy": -102},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        # Should only include parameter changes (encut matches "cutoff")
        paths = [c["path"] for c in result.data["changes"]]
        assert any("encut" in p for p in paths)

    @pytest.mark.asyncio
    async def test_results_filter(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="results",
            inline_a={"encut": 400, "energy": -100},
            inline_b={"encut": 520, "energy": -102},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        paths = [c["path"] for c in result.data["changes"]]
        assert any("energy" in p for p in paths)

    @pytest.mark.asyncio
    async def test_file_path_loading(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"param": 1}, f)
            f.flush()
            path_a = f.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"param": 2}, f)
            f.flush()
            path_b = f.name

        args = DiffToolInput(calc_a=path_a, calc_b=path_b, comparison_type="full")
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert result.data["summary"]["total_changes"] > 0

        Path(path_a).unlink()
        Path(path_b).unlink()

    @pytest.mark.asyncio
    async def test_plain_identifier_fallback(self):
        args = DiffToolInput(
            calc_a="run_001", calc_b="run_002",
            comparison_type="full",
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success

    @pytest.mark.asyncio
    async def test_read_only(self):
        args = DiffToolInput(calc_a="a", calc_b="b", comparison_type="full")
        assert self.tool.is_read_only(args) is True

    @pytest.mark.asyncio
    async def test_critical_summary(self):
        args = DiffToolInput(
            calc_a="a", calc_b="b",
            comparison_type="full",
            inline_a={"equation": "x+y"},
            inline_b={"equation": "x+z"},
        )
        result = await self.tool.call(args, self.ctx)
        assert result.success
        assert "critical" in result.data["semantic_summary"].lower()
