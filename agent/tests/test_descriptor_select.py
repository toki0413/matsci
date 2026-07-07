"""Tests for DescriptorTool select action — SR -> descriptor feature selection."""

from __future__ import annotations

import pytest

from huginn.tools.descriptor_tool import DescriptorTool
from huginn.types import ToolContext


@pytest.fixture
def tool():
    return DescriptorTool()


@pytest.fixture
def context(tmp_path):
    return ToolContext(session_id="test", workspace=str(tmp_path))


@pytest.mark.asyncio
async def test_select_basic_ranking(tool, context):
    # Five descriptors with distinct importances — order must follow |score|
    importances = {
        "magpie_mean_Number": 0.42,
        "magpie_mean_Electronegativity": 0.18,
        "magpie_mean_CovalentRadius": 0.27,
        "magpie_mean_AtomicWeight": 0.09,
        "magpie_mean_Row": 0.04,
    }
    result = await tool.call(
        tool.input_schema(action="select", feature_importances=importances),
        context,
    )
    assert result.success is True
    data = result.data
    names = [d["descriptor"] for d in data["selected_descriptors"]]
    assert names == [
        "magpie_mean_Number",
        "magpie_mean_CovalentRadius",
        "magpie_mean_Electronegativity",
        "magpie_mean_AtomicWeight",
        "magpie_mean_Row",
    ]
    assert data["n_total"] == 5
    assert data["n_selected"] == 5


@pytest.mark.asyncio
async def test_select_top_k(tool, context):
    importances = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    result = await tool.call(
        tool.input_schema(
            action="select", feature_importances=importances, top_k=2
        ),
        context,
    )
    assert result.success is True
    data = result.data
    assert data["n_selected"] == 2
    assert [d["descriptor"] for d in data["selected_descriptors"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_select_threshold(tool, context):
    importances = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    result = await tool.call(
        tool.input_schema(
            action="select", feature_importances=importances, threshold=0.1
        ),
        context,
    )
    assert result.success is True
    data = result.data
    # Only a (0.5), b (0.3), c (0.15) clear the |score| >= 0.1 bar
    names = [d["descriptor"] for d in data["selected_descriptors"]]
    assert names == ["a", "b", "c"]
    assert data["n_total"] == 3
    assert data["n_selected"] == 3


@pytest.mark.asyncio
async def test_select_descriptor_names_subset(tool, context):
    importances = {
        "feat_A": 0.5,
        "feat_B": 0.3,
        "feat_C": 0.15,
        "feat_D": 0.05,
    }
    # Only rank a subset — feat_B and feat_D should be dropped before ranking
    result = await tool.call(
        tool.input_schema(
            action="select",
            feature_importances=importances,
            descriptor_names=["feat_A", "feat_C"],
        ),
        context,
    )
    assert result.success is True
    data = result.data
    names = [d["descriptor"] for d in data["selected_descriptors"]]
    assert names == ["feat_A", "feat_C"]
    assert data["n_total"] == 2


@pytest.mark.asyncio
async def test_select_empty_input(tool, context):
    result = await tool.call(
        tool.input_schema(action="select", feature_importances=None),
        context,
    )
    assert result.success is False
    assert "feature_importances is required" in result.error

    # Empty dict is the same as missing — should fail gracefully, not crash
    result_empty = await tool.call(
        tool.input_schema(action="select", feature_importances={}),
        context,
    )
    assert result_empty.success is False
    assert "feature_importances is required" in result_empty.error


@pytest.mark.asyncio
async def test_select_cumulative_importance(tool, context):
    # scores sum to 1.0 so normalised values are easy to check by hand
    importances = {"a": 0.6, "b": 0.3, "c": 0.1}
    result = await tool.call(
        tool.input_schema(action="select", feature_importances=importances),
        context,
    )
    assert result.success is True
    sel = result.data["selected_descriptors"]
    # Cumulative should be a running sum of the per-row normalised importances
    assert sel[0]["normalized_importance"] == pytest.approx(0.6)
    assert sel[0]["cumulative_importance"] == pytest.approx(0.6)
    assert sel[1]["normalized_importance"] == pytest.approx(0.3)
    assert sel[1]["cumulative_importance"] == pytest.approx(0.9)
    assert sel[2]["normalized_importance"] == pytest.approx(0.1)
    assert sel[2]["cumulative_importance"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_select_negative_scores_ranked_by_abs(tool, context):
    # Importance can be signed (e.g. correlation); ranking uses |score|
    importances = {"a": -0.6, "b": 0.4, "c": 0.1}
    result = await tool.call(
        tool.input_schema(action="select", feature_importances=importances),
        context,
    )
    assert result.success is True
    names = [d["descriptor"] for d in result.data["selected_descriptors"]]
    assert names == ["a", "b", "c"]
    # The original (signed) score is preserved in the output
    assert result.data["selected_descriptors"][0]["importance"] == pytest.approx(-0.6)
