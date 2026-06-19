"""Tests for ExperimentalDataTool."""

from __future__ import annotations

import numpy as np
import pytest

from huginn.tools.experimental_data_tool import ExperimentalDataTool
from huginn.types import ToolContext


@pytest.fixture
def tool():
    return ExperimentalDataTool()


@pytest.fixture
def context(tmp_path):
    return ToolContext(session_id="test", workspace=str(tmp_path))


@pytest.mark.asyncio
async def test_xrd_load(tool, context, tmp_path):
    path = tmp_path / "xrd.txt"
    lines = ["# 2theta intensity", "10.0 100", "20.0 500", "30.0 120"]
    path.write_text("\n".join(lines), encoding="utf-8")

    result = await tool.call(
        tool.input_schema(action="xrd_load", file_path=str(path)), context
    )
    assert result.success is True
    assert result.data["count"] == 3
    assert result.data["data"]["intensity_max"] == 500.0


@pytest.mark.asyncio
async def test_xrd_peaks(tool, context):
    two_theta = np.linspace(10, 80, 1000).tolist()
    intensity = np.zeros(1000).tolist()
    # Inject two gaussian-like peaks
    for i, t in enumerate(two_theta):
        intensity[i] += 100 * np.exp(-((t - 25.0) ** 2) / 0.5)
        intensity[i] += 50 * np.exp(-((t - 45.0) ** 2) / 0.5)

    result = await tool.call(
        tool.input_schema(
            action="xrd_peaks", two_theta=two_theta, intensity=intensity, max_peaks=5
        ),
        context,
    )
    assert result.success is True
    peaks = result.data["data"]["peaks"]
    assert len(peaks) >= 2
    # Highest peak near 25 degrees
    assert abs(peaks[0]["two_theta"] - 25.0) < 1.0


@pytest.mark.asyncio
async def test_xrd_d_spacing(tool, context):
    two_theta = [27.44, 28.44, 29.44]  # Si (111) approx, peak in middle
    intensity = [10.0, 1000.0, 10.0]
    result = await tool.call(
        tool.input_schema(
            action="xrd_d_spacing", two_theta=two_theta, intensity=intensity
        ),
        context,
    )
    assert result.success is True
    d = result.data["data"]["d_spacings"][0]["d_spacing_A"]
    assert 3.1 < d < 3.2


@pytest.mark.asyncio
async def test_image_metadata_not_found(tool, context):
    result = await tool.call(
        tool.input_schema(action="image_metadata", file_path="missing.tif"), context
    )
    assert result.success is False
    assert "not found" in result.error.lower() or "File not found" in result.error
