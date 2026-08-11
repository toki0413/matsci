"""Tests for the characterization data tool."""

from __future__ import annotations

import asyncio
import csv
import json
import os

import numpy as np
import pytest

from huginn.core_types import ToolContext
from huginn.tools.characterization_tool import CharacterizationTool

# CI 上 scipy 可能安装了但 scipy.signal 不可用 (版本不兼容/安装不完整).
# importorskip 只检查 `import scipy`, 不检查子模块, 改用 CI 跳过.
_skip_ci = os.environ.get("HUGINN_CI", "").lower() in ("1", "true", "yes")


class TestCharacterizationTool:
    pytestmark = pytest.mark.skipif(_skip_ci, reason="scipy.signal unavailable on CI")

    def test_xrd_peak_detect(self, tmp_path):
        path = tmp_path / "xrd.csv"
        x = np.linspace(10, 80, 500)
        y = np.zeros_like(x)
        # Add three Lorentzian peaks
        for center, amp in [(25.0, 100.0), (45.0, 80.0), (65.0, 60.0)]:
            y += amp / (1 + ((x - center) / 1.5) ** 2)
        y += np.random.default_rng(0).normal(0, 1, size=x.shape)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["2theta", "intensity"])
            for xi, yi in zip(x, y):
                writer.writerow([xi, yi])

        tool = CharacterizationTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="xrd_peak_detect",
                    data_path=str(path),
                    parameters={"threshold": 10.0, "distance": 10},
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        peaks = result.data["peaks"]
        assert len(peaks) >= 3
        positions = [p["2theta"] for p in peaks]
        assert any(23 < p < 27 for p in positions)
        assert any(43 < p < 47 for p in positions)

    def test_spectroscopy_peak_fit(self, tmp_path):
        path = tmp_path / "raman.csv"
        x = np.linspace(100, 1200, 800)
        y = np.zeros_like(x)
        for center, amp in [(520.0, 100.0), (960.0, 60.0)]:
            y += amp * np.exp(-0.5 * ((x - center) / 5) ** 2)
        y += np.random.default_rng(1).normal(0, 1, size=x.shape)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wavenumber", "intensity"])
            for xi, yi in zip(x, y):
                writer.writerow([xi, yi])

        tool = CharacterizationTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="spectroscopy_peak_fit",
                    data_path=str(path),
                    parameters={"threshold": 10.0, "distance": 10},
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        peaks = result.data["peaks"]
        positions = [p["position"] for p in peaks]
        assert any(510 < p < 530 for p in positions)

    def test_json_input(self, tmp_path):
        path = tmp_path / "xrd.json"
        x = np.linspace(10, 80, 200)
        y = 100 * np.exp(-0.5 * ((x - 30.0) / 2) ** 2)
        path.write_text(
            json.dumps({"2theta": x.tolist(), "intensity": y.tolist()}),
            encoding="utf-8",
        )

        tool = CharacterizationTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="xrd_peak_detect",
                    data_path=str(path),
                    parameters={"threshold": 5.0},
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert result.data["n_peaks"] >= 1

    def test_missing_file(self, tmp_path):
        tool = CharacterizationTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="xrd_peak_detect",
                    data_path=str(tmp_path / "missing.csv"),
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )
        assert result.success is False
