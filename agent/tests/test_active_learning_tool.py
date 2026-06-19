"""Tests for the active-learning / synthesis-planning tool."""

from __future__ import annotations

import asyncio
import csv

import numpy as np
import pytest

from huginn.tools.active_learning_tool import ActiveLearningTool
from huginn.types import ToolContext


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "experiments.csv"
    rows = []
    rng = np.random.default_rng(0)
    for _ in range(10):
        temp = float(rng.uniform(300, 800))
        time = float(rng.uniform(1, 10))
        # Synthetic target: higher temp+time -> higher yield
        target = temp * 0.01 + time * 0.5 + rng.normal(0, 0.1)
        rows.append({"temperature": temp, "time": time, "yield": target})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["temperature", "time", "yield"])
        writer.writeheader()
        writer.writerows(rows)
    return path


class TestActiveLearningTool:
    def test_load_csv(self, tmp_path, sample_csv):
        tool = ActiveLearningTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="load_csv",
                    data_path=str(sample_csv),
                    target_column="yield",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )
        assert result.success is True
        assert result.data["rows"] == 10
        assert result.data["target_column"] == "yield"
        assert "temperature" in result.data["feature_columns"]
        assert "time" in result.data["feature_columns"]

    def test_recommend(self, tmp_path, sample_csv):
        tool = ActiveLearningTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="recommend",
                    data_path=str(sample_csv),
                    target_column="yield",
                    feature_columns=["temperature", "time"],
                    bounds={"temperature": (300.0, 800.0), "time": (1.0, 10.0)},
                    n_recommendations=3,
                    maximize=True,
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )
        assert result.success is True
        recs = result.data["recommendations"]
        assert len(recs) == 3
        for r in recs:
            assert "expected_improvement" in r
            assert "predicted_mean" in r
            assert "temperature" in r

    def test_recommend_fails_without_enough_data(self, tmp_path):
        path = tmp_path / "small.csv"
        path.write_text("temperature,yield\n300,1.0\n", encoding="utf-8")
        tool = ActiveLearningTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="recommend",
                    data_path=str(path),
                    target_column="yield",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )
        assert result.success is False
        assert "at least 2" in (result.error or "")
