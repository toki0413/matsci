"""Phase 5b Pareto 多目标 BO 测试.

5 测:
  1. fast_non_dominated_sort 简单 (4 点, 2 前沿)
  2. crowding_distance 边界 (2 点, inf)
  3. hypervolume 已知解 (2D, 手算验证)
  4. pareto_suggest 多目标 (y_multi 2 维)
  5. y_multi=None 退回单目标 (走原 suggest)
"""
from __future__ import annotations

import numpy as np
import pytest

from huginn.tools.sci.gp_tool import (
    GPTool,
    crowding_distance,
    fast_non_dominated_sort,
    hypervolume,
)


class TestFastNonDominatedSort:
    def test_simple_two_fronts(self) -> None:
        # 4 点: A=(1,1) B=(2,2) 在前沿1; C=(3,3) D=(4,4) 在前沿2
        y = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        fronts = fast_non_dominated_sort(y)
        assert len(fronts) >= 1
        # A 和 B 互不支配 (A 支配 B? A<=B 且 A<B -> 是! A=(1,1) < B=(2,2))
        # 所以前沿1 只有 A, 前沿2 只有 B, 前沿3 只有 C, 前沿4 只有 D
        # A 支配 B,C,D; B 支配 C,D; C 支配 D
        assert len(fronts) == 4
        assert set(fronts[0].tolist()) == {0}  # A
        assert set(fronts[1].tolist()) == {1}  # B

    def test_non_dominated_pair(self) -> None:
        # A=(1,3) B=(3,1): 互不支配, 都在前沿1
        y = np.array([[1.0, 3.0], [3.0, 1.0]])
        fronts = fast_non_dominated_sort(y)
        assert len(fronts) == 1
        assert set(fronts[0].tolist()) == {0, 1}


class TestCrowdingDistance:
    def test_two_points_inf(self) -> None:
        y = np.array([[1.0, 3.0], [3.0, 1.0]])
        cd = crowding_distance(y)
        assert len(cd) == 2
        assert np.all(np.isinf(cd))

    def test_three_points_middle_finite(self) -> None:
        y = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        cd = crowding_distance(y)
        # 边界 inf, 中间有限
        assert np.isinf(cd[0])
        assert np.isinf(cd[2])
        assert np.isfinite(cd[1])


class TestHypervolume:
    def test_2d_known(self) -> None:
        # ref=(10,10), 单点 (1,1): HV = (10-1)*(10-1) = 81
        pts = np.array([[1.0, 1.0]])
        ref = np.array([10.0, 10.0])
        hv = hypervolume(pts, ref)
        assert abs(hv - 81.0) < 1e-6

    def test_2d_two_points(self) -> None:
        # ref=(10,10), pts=(1,5) 和 (5,1)
        # 排序后: (1,5) 宽=5-1=4, 高=10-5=5 -> 20
        #         (5,1) 宽=10-5=5, 高=10-1=9 -> 45
        # HV = 20 + 45 = 65
        pts = np.array([[1.0, 5.0], [5.0, 1.0]])
        ref = np.array([10.0, 10.0])
        hv = hypervolume(pts, ref)
        assert abs(hv - 65.0) < 1e-6


class TestParetoSuggest:
    def test_multi_objective(self) -> None:
        tool = GPTool()
        # 4 个样本, 2 目标 (最小化)
        result = tool.call(
            {
                "action": "pareto_suggest",
                "y_multi": [[1.0, 5.0], [5.0, 1.0], [3.0, 3.0], [2.0, 2.0]],
                "reference_point": [10.0, 10.0],
                "population_x": [[0.0], [1.0], [2.0], [3.0]],
            },
            context=None,
        )
        assert result.success, f"pareto_suggest failed: {result.error}"
        data = result.data
        assert "pareto_front" in data
        assert data["n_fronts"] >= 1
        assert data["hypervolume"] is not None
        assert data["hypervolume"] > 0
        assert data["suggested_point"] is not None

    def test_y_multi_none_falls_back(self) -> None:
        """没给 y_multi -> 走单目标 suggest."""
        tool = GPTool()
        result = tool.call(
            {
                "action": "pareto_suggest",
                "X": [[0.0], [1.0], [2.0]],
                "y": [1.0, 0.5, 2.0],
                "X_new": [[0.5], [1.5]],
            },
            context=None,
        )
        assert result.success, f"fallback suggest failed: {result.error}"
        # 单目标 suggest 返回 suggested_index + acquisition
        assert "suggested_index" in result.data
