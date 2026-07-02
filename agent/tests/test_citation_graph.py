"""Phase 4b 引文图 (citation_graph) 测试.

4 测:
  1. 种子解析失败报错 (没 doi/arxiv_id/paper)
  2. BFS 单层 (mock S2 references API 返回 1 层)
  3. BFS 双层 + max_nodes 触顶 (mock 2 层, max_nodes 卡住)
  4. S2 API 失败降级 (mock _http_get_json 抛错, 仍返回部分图 + errors)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from huginn.tools.literature.tool import LiteratureInput, LiteratureTool


class TestCitationGraph:
    """citation_graph action BFS 行为."""

    @pytest.mark.asyncio
    async def test_seed_resolution_failure_returns_error(self) -> None:
        """没给 doi/arxiv_id/paper → 种子解析失败, 返回 success=False."""
        tool = LiteratureTool()
        args = LiteratureInput(action="citation_graph")
        result = await tool.call(args, context=None)
        assert not result.success
        assert "无法解析" in (result.error or "")

    @pytest.mark.asyncio
    async def test_bfs_single_layer(self) -> None:
        """mock S2 references API 返回 1 层 2 个引用, 验证 nodes/edges 正确."""
        tool = LiteratureTool()
        args = LiteratureInput(
            action="citation_graph",
            doi="10.1000/test1",
            max_depth=1,
            max_nodes=50,
        )
        # S2 references API 返回结构: {"data": [{"citedPaper": {...}}, ...]}
        mock_response = {
            "data": [
                {
                    "citedPaper": {
                        "paperId": "child1",
                        "title": "Cited Paper 1",
                        "year": 2020,
                        "externalIds": {"DOI": "10.1000/c1"},
                    }
                },
                {
                    "citedPaper": {
                        "paperId": "child2",
                        "title": "Cited Paper 2",
                        "year": 2019,
                        "externalIds": {"DOI": "10.1000/c2"},
                    }
                },
            ]
        }
        with patch(
            "huginn.tools.literature.tool._http_get_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tool.call(args, context=None)
        assert result.success
        data = result.data
        assert data["action"] == "citation_graph"
        assert data["seed_paper_id"] == "DOI:10.1000/test1"
        assert data["depth_reached"] == 1
        # 种子 + 2 个子节点 = 3
        assert data["n_unique_papers"] == 3
        assert data["n_edges"] == 2
        assert data["truncated"] is False
        # 节点深度正确
        depths = [n["depth"] for n in data["nodes"]]
        assert 0 in depths  # 种子
        assert 1 in depths  # 第一层

    @pytest.mark.asyncio
    async def test_bfs_max_nodes_truncation(self) -> None:
        """max_nodes=5 时, 种子+4 个子节点触顶, 第 5 个子节点被截断, truncated=True."""
        tool = LiteratureTool()
        args = LiteratureInput(
            action="citation_graph",
            doi="10.1000/test2",
            max_depth=2,
            max_nodes=5,  # 种子 + 4 个子节点就满 (ge=5 是 schema 下限)
        )
        # 给 6 个子节点, 但 max_nodes=5 应只收 4 个就停
        mock_response = {
            "data": [
                {"citedPaper": {"paperId": f"c{i}", "title": f"Child {i}", "year": 2020}}
                for i in range(1, 7)
            ]
        }
        with patch(
            "huginn.tools.literature.tool._http_get_json",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await tool.call(args, context=None)
        assert result.success
        data = result.data
        # 种子 + 4 个子节点 = 5 (触顶)
        assert data["n_unique_papers"] == 5
        assert data["truncated"] is True

    @pytest.mark.asyncio
    async def test_s2_api_failure_degrades_gracefully(self) -> None:
        """S2 API 全部失败时, 仍返回种子节点 + errors 列表."""
        tool = LiteratureTool()
        args = LiteratureInput(
            action="citation_graph",
            doi="10.1000/test3",
            max_depth=2,
        )

        async def _boom(url):
            raise RuntimeError("S2 timeout")

        with patch(
            "huginn.tools.literature.tool._http_get_json",
            new_callable=AsyncMock,
            side_effect=_boom,
        ):
            result = await tool.call(args, context=None)
        assert result.success  # 部分结果仍算成功
        data = result.data
        # 只有种子节点
        assert data["n_unique_papers"] == 1
        assert data["n_edges"] == 0
        # errors 记录了 API 失败
        assert len(data["errors"]) >= 1
        assert "S2 timeout" in data["errors"][0] or "timeout" in data["errors"][0].lower()
