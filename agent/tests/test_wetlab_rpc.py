"""Phase 5d 湿实验 RPC 工具测试.

4 测:
  1. 未配置 endpoint (env 未设 → success=False)
  2. submit_request mock (mock aiohttp POST)
  3. check_status poll (mock GET 先 pending 后 done)
  4. list_labs 返回 (mock GET /labs)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from huginn.tools.wetlab_rpc_tool import WetlabInput, WetlabRpcTool


class TestWetlabRpc:
    """wetlab_rpc_tool 4 个 action."""

    @pytest.mark.asyncio
    async def test_endpoint_not_configured(self, monkeypatch) -> None:
        monkeypatch.delenv("HUGINN_WETLAB_ENDPOINT", raising=False)
        tool = WetlabRpcTool()
        args = WetlabInput(action="list_labs")
        result = await tool.call(args, context=None)
        assert not result.success
        assert "not set" in (result.error or "") or "not configured" in (
            result.error or ""
        )

    @pytest.mark.asyncio
    async def test_submit_request_mock(self, monkeypatch) -> None:
        monkeypatch.setenv("HUGINN_WETLAB_ENDPOINT", "http://fake-lab.local")
        tool = WetlabRpcTool()
        args = WetlabInput(
            action="submit_request",
            lab_id="lab_1",
            request_type="synthesis",
            payload={"formula": "GaN", "temp": 800},
        )

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"request_id": "req_123", "ok": True})

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.post = MagicMock(return_value=mock_resp)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await tool.call(args, context=None)
        assert result.success, f"submit failed: {result.error}"
        assert result.data["request_id"] == "req_123"

    @pytest.mark.asyncio
    async def test_check_status_poll(self, monkeypatch) -> None:
        """先返回 pending, 第二次返回 done."""
        monkeypatch.setenv("HUGINN_WETLAB_ENDPOINT", "http://fake-lab.local")
        tool = WetlabRpcTool()
        args = WetlabInput(action="check_status", request_id="req_1", poll_timeout=5)

        call_count = [0]

        def make_resp():
            call_count[0] += 1
            resp = MagicMock()
            resp.status = 200
            if call_count[0] == 1:
                resp.json = AsyncMock(return_value={"status": "pending"})
            else:
                resp.json = AsyncMock(return_value={"status": "done", "data": "ok"})
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=None)
            return resp

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(side_effect=lambda *a, **kw: make_resp())

        with patch("aiohttp.ClientSession", return_value=mock_session), patch(
            "asyncio.sleep", new=AsyncMock()
        ):
            result = await tool.call(args, context=None)
        assert result.success, f"check_status failed: {result.error}"
        assert result.data["status"] == "done"

    @pytest.mark.asyncio
    async def test_list_labs_mock(self, monkeypatch) -> None:
        monkeypatch.setenv("HUGINN_WETLAB_ENDPOINT", "http://fake-lab.local")
        tool = WetlabRpcTool()
        args = WetlabInput(action="list_labs")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value=[{"lab_id": "lab_1", "name": "Synthesis Lab"}]
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await tool.call(args, context=None)
        assert result.success, f"list_labs failed: {result.error}"
        assert "labs" in result.data
        assert len(result.data["labs"]) == 1
