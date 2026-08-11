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

    @pytest.mark.asyncio
    async def test_fetch_result_mock(self, monkeypatch) -> None:
        """fetch_result: 补齐 RPC 六 action 中最后一个未覆盖的网络动作."""
        monkeypatch.setenv("HUGINN_WETLAB_ENDPOINT", "http://fake-lab.local")
        tool = WetlabRpcTool()
        args = WetlabInput(action="fetch_result", request_id="req_9")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={"peaks": [{"two_theta": 34.5, "intensity": 100}]}
        )
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.get = MagicMock(return_value=mock_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await tool.call(args, context=None)
        assert result.success, f"fetch_result failed: {result.error}"
        assert result.data["result"]["peaks"][0]["two_theta"] == 34.5

    @pytest.mark.asyncio
    async def test_submit_protocol_local_validation(self) -> None:
        """submit_protocol: 本地协议校验 (dry-lab → wetlab 闭环的提交端)."""
        tool = WetlabRpcTool()
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            lab_id="lab_1",
            params={
                "scan_range": [10, 90],
                "step_size": 0.02,
                "dwell_time": 1.0,
            },
            sample={
                "sample_id": "s1",
                "composition": "GaN",
                "preparation_method": "MOCVD",
            },
        )
        result = await tool.call(args, context=None)
        assert result.success, f"submit_protocol failed: {result.error}"
        assert result.data["validated"] is True
        assert result.data["n_required_sample_fields"] == 3
        tmpl = result.data["request_template"]
        assert tmpl["protocol"] == "XRD"
        assert tmpl["lims_format"] is True
        # 默认值应被填入 (wavelength=1.5406)
        assert tmpl["params"]["wavelength"] == 1.5406

    @pytest.mark.asyncio
    async def test_submit_protocol_invalid_params(self) -> None:
        """submit_protocol: 超出范围的参数应被拦截."""
        tool = WetlabRpcTool()
        args = WetlabInput(
            action="submit_protocol",
            protocol="XRD",
            params={"scan_range": [10, 90], "step_size": 5.0, "dwell_time": 1.0},
            sample={"sample_id": "s1", "composition": "GaN", "preparation_method": "MOCVD"},
        )
        result = await tool.call(args, context=None)
        assert not result.success
        assert "step_size" in (result.error or "")

    @pytest.mark.asyncio
    async def test_parse_result_local(self) -> None:
        """parse_result: 本地结构化解析 (wetlab → dry-lab 闭环的结果端)."""
        tool = WetlabRpcTool()
        args = WetlabInput(
            action="parse_result",
            protocol="XRD",
            raw_result={
                "peaks": [{"two_theta": 34.5, "intensity": 100, "fwhm": 0.2}],
                "crystallite_size_nm": 42.0,
                "phase_ids": ["GaN"],
            },
        )
        result = await tool.call(args, context=None)
        assert result.success, f"parse_result failed: {result.error}"
        assert result.data["parsed"]["crystallite_size_nm"] == 42.0
        assert result.data["n_fields_extracted"] == 3

    @pytest.mark.asyncio
    async def test_wetlab_drylab_closed_loop(self) -> None:
        """闭环: submit_protocol 模板 → (mock wetlab) → parse_result 解析.

        验证 dry-lab 端生成的 LIMS 模板能经湿实验返回后, 被本地解析回结构化字段,
        证明 wetlab/dry-lab 协议 schema 一致、闭环可复现.
        """
        tool = WetlabRpcTool()
        # 1) dry-lab 端: 生成 XRD 请求模板
        sub = await tool.call(
            WetlabInput(
                action="submit_protocol",
                protocol="XRD",
                lab_id="lab_1",
                params={"scan_range": [10, 90], "step_size": 0.02, "dwell_time": 1.0},
                sample={"sample_id": "s1", "composition": "GaN", "preparation_method": "MOCVD"},
            ),
            context=None,
        )
        assert sub.success
        result_schema = sub.data["request_template"]["result_schema"]
        assert "peaks" in result_schema and "crystallite_size_nm" in result_schema

        # 2) 模拟湿实验返回 (字段与 result_schema 一致)
        raw = {
            "peaks": [{"two_theta": 34.5, "intensity": 100}],
            "crystallite_size_nm": 42.0,
            "phase_ids": ["GaN"],
        }

        # 3) dry-lab 端: 解析结果, 抽出 schema 中声明的字段
        par = await tool.call(
            WetlabInput(action="parse_result", protocol="XRD", raw_result=raw),
            context=None,
        )
        assert par.success
        parsed = par.data["parsed"]
        assert parsed["crystallite_size_nm"] == 42.0
        # 每个 schema 字段都应有对应解析键
        for field in result_schema:
            assert field in parsed["parsed_fields"]
