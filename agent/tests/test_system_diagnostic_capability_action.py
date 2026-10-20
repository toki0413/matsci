"""system_diagnostic_tool 新增 capability_runtime action 的行为."""

from __future__ import annotations

import asyncio

from huginn.tools.system_diagnostic_tool import (
    SystemDiagnosticInput,
    SystemDiagnosticTool,
)


def test_action_is_accepted_by_schema():
    SystemDiagnosticInput(action="capability_runtime")  # 不得抛 ValidationError


def test_action_returns_degraded_list():
    tool = SystemDiagnosticTool()
    result = asyncio.run(tool.call({"action": "capability_runtime"}))
    assert result.success is True
    assert "summary" in result.data
    assert isinstance(result.data["degraded"], list)
