"""通信契约层 —— 把 ASD-STE100 纪律从"语言"扩到"结构化通信"。

见 ``contract`` 模块: 对模块事件 / 工作流 / agent 间消息做确定性契约检查
(一词一义 / 信封完备 / 载荷契约 / 结果自洽), 只报告不拦截。
"""
from __future__ import annotations

from huginn.comms.contract import (
    FIELD_ALIASES,
    KNOWN_EVENT_TYPES,
    PAYLOAD_CONTRACTS,
    ContractReport,
    Violation,
    audit_agent_event,
    audit_agent_message,
    audit_plugin_event,
    audit_subagent_result,
    lint,
)

__all__ = [
    "ContractReport",
    "Violation",
    "KNOWN_EVENT_TYPES",
    "PAYLOAD_CONTRACTS",
    "FIELD_ALIASES",
    "lint",
    "audit_agent_event",
    "audit_agent_message",
    "audit_plugin_event",
    "audit_subagent_result",
]