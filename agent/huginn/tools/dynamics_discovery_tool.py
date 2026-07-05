"""shim: 实现在 huginn.tools.sci.dynamics_discovery_tool 子包.

按 sci/ 子包约定, 这里只做 re-export, 供 register_all_tools 按
huginn.tools.dynamics_discovery_tool.DynamicsDiscoveryTool 注册.
"""
from huginn.tools.sci.dynamics_discovery_tool import (
    DynamicsDiscoveryInput,
    DynamicsDiscoveryTool,
)

__all__ = ["DynamicsDiscoveryInput", "DynamicsDiscoveryTool"]
