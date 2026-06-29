"""插件权限模型 —— 反 AstrBot "运行时无沙箱" 反模式。

AstrBot 插件能直接摸 LLM / 文件 / 子进程, 没有声明也没有强制,
导致恶意/有 bug 的插件能搞坏宿主。这里强制:
  - 插件在 metadata.yaml 的 permissions 字段声明需要的能力
  - PermissionChecker 在 handler 执行前检查, 缺权限直接抛 PermissionError
  - 默认给基础权限 (LLM_CALL / 读工具列表), 其他都要显式声明

注意: 跟 huginn/permissions.py (工具 AUTO/ASK/DENY) 是不同维度,
那个管工具执行确认, 这个管插件能力边界, 互不干扰。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from huginn.plugins.metadata import PluginMetadata, _perm_match


class PluginPermission(Enum):
    """插件能申请的能力。值即 metadata.yaml 里写的字符串。"""

    LLM_CALL = "llm_call"               # 调用 LLM
    TOOL_CALL = "tool_call"             # 调用工具 (可带 scope: tool_call:vasp_tool)
    FILE_WRITE = "file_write"           # 写文件 (可带路径 scope)
    NETWORK = "network"                 # 网络访问
    SUBPROCESS = "subprocess"           # 子进程执行


# 默认所有插件都有的基础权限, 不需要在 metadata.yaml 声明
DEFAULT_PERMISSIONS: list[str] = [
    "llm_call",         # 读写 LLM 是基础能力
    "tool_call:list",   # 读工具列表 (只读, 无副作用)
]

# 权限字符串里冒号前的 kind 必须落在这个集合, 防止插件瞎写
_VALID_KINDS = {p.value for p in PluginPermission}


@dataclass
class PermissionChecker:
    """权限检查器。强制模型: 不通过就抛, 不静默返回 False。

    用法:
        checker = PermissionChecker()
        checker.register(metadata)  # 加载插件时注册
        checker.enforce(plugin_name, "tool_call:vasp_tool")  # 不通过抛错
    """

    # plugin_name -> 声明的权限列表 (已合并默认权限)
    _granted: dict[str, list[str]] = field(default_factory=dict)
    # 调试 / 审计用, 记录最近一次拒绝原因
    last_denied_reason: str = ""

    def register(self, metadata: PluginMetadata) -> None:
        """注册一个插件的权限声明。合并默认权限。"""
        perms = list(DEFAULT_PERMISSIONS)
        for p in metadata.permissions:
            kind = p.split(":", 1)[0] if ":" in p else p
            if kind not in _VALID_KINDS:
                # 未知权限 kind 不致命, 但忽略并记录 (开发期容易拼错)
                continue
            if p not in perms:
                perms.append(p)
        self._granted[metadata.name] = perms

    def unregister(self, plugin_name: str) -> None:
        self._granted.pop(plugin_name, None)

    def check(self, plugin_name: str, perm: str) -> bool:
        """检查插件是否有 perm 权限。perm 形如 'tool_call:vasp_tool'。"""
        granted = self._granted.get(plugin_name, list(DEFAULT_PERMISSIONS))
        for declared in granted:
            if _perm_match(declared, perm):
                return True
        self.last_denied_reason = (
            f"plugin {plugin_name!r} lacks permission {perm!r}; "
            f"declared: {granted}"
        )
        return False

    def enforce(self, plugin_name: str, perm: str) -> None:
        """强制检查。不通过抛 PermissionError, 调用方必须接住或让它冒泡。"""
        if not self.check(plugin_name, perm):
            raise PermissionError(self.last_denied_reason)

    def check_handler(self, plugin_name: str, metadata: Any) -> bool:
        """检查一个 StarHandlerMetadata 声明的所有权限是否都满足。

        返回 True 表示全部通过, False 表示有缺 (调用方决定是跳过还是抛)。
        """
        perms = getattr(metadata, "permissions", []) or []
        for perm in perms:
            if not self.check(plugin_name, perm):
                return False
        return True

    def list_granted(self, plugin_name: str) -> list[str]:
        """列出某插件已授予的权限。调试用。"""
        return list(self._granted.get(plugin_name, DEFAULT_PERMISSIONS))


__all__ = ["PluginPermission", "PermissionChecker", "DEFAULT_PERMISSIONS"]
