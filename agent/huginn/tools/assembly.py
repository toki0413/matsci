"""工具装配阶段 —— 单一入口统一过滤、排序、去重。

参考 Claude Code 的 tools.ts：deny 规则在装配阶段就被过滤掉，避免运行期
才决定工具可见性导致 cache 不稳定、提示词里的工具列表与实际可用工具不一致
等问题。
"""

from __future__ import annotations

from typing import Any

from huginn.permissions import PermissionMode
from huginn.tools.defaults import ToolMetadata


def _is_deny_mode(mode_value: Any) -> bool:
    """判断某个 mode 值是否表示 deny。

    兼容两种输入：
      - PermissionMode 枚举（permissions.py 内部使用）
      - 字符串 "deny"（外部配置文件常用）
    """
    if isinstance(mode_value, PermissionMode):
        return mode_value == PermissionMode.DENY
    if isinstance(mode_value, str):
        return mode_value.lower() == PermissionMode.DENY.value
    return False


def _get_tool_name(tool: dict) -> str:
    """从工具定义中提取名称, 兼容扁平和嵌套两种格式.

    - 扁平格式: {"name": "foo", ...}
    - 嵌套格式(LangChain/OpenAI 风格): {"type": "function", "function": {"name": "foo", ...}}
    """
    name = tool.get("name")
    if name:
        return str(name)
    func = tool.get("function")
    if isinstance(func, dict):
        return str(func.get("name", ""))
    return ""


def assemble_tool_pool(
    builtin_tools: list[dict],
    mcp_tools: list[dict] | None = None,
    permission_rules: dict | None = None,
) -> list[dict]:
    """单一装配点 —— 在装配阶段就把 deny 工具过滤掉。

    流程：
      1. 合并内置工具与 MCP 工具
      2. 根据 permission_rules 过滤掉 deny 工具
      3. 按名称排序，保证下游缓存键稳定
      4. 同名工具去重，保留第一个出现的（内置工具优先级高于 MCP）

    Args:
        builtin_tools: 内置工具定义列表，每个元素至少含 "name" 字段
            （扁平格式）或 "function"."name" 字段（嵌套格式）。
        mcp_tools: 通过 MCP 接入的扩展工具列表，可为 None。
        permission_rules: 权限规则字典，形如
            {"file_delete_tool": {"mode": "deny"}, ...}
            或 {"file_delete_tool": PermissionMode.DENY, ...}。
            未提供时不过滤。

    Returns:
        装配后的工具列表，已排序、去重。
    """
    denied = collect_denied_tool_names(permission_rules)

    all_tools: list[dict] = list(builtin_tools) + list(mcp_tools or [])
    allowed = [t for t in all_tools if _get_tool_name(t) not in denied]

    # 按名称排序，确保 LLM 提示词里的工具顺序稳定，便于结果缓存
    allowed.sort(key=_get_tool_name)

    # 去重：同名工具保留第一个出现的（内置工具已经排在前面）
    seen: set[str] = set()
    result: list[dict] = []
    for tool in allowed:
        name = _get_tool_name(tool)
        if name not in seen:
            seen.add(name)
            result.append(tool)
    return result


def collect_denied_tool_names(permission_rules: dict | None) -> set[str]:
    """从权限规则中收集所有 deny 工具的名称集合。

    抽出来便于在装配前后单独调用，也方便单测。
    """
    denied: set[str] = set()
    if not permission_rules:
        return denied

    for tool_name, rule in permission_rules.items():
        # 兼容两种 rule 形式：
        #   1) {"mode": "deny"} / {"mode": PermissionMode.DENY}
        #   2) 直接是 PermissionMode / 字符串（如 DEFAULT_PERMISSION_RULES 那种）
        if isinstance(rule, dict):
            mode_value = rule.get("mode")
        else:
            mode_value = rule
        if _is_deny_mode(mode_value):
            denied.add(tool_name)
    return denied


def filter_tools_by_deny_rules(
    tools: list[dict], denied: set[str]
) -> list[dict]:
    """过滤掉被 deny 的工具。

    用于装配后再次裁剪，例如运行期用户临时把某工具改成 deny。
    """
    return [t for t in tools if _get_tool_name(t) not in denied]


def annotate_metadata(tools: list[dict]) -> list[dict]:
    """给装配后的工具列表补上 ToolMetadata 默认值。

    未显式提供 metadata 的工具会拿到 fail-closed 默认值，便于下游权限检查
    统一处理。
    """
    enriched: list[dict] = []
    for tool in tools:
        if "metadata" not in tool or not isinstance(
            tool.get("metadata"), ToolMetadata
        ):
            tool = {**tool, "metadata": ToolMetadata()}
        enriched.append(tool)
    return enriched
