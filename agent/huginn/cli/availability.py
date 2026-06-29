"""命令 availability 过滤 —— 根据环境状态决定哪些命令可见。

参考 Claude Code 的 commands.ts：每条命令可以声明 availability 条件，
只有满足条件的命令才会展示给用户。这样能在没有 HPC、没认证的环境下
隐藏掉一堆用不了的命令，避免误触。
"""
from __future__ import annotations

import os
from typing import Any

# 命令名 -> availability 条件列表
# 条件之间是 OR 关系：满足任意一个即视为可见。
# 不列在这里的命令默认总是可用。
COMMAND_AVAILABILITY: dict[str, list[str]] = {
    # 需要 HPC 连接
    "hpc": ["hpc"],
    "remote": ["hpc"],
    "scheduler": ["hpc"],
    # 需要某种可用认证
    "chat": ["api_key", "oauth"],
    "autoresearch": ["api_key", "oauth"],
    "evolve": ["api_key", "oauth"],
    "swarm": ["api_key", "oauth"],
    "team": ["api_key", "oauth"],
    "workflow": ["api_key", "oauth"],
    "coder": ["api_key", "oauth"],
    "refactor": ["api_key", "oauth"],
    "explore": ["api_key", "oauth"],
    "autoloop": ["api_key", "oauth"],
    # 仅开发模式可见
    "telemetry": ["dev_mode"],
    "diagnose": ["dev_mode"],
    # 其余命令（serve、kg、version、configure、bench、export 等）默认可见
}


def get_auth_state() -> dict[str, bool]:
    """检测当前环境的认证 / 能力状态。

    返回字典的键即 availability 条件名。检测依据是环境变量；
    测试时可通过设置环境变量或直接传 auth_state 覆盖。
    """
    def _flag(name: str) -> bool:
        return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")

    return {
        "api_key": bool(
            os.environ.get("HUGINN_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        ),
        "oauth": bool(os.environ.get("HUGINN_OAUTH_TOKEN")),
        "hpc": bool(os.environ.get("HUGINN_HPC_HOST")),
        "rag": _flag("HUGINN_RAG_ENABLED"),
        "dev_mode": _flag("HUGINN_DEV_MODE"),
    }


def meets_availability(
    availability: list[str] | None,
    auth_state: dict[str, bool] | None = None,
) -> bool:
    """检查命令是否满足 availability 要求。

    availability 是条件列表，OR 关系：满足任意一个即视为可用。
    为 None 或空列表表示总是可用。
    """
    if not availability:
        return True
    if auth_state is None:
        auth_state = get_auth_state()
    for condition in availability:
        if auth_state.get(condition, False):
            return True
    return False


def filter_commands_by_availability(
    commands: Any,
    auth_state: dict[str, bool] | None = None,
) -> Any:
    """过滤命令, 只保留满足 availability 的.

    支持两种调用方式:
    1. 传 list[dict]: 每个元素是字典, 返回过滤后的 list[dict]
    2. 传 click.Group: 直接在 group 上操作, 把不可用的命令删掉, 返回 group

    第二种是 CLI 启动时的主路径, 避免在外面再写一遍遍历逻辑.
    """
    if auth_state is None:
        auth_state = get_auth_state()

    # click.Group 路径: 直接改 group.commands
    if hasattr(commands, "commands") and hasattr(commands, "list_commands"):
        to_remove = []
        for name in commands.list_commands(None):
            avail = COMMAND_AVAILABILITY.get(name)
            if avail is not None and not meets_availability(avail, auth_state):
                to_remove.append(name)
        for name in to_remove:
            commands.commands.pop(name, None)
        return commands

    # list[dict] 路径: 纯函数, 返回过滤后的列表
    result: list[dict] = []
    for cmd in commands:
        avail = cmd.get("availability")
        if avail is None:
            avail = COMMAND_AVAILABILITY.get(cmd.get("name", ""))
        if meets_availability(avail, auth_state):
            result.append(cmd)
    return result


def command_is_available(
    name: str, auth_state: dict[str, bool] | None = None
) -> bool:
    """单条命令是否可见（按 COMMAND_AVAILABILITY 查询）。"""
    return meets_availability(COMMAND_AVAILABILITY.get(name), auth_state)
