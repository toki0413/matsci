"""自定义 slash 命令加载器。

从 `.huginn/commands/` 目录加载 markdown 文件, 文件名就是命令名,
内容是 prompt 模板, 可以用 $ARGUMENTS 占位符接收用户参数。

用法示例:
  在 workspace 下放 `.huginn/commands/test-material.md`:
    请对以下材料进行全面分析：
    $ARGUMENTS

    要求：
    1. 结构分析
    2. 性质预测
    3. 稳定性评估

  chat 里输入 `/test-material CaTiO3` 会被展开成上面的模板, $ARGUMENTS 替换为 CaTiO3。
"""

from __future__ import annotations

import re
from pathlib import Path

# 内置 slash 命令名, 自定义命令不能跟这些重名
BUILTIN_COMMANDS: frozenset[str] = frozenset(
    {
        "help",
        "compact",
        "clear",
        "context",
        "cost",
        "undo",
        "tools",
        "sessions",
        "bg",
    }
)


def _default_commands_dir(workspace: str | Path = ".") -> Path:
    """返回 workspace 下默认的自定义命令目录。"""
    return Path(workspace) / ".huginn" / "commands"


def load_custom_commands(commands_dir: Path | None = None) -> dict[str, str]:
    """从 `.huginn/commands/` 目录加载自定义命令模板。

    每个文件是一个 markdown 文件, 文件名（不含 .md）是命令名。
    文件内容是命令模板, 可以包含 $ARGUMENTS 占位符。

    返回 {command_name: template_content}。目录不存在或为空返回空 dict。
    """
    if commands_dir is None:
        commands_dir = _default_commands_dir()
    commands_dir = Path(commands_dir)

    if not commands_dir.exists() or not commands_dir.is_dir():
        return {}

    result: dict[str, str] = {}
    for path in commands_dir.iterdir():
        if not path.is_file():
            continue
        # 只认 .md 后缀, 其他文件忽略
        if path.suffix.lower() != ".md":
            continue
        name = path.stem.lower()
        # 跳过跟内置命令重名的, 避免覆盖
        if name in BUILTIN_COMMANDS:
            continue
        # 跳过名字带特殊字符的, 防止解析时出意外
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # 读不了的文件直接跳过, 不影响其他命令加载
            continue
        result[name] = content
    return result


def resolve_custom_command(
    user_input: str, workspace: str | Path = "."
) -> str | None:
    """检查用户输入是否是自定义命令（/custom-name args）。

    如果是, 返回展开 $ARGUMENTS 后的 prompt。
    如果不是自定义命令, 返回 None。

    不会处理内置命令 (help/compact 等), 那些由 slash_commands.py 自己处理。
    """
    if not user_input.startswith("/"):
        return None

    body = user_input.lstrip("/")
    # 按空格切出命令名和参数, 跟 slash_commands.py 保持一致
    parts = body.split(None, 1)
    if not parts:
        return None
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # 内置命令不在这里处理
    if name in BUILTIN_COMMANDS:
        return None

    commands = load_custom_commands(_default_commands_dir(workspace))
    template = commands.get(name)
    if template is None:
        return None

    # 把 $ARGUMENTS 替换为用户传入的参数, 没传就替换为空串
    # 同时支持 ${ARGUMENTS} 写法, 跟 shell 变量风格对齐
    expanded = template.replace("${ARGUMENTS}", args)
    expanded = expanded.replace("$ARGUMENTS", args)
    return expanded


def list_custom_commands(workspace: str | Path = ".") -> list[str]:
    """列出当前 workspace 下所有可用的自定义命令名。

    给 /help 之类的地方用, 让用户能看到自己定义了哪些命令。
    """
    commands = load_custom_commands(_default_commands_dir(workspace))
    return sorted(commands.keys())


__all__ = [
    "BUILTIN_COMMANDS",
    "list_custom_commands",
    "load_custom_commands",
    "resolve_custom_command",
]
