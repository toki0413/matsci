"""命令懒加载 —— 延迟导入，避免启动时把所有命令模块都拉起来。

思路参考 Claude Code 的 commands.ts：命令模块只在真正被调用、或查看
其帮助时才导入，从而加快 CLI 启动速度、减少不必要的副作用初始化。

与 commands/__init__.py 里的 register_commands 互斥使用：
- register_commands 会 eagerly 导入全部命令模块（启动慢但兼容性好）
- register_lazy_commands 不在启动时导入任何命令模块（启动快）
两者不要同时调用，否则 eager 导入会抹平懒加载的收益。
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import click

# 命令名 -> 模块路径 的映射，import 时由 scan_command_modules 自动填充
_LAZY_COMMANDS: dict[str, str] = {}

# 模块名 -> 命令属性名 的显式覆盖
# 多数模块用模块名做属性（chat.chat、serve.serve），少数例外在这里登记
_COMMAND_ATTR_OVERRIDES: dict[str, str] = {
    "export": "export_data",
}


CommandType = Literal["prompt", "local", "ui"]


@dataclass
class Command:
    """命令的元信息描述，便于在 UI/帮助层统一过滤与展示。"""

    name: str
    description: str
    type: CommandType = "local"
    # builtin | plugin | skill | mcp
    source: str = "builtin"
    is_enabled: Callable[[], bool] = field(default=lambda: True)
    is_hidden: bool = False
    # availability 条件列表，OR 关系；为空表示总是可用
    availability: list[str] | None = None


def scan_command_modules() -> dict[str, str]:
    """扫描 cli/commands/ 目录，生成 命令名 -> 模块路径 的映射。

    跳过以 _ 开头的文件和 __init__.py。只做文件系统扫描，不导入模块。
    """
    commands_dir = Path(__file__).parent / "commands"
    mapping: dict[str, str] = {}
    if not commands_dir.is_dir():
        return mapping
    for f in commands_dir.glob("*.py"):
        if f.name.startswith("_"):
            continue
        name = f.stem
        mapping[name] = f"huginn.cli.commands.{name}"
    return mapping


def _resolve_command_attr(module: Any, name: str) -> click.Command | None:
    """从命令模块里挑出对应的 Click 命令对象。

    查找顺序：
    1. _COMMAND_ATTR_OVERRIDES 里的显式覆盖
    2. 与模块同名的属性
    3. 模块内名为 command 的属性
    4. 模块里出现的第一个 click 命令（兜底）

    用 click.Command 而非 BaseCommand：Group 是 Command 的子类，
    所以 Command 既能匹配普通命令也能匹配命令组，且避开 Click 9.0 的弃用告警。
    """
    attr = _COMMAND_ATTR_OVERRIDES.get(name)
    if attr:
        cmd = getattr(module, attr, None)
        if isinstance(cmd, click.Command):
            return cmd

    cmd = getattr(module, name, None)
    if isinstance(cmd, click.Command):
        return cmd

    cmd = getattr(module, "command", None)
    if isinstance(cmd, click.Command):
        return cmd

    for value in vars(module).values():
        if isinstance(value, click.Command):
            return value
    return None


def get_lazy_command(name: str) -> click.Command | None:
    """按需加载命令模块并返回 Click 命令对象，找不到返回 None。"""
    mapping = _LAZY_COMMANDS or scan_command_modules()
    module_path = mapping.get(name)
    if not module_path:
        return None
    module = importlib.import_module(module_path)
    return _resolve_command_attr(module, name)


def register_lazy_commands(cli_group: click.Group) -> None:
    """把所有命令以懒加载方式挂到 Click group 上。

    实现方式：在 group 实例上打补丁，改写 get_command / list_commands，
    使得命令名出现在 --help 里、但对应模块只在真正取用时才导入。
    已经显式注册过的命令优先走原逻辑，不会被懒加载覆盖。
    """
    lazy_map: dict[str, str] = dict(scan_command_modules())
    # 已显式注册的命令不走懒加载
    for existing in list(cli_group.commands.keys()):
        lazy_map.pop(existing, None)

    # 把映射挂在实例上，方便外部排查
    cli_group._huginn_lazy_commands = lazy_map  # type: ignore[attr-defined]

    original_get_command = cli_group.get_command
    original_list_commands = cli_group.list_commands

    def patched_get_command(
        ctx: click.Context, cmd_name: str
    ) -> click.Command | None:
        # 先走原有逻辑，保留已显式注册的命令
        cmd = original_get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        if cmd_name in lazy_map:
            return get_lazy_command(cmd_name)
        return None

    def patched_list_commands(ctx: click.Context) -> list[str]:
        base = original_list_commands(ctx)
        merged = set(base) | set(lazy_map.keys())
        return sorted(merged)

    cli_group.get_command = patched_get_command  # type: ignore[assignment]
    cli_group.list_commands = patched_list_commands  # type: ignore[assignment]


def preload_command(name: str) -> None:
    """显式预加载某个命令模块。

    适用于既想保留懒加载、又确定马上要用到某条命令的场景。
    """
    get_lazy_command(name)


def init_lazy_registry() -> None:
    """填充 _LAZY_COMMANDS 注册表，避免每次调用都重新扫目录。"""
    global _LAZY_COMMANDS
    if not _LAZY_COMMANDS:
        _LAZY_COMMANDS = scan_command_modules()


# 导入时填充一次，方便直接读取 _LAZY_COMMANDS
init_lazy_registry()
