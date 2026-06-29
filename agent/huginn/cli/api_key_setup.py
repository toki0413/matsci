"""交互式 API key 配置入口。

支持从 .env 文件读写, 检查各服务 key 是否已配置,
并提供一个交互式向导让用户逐个补全。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

# 需要配置 API key 的服务列表
# 新增服务时直接 append 一项即可, 不用改其他地方
API_KEY_SERVICES: list[dict[str, Any]] = [
    {
        "name": "tavily",
        "env_var": "TAVILY_API_KEY",
        "description": "Tavily Search API (用于 web_search_tool 的高级搜索)",
        "url": "https://tavily.com",
        "required": False,
    },
    {
        "name": "openai",
        "env_var": "OPENAI_API_KEY",
        "description": "OpenAI API (GPT 系列模型)",
        "url": "https://platform.openai.com/api-keys",
        "required": False,
    },
    {
        "name": "anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "description": "Anthropic API (Claude 系列模型)",
        "url": "https://console.anthropic.com/",
        "required": False,
    },
    {
        "name": "deepseek",
        "env_var": "DEEPSEEK_API_KEY",
        "description": "DeepSeek API (DeepSeek 系列模型)",
        "url": "https://platform.deepseek.com/",
        "required": False,
    },
    {
        "name": "materials_project",
        "env_var": "MP_API_KEY",
        "description": "Materials Project API (材料数据库查询)",
        "url": "https://nextgen.materialsproject.org/api",
        "required": False,
    },
]


def check_api_keys() -> dict[str, bool]:
    """检查所有 API key 的配置状态。

    只看环境变量里有没有值, 不验证 key 是否有效。
    返回 {service_name: is_configured}。
    """
    status: dict[str, bool] = {}
    for svc in API_KEY_SERVICES:
        val = os.environ.get(svc["env_var"], "").strip()
        status[svc["name"]] = bool(val)
    return status


def _read_env_file(env_file: str | Path) -> dict[str, str]:
    """读 .env 文件成 dict, 解析 KEY=VALUE 行。

    文件不存在或解析失败都返回空 dict, 不抛异常。
    """
    path = Path(env_file)
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            # 去掉两端引号, .env 里常见 KEY="value" 写法
            value = value.strip().strip('"').strip("'")
            result[key.strip()] = value
    except OSError:
        return {}
    return result


def save_api_key(env_var: str, key: str, env_file: str = ".env") -> None:
    """把 API key 写入 .env 文件。

    - 已存在该 key 就更新值
    - 不存在就追加一行
    - 自动加引号防止特殊字符 (空格 / # 等) 把行截断
    """
    path = Path(env_file)
    existing = _read_env_file(path)

    # 读原始行, 用于保留注释和顺序
    if path.exists():
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    else:
        raw_lines = []

    pattern = re.compile(rf"^\s*{re.escape(env_var)}\s*=", re.IGNORECASE)
    updated = False
    new_lines: list[str] = []
    for line in raw_lines:
        if pattern.match(line):
            new_lines.append(f'{env_var}="{key}"')
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        # 文件末尾留个空行, 避免追加时跟上一行挤在一起
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f'{env_var}="{key}"')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 同步到当前进程环境, 这样不用重启就能用上新的 key
    os.environ[env_var] = key


def _render_status_table(console: Console, status: dict[str, bool]) -> None:
    """打印 API key 配置状态表格。"""
    table = Table(title="API Key 配置状态", show_lines=False)
    table.add_column("#", style="dim", justify="right")
    table.add_column("服务", style="cyan")
    table.add_column("环境变量", style="blue")
    table.add_column("状态", justify="center")
    table.add_column("说明")

    for idx, svc in enumerate(API_KEY_SERVICES, start=1):
        configured = status.get(svc["name"], False)
        if configured:
            state = "[green]✓ 已配置[/green]"
        elif svc.get("required"):
            state = "[red]✗ 缺失 (必需)[/red]"
        else:
            state = "[yellow]○ 未配置[/yellow]"
        table.add_row(
            str(idx),
            svc["name"],
            svc["env_var"],
            state,
            svc["description"],
        )
    console.print(table)


def interactive_setup(console: Console | None = None) -> None:
    """交互式引导用户配置 API key。

    流程:
      1. 显示当前所有 key 的状态
      2. 让用户选择要配置的服务编号
      3. 输入 key (输入留空则跳过)
      4. 保存到 .env
      5. 循环直到用户退出
    """
    con = console or Console()
    _render_status_table(con, check_api_keys())

    con.print("\n[bold]操作:[/bold]")
    con.print("  输入 [cyan]编号[/cyan] (1-{n}) 配置对应服务".format(
        n=len(API_KEY_SERVICES)
    ))
    con.print("  输入 [cyan]a[/cyan] / [cyan]all[/cyan] 依次配置所有未填的服务")
    con.print("  输入 [cyan]q[/cyan] / [cyan]quit[/cyan] 退出\n")

    while True:
        try:
            choice = con.input("[bold green]选择:[/bold green] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            con.print("\n[yellow]已退出[/yellow]")
            return

        if choice in ("q", "quit", "exit"):
            con.print("[dim]再见[/dim]")
            return

        if choice in ("a", "all"):
            targets = [s for s in API_KEY_SERVICES if not os.environ.get(s["env_var"])]
            if not targets:
                con.print("[green]所有 key 都已配置, 没什么可填的[/green]")
                continue
            for svc in targets:
                _prompt_and_save(svc, con)
            _render_status_table(con, check_api_keys())
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(API_KEY_SERVICES):
                _prompt_and_save(API_KEY_SERVICES[idx - 1], con)
                _render_status_table(con, check_api_keys())
                continue

        con.print("[yellow]无效输入, 请输入编号 / a / q[/yellow]")


def _prompt_and_save(svc: dict[str, Any], console: Console) -> None:
    """对一个服务走一遍输入 + 保存流程。"""
    console.print(
        f"\n[bold blue]{svc['name']}[/bold blue] - {svc['description']}"
    )
    console.print(f"[dim]申请地址: {svc['url']}[/dim]")
    try:
        key = console.input("[bold]API key (留空跳过): [/bold]").strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]已跳过[/yellow]")
        return
    if not key:
        console.print("[dim]跳过[/dim]")
        return
    try:
        save_api_key(svc["env_var"], key)
        console.print(f"[green]✓ 已保存到 .env ({svc['env_var']})[/green]")
    except Exception as e:
        console.print(f"[red]保存失败: {e}[/red]")


@click.command(name="api-keys")
@click.option(
    "--env-file",
    default=".env",
    help="目标 .env 文件路径 (默认当前目录 .env)",
)
@click.option(
    "--check",
    is_flag=True,
    help="只检查配置状态, 不进入交互向导",
)
@click.pass_obj
def api_keys(ctx: Any, env_file: str, check: bool) -> None:
    """Configure API keys for external services."""
    # 拿 ctx 上的 console, 没有就用 rich 默认的
    console = getattr(ctx, "console", None) or Console()

    if check:
        _render_status_table(console, check_api_keys())
        return

    interactive_setup(console=console)


__all__ = [
    "API_KEY_SERVICES",
    "api_keys",
    "check_api_keys",
    "interactive_setup",
    "save_api_key",
]
