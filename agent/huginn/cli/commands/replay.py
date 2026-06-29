"""Trajectory replay 命令 — 回放 agent 的决策过程。

读 save_trajectory 写出的 JSON, 按步骤把每个工具调用的参数、结果、耗时
打出来, 配合进度条直观展示执行轨迹。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from huginn.cli.context import CliContext


def _format_progress(step: int, total: int, width: int = 10) -> str:
    """画 [████░░░░░░] 风格的进度条, width 是格子数。"""
    if total <= 0:
        return "[" + " " * width + "]"
    filled = int(width * step / total)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _truncate(s: str, max_len: int = 200) -> str:
    """长字符串截断加省略号, 一行能放下。"""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _dumps(obj: Any) -> str:
    """把任意对象转成可读字符串, dict/list 走 JSON。"""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _print_step(
    console: Console,
    step: int,
    total: int,
    call: dict[str, Any],
    verbose: bool,
) -> None:
    """打印一个工具调用步骤, 带框线 + 进度条。"""
    progress = _format_progress(step, total)
    pct = int(100 * step / total) if total else 0
    console.print(f"\nStep {step}/{total}  {progress}  {pct}%")

    tool = call.get("tool", "unknown")
    duration = call.get("duration_ms", 0.0) or 0.0
    success = call.get("success", True)
    mark = "[green]✓[/green]" if success else "[red]✗[/red]"
    console.print(f"[cyan]┌─[/cyan] {tool} ({duration:.1f}ms) {mark}")

    args = call.get("args")
    if args is not None:
        args_str = _dumps(args)
        if not verbose:
            args_str = _truncate(args_str)
        console.print(f"[cyan]│[/cyan]  args: {args_str}")

    result = call.get("result")
    if result is not None:
        result_str = _dumps(result)
        if not verbose:
            result_str = _truncate(result_str)
        console.print(f"[cyan]│[/cyan]  result: {result_str}")

    error = call.get("error")
    if error:
        console.print(f"[cyan]│[/cyan]  [red]error: {error}[/red]")

    console.print(f"[cyan]└─[/cyan]")


@click.command()
@click.argument("trajectory_file")
@click.option(
    "--step", "-s", "target_step", type=int, default=None,
    help="跳到指定步骤 (只详细显示该步)",
)
@click.option(
    "--verbose", "-v", is_flag=True,
    help="显示完整参数和结果 (默认截断到 200 字符)",
)
@click.pass_obj
def replay(
    ctx: CliContext,
    trajectory_file: str,
    target_step: int | None,
    verbose: bool,
) -> None:
    """回放 agent 的决策过程。

    \b
    显示格式:
      Step 1/8  [████░░░░░░] 25%
      ┌─ structure_tool (123.4ms) ✓
      │  args: {"formula": "Si", "a": 5.43}
      │  result: {"structure": "diamond", ...}
      └─

    TRAJECTORY_FILE 是 save_trajectory 写出的 JSON 路径。
    """
    console: Console = getattr(ctx, "console", None) or Console()

    path = Path(trajectory_file)
    if not path.exists():
        console.print(f"[red]轨迹文件不存在: {path}[/red]")
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        console.print(f"[red]JSON 解析失败: {e}[/red]")
        return

    tool_calls = data.get("tool_calls", []) or []
    if not tool_calls:
        console.print("[yellow]轨迹里没有工具调用记录[/yellow]")
        return

    total = len(tool_calls)

    # 文件头: 时间戳 + 路径 + 总步数, 让用户一眼看到在回放哪个轨迹
    ts = data.get("timestamp", "")
    console.print(f"[bold]轨迹回放[/bold]  {ts}")
    console.print(f"[dim]{path}[/dim]")
    console.print(f"[bold]共 {total} 步[/bold]")

    # --step N: 只详细打印第 N 步, 配合进度条看上下文位置
    if target_step is not None:
        if target_step < 1 or target_step > total:
            console.print(
                f"[red]步骤 {target_step} 超出范围 (1-{total})[/red]"
            )
            return
        _print_step(console, target_step, total, tool_calls[target_step - 1], verbose)
        return

    # 默认: 从第 1 步到末尾全量回放
    for i, call in enumerate(tool_calls, start=1):
        _print_step(console, i, total, call, verbose)

    # 末尾附 summary, 让用户看到总 span 数和分类统计
    summary = data.get("summary", {}) or {}
    total_spans = summary.get("total_spans", 0)
    console.print(f"\n[bold]Summary[/bold]  total_spans={total_spans}")
