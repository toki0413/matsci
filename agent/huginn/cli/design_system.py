"""统一设计系统 — 基于 Rich 的组件库，保证视觉一致性。"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


class DesignSystem:
    """统一视觉语言"""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    # --- 基础组件 ---

    def dialog(self, title: str, content: str, color: str = "blue") -> None:
        """标准对话框"""
        self.console.print(Panel(content, title=title, border_style=color))

    def info(self, message: str) -> None:
        """信息提示"""
        self.console.print(f"[blue]ℹ[/blue] {message}")

    def success(self, message: str) -> None:
        """成功提示"""
        self.console.print(f"[green]✓[/green] {message}")

    def warning(self, message: str) -> None:
        """警告提示"""
        self.console.print(f"[yellow]⚠[/yellow] {message}")

    def error(self, message: str) -> None:
        """错误提示"""
        self.console.print(f"[red]✗[/red] {message}")

    # --- 权限请求 ---

    def permission_request(
        self,
        tool: str,
        reason: str,
        diff: str | None = None,
    ) -> bool:
        """权限请求对话框，带可选 diff 预览"""
        self.console.print(Panel(
            f"[yellow]需要审批:[/yellow] {tool}\n原因: {reason}",
            title="权限请求",
            border_style="yellow",
        ))
        if diff:
            self.render_diff(diff)
        response = self.console.input("[bold]是否允许? [y/N]: [/bold]")
        return response.strip().lower() in ("y", "yes")

    # --- 结构化 Diff ---

    def render_diff(self, diff_text: str) -> None:
        """颜色化的 diff 渲染"""
        for line in diff_text.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                self.console.print(f"[green]{line}[/green]")
            elif line.startswith("-") and not line.startswith("---"):
                self.console.print(f"[red]{line}[/red]")
            elif line.startswith("@@"):
                self.console.print(f"[cyan]{line}[/cyan]")
            else:
                self.console.print(line)

    def render_code(self, code: str, language: str = "python") -> None:
        """语法高亮的代码渲染"""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)

    # --- 进度条 ---

    def progress_bar(self, current: int, total: int, label: str = "") -> None:
        """标准进度条"""
        pct = current / total * 100 if total else 0
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        self.console.print(f"[blue]{label}[/blue] [{bar}] {pct:.1f}% ({current}/{total})")

    # --- 表格 ---

    def table(self, title: str, headers: list[str], rows: list[list[str]]) -> None:
        """标准表格"""
        table = Table(title=title)
        for h in headers:
            table.add_column(h)
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    # --- 上下文可视化 ---

    def render_context_usage(self, usage: dict, window: int) -> None:
        """token 使用情况可视化"""
        total = sum(usage.values())
        pct = total / window * 100 if window else 0
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        self.console.print(f"\n[blue]上下文:[/blue] [{bar}] {pct:.1f}% ({total:,}/{window:,})")

        for category, tokens in usage.items():
            cat_pct = tokens / total * 100 if total else 0
            self.console.print(f"  [dim]{category:20s}:[/dim] {tokens:>8,} ({cat_pct:.1f}%)")

    # --- 消息类型化渲染 ---

    def render_message(self, msg_type: str, content: str, **kwargs) -> None:
        """类型化消息渲染"""
        renderers = {
            "user": lambda: f"[bold cyan]You:[/bold cyan] {content}",
            "assistant": lambda: f"[bold magenta]Huginn:[/bold magenta] {content}",
            "tool_use": lambda: f"[dim]→ 调用 {kwargs.get('tool', 'unknown')}...[/dim]",
            "tool_result": lambda: f"[green]✓ {kwargs.get('tool', 'unknown')}[/green] ({kwargs.get('duration_ms', 0)}ms)",
            "tool_error": lambda: f"[red]✗ {kwargs.get('tool', 'unknown')}: {content}[/red]",
            "rate_limit": lambda: f"[yellow]⚠ 限流，{kwargs.get('retry_after', 60)}秒后重试...[/yellow]",
            "system": lambda: f"[dim blue][系统] {content}[/dim blue]",
        }
        renderer = renderers.get(msg_type)
        if renderer:
            self.console.print(renderer())
        else:
            self.console.print(content)


# 全局单例
_ds: DesignSystem | None = None


def get_design_system() -> DesignSystem:
    """获取全局 DesignSystem 单例"""
    global _ds
    if _ds is None:
        _ds = DesignSystem()
    return _ds
