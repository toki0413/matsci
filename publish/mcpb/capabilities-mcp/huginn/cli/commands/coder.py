"""Autonomous coder command."""

from __future__ import annotations

import click
from rich.markdown import Markdown
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.permissions import PermissionConfig


@click.command()
@click.argument("task", required=False)
@click.option(
    "--auto-approve", is_flag=True, help="Auto-approve destructive coder actions"
)
@click.option(
    "--max-iterations", "-i", type=int, default=None, help="Maximum coder iterations"
)
@click.pass_obj
def coder(
    ctx: CliContext, task: str | None, auto_approve: bool, max_iterations: int | None
) -> None:
    """Start an autonomous coding session (Codex-like)."""
    from huginn.coder import CoderRunner
    from huginn.config import get_settings

    settings = get_settings()
    if ctx.config_path:
        from huginn.config import HuginnConfig

        try:
            file_cfg = HuginnConfig.load(ctx.config_path)
            settings.config = file_cfg
            ctx.console.print(f"[green]✓[/green] Loaded config from {ctx.config_path}")
        except Exception as e:
            ctx.console.print(f"[yellow]Config load warning: {e}[/yellow]")

    # Apply CLI overrides to the loaded config
    settings.config.apply_overrides(
        provider=ctx.provider,
        model=ctx.model,
        base_url=ctx.base_url,
        ollama_url=ctx.ollama_url,
        thinking=ctx.thinking,
    )
    if max_iterations is not None:
        settings.coder.max_iterations = max_iterations

    permission_config = PermissionConfig(auto_approve_all=auto_approve)
    if auto_approve:
        ctx.console.print(
            "[yellow]Auto-approve enabled for this coder session[/yellow]"
        )

    def approval_callback(tool_name: str, reason: str) -> bool:
        ctx.console.print(f"[yellow]{reason}[/yellow]")
        answer = (
            ctx.console.input(f"Approve [bold]{tool_name}[/bold]? [y/N]: ")
            .strip()
            .lower()
        )
        return answer in ("y", "yes")

    if not task:
        try:
            task = ctx.console.input("[bold green]Coder task:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            ctx.console.print("\n[yellow]Cancelled.[/yellow]")
            return

    if not task:
        ctx.console.print("[yellow]No task provided.[/yellow]")
        return

    ctx.console.print(
        Panel(
            f"[bold blue]Coder Mode[/bold blue]\n"
            f"Provider: {settings.config.provider}\n"
            f"Model: {settings.config.model or 'auto'}\n"
            f"Max iterations: {settings.coder.max_iterations}",
            title="Autonomous Coding",
            border_style="blue",
        )
    )

    try:
        runner = CoderRunner(
            settings=settings,
            permission_config=permission_config,
            approval_callback=None if auto_approve else approval_callback,
        )
        result = runner.run(task)
        final = result.get("final_answer", "")
        ctx.console.print(Markdown(final))
    except Exception as e:
        ctx.console.print(f"[red]Coder failed: {e}[/red]")
