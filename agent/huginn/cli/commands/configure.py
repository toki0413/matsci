"""Interactive configuration wizard."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.config import HuginnConfig


@click.command()
@click.option("--path", "-p", default="huginn.toml", help="Config file path to write")
@click.pass_obj
def configure(ctx: CliContext, path: str) -> None:
    """Interactive first-run configuration wizard."""
    ctx.console.print(
        Panel(
            "[bold blue]Huginn Configuration Wizard[/bold blue]",
            title="Setup",
            border_style="blue",
        )
    )

    # Try to load existing config
    try:
        cfg = HuginnConfig.load(path)
        ctx.console.print(f"[dim]Loaded existing config from {path}[/dim]")
    except Exception:
        cfg = HuginnConfig()

    provider = (
        ctx.console.input(f"Provider [cyan]{cfg.provider}[/cyan]: ").strip()
        or cfg.provider
    )
    model = (
        ctx.console.input(f"Model [cyan]{cfg.model or 'auto'}[/cyan]: ").strip()
        or cfg.model
    )
    api_key = (
        ctx.console.input(
            f"API key [cyan]{'***' if cfg.api_key else 'none'}[/cyan]: "
        ).strip()
        or cfg.api_key
    )
    base_url = (
        ctx.console.input(f"Base URL [cyan]{cfg.base_url or 'none'}[/cyan]: ").strip()
        or cfg.base_url
    )
    ollama_host = (
        ctx.console.input(f"Ollama host [cyan]{cfg.ollama_host}[/cyan]: ").strip()
        or cfg.ollama_host
    )
    workspace = (
        ctx.console.input(f"Workspace [cyan]{cfg.workspace}[/cyan]: ").strip()
        or cfg.workspace
    )

    thinking_raw = ctx.console.input(
        f"Thinking intensity [cyan]{cfg.thinking or 'none'}[/cyan] (low/medium/high or none): "
    ).strip() or (cfg.thinking if cfg.thinking else "")
    thinking: str | None = (
        thinking_raw if thinking_raw in ("low", "medium", "high") else None
    )

    new_cfg = HuginnConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        ollama_host=ollama_host,
        workspace=workspace,
        thinking=thinking,  # type: ignore[arg-type]
    )

    fmt = "toml" if path.endswith(".toml") else "json"
    new_cfg.save(path, format=fmt)
    ctx.console.print(f"[green]✓[/green] Config saved to [bold]{path}[/bold]")
    ctx.console.print("[dim]Run: huginn chat --config " + path + "[/dim]")
