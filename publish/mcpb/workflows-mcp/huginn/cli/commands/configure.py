"""Interactive configuration wizard."""

from __future__ import annotations

import click

from huginn.cli.context import CliContext
from huginn.cli.design_system import get_design_system
from huginn.config import HuginnConfig


@click.command()
@click.option("--path", "-p", default="huginn.toml", help="Config file path to write")
@click.pass_obj
def configure(ctx: CliContext, path: str) -> None:
    """Interactive first-run configuration wizard."""
    ds = get_design_system()
    ds.dialog(
        title="Setup",
        content="[bold blue]Huginn Configuration Wizard[/bold blue]",
    )

    # Try to load existing config
    try:
        cfg = HuginnConfig.load(path)
        ds.info(f"Loaded existing config from {path}")
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
    ds.success(f"Config saved to {path}")
    ds.info("Run: huginn chat --config " + path)
