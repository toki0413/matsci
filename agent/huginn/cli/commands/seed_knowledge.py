"""Pre-load built-in knowledge-base seeds."""

from __future__ import annotations

import click
from rich.panel import Panel

from huginn.cli.context import CliContext


@click.command(name="seed-knowledge")
@click.option(
    "--force", is_flag=True, help="Re-ingest seed documents even if already present"
)
@click.pass_obj
def seed_knowledge(ctx: CliContext, force: bool) -> None:
    """Pre-load built-in materials science reference documents into the RAG KB."""
    from huginn.knowledge import get_knowledge_base, seed_knowledge_base

    kb = get_knowledge_base(str(ctx.workspace))
    result = seed_knowledge_base(kb, force=force)
    ctx.console.print(
        Panel(
            f"[bold blue]Seed Knowledge Base[/bold blue]\n"
            f"Added: {result['added']}\n"
            f"Skipped: {result['skipped']}\n"
            f"Failed: {result['failed']}",
            border_style="blue",
        )
    )
