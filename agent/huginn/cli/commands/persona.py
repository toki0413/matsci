"""Persona management commands."""

from __future__ import annotations

from pathlib import Path

import click
from rich.panel import Panel
from rich.table import Table

from huginn.cli.context import CliContext


@click.group(name="persona")
@click.pass_obj
def persona(ctx: CliContext) -> None:
    """Manage Huginn personas."""


@persona.command("list")
@click.pass_obj
def persona_list(ctx: CliContext) -> None:
    """List available personas."""
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    ctx.console.print(Panel("[bold blue]Personas[/bold blue]", border_style="blue"))
    for name in mgr.list():
        p = mgr.get(name)
        marker = " (default)" if name == mgr.get_default_name() else ""
        kind_tag = f" [{p.kind}]" if p.kind != "json" else ""
        ctx.console.print(f"  - {name}{kind_tag}{marker}")
        if p.description:
            ctx.console.print(f"      {p.description}")


@persona.command("show")
@click.argument("name")
@click.pass_obj
def persona_show(ctx: CliContext, name: str) -> None:
    """Show a persona details."""
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    p = mgr.get(name)
    ctx.console.print(
        Panel(
            f"[bold blue]{p.name}[/bold blue] [{p.kind}]",
            title="Persona",
            border_style="blue",
        )
    )
    if p.description:
        ctx.console.print(f"[italic]{p.description}[/italic]\n")
    ctx.console.print(p.system_prompt)
    if p.when_to_use:
        ctx.console.print("\n[bold]Use when:[/bold]")
        for trigger in p.when_to_use:
            ctx.console.print(f"  - {trigger}")
    if p.begin_dialogs:
        ctx.console.print("\n[bold]Begin dialogs:[/bold]")
        for d in p.begin_dialogs:
            ctx.console.print(f"  {d['role']}: {d['content']}")


@persona.command("set-default")
@click.argument("name")
@click.pass_obj
def persona_set_default(ctx: CliContext, name: str) -> None:
    """Set the default persona."""
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    try:
        mgr.set_default(name)
        ctx.console.print(f"[green]Default persona set to {name}[/green]")
    except ValueError as e:
        ctx.console.print(f"[red]{e}[/red]")


# Alias that matches Nuwa-style switching vocabulary.
@persona.command("use")
@click.argument("name")
@click.pass_obj
def persona_use(ctx: CliContext, name: str) -> None:
    """Alias for set-default: activate a persona."""
    persona_set_default.callback(ctx, name)  # type: ignore[arg-type]


@persona.command("match")
@click.argument("query")
@click.option("--threshold", default=0.3, help="Minimum match score")
@click.pass_obj
def persona_match(ctx: CliContext, query: str, threshold: float) -> None:
    """Find the persona that best matches a query."""
    from huginn.persona_matcher import PersonaMatcher
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    matcher = PersonaMatcher(manager=mgr)
    results = matcher.match(query, top_k=3, score_threshold=threshold)
    if not results:
        ctx.console.print("[yellow]No strong persona match found.[/yellow]")
        return
    table = Table(title="Matching Personas")
    table.add_column("Rank")
    table.add_column("Name")
    table.add_column("Score")
    table.add_column("Description")
    for i, (p, score) in enumerate(results, 1):
        table.add_row(
            str(i), p.name, f"{score:.3f}", (p.description or "")[:60]
        )
    ctx.console.print(table)


@persona.command("switch")
@click.argument("name")
@click.pass_obj
def persona_switch(ctx: CliContext, name: str) -> None:
    """Switch the active runtime persona for the current workspace session."""
    from huginn.persona_emotion import EmotionTracker
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    p = mgr.get(name)

    # Update config default so subsequent server/chat starts use this persona.
    import os

    os.environ["HUGINN_PERSONA"] = name

    ctx.console.print(
        f"[green]Switched active persona to {p.name}[/green]"
    )
    ctx.console.print(
        f"[dim]Description:[/dim] {p.description or '(none)'}"
    )
    tracker = EmotionTracker(name, workspace=ctx.workspace)
    ctx.console.print(f"[dim]Mood:[/dim] {tracker.context_prompt()}")


@persona.command("import")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.pass_obj
def persona_import(ctx: CliContext, path: Path) -> None:
    """Import a Nuwa-style SKILL.md persona."""
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    try:
        persona = mgr.import_skill(path)
        ctx.console.print(f"[green]Imported persona {persona.name} from {path}[/green]")
    except Exception as e:
        ctx.console.print(f"[red]Failed to import persona: {e}[/red]")


@persona.command("create")
@click.argument("name")
@click.option("--prompt", required=True, help="System prompt text")
@click.option(
    "--begin-dialog",
    "begin_dialogs",
    multiple=True,
    help="Begin dialog as role:content",
)
@click.pass_obj
def persona_create(
    ctx: CliContext,
    name: str,
    prompt: str,
    begin_dialogs: tuple[str, ...],
) -> None:
    """Create a new persona."""
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    parsed = []
    for d in begin_dialogs:
        if ":" not in d:
            ctx.console.print(
                f"[red]Invalid begin dialog (use role:content): {d}[/red]"
            )
            return
        role, content = d.split(":", 1)
        parsed.append({"role": role.strip(), "content": content.strip()})
    try:
        mgr.create(name, system_prompt=prompt, begin_dialogs=parsed)
        ctx.console.print(f"[green]Created persona {name}[/green]")
    except ValueError as e:
        ctx.console.print(f"[red]{e}[/red]")


@persona.command("delete")
@click.argument("name")
@click.pass_obj
def persona_delete(ctx: CliContext, name: str) -> None:
    """Delete a user-defined persona."""
    from huginn.personas import PersonaManager

    mgr = PersonaManager(workspace=ctx.workspace)
    try:
        mgr.delete(name)
        ctx.console.print(f"[green]Deleted persona {name}[/green]")
    except ValueError as e:
        ctx.console.print(f"[red]{e}[/red]")


@persona.command("emotion")
@click.argument("name")
@click.pass_obj
def persona_emotion(ctx: CliContext, name: str) -> None:
    """Show the computational emotional trajectory of a persona."""
    from huginn.persona_emotion import EmotionTracker

    tracker = EmotionTracker(name, workspace=ctx.workspace)
    state = tracker.current_state()
    ctx.console.print(
        Panel(
            f"[bold blue]{name}[/bold blue] current mood\n"
            f"[italic]{tracker.context_prompt()}[/italic]",
            title="Emotional Trajectory",
            border_style="blue",
        )
    )
    table = Table(title="Dimensions")
    table.add_column("Dimension")
    table.add_column("Value")
    for dim in (
        "valence",
        "arousal",
        "trust",
        "affection",
        "fatigue",
        "loneliness",
        "interest",
    ):
        table.add_row(dim, f"{getattr(state, dim):.3f}")
    ctx.console.print(table)
    ctx.console.print(f"\n[dim]Last updated: {state.timestamp}[/dim]")
    ctx.console.print(f"[dim]Recent events: {len(state.events)}[/dim]")
