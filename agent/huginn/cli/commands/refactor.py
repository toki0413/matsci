"""Refactor command — plan and apply cross-file code changes."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from huginn.cli.context import CliContext
from huginn.coder.refactor_engine import RefactorEngine


@click.command(name="refactor")
@click.argument("task")
@click.option(
    "--file",
    "-f",
    multiple=True,
    help="Target file (relative to workspace); repeatable. Defaults to whole workspace.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show the planned diff without modifying files.",
)
@click.option(
    "--rollback",
    is_flag=True,
    help="Revert the most recent apply using saved snapshots.",
)
@click.option(
    "--snapshot-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save/load JSON snapshots for rollback.",
)
@click.pass_obj
@click.pass_context
def refactor(
    ctx: click.Context,
    obj: CliContext,
    task: str,
    file: tuple[str, ...],
    dry_run: bool,
    rollback: bool,
    snapshot_file: Path | None,
) -> None:
    """Plan and execute a cross-file refactoring task.

    Example:

        huginn refactor "rename class Foo to Bar" -f src/foo.py -f src/bar.py --dry-run
    """
    console = Console()

    if rollback:
        _do_rollback(obj, snapshot_file, console)
        return

    engine = RefactorEngine(root=obj.workspace, config=obj.load_config())
    target_files = list(file) if file else None

    with console.status("[bold blue]Planning refactor..."):
        try:
            plan = engine.plan(task, target_files=target_files)
        except Exception as exc:
            console.print(Panel(f"Planning failed: {exc}", border_style="red"))
            sys.exit(1)

    if not plan:
        console.print("[yellow]No edits proposed.[/yellow]")
        return

    console.print(f"[bold green]Proposed {len(plan)} edit(s)[/bold green]")
    for edit in plan:
        console.print(f"  - {edit.path}")

    result = engine.apply(plan, dry_run=dry_run or obj.dry_run)

    if result["diff"]:
        console.print(
            Panel(
                Syntax(result["diff"], "diff", theme="monokai", line_numbers=False),
                title="Diff",
                border_style="blue",
            )
        )

    if result["errors"]:
        for error in result["errors"]:
            console.print(f"[red]Error:[/red] {error}")

    if dry_run or obj.dry_run:
        console.print("[cyan]Dry run — no files modified.[/cyan]")
    else:
        console.print(f"[bold green]Applied {result['applied']} edit(s).[/bold green]")
        _save_snapshots(obj, snapshot_file, result["snapshots"], console)


def _save_snapshots(
    obj: CliContext,
    snapshot_file: Path | None,
    snapshots: dict[str, str],
    console: Console,
) -> None:
    import json

    path = snapshot_file or obj.workspace / ".huginn_refactor_snapshots.json"
    path.write_text(
        json.dumps(snapshots, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"[dim]Snapshots saved to {path}[/dim]")


def _do_rollback(
    obj: CliContext,
    snapshot_file: Path | None,
    console: Console,
) -> None:
    import json

    path = snapshot_file or obj.workspace / ".huginn_refactor_snapshots.json"
    if not path.exists():
        console.print(f"[red]No snapshot file found at {path}[/red]")
        sys.exit(1)

    snapshots = json.loads(path.read_text(encoding="utf-8"))
    engine = RefactorEngine(root=obj.workspace, config=obj.load_config())
    engine.rollback(snapshots)
    console.print(f"[bold green]Rolled back {len(snapshots)} file(s).[/bold green]")
