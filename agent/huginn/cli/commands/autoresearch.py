"""AutoResearch CLI — drive an autoresearch workspace from Huginn."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.plugins.autoresearch import AutoresearchInput, AutoresearchTool
from huginn.types import ToolContext


@click.group(name="autoresearch")
def autoresearch() -> None:
    """Manage an AutoResearch workspace (init, prepare, run, step, loop)."""
    pass


def _tool_context(obj: CliContext) -> ToolContext:
    return ToolContext(
        session_id=f"cli-{id(obj)}",
        workspace=str(obj.workspace),
        config=obj.load_config(),
    )


async def _run_tool(action: str, ctx: ToolContext, **kwargs: Any) -> Any:
    tool = AutoresearchTool()
    inp = AutoresearchInput(action=action, **kwargs)
    return await tool.call(inp, ctx)


@autoresearch.command(name="init")
@click.option(
    "--workspace",
    "-w",
    default="./autoresearch",
    help="Directory to initialize as an autoresearch workspace",
)
@click.option(
    "--repo-url",
    "-r",
    default=None,
    help="Git URL to clone (default: karpathy/autoresearch)",
)
@click.option(
    "--branch",
    "-b",
    default=None,
    help="Experiment branch name (default: autoresearch/huginn-YYYYMMDD)",
)
@click.option(
    "--program-append",
    "-p",
    default=None,
    help="Extra instructions to append to program.md",
)
@click.pass_obj
def autoresearch_init(
    obj: CliContext,
    workspace: str,
    repo_url: str | None,
    branch: str | None,
    program_append: str | None,
) -> None:
    """Initialize a fresh autoresearch workspace."""
    ctx = _tool_context(obj)
    result = asyncio.run(
        _run_tool(
            "init_workspace",
            ctx,
            workspace=workspace,
            repo_url=repo_url,
            branch=branch,
            program_append=program_append,
        )
    )
    console = Console()
    if result.success:
        console.print(
            Panel.fit(
                f"Initialized autoresearch workspace at [green]{result.data['workspace']}[/green]\n"
                f"Branch: [blue]{result.data['branch']}[/blue]",
                title="AutoResearch",
                border_style="green",
            )
        )
    else:
        console.print(Panel(result.error or "Init failed", border_style="red"))
        sys.exit(1)


@autoresearch.command(name="prepare")
@click.option(
    "--workspace", "-w", default="./autoresearch", help="Autoresearch workspace"
)
@click.pass_obj
def autoresearch_prepare(obj: CliContext, workspace: str) -> None:
    """Run the autoresearch data-prep step (prepare.py)."""
    ctx = _tool_context(obj)
    result = asyncio.run(_run_tool("prepare", ctx, workspace=workspace))
    console = Console()
    if result.success:
        console.print("[green]Data preparation completed.[/green]")
    else:
        console.print(Panel(result.error or "Prepare failed", border_style="red"))
        sys.exit(1)


@autoresearch.command(name="run")
@click.option(
    "--workspace", "-w", default="./autoresearch", help="Autoresearch workspace"
)
@click.option(
    "--timeout",
    "-t",
    default=600,
    type=int,
    help="Maximum seconds to wait for the training run",
)
@click.pass_obj
def autoresearch_run(obj: CliContext, workspace: str, timeout: int) -> None:
    """Run a single fixed-time autoresearch experiment."""
    ctx = _tool_context(obj)
    result = asyncio.run(
        _run_tool("run_experiment", ctx, workspace=workspace, timeout=timeout)
    )
    console = Console()
    console.print_json(data=result.data)
    if not result.success:
        sys.exit(1)


@autoresearch.command(name="results")
@click.option(
    "--workspace", "-w", default="./autoresearch", help="Autoresearch workspace"
)
@click.pass_obj
def autoresearch_results(obj: CliContext, workspace: str) -> None:
    """Show the autoresearch results.tsv log."""
    ctx = _tool_context(obj)
    result = asyncio.run(_run_tool("results", ctx, workspace=workspace))
    console = Console()
    rows = result.data.get("rows", [])
    if not rows:
        console.print("[yellow]No results logged yet.[/yellow]")
        return
    console.print_json(data=result.data)


@autoresearch.command(name="status")
@click.option(
    "--workspace", "-w", default="./autoresearch", help="Autoresearch workspace"
)
@click.pass_obj
def autoresearch_status(obj: CliContext, workspace: str) -> None:
    """Show git status of the autoresearch workspace."""
    ctx = _tool_context(obj)
    result = asyncio.run(_run_tool("status", ctx, workspace=workspace))
    console = Console()
    console.print_json(data=result.data)


@autoresearch.command(name="propose")
@click.option(
    "--workspace", "-w", default="./autoresearch", help="Autoresearch workspace"
)
@click.option(
    "--hint",
    "-h",
    default=None,
    help="Hint to guide the next proposed edit",
)
@click.pass_obj
def autoresearch_propose(obj: CliContext, workspace: str, hint: str | None) -> None:
    """Ask the configured LLM to propose the next train.py edit."""
    ctx = _tool_context(obj)
    result = asyncio.run(
        _run_tool("propose_edit", ctx, workspace=workspace, user_hint=hint)
    )
    console = Console()
    if result.success:
        console.print(f"[bold]Hypothesis:[/bold] {result.data.get('hypothesis', '')}")
        console.print(f"[bold]Description:[/bold] {result.data.get('description', '')}")
        console.print(
            f"[bold]Proposed train.py length:[/bold] "
            f"{len(result.data.get('train_py', ''))} chars"
        )
    else:
        console.print(Panel(result.error or "Proposal failed", border_style="red"))
        sys.exit(1)


@autoresearch.command(name="step")
@click.option(
    "--workspace", "-w", default="./autoresearch", help="Autoresearch workspace"
)
@click.option(
    "--hint",
    "-h",
    default=None,
    help="Hint for the next experimental edit",
)
@click.option(
    "--timeout",
    "-t",
    default=600,
    type=int,
    help="Maximum seconds to wait for the training run",
)
@click.option(
    "--train-py",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Apply a specific train.py file instead of asking the LLM",
)
@click.option(
    "--description",
    "-d",
    default=None,
    help="Description for a manual step",
)
@click.pass_obj
def autoresearch_step(
    obj: CliContext,
    workspace: str,
    hint: str | None,
    timeout: int,
    train_py: str | None,
    description: str | None,
) -> None:
    """Run one full autoresearch step: propose, commit, train, keep/discard."""
    ctx = _tool_context(obj)
    kwargs: dict[str, Any] = {
        "workspace": workspace,
        "user_hint": hint,
        "timeout": timeout,
    }
    if train_py:
        kwargs["train_py"] = Path(train_py).read_text(encoding="utf-8")
    if description:
        kwargs["description"] = description

    result = asyncio.run(_run_tool("step", ctx, **kwargs))
    console = Console()
    console.print_json(data=result.data)
    if not result.success:
        sys.exit(1)


@autoresearch.command(name="loop")
@click.option(
    "--workspace", "-w", default="./autoresearch", help="Autoresearch workspace"
)
@click.option(
    "--iterations",
    "-n",
    default=3,
    type=int,
    help="Number of experiments to run",
)
@click.option(
    "--hint",
    "-h",
    default=None,
    help="Hint to guide each experimental edit",
)
@click.option(
    "--timeout",
    "-t",
    default=600,
    type=int,
    help="Maximum seconds to wait for each training run",
)
@click.pass_obj
def autoresearch_loop(
    obj: CliContext,
    workspace: str,
    iterations: int,
    hint: str | None,
    timeout: int,
) -> None:
    """Run multiple autoresearch steps back-to-back."""
    ctx = _tool_context(obj)
    result = asyncio.run(
        _run_tool(
            "loop",
            ctx,
            workspace=workspace,
            max_iterations=iterations,
            user_hint=hint,
            timeout=timeout,
        )
    )
    console = Console()
    console.print_json(data=result.data)
