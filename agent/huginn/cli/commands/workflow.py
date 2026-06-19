"""Workflow template command."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from huginn.cli.context import CliContext
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext
from huginn.workflows.checkpoint import WorkflowCheckpoint
from huginn.workflows.engine import WorkflowEngine
from huginn.workflows.templates import get_template


@click.command("workflow")
@click.argument("template")
@click.argument("args", nargs=-1)
@click.option(
    "--checkpoint",
    "-c",
    help="Path to checkpoint file (auto-generated if omitted)",
)
@click.option(
    "--resume",
    "-r",
    is_flag=True,
    help="Resume from the checkpoint file if it exists",
)
@click.pass_obj
def workflow(
    ctx: CliContext,
    template: str,
    args: tuple[str, ...],
    checkpoint: str | None,
    resume: bool,
) -> None:
    """Run a workflow template with KEY=VALUE arguments."""
    template_fn = get_template(template)
    if not template_fn:
        ctx.console.print(f"[red]Template '{template}' not found[/red]")
        return

    kwargs: dict[str, Any] = {}
    for a in args:
        if "=" not in a:
            ctx.console.print(f"[yellow]Ignoring malformed arg: {a}[/yellow]")
            continue
        key, value = a.split("=", 1)
        with contextlib.suppress(Exception):
            value = json.loads(value)
        kwargs[key] = value

    try:
        stages = template_fn(**kwargs)
    except Exception as e:
        ctx.console.print(f"[red]Failed to build workflow: {e}[/red]")
        return

    checkpoint_path: Path
    if checkpoint:
        checkpoint_path = Path(checkpoint)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = WorkflowCheckpoint.default_path(
            ctx.workspace, f"{template}_{ts}"
        )

    engine = WorkflowEngine(ToolRegistry)
    context = ToolContext(session_id="workflow", workspace=str(ctx.workspace))

    if resume and checkpoint_path.exists():
        ctx.console.print(f"[dim]Resuming workflow from {checkpoint_path.name}[/dim]")
        result = asyncio.run(engine.resume(stages, context, checkpoint_path))
    else:
        result = asyncio.run(
            engine.execute(stages, context, checkpoint_path=checkpoint_path)
        )

    ctx.console.print(
        json.dumps(
            {
                "success": result.success,
                "total_walltime": result.total_walltime,
                "stages": {
                    sid: {
                        "name": s.name,
                        "status": s.status,
                        "attempts": s.attempts,
                        "error": s.result.error if s.result else None,
                    }
                    for sid, s in result.stages.items()
                },
                "outputs": result.outputs,
                "error": result.error,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
