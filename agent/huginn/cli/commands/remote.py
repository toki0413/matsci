"""Remote execution job management commands.

These commands let users inspect and control jobs submitted via the remote
HPC backend across CLI invocations, because ``RemoteExecutor`` persists job
records to ``<workspace>/.huginn/remote_jobs.json``.
"""

from __future__ import annotations

import contextlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.table import Table

from huginn.cli.context import CliContext


def _job_store(ctx: CliContext) -> Any:
    """Return a RemoteJobStore for the current workspace."""
    from huginn.execution.remote_job_store import RemoteJobStore

    return RemoteJobStore(workspace=ctx.workspace)


def _maybe_executor(ctx: CliContext) -> Any | None:
    """Build a RemoteExecutor if the config has HPC credentials."""
    from huginn.execution.remote_executor import build_executor

    cfg = ctx.load_config()
    executor = build_executor(cfg)
    # build_executor falls back to SandboxExecutor when remote is not configured.
    if executor.__class__.__name__ == "RemoteExecutor":
        return executor
    return None


@click.group(name="remote")
@click.pass_obj
def remote(ctx: CliContext) -> None:
    """Manage remote HPC jobs submitted by Huginn."""


@remote.command("list")
@click.option("--limit", "-n", default=20, type=int, help="Max jobs to show")
@click.pass_obj
def remote_list(ctx: CliContext, limit: int) -> None:
    """List tracked remote jobs."""
    store = _job_store(ctx)
    jobs = store.list_jobs()[:limit]

    if not jobs:
        ctx.console.print("[dim]No remote jobs tracked yet.[/dim]")
        return

    table = Table(title="Remote Jobs")
    table.add_column("Local ID", style="cyan")
    table.add_column("Scheduler ID", style="magenta")
    table.add_column("Status")
    table.add_column("Queue", style="dim")
    table.add_column("Submitted")
    table.add_column("Command", style="dim")

    for job in jobs:
        submitted = datetime.fromtimestamp(job.submitted_at).strftime("%Y-%m-%d %H:%M")
        cmd = " ".join(job.command)
        if len(cmd) > 50:
            cmd = cmd[:47] + "..."
        status_style = {
            "PENDING": "yellow",
            "RUNNING": "blue",
            "COMPLETED": "green",
            "FAILED": "red",
            "CANCELLED": "dim",
        }.get(job.status, "white")
        table.add_row(
            job.local_id,
            job.scheduler_id,
            f"[{status_style}]{job.status}[/{status_style}]",
            job.queue or "-",
            submitted,
            cmd,
        )

    ctx.console.print(table)


@remote.command("status")
@click.argument("local_id")
@click.option(
    "--refresh/--no-refresh",
    default=True,
    help="Poll the scheduler for the latest status",
)
@click.pass_obj
def remote_status(ctx: CliContext, local_id: str, refresh: bool) -> None:
    """Show details for a tracked remote job."""
    store = _job_store(ctx)
    record = store.get(local_id)
    if record is None:
        ctx.console.print(f"[red]Job '{local_id}' not found.[/red]")
        return

    if refresh and record.status not in ("COMPLETED", "FAILED", "CANCELLED"):
        executor = _maybe_executor(ctx)
        if executor is not None:
            try:
                record = executor.refresh_job(local_id) or record
            except Exception as e:
                ctx.console.print(f"[yellow]Could not refresh status: {e}[/yellow]")

    ctx.console.print(
        {
            "local_id": record.local_id,
            "scheduler_id": record.scheduler_id,
            "status": record.status,
            "exit_code": record.exit_code,
            "queue": record.queue,
            "cwd": record.cwd,
            "command": record.command,
            "submitted_at": datetime.fromtimestamp(record.submitted_at).isoformat(),
            "completed_at": (
                datetime.fromtimestamp(record.completed_at).isoformat()
                if record.completed_at
                else None
            ),
            "message": record.message,
        }
    )


@remote.command("cancel")
@click.argument("local_id")
@click.pass_obj
def remote_cancel(ctx: CliContext, local_id: str) -> None:
    """Cancel a running remote job."""
    store = _job_store(ctx)
    record = store.get(local_id)
    if record is None:
        ctx.console.print(f"[red]Job '{local_id}' not found.[/red]")
        return

    executor = _maybe_executor(ctx)
    if executor is None:
        ctx.console.print(
            "[yellow]Remote HPC not configured; cancelling only the local record.[/yellow]"
        )
        from huginn.execution.remote_job_store import RemoteJobRecord

        updated = RemoteJobRecord(
            local_id=record.local_id,
            scheduler_id=record.scheduler_id,
            command=record.command,
            cwd=record.cwd,
            queue=record.queue,
            status="CANCELLED",
            exit_code=record.exit_code,
            submitted_at=record.submitted_at,
            completed_at=datetime.now().timestamp(),
        )
        store.add_or_update(updated)
        ctx.console.print(f"[green]✓[/green] Marked '{local_id}' as cancelled locally.")
        return

    if executor.cancel_job(local_id):
        ctx.console.print(f"[green]✓[/green] Cancelled '{local_id}'.")
    else:
        ctx.console.print(f"[red]Failed to cancel '{local_id}'.[/red]")


@remote.command("logs")
@click.argument("local_id")
@click.option(
    "--stderr", "show_stderr", is_flag=True, help="Show stderr instead of stdout"
)
@click.pass_obj
def remote_logs(ctx: CliContext, local_id: str, show_stderr: bool) -> None:
    """Show stdout/stderr for a tracked remote job.

    Looks for scheduler output files in the job's local working directory.
    """
    kind = "stderr" if show_stderr else "stdout"
    store = _job_store(ctx)
    record = store.get(local_id)
    if record is None:
        ctx.console.print(f"[red]Job '{local_id}' not found.[/red]")
        return

    cwd = Path(record.cwd)
    scheduler_id = record.scheduler_id

    candidates: list[Path] = []
    if kind == "stdout":
        candidates = [
            cwd / f"slurm-{scheduler_id}.out",
            cwd / f"pbs-{scheduler_id}.out",
        ]
    else:
        candidates = [
            cwd / f"slurm-{scheduler_id}.err",
            cwd / f"pbs-{scheduler_id}.err",
        ]

    for path in candidates:
        if path.exists():
            ctx.console.print(path.read_text(encoding="utf-8", errors="ignore"))
            return

    ctx.console.print(f"[yellow]No {kind} log found for {local_id} in {cwd}.[/yellow]")


@remote.command("watch")
@click.option(
    "--interval",
    "-i",
    default=5,
    type=int,
    help="Refresh interval in seconds",
)
@click.option(
    "--exit-on-done/--no-exit-on-done",
    default=False,
    help="Exit automatically when all tracked jobs are terminal",
)
@click.option("--limit", "-n", default=20, type=int, help="Max jobs to show")
@click.pass_obj
def remote_watch(
    ctx: CliContext, interval: int, exit_on_done: bool, limit: int
) -> None:
    """Live-watch tracked remote jobs."""
    from rich.live import Live

    store = _job_store(ctx)
    executor = _maybe_executor(ctx)

    def _build_table() -> Table:
        jobs = store.list_jobs()[:limit]
        table = Table(title="Remote Jobs (live)")
        table.add_column("Local ID", style="cyan")
        table.add_column("Scheduler ID", style="magenta")
        table.add_column("Status")
        table.add_column("Queue", style="dim")
        table.add_column("Submitted")
        table.add_column("Command", style="dim")

        if not jobs:
            table.add_row("-", "-", "[dim]no jobs[/dim]", "-", "-", "-")
            return table

        for job in jobs:
            submitted = datetime.fromtimestamp(job.submitted_at).strftime(
                "%Y-%m-%d %H:%M"
            )
            cmd = " ".join(job.command)
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            status_style = {
                "PENDING": "yellow",
                "RUNNING": "blue",
                "COMPLETED": "green",
                "FAILED": "red",
                "CANCELLED": "dim",
            }.get(job.status, "white")
            table.add_row(
                job.local_id,
                job.scheduler_id,
                f"[{status_style}]{job.status}[/{status_style}]",
                job.queue or "-",
                submitted,
                cmd,
            )
        return table

    def _refresh_active() -> None:
        if executor is None:
            return
        for job in store.list_jobs()[:limit]:
            if job.status not in ("COMPLETED", "FAILED", "CANCELLED"):
                with contextlib.suppress(Exception):
                    executor.refresh_job(job.local_id)

    ctx.console.print(
        f"[dim]Watching remote jobs every {interval}s (Ctrl+C to stop)[/dim]"
    )
    try:
        with Live(_build_table(), refresh_per_second=1, console=ctx.console) as live:
            while True:
                _refresh_active()
                live.update(_build_table())
                if exit_on_done:
                    jobs = store.list_jobs()[:limit]
                    if jobs and all(
                        j.status in ("COMPLETED", "FAILED", "CANCELLED") for j in jobs
                    ):
                        break
                time.sleep(interval)
    except KeyboardInterrupt:
        ctx.console.print("[dim]Stopped watching.[/dim]")


@remote.command("export")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "csv"]),
    default="json",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.File("w", encoding="utf-8"),
    default="-",
    help="Output file (default: stdout)",
)
@click.option("--limit", "-n", default=0, type=int, help="Limit rows (0 = all)")
@click.pass_obj
def remote_export(ctx: CliContext, fmt: str, output: click.File, limit: int) -> None:
    """Export tracked remote jobs to JSON or CSV."""
    store = _job_store(ctx)
    jobs = store.list_jobs()
    if limit > 0:
        jobs = jobs[:limit]

    rows = [
        {
            "local_id": j.local_id,
            "scheduler_id": j.scheduler_id,
            "status": j.status,
            "exit_code": j.exit_code,
            "queue": j.queue,
            "cwd": j.cwd,
            "command": " ".join(j.command),
            "submitted_at": datetime.fromtimestamp(j.submitted_at).isoformat(),
            "completed_at": (
                datetime.fromtimestamp(j.completed_at).isoformat()
                if j.completed_at
                else None
            ),
            "message": j.message,
        }
        for j in jobs
    ]

    if fmt == "json":
        import json

        output.write(json.dumps(rows, indent=2, ensure_ascii=False))
        output.write("\n")
    else:
        import csv

        writer = csv.DictWriter(
            output,
            fieldnames=[
                "local_id",
                "scheduler_id",
                "status",
                "exit_code",
                "queue",
                "cwd",
                "command",
                "submitted_at",
                "completed_at",
                "message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


@remote.command("sync")
@click.option("--limit", "-n", default=50, type=int, help="Max jobs to refresh")
@click.pass_obj
def remote_sync(ctx: CliContext, limit: int) -> None:
    """Refresh statuses of all non-terminal jobs from the remote scheduler."""
    store = _job_store(ctx)
    executor = _maybe_executor(ctx)
    if executor is None:
        ctx.console.print(
            "[yellow]Remote HPC not configured; cannot sync with scheduler.[/yellow]"
        )
        return

    jobs = [
        j
        for j in store.list_jobs()[:limit]
        if j.status not in ("COMPLETED", "FAILED", "CANCELLED")
    ]
    if not jobs:
        ctx.console.print("[dim]No active jobs to sync.[/dim]")
        return

    refreshed = 0
    failed = 0
    for job in jobs:
        try:
            executor.refresh_job(job.local_id)
            refreshed += 1
        except Exception as e:
            ctx.console.print(f"[yellow]Could not refresh {job.local_id}: {e}[/yellow]")
            failed += 1

    ctx.console.print(
        f"[green]✓[/green] Synced {refreshed} job(s)"
        + (f", {failed} failed" if failed else "")
    )


@remote.command("prune")
@click.option(
    "--status",
    default="COMPLETED,FAILED,CANCELLED",
    help="Comma-separated terminal statuses to prune",
)
@click.option(
    "--older-than-days",
    type=int,
    help="Only prune records older than N days",
)
@click.option(
    "--delete-logs/--keep-logs",
    default=False,
    help="Also delete local scheduler log files",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_obj
def remote_prune(
    ctx: CliContext,
    status: str,
    older_than_days: int | None,
    delete_logs: bool,
    yes: bool,
) -> None:
    """Remove old or terminal remote job records from the local store."""
    store = _job_store(ctx)
    target_statuses = {s.strip().upper() for s in status.split(",") if s.strip()}
    now = datetime.now().timestamp()
    cutoff = (now - older_than_days * 86400) if older_than_days else None

    to_prune = []
    for job in store.list_jobs():
        if job.status not in target_statuses:
            continue
        if cutoff is not None and (job.completed_at or job.submitted_at) > cutoff:
            continue
        to_prune.append(job)

    if not to_prune:
        ctx.console.print("[dim]No matching jobs to prune.[/dim]")
        return

    if not yes:
        ctx.console.print(
            f"Will prune {len(to_prune)} job record(s) "
            f"(status in {target_statuses}{', logs too' if delete_logs else ''})."
        )
        click.confirm("Continue?", abort=True)

    removed = 0
    for job in to_prune:
        if store.remove(job.local_id):
            removed += 1
        if delete_logs:
            cwd = Path(job.cwd)
            for suffix in (".out", ".err"):
                for prefix in ("slurm-", "pbs-"):
                    (cwd / f"{prefix}{job.scheduler_id}{suffix}").unlink(
                        missing_ok=True
                    )

    ctx.console.print(f"[green]✓[/green] Pruned {removed} job record(s).")
