"""Scheduler command — manage recurring Huginn tasks."""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from huginn.cli.context import CliContext
from huginn.scheduler import ScheduleManager


@click.group(name="scheduler")
def scheduler():
    """Schedule recurring commands (workflows, backups, reports)."""
    pass


@scheduler.command(name="add")
@click.option(
    "--cron",
    required=True,
    help="Cron expression with 5 fields: minute hour day month day-of-week",
)
@click.option("--command", required=True, help="Shell command to execute")
@click.pass_obj
def scheduler_add(obj: CliContext, cron: str, command: str) -> None:
    """Add a scheduled command."""
    manager = ScheduleManager(obj.workspace)
    job_id = manager.add(cron, command)
    Console().print(f"[green]Scheduled job {job_id}:[/green] {cron} -> {command}")


@scheduler.command(name="list")
@click.pass_obj
def scheduler_list(obj: CliContext) -> None:
    """List scheduled jobs."""
    manager = ScheduleManager(obj.workspace)
    jobs = manager.list()
    console = Console()
    if not jobs:
        console.print("[yellow]No scheduled jobs.[/yellow]")
        return

    table = Table(title="Scheduled Jobs")
    table.add_column("ID")
    table.add_column("Cron")
    table.add_column("Command")
    table.add_column("Enabled")
    table.add_column("Runs")
    table.add_column("Last Run")

    for job in jobs:
        table.add_row(
            job.id,
            job.cron,
            job.command,
            "yes" if job.enabled else "no",
            str(job.run_count),
            job.last_run or "never",
        )
    console.print(table)


@scheduler.command(name="remove")
@click.argument("job_id")
@click.pass_obj
def scheduler_remove(obj: CliContext, job_id: str) -> None:
    """Remove a scheduled job by ID."""
    manager = ScheduleManager(obj.workspace)
    if manager.remove(job_id):
        Console().print(f"[green]Removed job {job_id}.[/green]")
    else:
        Console().print(f"[red]Job {job_id} not found.[/red]")
        sys.exit(1)


@scheduler.command(name="enable")
@click.argument("job_id")
@click.pass_obj
def scheduler_enable(obj: CliContext, job_id: str) -> None:
    """Enable a scheduled job."""
    manager = ScheduleManager(obj.workspace)
    if manager.enable(job_id, True):
        Console().print(f"[green]Enabled job {job_id}.[/green]")
    else:
        Console().print(f"[red]Job {job_id} not found.[/red]")
        sys.exit(1)


@scheduler.command(name="disable")
@click.argument("job_id")
@click.pass_obj
def scheduler_disable(obj: CliContext, job_id: str) -> None:
    """Disable a scheduled job."""
    manager = ScheduleManager(obj.workspace)
    if manager.enable(job_id, False):
        Console().print(f"[green]Disabled job {job_id}.[/green]")
    else:
        Console().print(f"[red]Job {job_id} not found.[/red]")
        sys.exit(1)


@scheduler.command(name="run-now")
@click.pass_obj
def scheduler_run_now(obj: CliContext) -> None:
    """Execute any currently due jobs once."""
    manager = ScheduleManager(obj.workspace)
    console = Console()
    try:
        results = manager.run_due()
    except Exception as exc:
        console.print(Panel(f"Scheduler error: {exc}", border_style="red"))
        sys.exit(1)

    if not results:
        console.print("[yellow]No jobs due.[/yellow]")
        return

    for result in results:
        status = "[green]OK[/green]" if result.get("success") else "[red]FAIL[/red]"
        console.print(f"{status} {result['job_id']}: {result['command']}")
        if "error" in result:
            console.print(f"  {result['error']}")


@scheduler.command(name="start")
@click.option(
    "--interval",
    default=60,
    help="Polling interval in seconds",
)
@click.pass_obj
def scheduler_start(obj: CliContext, interval: int) -> None:
    """Start the scheduler daemon (runs until interrupted)."""
    manager = ScheduleManager(obj.workspace)
    console = Console()
    console.print(
        f"[bold blue]Scheduler started for {obj.workspace}[/bold blue] "
        f"(interval={interval}s; Ctrl+C to stop)"
    )
    try:
        manager.run_blocking(interval=interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Scheduler stopped.[/yellow]")
