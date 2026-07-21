"""Autoloop CLI command — start the autonomous closed-loop engine.

Usage:
    huginn autoloop "Optimize C-S-H defect kinetics" --iterations 5
    huginn autoloop --watch  # Watch mode: continuously monitor workspace
"""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from huginn.autoloop import AutoloopEngine, save_autoloop_snapshot
from huginn.cli.context import CliContext


@click.command()
@click.argument("objective", required=False, default="")
@click.option(
    "--iterations",
    "-i",
    default=20,
    type=int,
    help="Maximum autonomous loop iterations (default: 20)",
)
@click.option(
    "--watch",
    "-w",
    is_flag=True,
    help="Watch mode: continuously monitor workspace and auto-trigger loops",
)
@click.option(
    "--interval",
    default=30,
    type=int,
    help="Watch mode: check interval in seconds (default: 30)",
)
@click.option(
    "--no-progressive-budget",
    "no_progressive_budget",
    is_flag=True,
    help="Disable progressive budget tiering (allow all plan modes at every iteration)",
)
@click.option(
    "--goal",
    "goal_id",
    default=None,
    help="Resume a persisted goal by ID (from goals.json) instead of creating a new one",
)
@click.option(
    "--success-criteria",
    "-s",
    "success_criteria",
    multiple=True,
    help="Success criterion (keyword that must appear in validation output). Repeatable: -s foo -s bar",
)
@click.pass_obj
def autoloop(
    obj: CliContext,
    objective: str,
    iterations: int,
    watch: bool,
    interval: int,
    no_progressive_budget: bool,
    goal_id: str | None,
    success_criteria: tuple[str, ...],
) -> None:
    """Run the autonomous closed-loop engine.

    OBJECTIVE: Natural language goal for the autonomous loop.
    If not provided, the agent will infer a goal from the workspace state.

    Examples:
        huginn autoloop "Optimize C-S-H defect kinetics"
        huginn autoloop "Find stable phase" -s tests_passed -s r_phys
        huginn autoloop --watch --interval 60
        huginn autoloop --goal goal_abc12345
    """
    console = obj.console

    if not objective and not watch:
        console.print(
            Panel(
                "[bold yellow]Usage:[/bold yellow]\n"
                "  huginn autoloop [OBJECTIVE]\n"
                "  huginn autoloop --watch\n\n"
                "[dim]Examples:[/dim]\n"
                "  huginn autoloop 'Optimize C-S-H defect kinetics'\n"
                "  huginn autoloop --watch --interval 60",
                title="Autoloop Help",
                border_style="blue",
            )
        )
        return

    # H3: 统一 resume 入口 — 有 checkpoint 就走 resume_engine_from_checkpoint,
    # 让 audit 校验 + drift 检测 + engine_state + hypothesis_graph 一起跑.
    # task_id 用 workspace.name (跟 rcb_runner 一致). 无 checkpoint / resume 失败
    # → 退回 fresh engine, 不阻塞用户.
    engine = None
    try:
        from huginn.runtime.checkpoint import (
            load_checkpoint, resume_engine_from_checkpoint,
        )
        _cp = load_checkpoint(obj.workspace.name, obj.workspace)
        if _cp is not None:
            console.print(
                f"[dim]Found checkpoint at step {_cp.step_id}, resuming...[/dim]"
            )
            engine = resume_engine_from_checkpoint(_cp, obj.workspace)
            console.print(
                f"[green]Resumed from checkpoint[/green] "
                f"(task_id={obj.workspace.name}, step={_cp.step_id})"
            )
    except Exception as _e:
        console.print(
            f"[yellow]Checkpoint resume failed, starting fresh:[/yellow] {_e}"
        )
    if engine is None:
        engine = AutoloopEngine(workspace=obj.workspace)

    # Goal resolution: --goal resumes a persisted goal; --success-criteria
    # creates a new one. Neither → no goal, run() behaves as before.
    goal = None
    from huginn.autoloop.goal_scheduler import GoalScheduler

    scheduler = GoalScheduler()
    if goal_id:
        goal = scheduler.get_goal(goal_id)
        if goal is None:
            console.print(f"[red]Goal not found: {goal_id}[/red]")
            return
        if not objective:
            objective = goal.objective
        console.print(f"[blue]Resuming goal:[/blue] {goal.id} ({goal.objective})")
    elif success_criteria and objective:
        goal = scheduler.create_goal(
            objective=objective,
            success_criteria=list(success_criteria),
            max_iterations=iterations,
        )
        console.print(
            f"[blue]Created goal:[/blue] {goal.id}\n"
            f"  criteria: {list(success_criteria)}"
        )
    engine._goal_scheduler = scheduler

    if watch:
        console.print(
            Panel(
                f"[bold green]Autoloop Watch Mode[/bold green]\n"
                f"Workspace: {obj.workspace}\n"
                f"Check interval: {interval}s\n"
                f"Press Ctrl+C to stop",
                title="Watching",
                border_style="green",
            )
        )
        try:
            asyncio.run(_watch_loop(
                engine, console, interval, iterations,
                progressive_budget=not no_progressive_budget,
            ))
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch mode stopped.[/yellow]")
    else:
        console.print(
            Panel(
                f"[bold green]Autoloop[/bold green]\n"
                f"Objective: {objective}\n"
                f"Max iterations: {iterations}",
                title="Starting",
                border_style="green",
            )
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running autonomous loop...", total=None)
            try:
                result = asyncio.run(engine.run_cognitive(
                    objective=objective,
                    max_iterations=iterations,
                    progressive_budget=not no_progressive_budget,
                    goal=goal,
                ))
                progress.update(task, completed=True)

                # Persist a reusable snapshot so DeliAutoResearch (or a re-run)
                # can pick up the result without re-instantiating AutoloopEngine.
                snap_path = save_autoloop_snapshot(result, obj.workspace)

                console.print(
                    Panel(
                        f"[bold green]Loop Complete[/bold green]\n"
                        f"Run ID: {result.run_id}\n"
                        f"Success: {'Yes' if result.success else 'No'}\n"
                        f"Total time: {result.total_time_seconds:.1f}s\n"
                        f"Report: {result.report_path or 'N/A'}\n"
                        f"Snapshot: {snap_path or 'N/A'}",
                        title="Result",
                        border_style="green" if result.success else "red",
                    )
                )

                # Show phase summary
                console.print("\n[bold]Phase Summary:[/bold]")
                for phase in result.phases:
                    status_color = "green" if phase.status == "completed" else "red" if phase.status == "failed" else "yellow"
                    duration = (phase.end_time or 0) - (phase.start_time or 0) if phase.start_time and phase.end_time else 0
                    console.print(
                        f"  [{status_color}]{phase.status:12}[/{status_color}] "
                        f"{phase.name:15} ({duration:.1f}s)"
                        f"{f' [red]{phase.error}[/red]' if phase.error else ''}"
                    )

            except Exception as e:
                progress.update(task, completed=True)
                console.print(f"[red]Autoloop failed: {e}[/red]")


async def _watch_loop(
    engine: AutoloopEngine,
    console: Console,
    interval: int,
    max_iterations: int,
    progressive_budget: bool = True,
) -> None:
    """Continuously watch workspace and trigger loops when changes detected."""
    import time

    iteration = 0
    while True:
        iteration += 1
        console.print(f"\n[dim]Watch check #{iteration}...[/dim]")

        # Quick perceive check
        context = engine._perceive()
        if context:
            console.print(
                f"[green]Changes detected:[/green] {len(context.get('changed_files', []))} files"
            )

            # Infer objective from changes if not provided
            objective = _infer_objective(context)
            console.print(f"[blue]Inferred objective:[/blue] {objective}")

            result = await engine.run_cognitive(
                objective=objective,
                max_iterations=max_iterations,
                progressive_budget=progressive_budget,
            )

            console.print(
                f"[green]Loop #{iteration} complete:[/green] "
                f"success={result.success}, time={result.total_time_seconds:.1f}s"
            )
            if result.report_path:
                console.print(f"  Report: {result.report_path}")
            snap = save_autoloop_snapshot(result, engine.workspace)
            if snap:
                console.print(f"  [dim]Snapshot: {snap}[/dim]")
        else:
            console.print("[dim]No changes detected.[/dim]")

        await asyncio.sleep(interval)


def _infer_objective(context: dict[str, Any]) -> str:
    """Infer a research objective from the perceived context."""
    changed = context.get("changed_files", [])
    errors = context.get("error_patterns", [])

    if errors:
        return f"Fix errors in {len(errors)} log files and validate changes"

    if changed:
        # Try to infer from file names
        code_files = [f for f in changed if any(f.endswith(ext) for ext in [".py", ".rs", ".ts"])]
        if code_files:
            return f"Analyze and improve code changes in {len(code_files)} files"

        data_files = [f for f in changed if any(f.endswith(ext) for ext in [".cif", ".poscar", ".vasp", ".json"])]
        if data_files:
            return f"Process and validate {len(data_files)} new data files"

    return "Analyze workspace changes and suggest improvements"
