"""Autoloop CLI command — start the autonomous closed-loop engine.

Usage:
    huginn autoloop "Optimize C-S-H defect kinetics" --iterations 5
    huginn autoloop --watch  # Watch mode: continuously monitor workspace
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from huginn.autoloop import AutoloopEngine, save_autoloop_snapshot
from huginn.cli.context import CliContext

logger = logging.getLogger(__name__)


def _load_star_plugins() -> None:
    """把 Star 插件挂进本进程 —— 与 serve 启动时加载的是同一套.

    autoloop CLI 此前**从不**加载 Star 插件, 于是插件面 (prompt 段 / plugin tools,
    如 asd_ste100 的 ste_lint、comms_lint) 在长程 run 里全部缺席. 这里复用服务端
    ``lifespan._load_star_plugins`` 的同一实现, 让 autoloop 与 serve 看到同一套插件
    (单一来源, 不在这里另写一份发现/加载逻辑).

    插件各自用 feature flag 门控自身行为, 加载本身不改默认语义. 设
    ``HUGINN_AUTOLOOP_NO_PLUGINS=1`` 可关闭 (回到插件缺席的旧行为).
    """
    if os.environ.get("HUGINN_AUTOLOOP_NO_PLUGINS", "0") == "1":
        return
    try:
        from huginn.lifespan import _load_star_plugins as _load

        asyncio.run(_load())
    except Exception:  # 防御: 插件加载失败不阻塞 autoloop 启动
        logger.warning("Star plugin loading failed; autoloop runs without plugins", exc_info=True)


def _maybe_agent_factory() -> Any:
    """多智能体协作通电开关 — 默认 None (纯单 agent, 零额外成本).

    盲重建 / failure_inverter / BranchIncubator 三条协作路径都靠
    ``engine._agent_factory``; 但 CLI 此前**从没注入**它, 于是它们在长程 run 里
    全部静默空转 (engine.py 注释声称"由 RCBench runner / CLI 注入"—— 实际未接).
    置 ``HUGINN_ENABLE_AGENT_COLLAB=1`` 才构造, 默认关 → 默认行为/成本不变.
    """
    if os.environ.get("HUGINN_ENABLE_AGENT_COLLAB", "0") != "1":
        return None
    try:
        from huginn.server_core import get_agent_factory

        return get_agent_factory()
    except Exception:  # 防御: 工厂构造失败退回单 agent
        import logging

        logging.getLogger(__name__).warning(
            "agent collab factory build failed; fall back to single-agent",
            exc_info=True,
        )
        return None


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
@click.option(
    "--wall-clock-budget",
    "wall_clock_budget",
    default=0,
    type=int,
    help="Long-horizon mode: wall-clock budget in SECONDS. >0 creates a persistent "
    "goal and enables persistent-goal mode, so heuristic early-stops (darwin "
    "stagnation / belief / surprise convergence) defer until the budget is spent "
    "or the -i iteration cap is reached.",
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
    wall_clock_budget: int,
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

    # 挂载 Star 插件 (prompt 段 / plugin tools), 与 serve 同一套.
    _load_star_plugins()

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
            load_checkpoint,
            resume_engine_from_checkpoint,
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
        engine = AutoloopEngine(
            workspace=obj.workspace,
            agent_factory=_maybe_agent_factory(),
        )

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
    elif objective and (success_criteria or wall_clock_budget > 0):
        goal = scheduler.create_goal(
            objective=objective,
            success_criteria=list(success_criteria),
            max_iterations=iterations,
        )
        if wall_clock_budget > 0:
            # 长程探索: 目标挂上挂钟预算并置 active, 同时打开持久目标模式.
            # 这样 darwin/belief/surprise 这类启发式早停在预算未耗尽时不再终止
            # 整个 run, 循环自主推进到目标达成或预算/迭代上限耗尽.
            from huginn.utils.common import now_iso

            scheduler.update_goal(
                goal.id,
                wall_clock_budget_seconds=float(wall_clock_budget),
                started_at=now_iso(),
                status="active",
            )
            os.environ["HUGINN_PERSISTENT_GOAL_MODE"] = "1"
            console.print(
                f"[blue]Long-horizon goal:[/blue] {goal.id}\n"
                f"  wall-clock budget: {wall_clock_budget}s, "
                f"iteration cap: {iterations}"
            )
        else:
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
