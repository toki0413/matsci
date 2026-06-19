"""Multi-agent swarm commands."""

from __future__ import annotations

import asyncio

import click
from rich.panel import Panel

from huginn.cli.context import CliContext, build_agent_from_ctx


@click.group()
@click.pass_obj
def swarm(ctx: CliContext) -> None:
    """Multi-agent swarm commands."""


@swarm.command("run")
@click.argument("task")
@click.option("--profile", "-p", default="lead", help="Agent profile to use as workers")
@click.pass_obj
def swarm_run(ctx: CliContext, task: str, profile: str) -> None:
    """Run a task through a multi-agent swarm."""
    from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent

    agent = build_agent_from_ctx(ctx, profile_id=profile)
    if agent is None:
        return
    try:
        workers = [
            SwarmAgent(
                "planner", AgentRole.PLANNER, agent, "Break the task into steps."
            ),
            SwarmAgent(
                "scientist", AgentRole.SCIENTIST, agent, "Choose physical models."
            ),
            SwarmAgent("coder", AgentRole.CODER, agent, "Write code or tool calls."),
            SwarmAgent("executor", AgentRole.EXECUTOR, agent, "Run the solution."),
            SwarmAgent("critic", AgentRole.CRITIC, agent, "Review correctness."),
        ]
        result = asyncio.run(HuginnSwarm(workers).run(task))
        ctx.console.print(
            Panel(
                f"[bold blue]Swarm Result[/bold blue]\n{result['final_output']}",
                border_style="blue",
            )
        )
        for step in result["trace"]:
            ctx.console.print(
                f"  [{step['role']}] {step['agent_name']} ({step['duration_ms']:.0f}ms)"
            )
    finally:
        agent.close()
