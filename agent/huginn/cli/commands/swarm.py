"""Multi-agent swarm & multi-model team commands.

swarm: 老的单模型多角色模式 (向后兼容)
team:  新的多模型团队模式, 每个 role 可以用不同的 LLM
"""

from __future__ import annotations

import asyncio

import click

from huginn.cli.context import CliContext, build_agent_from_ctx
from huginn.cli.design_system import get_design_system


# ── 老的 swarm 命令 (向后兼容) ─────────────────────────


@click.group()
@click.pass_obj
def swarm(ctx: CliContext) -> None:
    """Multi-agent swarm commands (legacy single-model mode)."""


@swarm.command("run")
@click.argument("task")
@click.option("--profile", "-p", default="lead", help="Agent profile to use as workers")
@click.pass_obj
def swarm_run(ctx: CliContext, task: str, profile: str) -> None:
    """Run a task through a multi-agent swarm (single model, multiple roles)."""
    from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent

    ds = get_design_system()
    agent = build_agent_from_ctx(ctx, profile_id=profile)
    if agent is None:
        ds.error("No provider configured.")
        return
    try:
        workers = [
            SwarmAgent("planner", AgentRole.PLANNER, agent, "Break the task into steps."),
            SwarmAgent("scientist", AgentRole.SCIENTIST, agent, "Choose physical models."),
            SwarmAgent("coder", AgentRole.CODER, agent, "Write code or tool calls."),
            SwarmAgent("executor", AgentRole.EXECUTOR, agent, "Run the solution."),
            SwarmAgent("critic", AgentRole.CRITIC, agent, "Review correctness."),
        ]
        result = asyncio.run(HuginnSwarm(workers).run(task))
        ds.dialog(title="Swarm Result", content=result["final_output"])
        for step in result["trace"]:
            ds.info(f"[{step['role']}] {step['agent_name']} ({step['duration_ms']:.0f}ms)")
    finally:
        agent.close()


# ── 新的 team 命令 (多模型团队) ────────────────────────


@click.group()
@click.pass_obj
def team(ctx: CliContext) -> None:
    """Multi-model team commands.

    让不同 LLM 各司其职: 规划用强推理模型, 写代码用工具调用强的模型,
    审查用另一个模型避免自我偏见. 在 huginn.toml 里配置多个 [[agents]]
    profile (每个指向不同的 [[models]] alias) 即可自动组队.
    """


@team.command("list")
@click.pass_obj
def team_list(ctx: CliContext) -> None:
    """Show team members and their model capabilities."""
    from huginn.agents.team import ModelTeam

    ds = get_design_system()
    cfg = ctx.load_config()
    model_team = ModelTeam.from_config(cfg)
    members = model_team.list_members()

    if not members:
        ds.warning("No team members. Configure multiple [[agents]] profiles in huginn.toml.")
        ds.info("Each profile should point to a different [[models]] alias.")
        return

    rows = []
    for m in members:
        caps_str = " ".join(
            k for k, v in m["caps"].items() if v
        ) or "none"
        rows.append([
            m["role"],
            m["name"],
            m["model"] or "(unknown)",
            caps_str,
        ])

    ds.table(
        title="Model Team",
        headers=["Role", "Member", "Model", "Capabilities"],
        rows=rows,
    )


@team.command("run")
@click.argument("task")
@click.option(
    "--show-plan",
    is_flag=True,
    help="Print the execution plan before running",
)
@click.pass_obj
def team_run(ctx: CliContext, task: str, show_plan: bool) -> None:
    """Run a task through the multi-model team."""
    from huginn.agents.team import ModelTeam

    ds = get_design_system()
    cfg = ctx.load_config()
    model_team = ModelTeam.from_config(cfg)

    if not model_team.members:
        ds.error("No team members. Configure multiple [[agents]] profiles first.")
        ds.info("Run `huginn configure` or edit huginn.toml to add models and agent profiles.")
        return

    # 列出团队阵容
    ds.info(f"Team assembled with {len(model_team.members)} members:")
    for m in model_team.list_members():
        caps_str = " ".join(
            k for k, v in m["caps"].items() if v
        ) or "none"
        ds.info(f"  [{m['role']}] {m['name']} → {m['model'] or '(unknown)'} ({caps_str})")

    try:
        result = asyncio.run(model_team.run(task))

        if show_plan and result["context"].get("plan"):
            ds.dialog(title="Plan", content=result["context"]["plan"])

        ds.dialog(title="Team Result", content=result["final_output"])

        if result.get("review"):
            ds.dialog(title="Review", content=result["review"], color="yellow")

        ds.info("Execution trace:")
        for step in result["trace"]:
            ds.info(
                f"  [{step['role']}] {step['member']} "
                f"({step['model']}) {step['duration_ms']:.0f}ms"
            )
    except Exception as e:
        ds.error(f"Team execution failed: {e}")
        raise
