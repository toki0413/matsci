"""Interactive chat command."""

from __future__ import annotations

import asyncio
import contextlib

import click
from rich.panel import Panel

from huginn import __version__
from huginn.cli.context import CliContext, build_agent_from_ctx, init_mcp, shutdown_mcp
from huginn.pet import get_pet_avatar


@click.command()
@click.pass_obj
def chat(ctx: CliContext) -> None:
    """Start interactive chat with the Agent."""
    avatar = get_pet_avatar()
    ctx.console.print(
        Panel(
            f"[dim]{avatar}[/dim]\n"
            f"[bold blue]Huginn[/bold blue] v{__version__}\n"
            f"Workspace: {ctx.workspace}\n"
            f"Type your materials science questions or 'exit' to quit.",
            title="Welcome",
            border_style="blue",
        )
    )

    agent = build_agent_from_ctx(ctx)
    if agent is None:
        ctx.console.print("[yellow]No provider configured.[/yellow]")
        ctx.console.print("[dim]Examples:[/dim]")
        ctx.console.print("  huginn chat --provider openai --model gpt-4o")
        ctx.console.print(
            "  huginn chat --provider ollama --ollama-url http://localhost:11434"
        )
        ctx.console.print(
            "  huginn chat --provider vllm --base-url http://localhost:8000/v1 --model llama-3-8b"
        )
        ctx.console.print("  huginn chat --provider moonshot --model moonshot-v1-8k")
        ctx.console.print("  huginn chat --provider dashscope --model qwen-max")
        ctx.console.print("  huginn chat --config huginn.toml")
    else:
        agent.register_tools_from_registry()
        ctx.console.print(
            f"[green]✓[/green] Agent initialized with {len(agent.langchain_tools)} tools"
        )

    # Initialize MCP servers
    try:
        cfg = ctx.load_config()
        asyncio.run(init_mcp(cfg.abaqus_mcp_server))
        if agent:
            agent.register_tools_from_registry()
            ctx.console.print(
                f"[green]✓[/green] Total tools: {len(agent.langchain_tools)}"
            )
    except Exception as e:
        ctx.console.print(f"[yellow]MCP init skipped: {e}[/yellow]")

    # Chat loop
    try:
        while True:
            try:
                user_input = ctx.console.input("[bold green]You:[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                ctx.console.print("\n[yellow]Goodbye![/yellow]")
                break

            if user_input.lower() in ("exit", "quit", "q"):
                ctx.console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

            if agent is None:
                ctx.console.print(
                    "[red]Agent not available. Please check configuration.[/red]"
                )
                continue

            # Run the agent
            ctx.console.print("[dim]Thinking...[/dim]")
            try:
                result = agent.invoke(user_input)

                # Extract the final AI message
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        ctx.console.print(
                            f"[bold blue]Agent:[/bold blue] {last_msg.content}"
                        )
                    else:
                        ctx.console.print(f"[bold blue]Agent:[/bold blue] {last_msg}")
                else:
                    ctx.console.print(f"[dim]{result}[/dim]")

            except Exception as e:
                ctx.console.print(f"[red]Error: {e}[/red]")
    finally:
        with contextlib.suppress(Exception):
            asyncio.run(shutdown_mcp())
