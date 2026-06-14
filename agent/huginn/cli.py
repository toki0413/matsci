"""CLI entry point for Huginn.

Inspired by EvoScientist's CLI and Claude Code's main.tsx.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from huginn import __version__
from huginn.agent import HuginnAgent
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry
from huginn.coder import CoderRunner
from huginn.permissions import PermissionConfig

console = Console()


def _resolve_abaqus_mcp_path(config_path: str | None = None) -> Path:
    """Resolve the Abaqus MCP server path.

    Priority:
    1. ABAQUS_MCP_SERVER_PATH environment variable
    2. abaqus_mcp_server from config
    3. Default: ~/.abaqus-mcp/mcp_server.py
    """
    env_path = os.environ.get("ABAQUS_MCP_SERVER_PATH")
    if env_path:
        return Path(env_path)
    if config_path:
        return Path(config_path)
    return Path.home() / ".abaqus-mcp" / "mcp_server.py"


_mcp_manager = None


async def _init_mcp(abaqus_mcp_server: str | None = None) -> None:
    """Initialize MCP servers and register their tools."""
    global _mcp_manager
    try:
        from huginn.mcp_client import MCPClientManager, MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_tools

        _mcp_manager = MCPClientManager()
        base = Path(__file__).parent.parent.parent  # repo root

        mat_db_path = base / "servers" / "mat-db-mcp" / "server.py"
        if mat_db_path.exists():
            await _mcp_manager.connect(MCPServerConfig(
                name="mat-db",
                command="python",
                args=[str(mat_db_path)],
            ))
            console.print(f"[green]✓[/green] MCP mat-db connected")

        math_path = base / "servers" / "math-anything-mcp" / "server.py"
        if math_path.exists():
            await _mcp_manager.connect(MCPServerConfig(
                name="math-anything",
                command="python",
                args=[str(math_path)],
            ))
            console.print(f"[green]✓[/green] MCP math-anything connected")

        # Abaqus MCP server (external project in home directory)
        abaqus_path = _resolve_abaqus_mcp_path(abaqus_mcp_server)
        if abaqus_path.exists():
            try:
                await _mcp_manager.connect(MCPServerConfig(
                    name="abaqus",
                    command="python",
                    args=[str(abaqus_path)],
                ))
                console.print(f"[green]✓[/green] MCP abaqus connected")
            except Exception as e:
                console.print(f"[yellow]MCP abaqus connection skipped: {e}[/yellow]")
        else:
            console.print("[dim]Abaqus MCP server not found; install from .abaqus-mcp to enable[/dim]")

        registered = register_mcp_tools(_mcp_manager)
        if registered:
            console.print(f"[green]✓[/green] Registered {len(registered)} MCP tools")
    except Exception as e:
        console.print(f"[yellow]MCP init warning: {e}[/yellow]")


async def _shutdown_mcp() -> None:
    """Shutdown MCP connections."""
    global _mcp_manager
    if _mcp_manager:
        await _mcp_manager.disconnect_all()
        _mcp_manager = None


@click.group()
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.option("--config", "-c", help="Config file path")
@click.option("--model", "-m", help="Model name (e.g., claude-sonnet-4-6, gpt-5.4)")
@click.option("--provider", "-p", help="Provider (anthropic, openai, ollama)")
@click.option("--dry-run", is_flag=True, help="Show what would be executed without running commands")
@click.option("--base-url", "-u", help="Base URL for OpenAI-compatible endpoints (vLLM, LM Studio, etc.)")
@click.option("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
@click.pass_context
def cli(ctx: click.Context, workspace: str, config: str | None,
        model: str | None, provider: str | None, base_url: str | None,
        dry_run: bool, ollama_url: str) -> None:
    """Huginn: Material Science specialized AI Agent Harness."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = Path(workspace).resolve()
    ctx.obj["config"] = config
    ctx.obj["model"] = model
    ctx.obj["provider"] = provider
    ctx.obj["base_url"] = base_url
    ctx.obj["dry_run"] = dry_run
    ctx.obj["ollama_url"] = ollama_url
    
    # Register all tools
    register_all_tools()


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start interactive chat with the Agent."""
    workspace = ctx.obj["workspace"]
    model_name = ctx.obj["model"]
    provider = ctx.obj["provider"]
    base_url = ctx.obj["base_url"]
    ollama_url = ctx.obj["ollama_url"]
    config_path = ctx.obj["config"]
    dry_run = ctx.obj["dry_run"]

    # Load config file if provided
    cfg = None
    if config_path:
        try:
            from huginn.config import HuginnConfig
            cfg = HuginnConfig.load(config_path)
            console.print(f"[green]✓[/green] Loaded config from {config_path}")
        except Exception as e:
            console.print(f"[yellow]Config load warning: {e}[/yellow]")

    console.print(Panel(
        f"[bold blue]Huginn[/bold blue] v{__version__}\n"
        f"Workspace: {workspace}\n"
        f"Type your materials science questions or 'exit' to quit.",
        title="Welcome",
        border_style="blue"
    ))
    
    # Initialize agent
    try:
        # Resolve settings: CLI flags > config file > env
        from huginn.config import HuginnConfig
        from huginn.security import SandboxConfig, SandboxExecutor, AuditLogger
        env_cfg = HuginnConfig.from_env()
        resolved_provider = provider or (cfg.provider if cfg else None) or env_cfg.provider
        resolved_model = model_name or (cfg.model if cfg else None) or env_cfg.model
        resolved_key = (cfg.api_key if cfg else None) or env_cfg.api_key
        resolved_base = base_url or (cfg.base_url if cfg else None) or env_cfg.base_url
        resolved_ollama = ollama_url or (cfg.ollama_host if cfg else None) or env_cfg.ollama_host

        # Resolve API key with env:/keyring: prefix support
        resolved_key = HuginnConfig.resolve_key(resolved_key)

        # Security layer
        sandbox_cfg = SandboxConfig(dry_run=dry_run)
        sandbox = SandboxExecutor(sandbox_cfg)
        audit = AuditLogger(workspace / "huginn_audit.jsonl")
        audit.log("config_load", "user", "chat_init", details={
            "provider": resolved_provider,
            "dry_run": dry_run,
        })

        if dry_run:
            console.print("[yellow]Dry-run mode: commands will be logged but not executed[/yellow]")

        if resolved_provider == "ollama":
            agent = HuginnAgent.from_ollama(
                model=resolved_model or "qwen2.5:14b",
                base_url=resolved_ollama,
                sandbox=sandbox,
                audit=audit,
            )
        elif resolved_provider and resolved_provider != "default":
            agent = HuginnAgent.from_provider(
                provider=resolved_provider,
                model=resolved_model,
                api_key=resolved_key,
                base_url=resolved_base,
                sandbox=sandbox,
                audit=audit,
            )
        else:
            console.print("[yellow]No provider configured.[/yellow]")
            console.print("[dim]Examples:[/dim]")
            console.print("  huginn chat --provider openai --model gpt-4o")
            console.print("  huginn chat --provider ollama --ollama-url http://localhost:11434")
            console.print("  huginn chat --provider vllm --base-url http://localhost:8000/v1 --model llama-3-8b")
            console.print("  huginn chat --config huginn.toml")
            agent = None
        
        if agent:
            agent.register_tools_from_registry()
            console.print(f"[green]✓[/green] Agent initialized with {len(agent.langchain_tools)} tools")
        
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize agent: {e}[/red]")
        console.print("[yellow]Falling back to mock mode (no LLM)[/yellow]")
        agent = None
    
    # Initialize MCP servers
    try:
        abaqus_mcp_path = cfg.abaqus_mcp_server if cfg else None
        asyncio.run(_init_mcp(abaqus_mcp_path))
        if agent:
            agent.register_tools_from_registry()
            console.print(f"[green]✓[/green] Total tools: {len(agent.langchain_tools)}")
    except Exception as e:
        console.print(f"[yellow]MCP init skipped: {e}[/yellow]")
    
    # Chat loop
    try:
        while True:
            try:
                user_input = console.input("[bold green]You:[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            
            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not user_input.strip():
                continue
            
            if agent is None:
                console.print("[red]Agent not available. Please check configuration.[/red]")
                continue
            
            # Run the agent
            console.print("[dim]Thinking...[/dim]")
            try:
                result = agent.invoke(user_input)
                
                # Extract the final AI message
                messages = result.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if hasattr(last_msg, "content"):
                        console.print(f"[bold blue]Agent:[/bold blue] {last_msg.content}")
                    else:
                        console.print(f"[bold blue]Agent:[/bold blue] {last_msg}")
                else:
                    console.print(f"[dim]{result}[/dim]")
                    
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
    finally:
        try:
            asyncio.run(_shutdown_mcp())
        except Exception:
            pass


@cli.command()
@click.argument("objective")
@click.option("--strategy", "-s", default="pareto", type=click.Choice(["pareto", "bayesian", "grid"]))
@click.option("--max-branches", "-b", default=10, help="Maximum parallel branches")
@click.pass_context
def explore(ctx: click.Context, objective: str, strategy: str, max_branches: int) -> None:
    """Enter exploration mode to systematically search a design space."""
    console.print(Panel(
        f"[bold green]Exploration Mode[/bold green]\n"
        f"Objective: {objective}\n"
        f"Strategy: {strategy}\n"
        f"Max branches: {max_branches}",
        title="Exploration",
        border_style="green"
    ))
    
    console.print("[yellow]Exploration engine initialization pending...[/yellow]")
    console.print("[dim]This will activate the multi-branch async exploration system.[/dim]")


@cli.command()
@click.argument("task", required=False)
@click.option("--auto-approve", is_flag=True, help="Auto-approve destructive coder actions")
@click.option("--max-iterations", "-i", type=int, default=None, help="Maximum coder iterations")
@click.pass_context
def coder(ctx: click.Context, task: str | None, auto_approve: bool, max_iterations: int | None) -> None:
    """Start an autonomous coding session (Codex-like)."""
    from huginn.config import HuginnConfig, get_settings

    settings = get_settings()
    config_path = ctx.obj["config"]
    if config_path:
        try:
            file_cfg = HuginnConfig.load(config_path)
            settings.config = file_cfg
            console.print(f"[green]✓[/green] Loaded config from {config_path}")
        except Exception as e:
            console.print(f"[yellow]Config load warning: {e}[/yellow]")

    # CLI flags override config/env values
    if ctx.obj["provider"]:
        settings.config.provider = ctx.obj["provider"]
    if ctx.obj["model"]:
        settings.config.model = ctx.obj["model"]
    if ctx.obj["base_url"]:
        settings.config.base_url = ctx.obj["base_url"]
    if ctx.obj["ollama_url"]:
        settings.config.ollama_host = ctx.obj["ollama_url"]
    if max_iterations is not None:
        settings.coder.max_iterations = max_iterations

    permission_config = PermissionConfig(auto_approve_all=auto_approve)
    if auto_approve:
        console.print("[yellow]Auto-approve enabled for this coder session[/yellow]")

    def approval_callback(tool_name: str, reason: str) -> bool:
        console.print(f"[yellow]{reason}[/yellow]")
        answer = console.input(f"Approve [bold]{tool_name}[/bold]? [y/N]: ").strip().lower()
        return answer in ("y", "yes")

    if not task:
        try:
            task = console.input("[bold green]Coder task:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Cancelled.[/yellow]")
            return

    if not task:
        console.print("[yellow]No task provided.[/yellow]")
        return

    console.print(Panel(
        f"[bold blue]Coder Mode[/bold blue]\n"
        f"Provider: {settings.config.provider}\n"
        f"Model: {settings.config.model or 'auto'}\n"
        f"Max iterations: {settings.coder.max_iterations}",
        title="Autonomous Coding",
        border_style="blue"
    ))

    try:
        runner = CoderRunner(
            settings=settings,
            permission_config=permission_config,
            approval_callback=None if auto_approve else approval_callback,
        )
        result = runner.run(task)
        final = result.get("final_answer", "")
        console.print(Markdown(final))
    except Exception as e:
        console.print(f"[red]Coder failed: {e}[/red]")


@cli.command()
@click.option("--port", "-p", default=8000, help="Server port")
@click.option("--host", "-h", default="127.0.0.1", help="Server host")
@click.pass_context
def serve(ctx: click.Context, port: int, host: str) -> None:
    """Start the HTTP/WebSocket server for the desktop app."""
    register_all_tools()
    
    console.print(Panel(
        f"[bold blue]Huginn Server[/bold blue]\n"
        f"URL: http://{host}:{port}\n"
        f"WebSocket: ws://{host}:{port}/ws/agent\n"
        f"Tools: {len(ToolRegistry.list_tools())}",
        title="Server",
        border_style="blue"
    ))
    
    try:
        import uvicorn
        from huginn.server import app
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn fastapi[/red]")


@cli.command()
def tools() -> None:
    """List all available tools."""
    register_all_tools()
    
    console.print(Panel(
        f"[bold blue]Available Tools ({len(ToolRegistry.list_tools())})[/bold blue]",
        border_style="blue"
    ))
    
    for name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(name)
        if tool:
            try:
                is_ro = hasattr(tool, 'is_read_only') and tool.is_read_only(None)
            except Exception:
                is_ro = False
            read_only = "[green]read-only[/green]" if is_ro else ""
            console.print(f"  [bold]{name}[/bold] — {tool.description[:60]}... {read_only}")


@cli.command()
def version() -> None:
    """Show version information."""
    console.print(f"Huginn [bold]{__version__}[/bold]")
    
    try:
        import langchain
        console.print(f"  langchain: {langchain.__version__}")
    except Exception:
        pass
    
    try:
        import langgraph
        console.print(f"  langgraph: {langgraph.__version__}")
    except Exception:
        pass
    
    try:
        import pydantic
        console.print(f"  pydantic: {pydantic.__version__}")
    except Exception:
        pass



@cli.command()
@click.option("--path", "-p", default="huginn.toml", help="Config file path to write")
@click.pass_context
def configure(ctx: click.Context, path: str) -> None:
    """Interactive first-run configuration wizard."""
    console.print(Panel(
        "[bold blue]Huginn Configuration Wizard[/bold blue]",
        title="Setup",
        border_style="blue"
    ))
    
    from huginn.config import HuginnConfig
    
    # Try to load existing config
    try:
        cfg = HuginnConfig.load(path)
        console.print(f"[dim]Loaded existing config from {path}[/dim]")
    except Exception:
        cfg = HuginnConfig()
    
    provider = console.input(
        f"Provider [cyan]{cfg.provider}[/cyan]: "
    ).strip() or cfg.provider
    
    model = console.input(
        f"Model [cyan]{cfg.model or 'auto'}[/cyan]: "
    ).strip() or cfg.model
    
    api_key = console.input(
        f"API key [cyan]{'***' if cfg.api_key else 'none'}[/cyan]: "
    ).strip() or cfg.api_key
    
    base_url = console.input(
        f"Base URL [cyan]{cfg.base_url or 'none'}[/cyan]: "
    ).strip() or cfg.base_url
    
    ollama_host = console.input(
        f"Ollama host [cyan]{cfg.ollama_host}[/cyan]: "
    ).strip() or cfg.ollama_host
    
    workspace = console.input(
        f"Workspace [cyan]{cfg.workspace}[/cyan]: "
    ).strip() or cfg.workspace
    
    new_cfg = HuginnConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        ollama_host=ollama_host,
        workspace=workspace,
    )
    
    fmt = "toml" if path.endswith(".toml") else "json"
    new_cfg.save(path, format=fmt)
    console.print(f"[green]✓[/green] Config saved to [bold]{path}[/bold]")
    console.print("[dim]Run: huginn chat --config " + path + "[/dim]")

def main() -> None:
    """Entry point."""
    env_path = Path(".env")
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass
    
    cli()


if __name__ == "__main__":
    main()
