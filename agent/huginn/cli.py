"""CLI entry point for Huginn.

Inspired by EvoScientist's CLI and Claude Code's main.tsx.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from huginn import __version__
from huginn.agent import HuginnAgent
from huginn.checkpointer import create_in_memory_checkpointer
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
@click.option("--thinking", type=click.Choice(["low", "medium", "high"]), help="Reasoning intensity (Anthropic extended thinking / OpenAI reasoning effort)")
@click.pass_context
def cli(ctx: click.Context, workspace: str, config: str | None,
        model: str | None, provider: str | None, base_url: str | None,
        dry_run: bool, ollama_url: str, thinking: str | None) -> None:
    """Huginn: Material Science specialized AI Agent Harness."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = Path(workspace).resolve()
    ctx.obj["config"] = config
    ctx.obj["model"] = model
    ctx.obj["provider"] = provider
    ctx.obj["base_url"] = base_url
    ctx.obj["dry_run"] = dry_run
    ctx.obj["ollama_url"] = ollama_url
    ctx.obj["thinking"] = thinking

    # Register all tools
    register_all_tools()


def _load_config(ctx: click.Context) -> Any:
    """Load HuginnConfig from --config or environment."""
    from huginn.config import HuginnConfig
    config_path = ctx.obj.get("config")
    if config_path:
        try:
            return HuginnConfig.load(config_path)
        except Exception as e:
            console.print(f"[yellow]Config load warning: {e}[/yellow]")
    return HuginnConfig.from_env()


def _apply_cli_overrides(ctx: click.Context, cfg: Any) -> None:
    """Let CLI flags override config/env values."""
    if ctx.obj.get("provider"):
        cfg.provider = ctx.obj["provider"]
    if ctx.obj.get("model"):
        cfg.model = ctx.obj["model"]
    if ctx.obj.get("base_url"):
        cfg.base_url = ctx.obj["base_url"]
    if ctx.obj.get("ollama_url"):
        cfg.ollama_host = ctx.obj["ollama_url"]
    if ctx.obj.get("thinking"):
        cfg.thinking = ctx.obj["thinking"]


def _build_agent_from_ctx(ctx: click.Context, profile_id: str = "lead") -> HuginnAgent | None:
    """Build a HuginnAgent from the resolved configuration."""
    from huginn.security import SandboxConfig, SandboxExecutor, AuditLogger

    cfg = _load_config(ctx)
    _apply_cli_overrides(ctx, cfg)

    if cfg.provider == "default" and not cfg.models:
        console.print("[yellow]No provider or model pool configured.[/yellow]")
        console.print("[dim]Run `huginn configure` or set HUGINN_PROVIDER / HUGINN_MODELS.[/dim]")
        return None

    sandbox_cfg = SandboxConfig(dry_run=ctx.obj.get("dry_run", False))
    sandbox = SandboxExecutor(sandbox_cfg)
    audit = AuditLogger(ctx.obj["workspace"] / "huginn_audit.jsonl")

    try:
        return HuginnAgent.from_config(
            cfg,
            profile_id=profile_id,
            sandbox=sandbox,
            audit=audit,
        )
    except Exception as e:
        console.print(f"[red]Failed to build agent: {e}[/red]")
        return None


@cli.command()
@click.pass_context
def chat(ctx: click.Context) -> None:
    """Start interactive chat with the Agent."""
    workspace = ctx.obj["workspace"]

    console.print(Panel(
        f"[bold blue]Huginn[/bold blue] v{__version__}\n"
        f"Workspace: {workspace}\n"
        f"Type your materials science questions or 'exit' to quit.",
        title="Welcome",
        border_style="blue"
    ))

    agent = _build_agent_from_ctx(ctx)
    if agent is None:
        console.print("[yellow]No provider configured.[/yellow]")
        console.print("[dim]Examples:[/dim]")
        console.print("  huginn chat --provider openai --model gpt-4o")
        console.print("  huginn chat --provider ollama --ollama-url http://localhost:11434")
        console.print("  huginn chat --provider vllm --base-url http://localhost:8000/v1 --model llama-3-8b")
        console.print("  huginn chat --config huginn.toml")
    else:
        agent.register_tools_from_registry()
        console.print(f"[green]✓[/green] Agent initialized with {len(agent.langchain_tools)} tools")

    # Initialize MCP servers
    try:
        cfg = _load_config(ctx)
        asyncio.run(_init_mcp(cfg.abaqus_mcp_server))
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
@click.option("--max-iterations", "-i", default=20, help="Maximum exploration iterations")
@click.pass_context
def explore(ctx: click.Context, objective: str, strategy: str, max_branches: int, max_iterations: int) -> None:
    """Enter exploration mode to systematically search a design space."""
    from huginn.exploration.orchestrator import ExplorationOrchestrator
    from huginn.exploration.strategies import ParetoPruningStrategy

    console.print(Panel(
        f"[bold green]Exploration Mode[/bold green]\n"
        f"Objective: {objective}\n"
        f"Strategy: {strategy}\n"
        f"Max branches: {max_branches}\n"
        f"Max iterations: {max_iterations}",
        title="Exploration",
        border_style="green"
    ))

    try:
        orch = ExplorationOrchestrator(
            strategy=ParetoPruningStrategy(max_active=max_branches),
            max_parallel=min(5, max_branches),
        )
        result = asyncio.run(orch.explore(
            objective=objective,
            initial_branches=[{"name": "baseline", "hypothesis": f"Baseline for: {objective}"}],
            objectives_config={"score": "maximize"},
            max_iterations=max_iterations,
        ))
        console.print(f"[green]✓[/green] Exploration complete: {result.convergence_reason}")
        console.print(f"  Branches explored: {result.n_branches_explored}")
        console.print(f"  Branches pruned: {result.n_branches_pruned}")
        console.print(f"  Pareto front size: {len(result.pareto_front)}")
        if result.best_branch:
            console.print(f"  Best branch: {result.best_branch['name']}")
    except Exception as e:
        console.print(f"[red]Exploration failed: {e}[/red]")


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
    if ctx.obj.get("thinking"):
        settings.config.thinking = ctx.obj["thinking"]
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

    thinking_raw = console.input(
        f"Thinking intensity [cyan]{cfg.thinking or 'none'}[/cyan] (low/medium/high or none): "
    ).strip() or (cfg.thinking if cfg.thinking else "")
    thinking: str | None = thinking_raw if thinking_raw in ("low", "medium", "high") else None

    new_cfg = HuginnConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        ollama_host=ollama_host,
        workspace=workspace,
        thinking=thinking,  # type: ignore[arg-type]
    )
    
    fmt = "toml" if path.endswith(".toml") else "json"
    new_cfg.save(path, format=fmt)
    console.print(f"[green]✓[/green] Config saved to [bold]{path}[/bold]")
    console.print("[dim]Run: huginn chat --config " + path + "[/dim]")


@cli.command()
@click.option("--evolve", is_flag=True, help="Run evolution cycle after benchmarking")
@click.option("--categories", "-c", help="Comma-separated categories to run")
@click.option("--output", "-o", default="bench_report.json", help="Report output path")
@click.pass_context
def bench(ctx: click.Context, evolve: bool, categories: str | None, output: str) -> None:
    """Run the benchmark suite and optionally trigger self-evolution."""
    from huginn.bench.runner import BenchmarkRunner
    from huginn.config import HuginnConfig

    config_path = ctx.obj["config"]
    cfg = HuginnConfig.load(config_path) if config_path else HuginnConfig.from_env()
    runner = BenchmarkRunner(config=cfg)
    cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else None

    report = runner.run(evolve=evolve, categories=cats)
    report_path = ctx.obj["workspace"] / output
    runner.save_report(report, report_path)

    console.print(Panel(
        f"[bold blue]Benchmark Report[/bold blue]\n"
        f"Run ID: {report.run_id}\n"
        f"Total: {report.total}  Passed: {report.passed}  Failed: {report.failed}  Skipped: {report.skipped}\n"
        f"Pass rate: {report.metrics.get('pass_rate', 0):.0%}\n"
        f"Avg task time: {report.metrics.get('avg_task_time_seconds', 0):.2f}s\n"
        f"Report saved to: {report_path}",
        title="Bench",
        border_style="blue"
    ))

    for r in report.results:
        icon = "[green]✓[/green]" if r.passed else "[red]✗[/red]"
        console.print(f"{icon} {r.task_id}: {r.reason}")


@cli.command()
@click.option("--logs-dir", help="Execution log directory to evolve from")
@click.pass_context
def evolve(ctx: click.Context, logs_dir: str | None) -> None:
    """Run a self-evolution cycle from execution logs."""
    from huginn.evolution.engine import EvolutionEngine
    from huginn.evolution.logger import ExecutionLogger

    logger = ExecutionLogger(persist_dir=logs_dir) if logs_dir else ExecutionLogger()
    engine = EvolutionEngine(logger=logger)
    report = engine.run_full_evolution_cycle()

    console.print(Panel(
        f"[bold blue]Evolution Report[/bold blue]\n"
        f"Failure rules learned: {len(report['failure_rules'])}\n"
        f"Success skills extracted: {len(report['success_skills'])}\n"
        f"Prompt patches: {len(report['prompt_patches'])}\n"
        f"Total rules/skills: {report['total_rules_after']}/{report['total_skills_after']}",
        title="Evolve",
        border_style="blue"
    ))


@cli.command()
@click.argument("stages")
@click.option("--working-dir", "-w", default=".", help="Working directory")
@click.option("--name", "-n", default="execute", help="Workflow name")
@click.pass_context
def execute(ctx: click.Context, stages: str, working_dir: str, name: str) -> None:
    """Run a list of workflow stages via the execution orchestrator.

    STAGES can be a JSON file path or an inline JSON array of stage dicts.
    """
    from huginn.execution.orchestrator import ExecutionOrchestrator
    from huginn.security.audit import AuditLogger
    from huginn.tools.registry import ToolRegistry
    from huginn.types import ToolContext

    stage_path = Path(stages)
    raw = stage_path.read_text(encoding="utf-8") if stage_path.exists() else stages
    stage_list = json.loads(raw)

    def _wrap_tool(tool):
        async def _run(action: str = "", **params):
            if tool.input_schema:
                input_data = tool.input_schema(action=action, **params)
            else:
                input_data = {"action": action, **params}
            context = ToolContext(
                session_id="execute",
                workspace=working_dir,
                audit_logger=AuditLogger(Path(working_dir) / "huginn_audit.jsonl"),
            )
            result = await tool.call(input_data, context)
            if not result.success:
                raise RuntimeError(result.error or f"{tool.name} failed")
            return result.data
        return _run

    orch = ExecutionOrchestrator(working_dir=working_dir)
    for tool_name in ToolRegistry.list_tools():
        tool = ToolRegistry.get(tool_name)
        if tool:
            orch.register_tool(tool_name, _wrap_tool(tool))

    record = asyncio.run(orch.run(stage_list, workflow_name=name))
    console.print(json.dumps({
        "workflow_name": record.workflow_name,
        "overall_success": record.overall_success,
        "stages": [r.to_dict() for r in record.stage_results],
    }, indent=2, ensure_ascii=False, default=str))


@cli.command("workflow")
@click.argument("template")
@click.argument("args", nargs=-1)
@click.pass_context
def workflow(ctx: click.Context, template: str, args: tuple[str, ...]) -> None:
    """Run a workflow template with KEY=VALUE arguments."""
    from huginn.types import ToolContext
    from huginn.workflows.engine import WorkflowEngine
    from huginn.workflows.templates import get_template

    template_fn = get_template(template)
    if not template_fn:
        console.print(f"[red]Template '{template}' not found[/red]")
        return

    kwargs: dict[str, Any] = {}
    for a in args:
        if "=" not in a:
            console.print(f"[yellow]Ignoring malformed arg: {a}[/yellow]")
            continue
        key, value = a.split("=", 1)
        try:
            value = json.loads(value)
        except Exception:
            pass
        kwargs[key] = value

    try:
        stages = template_fn(**kwargs)
    except Exception as e:
        console.print(f"[red]Failed to build workflow: {e}[/red]")
        return

    engine = WorkflowEngine(ToolRegistry)
    context = ToolContext(session_id="workflow", workspace=str(ctx.obj["workspace"]))
    result = asyncio.run(engine.execute(stages, context))

    console.print(json.dumps({
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
    }, indent=2, ensure_ascii=False, default=str))


@cli.command()
@click.argument("error_message")
@click.option("--software", "-s", help="Software (e.g., VASP, LAMMPS, Gaussian)")
@click.option("--calculation-type", "-t", help="Calculation type (e.g., DFT, MD)")
@click.option("--context", "-c", help="Additional context")
@click.pass_context
def diagnose(ctx: click.Context, error_message: str, software: str | None, calculation_type: str | None, context: str | None) -> None:
    """Diagnose a computational chemistry/MD error."""
    from huginn.tools.diagnose_tool import DiagnoseInput, DiagnoseTool
    from huginn.types import ToolContext

    tool = DiagnoseTool()
    input_data = DiagnoseInput(
        error_message=error_message,
        software=software,
        calculation_type=calculation_type,
        context=context,
    )
    result = asyncio.run(tool.call(input_data, ToolContext(
        session_id="diagnose",
        workspace=str(ctx.obj["workspace"]),
    )))
    console.print(json.dumps(result.data, indent=2, ensure_ascii=False, default=str))


@cli.group(name="hpc")
@click.pass_context
def hpc(ctx: click.Context) -> None:
    """HPC cluster job submission commands."""


@hpc.command("test")
@click.option("--host", required=True, help="HPC host")
@click.option("--username", "-u", required=True, help="SSH username")
@click.option("--scheduler", default="slurm", type=click.Choice(["slurm", "pbs"]))
@click.option("--key-path", help="SSH private key path")
@click.option("--port", default=22, type=int)
@click.pass_context
def hpc_test(ctx: click.Context, host: str, username: str, scheduler: str, key_path: str | None, port: int) -> None:
    """Test SSH connection to an HPC cluster."""
    from huginn.hpc.client import HPCClient, HPCConfig

    cfg = HPCConfig(host=host, username=username, scheduler=scheduler, key_path=key_path, port=port)
    try:
        with HPCClient(cfg) as client:
            stdout, stderr, rc = client._exec("hostname")
            if rc == 0:
                console.print(f"[green]✓[/green] Connected to {host}: {stdout}")
            else:
                console.print(f"[red]✗[/red] {stderr or 'Connection failed'}")
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")


@hpc.command("submit")
@click.option("--host", required=True)
@click.option("--username", "-u", required=True)
@click.option("--command", required=True, help="Command to run on the cluster")
@click.option("--job-name", default="huginn_job")
@click.option("--walltime", default="01:00:00")
@click.option("--nodes", default=1, type=int)
@click.option("--ntasks-per-node", default=4, type=int)
@click.option("--queue", help="Queue/partition")
@click.option("--scheduler", default="slurm", type=click.Choice(["slurm", "pbs"]))
@click.option("--key-path")
@click.option("--remote-work-dir", default="~/huginn_jobs")
@click.pass_context
def hpc_submit(
    ctx: click.Context,
    host: str,
    username: str,
    command: str,
    job_name: str,
    walltime: str,
    nodes: int,
    ntasks_per_node: int,
    queue: str | None,
    scheduler: str,
    key_path: str | None,
    remote_work_dir: str,
) -> None:
    """Submit a job to a remote HPC cluster."""
    from huginn.hpc.client import HPCClient, HPCConfig

    cfg = HPCConfig(
        host=host,
        username=username,
        scheduler=scheduler,
        key_path=key_path,
        remote_work_dir=remote_work_dir,
    )
    try:
        with HPCClient(cfg) as client:
            script = client.generate_job_script(
                command=command,
                job_name=job_name,
                walltime=walltime,
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
                queue=queue,
            )
            job_id = client.submit_job(script, job_name=job_name)
            console.print(f"[green]✓[/green] Submitted {job_name}: {job_id}")
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")


@hpc.command("status")
@click.option("--host", required=True)
@click.option("--username", "-u", required=True)
@click.option("--job-id", required=True)
@click.option("--scheduler", default="slurm", type=click.Choice(["slurm", "pbs"]))
@click.option("--key-path")
@click.pass_context
def hpc_status(
    ctx: click.Context,
    host: str,
    username: str,
    job_id: str,
    scheduler: str,
    key_path: str | None,
) -> None:
    """Poll status of a remote HPC job."""
    from huginn.hpc.client import HPCClient, HPCConfig

    cfg = HPCConfig(host=host, username=username, scheduler=scheduler, key_path=key_path)
    try:
        with HPCClient(cfg) as client:
            status = client.poll_status(job_id)
            console.print(json.dumps({
                "job_id": status.job_id,
                "state": status.state,
                "exit_code": status.exit_code,
                "runtime": status.runtime,
                "message": status.message,
            }, indent=2, default=str))
    except Exception as e:
        console.print(f"[red]✗[/red] {e}")


@cli.command("encrypt-config")
@click.argument("path", default="huginn.toml")
@click.option("--password", prompt=True, hide_input=True, help="Encryption password")
@click.pass_context
def encrypt_config(ctx: click.Context, path: str, password: str) -> None:
    """Encrypt a configuration file."""
    from huginn.config import HuginnConfig

    target = Path(path)
    cfg = HuginnConfig.load(path) if target.exists() else HuginnConfig.from_env()
    cfg.encrypt_config = True
    cfg.encryption_password = password
    out = target if str(target).endswith(".enc") else target.with_suffix(target.suffix + ".enc")
    cfg.save(out, format="json")
    console.print(f"[green]✓[/green] Encrypted config saved to {out}")


@cli.group(name="unified")
@click.pass_context
def unified(ctx: click.Context) -> None:
    """Unified scientific computing framework."""


@unified.command("list")
def unified_list() -> None:
    """List available unified models and bridges."""
    from huginn.unified.models import list_models
    from huginn.unified.bridge import list_bridges

    console.print(Panel("[bold blue]Unified Models[/bold blue]", border_style="blue"))
    for name in list_models():
        console.print(f"  - {name}")
    console.print(Panel("[bold blue]Multiscale Bridges[/bold blue]", border_style="blue"))
    for name in list_bridges():
        console.print(f"  - {name}")


@unified.command("derive")
@click.argument("model")
@click.pass_context
def unified_derive(ctx: click.Context, model: str) -> None:
    """Derive governing equations for a unified model."""
    from huginn.unified.models import get_model
    from huginn.unified import derive_equations

    factory = get_model(model)
    if not factory:
        console.print(f"[red]Model '{model}' not found[/red]")
        return
    problem = factory()
    result = derive_equations(problem)
    console.print(Panel(
        f"[bold blue]{problem.name}[/bold blue]\n"
        f"Principle: {result['principle']}\n"
        f"Equations:",
        title="Unified Derivation",
        border_style="blue"
    ))
    for key, eq in result["equations"].items():
        console.print(f"  [bold]{key}:[/bold] {eq}")


@unified.command("bridge")
@click.argument("name")
@click.option("--model", help="Model name required by some bridges (e.g. dft_to_md)")
@click.option("--expression", help="Potential expression for cauchy_born / md_to_elasticity")
@click.option("--symbols", help="Comma-separated symbol list for --expression")
@click.pass_context
def unified_bridge(
    ctx: click.Context,
    name: str,
    model: str | None,
    expression: str | None,
    symbols: str | None,
) -> None:
    """Compute a multiscale bridge relation."""
    import sympy as sp

    from huginn.unified.bridge import ConstitutiveModel, get_bridge
    from huginn.unified.models import get_model

    bridge_name = name.lower().replace("-", "_")
    bridge_fn = get_bridge(bridge_name)
    if not bridge_fn:
        console.print(f"[red]Bridge '{name}' not found[/red]")
        return

    kwargs: dict[str, Any] = {}
    if bridge_name == "dft_to_md":
        if model:
            factory = get_model(model)
            if not factory:
                console.print(f"[red]Model '{model}' not found[/red]")
                return
            kwargs["dft_problem"] = factory()
        else:
            from huginn.unified.models import one_d_kohn_sham_dft
            kwargs["dft_problem"] = one_d_kohn_sham_dft()
    elif bridge_name in ("cauchy_born", "md_to_elasticity"):
        if not expression or not symbols:
            console.print("[red]--expression and --symbols required for this bridge[/red]")
            return
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        sym_dict = {s: sp.Symbol(s) for s in sym_list}
        expr = sp.sympify(expression, locals=sym_dict)
        kwargs["potential"] = ConstitutiveModel(
            name="user_potential",
            expression=expr,
            parameters={s: str(sym_dict[s]) for s in sym_list},
        )

    result = bridge_fn(**kwargs)
    console.print(Panel(
        f"[bold blue]{name}[/bold blue]\n{result.get('interpretation', '')}",
        title="Multiscale Bridge",
        border_style="blue"
    ))
    for key, val in result.items():
        if key == "interpretation":
            continue
        console.print(f"  [bold]{key}:[/bold] {val}")


@unified.command("solve")
@click.argument("model")
@click.option("--method", default="fem", help="fem | fd")
@click.option("--n", default=10, help="Number of elements/points")
@click.option("--plot", "plot_path", default=None, help="Path to save solution plot")
@click.pass_context
def unified_solve(ctx: click.Context, model: str, method: str, n: int, plot_path: str | None) -> None:
    """Discretize and solve a unified model."""
    from huginn.unified import solve, solve_and_plot
    from huginn.unified.models import get_model

    factory = get_model(model)
    if not factory:
        console.print(f"[red]Model '{model}' not found[/red]")
        return
    problem = factory()
    try:
        if plot_path:
            result = solve_and_plot(problem, method=method, n=n, output_path=plot_path)
        else:
            result = solve(problem, method=method, n=n)
    except Exception as e:
        console.print(f"[red]Solve failed: {e}[/red]")
        return

    info = f"Method: {result['method']}, DOFs: {result['n_dof']}, Residual: {result['residual']:.3e}"
    console.print(Panel(f"[bold blue]{model}[/bold blue]", title="Unified Solve", subtitle=info, border_style="blue"))
    console.print(f"[bold]Mesh:[/bold] {result['mesh']}")
    console.print(f"[bold]Solution:[/bold] {result['solution']}")
    if plot_path:
        console.print(f"[green]Plot saved to {result['plot_path']}[/green]")


@unified.command("discretize")
@click.argument("model")
@click.option("--method", default="fem", help="fem | fd")
@click.option("--n", default=10, help="Number of elements/points")
@click.pass_context
def unified_discretize(ctx: click.Context, model: str, method: str, n: int) -> None:
    """Discretize a unified model into a linear algebraic system."""
    from huginn.unified import discretize
    from huginn.unified.models import get_model

    factory = get_model(model)
    if not factory:
        console.print(f"[red]Model '{model}' not found[/red]")
        return
    problem = factory()
    try:
        result = discretize(problem, method=method, n=n)
    except Exception as e:
        console.print(f"[red]Discretization failed: {e}[/red]")
        return

    info = f"Method: {result['method']}, DOFs: {result['n_dof']}"
    console.print(Panel(f"[bold blue]{model}[/bold blue]", title="Discretization", subtitle=info, border_style="blue"))
    console.print("[bold]Stiffness matrix:[/bold]")
    for row in result["stiffness_matrix"]:
        console.print(f"  {row}")
    console.print(f"[bold]Load vector:[/bold] {result['load_vector']}")


@cli.group(name="persona")
@click.pass_context
def persona(ctx: click.Context) -> None:
    """Manage Huginn personas."""


@persona.command("list")
def persona_list() -> None:
    """List available personas."""
    from huginn.personas import PersonaManager
    mgr = PersonaManager()
    console.print(Panel("[bold blue]Personas[/bold blue]", border_style="blue"))
    for name in mgr.list():
        marker = " (default)" if name == mgr.get_default_name() else ""
        console.print(f"  - {name}{marker}")


@persona.command("show")
@click.argument("name")
def persona_show(name: str) -> None:
    """Show a persona details."""
    from huginn.personas import PersonaManager
    mgr = PersonaManager()
    p = mgr.get(name)
    console.print(Panel(f"[bold blue]{p.name}[/bold blue]", title="Persona", border_style="blue"))
    console.print(p.system_prompt)
    if p.begin_dialogs:
        console.print("[bold]Begin dialogs:[/bold]")
        for d in p.begin_dialogs:
            console.print(f"  {d['role']}: {d['content']}")


@persona.command("set-default")
@click.argument("name")
def persona_set_default(name: str) -> None:
    """Set the default persona."""
    from huginn.personas import PersonaManager
    mgr = PersonaManager()
    try:
        mgr.set_default(name)
        console.print(f"[green]Default persona set to {name}[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")


@persona.command("create")
@click.argument("name")
@click.option("--prompt", required=True, help="System prompt text")
@click.option("--begin-dialog", "begin_dialogs", multiple=True, help="Begin dialog as role:content")
def persona_create(name: str, prompt: str, begin_dialogs: tuple[str, ...]) -> None:
    """Create a new persona."""
    from huginn.personas import PersonaManager
    mgr = PersonaManager()
    parsed = []
    for d in begin_dialogs:
        if ":" not in d:
            console.print(f"[red]Invalid begin dialog (use role:content): {d}[/red]")
            return
        role, content = d.split(":", 1)
        parsed.append({"role": role.strip(), "content": content.strip()})
    try:
        mgr.create(name, system_prompt=prompt, begin_dialogs=parsed)
        console.print(f"[green]Created persona {name}[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")


@persona.command("delete")
@click.argument("name")
def persona_delete(name: str) -> None:
    """Delete a user-defined persona."""
    from huginn.personas import PersonaManager
    mgr = PersonaManager()
    try:
        mgr.delete(name)
        console.print(f"[green]Deleted persona {name}[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")


@cli.command("model-list")
@click.pass_context
def model_list(ctx: click.Context) -> None:
    """List configured models and agent profiles."""
    cfg = _load_config(ctx)
    console.print(Panel("[bold blue]Configured Models[/bold blue]", border_style="blue"))
    if cfg.models:
        for m in cfg.models:
            status = "[green]enabled[/green]" if m.enabled else "[red]disabled[/red]"
            console.print(f"  [bold]{m.alias}[/bold] {m.provider}:{m.model or 'auto'} ({status})")
    else:
        console.print(f"  [dim]No model pool. Legacy provider: {cfg.provider} / {cfg.model or 'auto'}[/dim]")

    console.print(Panel("[bold blue]Agent Profiles[/bold blue]", border_style="blue"))
    if cfg.agents:
        for a in cfg.agents:
            status = "[green]enabled[/green]" if a.enabled else "[red]disabled[/red]"
            console.print(f"  [bold]{a.id}[/bold] -> {a.model_alias} persona={a.persona} tools={len(a.tools)} ({status})")
    else:
        console.print("  [dim]No agent profiles configured.[/dim]")


@cli.command("memory-maintenance")
@click.option("--prune-threshold", type=float, default=None, help="Importance threshold for pruning")
@click.pass_context
def memory_maintenance(ctx: click.Context, prune_threshold: float | None) -> None:
    """Run long-term memory decay, prune, and deduplication."""
    cfg = _load_config(ctx)
    threshold = prune_threshold if prune_threshold is not None else cfg.memory_decay_prune_threshold
    agent = HuginnAgent(model=None, tools=[], checkpointer=create_in_memory_checkpointer())
    try:
        summary = agent.memory.maintenance(prune_threshold=threshold)
        console.print(Panel(
            f"[bold blue]Memory Maintenance[/bold blue]\n"
            f"Decayed: {summary.get('decayed', 0)}\n"
            f"Pruned: {summary.get('pruned', 0)}\n"
            f"Expired: {summary.get('expired', 0)}\n"
            f"Deduplicated: {summary.get('deduplicated', 0)}",
            border_style="blue",
        ))
    finally:
        agent.close()


@cli.command("telemetry")
@click.pass_context
def telemetry(ctx: click.Context) -> None:
    """Show telemetry summary for the default agent profile."""
    cfg = _load_config(ctx)
    _apply_cli_overrides(ctx, cfg)
    agent = HuginnAgent.from_config(cfg)
    try:
        summary = agent.telemetry_summary()
        console.print(Panel(
            f"[bold blue]Telemetry Summary[/bold blue]\n"
            f"Total spans: {summary.get('total_spans', 0)}",
            border_style="blue",
        ))
        for name, info in summary.get("by_name", {}).items():
            console.print(f"  {name}: count={info['count']} duration_ms={info['duration_ms']:.1f}")
    finally:
        agent.close()


@cli.group()
def swarm() -> None:
    """Multi-agent swarm commands."""
    pass


@swarm.command("run")
@click.argument("task")
@click.option("--profile", "-p", default="lead", help="Agent profile to use as workers")
@click.pass_context
def swarm_run(ctx: click.Context, task: str, profile: str) -> None:
    """Run a task through a multi-agent swarm."""
    from huginn.agents.swarm import AgentRole, HuginnSwarm, SwarmAgent

    agent = _build_agent_from_ctx(ctx, profile_id=profile)
    if agent is None:
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
        console.print(Panel(
            f"[bold blue]Swarm Result[/bold blue]\n{result['final_output']}",
            border_style="blue",
        ))
        for step in result["trace"]:
            console.print(f"  [{step['role']}] {step['agent_name']} ({step['duration_ms']:.0f}ms)")
    finally:
        agent.close()


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
