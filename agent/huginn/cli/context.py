"""Shared CLI context and helpers used by all command modules."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console

from huginn.config import HuginnConfig

console = Console()


@dataclass
class CliContext:
    """Context object shared across all CLI commands."""

    workspace: Path
    config_path: str | None = None
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    ollama_url: str | None = None
    thinking: str | None = None
    dry_run: bool = False
    console: Console = field(default_factory=Console)
    # 新增的 flags, 默认值都是关 / None, 不影响现有命令
    plan_mode: bool = False  # 只读模式, 写工具一律走 ASK
    yolo: bool = False  # 跳过所有确认, 等价于 auto_approve_all
    prompt_text: str | None = None  # headless 模式: 跑一次就退出
    resume_thread_id: str | None = None  # 恢复指定会话
    allowed_tools: list[str] | None = None  # 工具白名单
    disallowed_tools: list[str] | None = None  # 工具黑名单

    def load_config(self) -> HuginnConfig:
        """Load config from --config or environment, then apply CLI overrides."""
        if self.config_path:
            try:
                cfg = HuginnConfig.load(self.config_path)
            except Exception as e:
                self.console.print(f"[yellow]Config load warning: {e}[/yellow]")
                cfg = HuginnConfig.from_env()
        else:
            cfg = HuginnConfig.from_env()

        cfg.apply_overrides(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            ollama_url=self.ollama_url,
            thinking=self.thinking,
        )
        return cfg


def resolve_abaqus_mcp_path(config_path: str | None = None) -> Path:
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


_mcp_manager: Any | None = None


async def init_mcp(abaqus_mcp_server: str | None = None) -> None:
    """Initialize MCP servers and register their tools."""
    global _mcp_manager
    try:
        from huginn.mcp_client import MCPClientManager, MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_tools

        _mcp_manager = MCPClientManager()
        base = Path(__file__).parent.parent.parent.parent  # repo root

        mat_db_path = base / "servers" / "mat-db-mcp" / "server.py"
        if mat_db_path.exists():
            await _mcp_manager.connect(
                MCPServerConfig(
                    name="mat-db",
                    command="python",
                    args=[str(mat_db_path)],
                )
            )
            console.print("[green]✓[/green] MCP mat-db connected")

        math_path = base / "servers" / "math-anything-mcp" / "server.py"
        if math_path.exists():
            await _mcp_manager.connect(
                MCPServerConfig(
                    name="math-anything",
                    command="python",
                    args=[str(math_path)],
                )
            )
            console.print("[green]✓[/green] MCP math-anything connected")

        abaqus_path = resolve_abaqus_mcp_path(abaqus_mcp_server)
        if abaqus_path.exists():
            try:
                await _mcp_manager.connect(
                    MCPServerConfig(
                        name="abaqus",
                        command="python",
                        args=[str(abaqus_path)],
                    )
                )
                console.print("[green]✓[/green] MCP abaqus connected")
            except Exception as e:
                console.print(f"[yellow]MCP abaqus connection skipped: {e}[/yellow]")
        else:
            console.print(
                "[dim]Abaqus MCP server not found; install from .abaqus-mcp to enable[/dim]"
            )

        registered = register_mcp_tools(_mcp_manager)
        if registered:
            console.print(f"[green]✓[/green] Registered {len(registered)} MCP tools")
    except Exception as e:
        console.print(f"[yellow]MCP init warning: {e}[/yellow]")


async def shutdown_mcp() -> None:
    """Shutdown MCP connections."""
    global _mcp_manager
    if _mcp_manager:
        await _mcp_manager.disconnect_all()
        _mcp_manager = None


def build_agent_from_ctx(
    ctx: CliContext,
    profile_id: str = "lead",
    approval_callback: Callable[[str, str], bool] | None = None,
) -> Any | None:
    """Build a HuginnAgent from the resolved configuration."""
    from huginn.agent import HuginnAgent
    from huginn.security import AuditLogger, SandboxConfig, SandboxExecutor

    cfg = ctx.load_config()

    if cfg.provider == "default" and not cfg.models:
        ctx.console.print("[yellow]No provider or model pool configured.[/yellow]")
        ctx.console.print(
            "[dim]Run `huginn configure` or set HUGINN_PROVIDER / HUGINN_MODELS.[/dim]"
        )
        return None

    sandbox_cfg = SandboxConfig(dry_run=ctx.dry_run)
    sandbox = SandboxExecutor(sandbox_cfg)
    audit = AuditLogger(ctx.workspace / "huginn_audit.jsonl")

    overrides: dict[str, Any] = {}
    if approval_callback is not None:
        overrides["approval_callback"] = approval_callback

    # --resume: 把 thread_id 传给 agent, 让 LangGraph 恢复那个会话的状态
    if ctx.resume_thread_id:
        overrides["thread_id"] = ctx.resume_thread_id

    # --yolo: auto_approve=True, 跳过所有 ASK 确认
    # 注意: plan_mode 优先级更高, 同时开 yolo + plan 时写工具仍需确认
    if ctx.yolo:
        overrides["auto_approve"] = True

    # 工具白名单: 直接传 tool_filter, agent 注册工具时会按它过滤
    if ctx.allowed_tools:
        overrides["tool_filter"] = list(ctx.allowed_tools)

    # 工具黑名单: agent 没有 disallow 参数, 这里折算成白名单
    # 全量工具减去黑名单, 得到的就是允许注册的工具
    if ctx.disallowed_tools:
        disallowed_set = set(ctx.disallowed_tools)
        try:
            from huginn.tools.registry import ToolRegistry

            all_tools = set(ToolRegistry.list_tools())
        except Exception:
            all_tools = set()
        allowed = all_tools - disallowed_set
        # 跟 --allowed-tools 合并: 取交集, 没给白名单就用算出来的这批
        if overrides.get("tool_filter"):
            overrides["tool_filter"] = [
                t for t in overrides["tool_filter"] if t in allowed
            ]
        else:
            overrides["tool_filter"] = sorted(allowed)

    try:
        agent = HuginnAgent.from_config(
            cfg,
            profile_id=profile_id,
            sandbox=sandbox,
            audit=audit,
            **overrides,
        )
    except Exception as e:
        ctx.console.print(f"[red]Failed to build agent: {e}[/red]")
        return None

    # plan_mode 不能通过构造器传 (PermissionConfig 是 agent 内部创建的),
    # 在 agent 建好之后直接改 _permission_config.plan_mode
    if ctx.plan_mode and agent is not None:
        perm_cfg = getattr(agent, "_permission_config", None)
        if perm_cfg is not None:
            perm_cfg.plan_mode = True

    return agent
