"""CLI entry point for Huginn.

Inspired by EvoScientist's CLI and Claude Code's main.tsx.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from huginn.cli.availability import filter_commands_by_availability, get_auth_state
from huginn.cli.commands import register_commands
from huginn.cli.context import CliContext
from huginn.tools import register_all_tools


@click.group()
@click.option("--workspace", "-w", default=".", help="Workspace directory")
@click.option("--config", "-c", help="Config file path")
@click.option("--model", "-m", help="Model name (e.g., claude-sonnet-4-6, gpt-5.4)")
@click.option(
    "--provider",
    "-p",
    help=(
        "Provider (anthropic, openai, ollama, deepseek, "
        "siliconflow, moonshot, zhipu, baichuan, dashscope, "
        "qianfan, doubao, hunyuan, openai-compatible)"
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be executed without running commands",
)
@click.option(
    "--base-url",
    "-u",
    help="Base URL for OpenAI-compatible endpoints (vLLM, LM Studio, etc.)",
)
@click.option("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
@click.option(
    "--thinking",
    type=click.Choice(["low", "medium", "high"]),
    help="Reasoning intensity (Anthropic extended thinking / OpenAI reasoning effort)",
)
@click.option(
    "--plan",
    is_flag=True,
    help="Plan mode: 只读模式, 所有写工具强制 ASK, 需要人工确认",
)
@click.option(
    "--yolo",
    is_flag=True,
    help="跳过所有确认, 等价于 auto_approve_all (跟 --plan 一起用时 plan 优先)",
)
@click.option(
    "--prompt",
    "-P",
    default=None,
    help="Headless 模式: 执行一次 prompt 后退出, 不进入交互循环",
)
@click.option(
    "--resume",
    default=None,
    help="恢复指定会话 (thread_id), 接着上次的上下文继续聊",
)
@click.option(
    "--allowed-tools",
    default=None,
    help="工具白名单, 逗号分隔 (e.g. file_read_tool,structure_tool)",
)
@click.option(
    "--disallowed-tools",
    default=None,
    help="工具黑名单, 逗号分隔 (e.g. bash_tool,git_commit_tool)",
)
@click.pass_context
def cli(
    ctx: click.Context,
    workspace: str,
    config: str | None,
    model: str | None,
    provider: str | None,
    base_url: str | None,
    dry_run: bool,
    ollama_url: str,
    thinking: str | None,
    plan: bool,
    yolo: bool,
    prompt: str | None,
    resume: str | None,
    allowed_tools: str | None,
    disallowed_tools: str | None,
) -> None:
    """Huginn: Material Science specialized AI Agent Harness."""
    ctx.ensure_object(dict)

    # 逗号分隔的工具列表拆成 list[str], 去掉空白和空串
    def _split_csv(s: str | None) -> list[str] | None:
        if not s:
            return None
        items = [x.strip() for x in s.split(",")]
        return [x for x in items if x] or None

    ctx.obj = CliContext(
        workspace=Path(workspace).resolve(),
        config_path=config,
        model=model,
        provider=provider,
        base_url=base_url,
        ollama_url=ollama_url,
        thinking=thinking,
        dry_run=dry_run,
        plan_mode=plan,
        yolo=yolo,
        prompt_text=prompt,
        resume_thread_id=resume,
        allowed_tools=_split_csv(allowed_tools),
        disallowed_tools=_split_csv(disallowed_tools),
    )

    # Register all tools, wiring in the configured execution backend
    # (local sandbox or remote HPC) before subcommands run.
    register_all_tools(ctx.obj.load_config())

    # 按 availability 隐藏不可用的命令, 避免在没认证/HPC 的环境下
    # 把一堆用不了的命令暴露给用户
    auth_state = get_auth_state()
    filter_commands_by_availability(cli, auth_state)


# 懒加载模式: 通过 HUGINN_LAZY_CLI=1 启用, 启动时不导入任何命令模块
# 与 eager 模式(register_commands)互斥, 不能同时使用
if os.environ.get("HUGINN_LAZY_CLI", "").lower() in ("1", "true", "yes"):
    from huginn.cli.lazy_loader import register_lazy_commands

    register_lazy_commands(cli)
else:
    register_commands(cli)


def main() -> None:
    """Entry point."""
    # Ensure UTF-8 output on Windows to handle scientific symbols (Å, °, etc.)
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")

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
