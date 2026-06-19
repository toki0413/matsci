"""CLI entry point for Huginn.

Inspired by EvoScientist's CLI and Claude Code's main.tsx.
"""

from __future__ import annotations

from pathlib import Path

import click

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
) -> None:
    """Huginn: Material Science specialized AI Agent Harness."""
    ctx.ensure_object(dict)
    ctx.obj = CliContext(
        workspace=Path(workspace).resolve(),
        config_path=config,
        model=model,
        provider=provider,
        base_url=base_url,
        ollama_url=ollama_url,
        thinking=thinking,
        dry_run=dry_run,
    )

    # Register all tools, wiring in the configured execution backend
    # (local sandbox or remote HPC) before subcommands run.
    register_all_tools(ctx.obj.load_config())


register_commands(cli)


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
