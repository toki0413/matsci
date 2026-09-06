"""Execute workflow stages command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from huginn.cli.context import CliContext


@click.command()
@click.argument("stages")
@click.option("--working-dir", "-w", default=".", help="Working directory")
@click.option("--name", "-n", default="execute", help="Workflow name")
@click.pass_obj
def execute(ctx: CliContext, stages: str, working_dir: str, name: str) -> None:
    """Run a list of workflow stages via the execution orchestrator.

    STAGES can be a JSON file path or an inline JSON array of stage dicts.
    """
    from huginn.execution.orchestrator import ExecutionOrchestrator

    stage_path = Path(stages)
    raw = stage_path.read_text(encoding="utf-8") if stage_path.exists() else stages
    stage_list = json.loads(raw)

    orch = ExecutionOrchestrator(working_dir=working_dir)
    # 注: 不调用 orch.register_tool() — orchestrator 默认接全局 ToolRegistry
    # 类, register_tool 仅对 dict 模式有效, 这里逐个注册纯属无效且每个都打
    # warning。run() 内部用 ToolRegistry.get() 取工具, 由 _invoke_tool 桥接。

    record = asyncio.run(orch.run(stage_list, workflow_name=name))
    ctx.console.print(
        json.dumps(
            {
                "workflow_name": record.workflow_name,
                "overall_success": record.overall_success,
                "stages": [r.to_dict() for r in record.stage_results],
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )
