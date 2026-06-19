"""Execute workflow stages command."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from huginn.cli.context import CliContext
from huginn.security.audit import AuditLogger
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext


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
