"""Diagnose computational errors command."""

from __future__ import annotations

import asyncio
import json

import click

from huginn.cli.context import CliContext
from huginn.types import ToolContext


@click.command()
@click.argument("error_message")
@click.option("--software", "-s", help="Software (e.g., VASP, LAMMPS, Gaussian)")
@click.option("--calculation-type", "-t", help="Calculation type (e.g., DFT, MD)")
@click.option("--context", "-c", "context_str", help="Additional context")
@click.pass_obj
def diagnose(
    ctx: CliContext,
    error_message: str,
    software: str | None,
    calculation_type: str | None,
    context_str: str | None,
) -> None:
    """Diagnose a computational chemistry/MD error."""
    from huginn.tools.diagnose_tool import DiagnoseInput, DiagnoseTool

    tool = DiagnoseTool()
    input_data = DiagnoseInput(
        error_message=error_message,
        software=software,
        calculation_type=calculation_type,
        context=context_str,
    )
    result = asyncio.run(
        tool.call(
            input_data,
            ToolContext(session_id="diagnose", workspace=str(ctx.workspace)),
        )
    )
    ctx.console.print(
        json.dumps(result.data, indent=2, ensure_ascii=False, default=str)
    )
