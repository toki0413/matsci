"""OpenAPI contract export command.

Dumps the runtime FastAPI OpenAPI schema to a JSON file. This is the single
source of truth for the frontend contract: the committed ``openapi.json`` and
the generated TypeScript types are both derived from this command, so they can
never silently drift from the actual backend routes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console


@click.command(name="openapi")
@click.option(
    "--output",
    "-o",
    default="openapi.json",
    help="Output JSON file path (default: openapi.json)",
)
@click.option(
    "--pretty",
    is_flag=True,
    default=True,
    help="Pretty-print the JSON (default: on)",
)
def openapi_export(output: str, pretty: bool) -> None:
    """Export the runtime OpenAPI schema (all mounted routes incl. /v1)."""
    console = Console()

    try:
        from huginn.server import app
    except Exception as exc:  # pragma: no cover - import failure path
        console.print(f"[red]Failed to import server app:[/red] {exc}")
        sys.exit(1)

    try:
        schema = app.openapi()
    except Exception as exc:  # pragma: no cover - schema generation failure path
        console.print(f"[red]Failed to generate OpenAPI schema:[/red] {exc}")
        sys.exit(1)

    paths = schema.get("paths", {})
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        out.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        out.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")

    console.print(
        f"[green]Exported OpenAPI schema[/green] -> {out} "
        f"({len(paths)} paths, openapi {schema.get('openapi', '?')})"
    )
