"""Knowledge graph commands."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.panel import Panel

from huginn.cli.context import CliContext
from huginn.kg import (
    ProjectKnowledgeGraph,
    build_from_logs,
    build_from_memory,
    build_from_seeds,
)
from huginn.kg.query import GraphQuery


@click.group(name="kg")
@click.pass_obj
def kg(ctx: CliContext) -> None:
    """Project knowledge graph commands."""


@click.command(name="build-kg")
@click.option("--from-memory", is_flag=True, help="Import from long-term memory")
@click.option("--from-logs", is_flag=True, help="Import from execution logs")
@click.option("--from-seeds", is_flag=True, help="Import from built-in seed documents")
@click.pass_obj
def build_kg(
    ctx: CliContext,
    from_memory: bool,
    from_logs: bool,
    from_seeds: bool,
) -> None:
    """Build or update the project knowledge graph."""
    kg_root = ctx.workspace / ".huginn"
    kg = ProjectKnowledgeGraph(kg_root)

    total = {
        "facts": 0,
        "sessions": 0,
        "tools": 0,
        "errors": 0,
        "topics": 0,
        "links": 0,
    }

    if from_seeds:
        stats = build_from_seeds(kg)
        total["topics"] += stats.get("topics", 0)
        total["links"] += stats.get("links", 0)

    if from_memory:
        from huginn.memory.longterm import LongTermMemory

        ltm = LongTermMemory()
        stats = build_from_memory(kg, ltm)
        total["facts"] += stats.get("facts", 0)
        total["links"] += stats.get("links", 0)

    if from_logs:
        from huginn.evolution.logger import ExecutionLogger

        logger = ExecutionLogger()
        stats = build_from_logs(kg, logger)
        for key in ("sessions", "tools", "errors", "links"):
            total[key] += stats.get(key, 0)

    kg.save()
    ctx.console.print(
        Panel(
            f"[bold blue]Project Knowledge Graph[/bold blue]\n"
            f"Saved to: {kg.path}\n"
            f"Facts: {total['facts']}  Topics: {total['topics']}\n"
            f"Sessions: {total['sessions']}  Tools: {total['tools']}  Errors: {total['errors']}\n"
            f"Entity links: {total['links']}",
            border_style="blue",
        )
    )


@kg.command("query")
@click.argument("seed")
@click.option("--depth", default=1, help="Neighborhood depth")
@click.option("--top-k", default=10, help="Max nodes to return")
@click.pass_obj
def kg_query(ctx: CliContext, seed: str, depth: int, top_k: int) -> None:
    """Query the knowledge graph for a given seed."""
    kg_root = ctx.workspace / ".huginn"
    if not (kg_root / ProjectKnowledgeGraph.FILENAME).exists():
        ctx.console.print(
            "[yellow]No knowledge graph found. Run `huginn build-kg` first.[/yellow]"
        )
        return
    kg = ProjectKnowledgeGraph(kg_root)
    query = GraphQuery(kg._graph)
    result = query.query(seed, depth=depth, top_k=top_k)
    ctx.console.print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


@kg.command("stats")
@click.pass_obj
def kg_stats(ctx: CliContext) -> None:
    """Show knowledge graph statistics."""
    kg_root = ctx.workspace / ".huginn"
    if not (kg_root / ProjectKnowledgeGraph.FILENAME).exists():
        ctx.console.print(
            "[yellow]No knowledge graph found. Run `huginn build-kg` first.[/yellow]"
        )
        return
    kg = ProjectKnowledgeGraph(kg_root)
    stats = kg.stats()
    ctx.console.print(
        Panel(
            f"[bold blue]Knowledge Graph Stats[/bold blue]\n"
            f"Nodes: {stats['nodes']}\n"
            f"Edges: {stats['edges']}\n"
            f"Node types: {stats['node_types']}",
            border_style="blue",
        )
    )


@kg.command("export")
@click.option("--format", "fmt", default="json", type=click.Choice(["json", "gml"]))
@click.option("--output", "-o", help="Output file (default: stdout)")
@click.pass_obj
def kg_export(ctx: CliContext, fmt: str, output: str | None) -> None:
    """Export the knowledge graph as JSON or GML."""
    kg_root = ctx.workspace / ".huginn"
    if not (kg_root / ProjectKnowledgeGraph.FILENAME).exists():
        ctx.console.print(
            "[yellow]No knowledge graph found. Run `huginn build-kg` first.[/yellow]"
        )
        return
    kg = ProjectKnowledgeGraph(kg_root)
    data = kg.export(fmt=fmt)
    text = (
        json.dumps(data, indent=2, ensure_ascii=False, default=str)
        if fmt == "json"
        else str(data)
    )
    if output:
        Path(output).write_text(text, encoding="utf-8")
        ctx.console.print(f"[green]✓[/green] Exported to {output}")
    else:
        ctx.console.print(text)
