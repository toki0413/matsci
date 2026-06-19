"""Unified scientific computing framework commands."""

from __future__ import annotations

from typing import Any

import click
from rich.panel import Panel

from huginn.cli.context import CliContext


@click.group(name="unified")
@click.pass_obj
def unified(ctx: CliContext) -> None:
    """Unified scientific computing framework."""


@unified.command("list")
@click.pass_obj
def unified_list(ctx: CliContext) -> None:
    """List available unified models and bridges."""
    from huginn.unified.bridge import list_bridges
    from huginn.unified.models import list_models

    ctx.console.print(
        Panel("[bold blue]Unified Models[/bold blue]", border_style="blue")
    )
    for name in list_models():
        ctx.console.print(f"  - {name}")
    ctx.console.print(
        Panel("[bold blue]Multiscale Bridges[/bold blue]", border_style="blue")
    )
    for name in list_bridges():
        ctx.console.print(f"  - {name}")


@unified.command("derive")
@click.argument("model")
@click.pass_obj
def unified_derive(ctx: CliContext, model: str) -> None:
    """Derive governing equations for a unified model."""
    from huginn.unified import derive_equations
    from huginn.unified.models import get_model

    factory = get_model(model)
    if not factory:
        ctx.console.print(f"[red]Model '{model}' not found[/red]")
        return
    problem = factory()
    result = derive_equations(problem)
    ctx.console.print(
        Panel(
            f"[bold blue]{problem.name}[/bold blue]\n"
            f"Principle: {result['principle']}\n"
            f"Equations:",
            title="Unified Derivation",
            border_style="blue",
        )
    )
    for key, eq in result["equations"].items():
        ctx.console.print(f"  [bold]{key}:[/bold] {eq}")


@unified.command("bridge")
@click.argument("name")
@click.option("--model", help="Model name required by some bridges (e.g. dft_to_md)")
@click.option(
    "--expression", help="Potential expression for cauchy_born / md_to_elasticity"
)
@click.option("--symbols", help="Comma-separated symbol list for --expression")
@click.pass_obj
def unified_bridge(
    ctx: CliContext,
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
        ctx.console.print(f"[red]Bridge '{name}' not found[/red]")
        return

    kwargs: dict[str, Any] = {}
    if bridge_name == "dft_to_md":
        if model:
            factory = get_model(model)
            if not factory:
                ctx.console.print(f"[red]Model '{model}' not found[/red]")
                return
            kwargs["dft_problem"] = factory()
        else:
            from huginn.unified.models import one_d_kohn_sham_dft

            kwargs["dft_problem"] = one_d_kohn_sham_dft()
    elif bridge_name in ("cauchy_born", "md_to_elasticity"):
        if not expression or not symbols:
            ctx.console.print(
                "[red]--expression and --symbols required for this bridge[/red]"
            )
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
    ctx.console.print(
        Panel(
            f"[bold blue]{name}[/bold blue]\n{result.get('interpretation', '')}",
            title="Multiscale Bridge",
            border_style="blue",
        )
    )
    for key, val in result.items():
        if key == "interpretation":
            continue
        ctx.console.print(f"  [bold]{key}:[/bold] {val}")


@unified.command("solve")
@click.argument("model")
@click.option("--method", default="fem", help="fem | fd")
@click.option("--n", default=10, help="Number of elements/points")
@click.option("--plot", "plot_path", default=None, help="Path to save solution plot")
@click.pass_obj
def unified_solve(
    ctx: CliContext,
    model: str,
    method: str,
    n: int,
    plot_path: str | None,
) -> None:
    """Discretize and solve a unified model."""
    from huginn.unified import solve, solve_and_plot
    from huginn.unified.models import get_model

    factory = get_model(model)
    if not factory:
        ctx.console.print(f"[red]Model '{model}' not found[/red]")
        return
    problem = factory()
    try:
        if plot_path:
            result = solve_and_plot(problem, method=method, n=n, output_path=plot_path)
        else:
            result = solve(problem, method=method, n=n)
    except Exception as e:
        ctx.console.print(f"[red]Solve failed: {e}[/red]")
        return

    info = (
        f"Method: {result['method']}, DOFs: {result['n_dof']}, "
        f"Residual: {result['residual']:.3e}"
    )
    ctx.console.print(
        Panel(
            f"[bold blue]{model}[/bold blue]",
            title="Unified Solve",
            subtitle=info,
            border_style="blue",
        )
    )
    ctx.console.print(f"[bold]Mesh:[/bold] {result['mesh']}")
    ctx.console.print(f"[bold]Solution:[/bold] {result['solution']}")
    if plot_path:
        ctx.console.print(f"[green]Plot saved to {result['plot_path']}[/green]")


@unified.command("discretize")
@click.argument("model")
@click.option("--method", default="fem", help="fem | fd")
@click.option("--n", default=10, help="Number of elements/points")
@click.pass_obj
def unified_discretize(ctx: CliContext, model: str, method: str, n: int) -> None:
    """Discretize a unified model into a linear algebraic system."""
    from huginn.unified import discretize
    from huginn.unified.models import get_model

    factory = get_model(model)
    if not factory:
        ctx.console.print(f"[red]Model '{model}' not found[/red]")
        return
    problem = factory()
    try:
        result = discretize(problem, method=method, n=n)
    except Exception as e:
        ctx.console.print(f"[red]Discretization failed: {e}[/red]")
        return

    info = f"Method: {result['method']}, DOFs: {result['n_dof']}"
    ctx.console.print(
        Panel(
            f"[bold blue]{model}[/bold blue]",
            title="Discretization",
            subtitle=info,
            border_style="blue",
        )
    )
    ctx.console.print("[bold]Stiffness matrix:[/bold]")
    for row in result["stiffness_matrix"]:
        ctx.console.print(f"  {row}")
    ctx.console.print(f"[bold]Load vector:[/bold] {result['load_vector']}")
