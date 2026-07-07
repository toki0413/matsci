"""Show version information."""

from __future__ import annotations

import click

from huginn import __version__
from huginn.cli.context import console
from huginn.pet import get_pet_avatar
import logging
logger = logging.getLogger(__name__)



@click.command()
def version() -> None:
    """Show version information."""
    console.print(f"[dim]{get_pet_avatar()}[/dim]")
    console.print(f"Huginn [bold]{__version__}[/bold]")

    try:
        import langchain

        console.print(f"  langchain: {langchain.__version__}")
    except Exception:
        logger.debug("version failed", exc_info=True)

    try:
        import langgraph

        console.print(f"  langgraph: {langgraph.__version__}")
    except Exception:
        logger.debug("version failed", exc_info=True)

    try:
        import pydantic

        console.print(f"  pydantic: {pydantic.__version__}")
    except Exception:
        logger.debug("version failed", exc_info=True)
