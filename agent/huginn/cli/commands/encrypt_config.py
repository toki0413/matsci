"""Encrypt a configuration file."""

from __future__ import annotations

from pathlib import Path

import click

from huginn.cli.context import CliContext
from huginn.config import HuginnConfig


@click.command("encrypt-config")
@click.argument("path", default="huginn.toml")
@click.option("--password", prompt=True, hide_input=True, help="Encryption password")
@click.pass_obj
def encrypt_config(ctx: CliContext, path: str, password: str) -> None:
    """Encrypt a configuration file."""
    target = Path(path)
    cfg = HuginnConfig.load(path) if target.exists() else HuginnConfig.from_env()
    cfg.encrypt_config = True
    cfg.encryption_password = password
    out = (
        target
        if str(target).endswith(".enc")
        else target.with_suffix(target.suffix + ".enc")
    )
    cfg.save(out, format="json")
    ctx.console.print(f"[green]✓[/green] Encrypted config saved to {out}")
