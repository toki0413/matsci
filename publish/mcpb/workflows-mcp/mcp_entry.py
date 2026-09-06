"""Huginn workflows-mcp MCPB entry — runs the real huginn workflows-mcp CLI over stdio."""
from __future__ import annotations

import sys


def main() -> None:
    from huginn.cli.commands.workflows_mcp import workflows_mcp as cmd

    args = ["--transport", "stdio"] + sys.argv[1:]
    cmd.main(args=args, prog_name="workflows-mcp", standalone_mode=False)


if __name__ == "__main__":
    main()