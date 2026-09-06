"""Huginn capabilities-mcp MCPB entry — runs the real huginn capabilities-mcp CLI over stdio."""
from __future__ import annotations

import sys


def main() -> None:
    from huginn.cli.commands.capabilities_mcp import capabilities_mcp as cmd

    args = ["--transport", "stdio"] + sys.argv[1:]
    cmd.main(args=args, prog_name="capabilities-mcp", standalone_mode=False)


if __name__ == "__main__":
    main()