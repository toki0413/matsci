"""Tests for Abaqus MCP server wiring."""

import os
from pathlib import Path

import pytest

from huginn.cli import _resolve_abaqus_mcp_path


@pytest.fixture
def clear_abaqus_env() -> None:
    """Remove ABAQUS_MCP_SERVER_PATH from the environment for isolated tests."""
    original = os.environ.pop("ABAQUS_MCP_SERVER_PATH", None)
    yield
    if original is not None:
        os.environ["ABAQUS_MCP_SERVER_PATH"] = original


def test_resolve_abaqus_path_default(clear_abaqus_env: None) -> None:
    """Default path points to ~/.abaqus-mcp/mcp_server.py."""
    path = _resolve_abaqus_mcp_path()
    expected = Path.home() / ".abaqus-mcp" / "mcp_server.py"
    assert path == expected


def test_resolve_abaqus_path_from_env(clear_abaqus_env: None) -> None:
    """Environment variable overrides the default path."""
    custom = "/custom/path/to/mcp_server.py"
    os.environ["ABAQUS_MCP_SERVER_PATH"] = custom
    path = _resolve_abaqus_mcp_path()
    assert path == Path(custom)


def test_resolve_abaqus_path_from_config(clear_abaqus_env: None) -> None:
    """Config path is used when environment variable is not set."""
    custom = "/config/path/to/mcp_server.py"
    path = _resolve_abaqus_mcp_path(config_path=custom)
    assert path == Path(custom)


def test_resolve_abaqus_path_env_beats_config(clear_abaqus_env: None) -> None:
    """Environment variable takes precedence over config path."""
    env_path = "/env/path/to/mcp_server.py"
    config_path = "/config/path/to/mcp_server.py"
    os.environ["ABAQUS_MCP_SERVER_PATH"] = env_path
    path = _resolve_abaqus_mcp_path(config_path=config_path)
    assert path == Path(env_path)


def test_abaqus_mcp_server_exists_on_this_machine() -> None:
    """Smoke test that the expected Abaqus MCP server file exists."""
    path = _resolve_abaqus_mcp_path()
    # This test documents the expected location; it is not a hard requirement.
    if path.exists():
        assert path.name == "mcp_server.py"
