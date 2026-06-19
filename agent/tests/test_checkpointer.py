"""Tests for persistent agent state checkpointing."""

from __future__ import annotations

import importlib.util
import pytest

_sqlite_saver_available = importlib.util.find_spec("langgraph.checkpoint.sqlite") is not None

import tempfile
from pathlib import Path

from huginn.agent import HuginnAgent
from huginn.checkpointer import create_checkpointer, create_in_memory_checkpointer


class TestCheckpointerFactory:
    def test_in_memory_checkpointer(self):
        cp = create_in_memory_checkpointer()
        assert cp is not None

    @pytest.mark.skipif(not _sqlite_saver_available, reason="langgraph sqlite checkpointer not available")
    def test_sqlite_checkpointer_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.sqlite"
            cp = create_checkpointer(path)
            assert cp is not None
            assert path.exists()

    @pytest.mark.skipif(not _sqlite_saver_available, reason="langgraph sqlite checkpointer not available")
    def test_env_path(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env.sqlite"
            monkeypatch.setenv("HUGINN_CHECKPOINTER_PATH", str(path))
            cp = create_checkpointer()
            assert cp is not None
            assert path.exists()


class TestHuginnAgentCheckpointer:
    def test_default_is_in_memory(self):
        agent = HuginnAgent()
        # InMemorySaver class name; avoid importing langgraph internals.
        assert "InMemory" in type(agent.checkpointer).__name__

    @pytest.mark.skipif(not _sqlite_saver_available, reason="langgraph sqlite checkpointer not available")
    def test_persistent_by_path(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            HuginnAgent(checkpointer_path=str(Path(tmp) / "cp.sqlite")) as agent,
        ):
            assert "Sqlite" in type(agent.checkpointer).__name__

    @pytest.mark.skipif(not _sqlite_saver_available, reason="langgraph sqlite checkpointer not available")
    def test_persistent_by_env(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv(
                "HUGINN_CHECKPOINTER_PATH", str(Path(tmp) / "env.sqlite")
            )
            with HuginnAgent() as agent:
                assert "Sqlite" in type(agent.checkpointer).__name__