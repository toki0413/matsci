"""Tests for the Huginn persona system (inspired by AstrBot)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from huginn.personas import Persona, PersonaManager


def test_builtin_personas() -> None:
    mgr = PersonaManager()
    assert "default" in mgr.list()
    assert "dft_expert" in mgr.list()
    default = mgr.get()
    assert "computational materials science" in default.system_prompt


def test_default_fallback() -> None:
    mgr = PersonaManager()
    p = mgr.get("nonexistent")
    assert p.name == "default"


def test_create_and_persist() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "personas.json"
        mgr = PersonaManager(personas_path=path)
        mgr.create(
            name="test_bot",
            system_prompt="You are a test bot.",
            begin_dialogs=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        )
        assert "test_bot" in mgr.list()

        # Re-load from disk
        mgr2 = PersonaManager(personas_path=path)
        p = mgr2.get("test_bot")
        assert p.system_prompt == "You are a test bot."
        assert len(p.begin_dialogs) == 2


def test_set_default_and_delete() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "personas.json"
        mgr = PersonaManager(personas_path=path)
        mgr.create(name="custom", system_prompt="custom prompt")
        mgr.set_default("custom")
        assert mgr.get_default_name() == "custom"
        assert mgr.get().name == "custom"

        mgr.delete("custom")
        assert "custom" not in mgr.list()
        assert mgr.get_default_name() == "default"


def test_cannot_delete_builtin() -> None:
    mgr = PersonaManager()
    with pytest.raises(ValueError, match="built-in"):
        mgr.delete("default")


def test_persona_from_dict() -> None:
    p = Persona.from_dict({
        "name": "legacy",
        "prompt": "legacy prompt",
        "begin_dialogs": [{"role": "user", "content": "hello"}],
    })
    assert p.name == "legacy"
    assert p.system_prompt == "legacy prompt"
