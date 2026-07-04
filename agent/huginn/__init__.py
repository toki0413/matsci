"""Huginn: Material Science specialized AI Agent Harness.

Architecture inspired by Claude Code's Tool System, QueryEngine,
and Exploration patterns, built on Python/LangGraph/EvoScientist.
"""

import enum as _enum

# StrEnum was added in Python 3.11.  Patch a minimal backport onto the
# ``enum`` module so all submodules can use ``enum.StrEnum`` (or
# ``from enum import StrEnum``) regardless of the running Python version.
if not hasattr(_enum, "StrEnum"):  # pragma: no cover

    class _StrEnumBackport(str, _enum.Enum):
        __str__ = str.__str__

    _enum.StrEnum = _StrEnumBackport  # type: ignore[attr-defined]

__version__ = "0.1.0"
