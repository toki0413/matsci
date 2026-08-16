"""Huginn: General Scientific Research Agent Harness.

Evolved from a computational-materials-science agent into a multi-domain
scientific research automation agent (simulation, symbolic math, causal
analysis, knowledge distillation, multi-agent collaboration), with Lean 4
formal verification of the underlying math.
"""

import enum as _enum

# StrEnum was added in Python 3.11.  Patch a minimal backport onto the
# ``enum`` module so all submodules can use ``enum.StrEnum`` (or
# ``from enum import StrEnum``) regardless of the running Python version.
if not hasattr(_enum, "StrEnum"):  # pragma: no cover

    class _StrEnumBackport(_enum.StrEnum):
        __str__ = str.__str__

    _enum.StrEnum = _StrEnumBackport  # type: ignore[attr-defined]

__version__ = "1.3.0"
