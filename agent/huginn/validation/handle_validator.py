"""Handle validation — pre-flight checks for tool inputs.

Validates that handles (file paths, job IDs, material IDs) exist
or are well-formed before tool execution begins.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from huginn.types import HandleType, ToolContext, ValidationResult


class HandleValidator:
    """Registry of handle-type validators.

    Usage::

        HandleValidator.validate(HandleType.FILE_PATH, "/some/path", context)
    """

    _checkers: dict[HandleType, Callable[[str, ToolContext], ValidationResult]] = {}

    @classmethod
    def register(
        cls,
        handle_type: HandleType,
        checker: Callable[[str, ToolContext], ValidationResult],
    ) -> None:
        cls._checkers[handle_type] = checker

    @classmethod
    def validate(
        cls, handle_type: HandleType, value: str, context: ToolContext
    ) -> ValidationResult:
        checker = cls._checkers.get(handle_type)
        if not checker:
            return ValidationResult(
                result=True, message=f"No checker for {handle_type.value}"
            )
        return checker(value, context)

    @classmethod
    def list_types(cls) -> list[str]:
        return [ht.value for ht in cls._checkers]


# ── Built-in checkers ──────────────────────────────────────────────


def _check_file_path(value: str, context: ToolContext) -> ValidationResult:
    """Check that a file or directory path exists."""
    p = Path(value)
    if p.exists():
        return ValidationResult(result=True)
    if context and context.workspace:
        p = Path(context.workspace) / value
        if p.exists():
            return ValidationResult(result=True)
    return ValidationResult(
        result=False,
        message=f"Path does not exist: {value}",
        error_code=404,
    )


def _check_job_id(value: str, context: ToolContext) -> ValidationResult:
    if not value or not value.strip():
        return ValidationResult(
            result=False, message="Job ID cannot be empty", error_code=400
        )
    return ValidationResult(result=True)


def _check_material_id(value: str, context: ToolContext) -> ValidationResult:
    if not value or not value.strip():
        return ValidationResult(
            result=False, message="Material ID cannot be empty", error_code=400
        )
    return ValidationResult(result=True)


def _check_formula(value: str, context: ToolContext) -> ValidationResult:
    if not value or not value.strip():
        return ValidationResult(
            result=False, message="Formula cannot be empty", error_code=400
        )
    if not re.search(r"[A-Z]", value):
        return ValidationResult(
            result=False,
            message=f"Formula '{value}' must contain at least one element symbol",
            error_code=400,
        )
    return ValidationResult(result=True)


# Register built-in checkers
HandleValidator.register(HandleType.FILE_PATH, _check_file_path)
HandleValidator.register(HandleType.JOB_ID, _check_job_id)
HandleValidator.register(HandleType.MATERIAL_ID, _check_material_id)
HandleValidator.register(HandleType.FORMULA, _check_formula)
