"""Semantic diff tool — compare calculations using math-anything's MathDiffer.

Read-only. Safe to auto-execute.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool, ResearchPhase, ToolProfile


class DiffToolInput(BaseModel):
    calc_a: str = Field(..., description="Path or ID of first calculation result")
    calc_b: str = Field(..., description="Path or ID of second calculation result")
    comparison_type: Literal["parameters", "results", "full"] = Field(default="full")
    inline_a: dict[str, Any] | None = Field(
        default=None, description="Inline dict for calc_a (overrides file path)"
    )
    inline_b: dict[str, Any] | None = Field(
        default=None, description="Inline dict for calc_b (overrides file path)"
    )


def _diff_entry(
    change_type: str, path: str, severity: str, **values: Any
) -> dict[str, Any]:
    return {"type": change_type, "path": path, **values, "severity": severity}


def _severity_for_path(path: str) -> str:
    severity = "info"
    if path and any(
        kw in path.lower()
        for kw in ("equation", "boundary", "conservation", "symmetry")
    ):
        severity = "critical"
    return severity


def _deep_diff(old: Any, new: Any, path: str = "") -> list[dict[str, Any]]:
    """Built-in recursive dict diff fallback when MathDiffer is unavailable."""
    changes: list[dict[str, Any]] = []
    if isinstance(old, dict) and isinstance(new, dict):
        all_keys = set(old) | set(new)
        for key in sorted(all_keys):
            child_path = f"{path}.{key}" if path else key
            if key not in old:
                changes.append(
                    _diff_entry("added", child_path, "info", new_value=new[key])
                )
            elif key not in new:
                changes.append(
                    _diff_entry("removed", child_path, "warning", old_value=old[key])
                )
            else:
                changes.extend(_deep_diff(old[key], new[key], child_path))
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            changes.append(
                _diff_entry(
                    "list_length_changed",
                    path,
                    "info",
                    old_value=len(old),
                    new_value=len(new),
                )
            )
        for i, (a, b) in enumerate(zip(old, new)):
            changes.extend(_deep_diff(a, b, f"{path}[{i}]"))
    elif old != new:
        changes.append(
            _diff_entry(
                "value_changed",
                path,
                _severity_for_path(path),
                old_value=old,
                new_value=new,
            )
        )
    return changes


def _try_math_differ(
    data_a: dict[str, Any],
    data_b: dict[str, Any],
    old_source: str,
    new_source: str,
) -> dict[str, Any] | None:
    try:
        from math_anything.utils.math_diff import MathDiffer

        differ = MathDiffer()
        diff_report = differ.compare(
            data_a,
            data_b,
            old_source=old_source,
            new_source=new_source,
        )
        # Only use MathDiffer if it found changes (non-schema data may yield 0)
        if diff_report.all_changes:
            return diff_report.to_dict()
    except (ImportError, Exception):
        # math-anything not available or schemas incompatible
        pass
    return None


def _filter_changes(
    changes: list[dict[str, Any]], comparison_type: str
) -> list[dict[str, Any]]:
    # Filter by comparison_type
    if comparison_type == "parameters":
        param_keywords = {
            "encut",
            "kpoint",
            "cutoff",
            "threshold",
            "tolerance",
            "convergence",
            "smearing",
            "functional",
            "pseudopotential",
        }
        changes = [
            c
            for c in changes
            if any(kw in c.get("path", "").lower() for kw in param_keywords)
        ]
    elif comparison_type == "results":
        result_keywords = {
            "energy",
            "force",
            "stress",
            "band_gap",
            "dos",
            "phonon",
            "elastic",
            "magnetization",
            "charge",
        }
        changes = [
            c
            for c in changes
            if any(kw in c.get("path", "").lower() for kw in result_keywords)
        ]
    return changes


def _builtin_diff_report(
    args: DiffToolInput, data_a: dict[str, Any], data_b: dict[str, Any]
) -> dict[str, Any]:
    changes = _filter_changes(_deep_diff(data_a, data_b), args.comparison_type)

    critical = [c for c in changes if c.get("severity") == "critical"]
    return {
        "old_source": args.calc_a,
        "new_source": args.calc_b,
        "summary": {
            "total_changes": len(changes),
            "critical": len(critical),
        },
        "changes": changes,
    }


def _semantic_summary(report_dict: dict[str, Any]) -> str:
    total = report_dict.get("summary", {}).get("total_changes", 0)
    critical_count = report_dict.get("summary", {}).get("critical", 0)
    if total == 0:
        return "No differences detected between the two calculations."
    elif critical_count > 0:
        return (
            f"Found {total} change(s) including {critical_count} critical. "
            "Critical changes may affect physical validity."
        )
    return f"Found {total} non-critical change(s) between calculations."


class DiffTool(HuginnTool):
    """Compare two calculations semantically (not just text diff)."""

    name = "diff_tool"
    category = "core"
    profile = ToolProfile(
        phases=frozenset({ResearchPhase.VALIDATION, ResearchPhase.REPORTING})
    )
    description = "Semantically compare two calculations: parameter changes, mathematical structure differences, and physical implications"
    input_schema = DiffToolInput

    def is_read_only(self, args: DiffToolInput) -> bool:
        return True

    def _load_data(self, ref: str, inline: dict | None) -> dict[str, Any]:
        """Load data from inline dict or file path."""
        if inline is not None:
            return inline
        path = Path(ref)
        if path.exists() and path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        # Return as-is if it looks like a plain identifier
        return {"_id": ref}

    async def call(self, args: DiffToolInput, context: ToolContext) -> ToolResult:
        data_a = self._load_data(args.calc_a, args.inline_a)
        data_b = self._load_data(args.calc_b, args.inline_b)

        # Try math-anything MathDiffer first for full schema-aware comparison
        report_dict: dict[str, Any] | None = None
        if args.comparison_type == "full":
            report_dict = _try_math_differ(
                data_a, data_b, old_source=args.calc_a, new_source=args.calc_b
            )
        math_differ_used = report_dict is not None

        # Fallback: built-in deep diff
        if report_dict is None:
            report_dict = _builtin_diff_report(args, data_a, data_b)

        # Build semantic summary
        report_dict["semantic_summary"] = _semantic_summary(report_dict)
        report_dict["comparison_type"] = args.comparison_type
        report_dict["engine"] = (
            "math_differ" if math_differ_used else "builtin_deep_diff"
        )

        return ToolResult(data=report_dict, success=True)
