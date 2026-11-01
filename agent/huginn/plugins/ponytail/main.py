"""Ponytail plugin — lazy-engineer decision ladder as commands + audit.

The ladder text the model reads lives in ``SKILL.md`` and reaches the
model through the skill loader.  This plugin adds the runtime half:
mode state (lite/full/ultra/off) persisted in plugin storage, plus
``/ponytail-review`` / ``/ponytail-audit`` AST scans for
over-engineering in the workspace.  Mode is reported on demand; it is
not injected into the system prompt.

Commands:
  /ponytail [lite|full|ultra|off]  — switch mode
  /ponytail-review                  — review last diff for over-engineering
  /ponytail-audit                   — scan workspace for over-engineering
  /ponytail-gain                    — show cumulative savings
"""

from __future__ import annotations

import ast
import datetime
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.api.context import PluginContext
from huginn.api.event import Event
from huginn.api.filter import filter
from huginn.api.star import Star

logger = logging.getLogger(__name__)


# Modes accepted by /ponytail; rules text lives in SKILL.md, not here.
_VALID_MODES = frozenset({"lite", "full", "ultra", "off"})


@dataclass
class PonytailState:
    """Track ponytail mode and cumulative savings."""

    mode: str = "full"
    # Count of times over-engineering was prevented
    skipped_count: int = 0
    # Estimated lines saved
    lines_saved: int = 0
    # History of mode changes for /ponytail-gain
    history: list[dict[str, Any]] = field(default_factory=list)


class PonytailStar(Star):
    """Ponytail mode state + AST audit commands.

    Mode is persisted in plugin storage so it survives across sessions.
    The rules text itself is a skill (SKILL.md), loaded by the skill
    loader; this class does not touch the system prompt.
    """

    name = "ponytail"
    author = "Huginn Integration"
    version = "1.0.0"
    description = "Enforces a lazy-engineer decision ladder to prevent over-engineering"
    priority = 50

    def __init__(self, context: PluginContext | None = None) -> None:
        super().__init__(context)
        self._state = PonytailState()

    async def on_load(self) -> None:
        # Restore mode from storage if available
        try:
            stored = self.storage.get("ponytail_state")
            if stored and isinstance(stored, dict):
                self._state = PonytailState(
                    **{
                        k: v
                        for k, v in stored.items()
                        if k in {"mode", "skipped_count", "lines_saved", "history"}
                    }
                )
        except Exception:
            # Storage might not be injected in test mode — fine
            logger.debug("best-effort op failed", exc_info=True)
        self.logger.info("ponytail loaded (mode=%s)", self._state.mode)

    async def on_unload(self) -> None:
        self._persist()

    def _persist(self) -> None:
        try:
            self.storage.set(
                "ponytail_state",
                {
                    "mode": self._state.mode,
                    "skipped_count": self._state.skipped_count,
                    "lines_saved": self._state.lines_saved,
                    "history": self._state.history[-20:],  # cap history
                },
            )
        except Exception:
            logger.debug("persist failed", exc_info=True)

    # ── Commands ──────────────────────────────────────────────────────

    @filter.command("/ponytail")
    @filter.event_message_type()
    async def handle_ponytail(self, event: Event) -> None:
        """Handle /ponytail [lite|full|ultra|off] — switch mode."""
        text = getattr(event, "text", "") or ""
        parts = text.strip().split()

        if len(parts) < 2:
            # No argument — show current mode
            reply = (
                f"Ponytail mode: **{self._state.mode}**\n"
                f"Skipped: {self._state.skipped_count} | "
                f"Lines saved: ~{self._state.lines_saved}\n"
                f"Switch: /ponytail [lite|full|ultra|off]"
            )
            self._set_reply(event, reply)
            return

        new_mode = parts[1].lower()
        if new_mode not in _VALID_MODES:
            self._set_reply(
                event, f"Unknown mode '{new_mode}'. Use: lite, full, ultra, off"
            )
            return

        old_mode = self._state.mode
        self._state.mode = new_mode
        self._state.history.append(
            {
                "from": old_mode,
                "to": new_mode,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )
        self._persist()

        reply = f"Ponytail: {old_mode} → **{new_mode}**"
        if new_mode == "off":
            reply += " (rules disabled)"
        elif new_mode == "ultra":
            reply += " (YAGNI extremist — deletion before addition)"
        elif new_mode == "lite":
            reply += " (suggest only — user picks)"
        self._set_reply(event, reply)

    @filter.command("/ponytail-review")
    @filter.event_message_type()
    async def handle_review(self, event: Event) -> None:
        """Review the last diff for over-engineering patterns."""
        workspace = Path(os.environ.get("HUGINN_WORKSPACE", "."))
        # Look for recently modified .py files
        py_files = sorted(
            workspace.rglob("*.py"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )[:5]

        findings: list[str] = []
        for f in py_files:
            try:
                content = f.read_text(encoding="utf-8")
                tree = ast.parse(content)
                issues = _audit_python_tree(tree, str(f))
                findings.extend(issues)
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                continue

        if not findings:
            self._set_reply(
                event, "Ponytail review: no over-engineering found in recent files. ✅"
            )
        else:
            lines = [f"Ponytail review: {len(findings)} issue(s) found:"]
            for issue in findings[:10]:
                lines.append(f"  ⚠ {issue}")
            if len(findings) > 10:
                lines.append(f"  ... and {len(findings) - 10} more")
            self._set_reply(event, "\n".join(lines))

    @filter.command("/ponytail-audit")
    @filter.event_message_type()
    async def handle_audit(self, event: Event) -> None:
        """Scan the workspace for over-engineering patterns."""
        workspace = Path(os.environ.get("HUGINN_WORKSPACE", "."))
        py_files = list(workspace.rglob("*.py"))
        py_files = [
            f
            for f in py_files
            if not any(
                p in str(f) for p in ("venv", "site-packages", "__pycache__", "test_")
            )
        ]

        all_findings: list[str] = []
        for f in py_files[:50]:
            try:
                content = f.read_text(encoding="utf-8")
                tree = ast.parse(content)
                issues = _audit_python_tree(tree, str(f.relative_to(workspace)))
                all_findings.extend(issues)
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                continue

        if not all_findings:
            self._set_reply(
                event,
                f"Ponytail audit: scanned {len(py_files)} files, no over-engineering found. ✅",
            )
        else:
            lines = [
                f"Ponytail audit: {len(all_findings)} issue(s) in {len(py_files)} files:"
            ]
            for issue in all_findings[:15]:
                lines.append(f"  ⚠ {issue}")
            if len(all_findings) > 15:
                lines.append(f"  ... and {len(all_findings) - 15} more")
            self._set_reply(event, "\n".join(lines))

    @filter.command("/ponytail-gain")
    @filter.event_message_type()
    async def handle_gain(self, event: Event) -> None:
        """Show cumulative ponytail savings."""
        self._set_reply(
            event,
            (
                f"Ponytail cumulative savings:\n"
                f"  Mode: {self._state.mode}\n"
                f"  Over-engineering prevented: {self._state.skipped_count}\n"
                f"  Estimated lines saved: ~{self._state.lines_saved}\n"
                f"  Mode changes: {len(self._state.history)}"
            ),
        )

    def _set_reply(self, event: Event, text: str) -> None:
        """Write reply to wherever the event expects it (extra or context)."""
        if hasattr(event, "extra"):
            event.extra["reply"] = text
        elif hasattr(event, "context"):
            event.context["reply"] = text


# ── Static analysis helpers ─────────────────────────────────────────


def _audit_python_tree(tree: ast.AST, filename: str) -> list[str]:
    """Walk an AST and flag over-engineering patterns."""
    findings: list[str] = []

    for node in ast.walk(tree):
        findings.extend(_audit_abstract_class(node, tree, filename))
        findings.extend(_audit_factory(node, filename))
        findings.extend(_audit_config_dict(node, filename))

    return findings


def _audit_abstract_class(node: ast.AST, tree: ast.AST, filename: str) -> list[str]:
    """Interface with single implementation."""
    findings: list[str] = []
    if isinstance(node, ast.ClassDef):
        bases = [getattr(b, "id", "") for b in node.bases if isinstance(b, ast.Name)]
        if "ABC" in bases:
            # Check if there's only one concrete subclass in the same file
            concrete = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef)
                and any(
                    getattr(b, "id", "") == node.name
                    for b in n.bases
                    if isinstance(b, ast.Name)
                )
            ]
            if len(concrete) <= 1:
                findings.append(
                    f"{filename}:{node.lineno} — ABC '{node.name}' has "
                    f"≤1 implementation (premature abstraction)"
                )
    return findings


def _audit_factory(node: ast.AST, filename: str) -> list[str]:
    """Factory with single product."""
    findings: list[str] = []
    if (
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and "factory" in node.name.lower()
    ):
        returns = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
        ]
        if len(returns) <= 1:
            findings.append(
                f"{filename}:{node.lineno} — '{node.name}' factory "
                f"creates ≤1 product (unnecessary indirection)"
            )
    return findings


def _audit_config_dict(node: ast.AST, filename: str) -> list[str]:
    """Config dict for a value that never changes."""
    findings: list[str] = []
    if isinstance(node, ast.Dict) and len(node.keys) <= 2:
        for key in node.keys:
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.lower() in {"timeout", "retries", "max_retries"}
            ):
                findings.append(
                    f"{filename}:{node.lineno} — config dict for "
                    f"'{key.value}' (hardcode or use kwargs)"
                )
    return findings


__all__ = ["PonytailStar", "PonytailState"]
