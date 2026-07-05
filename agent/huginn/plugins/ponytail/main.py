"""Ponytail plugin — enforces a lazy-engineer decision ladder.

Integrates the ponytail behavior-constraint rules into Huginn's LLM
pipeline.  The core mechanism is a 7-rung ladder that the LLM must
climb before writing any code: YAGNI → stdlib → platform → existing dep
→ one-liner → minimal new code.  This eliminates over-engineering at
the prompt level rather than post-hoc review.

Modes:
  lite  — build what's asked, name the lazier alternative
  full  — ladder enforced, stdlib/native first (default)
  ultra — YAGNI extremist, deletion before addition
  off   — no injection

Commands:
  /ponytail [lite|full|ultra|off]  — switch mode
  /ponytail-review                  — review last diff for over-engineering
  /ponytail-audit                   — scan workspace for over-engineering
  /ponytail-gain                    — show cumulative savings
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.api.context import PluginContext
from huginn.api.event import Event, LLMRequestEvent
from huginn.api.filter import filter
from huginn.api.star import Star

# ── Rule text ────────────────────────────────────────────────────────
# Adapted from DietrichGebert/ponytail SKILL.md (MIT license).
# Trimmed to the essential ladder + rules + safety boundaries so it
# stays under ~800 tokens in the system prompt.

_RULES_LITE = """\
## Ponytail (lite)

You are a lazy senior developer. Lazy means efficient, not careless.
Before writing code, consider whether a simpler solution exists:
stdlib, native platform feature, or existing dependency. If you built
something complex, name the lazier alternative in one line. User picks.
"""

_RULES_FULL = """\
## Ponytail (full)

You are a lazy senior developer. Lazy means efficient, not careless.
The best code is the code never written.

### The Ladder
Stop at the first rung that holds:
1. Does this need to exist? Speculative need = skip it (YAGNI).
2. Stdlib does it? Use it.
3. Native platform feature covers it? Use it.
4. Already-installed dependency solves it? Use it. Never add new deps.
5. Can it be one line? One line.
6. Only then: the minimum code that works.

### Rules
- No unrequested abstractions (interface with one impl, factory for one product).
- Deletion over addition. Boring over clever.
- Fewest files possible. Shortest working diff wins.
- Mark deliberate simplifications with `# ponytail:` comment.
- Non-trivial logic leaves ONE runnable check behind (assert or test_*.py).

### Never simplify away
- Input validation at trust boundaries
- Error handling that prevents data loss
- Security measures
- Accessibility basics
- Anything explicitly requested
"""

_RULES_ULTRA = """\
## Ponytail (ultra)

You are a lazy senior developer in YAGNI extremist mode.
Deletion before addition. Ship the one-liner and challenge the rest.

### The Ladder
1. Does this need to exist? Probably not. Say so.
2. Stdlib. Always.
3. One line if possible.
4. If code is unavoidable: the absolute minimum.

### Rules
- No abstractions, ever, unless there are 3+ implementations TODAY.
- No config files for values that never change.
- No scaffolding "for later". Later can scaffold for itself.
- Ship the lazy version, then challenge the requirement in the same breath.
- Mark simplifications with `# ponytail:` comment naming the ceiling.

### Never simplify away
- Input validation, error handling, security, accessibility.
"""

_MODE_MAP: dict[str, str] = {
    "lite": _RULES_LITE,
    "full": _RULES_FULL,
    "ultra": _RULES_ULTRA,
}


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
    """Ponytail behavior-constraint plugin.

    Hooks into ON_LLM_REQUEST to inject the decision-ladder rules into
    the system prompt.  Mode is persisted in plugin storage so it
    survives across sessions.
    """

    name = "ponytail"
    author = "Huginn Integration"
    version = "1.0.0"
    description = "Enforces a lazy-engineer decision ladder to prevent over-engineering"
    priority = 50  # Run before most other LLM request handlers

    def __init__(self, context: PluginContext | None = None) -> None:
        super().__init__(context)
        self._state = PonytailState()

    async def on_load(self) -> None:
        # Restore mode from storage if available
        try:
            stored = self.storage.get("ponytail_state")
            if stored and isinstance(stored, dict):
                self._state = PonytailState(**{
                    k: v for k, v in stored.items()
                    if k in {"mode", "skipped_count", "lines_saved", "history"}
                })
        except Exception:
            # Storage might not be injected in test mode — fine
            pass
        self.logger.info("ponytail loaded (mode=%s)", self._state.mode)

    async def on_unload(self) -> None:
        self._persist()

    def _persist(self) -> None:
        try:
            self.storage.set("ponytail_state", {
                "mode": self._state.mode,
                "skipped_count": self._state.skipped_count,
                "lines_saved": self._state.lines_saved,
                "history": self._state.history[-20:],  # cap history
            })
        except Exception:
            pass

    # ── LLM prompt injection ──────────────────────────────────────────

    @filter.on_llm_request()
    async def inject_rules(self, event: LLMRequestEvent) -> None:
        """Inject ponytail rules into the system prompt based on current mode."""
        if self._state.mode == "off":
            return
        rules = _MODE_MAP.get(self._state.mode, _RULES_FULL)
        # Append rules to the existing system prompt
        if event.system_prompt:
            event.system_prompt = event.system_prompt + "\n\n" + rules
        else:
            event.system_prompt = rules
        # Tag context so downstream handlers know ponytail is active
        event.context["ponytail_mode"] = self._state.mode

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
        if new_mode not in _MODE_MAP and new_mode != "off":
            self._set_reply(event, f"Unknown mode '{new_mode}'. Use: lite, full, ultra, off")
            return

        old_mode = self._state.mode
        self._state.mode = new_mode
        self._state.history.append({
            "from": old_mode,
            "to": new_mode,
            "timestamp": datetime.datetime.now().isoformat(),
        })
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
                continue

        if not findings:
            self._set_reply(event, "Ponytail review: no over-engineering found in recent files. ✅")
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
            f for f in py_files
            if not any(p in str(f) for p in ("venv", "site-packages", "__pycache__", "test_"))
        ]

        all_findings: list[str] = []
        for f in py_files[:50]:
            try:
                content = f.read_text(encoding="utf-8")
                tree = ast.parse(content)
                issues = _audit_python_tree(tree, str(f.relative_to(workspace)))
                all_findings.extend(issues)
            except Exception:
                continue

        if not all_findings:
            self._set_reply(event, f"Ponytail audit: scanned {len(py_files)} files, no over-engineering found. ✅")
        else:
            lines = [f"Ponytail audit: {len(all_findings)} issue(s) in {len(py_files)} files:"]
            for issue in all_findings[:15]:
                lines.append(f"  ⚠ {issue}")
            if len(all_findings) > 15:
                lines.append(f"  ... and {len(all_findings) - 15} more")
            self._set_reply(event, "\n".join(lines))

    @filter.command("/ponytail-gain")
    @filter.event_message_type()
    async def handle_gain(self, event: Event) -> None:
        """Show cumulative ponytail savings."""
        self._set_reply(event, (
            f"Ponytail cumulative savings:\n"
            f"  Mode: {self._state.mode}\n"
            f"  Over-engineering prevented: {self._state.skipped_count}\n"
            f"  Estimated lines saved: ~{self._state.lines_saved}\n"
            f"  Mode changes: {len(self._state.history)}"
        ))

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
        # Interface with single implementation
        if isinstance(node, ast.ClassDef):
            bases = [getattr(b, "id", "") for b in node.bases if isinstance(b, ast.Name)]
            if "ABC" in bases:
                # Check if there's only one concrete subclass in the same file
                concrete = [
                    n for n in ast.walk(tree)
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

        # Factory with single product
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if "factory" in node.name.lower():
                returns = [
                    n for n in ast.walk(node)
                    if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
                ]
                if len(returns) <= 1:
                    findings.append(
                        f"{filename}:{node.lineno} — '{node.name}' factory "
                        f"creates ≤1 product (unnecessary indirection)"
                    )

        # Config dict for a value that never changes
        if isinstance(node, ast.Dict) and len(node.keys) <= 2:
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    if key.value.lower() in {"timeout", "retries", "max_retries"}:
                        findings.append(
                            f"{filename}:{node.lineno} — config dict for "
                            f"'{key.value}' (hardcode or use kwargs)"
                        )

    return findings


__all__ = ["PonytailStar", "PonytailState"]
