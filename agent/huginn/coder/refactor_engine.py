"""Refactor engine for planning and applying cross-file edits.

The engine builds a symbol index, gathers relevant source context, asks an LLM
for a minimal edit plan, and then applies the plan with exact-string matching
and snapshot-based rollback.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from huginn.coder.symbol_index import SymbolIndex
from huginn.config import HuginnConfig
from huginn.llm import get_model
from huginn.utils.tokens import rough_token_count


@dataclass
class PlannedEdit:
    """A single exact-string replacement in a file."""

    path: Path
    old_string: str
    new_string: str


class RefactorEngine:
    """Plan and execute cross-file refactorings safely."""

    MAX_FILES = 30
    MAX_CONTEXT_TOKENS = 8000

    def __init__(
        self,
        root: str | Path,
        config: HuginnConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.config = config or HuginnConfig.from_env()
        self.model = model or get_model(self.config)
        self.index = SymbolIndex(self.root)
        self.index.build()

    def plan(
        self,
        task: str,
        target_files: list[str] | None = None,
    ) -> list[PlannedEdit]:
        """Generate a refactor plan from a natural language task."""
        context = self._build_context(task, target_files)
        prompt = self._make_plan_prompt(task, context)

        response = self.model.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a careful code-refactoring assistant. "
                        "Respond only with a JSON object containing an "
                        "'edits' array. Each edit must have 'path', "
                        "'old_string', and 'new_string'. The old_string "
                        "must match exactly once in the file. Prefer minimal "
                        "changes."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        content = response.content if isinstance(response.content, str) else ""
        return self._parse_plan(content)

    def apply(
        self,
        edits: list[PlannedEdit],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Apply a plan and return results including a diff and snapshots."""
        snapshots: dict[str, str] = {}
        applied: list[PlannedEdit] = []
        errors: list[str] = []

        for edit in edits:
            abs_path = self.root / edit.path
            if not abs_path.exists():
                errors.append(f"File not found: {edit.path}")
                continue

            try:
                original = abs_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                errors.append(f"Could not read {edit.path}: {exc}")
                continue

            if str(abs_path) not in snapshots:
                snapshots[str(abs_path)] = original

            if edit.old_string not in original:
                errors.append(
                    f"old_string not found in {edit.path}: {edit.old_string[:80]!r}"
                )
                continue

            if original.count(edit.old_string) > 1:
                errors.append(
                    f"old_string occurs multiple times in {edit.path}: {edit.old_string[:80]!r}"
                )
                continue

            new_content = original.replace(edit.old_string, edit.new_string, 1)
            if not dry_run:
                abs_path.write_text(new_content, encoding="utf-8")
            applied.append(edit)

        return {
            "dry_run": dry_run,
            "applied": len(applied),
            "errors": errors,
            "snapshots": snapshots,
            "diff": self._render_diff(snapshots, edits),
        }

    def rollback(self, snapshots: dict[str, str]) -> None:
        """Restore files from snapshots captured during ``apply``."""
        for path_str, content in snapshots.items():
            Path(path_str).write_text(content, encoding="utf-8")

    def _build_context(
        self,
        task: str,
        target_files: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Collect file contents for the planning prompt."""
        if target_files:
            candidate_paths = [self.root / f for f in target_files]
        else:
            candidate_paths = [f.path for f in self.index._files]

        candidate_paths = [p for p in candidate_paths if p.exists()]
        # Prefer files that mention task keywords.
        keywords = {w.lower() for w in re.findall(r"[A-Za-z_]\w+", task)}
        scored: list[tuple[int, Path]] = []
        for path in candidate_paths:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            score = sum(1 for kw in keywords if kw in text.lower())
            scored.append((score, path))

        scored.sort(key=lambda x: x[0], reverse=True)
        paths = [p for _, p in scored[: self.MAX_FILES]]

        entries: list[dict[str, Any]] = []
        total_tokens = 0
        for path in paths:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = path.relative_to(self.root)
            tokens = rough_token_count(content)
            if total_tokens + tokens > self.MAX_CONTEXT_TOKENS:
                break
            total_tokens += tokens
            symbols = [
                {"name": s.name, "kind": s.kind, "line": s.line}
                for s in self.index.symbols()
                if s.file == str(path)
            ]
            entries.append(
                {
                    "path": str(rel),
                    "symbols": symbols,
                    "content": content,
                }
            )

        return entries

    def _make_plan_prompt(
        self,
        task: str,
        context: list[dict[str, Any]],
    ) -> str:
        lines = [
            "Task:",
            task,
            "",
            "Workspace source files:",
        ]
        for entry in context:
            lines.append(f"\n--- {entry['path']} ---")
            for sym in entry["symbols"]:
                lines.append(f"# {sym['kind']} {sym['name']} (line {sym['line']})")
            lines.append(entry["content"])
        lines.append(
            "\nReturn only JSON in this exact shape:\n"
            '{"edits": [{"path": "relative/path.py", '
            '"old_string": "exact existing text", '
            '"new_string": "replacement text"}]}'
        )
        return "\n".join(lines)

    def _parse_plan(self, content: str) -> list[PlannedEdit]:
        """Extract JSON edit plan from model output."""
        # Strip markdown fences if present.
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.M)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse refactor plan as JSON: {exc}") from exc

        edits: list[PlannedEdit] = []
        for item in data.get("edits", []):
            path = Path(item["path"])
            edits.append(
                PlannedEdit(
                    path=path,
                    old_string=item["old_string"],
                    new_string=item["new_string"],
                )
            )
        return edits

    def _render_diff(
        self,
        snapshots: dict[str, str],
        edits: list[PlannedEdit],
    ) -> str:
        """Render a unified diff for all files touched by the plan."""
        diff_parts: list[str] = []
        for path_str, original in snapshots.items():
            path = Path(path_str)
            new = original
            for edit in edits:
                edit_abs = self.root / edit.path
                if edit_abs == path and edit.old_string in new:
                    new = new.replace(edit.old_string, edit.new_string, 1)
            rel = path.relative_to(self.root)
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=str(rel),
                tofile=str(rel),
            )
            diff_parts.extend(diff)
        return "".join(diff_parts)
