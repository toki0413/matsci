"""Load Nuwa-style persona skills from markdown SKILL.md files.

A persona skill is a markdown file with YAML frontmatter such as:

---
name: paul-graham
description: Startup and writing perspective
when_to_use:
  - evaluating startup ideas
  - writing clearly
---

# Paul Graham

## Who Paul Graham Is
...

The frontmatter drives registration; the markdown body becomes the system prompt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huginn.personas import Persona


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter from markdown body."""
    if not text.startswith("---"):
        return {}, text

    # Find the closing --- after the first line.
    match = re.search(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return {}, text

    front_text = match.group(1)
    body = text[match.end() :]

    try:
        import yaml

        meta = yaml.safe_load(front_text) or {}
    except Exception:
        meta = {}

    return meta, body


def load_persona_skill(path: Path) -> Persona:
    """Parse a Nuwa-style SKILL.md into a Huginn Persona."""
    from huginn.personas import Persona

    text = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(text)

    name = str(meta.get("name") or path.stem).strip()
    description = str(meta.get("description", "")).strip()
    when_to_use = meta.get("when_to_use") or []
    if isinstance(when_to_use, str):
        when_to_use = [when_to_use]
    when_to_use = [str(item).strip() for item in when_to_use if str(item).strip()]

    # Trim excessive whitespace from the body while preserving structure.
    system_prompt = re.sub(r"\n{3,}", "\n\n", body).strip()

    # If the body is empty, build a minimal prompt from metadata so the
    # persona is still usable.
    if not system_prompt:
        system_prompt = f"You are the {name} persona."
        if description:
            system_prompt += f" {description}"

    return Persona(
        name=name,
        system_prompt=system_prompt,
        description=description,
        when_to_use=when_to_use,
        source_path=str(path.resolve()),
        kind="nuwa",
    )


def scan_persona_skills(*directories: Path) -> dict[str, Persona]:
    """Scan directories for *.md persona skills and return a name->Persona map."""
    found: dict[str, Persona] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            try:
                persona = load_persona_skill(path)
                found[persona.name] = persona
            except Exception:
                # Skip malformed skill files rather than breaking startup.
                continue
    return found
