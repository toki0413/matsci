"""Bridge to google-deepmind/science-skills plugin.

Auto-discovers 37 scientific database/literature skills from the cloned
``science-skills`` repository and registers each as a ``HuginnTool``.
Execution delegates to the original Python CLI scripts via ``uv run``.

Architecture::

    ScienceSkillsLoader  → scans skills/ for SKILL.md + scripts/*.py
    ScienceSkillTool     → one HuginnTool instance per discovered skill
    ScienceSkillInput    → generic Pydantic model (query/action/identifiers/…)
    register_science_skills()  → batch-registers all into ToolRegistry

Requirements:
- ``uv`` package manager on PATH (https://astral.sh/uv)
- science-skills repo cloned at ``SCIENCE_SKILLS_DIR`` (env var) or the
  bundled ``agent/huginn/plugins/science-skills/`` location.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext, ToolResult

# 复用 skill_loader 的 frontmatter 解析和条件激活引擎
from huginn.plugins.skill_loader import (
    activate_conditional_skills,
    parse_skill_file,
    register_conditional_skills,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BUNDLED_DIR = Path(__file__).parent / "science-skills"
_SKIP_DIRS = {"scienceskillscommon", "uv", "workflow_skill_creator", "__pycache__"}


def _resolve_skills_dir() -> Path | None:
    """Locate the science-skills ``skills/`` directory."""
    env_dir = os.environ.get("SCIENCE_SKILLS_DIR")
    if env_dir:
        p = Path(env_dir) / "skills"
        if p.is_dir():
            return p

    bundled = _BUNDLED_DIR / "skills"
    if bundled.is_dir():
        return bundled

    return None


# ---------------------------------------------------------------------------
# SKILL.md parser
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


@dataclass
class SkillMetadata:
    """Parsed metadata from a single skill directory."""

    name: str                       # kebab-case from frontmatter
    description: str                # from frontmatter
    directory: Path                 # absolute path to skill dir
    scripts: list[Path] = field(default_factory=list)
    primary_script: Path | None = None

    @property
    def tool_name(self) -> str:
        """Huginn tool name: ``science_{underscore_name}``."""
        return "science_" + self.name.replace("-", "_")


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Extract YAML frontmatter fields (name, description) without PyYAML."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    block = m.group(1)
    result: dict[str, str] = {}
    current_key = ""
    current_val_lines: list[str] = []

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # New key: value pair
        if ":" in stripped and not stripped.startswith(" "):
            if current_key:
                result[current_key] = " ".join(current_val_lines).strip()
            key, _, val = stripped.partition(":")
            current_key = key.strip()
            val = val.strip()
            if val and val != ">":
                current_val_lines = [val]
            else:
                current_val_lines = []
        else:
            current_val_lines.append(stripped)

    if current_key:
        result[current_key] = " ".join(current_val_lines).strip()

    return result


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ScienceSkillsLoader:
    """Discover and load metadata from science-skills ``skills/`` directory."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir = skills_dir or _resolve_skills_dir()

    def discover(self) -> list[SkillMetadata]:
        """Scan skills/ and return metadata for each valid skill."""
        if self.skills_dir is None or not self.skills_dir.is_dir():
            return []

        skills: list[SkillMetadata] = []
        for entry in sorted(self.skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in _SKIP_DIRS:
                continue

            skill_md = entry / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                # 用 skill_loader.parse_skill_file 统一解析 frontmatter,
                # 支持 paths/allowed_tools/model/effort 等扩展字段
                parsed = parse_skill_file(skill_md)
            except OSError:
                continue
            except Exception as exc:
                logger.warning("解析 SKILL.md 失败 %s: %s", skill_md, exc)
                continue

            name = parsed.get("name", entry.name)
            description = parsed.get("description", f"Science skill: {entry.name}")

            # Discover scripts
            scripts_dir = entry / "scripts"
            scripts: list[Path] = []
            if scripts_dir.is_dir():
                scripts = sorted(scripts_dir.glob("*.py"))

            if not scripts:
                continue  # No executable scripts → skip

            meta = SkillMetadata(
                name=name,
                description=description,
                directory=entry,
                scripts=scripts,
                primary_script=scripts[0],
            )
            skills.append(meta)

        return skills


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

class ScienceSkillInput(BaseModel):
    """Generic input for any science-skills bridge tool."""

    query: str | None = Field(
        default=None,
        description="Search term, gene/protein/chemical name, or identifier",
    )
    action: str | None = Field(
        default=None,
        description="Sub-command to run (e.g. resolve, search, summary, filter, view, properties)",
    )
    identifiers: list[str] | None = Field(
        default=None,
        description="List of IDs (e.g. CIDs, gene names, variant IDs) for batch operations",
    )
    extra_args: dict[str, str] | None = Field(
        default=None,
        description="Additional CLI flags as key-value pairs (e.g. {'--smiles': 'CCO', '--limit': '10'})",
    )
    output_file: str | None = Field(
        default=None,
        description="Output file path (auto-generated if omitted)",
    )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class ScienceSkillTool(HuginnTool):
    """A single science-skills skill exposed as a HuginnTool.

    Each instance wraps one skill directory and delegates execution to its
    primary script via ``uv run``.
    """

    destructive: bool = False
    read_only: bool = True
    input_schema: type = ScienceSkillInput

    def __init__(self, meta: SkillMetadata, **kwargs: Any) -> None:
        self.name = meta.tool_name
        self.description = (
            f"[science-skills] {meta.description}\n\n"
            f"Skill: {meta.name} | Script: {meta.primary_script.name if meta.primary_script else 'N/A'}"
        )
        self._meta = meta
        self._skills_root = meta.directory.parent  # science-skills/skills/

    def _build_command(
        self,
        args: ScienceSkillInput,
        output_path: Path,
        uv_exe: str,
    ) -> list[str]:
        """Build the ``uv run`` command line."""
        script = self._meta.primary_script
        if script is None:
            raise RuntimeError(f"No script found for skill '{self._meta.name}'")

        cmd = [uv_exe, "run", str(script)]

        # Action / sub-command
        if args.action:
            cmd.append(args.action)

        # Query → positional or --name/--query flag
        if args.query:
            cmd.extend(["--query", args.query])

        # Identifiers
        if args.identifiers:
            for ident in args.identifiers:
                cmd.extend(["--id", ident])

        # Output
        cmd.extend(["--output", str(output_path)])

        # Extra args
        if args.extra_args:
            for k, v in args.extra_args.items():
                flag = k if k.startswith("-") else f"--{k}"
                if v.lower() in ("true", "yes", ""):
                    cmd.append(flag)
                else:
                    cmd.extend([flag, v])

        return cmd

    async def call(
        self, args: ScienceSkillInput, context: ToolContext
    ) -> ToolResult:
        """Execute the skill's CLI script via ``uv run``."""

        # Locate uv
        uv_exe = shutil.which("uv")
        if uv_exe is None:
            return ToolResult(
                data=None,
                success=False,
                error=(
                    "uv package manager not found. Install it from "
                    "https://astral.sh/uv or run: "
                    "curl -LsSf https://astral.sh/uv/install.sh | sh"
                ),
            )

        # Prepare output path
        # Use TemporaryDirectory for auto-cleanup when no explicit output_file
        _tmp_cleanup = None
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            _tmp_dir = tempfile.TemporaryDirectory(prefix="science_skill_")
            _tmp_cleanup = _tmp_dir
            output_path = Path(_tmp_dir.name) / "output.json"

        # Build command
        try:
            cmd = self._build_command(args, output_path, uv_exe)
        except RuntimeError as exc:
            if _tmp_cleanup:
                _tmp_cleanup.cleanup()
            return ToolResult(data=None, success=False, error=str(exc))

        # Execute via subprocess (in executor to avoid blocking)
        cwd = str(self._skills_root.parent)  # science-skills root for relative imports
        timeout = 120  # 2 minutes per query

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: _run_subprocess(cmd, cwd, timeout),
            )
        except asyncio.TimeoutError:
            if _tmp_cleanup:
                _tmp_cleanup.cleanup()
            return ToolResult(
                data=None,
                success=False,
                error=f"Skill '{self._meta.name}' timed out after {timeout}s",
            )
        except Exception as exc:
            if _tmp_cleanup:
                _tmp_cleanup.cleanup()
            return ToolResult(
                data=None,
                success=False,
                error=f"Execution error: {exc}",
            )

        if result["returncode"] != 0:
            stderr = result.get("stderr", "")
            if _tmp_cleanup:
                _tmp_cleanup.cleanup()
            return ToolResult(
                data=None,
                success=False,
                error=f"Script failed (rc={result['returncode']}): {stderr[:1000]}",
            )

        # Read output file before temp directory is cleaned up
        output_data: Any = None
        output_file_str: str | None = None
        if output_path.exists():
            output_file_str = str(output_path)
            try:
                text = output_path.read_text(encoding="utf-8")
                try:
                    output_data = json.loads(text)
                except json.JSONDecodeError:
                    output_data = text
            except OSError as exc:
                output_data = f"Could not read output: {exc}"
        elif result.get("stdout"):
            output_data = result["stdout"]

        # Clean up temp directory if we created one
        if _tmp_cleanup:
            _tmp_cleanup.cleanup()

        return ToolResult(
            data={
                "skill": self._meta.name,
                "action": args.action,
                "output_file": output_file_str,
                "result": output_data,
            },
            success=True,
        )


def _run_subprocess(
    cmd: list[str], cwd: str, timeout: int
) -> dict[str, Any]:
    """Synchronous subprocess runner (called from executor)."""
    import subprocess

    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": "Process timed out"}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


# ---------------------------------------------------------------------------
# Batch registration
# ---------------------------------------------------------------------------

_registered = False


def register_science_skills() -> list[str]:
    """Discover all science-skills and register them in ToolRegistry.

    Returns the list of registered tool names.  Idempotent — calling
    multiple times only registers once.
    """
    global _registered

    # Idempotency: if already registered and still present in registry
    if _registered:
        existing = [t for t in ToolRegistry.list_tools() if t.startswith("science_")]
        if existing:
            return existing
        # Registry was cleared externally — re-register
        _registered = False

    loader = ScienceSkillsLoader()
    skills = loader.discover()

    if not skills:
        logger.warning(
            "No science-skills found. Clone the repo or set SCIENCE_SKILLS_DIR."
        )
        return []

    registered: list[str] = []
    for meta in skills:
        try:
            tool = ScienceSkillTool(meta)
            ToolRegistry.register(tool)
            registered.append(tool.name)
        except Exception as exc:
            logger.warning("Failed to register science skill '%s': %s", meta.name, exc)

    # 用 skill_loader 注册带 paths 字段的条件技能, 后续工具调用时
    # activate_conditional_skills 会按文件路径自动激活匹配的技能
    try:
        skills_dir = loader.skills_dir
        if skills_dir:
            from huginn.plugins.skill_loader import load_skills_from_dir

            all_parsed = load_skills_from_dir(skills_dir)
            register_conditional_skills(all_parsed)
    except Exception as exc:
        logger.debug("条件技能注册失败(非致命): %s", exc)

    _registered = True
    logger.info("Registered %d science-skills tools: %s", len(registered), registered)
    return registered


def get_science_skills_info() -> list[dict[str, Any]]:
    """Return metadata about all discovered science skills (for API listing)."""
    loader = ScienceSkillsLoader()
    skills = loader.discover()
    return [
        {
            "name": meta.name,
            "tool_name": meta.tool_name,
            "description": meta.description,
            "directory": str(meta.directory),
            "scripts": [str(s) for s in meta.scripts],
            "primary_script": str(meta.primary_script) if meta.primary_script else None,
        }
        for meta in skills
    ]
