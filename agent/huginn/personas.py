"""Persona system for Huginn.

Inspired by AstrBot's persona/personality mechanism:
  - Each persona has a name, system prompt, and optional begin/mood dialogs.
  - A default persona is selected by name.
  - Personas can be loaded from and persisted to a JSON file.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huginn.prompts import HUGINN_SYSTEM_PROMPT, MATH_DEPTH_GUIDE

logger = logging.getLogger(__name__)



@dataclass
class Persona:
    """A character/personality configuration for Huginn.

    Field layering (SillyTavern-inspired):
    - permanent_core: identity, role, safety constraints — always injected.
      If empty, falls back to system_prompt (backward compat).
    - system_prompt: legacy field, holds the full prompt for old personas.
    - adaptive_layer: session-specific style/preferences — conditionally
      injected after the core. Populated by StyleLearner/TasteProfile.
    """

    name: str
    system_prompt: str = ""
    permanent_core: str = ""
    adaptive_layer: str = ""
    begin_dialogs: list[dict[str, str]] = field(default_factory=list)
    mood_dialogs: list[dict[str, str]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    avatar: str | None = None
    description: str = ""
    when_to_use: list[str] = field(default_factory=list)
    source_path: str | None = None
    kind: str = "json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Persona:
        return cls(
            name=data.get("name", "default"),
            system_prompt=data.get("system_prompt", data.get("prompt", "")),
            permanent_core=data.get("permanent_core", ""),
            adaptive_layer=data.get("adaptive_layer", ""),
            begin_dialogs=data.get("begin_dialogs", []),
            mood_dialogs=data.get("mood_dialogs", []),
            variables=data.get("variables", {}),
            avatar=data.get("avatar"),
            description=data.get("description", ""),
            when_to_use=data.get("when_to_use", []),
            source_path=data.get("source_path"),
            kind=data.get("kind", "json"),
        )


BUILT_IN_PERSONAS: list[Persona] = [
    Persona(
        name="default",
        system_prompt=HUGINN_SYSTEM_PROMPT,
    ),
    Persona(
        name="dft_expert",
        system_prompt=MATH_DEPTH_GUIDE + """
You are an expert in computational materials science with deep specialization in density functional theory (DFT).

When answering questions:
- Prefer first-principles methods and explain which exchange-correlation functional and pseudopotentials are appropriate.
- Give concrete VASP, Quantum ESPRESSO, or CP2K input examples when relevant.
- Discuss convergence with respect to plane-wave cutoff, k-point sampling, and total-energy thresholds.
- Interpret band structures, density of states, and structural relaxations critically.
- Mention known pitfalls (DFT band gap problem, dispersion corrections, spin states).""",
    ),
    Persona(
        name="md_expert",
        system_prompt="""You are an expert in atomistic molecular dynamics (MD) simulations for materials.

When answering questions:
- Recommend suitable force fields, interatomic potentials, or machine-learning potentials.
- Provide LAMMPS input script patterns and explain ensembles, thermostats, and barostats.
- Discuss equilibration, timestep choice, and trajectory analysis (RDF, MSD, viscosity, elastic constants).
- Link simulation setup to the material property the user wants to compute.""",
    ),
    Persona(
        name="reviewer",
        system_prompt=MATH_DEPTH_GUIDE + """
You are a critical peer reviewer for computational materials-science manuscripts and workflows.

When evaluating a method or result:
- Point out missing convergence tests, questionable approximations, or incomplete validation.
- Ask for uncertainty quantification, benchmarks against known references, and reproducibility details.
- Suggest stronger experimental or literature comparisons when appropriate.
- Be concise, direct, and constructive.""",
    ),
    Persona(
        name="tutor",
        system_prompt="""You are a patient tutor explaining computational materials science to a graduate student.

When answering questions:
- Break concepts into clear, logical steps.
- Use analogies and simple examples before diving into equations.
- Encourage the student to check convergence, validate against literature, and understand limitations.
- Keep a supportive, conversational tone.""",
    ),
    Persona(
        name="planner",
        system_prompt="""You are a research planner. Decompose the user's objective into a sequence of concrete, executable steps.

For each step specify:
- id: short identifier (s1, s2, ...)
- description: what to do, written so an executor agent can run it without further context
- tool: which tool to call (vasp_tool, lammps_tool, web_search, file_read_tool, ...) or null for a reasoning step
- parameters: dict of tool arguments, or {} if no tool
- dependencies: list of step ids that must complete first

Respond with ONLY a JSON object: {"steps": [...]}""",
    ),
    Persona(
        name="executor",
        system_prompt=MATH_DEPTH_GUIDE + """
You are a step executor. You receive one step from a confirmed plan.
Read the description and tool/parameters, execute precisely, and report the result.
Do not redesign the plan or skip steps. If a step fails, report the error clearly.""",
    ),
    Persona(
        name="research",
        system_prompt=(
            "You are a scientific research companion specialized in materials science.\n\n"
            "## Research Principles\n"
            "- Support the researcher's process: intuition -> analogy -> hypothesis -> formalization\n"
            "- Cite literature sources (DOI, title, authors, year) when making claims\n"
            "- Quantify uncertainty in computational results when possible\n"
            "- Compare results to published values; flag anomalies as potential discoveries\n"
            "- Different algorithms suit different data structures — there is no universal 'best'\n\n"
            "## Researcher's Intuition\n"
            "Researchers often start with a fuzzy intuition, a cross-domain analogy, or a\n"
            "technical preference before formalizing the problem. This exploratory phase is\n"
            "valuable and hard to evaluate with standard metrics — respect it. When the\n"
            "researcher provides intuition (via research_intuition), use it as a hint, not a\n"
            "constraint. The system's structure identification is advisory: 'exploratory' is a\n"
            "valid classification, not a failure state.\n\n"
            "## Deli Research Pipeline\n"
            "The Deli 9-stage pipeline is a flexible framework, not a rigid sequence:\n"
            "  1. Topic Analysis  -- extract research question and keywords\n"
            "  2. Literature Search -- retrieve and cluster relevant papers\n"
            "  3. Gap Analysis  -- identify unaddressed research gaps\n"
            "  4. Outline  -- design paper structure around gaps\n"
            "  5. Drafting  -- write sections in parallel, cite literature\n"
            "  6. Citation Verify -- anti-hallucination check on all references\n"
            "  7. Peer Review  -- EIC + expert reviewers + devil's advocate\n"
            "  8. Revision  -- address must-fix items from review\n"
            "  9. Final  -- compliance check and polishing\n\n"
            "## Computational Gaps\n"
            "When gap analysis reveals a gap that needs computational data "
            "(DFT, MD, FEM, etc.), suggest running simulation tools "
            "(vasp_tool, lammps_tool, cp2k_tool) to fill the gap with quantitative "
            "results. Prefer filling gaps over leaving them as 'future work' when "
            "tools are available, but respect the researcher's priorities.\n\n"
            "## Mathematical Depth\n"
            "Mathematical structure identification (PDE, variational, conservation, etc.)\n"
            "guides tool selection but does not constrain the research direction. When a\n"
            "structure is identified, use the MATH_DEPTH_GUIDE for tool suggestions. When\n"
            "the structure is 'exploratory' or 'none', data-driven methods (SR, GP, etc.)\n"
            "are equally valid paths — the choice depends on the data, not on hierarchy."
        )
        + MATH_DEPTH_GUIDE,
    ),
]


def _default_personas_path(workspace: str | Path | None = None) -> Path:
    """Default file for user-defined personas."""
    base = Path(workspace) if workspace else Path.cwd()
    return base / ".huginn" / "personas.json"


def _default_skill_dirs(workspace: str | Path | None = None) -> list[Path]:
    """Default directories to scan for Nuwa-style persona skills."""
    base = Path(workspace) if workspace else Path.cwd()
    return [base / ".huginn" / "personas"]


class PersonaManager:
    """Manage persona definitions: built-ins plus user-defined overrides and skills."""

    def __init__(
        self,
        personas_path: str | Path | None = None,
        default_persona: str = "default",
        skill_dirs: list[Path] | None = None,
        workspace: str | Path | None = None,
    ):
        self._path = (
            Path(personas_path) if personas_path else _default_personas_path(workspace)
        )
        self._default_name = default_persona
        self._skill_dirs = (
            skill_dirs if skill_dirs is not None else _default_skill_dirs(workspace)
        )
        self._workspace = Path(workspace) if workspace else Path.cwd()
        self._personas: dict[str, Persona] = {}
        self._load()

    def _load(self) -> None:
        """Load built-ins, JSON personas, and Nuwa-style skill personas."""
        from huginn.persona_loader import scan_persona_skills

        self._personas = {p.name: p for p in BUILT_IN_PERSONAS}
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                for entry in data.get("personas", []):
                    persona = Persona.from_dict(entry)
                    self._personas[persona.name] = persona
                if data.get("default_persona"):
                    self._default_name = data["default_persona"]
            except Exception:
                logger.debug("loads failed", exc_info=True)

        # Skill personas override JSON-defined personas of the same name.
        for name, persona in scan_persona_skills(*self._skill_dirs).items():
            self._personas[name] = persona

    def import_skill(
        self,
        source: Path,
        dest_dir: Path | None = None,
    ) -> Persona:
        """Copy a Nuwa-style SKILL.md into the persona store and load it."""
        from huginn.persona_loader import load_persona_skill

        dest_dir = dest_dir or self._skill_dirs[0]
        dest_dir.mkdir(parents=True, exist_ok=True)

        persona = load_persona_skill(source)
        dest_path = dest_dir / f"{persona.name}.md"
        dest_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

        persona.source_path = str(dest_path.resolve())
        self._personas[persona.name] = persona
        return persona

    def match_for_query(self, query: str, top_k: int = 1) -> list[Persona]:
        """Return skill personas whose description/when_to_use match the query.

        This method is kept for backward compatibility; new callers should use
        :class:`huginn.persona_matcher.PersonaMatcher` for semantic matching.
        """
        from huginn.persona_matcher import _keyword_score

        scored = [(_keyword_score(query, p), p) for p in self._personas.values()]
        scored.sort(key=lambda x: -x[0])
        return [p for score, p in scored[:top_k] if score > 0]

    def save(self) -> None:
        """Persist JSON-defined personas and default selection.

        Nuwa-style skill personas are file-backed and are not rewritten here.
        """
        builtin_names = {bp.name for bp in BUILT_IN_PERSONAS}
        data = {
            "default_persona": self._default_name,
            "personas": [
                p.to_dict()
                for p in self._personas.values()
                if p.name not in builtin_names and p.kind != "nuwa"
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def list(self) -> list[str]:
        return sorted(self._personas.keys())

    def get(self, name: str | None = None) -> Persona:
        name = name or self._default_name
        if name not in self._personas:
            name = "default"
        return self._personas[name]

    def get_default_name(self) -> str:
        return self._default_name

    def set_default(self, name: str) -> None:
        if name not in self._personas:
            raise ValueError(f"Persona '{name}' not found")
        self._default_name = name
        self.save()

    def create(
        self,
        name: str,
        system_prompt: str = "",
        begin_dialogs: list[dict[str, str]] | None = None,
        mood_dialogs: list[dict[str, str]] | None = None,
        variables: dict[str, Any] | None = None,
        avatar: str | None = None,
        description: str = "",
        when_to_use: list[str] | None = None,
    ) -> Persona:
        if not name:
            raise ValueError("Persona name is required")
        persona = Persona(
            name=name,
            system_prompt=system_prompt,
            begin_dialogs=begin_dialogs or [],
            mood_dialogs=mood_dialogs or [],
            variables=variables or {},
            avatar=avatar,
            description=description,
            when_to_use=when_to_use or [],
        )
        self._personas[name] = persona
        self.save()
        return persona

    def update(self, name: str, **kwargs: Any) -> Persona:
        if name not in self._personas:
            raise ValueError(f"Persona '{name}' not found")
        persona = self._personas[name]
        for key, value in kwargs.items():
            if hasattr(persona, key):
                setattr(persona, key, value)
        self._personas[name] = persona
        self.save()
        return persona

    def delete(self, name: str) -> None:
        if name in {bp.name for bp in BUILT_IN_PERSONAS}:
            raise ValueError(f"Cannot delete built-in persona '{name}'")
        if name not in self._personas:
            raise ValueError(f"Persona '{name}' not found")
        del self._personas[name]
        if self._default_name == name:
            self._default_name = "default"
        self.save()

    # ── 用户自设 persona (Nuwa 格式 .md 文件) ──────────────────────────

    def _persona_dir(self) -> Path:
        """用户 persona 文件存放目录 (.huginn/personas/)."""
        if self._skill_dirs:
            return self._skill_dirs[0]
        return self._workspace / ".huginn" / "personas"

    @staticmethod
    def _templates_path() -> Path:
        """persona 模板库 JSON 路径 (随包分发, 不随 workspace 变)."""
        return Path(__file__).resolve().parent / "data" / "persona_templates.json"

    @staticmethod
    def _render_template_string(template_str: str, values: dict[str, Any]) -> str:
        """把 {{var}} 占位符替换成 values 里的值, 未匹配的占位符原样保留."""
        def _replace(m: re.Match[str]) -> str:
            key = m.group(1).strip()
            if key in values:
                return str(values[key])
            return m.group(0)
        return re.sub(r"\{\{\s*(\w+)\s*\}\}", _replace, template_str)

    @staticmethod
    def _build_nuwa_markdown(
        name: str,
        description: str,
        when_to_use: list[str],
        system_prompt: str,
    ) -> str:
        """把 persona 字段拼成 Nuwa 格式 markdown (YAML frontmatter + body)."""
        import yaml

        front: dict[str, Any] = {"name": name}
        if description:
            front["description"] = description
        if when_to_use:
            front["when_to_use"] = when_to_use
        # sort_keys=False 保证 name/description/when_to_use 顺序稳定
        front_text = yaml.safe_dump(
            front, allow_unicode=True, sort_keys=False, default_flow_style=False
        )
        return f"---\n{front_text}---\n\n{system_prompt.strip()}\n"

    def create_persona(
        self,
        name: str,
        description: str,
        system_prompt: str,
        when_to_use: list[str] | None = None,
    ) -> Persona:
        """创建用户 persona 并写入 .huginn/personas/{name}.md (Nuwa 格式).

        内置 persona 同名时拒绝创建, 避免覆盖.
        """
        if not name:
            raise ValueError("Persona name is required")
        builtin_names = {bp.name for bp in BUILT_IN_PERSONAS}
        if name in builtin_names:
            raise ValueError(f"不能覆盖内置 persona '{name}'")

        when_to_use = when_to_use or []
        markdown = self._build_nuwa_markdown(
            name, description, when_to_use, system_prompt
        )

        persona_dir = self._persona_dir()
        persona_dir.mkdir(parents=True, exist_ok=True)
        file_path = persona_dir / f"{name}.md"
        file_path.write_text(markdown, encoding="utf-8")

        # 直接更新内存里的 persona, 不走全量 reload 以免丢掉其他运行时状态
        persona = Persona(
            name=name,
            system_prompt=system_prompt.strip(),
            description=description,
            when_to_use=when_to_use,
            source_path=str(file_path.resolve()),
            kind="nuwa",
        )
        self._personas[name] = persona
        return persona

    def update_persona(self, name: str, **fields: Any) -> Persona:
        """更新用户 persona 字段并写回 Nuwa 文件. 内置 persona 不允许改."""
        builtin_names = {bp.name for bp in BUILT_IN_PERSONAS}
        if name in builtin_names:
            raise ValueError(f"不能修改内置 persona '{name}'")
        if name not in self._personas:
            raise ValueError(f"Persona '{name}' not found")

        persona = self._personas[name]
        for key, value in fields.items():
            if hasattr(persona, key) and value is not None:
                setattr(persona, key, value)

        # 重写 Nuwa 文件
        file_path = self._persona_dir() / f"{name}.md"
        markdown = self._build_nuwa_markdown(
            persona.name,
            persona.description,
            list(persona.when_to_use),
            persona.system_prompt,
        )
        file_path.write_text(markdown, encoding="utf-8")
        persona.source_path = str(file_path.resolve())
        self._personas[name] = persona
        return persona

    def delete_persona(self, name: str) -> bool:
        """删除用户 persona 文件, 返回是否真的删了文件. 内置 persona 不可删."""
        builtin_names = {bp.name for bp in BUILT_IN_PERSONAS}
        if name in builtin_names:
            raise ValueError(f"不能删除内置 persona '{name}'")

        file_path = self._persona_dir() / f"{name}.md"
        deleted = False
        if file_path.exists():
            file_path.unlink()
            deleted = True
        # 内存里的也清掉
        self._personas.pop(name, None)
        # 默认 persona 被删了就回退到 default
        if self._default_name == name:
            self._default_name = "default"
            self.save()
        return deleted

    def list_templates(self) -> list[dict[str, Any]]:
        """列出 persona_templates.json 里的全部模板."""
        path = self._templates_path()
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return list(data.get("templates", []))
        except Exception:
            return []

    def get_template(self, template_name: str) -> dict[str, Any] | None:
        """按名字取单个模板, 找不到返回 None."""
        for tpl in self.list_templates():
            if tpl.get("name") == template_name:
                return tpl
        return None

    def instantiate_template(self, template_name: str, **overrides: Any) -> Persona:
        """从模板实例化 persona: 合并 default_values + overrides, 渲染占位符, 落盘."""
        tpl = self.get_template(template_name)
        if tpl is None:
            raise ValueError(f"模板 '{template_name}' 未找到")

        # 占位符取值: 模板默认值 < 调用方 overrides
        values: dict[str, Any] = dict(tpl.get("default_values", {}))
        values.update(overrides)

        prompt_template = tpl.get("system_prompt_template", "")
        system_prompt = self._render_template_string(prompt_template, values)

        # 名字默认用模板名, 允许 overrides 覆盖
        name = str(overrides.get("name", tpl.get("name", template_name)))
        description = str(overrides.get("description", tpl.get("description", "")))
        when_to_use = list(overrides.get("when_to_use", tpl.get("when_to_use", [])))

        return self.create_persona(
            name=name,
            description=description,
            system_prompt=system_prompt,
            when_to_use=when_to_use,
        )

    def export_persona(self, name: str) -> str:
        """把 persona 导出为 Nuwa 格式 markdown 字符串."""
        if name not in self._personas:
            raise ValueError(f"Persona '{name}' not found")
        persona = self._personas[name]

        # 用户 persona 直接读源文件, 保证 round-trip 一致
        if persona.source_path and Path(persona.source_path).exists():
            return Path(persona.source_path).read_text(encoding="utf-8")

        # 内置 persona 没有 .md 文件, 现场拼一份
        return self._build_nuwa_markdown(
            persona.name,
            persona.description,
            list(persona.when_to_use),
            persona.system_prompt,
        )

    def import_persona(self, markdown_text: str, overwrite: bool = False) -> Persona:
        """从 Nuwa 格式 markdown 文本导入 persona, 落盘到 .huginn/personas/."""
        from huginn.persona_loader import _split_frontmatter, load_persona_skill

        builtin_names = {bp.name for bp in BUILT_IN_PERSONAS}

        # 先解析出 name, 才能判断冲突
        meta, _body = _split_frontmatter(markdown_text)
        name = str(meta.get("name", "")).strip()
        if not name:
            raise ValueError("导入失败: markdown frontmatter 里缺少 name 字段")
        if name in builtin_names:
            raise ValueError(f"不能导入与内置 persona 同名的 '{name}'")

        dest_path = self._persona_dir() / f"{name}.md"
        if dest_path.exists() and not overwrite:
            raise ValueError(
                f"Persona '{name}' 已存在 ({dest_path}); 传 overwrite=True 覆盖"
            )

        self._persona_dir().mkdir(parents=True, exist_ok=True)
        dest_path.write_text(markdown_text, encoding="utf-8")

        # 用 load_persona_skill 解析落盘后的文件, 保证和 scan 路径一致
        persona = load_persona_skill(dest_path)
        self._personas[name] = persona
        return persona


# Backward-compatible flat mapping: name -> system prompt string.
def _personas_dict(manager: PersonaManager | None = None) -> dict[str, str]:
    mgr = manager or PersonaManager()
    return {name: mgr.get(name).system_prompt for name in mgr.list()}
