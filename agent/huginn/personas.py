"""Persona system for Huginn.

Inspired by AstrBot's persona/personality mechanism:
  - Each persona has a name, system prompt, and optional begin/mood dialogs.
  - A default persona is selected by name.
  - Personas can be loaded from and persisted to a JSON file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from huginn.prompts import HUGINN_SYSTEM_PROMPT


@dataclass
class Persona:
    """A character/personality configuration for Huginn."""

    name: str
    system_prompt: str = ""
    begin_dialogs: list[dict[str, str]] = field(default_factory=list)
    mood_dialogs: list[dict[str, str]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    avatar: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        return cls(
            name=data.get("name", "default"),
            system_prompt=data.get("system_prompt", data.get("prompt", "")),
            begin_dialogs=data.get("begin_dialogs", []),
            mood_dialogs=data.get("mood_dialogs", []),
            variables=data.get("variables", {}),
            avatar=data.get("avatar"),
        )


BUILT_IN_PERSONAS: list[Persona] = [
    Persona(
        name="default",
        system_prompt=HUGINN_SYSTEM_PROMPT,
    ),
    Persona(
        name="dft_expert",
        system_prompt="""You are an expert in computational materials science with deep specialization in density functional theory (DFT).

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
        system_prompt="""You are a critical peer reviewer for computational materials-science manuscripts and workflows.

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
]


def _default_personas_path() -> Path:
    """Default file for user-defined personas."""
    return Path.cwd() / "personas.json"


class PersonaManager:
    """Manage persona definitions: built-ins plus user-defined overrides."""

    def __init__(
        self,
        personas_path: str | Path | None = None,
        default_persona: str = "default",
    ):
        self._path = Path(personas_path) if personas_path else _default_personas_path()
        self._default_name = default_persona
        self._personas: dict[str, Persona] = {}
        self._load()

    def _load(self) -> None:
        """Load built-ins and user-defined personas."""
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
                pass

    def save(self) -> None:
        """Persist user-defined personas and default selection."""
        data = {
            "default_persona": self._default_name,
            "personas": [
                p.to_dict() for p in self._personas.values()
                if p.name not in {bp.name for bp in BUILT_IN_PERSONAS}
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

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


# Backward-compatible flat mapping: name -> system prompt string.
def _personas_dict(manager: PersonaManager | None = None) -> dict[str, str]:
    mgr = manager or PersonaManager()
    return {name: mgr.get(name).system_prompt for name in mgr.list()}


PERSONAS: dict[str, str] = _personas_dict()
