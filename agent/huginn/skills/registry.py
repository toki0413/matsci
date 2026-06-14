"""Skill registry — centralized discovery and registration of material science skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huginn.skills.base import SkillDefinition


class SkillRegistry:
    """Registry for all available skills."""

    _skills: dict[str, SkillDefinition] = {}

    @classmethod
    def register(cls, skill: SkillDefinition) -> SkillDefinition:
        if not skill.name:
            raise ValueError("Skill must have a name")
        cls._skills[skill.name] = skill
        return skill

    @classmethod
    def get(cls, name: str) -> SkillDefinition | None:
        return cls._skills.get(name)

    @classmethod
    def list_skills(cls, category: str | None = None) -> list[str]:
        if category:
            return [n for n, s in cls._skills.items() if s.category == category]
        return list(cls._skills.keys())

    @classmethod
    def get_by_category(cls, category: str) -> list[SkillDefinition]:
        return [s for s in cls._skills.values() if s.category == category]

    @classmethod
    def get_all_definitions(cls) -> list[SkillDefinition]:
        return list(cls._skills.values())

    @classmethod
    def search(cls, query: str) -> list[SkillDefinition]:
        """Fuzzy search skills by name, description, or tags."""
        query = query.lower()
        results = []
        for skill in cls._skills.values():
            if (
                query in skill.name.lower()
                or query in skill.description.lower()
                or any(query in t.lower() for t in skill.tags)
            ):
                results.append(skill)
        return results

    @classmethod
    def clear(cls) -> None:
        cls._skills.clear()


def register_skill(skill: SkillDefinition) -> SkillDefinition:
    """Decorator-style registration."""
    return SkillRegistry.register(skill)
