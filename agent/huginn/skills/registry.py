"""Skill registry — centralized discovery and registration of material science skills."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huginn.skills.base import SkillDefinition


class SkillValidationError(ValueError):
    """skill 注册前的静态校验失败."""


# 可疑 secret key 名 (api_key / password / secret / token 的各种变体)
_SECRET_KEY_RE = re.compile(
    r"(?:^|[_\-\s])(api[_-]?key|password|secret|token|passwd|access[_-]?key)(?:[_\-\s]|$)",
    re.IGNORECASE,
)
# 占位值, 不算真 secret (example / placeholder / changeme / <...> / env var 名)
_PLACEHOLDER_RE = re.compile(
    r"^(?:(?:example|placeholder|changeme|your[_\- ]?[\w]*)|<[^>]+>|[A-Z][A-Z0-9_]*$)",
    re.IGNORECASE,
)
# value 内联 secret 赋值, 如 "api_key: sk-xxxxxx" / "password = abc123"
_INLINE_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"]"
    r"(?!\s*(?:example|placeholder|changeme|your[-_ ]|<))[^'\"\s]{8,}['\"]"
)


def _find_secret_in_metadata(metadata: dict[str, Any]) -> str | None:
    """扫 metadata 里有没有可疑 secret. 命中返回 'metadata.<key>', 否则 None.

    两条路: key 名像 secret 字段且 value 非占位 / value 内联了 secret 赋值.
    ponytail: 不做真值检测 (entropy / 字符分布), 只看模式. 升级路径: detect-secrets.
    """
    for k, v in metadata.items():
        if not isinstance(k, str):
            continue
        if _SECRET_KEY_RE.search(k) and isinstance(v, str):
            val = v.strip()
            if val and not _PLACEHOLDER_RE.match(val):
                return f"metadata.{k}"
        if isinstance(v, str) and _INLINE_SECRET_RE.search(v):
            return f"metadata.{k} (inline)"
    return None


def _validate_skill(skill: "SkillDefinition") -> None:
    """注册前静态校验. 失败抛 SkillValidationError.

    只做 stdlib 能做的: 必填字段非空 + metadata secret 扫描.
    ponytail: 不做 SHA-256 manifest / file_assertions / tool_assertions,
    那些是外部 skill 评测才需要的. 升级路径: 接 skill_validator.py.
    """
    name = getattr(skill, "name", "")
    if not isinstance(name, str) or not name.strip():
        raise SkillValidationError("skill.name 不能为空")

    desc = getattr(skill, "description", "")
    if not isinstance(desc, str) or not desc.strip():
        raise SkillValidationError(f"skill {name!r}: description 不能为空")

    steps = getattr(skill, "steps", None) or []
    for i, step in enumerate(steps):
        tool = str(getattr(step, "tool", "") or "").strip()
        if not tool:
            raise SkillValidationError(
                f"skill {name!r}: steps[{i}].tool 不能为空"
            )

    params = getattr(skill, "parameters", None) or []
    for i, p in enumerate(params):
        if not str(getattr(p, "name", "") or "").strip():
            raise SkillValidationError(
                f"skill {name!r}: parameters[{i}].name 不能为空"
            )
        if not str(getattr(p, "type", "") or "").strip():
            raise SkillValidationError(
                f"skill {name!r}: parameters[{i}].type 不能为空"
            )

    meta = getattr(skill, "metadata", None) or {}
    if isinstance(meta, dict):
        leak = _find_secret_in_metadata(meta)
        if leak:
            raise SkillValidationError(
                f"skill {name!r}: {leak} 含可疑 secret, 拒绝注册"
            )


class SkillRegistry:
    """Registry for all available skills."""

    _skills: dict[str, SkillDefinition] = {}

    @classmethod
    def register(cls, skill: SkillDefinition) -> SkillDefinition:
        _validate_skill(skill)
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
