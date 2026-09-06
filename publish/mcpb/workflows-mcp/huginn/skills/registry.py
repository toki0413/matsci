"""Skill registry — centralized discovery and registration of material science skills."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from huginn.skills.base import SkillDefinition

logger = logging.getLogger(__name__)


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


def _validate_skill(skill: SkillDefinition) -> None:
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
        ensure_presets()
        return cls._skills.get(name)

    @classmethod
    def list_skills(cls, category: str | None = None) -> list[str]:
        ensure_presets()
        if category:
            return [n for n, s in cls._skills.items() if s.category == category]
        return list(cls._skills.keys())

    @classmethod
    def get_by_category(cls, category: str) -> list[SkillDefinition]:
        ensure_presets()
        return [s for s in cls._skills.values() if s.category == category]

    @classmethod
    def get_all_definitions(cls) -> list[SkillDefinition]:
        ensure_presets()
        return list(cls._skills.values())

    @classmethod
    def record_invocation(cls, name: str, success: bool) -> None:
        """运行时记录一次技能调用, 更新 metadata['evolution'] 的复用/成败计数.

        evolution 引擎 (SkillEvolutionLayer / EvolutionEngine) 用这些统计驱动
        元规则: 高复用低成本 → 提升为基元, 长时间零复用/高失败 → 淘汰.
        未注册的技能 / 无 metadata 的静默跳过, 不抛错.
        """
        skill = cls._skills.get(name)
        if skill is None:
            return
        meta = skill.metadata.setdefault("evolution", {})
        meta["usage_count"] = int(meta.get("usage_count", 0)) + 1
        if success:
            meta["success_count"] = int(meta.get("success_count", 0)) + 1

    @classmethod
    def search(cls, query: str) -> list[SkillDefinition]:
        """Fuzzy search skills by name, description, or tags."""
        ensure_presets()
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

    # ---- 技能树查询 (parent/children 层级) ----
    @classmethod
    def children(cls, name: str) -> list[str]:
        """返回 skill 的直接子技能名 (parent == name)."""
        ensure_presets()
        return sorted(
            s.name for s in cls._skills.values() if s.parent == name
        )

    @classmethod
    def descendants(cls, name: str) -> list[str]:
        """返回 skill 的全部后代技能名 (BFS, 含跨层)."""
        out: list[str] = []
        stack = [name]
        while stack:
            cur = stack.pop()
            kids = cls.children(cur)
            for k in kids:
                out.append(k)
                stack.append(k)
        return out

    @classmethod
    def subtree(cls, name: str) -> list[SkillDefinition]:
        """返回以 skill 为根的整棵子树 (根 + 全部后代)."""
        root = cls.get(name)
        if root is None:
            return []
        return [root] + [
            cls._skills[d] for d in cls.descendants(name) if d in cls._skills
        ]

    @classmethod
    def tree(cls) -> dict[str, list[str]]:
        """返回整个技能树的父子映射 {parent: [children]}.

        顶层技能 (parent is None) 用 key None 列出.
        """
        ensure_presets()
        mapping: dict[str, list[str]] = {}
        for s in cls._skills.values():
            key = s.parent or ""
            mapping.setdefault(key, []).append(s.name)
        for k in mapping:
            mapping[k].sort()
        return mapping

    @classmethod
    def clear(cls) -> None:
        cls._skills.clear()


def register_skill(skill: SkillDefinition) -> SkillDefinition:
    """Decorator-style registration."""
    return SkillRegistry.register(skill)


def ensure_presets() -> None:
    """懒加载注册预设技能.

    从 ``skills/__init__`` 移除 eager ``from ... import presets`` 后, 这里负责在
    查询/取用技能前按需导入 presets. 副作用：导入 ``huginn.skills.presets`` 会
    把 ~45 个预设 SkillDefinition 注册进 SkillRegistry. Python 模块缓存保证重复
    调用是 no-op（幂等）; import 失败静默吞掉, 不阻断查询（advisory）.
    """
    try:
        import huginn.skills.presets  # noqa: F401
    except Exception:
        logger.debug("ensure_presets: presets import failed", exc_info=True)
