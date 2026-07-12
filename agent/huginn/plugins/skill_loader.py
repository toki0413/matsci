"""技能加载器 —— 参考 Claude Code 的 skills/loadSkillsDir.ts 实现。

负责解析 SKILL.md 的 YAML frontmatter、根据当前文件路径做条件激活、
渲染变量占位符以及通过 realpath 去重符号链接。

设计要点：
- pathspec 是可选依赖，缺失时回退到标准库 fnmatch 做兜底匹配
- 模块级缓存保存条件技能和已激活的动态技能，避免每次调用都重新扫描
- 变量替换同时兼容 ${HUGINN_*} 和 ${CLAUDE_*} 两套命名
"""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# pathspec 可选，没有也能跑
try:
    import pathspec

    _HAVE_PATHSPEC = True
except ImportError:  # pragma: no cover - 环境差异
    _HAVE_PATHSPEC = False
    logger.debug("pathspec 未安装，条件激活回退到 fnmatch 匹配")

# frontmatter 正则：开头 --- 包裹的 YAML 块，后面是 markdown 正文
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)

# 模块级状态：注册过的条件技能 + 当前激活的动态技能
_conditional_skills: dict[str, dict[str, Any]] = {}
_dynamic_skills: dict[str, dict[str, Any]] = {}


def parse_skill_header(path: Path) -> dict[str, Any]:
    """Parse frontmatter only — skip body for lazy loading.

    Returns the same dict shape as parse_skill_file, but with content=""
    and skill_path set so the body can be loaded on demand via
    parse_skill_body().
    """
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {
            "name": path.stem,
            "description": "",
            "allowed_tools": [],
            "when_to_use": "",
            "paths": [],
            "model": None,
            "effort": None,
            "content": "",
            "skill_dir": str(path.parent),
            "skill_path": str(path),
        }

    frontmatter = yaml.safe_load(match.group(1)) or {}
    name = frontmatter.get("name") or path.stem

    return {
        "name": name,
        "description": frontmatter.get("description", ""),
        "allowed_tools": frontmatter.get("allowed-tools", []),
        "when_to_use": frontmatter.get("when_to_use", ""),
        "paths": frontmatter.get("paths", []) or [],
        "model": frontmatter.get("model"),
        "effort": frontmatter.get("effort"),
        "content": "",  # body not loaded — call parse_skill_body() to get it
        "skill_dir": str(path.parent),
        "skill_path": str(path),
    }


def parse_skill_body(path: Path) -> str:
    """Read only the body content, skipping frontmatter."""
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return text
    return match.group(2)


def parse_skill_file(path: Path) -> dict[str, Any]:
    """解析单个 SKILL.md，提取 frontmatter 字段和正文。

    缺失的字段会给合理的默认值，调用方不用做空值判断。
    """
    skill = parse_skill_header(path)
    skill["content"] = parse_skill_body(path)
    return skill


def load_skills_from_dir(skill_dir: Path) -> list[dict[str, Any]]:
    """递归扫描目录下所有 SKILL.md，返回解析后的技能列表。

    同一 realpath 只保留一次，避免符号链接造成重复加载。
    """
    seen: set[str] = set()
    skills: list[dict[str, Any]] = []

    for f in sorted(skill_dir.rglob("SKILL.md")):
        identity = get_file_identity(f)
        if identity in seen:
            continue
        seen.add(identity)

        try:
            skill = parse_skill_header(f)
        except Exception as exc:  # noqa: BLE001 - 单个文件挂了不影响其他
            logger.warning("解析技能文件失败 %s: %s", f, exc)
            continue
        skills.append(skill)

    return skills


def register_conditional_skills(skills: list[dict[str, Any]]) -> None:
    """把带 paths 字段的技能登记为条件技能。

    只登记不激活，激活要等到 activate_conditional_skills 被调用。
    """
    for skill in skills:
        if skill.get("paths"):
            _conditional_skills[skill["name"]] = skill


def _relative_path(file_path: str, cwd: str) -> str:
    """把任意文件路径转成相对 cwd 的路径字符串。

    绝对路径先 resolve 再取相对；相对路径原样返回。
    取相对失败（比如跨盘符）就退回 basename，至少能匹配通配。
    """
    p = Path(file_path)
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(Path(cwd).resolve())
            return str(rel)
        except ValueError:
            # 不在 cwd 子树下，退回文件名做兜底
            return p.name
    return file_path


def activate_conditional_skills(
    file_paths: list[str], cwd: str
) -> list[str]:
    """根据当前涉及的文件路径激活匹配的条件技能。

    返回这次新激活的技能名列表（已经在 _dynamic_skills 里的不会重复加）。
    """
    activated: list[str] = []

    if not _conditional_skills:
        return activated

    rel_paths = [_relative_path(fp, cwd) for fp in file_paths]

    for name, skill in _conditional_skills.items():
        # 已经激活过就跳过，避免重复入栈
        if name in _dynamic_skills:
            continue

        patterns = skill.get("paths", []) or []
        if not patterns:
            continue

        matched = False

        if _HAVE_PATHSPEC:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            for rel in rel_paths:
                if spec.match_file(rel):
                    matched = True
                    break
        else:
            # 没装 pathspec，用 fnmatch 兜底，语义差点但能用
            for pattern in patterns:
                for rel in rel_paths:
                    if fnmatch.fnmatch(rel, pattern):
                        matched = True
                        break
                if matched:
                    break

        if matched:
            _dynamic_skills[name] = skill
            activated.append(name)

    return activated


def get_dynamic_skills() -> dict[str, dict[str, Any]]:
    """返回当前激活的动态技能（拷贝，外部改不影响内部状态）。"""
    return dict(_dynamic_skills)


def clear_dynamic_skills() -> None:
    """清空已激活的动态技能。通常在会话切换或任务结束时调用。"""
    _dynamic_skills.clear()


def clear_conditional_skills() -> None:
    """清空条件技能注册表。重新加载技能目录前调用一次。"""
    _conditional_skills.clear()


def render_skill_content(
    skill: dict[str, Any], session_id: str = ""
) -> str:
    """把技能正文里的占位符替换成实际值.

    支持两套变量名：
    - ${HUGINN_SKILL_DIR} / ${HUGINN_SESSION_ID}  本项目风格
    - ${CLAUDE_SKILL_DIR} / ${CLAUDE_SESSION_ID}  兼容 Claude Code

    如果 content 为空 (header-only loading), 自动 lazy-load body.
    """
    content = skill.get("content", "")
    # lazy-load body if not yet loaded
    if not content and skill.get("skill_path"):
        try:
            content = parse_skill_body(Path(skill["skill_path"]))
        except Exception:
            logger.warning("lazy-load skill body failed: %s", skill.get("skill_path"))
            content = ""
    skill_dir = skill.get("skill_dir", ".")

    content = content.replace("${HUGINN_SKILL_DIR}", skill_dir)
    content = content.replace("${HUGINN_SESSION_ID}", session_id)
    content = content.replace("${CLAUDE_SKILL_DIR}", skill_dir)
    content = content.replace("${CLAUDE_SESSION_ID}", session_id)

    return content


def get_file_identity(path: Path) -> str:
    """用 realpath 解析符号链接，作为文件去重的 key。

    同一个技能可能通过软链接出现在多个扫描路径下，用 realpath
    能保证只算一次。
    """
    return str(path.resolve())
