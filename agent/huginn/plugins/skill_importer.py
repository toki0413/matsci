"""跨平台技能导入器。

把 OpenClaw / Hermes Agent 的 SKILL.md 转成 HuginnAgent 的 SkillDefinition，
反过来也能把 SkillDefinition 写回 Huginn 原生格式，方便技能在三套体系间搬运。

三家格式都是 YAML frontmatter + Markdown 正文，差别只在字段名：
- OpenClaw:  tools / trigger_conditions / steps
- Hermes:    trigger / steps / tags        (agentskills.io 标准)
- Huginn:    allowed-tools / when_to_use / paths / model / effort

缺失字段一律兜底，不抛异常；只在日志里 warn，调用方拿到的永远是一个可用对象。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from huginn.skills.base import SkillDefinition, SkillStep

logger = logging.getLogger(__name__)

# frontmatter 正则，和 skill_loader 里那一份保持一致，解析行为对齐
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)

# 描述性步骤没有 tool 绑定时的占位值。DeclarativeSkillExecutor 找不到这个工具，
# 这类导入技能主要靠正文当 prompt 用，不指望走结构化执行。
_MANUAL_TOOL = "manual"


class SkillImporter:
    """OpenClaw / Hermes / Huginn 三家 SKILL.md 的互转。"""

    # ── 单文件导入 ──────────────────────────────────────────────

    def import_file(self, skill_path: Path, platform: str = "auto") -> SkillDefinition:
        """导入单个 SKILL.md，platform=auto 时按 frontmatter 自动识别格式。"""
        frontmatter, content = self._parse_frontmatter(skill_path)
        fmt = platform if platform != "auto" else self._detect_platform(frontmatter)

        if fmt == "openclaw":
            skill = self._build_openclaw(frontmatter, content, skill_path)
        elif fmt == "hermes":
            skill = self._build_hermes(frontmatter, content, skill_path)
        else:
            # 认不出来或显式指定 huginn，都走自家格式，字段对得上就用
            skill = self._build_native(frontmatter, content, skill_path)

        self._validate(skill, frontmatter, skill_path)
        return skill

    def import_from_openclaw(self, skill_path: Path) -> SkillDefinition:
        """解析 OpenClaw 格式并映射成 SkillDefinition。

        字段映射：
        - tools            -> required_tools（导出时写回 allowed-tools）
        - trigger_conditions -> metadata.when_to_use（用 " OR " 拼接）
        - steps            -> list[SkillStep]
        """
        frontmatter, content = self._parse_frontmatter(skill_path)
        skill = self._build_openclaw(frontmatter, content, skill_path)
        self._validate(skill, frontmatter, skill_path)
        return skill

    def import_from_hermes(self, skill_path: Path) -> SkillDefinition:
        """解析 Hermes (agentskills.io) 格式并映射成 SkillDefinition。

        字段映射：
        - trigger -> metadata.when_to_use
        - tags    -> tags（同时留一份在 metadata 里）
        - steps   -> list[SkillStep]
        """
        frontmatter, content = self._parse_frontmatter(skill_path)
        skill = self._build_hermes(frontmatter, content, skill_path)
        self._validate(skill, frontmatter, skill_path)
        return skill

    # ── 批量导入 ────────────────────────────────────────────────

    def import_directory(
        self, dir_path: Path, platform: str = "auto"
    ) -> list[SkillDefinition]:
        """递归扫描目录下所有 SKILL.md，逐个导入。

        符号链接按 realpath 去重，避免同一个技能被软链接重复扫进来。
        单个文件解析失败不影响整批，只在日志里 warn。
        """
        if not dir_path.is_dir():
            logger.warning("导入路径不存在或不是目录: %s", dir_path)
            return []

        skills: list[SkillDefinition] = []
        seen: set[str] = set()
        for f in sorted(dir_path.rglob("SKILL.md")):
            identity = str(f.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            try:
                skills.append(self.import_file(f, platform))
            except Exception as exc:  # 单个文件挂了不影响其他
                logger.warning("导入技能失败 %s: %s", f, exc)
        return skills

    # ── 导出 ────────────────────────────────────────────────────

    def export_to_huginn(self, skill: SkillDefinition, output_path: Path) -> None:
        """把 SkillDefinition 写成 Huginn 原生 SKILL.md。"""
        md = skill.metadata
        # 只写非空字段，保持文件干净
        frontmatter: dict[str, Any] = {
            "name": skill.name,
            "description": skill.description,
        }
        if skill.required_tools:
            frontmatter["allowed-tools"] = skill.required_tools
        if md.get("when_to_use"):
            frontmatter["when_to_use"] = md["when_to_use"]
        if md.get("paths"):
            frontmatter["paths"] = md["paths"]
        if md.get("model"):
            frontmatter["model"] = md["model"]
        if md.get("effort"):
            frontmatter["effort"] = md["effort"]
        if skill.tags:
            frontmatter["tags"] = skill.tags

        body = md.get("content") or self._render_body(skill)
        text = (
            "---\n"
            + yaml.dump(
                frontmatter,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )
            + "---\n"
            + body
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        logger.info("已导出 Huginn 格式技能到 %s", output_path)

    # ── 各平台的字段映射 ────────────────────────────────────────

    def _build_openclaw(
        self, fm: dict, content: str, path: Path
    ) -> SkillDefinition:
        tools = fm.get("tools") or []
        triggers = fm.get("trigger_conditions") or []
        return SkillDefinition(
            name=self._skill_name(fm, path),
            description=fm.get("description", ""),
            category=fm.get("category", "general"),
            steps=self._steps_from_strings(fm.get("steps") or []),
            required_tools=list(tools),
            tags=fm.get("tags") or [],
            metadata={
                "platform": "openclaw",
                "when_to_use": " OR ".join(str(t) for t in triggers),
                "paths": fm.get("paths") or [],
                "model": fm.get("model"),
                "effort": fm.get("effort"),
                "content": content,
            },
        )

    def _build_hermes(
        self, fm: dict, content: str, path: Path
    ) -> SkillDefinition:
        tags = fm.get("tags") or []
        # trigger 可能是单个字符串，也可能给成列表，统一成字符串
        trigger = fm.get("trigger", "")
        if isinstance(trigger, list):
            trigger = " OR ".join(str(t) for t in trigger)
        return SkillDefinition(
            name=self._skill_name(fm, path),
            description=fm.get("description", ""),
            category=fm.get("category", "general"),
            steps=self._steps_from_strings(fm.get("steps") or []),
            required_tools=fm.get("tools") or [],
            tags=list(tags),
            metadata={
                "platform": "hermes",
                "when_to_use": str(trigger),
                "paths": fm.get("paths") or [],
                "model": fm.get("model"),
                "effort": fm.get("effort"),
                # 按 agentskills.io 习惯单独留一份原始 tags
                "tags": list(tags),
                "content": content,
            },
        )

    def _build_native(
        self, fm: dict, content: str, path: Path
    ) -> SkillDefinition:
        # Huginn 原生：allowed-tools / when_to_use / paths / model / effort
        return SkillDefinition(
            name=self._skill_name(fm, path),
            description=fm.get("description", ""),
            category=fm.get("category", "general"),
            steps=self._steps_from_strings(fm.get("steps") or []),
            required_tools=fm.get("allowed-tools") or [],
            tags=fm.get("tags") or [],
            metadata={
                "platform": "huginn",
                "when_to_use": fm.get("when_to_use", ""),
                "paths": fm.get("paths") or [],
                "model": fm.get("model"),
                "effort": fm.get("effort"),
                "content": content,
            },
        )

    # ── 工具方法 ────────────────────────────────────────────────

    @staticmethod
    def _skill_name(fm: dict, path: Path) -> str:
        """优先用 frontmatter 里的 name；没有就用所在目录名。

        SKILL.md 本身的 stem 恒为 "SKILL"，当名字没意义，所以退回父目录名。
        """
        if fm.get("name"):
            return str(fm["name"])
        return path.parent.name if path.name.upper() == "SKILL.MD" else path.stem

    @staticmethod
    def _steps_from_strings(steps: list[Any]) -> list[SkillStep]:
        """把外来的描述性步骤字符串转成 SkillStep。

        ponytail: OpenClaw/Hermes 的 steps 是纯文字说明，没有 tool 绑定，
        这里塞个 _MANUAL_TOOL 占位让它们能塞进 SkillDefinition。这类技能
        不会走 DeclarativeSkillExecutor 真正执行，要执行得手动补 tool 映射。
        """
        return [
            SkillStep(
                name=str(s),
                tool=_MANUAL_TOOL,
                input_mapping={},
                output_key=f"step_{i}",
            )
            for i, s in enumerate(steps, 1)
        ]

    def _parse_frontmatter(self, path: Path) -> tuple[dict, str]:
        """拆出 YAML frontmatter 和 Markdown 正文，没有 frontmatter 就全当正文。"""
        text = path.read_text(encoding="utf-8")
        match = _FRONTMATTER_RE.match(text)
        if not match:
            logger.debug("%s 没有 YAML frontmatter，按纯正文处理", path)
            return {}, text
        fm = yaml.safe_load(match.group(1)) or {}
        return fm, match.group(2)

    @staticmethod
    def _detect_platform(fm: dict) -> str:
        """按特征字段猜来源平台。"""
        # trigger_conditions 是 OpenClaw 独有的
        if "trigger_conditions" in fm:
            return "openclaw"
        # Hermes 用 trigger（单数），和 OpenClaw 的 trigger_conditions 区分开
        if "trigger" in fm:
            return "hermes"
        # allowed-tools 是 Huginn 原生
        if "allowed-tools" in fm:
            return "huginn"
        # 都没有就按 tools 字段猜：有 tools 偏 OpenClaw，否则偏 Hermes
        return "openclaw" if "tools" in fm else "hermes"

    @staticmethod
    def _validate(skill: SkillDefinition, fm: dict, path: Path) -> None:
        """缺失关键字段时 warn，但不阻断导入（已有兜底默认值）。"""
        if not fm.get("name"):
            logger.warning("%s 缺少 name 字段，已用目录名兜底", path)
        if not fm.get("description"):
            logger.warning("%s 缺少 description 字段", path)
        if not fm.get("steps"):
            logger.warning("%s 没有 steps，导入后为空步骤技能", path)

    @staticmethod
    def _render_body(skill: SkillDefinition) -> str:
        """没有原始正文时，用 SkillDefinition 自己拼一份 Markdown。"""
        lines = [f"# {skill.name}", "", skill.description, ""]
        if skill.steps:
            lines.append("## Steps")
            for i, s in enumerate(skill.steps, 1):
                lines.append(f"{i}. {s.name}")
            lines.append("")
        if skill.required_tools:
            lines.append(f"Tools: {', '.join(skill.required_tools)}")
        return "\n".join(lines)
