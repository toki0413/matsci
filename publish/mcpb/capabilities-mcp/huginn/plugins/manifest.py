"""统一 manifest —— 把任意接入目录归一成单一 spec dict。

三种来源按优先级读取, 结果约定一致, 向后兼容既有约定:
  1. ``manifest.yaml``  (新格式: 显式声明 kind/tools/prompts/scripts/entrypoint/paths)
  2. ``metadata.yaml``  (既有插件约定, 推断 kind=plugin)
  3. ``SKILL.md``       (既有技能约定 frontmatter, 推断 kind=skill)

归一后返回的 dict 字段稳定 (kind/name/version/description/tools/prompts/
scripts/entrypoint/paths/source), 供 catalog 路由与展示共用。只做"读 + 归一",
不发注册, 不碰执行层 —— 每种 kind 落到哪类注册表由 catalog 决定。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# 一个接入目录长啥样决定它是什么 kind; manifest.yaml 里显式写了就以它为准.
_PLUGIN_MARKER = "main.py"
_SKILL_MARKER = "SKILL.md"
# SKILL.md frontmatter 起始块 (与 skill_loader/_parse_frontmatter 同款).
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

# 稳定字段默认值 —— 没写就填默认, 调用方不用做空值判断.
_DEFAULTS: dict[str, Any] = {
    "kind": None,
    "name": "",
    "version": "0.0.0",
    "description": "",
    "tools": [],
    "prompts": [],
    "scripts": [],
    "entrypoint": None,
    "paths": [],
}


def _infer_kind(entry: Path) -> str | None:
    """根据目录内容猜 kind (manifest.yaml 缺省时兜底)."""
    if (entry / _PLUGIN_MARKER).is_file():
        return "plugin"
    if (entry / _SKILL_MARKER).is_file():
        return "skill"
    return None


def _read_manifest_yaml(path: Path) -> dict[str, Any] | None:
    """读新格式 manifest.yaml, 归一字段 + 推断 kind."""
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    spec = {**_DEFAULTS, **data}
    if not spec["kind"]:
        spec["kind"] = _infer_kind(path.parent) or "plugin"
    spec["name"] = spec["name"] or path.parent.name
    spec["tools"] = list(spec.get("tools") or [])
    spec["prompts"] = list(spec.get("prompts") or [])
    spec["scripts"] = list(spec.get("scripts") or [])
    spec["paths"] = list(spec.get("paths") or [])
    spec["version"] = str(spec.get("version") or "0.0.0")
    spec["source"] = "manifest.yaml"
    return spec


def _read_metadata_yaml(path: Path) -> dict[str, Any] | None:
    """读既有插件 metadata.yaml, 归一 (kind=plugin)."""
    if not path.is_file():
        return None
    try:
        from huginn.plugins.metadata import PluginMetadata

        meta = PluginMetadata.from_yaml(path)
    except Exception:
        return None
    scripts_dir = path.parent / "scripts"
    scripts = sorted(p.name for p in scripts_dir.glob("*.py")) if scripts_dir.is_dir() else []
    return {
        **_DEFAULTS,
        "kind": "plugin",
        "name": meta.name,
        "version": meta.version,
        "description": meta.description,
        "scripts": scripts,
        "entrypoint": "main.py",
        "source": "metadata.yaml",
    }


def _read_skill_md(path: Path) -> dict[str, Any] | None:
    """读既有 SKILL.md frontmatter, 归一 (kind=skill)."""
    if not path.is_file():
        return None
    try:
        from huginn.plugins.skill_loader import parse_skill_header

        hdr = parse_skill_header(path)
    except Exception:
        # skill_loader 失败就用本地轻量 frontmatter 兜底, 别整块挂掉.
        hdr = _fallback_skill_header(path)
    scripts_dir = path.parent / "scripts"
    scripts = sorted(p.name for p in scripts_dir.glob("*.py")) if scripts_dir.is_dir() else []
    return {
        **_DEFAULTS,
        "kind": "skill",
        "name": hdr.get("name") or path.parent.name,
        "version": str(hdr.get("version") or "0.0.0"),
        "description": hdr.get("description", ""),
        "scripts": scripts,
        "paths": list(hdr.get("paths") or []),
        "source": "SKILL.md",
    }


def _fallback_skill_header(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    name = path.parent.name
    description = ""
    if match:
        try:
            fm = yaml.safe_load(match.group(1)) or {}
            name = fm.get("name") or name
            description = fm.get("description") or ""
        except Exception:
            # frontmatter 解析失败就退回纯目录名, 不阻碍接入项发现.
            name = path.parent.name
            description = ""
    return {"name": name, "description": description, "version": "0.0.0", "paths": []}


def parse_dir(entry: str | Path) -> dict[str, Any] | None:
    """把单个接入目录归一成 spec dict; 识别不出就返回 None.

    兼容三种来源 (manifest.yaml > metadata.yaml > SKILL.md), 结果字段一致.
    """
    entry = Path(entry)
    if not entry.is_dir():
        return None
    path = entry / "manifest.yaml"
    spec = _read_manifest_yaml(path)
    if spec is not None:
        return spec
    return (
        _read_metadata_yaml(entry / "metadata.yaml")
        or _read_skill_md(entry / "SKILL.md")
    )


def discover(root: str | Path) -> list[dict[str, Any]]:
    """扫描 root 下的一级子目录, 返回可识别接入项的归一 spec (按名字排序)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return [
        spec
        for child in sorted(root.iterdir())
        if child.is_dir() and (spec := parse_dir(child)) is not None
    ]


def write_manifest(entry: str | Path, spec: dict[str, Any]) -> Path:
    """为接入目录生成/覆写 manifest.yaml.

    spec 至少要有 name; kind/tools/prompts/scripts/entrypoint/paths 可选.
    这是"新接入项"的第一步: 写份 manifest, 之后走 catalog 路由到具体注册表.
    """
    entry = Path(entry)
    entry.mkdir(parents=True, exist_ok=True)
    data = {k: spec[k] for k in _DEFAULTS if k in spec}
    data["name"] = spec["name"] or entry.name
    out = entry / "manifest.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


__all__ = ["parse_dir", "discover", "write_manifest"]
