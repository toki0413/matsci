"""PluginMetadata —— 插件元数据, 从 metadata.yaml 加载。

借鉴 AstrBot 的 metadata.yaml + 版本约束, 但字段裁剪到材料科研用得到的部分。
权限声明也放这里 (permissions 字段), 由 PermissionChecker 运行时强制检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml  # PyYAML
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load plugin metadata.yaml; "
        "install it via `pip install pyyaml`"
    ) from e

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version
    _HAS_PACKAGING = True
except ImportError:  # pragma: no cover
    _HAS_PACKAGING = False
    SpecifierSet = None  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]
    InvalidVersion = Exception  # type: ignore[assignment]


# 当前 huginn 插件 API 版本。loader 加载时跟 metadata.yaml 的
# huginn_version_range 做约束检查, 不兼容就拒载。
HUGINN_API_VERSION = "0.1.0"


@dataclass
class PluginMetadata:
    """插件元数据。

    permissions 是字符串列表, 形如:
      - "llm_call"
      - "tool_call:vasp_tool"
      - "file_write:/output"
      - "network"
      - "subprocess"
    具体 enum 见 huginn.plugins.permissions.PluginPermission。
    """

    name: str
    version: str = "0.0.0"
    author: str = ""
    description: str = ""
    huginn_version_range: str = f">={HUGINN_API_VERSION}"
    permissions: list[str] = field(default_factory=list)
    supported_platforms: list[str] = field(default_factory=list)
    homepage: str = ""
    repo: str = ""
    # 原始 yaml 内容, 调试 / 扩展字段用
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginMetadata":
        """从已解析的 dict 构造。字段缺失走默认值。"""
        return cls(
            name=data.get("name", ""),
            version=str(data.get("version", "0.0.0")),
            author=data.get("author", ""),
            description=data.get("description", ""),
            huginn_version_range=str(
                data.get("huginn_version_range")
                or data.get("astrbot_version_range")
                or f">={HUGINN_API_VERSION}"
            ),
            permissions=list(data.get("permissions", []) or []),
            supported_platforms=list(data.get("supported_platforms", []) or []),
            homepage=data.get("homepage", ""),
            repo=data.get("repo", ""),
            raw=dict(data),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PluginMetadata":
        """从 metadata.yaml 文件加载。

        path 应指向插件目录下的 metadata.yaml。
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"metadata.yaml not found: {p}")
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"metadata.yaml root must be a mapping, got {type(data)}")
        return cls.from_dict(data)

    def check_version_compatibility(self, api_version: str = HUGINN_API_VERSION) -> bool:
        """检查当前 API 版本是否落在 huginn_version_range 内。

        没装 packaging 时退化到字符串相等比较 (保守, 不兼容就拒载)。
        """
        if not self.huginn_version_range or self.huginn_version_range == "*":
            return True
        if not _HAS_PACKAGING:
            # 退化: 只支持 ">=x.y.z" 单条件, 而且要求完全匹配
            r = self.huginn_version_range.strip()
            if r.startswith(">="):
                target = r[2:].strip()
                return _ver_tuple(api_version) >= _ver_tuple(target)
            return False
        try:
            spec = SpecifierSet(self.huginn_version_range)
            return Version(api_version) in spec
        except (InvalidVersion, ValueError):
            return False

    def has_permission(self, perm: str) -> bool:
        """检查是否声明了某个权限。

        支持通配: 声明 "tool_call:*" 可命中 "tool_call:vasp_tool"。
        声明 "tool_call:vasp_tool" 不命中 "tool_call:lammps_tool"。
        """
        for declared in self.permissions:
            if _perm_match(declared, perm):
                return True
        return False


def _ver_tuple(v: str) -> tuple[int, ...]:
    """把 '0.1.0' 转 (0, 1, 0), 解析失败补 0。"""
    parts: list[int] = []
    for chunk in v.split("."):
        try:
            parts.append(int(chunk.split("-")[0].split("+")[0]))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def _perm_match(declared: str, requested: str) -> bool:
    """声明权限是否命中请求权限。支持 * 通配 (仅末段)。"""
    if declared == requested:
        return True
    if declared.endswith(":*"):
        prefix = declared[:-2]
        if ":" in requested and requested.rsplit(":", 1)[0] == prefix:
            return True
    return False


__all__ = ["PluginMetadata", "HUGINN_API_VERSION"]
