"""制度化闭环：插件准入门禁。

Star 插件（huginn/plugins/<name>/ 含 metadata.yaml + main.py）必须：
  1. metadata.yaml 合法：name==目录名、version 非占位默认、huginn 版本兼容；
  2. main.py 存在（Star 插件运行入口）；
  3. 出现在冻结的 SANCTIONED_PLUGINS 清单里 —— 新增插件必须显式准入，
     防止插件生态无序膨胀（跟"200 技能很多休眠"同样的教训）。

清单只增不减：想让新插件生效，先在这里显式登记 + 评审。
"""

from __future__ import annotations

from pathlib import Path

from huginn.plugins.metadata import HUGINN_API_VERSION, PluginMetadata

HUGINN_DIR = Path(__file__).resolve().parents[1] / "huginn"
PLUGINS_DIR = HUGINN_DIR / "plugins"

# 冻结的准入插件清单。新增插件必须显式登记；移除插件也要同步删。
SANCTIONED_PLUGINS = frozenset({"ponytail"})


def _discover_plugin_dirs() -> list[Path]:
    if not PLUGINS_DIR.is_dir():
        return []
    return sorted(
        child for child in PLUGINS_DIR.iterdir()
        if child.is_dir() and (child / "metadata.yaml").is_file()
    )


def test_plugin_manifest_is_frozen_and_complete():
    """仓库里的 Star 插件集合必须与准入清单完全一致。"""
    found = {d.name for d in _discover_plugin_dirs()}
    assert found == set(SANCTIONED_PLUGINS), (
        "插件与准入清单不符。新增插件必须显式加入 SANCTIONED_PLUGINS 并经评审；"
        "移除插件要同步删清单。"
        f"\n  仓库有清单没有: {sorted(found - set(SANCTIONED_PLUGINS))}"
        f"\n  清单有仓库没: {sorted(set(SANCTIONED_PLUGINS) - found)}"
    )


def test_every_plugin_metadata_is_valid():
    """每个插件目录的 metadata.yaml 必须合法且版本兼容。"""
    for d in _discover_plugin_dirs():
        meta = PluginMetadata.from_yaml(d / "metadata.yaml")
        assert meta.name == d.name, f"{d.name}: metadata.name 与目录名不一致"
        assert meta.version and meta.version != "0.0.0", f"{d.name}: version 缺失/默认"
        assert meta.check_version_compatibility(HUGINN_API_VERSION), (
            f"{d.name}: huginn_version_range {meta.huginn_version_range!r} "
            f"不含当前 API {HUGINN_API_VERSION}"
        )
        assert (d / "main.py").is_file(), f"{d.name}: 缺 main.py（Star 插件入口）"


def test_manifest_is_nonempty():
    """准入清单非空，防止误清空后失控。"""
    assert SANCTIONED_PLUGINS, "SANCTIONED_PLUGINS 不应为空"