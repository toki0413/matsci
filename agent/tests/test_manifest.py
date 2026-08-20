"""统一 manifest 测试 —— 三种来源归一 + kind 推断 + catalog 采集."""

from __future__ import annotations

from pathlib import Path

from huginn.plugins.manifest import discover, parse_dir, write_manifest


def test_manifest_yaml_kind_and_fields(tmp_path: Path):
    entry = tmp_path / "alpha"
    write_manifest(entry, {"kind": "tool", "name": "alpha"})
    spec = parse_dir(entry)
    assert spec is not None
    assert spec["kind"] == "tool"
    assert spec["name"] == "alpha"
    assert spec["source"] == "manifest.yaml"
    assert spec["version"] == "0.0.0"  # 缺省填默认值


def test_manifest_infers_kind_from_content(tmp_path: Path):
    # 有 SKILL.md → 推断 kind=skill (即便 manifest.yaml 没写 kind)
    entry = tmp_path / "beta"
    entry.mkdir()
    (entry / "SKILL.md").write_text(
        "---\nname: beta\ndescription: a skill\n---\n\nbody", encoding="utf-8"
    )
    spec = parse_dir(entry)
    assert spec is not None
    assert spec["kind"] == "skill"
    assert spec["name"] == "beta"
    assert spec["source"] == "SKILL.md"


def test_metadata_yaml_normalized_as_plugin(tmp_path: Path):
    entry = tmp_path / "gamma"
    entry.mkdir()
    (entry / "metadata.yaml").write_text(
        "name: gamma\nversion: 1.2.0\ndescription: a plugin\n", encoding="utf-8"
    )
    (entry / "main.py").write_text("class _:\n    pass\n", encoding="utf-8")
    spec = parse_dir(entry)
    assert spec is not None
    assert spec["kind"] == "plugin"
    assert spec["version"] == "1.2.0"


def test_discover_scans_subdirs_and_skips_unknown(tmp_path: Path):
    (tmp_path / "a").mkdir()
    write_manifest(tmp_path / "a", {"kind": "tool", "name": "a"})
    (tmp_path / "junk").mkdir()  # 无任何标记 → 跳过
    names = [s["name"] for s in discover(tmp_path)]
    assert names == ["a"]


def test_catalog_collects_plugins_dir(tmp_path: Path):
    from huginn.catalog.manager import CatalogManager

    write_manifest(tmp_path / "my-tool", {"kind": "tool", "name": "my-tool"})
    entry = write_manifest(tmp_path / "notes", {"kind": "plugin", "name": "notes"})
    (entry.parent / "main.py").write_text("class S:\n    pass\n", encoding="utf-8")

    mgr = CatalogManager()
    mgr.discover_all(plugins_dir=tmp_path)
    ids = {e.id for e in mgr.list()}
    assert "tool:my-tool" in ids
    assert "plugin:notes" in ids
    assert mgr.get("plugin:notes").origin == "dirs"