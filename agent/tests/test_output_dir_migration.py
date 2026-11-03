"""产物目录迁移层: 新名写入 + 旧名兼容读取.

覆盖 examples/_output_paths.py 的三条语义:
1. 写入永远落新目录 (ai4s_*);
2. 读取优先新目录; 新目录缺文件时回落旧目录 (shusheng_*) 并一次性迁移文件,
   旧目录原件保留;
3. 目录型条目只回落读取、不复制 (避免拷贝嵌入式数据库).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER = REPO_ROOT / "examples" / "_output_paths.py"
_spec = importlib.util.spec_from_file_location("_output_paths", _HELPER)
helper = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = helper
_spec.loader.exec_module(helper)


def _make_legacy(root: Path, name: str) -> Path:
    legacy = root / f"shusheng_{name}"
    (legacy / "taste_kb").mkdir(parents=True, exist_ok=True)
    (legacy / "state.json").write_text('{"round": 3}', encoding="utf-8")
    return legacy


def test_output_dir_always_creates_neutral_target(tmp_path: Path):
    out = helper.output_dir("demo", root=tmp_path)
    assert out == tmp_path / "ai4s_demo"
    assert out.is_dir()


def test_read_entry_migrates_legacy_file(tmp_path: Path):
    legacy = _make_legacy(tmp_path, "demo")
    migrated = helper.read_entry("demo", "state.json", root=tmp_path)
    assert migrated == tmp_path / "ai4s_demo" / "state.json"
    assert migrated.read_text(encoding="utf-8") == '{"round": 3}'
    assert (legacy / "state.json").exists(), "旧目录原件必须保留"


def test_new_file_wins_over_legacy(tmp_path: Path):
    _make_legacy(tmp_path, "demo")
    new = tmp_path / "ai4s_demo"
    new.mkdir(parents=True)
    (new / "state.json").write_text('{"round": 9}', encoding="utf-8")
    found = helper.read_entry("demo", "state.json", root=tmp_path)
    assert found == new / "state.json"
    assert found.read_text(encoding="utf-8") == '{"round": 9}'


def test_directory_entries_fall_back_without_copying(tmp_path: Path):
    legacy = _make_legacy(tmp_path, "demo")
    found = helper.read_entry("demo", "taste_kb", root=tmp_path)
    assert found == legacy / "taste_kb"
    assert not (tmp_path / "ai4s_demo" / "taste_kb").exists()


def test_missing_entry_points_at_new_target(tmp_path: Path):
    found = helper.read_entry("demo", "missing.json", root=tmp_path)
    assert found == tmp_path / "ai4s_demo" / "missing.json"
    assert not found.exists()
