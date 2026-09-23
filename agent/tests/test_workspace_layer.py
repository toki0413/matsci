"""P2: 整层快照 / 整层丢弃 (workspace layer) 单测.

覆盖 create_layer / discard_layer / layer_diff / list_layers:
不按后缀过滤拍整棵树, 丢弃时删新建 + 回写改动 + 补回删除 + 清新建空目录.
用独立 root (tmp_path) 建非单例 manager, 不碰真实 ~/.huginn/snapshots.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from huginn.snapshot.file_snapshot import SnapshotManager


@pytest.fixture
def mgr(tmp_path: Path) -> SnapshotManager:
    """独立 root 的 manager (非单例), 存储与全局隔离."""
    return SnapshotManager(root=tmp_path / "snapshots")


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    """一个小工作区: 两个文件 + 一个子目录."""
    d = tmp_path / "ws"
    d.mkdir()
    (d / "a.txt").write_text("alpha\n", encoding="utf-8")
    (d / "b.dat").write_text("beta\n", encoding="utf-8")
    sub = d / "sub"
    sub.mkdir()
    (sub / "c.py").write_text("print(1)\n", encoding="utf-8")
    return d


# ── create_layer ──────────────────────────────────────────────


class TestCreateLayer:
    def test_id_is_prefixed(self, mgr: SnapshotManager, ws: Path):
        """层 id 带 'L' 前缀, 与步骤快照 (纯 hex) 不会撞."""
        layer_id = mgr.create_layer(ws)
        assert layer_id.startswith("L")
        assert len(layer_id) == 17  # 'L' + 16 hex

    def test_captures_whole_tree_not_filtered(self, mgr: SnapshotManager, ws: Path):
        """整层快照不受 watch_patterns 过滤: .txt 这类非材料后缀也进清单."""
        layer_id = mgr.create_layer(ws)
        manifest = mgr._load_layer(layer_id)
        assert manifest is not None
        files = manifest["files"]
        assert set(files) == {"a.txt", "b.dat", "sub/c.py"}
        assert "sub" in manifest["dirs"]

    def test_records_hash_and_mode(self, mgr: SnapshotManager, ws: Path):
        os.chmod(ws / "a.txt", 0o600)
        layer_id = mgr.create_layer(ws)
        entry = mgr._load_layer(layer_id)["files"]["a.txt"]
        assert entry["kind"] == "file"
        assert entry["backed_up"] is True
        assert entry["mode"] == 0o600
        assert len(entry["hash"]) == 64

    def test_skips_skip_dirs(self, mgr: SnapshotManager, ws: Path):
        """__pycache__ / .git 这类目录不进清单."""
        cache = ws / "__pycache__"
        cache.mkdir()
        (cache / "x.pyc").write_bytes(b"\x00\x01")
        layer_id = mgr.create_layer(ws)
        files = mgr._load_layer(layer_id)["files"]
        assert not any("__pycache__" in rel for rel in files)


# ── discard_layer ─────────────────────────────────────────────


class TestDiscardLayer:
    def test_restores_modified_deleted_and_removes_created(
        self, mgr: SnapshotManager, ws: Path
    ):
        layer_id = mgr.create_layer(ws)
        # 改一个 / 删一个 / 建一个
        (ws / "a.txt").write_text("CHANGED\n", encoding="utf-8")
        (ws / "b.dat").unlink()
        (ws / "new.cif").write_text("new\n", encoding="utf-8")

        affected = mgr.discard_layer(layer_id)

        assert (ws / "a.txt").read_text(encoding="utf-8") == "alpha\n"
        assert (ws / "b.dat").exists() and (ws / "b.dat").read_text() == "beta\n"
        assert not (ws / "new.cif").exists()
        assert {"a.txt", "b.dat", "new.cif"} <= set(affected)

    def test_removes_new_empty_dirs(self, mgr: SnapshotManager, ws: Path):
        layer_id = mgr.create_layer(ws)
        (ws / "fresh" / "nested").mkdir(parents=True)
        assert (ws / "fresh" / "nested").is_dir()

        mgr.discard_layer(layer_id)

        assert not (ws / "fresh").exists()

    def test_keeps_preexisting_empty_dir(self, mgr: SnapshotManager, ws: Path):
        """建层时就存在的空目录不该被当成新建而删掉."""
        empty = ws / "keepme"
        empty.mkdir()
        layer_id = mgr.create_layer(ws)

        mgr.discard_layer(layer_id)

        assert empty.is_dir()

    def test_nonempty_new_dir_survives(self, mgr: SnapshotManager, ws: Path):
        """非空的新目录删不掉 (rmdir 失败), 不该把里面的文件误删."""
        layer_id = mgr.create_layer(ws)
        d = ws / "kept"
        d.mkdir()
        # 这个文件在清单外, 会被删; 目录随后空掉 → 会被清掉
        (d / "junk.tmp").write_text("junk\n", encoding="utf-8")

        mgr.discard_layer(layer_id)

        assert not (d / "junk.tmp").exists()
        assert not d.exists()

    def test_restores_mode(self, mgr: SnapshotManager, ws: Path):
        os.chmod(ws / "a.txt", 0o600)
        layer_id = mgr.create_layer(ws)
        os.chmod(ws / "a.txt", 0o644)

        mgr.discard_layer(layer_id)

        assert stat.S_IMODE((ws / "a.txt").stat().st_mode) == 0o600

    def test_unknown_layer_returns_empty(self, mgr: SnapshotManager, ws: Path):
        assert mgr.discard_layer("Ldeadbeefdeadbeef", ws) == []

    def test_discard_is_idempotent(self, mgr: SnapshotManager, ws: Path):
        """丢弃两次: 第二次工作区已复原, 不该报错也不该再删东西."""
        layer_id = mgr.create_layer(ws)
        (ws / "a.txt").write_text("CHANGED\n", encoding="utf-8")
        mgr.discard_layer(layer_id)

        second = mgr.discard_layer(layer_id)

        assert (ws / "a.txt").read_text(encoding="utf-8") == "alpha\n"
        assert second == []  # 无变化 → 无受影响路径

    @pytest.mark.skipif(sys.platform == "win32", reason="符号链接需权限")
    def test_symlink_recreated(self, mgr: SnapshotManager, ws: Path):
        link = ws / "link.txt"
        os.symlink("a.txt", link)
        layer_id = mgr.create_layer(ws)
        link.unlink()

        mgr.discard_layer(layer_id)

        assert link.is_symlink()
        assert os.readlink(link) == "a.txt"

    @pytest.mark.skipif(sys.platform == "win32", reason="符号链接需权限")
    def test_symlink_not_deleted_as_new(self, mgr: SnapshotManager, ws: Path):
        """符号链接在清单里, 不该被当作"新建文件"删掉."""
        link = ws / "link.txt"
        os.symlink("a.txt", link)
        layer_id = mgr.create_layer(ws)

        mgr.discard_layer(layer_id)

        assert link.is_symlink()


# ── 大文件 (超备份上限) ───────────────────────────────────────


class TestUnbackedFiles:
    def test_large_file_not_backed_up(self, mgr: SnapshotManager, ws: Path):
        (ws / "big.dat").write_text("0123456789", encoding="utf-8")
        layer_id = mgr.create_layer(ws, max_file_bytes=4)
        entry = mgr._load_layer(layer_id)["files"]["big.dat"]
        assert entry["backed_up"] is False

    def test_discard_skips_unbacked_but_keeps_file(
        self, mgr: SnapshotManager, ws: Path
    ):
        """超限文件无法回写内容, 但它在清单里 → 不该被删."""
        (ws / "big.dat").write_text("0123456789", encoding="utf-8")
        layer_id = mgr.create_layer(ws, max_file_bytes=4)
        (ws / "big.dat").write_text("MUTATED\n", encoding="utf-8")

        mgr.discard_layer(layer_id)

        assert (ws / "big.dat").exists()
        # 内容回不去 (没备份), 保持改动后的值
        assert (ws / "big.dat").read_text(encoding="utf-8") == "MUTATED\n"


# ── layer_diff ────────────────────────────────────────────────


class TestLayerDiff:
    def test_reports_created_modified_deleted(self, mgr: SnapshotManager, ws: Path):
        layer_id = mgr.create_layer(ws)
        (ws / "a.txt").write_text("CHANGED\n", encoding="utf-8")  # modified
        (ws / "b.dat").unlink()                                    # deleted
        (ws / "new.cif").write_text("new\n", encoding="utf-8")     # created

        patches = mgr.layer_diff(layer_id)
        by_rel = {p.file_path: p.change_type for p in patches}
        assert by_rel["a.txt"] == "modified"
        assert by_rel["b.dat"] == "deleted"
        assert by_rel["new.cif"] == "created"

    def test_clean_tree_no_patches(self, mgr: SnapshotManager, ws: Path):
        layer_id = mgr.create_layer(ws)
        assert mgr.layer_diff(layer_id) == []

    def test_unknown_layer_returns_empty(self, mgr: SnapshotManager, ws: Path):
        assert mgr.layer_diff("Lnope") == []


# ── list_layers / cap ─────────────────────────────────────────


class TestListLayers:
    def test_lists_metadata(self, mgr: SnapshotManager, ws: Path):
        layer_id = mgr.create_layer(ws, label="iter-1")
        infos = mgr.list_layers()
        assert len(infos) == 1
        info = infos[0]
        assert info.layer_id == layer_id
        assert info.label == "iter-1"
        assert info.file_count == 3
        assert info.workspace == str(ws.resolve())

    def test_sorted_by_time(self, mgr: SnapshotManager, ws: Path):
        first = mgr.create_layer(ws, label="a")
        second = mgr.create_layer(ws, label="b")
        ids = [li.layer_id for li in mgr.list_layers()]
        assert ids == [first, second]

    def test_cap_fifo(self, mgr: SnapshotManager, ws: Path, monkeypatch):
        monkeypatch.setattr("huginn.snapshot.file_snapshot._MAX_LAYERS", 2)
        ids = [mgr.create_layer(ws, label=f"l{i}") for i in range(3)]
        kept = [li.layer_id for li in mgr.list_layers()]
        assert kept == ids[1:]  # 最老的被淘汰
        # 日志也一并压缩
        assert len(mgr._load_layer_log()) == 2
