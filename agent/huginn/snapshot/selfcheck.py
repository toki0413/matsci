"""snapshot 模块的自检. 无框架, 直接 assert.

跑法::

    python -m huginn.snapshot.selfcheck

用一个独立 root (tempfile.mkdtemp) 跑完整 track→patch→revert→unrevert 流程,
不碰真实的 ~/.huginn/snapshots. 失败就抛 AssertionError, 退出码非 0.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path

from huginn.snapshot.file_snapshot import SnapshotManager


def _cleanup(root: Path) -> None:
    # 这台机器上 AV 扫刚写的文件会让 shutil.rmtree 单步卡几十秒,
    # 同步删会让自检看着像卡死. 放 daemon 线程后台删, 不阻塞断言结果.
    threading.Thread(
        target=shutil.rmtree, args=(root,), kwargs={"ignore_errors": True},
        daemon=True,
    ).start()


def _run() -> None:
    root = Path(tempfile.mkdtemp(prefix="huginn_snap_selfcheck_"))
    try:
        # 用独立 root 建一个非单例 manager, 不污染全局 ~/.huginn
        mgr = SnapshotManager(root=root)

        ws = root / "ws"
        ws.mkdir()
        (ws / "POSCAR").write_text("Cu\n 1.0\n", encoding="utf-8")
        (ws / "data.dat").write_text("old\n", encoding="utf-8")
        (ws / "skip.txt").write_text("not watched\n", encoding="utf-8")

        # track: 拍执行前状态
        sid = mgr.track("vasp_tool", ws)
        snap = mgr._load(sid)
        assert snap is not None and len(snap.files) == 2, snap  # POSCAR + data.dat, 不含 skip.txt
        assert "POSCAR" in snap.files and "data.dat" in snap.files

        # 模拟工具执行: 改一个 / 删一个 / 建一个
        (ws / "POSCAR").write_text("Cu\n 2.0\n", encoding="utf-8")  # modified
        (ws / "data.dat").unlink()                                  # deleted
        (ws / "new.cif").write_text("data_new\n", encoding="utf-8") # created

        patches = mgr.patch(sid, ws)
        kinds = sorted(p.change_type for p in patches)
        assert kinds == ["created", "deleted", "modified"], kinds
        assert not snap.reverted

        # revert: 回到执行前
        reverted = mgr.revert(sid, ws)
        assert set(reverted) >= {"POSCAR", "data.dat", "new.cif"}, reverted
        assert (ws / "POSCAR").read_text(encoding="utf-8") == "Cu\n 1.0\n"
        assert (ws / "data.dat").exists() and (ws / "data.dat").read_text() == "old\n"
        assert not (ws / "new.cif").exists()
        assert mgr._is_reverted(sid)

        # unrevert: 恢复到执行后
        restored = mgr.unrevert(sid, ws)
        assert set(restored) >= {"POSCAR", "data.dat", "new.cif"}, restored
        assert (ws / "POSCAR").read_text(encoding="utf-8") == "Cu\n 2.0\n"
        assert not (ws / "data.dat").exists()
        assert (ws / "new.cif").exists()
        assert not mgr._is_reverted(sid)

        # history
        hist = mgr.get_history(tool_name="vasp_tool")
        assert len(hist) == 1 and hist[0].step_id == sid
        assert len(hist[0].patches) == 3

        print("snapshot selfcheck OK:", sid, flush=True)
    finally:
        _cleanup(root)


if __name__ == "__main__":
    _run()
