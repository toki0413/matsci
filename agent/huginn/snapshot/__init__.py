"""步骤级文件快照与回滚系统.

给 agent 的工具调用做文件系统级 undo, 互补于已有的 turn 级 trajectory 日志.
设计借鉴 OpenCode/OpenScience 的 snapshot/ 机制.

主要入口::

    from huginn.snapshot import SnapshotManager, register_snapshot_hooks

    # 1) 注册到现有 hook 系统 (pre 拍照, post 比对)
    register_snapshot_hooks(hook_manager, workspace="/path/to/ws")

    # 2) 手动回滚某一步
    mgr = SnapshotManager()
    mgr.revert(step_id, Path("/path/to/ws"))
    mgr.unrevert(step_id, Path("/path/to/ws"))
"""

from huginn.snapshot.file_snapshot import (
    FilePatch,
    FileSnapshot,
    SnapshotManager,
)
from huginn.snapshot.integration import (
    register_snapshot_hooks,
    snapshot_post_hook,
    snapshot_pre_hook,
)

__all__ = [
    "FilePatch",
    "FileSnapshot",
    "SnapshotManager",
    "register_snapshot_hooks",
    "snapshot_pre_hook",
    "snapshot_post_hook",
]
