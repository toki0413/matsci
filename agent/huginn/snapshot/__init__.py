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

    # 3) 以"用户消息"为锚点回滚 (dry-run 先看影响清单)
    from huginn.snapshot.rewind import list_anchors, plan_rewind, apply_rewind
    anchors = list_anchors(thread_id)
    impact = plan_rewind(thread_id, anchors[0])   # 只读
    apply_rewind(thread_id, anchors[0])           # 真回滚
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
from huginn.snapshot.rewind import (
    FileImpact,
    RewindAnchor,
    RewindImpact,
    apply_rewind,
    list_anchors,
    plan_rewind,
)

__all__ = [
    "FileImpact",
    "FilePatch",
    "FileSnapshot",
    "RewindAnchor",
    "RewindImpact",
    "SnapshotManager",
    "apply_rewind",
    "list_anchors",
    "plan_rewind",
    "register_snapshot_hooks",
    "snapshot_pre_hook",
    "snapshot_post_hook",
]
