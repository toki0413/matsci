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

    # 3) 整层快照: 拍整棵工作区, 可一次性丢弃回到建层时状态
    layer_id = mgr.create_layer(Path("/path/to/ws"), label="iter-7")
    mgr.layer_diff(layer_id)          # 看看层后改了什么
    mgr.discard_layer(layer_id)       # 整层丢弃
"""

from huginn.snapshot.file_snapshot import (
    FilePatch,
    FileSnapshot,
    LayerInfo,
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
    "LayerInfo",
    "SnapshotManager",
    "register_snapshot_hooks",
    "snapshot_pre_hook",
    "snapshot_post_hook",
]
