"""把文件快照接到现有的 pre/post tool 钩子上.

用法::

    from huginn.snapshot.integration import register_snapshot_hooks
    register_snapshot_hooks(hook_manager)

复用 HookContext + run_pre/run_post, 不创建并行系统, 跟 science_hooks 平行.
也可以让 register_science_hooks 末尾顺手调一次 (已接好, 幂等).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from huginn.hooks import POST_TOOL_USE, PRE_TOOL_USE, HookContext, HookManager
from huginn.snapshot.file_snapshot import SnapshotManager

logger = logging.getLogger(__name__)

# 要拍照的工具: 仿真工具 + 会改文件的工具. 只读工具 (查库/读文件) 不值得拍.
# ponytail: 默认列表写死, 新增工具忘了登记会漏拍, 但总比全工具全拍省 IO.
# 升级路径: 给 HuginnTool 加个 destructive 标志位, 按标志自动纳入.
_SNAPSHOT_TOOLS: frozenset[str] = frozenset({
    # 仿真
    "vasp_tool", "qe_tool", "cp2k_tool", "lammps_tool", "gromacs_tool",
    "openmm_tool", "gaussian_tool", "orca_tool", "abaqus_tool", "comsol_tool",
    "elmer_tool", "fenics_tool", "openfoam_tool", "packing_tool", "vina_tool",
    "mechanical_tool", "plasma_tool", "transolver_tool",
    "convergence_test_tool", "fem_tool", "neb_tool",
    # 文件操作
    "file_write_tool", "file_edit_tool", "multi_edit_tool", "code_tool",
    "bash_tool",
})

# 模块级配置: register_snapshot_hooks 设一次, 钩子运行时读.
_workspace: Path | None = None
_watched_tools: frozenset[str] = _SNAPSHOT_TOOLS
_watch_patterns: tuple[str, ...] | None = None

# pending 关联: pre 拍照存 step_id, post 取出来比对.
# 单线程异步模型下 (HookManager 文档明确 agent 单线程跑), 同一 thread_id 的
# pre→execute→post 是顺序的, 用单槽即可, 不需要栈.
# ponytail: 假设单线程; 真多线程时改成 thread_id+call_id 显式 correlation.
_pending: dict[str, str] = {}

# post_hook patch 完后把 step_id 暂存这里, register_tool_output 来取.
# 单线程模型下一个 tool call 的 post→register 是顺序的, 单槽够用.
_last_step_id: str | None = None


def _emit(event_type: str, data: dict, thread_id: str = "") -> None:
    """Fire-and-forget 事件发布到 EventBus."""
    try:
        from huginn.events.integration import _publish
        import asyncio
        loop = asyncio.get_running_loop()
        asyncio.ensure_future(_publish(event_type, data, thread_id, source="snapshot"))
    except Exception:
        pass


def consume_last_snapshot_step_id() -> str | None:
    """取走最近一次 snapshot_post_hook 留下的 step_id (一次性消费).

    register_tool_output 调这个拿到快照关联, 建立
    provenance event ↔ file snapshot 的双向链.
    """
    global _last_step_id
    sid = _last_step_id
    _last_step_id = None
    return sid


def _thread_key(ctx: HookContext) -> str:
    return str(ctx.metadata.get("thread_id") or "_no_thread")


def _resolve_workspace() -> Path:
    if _workspace is not None:
        return _workspace
    # 没显式配就用 cwd —— agent 进程的 cwd 通常就是工作区.
    return Path(os.getcwd())


async def snapshot_pre_hook(ctx: HookContext) -> HookContext | None:
    """PRE_TOOL_USE: 仿真/文件工具执行前给工作区拍照.

    只对 _watched_tools 里的工具拍; 拍照本身出错不能把工具搞挂.
    """
    if ctx.tool_name not in _watched_tools:
        return None
    try:
        ws = _resolve_workspace()
        step_id = SnapshotManager().track(
            ctx.tool_name, ws,
            watch_patterns=list(_watch_patterns) if _watch_patterns else None,
        )
        _pending[_thread_key(ctx)] = step_id
        # 发布 snapshot.take 事件
        _emit("snapshot.take", {"step_id": step_id, "tool": ctx.tool_name},
              ctx.metadata.get("thread_id", ""))
    except Exception:
        logger.warning("snapshot track failed for %s", ctx.tool_name, exc_info=True)
    return None


async def snapshot_post_hook(ctx: HookContext) -> HookContext | None:
    """POST_TOOL_USE: 工具执行后比对文件变化, 产出 patch.

    配对 _pending 里同 thread 的 step_id (pre 存的). 取不到就说明
    这工具没被拍过 (非 watched 或 pre 失败), 直接放行.
    """
    key = _thread_key(ctx)
    step_id = _pending.pop(key, None)
    if not step_id:
        return None
    global _last_step_id
    try:
        SnapshotManager().patch(step_id, _resolve_workspace())
        _last_step_id = step_id
    except Exception:
        logger.warning("snapshot patch failed for %s", ctx.tool_name, exc_info=True)
    return None


def register_snapshot_hooks(
    hm: HookManager,
    workspace: str | os.PathLike[str] | None = None,
    watched_tools: frozenset[str] | None = None,
    watch_patterns: list[str] | None = None,
) -> None:
    """注册 pre/post 快照钩子. 幂等, 重复调用不会重复注册.

    Args:
        hm: HookManager 实例.
        workspace: 工作区路径; None 时钩子用 cwd.
        watched_tools: 要拍照的工具名集合; None 用默认仿真+文件工具集.
        watch_patterns: 盯的文件 glob; None 用 matsci 默认后缀.
    """
    global _workspace, _watched_tools, _watch_patterns
    if getattr(hm, "_snapshot_hooks_registered", False):
        return
    if workspace is not None:
        _workspace = Path(workspace)
    if watched_tools is not None:
        _watched_tools = frozenset(watched_tools)
    if watch_patterns is not None:
        _watch_patterns = tuple(watch_patterns)

    hm.register(PRE_TOOL_USE, snapshot_pre_hook)
    hm.register(POST_TOOL_USE, snapshot_post_hook)
    hm._snapshot_hooks_registered = True
    logger.info(
        "snapshot hooks registered: %d watched tools, workspace=%s",
        len(_watched_tools), _workspace or "<cwd>",
    )
