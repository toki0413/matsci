"""进程组回收工具 — 统一"超时不留孤儿"的跨平台实现.

动机: 多处 spawn 外部命令 (sandbox 执行、PersistentTerminal session、外部计算
工具) 都用 ``subprocess`` + 超时. 若命令是 ``sh -c '...; python heavy.py'``, 直接
kill 只杀直接子进程, 真正干活的孙进程会 orphan 到 init (PPID→1) 后**继续全速
运行** (run83/84 实测遗留多个 PPID=1 的 python 孤儿, 并拖长收尾). 根因: CPython
的 ``subprocess.run(timeout=...)`` 超时时只 ``kill()`` **直接**子进程, 不给进程组
语义. 因此需要统一"自立新组 + 整组回收".

单一实现, 多处以同一语义复用, 避免跨模块同名重复实现 (MECE 审计关注项).
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from typing import Any

__all__ = [
    "new_group_popen_kwargs",
    "kill_process_group",
    "track_live_child",
    "untrack_live_child",
    "kill_tracked_children",
]


def new_group_popen_kwargs() -> tuple[dict[str, Any], bool]:
    """返回 ``(Popen kwargs, own_group)``: 让子进程自立新会话/进程组.

    posix 用 ``start_new_session``, win32 用 ``CREATE_NEW_PROCESS_GROUP``; 其他
    平台返回 ``({}, False)``. ``own_group`` 供后续 :func:`kill_process_group` 判断
    能否整组回收 (为假时不 killpg, 避免误杀含自身的进程组).
    """
    if os.name == "posix":
        return {"start_new_session": True}, True
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}, True
    return {}, False


def kill_process_group(pid: int, *, own_group: bool) -> bool:
    """尽力整组回收 ``pid`` 所在进程组 (含孙进程); 返回是否已整组处理.

    前提: 子进程以 :func:`new_group_popen_kwargs` / pexpect ``pty.fork`` 自立新组
    → **pgid == pid**, 故可直接按 ``pid`` 整组杀. 安全闸: 若该组就是 agent 自身的
    组 (子进程未成功自立) 则拒绝 —— 绝不 killpg 一个含自身的组.

    返回 ``False`` = 调用方回退默认杀法 (只杀直接子进程). 全部尽力而为, 失败不
    影响主流程.
    """
    if not own_group:
        return False
    if os.name == "posix":
        with contextlib.suppress(OSError):
            if pid == os.getpgid(0):
                return False
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(pid, signal.SIGKILL)
            return True
        return False
    if os.name == "nt":
        with contextlib.suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
            )
            return True
        return False
    return False


# ── 在册子进程 (退出兜底回收) ─────────────────────────────────────────
# 动机: 子进程经 ``start_new_session``/``CREATE_NEW_PROCESS_GROUP`` 自立新组后,
# **父进程退出不会带走它们** —— 它们收不到任何信号, 直接成为 PPID→1 的孤儿继续
# 全速运行 (run87 实测: 主进程退出后实验子进程仍跑 15min). 超时路径已有整组回收,
# 但"父进程先退出"这条路径没有: 唯一解法是退出前主动按进程组杀掉在册子进程.
# 登记表放在本模块, 与 :func:`kill_process_group` 同一处理语义.
_LIVE_CHILDREN: dict[int, bool] = {}
_LIVE_LOCK = threading.Lock()


def track_live_child(pid: int, *, own_group: bool) -> None:
    """登记一个正在运行的子进程, 供进程退出时兜底整组回收."""
    with _LIVE_LOCK:
        _LIVE_CHILDREN[int(pid)] = bool(own_group)


def untrack_live_child(pid: int) -> None:
    """子进程已正常结束/已被回收 → 注销登记 (避免 pid 复用误杀)."""
    with _LIVE_LOCK:
        _LIVE_CHILDREN.pop(int(pid), None)


def kill_tracked_children() -> int:
    """整组回收所有仍在册的子进程, 返回成功处理的个数.

    在进程退出前调用 (CLI 无 lifespan 钩子时由调用方兜底). 幂等: 清空登记表后
    逐个 :func:`kill_process_group`; 已自行退出的 pid 会因 ProcessLookupError 被
    静默跳过. 全部尽力而为, 绝不抛异常打断退出流程.
    """
    with _LIVE_LOCK:
        items = list(_LIVE_CHILDREN.items())
        _LIVE_CHILDREN.clear()
    killed = 0
    for pid, own_group in items:
        if kill_process_group(pid, own_group=own_group):
            killed += 1
    return killed
