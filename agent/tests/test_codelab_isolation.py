"""Code Lab 隔离执行回归 — 超时可回收 + 孙进程不留孤儿.

背景: code_lab 的 ``run()`` 原用超时**子线程**跑, 而 Python 无法 kill 线程 —— 超时
后代码继续吃 CPU/BLAS 线程, 它 spawn 的子进程(torch DataLoader worker 等)更会在
宿主退出后 orphan 到 init 继续全速跑. 现改为**独立子进程**(``start_new_session``
自立新组): 超时 ``killpg`` 整组回收; 子进程写完结果后亦整组自杀, 连带孙进程.

本测试用一个自带原语的临时脚手架, 让被执行的代码真的 spawn 一个长睡孙进程, 验证:
  1. 超时在挂钟上被强制 (不等到孙进程自然结束), 且孙进程被整组回收;
  2. 正常完成路径下, 孙进程同样被回收 (子进程自杀式整组回收);
  3. 跑完后在册子进程表清空 (无残留登记).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from huginn.research.code_lab import load_scaffold, sandbox_run
from huginn.utils import process as proc_util

_SCAFFOLD = '''\
import subprocess
import sys

NAME = "test_spawn"
TEMPLATE = "def run(cfg):\\n    return None\\n"
PRIMITIVE_NAMES = ("spawn_sleeper",)


def primitives():
    def spawn_sleeper(cfg):
        p = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(cfg["pid_file"], "w") as fh:
            fh.write(str(p.pid))
        return {"pid": p.pid}
    return {"spawn_sleeper": spawn_sleeper}
'''

_CODE_HANG = '''\
def run(cfg):
    info = spawn_sleeper(cfg)
    import time as _t
    _t.sleep(120)
    return {"success": True, "summary": {}, "objectives": {"pid": float(info["pid"])}}
'''

_CODE_OK = '''\
def run(cfg):
    info = spawn_sleeper(cfg)
    return {"success": True, "summary": {"pid": info["pid"]},
            "objectives": {"pid": float(info["pid"])}}
'''


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid: int, timeout: float = 8.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def scaffold(tmp_path: Path):
    p = tmp_path / "spawn_scaffold.py"
    p.write_text(_SCAFFOLD, encoding="utf-8")
    return load_scaffold(str(p))


@pytest.mark.skipif(os.name != "posix", reason="POSIX 进程组语义")
def test_timeout_reaps_grandchild(scaffold, tmp_path: Path):
    """超时后孙进程被整组回收, 且挂钟不拖到孙进程自然结束."""
    import time

    pid_file = tmp_path / "pid.txt"
    t0 = time.monotonic()
    res, reason = sandbox_run(
        _CODE_HANG, {"pid_file": str(pid_file)}, timeout=3, scaffold=scaffold)
    elapsed = time.monotonic() - t0
    assert res is None and "超时" in (reason or ""), (res, reason)
    assert elapsed < 30, f"超时未被强制, 挂了 {elapsed:.1f}s"
    pid = int(pid_file.read_text())
    assert _wait_dead(pid), f"孙进程 {pid} 未被回收 (孤儿泄漏)"


@pytest.mark.skipif(os.name != "posix", reason="POSIX 进程组语义")
def test_normal_completion_also_reaps_grandchild(scaffold, tmp_path: Path):
    """正常完成也回收孙进程 —— 子进程写完结果后整组自杀."""
    pid_file = tmp_path / "pid2.txt"
    res, reason = sandbox_run(
        _CODE_OK, {"pid_file": str(pid_file)}, timeout=30, scaffold=scaffold)
    assert res is not None and reason is None, (res, reason)
    pid = int(pid_file.read_text())
    assert _wait_dead(pid), f"正常完成后孙进程 {pid} 仍存活"


def test_isolated_run_returns_real_value(tmp_path: Path):
    """隔离路径照旧跑出真实数值 (不因搬进子进程而失真)."""
    code = '''\
def run(cfg):
    import numpy as np
    xs = np.array([1.0, 2.0, 3.0])
    return {"success": True, "summary": {"m": float(xs.mean())},
            "objectives": {"mean": float(xs.mean())}}
'''
    res, reason = sandbox_run(code, {"seed": 0}, timeout=60)
    assert reason is None, reason
    assert res["objectives"]["mean"] == 2.0


def test_no_leftover_tracked_children(tmp_path: Path):
    """跑完后在册子进程表清空, 不留残项."""
    code = 'def run(cfg):\n    return {"success": True, "summary": {}, "objectives": {"x": 1.0}}\n'
    assert sandbox_run(code, {}, timeout=30)[0] is not None
    assert not proc_util._LIVE_CHILDREN, proc_util._LIVE_CHILDREN


# ── B: 证据链自修复 — 报错必须带回出错行号 (run88) ──────────────────────────
# 背景: run88 书生代码在沙箱报 ``TypeError: int() argument must be ... not list``,
# 但旧实现只回 "类型: 消息", 无行号 → 4 轮自修复全落空 → 整轮零证据 → 回落 DFT
# workflow 造成报告域漂移. 修法: ``_format_exec_error`` 从 traceback 取书生代码
# (<code_act>) 帧, 附上出错行号与附近源码, 让书生能对症定位.


def test_exec_error_reports_line_and_source():
    """沙箱报错须带"出错位置: 代码第 N 行"与出错源码行."""
    from huginn.research.code_lab import _format_exec_error

    code = (
        "def run(cfg):\n"
        "    xs = [1, 2, 3]\n"
        "    bad = int(xs)\n"  # 第 3 行: int() 收到 list
        "    return {'success': True, 'summary': {}, 'objectives': {'x': bad}}\n"
    )
    ns: dict = {}
    exec(compile(code, "<code_act>", "exec"), ns)  # noqa: S102 — 测试自有代码
    try:
        ns["run"]({})
    except Exception as e:
        msg = _format_exec_error(e, code)
    assert "出错位置: 代码第 3 行" in msg, msg
    assert "bad = int(xs)" in msg, msg


def test_sandbox_run_error_carries_line_number():
    """隔离执行路径把"出错行号"如实回灌 (证据链不断裂)."""
    code = (
        "def run(cfg):\n"
        "    xs = [1, 2, 3]\n"
        "    bad = int(xs)\n"  # 第 3 行
        "    return {'success': True, 'summary': {}, 'objectives': {'x': bad}}\n"
    )
    res, reason = sandbox_run(code, {}, timeout=60)
    assert res is None, "失败代码不应产出结果"
    assert reason and "出错位置: 代码第 3 行" in reason, reason


def test_repair_prompt_carries_line_and_advice():
    """自修复提示须含"出错行号"字样与针对 int(list) 的对症建议."""
    from huginn.research.code_lab import build_author_prompt

    hint = (
        "执行异常: TypeError: int() argument must be a string, a bytes-like "
        "object or a real number, not 'list'\n出错位置: 代码第 3 行; 附近源码:\n"
        "   3|>>    bad = int(xs)"
    )
    prompt = build_author_prompt("研究目标 X", repair_hint=hint)
    assert "出错行号" in prompt, "修复提示未强调行号"
    assert "int()" in prompt and "先取元素" in prompt, "缺少 int(list) 对症建议"
    assert "出错位置: 代码第 3 行" in prompt, "未把真实报错原文回灌"
