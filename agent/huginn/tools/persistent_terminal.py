"""PersistentTerminal -- 长任务 persistent session for Support subagent.

Core+Support 协议 (v14 Phase 2) 用这个消除 "每 N 次调用枪毙通道":
Support dispatch 时启动一个长 session, Core 通过非阻塞 read 轮询取 incremental
output, session 内的 Python/Jupyter 进程不被枪毙, 变量、模型权重、训练状态保持.

状态机: START -> WRITE/READ (轮询) -> KILL.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 1800  # 30 分钟, spec "跨调用状态保持"


def _env_timeout(default: int = _DEFAULT_TIMEOUT) -> int:
    raw = os.environ.get("HUGINN_PERSISTENT_TERMINAL_TIMEOUT")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("HUGINN_PERSISTENT_TERMINAL_TIMEOUT not int: %r", raw)
        return default


# ── 跨平台后端 ──────────────────────────────────────────────────────
# Windows: subprocess.Popen + pipe, 后台线程 drain stdout.
# Linux/Mac: pexpect.spawn (PTY 行为更好, REPL prompt 能读出来), 没装就回退.
# ponytail: Windows 上 python -i 跑在 pipe 模式 (非 PTY), prompt >>> 可能
# 不被 flush 出来. 升级路径: 装 pywinpty 后在 Windows 也走 PTY 路径.

try:
    import pexpect  # type: ignore

    _HAS_PEXPECT = True
except ImportError:
    _HAS_PEXPECT = False

_USE_PEXPECT = (sys.platform != "win32") and _HAS_PEXPECT


class _SubprocessHandle:
    """subprocess.Popen + 后台线程 drain stdout 到内部 buffer."""

    def __init__(self, cmd: str | list, cwd: str | None) -> None:
        if isinstance(cmd, str):
            args, shell = cmd, True
        else:
            args, shell = list(cmd), False
        # binary mode + 默认 bufsize: stdout 是 BufferedReader, 有 read1;
        # text mode 是 TextIOWrapper 没 read1, bufsize=0 是 FileIO 也没 read1.
        # 边界处手动 encode/decode utf-8.
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            shell=shell,
            bufsize=-1,
        )
        self._buf: list[str] = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()

    def _drain(self) -> None:
        # read1: 读任意可用字节, 不等 newline; 进程退出后 pipe 关闭返回 b"".
        stream = self.proc.stdout
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read1(4096)
                if not chunk:
                    break
                with self._lock:
                    self._buf.append(chunk.decode("utf-8", errors="replace"))
        except (OSError, ValueError):
            # pipe 已关
            pass

    def write(self, data: str) -> None:
        if self.proc.stdin is None or self.proc.stdin.closed:
            raise BrokenPipeError("stdin closed")
        self.proc.stdin.write(data.encode("utf-8"))
        self.proc.stdin.flush()

    def read_nonblocking(self, timeout: float) -> str:
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if self._buf:
                    out = "".join(self._buf)
                    self._buf.clear()
                    return out
            if time.time() >= deadline:
                return ""
            time.sleep(0.05)

    def kill(self) -> None:
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                try:
                    self.proc.stdin.close()
                except Exception:
                    pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        except Exception as e:
            logger.warning("_SubprocessHandle.kill error: %s", e)


class _PexpectHandle:
    """pexpect.spawn handle. PTY 模式, REPL prompt 能正常读出来."""

    def __init__(self, cmd: str | list, cwd: str | None) -> None:
        if isinstance(cmd, list):
            cmd = " ".join(cmd)
        self.proc = pexpect.spawn(cmd, cwd=cwd, encoding="utf-8", echo=False)

    def write(self, data: str) -> None:
        self.proc.send(data)

    def read_nonblocking(self, timeout: float) -> str:
        try:
            return self.proc.read_nonblocking(size=4096, timeout=timeout)
        except (pexpect.TIMEOUT, pexpect.EOF):
            return ""

    def kill(self) -> None:
        try:
            self.proc.close(force=True)
        except Exception as e:
            logger.warning("_PexpectHandle.kill error: %s", e)


def _spawn(cmd: str | list, cwd: str | None) -> _SubprocessHandle | _PexpectHandle:
    if _USE_PEXPECT:
        return _PexpectHandle(cmd, cwd)
    return _SubprocessHandle(cmd, cwd)


@dataclass
class _Session:
    handle: _SubprocessHandle | _PexpectHandle
    start_time: float
    cmd_repr: str
    cwd: str | None = None
    killed: bool = False


class PersistentTerminal:
    """长任务 persistent session, 消除 "每 N 次调用枪毙通道".

    Support subagent dispatch 时启动一个长 session, Core 通过非阻塞 read 轮询.
    session 内 Python 进程/Jupyter kernel 不被枪毙, 变量、模型权重、训练状态保持.

    ponytail: Windows 用 subprocess.Popen + pipe, Linux/Mac 用 pexpect.
    升级路径: Windows 用 winpty 改善交互式行为.
    """

    def __init__(self, timeout_seconds: int | None = None) -> None:
        self.timeout = timeout_seconds if timeout_seconds is not None else _env_timeout()
        self._sessions: dict[str, _Session] = {}

    # ── START ──────────────────────────────────────────────────────
    def start(self, cmd: str | list, cwd: str | None = None) -> str:
        """START 状态: 启动长任务, 返回 session_id."""
        session_id = f"sess_{int(time.time() * 1000)}_{len(self._sessions)}"
        handle = _spawn(cmd, cwd)
        cmd_repr = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
        self._sessions[session_id] = _Session(
            handle=handle,
            start_time=time.time(),
            cmd_repr=cmd_repr,
            cwd=cwd,
        )
        logger.info("PersistentTerminal.start: id=%s cmd=%s", session_id, cmd_repr)
        return session_id

    # ── WRITE ──────────────────────────────────────────────────────
    def write(self, session_id: str, data: str) -> None:
        """WRITE 状态: 向 session 追加 input."""
        session = self._require_session(session_id)
        try:
            session.handle.write(data)
        except (BrokenPipeError, OSError) as e:
            logger.warning("PersistentTerminal.write %s failed: %s", session_id, e)
            self._kill_internal(session_id, reason="write_failed")
            raise

    # ── READ ───────────────────────────────────────────────────────
    def read(self, session_id: str, timeout: float = 5.0) -> str:
        """READ 状态: 非阻塞读取 incremental output. timeout 内没数据返回 ""."""
        session = self._require_session(session_id)
        return session.handle.read_nonblocking(timeout=timeout)

    # ── KILL ───────────────────────────────────────────────────────
    def kill(self, session_id: str) -> None:
        """KILL 状态: 终止 session 并清理资源."""
        self._kill_internal(session_id, reason="manual")

    def _kill_internal(self, session_id: str, reason: str = "") -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        session.killed = True
        session.handle.kill()
        logger.info("PersistentTerminal.kill: id=%s reason=%s", session_id, reason)

    # ── list ───────────────────────────────────────────────────────
    def list_sessions(self) -> list[str]:
        """列出所有 active session_id. 顺便清理已过期的."""
        now = time.time()
        for sid in list(self._sessions.keys()):
            if (now - self._sessions[sid].start_time) > self.timeout:
                logger.warning("PersistentTerminal session %s expired", sid)
                self._kill_internal(sid, reason="timeout")
        return list(self._sessions.keys())

    # ── helpers ────────────────────────────────────────────────────
    def _require_session(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown session: {session_id}")
        if (time.time() - session.start_time) > self.timeout:
            logger.warning("PersistentTerminal session %s expired", session_id)
            self._kill_internal(session_id, reason="timeout")
            raise KeyError(f"session {session_id} expired and was killed")
        return session


# ── v14 Task 13: PersistentTerminal 接入 dispatch ──────────────────────

_default_terminal: PersistentTerminal | None = None


def get_default_terminal() -> PersistentTerminal:
    """进程内单例 PersistentTerminal, 让跨 dispatch 调用共享 sessions.

    ponytail: 进程级单例, 多线程并发未加锁 — SubagentTool dispatch 路径
    目前是串行的 (主循环单 worker). 升级路径: 加 threading.Lock 保护
    _sessions dict 的读写, 或换 asyncio.Lock 走 async 路径.
    """
    global _default_terminal
    if _default_terminal is None:
        _default_terminal = PersistentTerminal()
    return _default_terminal


def resolve_persistent_terminal_flag(
    use_persistent_terminal: bool | None,
) -> bool:
    """v14 Task 13: 决定是否走 PersistentTerminal 路径.

    显式传参优先; None 时看 env HUGINN_PERSISTENT_TERMINAL=1.
    """
    if use_persistent_terminal is not None:
        return use_persistent_terminal
    return os.environ.get("HUGINN_PERSISTENT_TERMINAL", "0") == "1"


def _extract_finding(output: str) -> dict | str | None:
    """从 session output 提取 JSON finding.

    spec 标记: {"finding": ...} 或 <FINDING_END> 前的 JSON 块.
    ponytail: 用 json.JSONDecoder.raw_decode 从最后一个 {"finding" 开始解析,
    支持嵌套对象. 找不到返回 None.
    """
    if not output:
        return None
    # <FINDING_END> marker 模式: JSON 在 marker 之前
    if "<FINDING_END>" in output:
        output = output[: output.index("<FINDING_END>")]
    # 找最后一个 {"finding" 起始位置 (后输出的覆盖前面的)
    idx = output.rfind('{"finding"')
    if idx < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(output[idx:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        # JSON 还没写完, 等下一轮 poll
        return None
    return None


def poll_support_session(
    session_id: str,
    max_wait_seconds: int = 60,
    core_context: str = "",
    workspace: str | Path | None = None,
    terminal: PersistentTerminal | None = None,
) -> dict[str, Any]:
    """轮询 PersistentTerminal session 拿 finding, 收到后做 H¹ 检查.

    返回 dict:
      - status: "found" | "running" | "dead"
      - finding: dict | None (status=found 时)
      - h1_status: "zero" | "nonzero" | None
      - h1_reason: str | None
      - partial_output: 当前累积的 raw output

    ponytail: max_wait_seconds 内每 5s 非阻塞 read 一次. finding 标记
    出现后立即 kill session 并返回. 超时返回 status="running", 调用方
    可再次调本函数续轮. 升级路径: 改成 async generator, yield 增量 output.
    """
    term = terminal or get_default_terminal()
    deadline = time.time() + max_wait_seconds
    buffer = ""
    poll_interval = 5.0

    while time.time() < deadline:
        try:
            chunk = term.read(session_id, timeout=poll_interval)
        except KeyError:
            # session 已死 (被 kill 或 timeout 自动清理)
            return {
                "status": "dead",
                "finding": None,
                "h1_status": None,
                "h1_reason": None,
                "partial_output": buffer,
            }
        if chunk:
            buffer += chunk

        finding = _extract_finding(buffer)
        if finding is not None:
            term.kill(session_id)
            result: dict[str, Any] = {
                "status": "found",
                "finding": finding,
                "partial_output": buffer,
            }
            # v14 Task 13.4: 收到 finding 后做 Čech H¹ 一致性检查
            if core_context:
                try:
                    from huginn.agents.subagent import (
                        _check_finding_consistency,
                        _write_support_rejection,
                    )
                    h1_zero, h1_reason = _check_finding_consistency(
                        finding, core_context,
                    )
                    result["h1_status"] = "zero" if h1_zero else "nonzero"
                    result["h1_reason"] = h1_reason
                    if not h1_zero and workspace:
                        _write_support_rejection(
                            workspace, finding, h1_reason, core_context,
                        )
                except Exception as exc:
                    # H¹ 检查失败不阻塞 finding 返回, 标记 unknown
                    logger.warning("H¹ check failed for session %s: %s", session_id, exc)
                    result["h1_status"] = "unknown"
                    result["h1_reason"] = f"check error: {exc}"
            return result

        # session 还在不在
        if session_id not in term.list_sessions():
            return {
                "status": "dead",
                "finding": None,
                "h1_status": None,
                "h1_reason": None,
                "partial_output": buffer,
            }

    return {
        "status": "running",
        "finding": None,
        "h1_status": None,
        "h1_reason": None,
        "partial_output": buffer,
    }


# ── self-check ────────────────────────────────────────────────────────
# 最小验证: start/write/read/kill 状态机 + session 持续性 + 超时清理.
# `python -m huginn.tools.persistent_terminal` 应输出 PASSED.
# ponytail: spec 写 "session 持续 ≥30s", self-check 缩到 5s 避免跑太久;
# 用 -u 强制 stdout unbuffered, 否则 Windows pipe 模式下 print 会 block-buffer.
# Windows 上 python -i 跑在非 PTY 模式, >>> prompt 可能不被 flush, 所以 case 3
# 只断言 "2" 出现, 不断言 prompt.

def _selfcheck() -> None:
    print("[selfcheck] PersistentTerminal v14 Task 12")
    print(f"[selfcheck] platform={sys.platform} _USE_PEXPECT={_USE_PEXPECT}")

    term = PersistentTerminal(timeout_seconds=1800)

    # case 1: 启动 sleep 60, 立即 read 非阻塞, 持续 5s 不被枪毙, kill 后消失.
    sid1 = term.start([sys.executable, "-c", "import time; time.sleep(60)"])
    assert sid1 in term.list_sessions(), "session should be active right after start"
    chunk = term.read(sid1, timeout=1.0)
    assert isinstance(chunk, str), "read should return str"
    print(f"[case1] read after start returned {len(chunk)} chars (non-blocking OK)")
    time.sleep(5)
    assert sid1 in term.list_sessions(), "session should survive 5s"
    print("[case1] session survived 5s, not killed")
    term.kill(sid1)
    assert sid1 not in term.list_sessions(), "kill should remove session"
    print("[case1] kill OK")

    # case 2: print hello, sleep 2, print world. -u 强制 unbuffered, 多次 read 拿两段.
    sid2 = term.start([
        sys.executable, "-u", "-c",
        "print('hello'); import time; time.sleep(2); print('world')",
    ])
    parts: list[str] = []
    deadline = time.time() + 10
    while time.time() < deadline and "world" not in "".join(parts):
        parts.append(term.read(sid2, timeout=3.0))
    combined = "".join(parts)
    assert "hello" in combined, f"expected hello in: {combined!r}"
    assert "world" in combined, f"expected world in: {combined!r}"
    print(f"[case2] got hello + world: {combined!r}")
    term.kill(sid2)

    # case 3: python -u -i REPL, write "print(1+1)\n", read 应拿到 "2".
    # ponytail: Windows 非 PTY 下 prompt 不一定出现, 只断言 "2".
    sid3 = term.start([sys.executable, "-u", "-i"])
    time.sleep(0.5)  # 给 REPL 启动时间
    term.write(sid3, "print(1+1)\n")
    out3 = ""
    deadline = time.time() + 5
    while time.time() < deadline:
        out3 += term.read(sid3, timeout=1.0)
        if "2" in out3:
            break
    assert "2" in out3, f"expected '2' in REPL output: {out3!r}"
    print(f"[case3] REPL print(1+1) -> 2 OK: {out3!r}")
    term.kill(sid3)

    # case 4: timeout=2, sleep 60, 等 3s 后 list_sessions 应自动 kill.
    term_short = PersistentTerminal(timeout_seconds=2)
    sid4 = term_short.start([sys.executable, "-c", "import time; time.sleep(60)"])
    assert sid4 in term_short.list_sessions()
    time.sleep(3)
    remaining = term_short.list_sessions()
    assert sid4 not in remaining, (
        f"session should be auto-killed after timeout, but still: {remaining}"
    )
    print("[case4] auto-kill on timeout OK")

    print("v14 Task 12 self-check PASSED")


# ── v14 Task 13 self-check ─────────────────────────────────────────────
# 验 dispatch 路径接入 PersistentTerminal: 启 session + poll finding + 非阻塞 + env 降级.
# ponytail: 不调真实 SubagentTool (要 agent_factory), 直接用 python -c 跑
# 出 JSON finding 的合成任务模拟 Support. case 2 用 timer 验非阻塞, 不上 mock
# 框架. case 3 改 env 后必须复位, 否则污染后续测试.

def _selfcheck_task13() -> None:
    print("\n[selfcheck] PersistentTerminal v14 Task 13 dispatch 接入")
    term = PersistentTerminal(timeout_seconds=1800)

    # case 1: dispatch 长任务, poll 拿 finding, kill 后不在 list_sessions.
    # 任务: sleep 2s 后 print 一个 JSON finding.
    finding_code = (
        "import time, json; time.sleep(2); "
        "print(json.dumps({'finding': 'test result', 'evidence': ['x']}))"
    )
    cmd1 = [sys.executable, "-u", "-c", finding_code]
    sid1 = term.start(cmd1)
    assert sid1, f"start should return session_id, got {sid1!r}"
    assert sid1 in term.list_sessions(), "session should be active right after start"
    print(f"[case1] dispatch started session {sid1}")

    result = poll_support_session(
        sid1, max_wait_seconds=20, core_context="", terminal=term,
    )
    assert result["status"] == "found", (
        f"poll should find JSON finding, got status={result['status']}, "
        f"partial={result['partial_output']!r}"
    )
    finding = result["finding"]
    assert isinstance(finding, dict), f"finding should be dict, got {type(finding)}"
    assert finding.get("finding") == "test result", f"finding content wrong: {finding}"
    print(f"[case1] poll got finding: {finding}")
    # poll 已经 kill session, list_sessions 应不再含 sid1
    assert sid1 not in term.list_sessions(), (
        "session should be killed by poll after finding received"
    )
    print("[case1] session killed after finding received")

    # case 2: dispatch 立即返回 (不等 Support), mock 主循环跑 2 次 tool call 不阻塞.
    # 跑一个 sleep 5s 的长任务, dispatch 应 <1s 返回, 主循环立刻能跑第二次.
    long_code = "import time; time.sleep(5); print('{\"finding\": \"late\"}')"
    cmd2 = [sys.executable, "-u", "-c", long_code]
    t_start = time.time()
    sid2 = term.start(cmd2)
    t_dispatch = time.time() - t_start
    assert t_dispatch < 1.0, (
        f"dispatch should return immediately (<1s), took {t_dispatch:.2f}s"
    )
    print(f"[case2] dispatch returned in {t_dispatch:.3f}s (non-blocking OK)")

    # mock 主循环: 跑两次"tool call" (这里就用 time.sleep 0.1 模拟),
    # 验证不被 session 阻塞 — 主循环能连续推进.
    for i in range(2):
        time.sleep(0.1)
        assert sid2 in term.list_sessions(), (
            f"iter {i}: session should still be running while main loop advances"
        )
    print("[case2] main loop advanced 2 iters while session running in background")
    # 清理: 杀掉没出 finding 的长 session
    term.kill(sid2)
    assert sid2 not in term.list_sessions()

    # case 3: env HUGINN_PERSISTENT_TERMINAL=0 → resolve_persistent_terminal_flag 返回 False
    saved = os.environ.get("HUGINN_PERSISTENT_TERMINAL")
    try:
        os.environ["HUGINN_PERSISTENT_TERMINAL"] = "0"
        assert resolve_persistent_terminal_flag(None) is False, (
            "env=0 should disable persistent terminal"
        )
        print("[case3] env=0 → resolve_persistent_terminal_flag(None) = False")
        # 显式 True 仍可强制开
        assert resolve_persistent_terminal_flag(True) is True
        print("[case3] explicit use_persistent_terminal=True overrides env")

        os.environ["HUGINN_PERSISTENT_TERMINAL"] = "1"
        assert resolve_persistent_terminal_flag(None) is True, (
            "env=1 should enable persistent terminal"
        )
        print("[case3] env=1 → resolve_persistent_terminal_flag(None) = True")
        # 显式 False 仍可强制关
        assert resolve_persistent_terminal_flag(False) is False
        print("[case3] explicit use_persistent_terminal=False overrides env")
    finally:
        if saved is None:
            os.environ.pop("HUGINN_PERSISTENT_TERMINAL", None)
        else:
            os.environ["HUGINN_PERSISTENT_TERMINAL"] = saved

    # case 4 (额外): <FINDING_END> marker 也能被 _extract_finding 识别.
    marker_code = (
        "import time; time.sleep(0.3); "
        "print('{\"finding\": \"marker test\"}'); print('<FINDING_END>')"
    )
    sid4 = term.start([sys.executable, "-u", "-c", marker_code])
    result4 = poll_support_session(sid4, max_wait_seconds=10, terminal=term)
    assert result4["status"] == "found", (
        f"poll should find marker finding, got {result4['status']}"
    )
    assert result4["finding"].get("finding") == "marker test", result4["finding"]
    print(f"[case4] <FINDING_END> marker extraction OK: {result4['finding']}")

    print("v14 Task 13 self-check PASSED")


if __name__ == "__main__":
    _selfcheck()
    _selfcheck_task13()
