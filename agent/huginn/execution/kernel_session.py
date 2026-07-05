"""有状态 ipykernel 会话 — 持久 Python 内核, 跨多次 execute 保留变量状态.

主路径用 jupyter_client 管理 ipykernel 子进程, 能捕获 matplotlib inline 图像.
ipykernel / jupyter_client 不在时降级到 subprocess + pickle 状态快照,
没有图像捕获但变量状态仍跨调用保留.

SWE-Vision 启发: 把"执行环境"当一等公民, 可查询状态 / 可重启 / 可并发.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty
from typing import Any

logger = logging.getLogger(__name__)

# jupyter_client + ipykernel 都是可选的, 缺了走 subprocess 降级
try:
    import jupyter_client  # type: ignore
    import ipykernel  # type: ignore  # noqa: F401

    _JUPYTER_AVAILABLE = True
except Exception:
    _JUPYTER_AVAILABLE = False


@dataclass
class KernelExecResult:
    """一次 execute 的输出."""

    stdout: str = ""
    stderr: str = ""
    images: list[str] = field(default_factory=list)  # base64 PNG
    error: str | None = None  # 异常 traceback 文本, 没错就是 None
    status: str = "ok"  # ok / error / timeout


class KernelSession:
    """单个 ipykernel 会话, 管理一个持久内核的生命周期."""

    def __init__(
        self,
        kernel_name: str = "python3",
        timeout: float = 30.0,
        session_id: str | None = None,
    ) -> None:
        self.kernel_name = kernel_name
        self.timeout = timeout
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self._started = False
        self._last_active = time.time()

        # jupyter 路径
        self._km: Any = None
        self._kc: Any = None
        # subprocess 降级路径
        self._state_file: str | None = None

    @property
    def alive(self) -> bool:
        return self._started

    @property
    def backend(self) -> str:
        return "jupyter" if _JUPYTER_AVAILABLE and self._km is not None else "subprocess"

    def start(self) -> None:
        """启动内核. jupyter 不可用时降级到 subprocess."""
        if self._started:
            return
        if _JUPYTER_AVAILABLE:
            try:
                self._start_jupyter()
                self._started = True
                return
            except Exception as exc:
                logger.warning("jupyter 内核启动失败, 降级 subprocess: %s", exc)
        self._start_subprocess()
        self._started = True

    # ── jupyter 路径 ─────────────────────────────────────────────

    def _start_jupyter(self) -> None:
        from jupyter_client import KernelManager

        self._km = KernelManager(kernel_name=self.kernel_name)
        self._km.start_kernel()
        self._kc = self._km.client()
        self._kc.start_channels()
        # 等内核就绪, 超时降级
        self._kc.wait_for_ready(timeout=self.timeout)
        # matplotlib inline: 图走 png, 能被我们捕获
        self._jupyter_exec("%matplotlib inline", silent=True)
        self._jupyter_exec(
            "import matplotlib; matplotlib.use('Agg')", silent=True
        )

    def _jupyter_exec(
        self, code: str, silent: bool = False, deadline: float | None = None
    ) -> KernelExecResult:
        if self._kc is None:
            return KernelExecResult(status="error", error="kernel not started")
        timeout = deadline if deadline is not None else self.timeout
        msg_id = self._kc.execute(code, silent=silent)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        images: list[str] = []
        error_tb: str | None = None
        end = time.time() + timeout

        while True:
            remaining = end - time.time()
            if remaining <= 0:
                return KernelExecResult(
                    stdout="".join(stdout_parts),
                    stderr="".join(stderr_parts),
                    images=images,
                    error="execution timeout",
                    status="timeout",
                )
            try:
                msg = self._kc.get_iopub_msg(timeout=remaining)
            except Empty:
                return KernelExecResult(
                    stdout="".join(stdout_parts),
                    stderr="".join(stderr_parts),
                    images=images,
                    error="iopub timeout",
                    status="timeout",
                )
            # 只看本次 execute 的消息
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype = msg["msg_type"]
            content = msg.get("content", {})
            if mtype == "stream":
                text = content.get("text", "")
                if content.get("name") == "stderr":
                    stderr_parts.append(text)
                else:
                    stdout_parts.append(text)
            elif mtype in ("execute_result", "display_data"):
                data = content.get("data", {})
                if "image/png" in data:
                    images.append(data["image/png"])
                if "text/plain" in data:
                    stdout_parts.append(data["text/plain"])
            elif mtype == "error":
                tb = content.get("traceback", [])
                error_tb = "\n".join(tb) if tb else content.get("ename", "error")
            elif mtype == "status" and content.get("execution_state") == "idle":
                break

        return KernelExecResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            images=images,
            error=error_tb,
            status="error" if error_tb else "ok",
        )

    # ── subprocess 降级路径 ──────────────────────────────────────

    def _start_subprocess(self) -> None:
        # 用临时文件存 pickle 的 globals, 跨 execute 保留状态
        fd, path = tempfile.mkstemp(suffix=".pkl", prefix="huginn_kernel_")
        os.close(fd)
        self._state_file = path
        # 空命名空间即可, exec 会自动注入 __builtins__; 不要 pickle
        # __builtins__ 本身——在模块上下文里它是 module, 含 PyCapsule 无法 pickle
        with open(path, "wb") as f:
            pickle.dump({}, f)

    def _subprocess_exec(self, code: str) -> KernelExecResult:
        if not self._state_file:
            return KernelExecResult(status="error", error="subprocess not started")
        # 每次执行: 加载状态 -> exec -> 存回状态 -> 打印输出
        # 用 repr 转义代码, 避免注入
        script = (
            "import pickle, sys, io, traceback\n"
            f"_sf = {self._state_file!r}\n"
            "try:\n"
            "    with open(_sf, 'rb') as _f:\n"
            "        _g = pickle.load(_f)\n"
            "except Exception:\n"
            "    _g = {}\n"
            "_out, _err = io.StringIO(), io.StringIO()\n"
            "_so, _se = sys.stdout, sys.stderr\n"
            "sys.stdout, sys.stderr = _out, _err\n"
            "try:\n"
            f"    exec({code!r}, _g)\n"
            "except Exception:\n"
            "    traceback.print_exc(file=_err)\n"
            "sys.stdout, sys.stderr = _so, _se\n"
            # 存回前剥掉 __builtins__: exec 会把它塞进 _g (module 形态),
            # 而 module 含 PyCapsule 无法 pickle, 必须排除
            "_g.pop('__builtins__', None)\n"
            # 原子写: pickle 到临时文件再 os.replace, 避免 dump 中途失败
            # (globals 里有不可 pickle 的对象如 module/file) 把状态文件写坏,
            # 写坏的文件下次 load 会变成空命名空间, 丢光所有变量
            "import os as _os, tempfile as _tf\n"
            "_fd, _tmp = _tf.mkstemp(dir=_os.path.dirname(_sf) or '.')\n"
            "_os.close(_fd)\n"
            "try:\n"
            "    with open(_tmp, 'wb') as _f:\n"
            "        pickle.dump(_g, _f)\n"
            "    _os.replace(_tmp, _sf)\n"
            "except Exception as _e:\n"
            "    _err.write('warning: state pickle failed: ' + repr(_e) + '\\n')\n"
            "    try: _os.unlink(_tmp)\n"
            "    except OSError: pass\n"
            "sys.stdout.write(_out.getvalue())\n"
            "sys.stderr.write(_err.getvalue())\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return KernelExecResult(
                status="timeout", error="subprocess execution timeout"
            )
        stderr = proc.stderr or ""
        # 用户代码抛异常时 traceback 走 stderr, 据此回填 error 状态,
        # 否则 subprocess 路径永远报 ok, 错误被吞掉
        had_error = "Traceback (most recent call last)" in stderr
        # subprocess 模式拿不到 inline 图像, 只有文本
        return KernelExecResult(
            stdout=proc.stdout or "",
            stderr=stderr,
            images=[],
            error=stderr if had_error else None,
            status="error" if had_error else "ok",
        )

    # ── 公共接口 ─────────────────────────────────────────────────

    def execute(self, code: str, silent: bool = False) -> KernelExecResult:
        """执行代码, 返回输出. 状态跨调用保留."""
        if not self._started:
            self.start()
        self._last_active = time.time()
        if self._km is not None:
            return self._jupyter_exec(code, silent=silent)
        return self._subprocess_exec(code)

    def get_state(self) -> dict[str, Any]:
        """返回当前顶层变量列表 (排除 dunder / 模块内建)."""
        probe = (
            "import json as _j\n"
            "_names = [k for k in list(globals().keys()) "
            "if not k.startswith('_')]\n"
            "print(_j.dumps(_names))\n"
        )
        res = self.execute(probe, silent=True)
        import json

        try:
            names = json.loads(res.stdout.strip().splitlines()[-1])
        except Exception:
            names = []
        # 顺手取每个变量的 repr (best effort, 失败就只给名字)
        state: dict[str, Any] = {}
        for n in names:
            r = self.execute(f"print(repr({n}))", silent=True)
            val = r.stdout.strip()
            if r.status == "error":
                val = "<unrepresentable>"
            state[n] = val
        return state

    def restart(self) -> None:
        """重启内核, 清空所有状态."""
        self.stop()
        self._started = False
        self._km = None
        self._kc = None
        self._state_file = None
        self.start()

    def stop(self) -> None:
        """关闭内核, 释放资源."""
        if self._kc is not None:
            try:
                self._kc.stop_channels()
            except Exception:
                pass
        if self._km is not None:
            try:
                self._km.shutdown_kernel(now=True)
            except Exception:
                pass
        if self._state_file and os.path.exists(self._state_file):
            try:
                os.remove(self._state_file)
            except OSError:
                pass
        self._started = False
        self._km = None
        self._kc = None
        self._state_file = None

    def touch(self) -> None:
        self._last_active = time.time()

    def idle_seconds(self) -> float:
        return time.time() - self._last_active


# ── 会话管理器 ─────────────────────────────────────────────────


class KernelSessionManager:
    """管理多个并发 kernel 会话, 按会话 ID 查找, 超时自动清理空闲会话."""

    def __init__(
        self,
        idle_timeout: float = 1800.0,  # 30 分钟没活动就回收
        cleanup_interval: float = 300.0,
    ) -> None:
        self._sessions: dict[str, KernelSession] = {}
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._cleaner: threading.Thread | None = None

    def create(
        self,
        kernel_name: str = "python3",
        timeout: float = 30.0,
        session_id: str | None = None,
    ) -> KernelSession:
        """创建并启动一个新会话, 返回 session."""
        sess = KernelSession(
            kernel_name=kernel_name,
            timeout=timeout,
            session_id=session_id,
        )
        sess.start()
        with self._lock:
            self._sessions[sess.session_id] = sess
        self._ensure_cleaner()
        return sess

    def get(self, session_id: str) -> KernelSession | None:
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is not None:
                sess.touch()
            return sess

    def close(self, session_id: str) -> bool:
        """关闭并移除一个会话. 返回是否存在过."""
        with self._lock:
            sess = self._sessions.pop(session_id, None)
        if sess is None:
            return False
        sess.stop()
        return True

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "backend": s.backend,
                    "alive": s.alive,
                    "idle_seconds": round(s.idle_seconds(), 1),
                }
                for s in self._sessions.values()
            ]

    def cleanup_idle(self) -> int:
        """清理超过 idle_timeout 没活动的会话, 返回清理数量."""
        reaped = 0
        with self._lock:
            stale = [
                sid for sid, s in self._sessions.items()
                if s.idle_seconds() > self._idle_timeout
            ]
            for sid in stale:
                sess = self._sessions.pop(sid, None)
                if sess is not None:
                    sess.stop()
                    reaped += 1
        if reaped:
            logger.info("kernel session reaper cleaned up %d idle session(s)", reaped)
        return reaped

    def close_all(self) -> None:
        """关闭所有会话 (服务停机时调)."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.stop()
        self._stop_flag.set()

    def _ensure_cleaner(self) -> None:
        """懒启动后台清理线程, 只在有会话时跑."""
        if self._cleaner is not None and self._cleaner.is_alive():
            return
        self._stop_flag.clear()
        self._cleaner = threading.Thread(
            target=self._cleaner_loop, name="kernel-session-reaper", daemon=True
        )
        self._cleaner.start()

    def _cleaner_loop(self) -> None:
        while not self._stop_flag.wait(self._cleanup_interval):
            try:
                self.cleanup_idle()
            except Exception:
                logger.warning("kernel session cleanup failed", exc_info=True)
                break
