"""远程终端路由 —— 仿 MobaXterm 内置终端的交互式 SSH 会话。

通过 WebSocket /ws/terminal 提供一个真实的交互式 shell:
- 前端连上后, 发送的每一行都会直接喂给远端 shell (invoke_shell)
- 远端 stdout/stderr 实时推回前端, 支持 vim / top 这类全屏程序
- 支持终端尺寸调整 (resize), 跟着前端窗口大小走

底层用 paramiko 的 Channel.invoke_shell(), 读写是阻塞的, 所以放线程里
跑, 用队列和 asyncio 事件循环衔接。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["terminal"])

logger = logging.getLogger(__name__)


@router.websocket("/ws/terminal")
async def terminal_websocket(websocket: WebSocket):
    """交互式 SSH 终端 WebSocket.

    query: credential_id (必填), cols/rows (默认 80/24), command (默认 login shell).
    客户端 -> 服务端: {"type":"input","data":...} / {"type":"resize","cols":n,"rows":n} /
    {"type":"signal","name":"SIGINT"}
    服务端 -> 客户端: {"type":"output","data":...} / {"type":"closed","exit_code":n} /
    {"type":"error","error":...}
    """
    # WebSocket 不走 router 级依赖, 走统一的 ws_auth_and_track
    from huginn.middleware.ws_governance import get_tracker, ws_auth_and_track

    identity = await ws_auth_and_track(websocket)
    if identity is None:
        return
    tracker = get_tracker()

    params = await _tw_resolve_params(websocket, tracker, identity)
    if params is None:
        return
    cols, rows, command, cfg = params

    # 用 paramiko 建一条带 PTY 的交互式 shell
    # invoke_shell 是阻塞读写的, 放线程里跑, 用队列桥接
    channel = await _tw_open_channel(
        websocket, tracker, identity, cfg, cols, rows, command
    )
    if channel is None:
        return
    await _tw_stream(websocket, tracker, identity, cfg, channel, cols, rows)


async def _tw_stream(
    websocket: WebSocket, tracker: Any, identity: Any, cfg: Any,
    channel: Any, cols: int, rows: int,
) -> None:
    """announce ready → 起读线程 → 双向 relay → 收尾."""
    from huginn.middleware.ws_governance import WSMessageRateLimiter

    await websocket.send_json({
        "type": "ready",
        "host": cfg.host,
        "username": cfg.username,
        "cols": cols,
        "rows": rows,
    })

    loop = asyncio.get_running_loop()
    output_queue: asyncio.Queue = asyncio.Queue()
    # 用来通知读线程该退出了
    closed = {"flag": False}
    rate_limiter = WSMessageRateLimiter()

    reader_thread = threading.Thread(
        target=_tw_read_output, args=(channel, output_queue, loop, closed), daemon=True
    )
    reader_thread.start()

    try:
        # 两个任务: 一个读 WS 输入, 一个从队列推输出到 WS
        await _tw_relay(websocket, channel, output_queue, cols, rows, rate_limiter)
    except WebSocketDisconnect:
        logger.debug("best-effort op failed", exc_info=True)
    except Exception as exc:
        logger.error("终端会话异常: %s", exc, exc_info=True)
    finally:
        await _tw_close(websocket, channel, closed, tracker, identity)


async def _tw_fail(websocket: WebSocket, tracker: Any, identity: Any, error: str) -> None:
    """发错误 → 关连接 → 释放跟踪名额."""
    await websocket.send_json({"type": "error", "error": error})
    await websocket.close()
    tracker.release(identity)


async def _tw_resolve_params(
    websocket: WebSocket, tracker: Any, identity: Any,
) -> tuple[int, int, str, Any] | None:
    """解析 query 参数并从凭据库取 SSH 配置; 出错时回错误并返回 None."""
    credential_id = websocket.query_params.get("credential_id")
    if not credential_id:
        await _tw_fail(websocket, tracker, identity, "缺少 credential_id 参数")
        return None

    cols = int(websocket.query_params.get("cols", "80"))
    rows = int(websocket.query_params.get("rows", "24"))
    command = websocket.query_params.get("command") or ""

    from huginn.routes.hpc import _resolve_hpc_config

    cfg, err = _resolve_hpc_config({"credential_id": credential_id})
    if err or cfg is None:
        await _tw_fail(websocket, tracker, identity, err or "无法解析 SSH 配置")
        return None
    if not cfg.host or not cfg.username:
        await _tw_fail(websocket, tracker, identity, "host 和 username 不能为空")
        return None
    return cols, rows, command, cfg


async def _tw_open_channel(
    websocket: WebSocket, tracker: Any, identity: Any, cfg: Any,
    cols: int, rows: int, command: str,
) -> Any:
    """开交互式 shell channel; 连接失败时回错误并返回 None."""
    try:
        return await asyncio.to_thread(_open_shell, cfg, cols, rows, command)
    except Exception as exc:
        logger.warning("终端连接失败 (%s): %s", cfg.host, exc)
        await _tw_fail(websocket, tracker, identity, f"SSH 连接失败: {exc}")
        return None


async def _tw_relay(
    websocket: WebSocket, channel: Any, output_queue: asyncio.Queue,
    cols: int, rows: int, rate_limiter: Any,
) -> None:
    """跑输入/输出两个任务, 任一结束就取消另一个."""
    input_task = asyncio.create_task(
        _tw_pump_input(websocket, channel, cols, rows, rate_limiter)
    )
    output_task = asyncio.create_task(_drain_output(websocket, output_queue))
    _done, pending = await asyncio.wait(
        {input_task, output_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for t in pending:
        t.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await t


async def _tw_pump_input(
    websocket: WebSocket, channel: Any, cols: int, rows: int, rate_limiter: Any,
) -> None:
    """从 WebSocket 读输入, 喂给 SSH channel."""
    while True:
        msg = await websocket.receive_text()
        if not rate_limiter.check():
            await websocket.send_json({"type": "error", "error": "消息频率过高, 请放慢"})
            continue
        try:
            import json
            data = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            # 纯文本就直接当输入发
            await asyncio.to_thread(channel.sendall, msg.encode("utf-8", "ignore"))
            continue
        await _tw_handle_message(channel, data, cols, rows)


async def _tw_handle_message(channel: Any, data: dict, cols: int, rows: int) -> None:
    """按消息类型分派: input / resize / signal."""
    mtype = data.get("type")
    if mtype == "input":
        payload = data.get("data", "")
        await asyncio.to_thread(channel.sendall, payload.encode("utf-8", "ignore"))
    elif mtype == "resize":
        new_cols = int(data.get("cols", cols))
        new_rows = int(data.get("rows", rows))
        await asyncio.to_thread(_safe_resize, channel, new_cols, new_rows)
    elif mtype == "signal":
        _send_signal(channel, data.get("name", ""))


def _tw_read_output(channel: Any, output_queue: asyncio.Queue, loop: Any, closed: dict) -> None:
    """在线程里读 channel 输出, 塞进队列."""
    try:
        while not closed["flag"]:
            # recv 是阻塞的, 拿到一块就推
            data = channel.recv(4096)
            if not data:
                break
            asyncio.run_coroutine_threadsafe(
                output_queue.put(("output", data.decode("utf-8", "ignore"))),
                loop,
            )
        # channel 关了, 取退出码
        exit_code = channel.recv_exit_status() if not closed["flag"] else None
        asyncio.run_coroutine_threadsafe(
            output_queue.put(("closed", exit_code)), loop
        )
    except (OSError, EOFError):
        logger.debug("best-effort op failed", exc_info=True)
    except Exception as exc:
        asyncio.run_coroutine_threadsafe(
            output_queue.put(("error", str(exc))), loop
        )


async def _tw_close(
    websocket: WebSocket, channel: Any, closed: dict, tracker: Any, identity: Any,
) -> None:
    """收尾: 标记关闭 → 关 channel → 关 WS → 释放跟踪名额."""
    closed["flag"] = True
    try:
        channel.close()
    except Exception:
        logger.debug("channel.close 收尾失败", exc_info=True)
    try:
        await websocket.close()
    except Exception:
        logger.debug("websocket.close 收尾失败", exc_info=True)
    tracker.release(identity)


# ── 内部工具函数 ─────────────────────────────────────────────────


def _open_shell(cfg: Any, cols: int, rows: int, command: str):
    """用 paramiko 打开一个交互式 shell channel (带 PTY)。

    阻塞调用, 必须在线程里跑。返回的 channel 已经连好, 可以直接 recv/send。
    """
    import paramiko

    client = paramiko.SSHClient()
    # 终端场景放宽 host key 校验: 很多集群第一次连还没 known_hosts
    if cfg.strict_host_key_checking:
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.load_system_host_keys()
    else:
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # nosec B507

    connect_kwargs = {
        "hostname": cfg.host,
        "username": cfg.username,
        "port": cfg.port,
        "timeout": 15,
        "look_for_keys": True,
    }
    if cfg.key_path:
        connect_kwargs["key_filename"] = cfg.key_path
    elif cfg.password:
        connect_kwargs["password"] = cfg.password

    client.connect(**connect_kwargs)
    transport = client.get_transport()
    if transport is None:
        raise RuntimeError("SSH transport 建立失败")

    channel = transport.open_session()
    # 申请 PTY, 让交互式程序 (vim/top) 能正常工作
    channel.get_pty(term="xterm-256color", width=cols, height=rows)
    # invoke_shell 开一个长连接的交互式 shell, 不会执行完就退出。
    # 注意: 不能同时调 exec_command, 两者互斥; 终端场景要的是持续会话。
    channel.invoke_shell()
    channel.settimeout(None)  # recv 永久阻塞等数据
    return channel


def _safe_resize(channel: Any, cols: int, rows: int) -> None:
    """安全地改终端尺寸, channel 不在就忽略。"""
    with contextlib.suppress(OSError, EOFError):
        channel.resize_pty(width=cols, height=rows)


def _send_signal(channel: Any, name: str) -> None:
    """给远端进程发信号 (目前只支持常见的几个)。"""
    # paramiko Channel 没有直接的 send_signal, 用 SIGINT 走 Ctrl-C 字符
    # 大多数 shell 收到 0x03 就会中断当前前台进程
    signal_map = {
        "SIGINT": b"\x03",
        "SIGQUIT": b"\x1c",
        "EOF": b"\x04",
    }
    payload = signal_map.get(name.upper())
    if payload:
        with contextlib.suppress(OSError, EOFError):
            channel.sendall(payload)


async def _drain_output(websocket: WebSocket, queue: asyncio.Queue) -> None:
    """从输出队列取消息, 推给 WebSocket, 直到收到 closed/error。"""
    while True:
        kind, content = await queue.get()
        if kind == "output":
            # send_json 已经做 JSON 序列化, 这里直接传 dict
            await websocket.send_json({
                "type": "output",
                "data": content,
            })
        elif kind == "closed":
            await websocket.send_json({
                "type": "closed",
                "exit_code": content,
            })
            break
        elif kind == "error":
            await websocket.send_json({"type": "error", "error": content})
            break
