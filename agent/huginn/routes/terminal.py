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
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from huginn.security.auth import require_api_key

router = APIRouter(tags=["terminal"])

logger = logging.getLogger(__name__)


@router.websocket("/ws/terminal")
async def terminal_websocket(websocket: WebSocket):
    """交互式 SSH 终端 WebSocket。

    连接参数走 query string (浏览器 WebSocket 没法发 body):
        ?credential_id=<id>          必填, 从凭据库取 SSH 连接信息
        &cols=80&rows=24             可选, 初始终端尺寸
        &command=bash                可选, 默认用 login shell

    客户端 -> 服务端的消息:
        {"type": "input", "data": "..."}      往 shell 喂字符
        {"type": "resize", "cols": n, "rows": n}  改终端大小
        {"type": "signal", "name": "SIGINT"}  发信号 (Ctrl-C 等)

    服务端 -> 客户端的消息:
        {"type": "output", "data": "..."}      shell 的 stdout/stderr
        {"type": "closed", "exit_code": n}     shell 结束
        {"type": "error", "error": "..."}     出错
    """
    # WebSocket 不走 router 级依赖, 手动鉴权
    try:
        require_api_key(request=None, websocket=websocket)
    except Exception:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # 从 query 取连接参数
    credential_id = websocket.query_params.get("credential_id")
    if not credential_id:
        await websocket.send_json({"type": "error", "error": "缺少 credential_id 参数"})
        await websocket.close()
        return

    cols = int(websocket.query_params.get("cols", "80"))
    rows = int(websocket.query_params.get("rows", "24"))
    command = websocket.query_params.get("command") or ""

    # 从凭据库取 SSH 配置
    from huginn.routes.hpc import _resolve_hpc_config

    cfg, err = _resolve_hpc_config({"credential_id": credential_id})
    if err or cfg is None:
        await websocket.send_json({"type": "error", "error": err or "无法解析 SSH 配置"})
        await websocket.close()
        return

    if not cfg.host or not cfg.username:
        await websocket.send_json({"type": "error", "error": "host 和 username 不能为空"})
        await websocket.close()
        return

    # 用 paramiko 建一条带 PTY 的交互式 shell
    # invoke_shell 是阻塞读写的, 放线程里跑, 用队列桥接
    try:
        channel = await asyncio.to_thread(_open_shell, cfg, cols, rows, command)
    except Exception as exc:
        logger.warning("终端连接失败 (%s): %s", cfg.host, exc)
        await websocket.send_json({"type": "error", "error": f"SSH 连接失败: {exc}"})
        await websocket.close()
        return

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

    async def _pump_input():
        """从 WebSocket 读输入, 喂给 SSH channel。"""
        while True:
            msg = await websocket.receive_text()
            try:
                import json
                data = json.loads(msg)
            except (json.JSONDecodeError, TypeError):
                # 纯文本就直接当输入发
                await asyncio.to_thread(channel.sendall, msg.encode("utf-8", "ignore"))
                continue

            mtype = data.get("type")
            if mtype == "input":
                payload = data.get("data", "")
                await asyncio.to_thread(channel.sendall, payload.encode("utf-8", "ignore"))
            elif mtype == "resize":
                new_cols = int(data.get("cols", cols))
                new_rows = int(data.get("rows", rows))
                await asyncio.to_thread(
                    _safe_resize, channel, new_cols, new_rows
                )
            elif mtype == "signal":
                _send_signal(channel, data.get("name", ""))

    def _read_output():
        """在线程里读 channel 输出, 塞进队列。"""
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
            pass
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                output_queue.put(("error", str(exc))), loop
            )

    import threading

    reader_thread = threading.Thread(target=_read_output, daemon=True)
    reader_thread.start()

    try:
        # 两个任务: 一个读 WS 输入, 一个从队列推输出到 WS
        input_task = asyncio.create_task(_pump_input())
        output_task = asyncio.create_task(_drain_output(websocket, output_queue))

        # 任一任务结束就收尾
        done, pending = await asyncio.wait(
            {input_task, output_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("终端会话异常: %s", exc, exc_info=True)
    finally:
        closed["flag"] = True
        try:
            channel.close()
        except Exception:
            logger.debug("channel.close 收尾失败", exc_info=True)
        try:
            await websocket.close()
        except Exception:
            logger.debug("websocket.close 收尾失败", exc_info=True)


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
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

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
    try:
        channel.resize_pty(width=cols, height=rows)
    except (OSError, EOFError):
        pass


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
        try:
            channel.sendall(payload)
        except (OSError, EOFError):
            pass


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
