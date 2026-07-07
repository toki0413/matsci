"""SSH 隧道管理路由 —— 仿 MobaXterm 的 MobaSSHTunnel 端口转发功能。

支持三种转发模式:
- local  : 本地端口转发 (对应 ssh -L), 把远程服务映射到本地端口
- remote : 远程端口转发 (对应 ssh -R), 把本地服务暴露给远端
- dynamic: SOCKS5 动态代理 (对应 ssh -D), 按需转发到任意目标

隧道底层用 paramiko 的 Transport 实现, 转发循环跑在独立线程里,
通过 asyncio.to_thread 暴露给异步路由, 这样既不阻塞事件循环,
也不强依赖 asyncssh (项目目前只装了 paramiko)。

状态落盘到 <workspace>/.huginn/tunnels.json, 重启后能恢复配置
(但不自动重连, 避免开机就打一堆 SSH 连接)。
"""

from __future__ import annotations

import json
import logging
import secrets
import socket
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

from huginn.security.auth import require_admin_key

router = APIRouter(tags=["tunnels"], dependencies=[Depends(require_admin_key)])

logger = logging.getLogger(__name__)

TunnelType = Literal["local", "remote", "dynamic"]
TunnelStatus = Literal["stopped", "starting", "running", "error", "reconnecting"]


# ── 请求 / 响应模型 ──────────────────────────────────────────────


class TunnelCreate(BaseModel):
    """创建隧道的请求体。"""

    name: str
    ssh_host: str
    ssh_port: int = 22
    ssh_user: str
    credential_id: str
    local_port: int
    remote_host: str = "127.0.0.1"
    remote_port: int = 0
    tunnel_type: TunnelType = "local"
    bind_address: str = "127.0.0.1"
    # 自动重连: 连接断了之后是否尝试恢复
    auto_reconnect: bool = True
    # 心跳间隔 (秒), 探活 SSH transport 是否还活着
    keepalive_interval: int = 30

    @field_validator("local_port", "ssh_port", "remote_port")
    @classmethod
    def _check_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("端口必须在 1-65535 范围内")
        return v

    @field_validator("tunnel_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        # dynamic 模式不需要 remote_host/remote_port, 其余需要
        if v not in ("local", "remote", "dynamic"):
            raise ValueError("tunnel_type 必须是 local / remote / dynamic")
        return v


# ── 隧道运行时记录 ───────────────────────────────────────────────


@dataclass
class TunnelRecord:
    """单条隧道的配置 + 运行状态, 既用于持久化也用于响应。"""

    tunnel_id: str
    name: str
    ssh_host: str
    ssh_port: int
    ssh_user: str
    credential_id: str
    local_port: int
    remote_host: str
    remote_port: int
    tunnel_type: TunnelType
    bind_address: str = "127.0.0.1"
    auto_reconnect: bool = True
    keepalive_interval: int = 30
    status: TunnelStatus = "stopped"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    # 统计: 转发了多少个连接, 收发了多少字节
    connections: int = 0
    bytes_in: int = 0
    bytes_out: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TunnelRecord:
        # 老数据可能缺字段, 用默认值兜底
        return cls(
            tunnel_id=d["tunnel_id"],
            name=d.get("name", "unnamed"),
            ssh_host=d["ssh_host"],
            ssh_port=d.get("ssh_port", 22),
            ssh_user=d["ssh_user"],
            credential_id=d["credential_id"],
            local_port=d["local_port"],
            remote_host=d.get("remote_host", "127.0.0.1"),
            remote_port=d.get("remote_port", 0),
            tunnel_type=d.get("tunnel_type", "local"),
            bind_address=d.get("bind_address", "127.0.0.1"),
            auto_reconnect=d.get("auto_reconnect", True),
            keepalive_interval=d.get("keepalive_interval", 30),
            status="stopped",  # 重启后一律从 stopped 开始, 不自动连
            error=None,
            created_at=d.get("created_at", time.time()),
        )


# ── 隧道引擎 ─────────────────────────────────────────────────────


def _resolve_credentials(credential_id: str) -> dict[str, Any] | None:
    """从凭据库取 SSH 明文 (host / username / password / key_path)。

    隧道需要真正的明文密码去建立 SSH 连接, 不能用脱敏后的值。
    to_hpc_config 内部已经把 secret 解密填到 HPCConfig.password 里了。
    """
    from huginn.security.credential_store import get_credential_store

    store = get_credential_store()
    cfg = store.to_hpc_config(credential_id)
    if cfg is None:
        return None
    return {
        "host": cfg.host,
        "port": cfg.port,
        "username": cfg.username,
        "password": cfg.password,
        "key_path": cfg.key_path,
    }


class _TunnelWorker:
    """单条隧道的运行时, 封装 paramiko Transport + 转发线程。

    local  : 本地起 socket 监听, 每个连接开一条 direct-tcpip channel
    remote : 让 SSH 服务端监听, 转发到本地 (用 request_port_forward)
    dynamic: 本地起 SOCKS5 监听, 解析目标地址后再开 channel

    线程模型:
    - _forward_loop 跑在 forward_thread 里, 负责 accept + 建连接
    - 每条已建立的连接单独起一个 _pump 线程搬运数据
    - keepalive 线程定期检查 transport 是否活着, 断了就重连
    """

    def __init__(self, record: TunnelRecord, manager: TunnelManager):
        self.record = record
        self.manager = manager
        self._transport: Any = None  # paramiko.Transport
        self._listen_sock: socket.socket | None = None
        self._forward_thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

    # ── 建立 SSH transport ────────────────────────────────────

    def _connect_transport(self) -> Any:
        """用 paramiko 建一条 SSH Transport, 供转发使用。"""
        import paramiko

        cred = _resolve_credentials(self.record.credential_id)
        if cred is None:
            raise RuntimeError(
                f"credential_id '{self.record.credential_id}' 取不到 SSH 凭据"
            )

        # host/user 以请求里传的为准, 没传就用凭据库里的
        host = self.record.ssh_host or cred["host"]
        port = self.record.ssh_port or cred["port"] or 22
        username = self.record.ssh_user or cred["username"]

        sock = socket.create_connection((host, port), timeout=15)
        transport = paramiko.Transport(sock)
        transport.use_compression(compress=True)
        transport.set_keepalive(self.record.keepalive_interval)

        # 认证: 优先用密钥, 没有密钥才用密码
        if cred.get("key_path"):
            try:
                pkey = paramiko.RSAKey.from_private_key_file(cred["key_path"])
            except Exception:
                # 可能是 ed25519 / ecdsa, 逐个试
                for kls in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.DSSKey):
                    try:
                        pkey = kls.from_private_key_file(cred["key_path"])
                        break
                    except Exception:
                        continue
                else:
                    raise RuntimeError(f"无法加载密钥文件: {cred['key_path']}")
            transport.connect(username=username, pkey=pkey)
        elif cred.get("password"):
            transport.connect(username=username, password=cred["password"])
        else:
            # 兜底: 让 paramiko 自己找 agent / known_keys
            transport.connect(username=username)

        return transport

    # ── 启动 / 停止 ────────────────────────────────────────────

    def start(self) -> None:
        """启动隧道: 连 SSH -> 起监听 -> 跑转发循环。"""
        with self._lock:
            if self._transport is not None:
                return  # 已经在跑了

        self._stop_evt.clear()
        self._set_status("starting")

        try:
            self._transport = self._connect_transport()
        except Exception as exc:
            self._set_status("error", str(exc))
            logger.warning("隧道 %s 连接 SSH 失败: %s", self.record.tunnel_id, exc)
            raise

        # 根据类型起监听 socket
        if self.record.tunnel_type in ("local", "dynamic"):
            self._listen_sock = self._bind_local_socket()

        self._set_status("running", started_at=time.time())

        self._forward_thread = threading.Thread(
            target=self._forward_loop,
            name=f"tunnel-{self.record.tunnel_id}-fwd",
            daemon=True,
        )
        self._forward_thread.start()

        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name=f"tunnel-{self.record.tunnel_id}-ka",
            daemon=True,
        )
        self._keepalive_thread.start()

        logger.info(
            "隧道 %s (%s) 已启动: %s:%d -> %s:%d",
            self.record.tunnel_id,
            self.record.tunnel_type,
            self.record.bind_address,
            self.record.local_port,
            self.record.remote_host,
            self.record.remote_port,
        )

    def stop(self) -> None:
        """停止隧道: 关监听 -> 关 transport -> 收线程。"""
        self._stop_evt.set()
        self._set_status("stopped")

        if self._listen_sock is not None:
            try:
                # 关 socket 唤醒 accept 阻塞
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None

        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:
                logger.debug("transport.close 收尾失败", exc_info=True)
            self._transport = None

        # 等线程退出, 不死等
        for t in (self._forward_thread, self._keepalive_thread):
            if t and t.is_alive():
                t.join(timeout=5)
        self._forward_thread = None
        self._keepalive_thread = None

    # ── 监听 socket ────────────────────────────────────────────

    def _bind_local_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.record.bind_address, self.record.local_port))
        sock.listen(100)
        sock.settimeout(1.0)  # 短超时, 方便检查 stop_evt
        return sock

    # ── 转发循环 ────────────────────────────────────────────────

    def _forward_loop(self) -> None:
        """主转发循环: accept 连接 -> 开 channel -> 搬运数据。"""
        if self.record.tunnel_type == "remote":
            # 远程转发: SSH 服务端监听, 不需要本地 accept
            self._remote_forward_loop()
            return

        assert self._listen_sock is not None
        while not self._stop_evt.is_set():
            try:
                client, _ = self._listen_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            # 每个连接单独起线程, 避免一个慢连接拖死整个隧道
            t = threading.Thread(
                target=self._handle_local_connection,
                args=(client,),
                daemon=True,
            )
            t.start()

    def _handle_local_connection(self, client: socket.socket) -> None:
        """处理 local / dynamic 转发的单个连接。"""
        import paramiko

        try:
            if self.record.tunnel_type == "dynamic":
                dest_host, dest_port = self._socks5_handshake(client)
                if dest_host is None:
                    return
            else:
                dest_host = self.record.remote_host
                dest_port = self.record.remote_port

            chan = self._transport.open_channel(
                "direct-tcpip",
                (dest_host, dest_port),
                client.getpeername(),
            )
            if chan is None:
                return

            self.manager._bump_stat(self.record.tunnel_id, "connections", 1)
            self._pump(client, chan)
        except (paramiko.SSHException, OSError, EOFError):
            pass
        finally:
            try:
                client.close()
            except OSError:
                pass

    def _socks5_handshake(self, client: socket.socket) -> tuple[str | None, int]:
        """SOCKS5 握手, 解析客户端要连的目标地址。

        只实现无认证模式, 返回 (host, port); 解析失败返回 (None, 0)。
        """
        try:
            # 握手: 客户端发版本 + 支持的认证方式
            ver = client.recv(1)
            if not ver or ver[0] != 0x05:
                return None, 0
            nmethods = client.recv(1)[0]
            client.recv(nmethods)  # 读掉方法列表
            client.sendall(b"\x05\x00")  # 不需要认证

            # 请求: VER CMD RSV ATYP DST.ADDR DST.PORT
            header = client.recv(4)
            if len(header) < 4 or header[1] != 0x01:
                # 只支持 CONNECT
                client.sendall(b"\x05\x07\x00\x01\x00\x00\x00\x00\x00\x00")
                return None, 0

            atyp = header[3]
            if atyp == 0x01:
                # IPv4
                addr = socket.inet_ntoa(client.recv(4))
            elif atyp == 0x03:
                # 域名
                length = client.recv(1)[0]
                addr = client.recv(length).decode("ascii", errors="ignore")
            elif atyp == 0x04:
                # IPv6
                addr = client.recv(16).hex()
            else:
                client.sendall(b"\x05\x08\x00\x01\x00\x00\x00\x00\x00\x00")
                return None, 0

            port = int.from_bytes(client.recv(2), "big")
            # 回复成功
            client.sendall(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            return addr, port
        except (OSError, IndexError):
            return None, 0

    def _remote_forward_loop(self) -> None:
        """远程转发 (-R): SSH 服务端监听, 把连接转回本地。

        paramiko 的 request_port_forward 注册一个回调, 每当远端收到
        连接就调一次。我们在回调里开 channel 连回本地对应服务。
        """
        import paramiko

        # 远程转发监听的是 ssh 服务端的 remote_port
        remote_port = self.record.remote_port

        def handler(channel: Any, origin: tuple, server: tuple) -> None:
            # channel 是远端过来的, 连回本地的 local_port
            dest = ("127.0.0.1", self.record.local_port)
            try:
                fwd = socket.create_connection(dest, timeout=10)
                self.manager._bump_stat(self.record.tunnel_id, "connections", 1)
                self._pump(fwd, channel)
            except OSError:
                channel.close()

        try:
            self._transport.request_port_forward(
                self.record.remote_host, remote_port, handler
            )
        except paramiko.SSHException as exc:
            self._set_status("error", str(exc))
            return

        # request_port_forward 是非阻塞注册, 主线程只需保持活着
        while not self._stop_evt.is_set():
            if not self._transport.is_active():
                break
            time.sleep(1.0)

        try:
            self._transport.cancel_port_forward(
                self.record.remote_host, remote_port
            )
        except Exception:
            logger.debug("cancel_port_forward 失败", exc_info=True)

    def _pump(self, sock: socket.socket, chan: Any) -> None:
        """在 socket 和 paramiko channel 之间双向搬运数据, 直到任一端断开。"""
        import select

        sock.setblocking(False)
        chan.setblocking(False)

        while not self._stop_evt.is_set():
            try:
                r, _, _ = select.select([sock, chan], [], [], 1.0)
            except (OSError, ValueError):
                break
            if not r:
                continue

            if sock in r:
                try:
                    data = sock.recv(65536)
                except (BlockingIOError, OSError):
                    data = b""
                if not data:
                    break
                try:
                    chan.sendall(data)
                except (OSError, EOFError):
                    break
                self.manager._bump_stat(
                    self.record.tunnel_id, "bytes_in", len(data)
                )

            if chan in r:
                try:
                    data = chan.recv(65536)
                except (OSError, EOFError):
                    break
                if not data:
                    break
                try:
                    sock.sendall(data)
                except (BlockingIOError, OSError):
                    break
                self.manager._bump_stat(
                    self.record.tunnel_id, "bytes_out", len(data)
                )

    # ── 保活 / 重连 ────────────────────────────────────────────

    def _keepalive_loop(self) -> None:
        """定期检查 transport 是否还活着, 断了就尝试重连。"""
        while not self._stop_evt.is_set():
            time.sleep(self.record.keepalive_interval)
            if self._stop_evt.is_set():
                break
            if self._transport is None or not self._transport.is_active():
                if not self.record.auto_reconnect:
                    self._set_status("error", "SSH 连接已断开, 未开启自动重连")
                    break
                # 尝试重连
                self._set_status("reconnecting")
                try:
                    self._transport = self._connect_transport()
                    self._set_status("running")
                    logger.info("隧道 %s 重连成功", self.record.tunnel_id)
                except Exception as exc:
                    self._set_status("error", f"重连失败: {exc}")
                    logger.warning("隧道 %s 重连失败: %s", self.record.tunnel_id, exc)
                    break

    # ── 状态写入 ────────────────────────────────────────────────

    def _set_status(
        self, status: TunnelStatus, error: str | None = None, started_at: float | None = None
    ) -> None:
        self.record.status = status
        if error is not None:
            self.record.error = error
        if status == "running" and started_at is not None:
            self.record.started_at = started_at
        elif status == "stopped":
            self.record.started_at = None
        self.manager._persist()


class TunnelManager:
    """所有隧道的管理中枢: 创建 / 启动 / 停止 / 删除 / 持久化。

    模块级单例, 和 RemoteJobStore 一样走 JSON 落盘。
    """

    def __init__(self, path: str | Path | None = None, workspace: str | Path = "."):
        if path is not None:
            self.path = Path(path).expanduser().resolve()
        else:
            self.path = (
                Path(workspace).expanduser().resolve() / ".huginn" / "tunnels.json"
            )
        self._records: dict[str, TunnelRecord] = {}
        self._workers: dict[str, _TunnelWorker] = {}
        self._lock = threading.Lock()
        self._load()

    # ── 持久化 ────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        rec = TunnelRecord.from_dict(item)
                        self._records[rec.tunnel_id] = rec
        except Exception as exc:
            logger.warning("加载隧道配置 %s 失败: %s", self.path, exc)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with self._lock, tmp.open("w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._records.values()], f, indent=2)
            tmp.replace(self.path)
        except Exception as exc:
            logger.warning("保存隧道配置失败: %s", exc)

    # ── CRUD ────────────────────────────────────────────────────

    def create(self, cfg: TunnelCreate) -> TunnelRecord:
        tid = secrets.token_hex(4)
        rec = TunnelRecord(
            tunnel_id=tid,
            name=cfg.name,
            ssh_host=cfg.ssh_host,
            ssh_port=cfg.ssh_port,
            ssh_user=cfg.ssh_user,
            credential_id=cfg.credential_id,
            local_port=cfg.local_port,
            remote_host=cfg.remote_host,
            remote_port=cfg.remote_port,
            tunnel_type=cfg.tunnel_type,
            bind_address=cfg.bind_address,
            auto_reconnect=cfg.auto_reconnect,
            keepalive_interval=cfg.keepalive_interval,
        )
        self._records[tid] = rec
        self._persist()
        return rec

    def list_all(self) -> list[TunnelRecord]:
        return list(self._records.values())

    def get(self, tunnel_id: str) -> TunnelRecord | None:
        return self._records.get(tunnel_id)

    def delete(self, tunnel_id: str) -> bool:
        self.stop(tunnel_id)
        existed = self._records.pop(tunnel_id, None) is not None
        if existed:
            self._persist()
        return existed

    # ── 生命周期 ────────────────────────────────────────────────

    def start(self, tunnel_id: str) -> tuple[bool, str | None]:
        rec = self._records.get(tunnel_id)
        if rec is None:
            return False, f"隧道 '{tunnel_id}' 不存在"

        with self._lock:
            worker = self._workers.get(tunnel_id)
            if worker is None:
                worker = _TunnelWorker(rec, self)
                self._workers[tunnel_id] = worker
        try:
            worker.start()
            return True, None
        except Exception as exc:
            return False, str(exc)

    def stop(self, tunnel_id: str) -> bool:
        worker = self._workers.get(tunnel_id)
        if worker is not None:
            worker.stop()
            return True
        rec = self._records.get(tunnel_id)
        if rec is not None:
            rec.status = "stopped"
            rec.started_at = None
            self._persist()
        return False

    # ── 内部统计 ────────────────────────────────────────────────

    def _bump_stat(self, tunnel_id: str, field: str, delta: int) -> None:
        rec = self._records.get(tunnel_id)
        if rec is None:
            return
        cur = getattr(rec, field, 0)
        setattr(rec, field, cur + delta)


# 模块级单例 —— 路由直接取, 不每次 new
_manager_singleton: TunnelManager | None = None
_manager_lock = threading.Lock()


def _get_manager() -> TunnelManager:
    global _manager_singleton
    if _manager_singleton is None:
        with _manager_lock:
            if _manager_singleton is None:
                import os

                workspace = Path(os.environ.get("HUGINN_WORKSPACE", "."))
                _manager_singleton = TunnelManager(workspace=workspace)
    return _manager_singleton


# ── 路由 ─────────────────────────────────────────────────────────


@router.post("/tunnels")
async def create_tunnel(cfg: TunnelCreate) -> dict[str, Any]:
    """创建一条新的 SSH 隧道 (默认不自动启动)。"""
    mgr = _get_manager()
    rec = mgr.create(cfg)
    return {"success": True, "tunnel": rec.to_dict()}


@router.get("/tunnels")
async def list_tunnels() -> dict[str, Any]:
    """列出全部隧道及当前状态。"""
    mgr = _get_manager()
    tunnels = [r.to_dict() for r in mgr.list_all()]
    return {"success": True, "tunnels": tunnels, "count": len(tunnels)}


@router.get("/tunnels/{tunnel_id}/status")
async def tunnel_status(tunnel_id: str) -> dict[str, Any]:
    """查询单条隧道的详细状态。"""
    mgr = _get_manager()
    rec = mgr.get(tunnel_id)
    if rec is None:
        return {"success": False, "error": f"隧道 '{tunnel_id}' 不存在"}
    return {"success": True, "tunnel": rec.to_dict()}


@router.post("/tunnels/{tunnel_id}/start")
async def start_tunnel(tunnel_id: str) -> dict[str, Any]:
    """启动 (或重启) 一条隧道。"""
    mgr = _get_manager()
    rec = mgr.get(tunnel_id)
    if rec is None:
        return {"success": False, "error": f"隧道 '{tunnel_id}' 不存在"}
    # 如果已经在跑, 先停掉再起, 走重启语义
    mgr.stop(tunnel_id)
    ok, err = await _async_start(mgr, tunnel_id)
    if not ok:
        return {"success": False, "error": err}
    return {"success": True, "tunnel": rec.to_dict()}


@router.post("/tunnels/{tunnel_id}/stop")
async def stop_tunnel(tunnel_id: str) -> dict[str, Any]:
    """停止一条正在运行的隧道。"""
    mgr = _get_manager()
    rec = mgr.get(tunnel_id)
    if rec is None:
        return {"success": False, "error": f"隧道 '{tunnel_id}' 不存在"}
    mgr.stop(tunnel_id)
    return {"success": True, "tunnel": rec.to_dict()}


@router.delete("/tunnels/{tunnel_id}")
async def delete_tunnel(tunnel_id: str) -> dict[str, Any]:
    """关闭并删除一条隧道。"""
    mgr = _get_manager()
    if mgr.delete(tunnel_id):
        return {"success": True}
    return {"success": False, "error": f"隧道 '{tunnel_id}' 不存在"}


# ── 工具: 把阻塞的 start 丢到线程池 ────────────────────────────


async def _async_start(mgr: TunnelManager, tunnel_id: str) -> tuple[bool, str | None]:
    """SSH 连接是阻塞的, 丢到线程池里跑, 不卡事件循环。"""
    import asyncio

    return await asyncio.to_thread(mgr.start, tunnel_id)
