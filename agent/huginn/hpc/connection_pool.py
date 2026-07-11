"""SSH connection pool — reuses paramiko connections across poll cycles.

Without a pool, the JobMonitor opens a fresh SSH connection on every poll
(30s for freshly submitted jobs), which is wasteful and can trip rate
limits on strict SSH servers.  This pool keeps idle connections around
for reuse and evicts them after a configurable idle timeout.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from huginn.hpc.client import HPCClient, HPCConfig

logger = logging.getLogger(__name__)


@dataclass
class _PooledConn:
    """A pooled SSH connection with bookkeeping."""

    client: HPCClient
    last_used: float = field(default_factory=time.time)
    in_use: bool = False


class SSHConnectionPool:
    """Thread-safe pool of SSH connections keyed by (host, user, port).

    Connections are created on demand and returned to the pool on release.
    A background daemon thread closes connections that have been idle for
    longer than ``idle_timeout`` seconds (default 5 min).
    """

    def __init__(self, max_per_host: int = 8, idle_timeout: float = 300):
        self._max_per_host = max_per_host
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()
        # (host, username, port) -> list of pooled connections
        self._pool: dict[tuple[str, str, int], list[_PooledConn]] = {}
        self._stop_event = threading.Event()
        self._cleaner: threading.Thread | None = None
        self._start_cleaner()

    # ── internal helpers ──────────────────────────────────────────

    @staticmethod
    def _key(config: HPCConfig) -> tuple[str, str, int]:
        return (config.host, config.username, config.port)

    @staticmethod
    def _is_healthy(client: HPCClient) -> bool:
        """True if the underlying SSH transport is still alive."""
        ssh = client._ssh
        if ssh is None:
            return False
        try:
            transport = ssh.get_transport()
            if transport is None or not transport.is_active():
                return False
        except Exception:
            return False
        return True

    def _discard(self, conn: _PooledConn) -> None:
        """Disconnect and forget a pooled connection."""
        try:
            conn.client.disconnect()
        except Exception:
            logger.debug("disconnect failed during discard", exc_info=True)

    # ── public API ────────────────────────────────────────────────

    def get_client(self, config: HPCConfig) -> HPCClient:
        """Return a connected, healthy client — from the pool if possible."""
        key = self._key(config)
        with self._lock:
            conns = self._pool.get(key)
            if conns:
                while conns:
                    conn = conns.pop()
                    if self._is_healthy(conn.client):
                        conn.in_use = True
                        conn.last_used = time.time()
                        # Keep the (possibly now-empty) list around so
                        # release_client can append back to it.
                        return conn.client
                    # dead connection — throw it away
                    self._discard(conn)

        # Nothing healthy in the pool — create a fresh one.
        # Done outside the lock so we don't block other callers during
        # the TCP handshake (can take a few seconds on slow networks).
        client = HPCClient(config)
        try:
            client.connect()
        except ImportError as exc:
            raise RuntimeError(
                "paramiko is not installed — run: pip install paramiko"
            ) from exc
        return client

    def release_client(self, config: HPCConfig, client: HPCClient) -> None:
        """Return a borrowed client to the pool for reuse.

        If the pool for this host is already at capacity, or the
        connection is no longer healthy, the client is closed instead.
        """
        key = self._key(config)
        with self._lock:
            conns = self._pool.setdefault(key, [])
            if len(conns) >= self._max_per_host:
                self._discard(_PooledConn(client=client))
                return
            if not self._is_healthy(client):
                self._discard(_PooledConn(client=client))
                return
            conns.append(_PooledConn(client=client, last_used=time.time()))

    @contextmanager
    def borrow(self, config: HPCConfig):
        """Borrow a connection, return it to the pool on exit."""
        client = self.get_client(config)
        try:
            yield client
        finally:
            self.release_client(config, client)

    # ── background cleanup ────────────────────────────────────────

    def _start_cleaner(self) -> None:
        self._cleaner = threading.Thread(
            target=self._clean_loop, daemon=True, name="ssh-pool-cleaner"
        )
        self._cleaner.start()

    def _clean_loop(self) -> None:
        """Wake up every 60s and close idle connections."""
        while not self._stop_event.wait(60.0):
            try:
                self._sweep_idle()
            except Exception:
                logger.debug("pool cleanup error", exc_info=True)

    def _sweep_idle(self) -> None:
        now = time.time()
        with self._lock:
            for key, conns in list(self._pool.items()):
                keep: list[_PooledConn] = []
                for conn in conns:
                    if conn.in_use:
                        keep.append(conn)
                        continue
                    if now - conn.last_used > self._idle_timeout:
                        self._discard(conn)
                    else:
                        keep.append(conn)
                if keep:
                    self._pool[key] = keep
                else:
                    del self._pool[key]

    def shutdown(self) -> None:
        """Close every pooled connection and stop the cleaner thread."""
        self._stop_event.set()
        with self._lock:
            for conns in self._pool.values():
                for conn in conns:
                    self._discard(conn)
            self._pool.clear()


# ── module-level singleton ──────────────────────────────────────

_pool: SSHConnectionPool | None = None


def get_pool() -> SSHConnectionPool:
    global _pool
    if _pool is None:
        _pool = SSHConnectionPool()
    return _pool
