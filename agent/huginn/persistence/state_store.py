"""Persistent dict-like stores for shared server state.

Backs the in-memory ``_threads`` / ``_checkpoints`` module globals in
``huginn.server_core`` with an optional SQLite backend so conversation
sessions and checkpoints survive restarts and are shared across multiple
uvicorn workers on the same host.

The default remains an in-memory ``dict`` (behaviour unchanged). Set
``HUGINN_STATE_BACKEND=sqlite`` to enable persistence.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable, Iterator, MutableMapping
from pathlib import Path
from typing import Any

from huginn.utils.runtime import get_runtime_home


def default_db_path() -> Path:
    """Return the per-user SQLite file backing the state stores."""
    home = get_runtime_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / "state.sqlite"


class SqliteStore(MutableMapping[str, Any]):
    """A ``dict``-like view over a single SQLite table, ordered by insertion.

    Values are serialized with a pluggable ``encode``/``decode`` pair
    (default JSON). ``check_same_thread=False`` plus an internal ``RLock`` let
    a FastAPI app touch it from the event loop and thread-pool workers; WAL
    journaling + ``busy_timeout`` let multiple worker processes share the file
    without "database is locked" errors.
    """

    def __init__(
        self,
        table: str,
        *,
        path: str | Path | None = None,
        encode: Callable[[Any], str] = lambda v: json.dumps(v),
        decode: Callable[[str], Any] = lambda s: json.loads(s),
    ) -> None:
        self._table = table
        self._encode = encode
        self._decode = decode
        self._lock = threading.RLock()
        resolved = Path(path) if path else default_db_path()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(resolved), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, seq INTEGER NOT NULL)"
        )
        self._conn.commit()

        # -- in-memory cache of decoded values -------------------------
        # Callers mutate values *in place* (e.g. ``_threads[tid]["label"] = x``,
        # ``meta["last_active"] = now`` in touch_thread). A dict would return a
        # live reference, so we cache decoded objects and hand back the same
        # object on each read — keeping ``__setitem__``/``__delitem__`` durable
        # (SQLite) while preserving dict-style reference semantics in-process.
        self._cache: dict[str, Any] = {}

    # -- MutableMapping core -------------------------------------------

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            row = self._conn.execute(
                f"SELECT value FROM {self._table} WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            raise KeyError(key)
        value = self._decode(row[0])
        with self._lock:
            self._cache[key] = value
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        encoded = self._encode(value)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {self._table} (key, value, seq) "
                "VALUES (?, ?, COALESCE((SELECT MAX(seq) FROM "
                f"{self._table}), 0) + 1) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, encoded),
            )
            self._conn.commit()
            self._cache[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            cur = self._conn.execute(
                f"DELETE FROM {self._table} WHERE key = ?", (key,)
            )
            self._conn.commit()
            self._cache.pop(key, None)
        if cur.rowcount == 0:
            raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key FROM {self._table} ORDER BY seq"
            ).fetchall()
        return (r[0] for r in rows)

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) FROM {self._table}"
            ).fetchone()
        return int(row[0])

    def __contains__(self, key: object) -> bool:
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM {self._table} WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    def __reversed__(self) -> Iterator[str]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT key FROM {self._table} ORDER BY seq DESC"
            ).fetchall()
        return (r[0] for r in rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# -- checkpoint value encoding ------------------------------------------
# A checkpoint value is ``(Path, dict[str, str])``; JSON cannot hold a Path,
# so we encode it explicitly.


def encode_checkpoint(value: tuple[Path, dict[str, str]]) -> str:
    base, snapshot = value
    return json.dumps({"base": str(base), "snapshot": snapshot})


def decode_checkpoint(raw: str) -> tuple[Path, dict[str, str]]:
    data = json.loads(raw)
    return (Path(data["base"]), data["snapshot"])
