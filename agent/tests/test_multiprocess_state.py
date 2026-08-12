"""T4: multi-process consistency of the SQLite shared-state backend.

With ``HUGINN_STATE_BACKEND=sqlite``, multiple uvicorn workers on the same host
must share one ``state.sqlite`` (WAL + busy_timeout) without "database is
locked", and reads after concurrent writes must be consistent.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from huginn.persistence.state_store import SqliteStore


def _worker_write(db: str, table: str, start: int, count: int) -> None:
    store = SqliteStore(table, path=db)
    try:
        for i in range(count):
            store[f"k-{start + i}"] = {"proc": start, "i": i}
    finally:
        store.close()


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.sqlite"


def test_multiprocess_concurrent_writes(db_path: Path) -> None:
    """Many processes hammering the same key space never lock the DB."""
    table = "t_threads"
    n_procs = 8
    per = 25
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_worker_write, args=(str(db_path), table, p * per, per))
        for p in range(n_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert all(p.exitcode == 0 for p in procs)

    store = SqliteStore(table, path=db_path)
    try:
        assert len(store) == n_procs * per
        keys = set(store)
        assert len(keys) == n_procs * per  # no lost updates
    finally:
        store.close()


def test_multiprocess_isolation_by_table(db_path: Path) -> None:
    """Thread store and checkpoint store share a file but not tables."""
    from huginn.persistence.state_store import encode_checkpoint

    threads = SqliteStore("t_threads", path=db_path)
    checkpoints = SqliteStore(
        "t_checkpoints", path=db_path, encode=encode_checkpoint
    )
    try:
        threads["sess"] = {"label": "thread"}
        checkpoints["cp"] = (Path("/tmp/x"), {"state": "ok"})
    finally:
        threads.close()
        checkpoints.close()

    threads = SqliteStore("t_threads", path=db_path)
    checkpoints = SqliteStore("t_checkpoints", path=db_path)
    try:
        assert "sess" in threads
        assert "cp" not in threads
        assert "cp" in checkpoints
        assert "sess" not in checkpoints
    finally:
        threads.close()
        checkpoints.close()
