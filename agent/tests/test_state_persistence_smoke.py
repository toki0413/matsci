"""T5: end-to-end shared-state persistence smoke across restart + workers.

Drives the *real* server_core accessors (``get_or_create_thread``,
``touch_thread``, ``_checkpoints``) through separate subprocesses with
``HUGINN_STATE_BACKEND=sqlite``, proving that sessions/checkpoints written by
one worker survive a process restart and are visible to other workers.

The store-level multiprocess behaviour is covered in
``test_multiprocess_state.py``; this file validates the wired-up contract.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path


def _writer_sub(ready: mp.Event, cache_dir: str) -> None:
    """Create a thread + a checkpoint and signal the parent."""
    os.environ["HUGINN_STATE_BACKEND"] = "sqlite"
    os.environ["HUGINN_CACHE_DIR"] = cache_dir
    import huginn.server_core as sc

    sc.get_or_create_thread("t-smoke", user_id="u1", label="smoke")
    sc._checkpoints["cp-smoke"] = (Path("/tmp/run"), {"state": "converged"})
    ready.set()


def _reader_sub(cache_dir: str, out: mp.Queue) -> None:
    """Fresh process (restart) — read back what the writer persisted."""
    os.environ["HUGINN_STATE_BACKEND"] = "sqlite"
    os.environ["HUGINN_CACHE_DIR"] = cache_dir
    import huginn.server_core as sc

    meta = sc.get_or_create_thread("t-smoke")
    base, snapshot = sc._checkpoints["cp-smoke"]
    out.put(
        {
            "thread_exists": meta is not None,
            "label": meta.get("label"),
            "thread_count": len(sc._threads),
            "cp_exists": "cp-smoke" in sc._checkpoints,
            "cp_base": str(base),
            "cp_state": snapshot.get("state"),
        }
    )


def test_state_persistence_across_restart_and_workers(tmp_path: Path) -> None:
    cache_dir = str(tmp_path / "huginn_cache")
    ctx = mp.get_context("spawn")

    # Worker 1 writes a thread + checkpoint.
    ready = ctx.Event()
    w1 = ctx.Process(target=_writer_sub, args=(ready, cache_dir))
    w1.start()
    ready.wait(timeout=60)
    w1.join(timeout=60)
    assert w1.exitcode == 0

    # "Restart": a brand-new process reads the same sqlite file. The thread
    # label is "smoke" only if the writer's entry was persisted; otherwise
    # get_or_create would have created a fresh one labelled "t-smoke".
    out = ctx.Queue()
    r = ctx.Process(target=_reader_sub, args=(cache_dir, out))
    r.start()
    r.join(timeout=60)
    assert r.exitcode == 0
    result = out.get(timeout=30)

    assert result["thread_exists"] is True
    assert result["label"] == "smoke"
    assert result["thread_count"] >= 1
    assert result["cp_exists"] is True
    assert result["cp_base"] == "/tmp/run"
    assert result["cp_state"] == "converged"