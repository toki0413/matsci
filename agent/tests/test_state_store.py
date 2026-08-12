"""Tests for the shared-state SQLite backend (P0-2).

Covers the ``SqliteStore`` MutableMapping contract, checkpoint value
round-tripping, persistence across store instances (process restart), and the
``HUGINN_STATE_BACKEND`` switch in ``server_core``. Multi-process consistency
is covered separately in ``test_multiprocess_state.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.persistence.state_store import (
    SqliteStore,
    decode_checkpoint,
    encode_checkpoint,
)


def test_mutablemapping_contract(tmp_path: Path) -> None:
    store = SqliteStore("t_threads", path=tmp_path / "state.sqlite")
    store["a"] = {"label": "alpha"}
    store["b"] = {"label": "beta"}
    assert "a" in store
    assert len(store) == 2
    # insertion-ordered iteration
    assert list(store) == ["a", "b"]
    assert list(store.values()) == [{"label": "alpha"}, {"label": "beta"}]
    assert store.get("missing") is None
    assert dict(store) == {
        "a": {"label": "alpha"},
        "b": {"label": "beta"},
    }
    # update-in-place on an existing value
    store["a"]["label"] = "gamma"
    assert store["a"]["label"] == "gamma"
    del store["a"]
    assert "a" not in store
    assert len(store) == 1
    with pytest.raises(KeyError):
        del store["a"]
    store.close()


def test_persistence_across_instances(tmp_path: Path) -> None:
    """Values survive the store object being replaced (process restart)."""
    db = tmp_path / "state.sqlite"
    s1 = SqliteStore("t_threads", path=db)
    s1["sess"] = {"thread_id": "abc", "label": "keep-me"}
    s1.close()

    s2 = SqliteStore("t_threads", path=db)
    assert s2["sess"] == {"thread_id": "abc", "label": "keep-me"}
    s2.close()


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    base = Path("/tmp/run-123")
    snapshot = {"state": "converged", "energy": "-5.2"}
    store = SqliteStore(
        "t_checkpoints",
        path=tmp_path / "state.sqlite",
        encode=encode_checkpoint,
        decode=decode_checkpoint,
    )
    store["cp-1"] = (base, snapshot)
    got_base, got_snapshot = store["cp-1"]
    assert got_base == base
    assert got_snapshot == snapshot
    store.close()


def test_pop_and_reversed(tmp_path: Path) -> None:
    store = SqliteStore("t_threads", path=tmp_path / "state.sqlite")
    store["x"] = 1
    store["y"] = 2
    assert list(reversed(store)) == ["y", "x"]
    assert store.pop("x") == 1
    assert list(store) == ["y"]
    assert store.pop("missing", None) is None
    store.close()


def _import_server_core():
    """Import server_core, skipping if the heavy agent stack is unavailable."""
    try:
        import huginn.server_core as sc

        return sc
    except ModuleNotFoundError as e:
        pytest.skip(f"full agent stack not installed: {e}")


def test_make_state_store_memory_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default backend is a plain dict (behaviour unchanged)."""
    sc = _import_server_core()
    monkeypatch.delenv("HUGINN_STATE_BACKEND", raising=False)
    assert type(sc._make_state_store("threads")) is dict
    assert type(sc._make_state_store("checkpoints")) is dict


def test_make_state_store_sqlite_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit sqlite backend returns a SqliteStore for both kinds."""
    sc = _import_server_core()
    for value in ("sqlite", "1", "true", "yes"):
        monkeypatch.setenv("HUGINN_STATE_BACKEND", value)
        threads = sc._make_state_store("threads")
        checkpoints = sc._make_state_store("checkpoints")
        assert isinstance(threads, SqliteStore)
        assert isinstance(checkpoints, SqliteStore)
        threads.close()
        checkpoints.close()
    monkeypatch.delenv("HUGINN_STATE_BACKEND", raising=False)
