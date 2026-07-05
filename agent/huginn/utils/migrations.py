"""SQLite migration framework using PRAGMA user_version.

Previously, schema changes were scattered across 6+ files as
`CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ADD COLUMN` wrapped in
`suppress(OperationalError)`. There was no version tracking, so it was
impossible to tell which migrations had been applied to a given database.

This module provides a lightweight migration system:

- SQLite's built-in `PRAGMA user_version` tracks the schema version (an
  integer stored in the database header, persists across connections).
- Each migration is a plain function `def migrate(conn): ...` that applies
  its schema changes on a `sqlite3.Connection`.
- `MigrationManager.run_migrations()` runs pending migrations in version
  order, each inside its own transaction. On success, `user_version` is
  bumped. On failure, the transaction rolls back and the old version is
  preserved.

Usage::

    from huginn.utils.migrations import MigrationManager

    mgr = MigrationManager(db_path)
    mgr.run_migrations([
        (1, migrate_v0_to_v1),
        (2, migrate_v1_to_v2),
    ])
    mgr.close()

Migration functions should NOT call `conn.commit()` -- the manager handles
that. They also should be idempotent where possible (check column existence
before adding), to handle databases that were partially migrated by the
old suppress(OperationalError) approach.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

# A migration is a (target_version, callable) tuple. The callable receives
# an open sqlite3.Connection and applies schema changes. It must not commit
# or rollback -- the manager wraps it in a transaction.
Migration = tuple[int, Callable[[sqlite3.Connection], None]]


def _get_user_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _set_user_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA user_version doesn't accept a parameter placeholder, so we
    # interpolate the int directly. version always comes from our own
    # code (never user input), so there's no injection risk.
    conn.execute(f"PRAGMA user_version = {int(version)}")


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether a column exists on a table.

    Migration functions should call this before ALTER TABLE ADD COLUMN
    to stay idempotent -- old databases may have been partially migrated
    by the previous suppress(OperationalError) approach.
    """
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check whether a table exists in the database."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


class MigrationManager:
    """Runs versioned schema migrations on a SQLite database.

    Each migration runs in its own transaction. If the migration function
    raises, the transaction rolls back and user_version is left unchanged,
    so a half-applied migration doesn't leave the DB in an inconsistent
    state.

    The manager opens its own connection (with WAL mode) and closes it
    when done. This keeps it independent of whatever connection the caller
    is using for its own work.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        # WAL so the manager's writes don't block readers (and vice versa)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._current_version = _get_user_version(self._conn)
        logger.debug(
            "MigrationManager: %s at version %d",
            self.db_path,
            self._current_version,
        )

    @property
    def current_version(self) -> int:
        """The schema version as recorded by PRAGMA user_version."""
        return self._current_version

    def run_migrations(self, migrations: Sequence[Migration]) -> int:
        """Run all pending migrations in version order.

        Returns the new schema version. Migrations whose target version
        is <= the current version are skipped.

        Each migration is preceded by an online backup (``.bak_v<n>``) and
        followed by a table/row-count sanity check that only logs warnings.
        ``schema_meta`` is stamped with the app version + migration timestamp
        inside the same transaction.

        Raises the original exception if a migration fails (after rolling
        back its transaction).
        """
        pending = [
            (v, fn)
            for v, fn in sorted(migrations, key=lambda m: m[0])
            if v > self._current_version
        ]
        if not pending:
            return self._current_version

        # schema_meta holds app version + migration log; PRAGMA user_version
        # still tracks the schema version. Created once up front (idempotent)
        # so per-migration stamps can INSERT into it inside the txn.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        self._conn.commit()

        for target_version, func in pending:
            logger.info(
                "running migration to v%d on %s", target_version, self.db_path
            )
            # Snapshot the pre-migration state so rollback_to() can restore it.
            self._backup(self._current_version)
            before = self._snapshot()
            try:
                # `with conn` is a transaction context: commit on clean exit,
                # rollback on exception. We set user_version INSIDE the
                # transaction so a crash mid-migration doesn't bump the version.
                with self._conn:
                    func(self._conn)
                    _set_user_version(self._conn, target_version)
                    self._record_migration(target_version)
            except Exception:
                logger.exception(
                    "migration to v%d failed on %s",
                    target_version,
                    self.db_path,
                )
                raise
            self._current_version = target_version
            self._check_integrity(before, target_version)
            logger.info(
                "%s migrated to v%d", self.db_path, target_version
            )

        return self._current_version

    # -- pre/post migration safety nets (backups + integrity checks) --

    def _backup(self, version: int) -> Path | None:
        """Online-backup the DB to ``<db_path>.bak_v<version>`` before migrating away.

        ``version`` is the *current* schema version (the state being saved),
        so ``rollback_to(version)`` restores exactly this snapshot. Uses
        sqlite3's online backup API so concurrent readers aren't blocked.
        """
        bak_path = Path(f"{self.db_path}.bak_v{version}")
        try:
            dst = sqlite3.connect(str(bak_path))
            try:
                with dst:
                    self._conn.backup(dst)
            finally:
                dst.close()
            logger.debug("backed up %s -> %s", self.db_path, bak_path)
            return bak_path
        except Exception:
            # backup is a safety net, not a gate — log and proceed
            logger.warning("backup failed for %s", self.db_path, exc_info=True)
            return None

    def _snapshot(self) -> dict:
        """Capture table count + per-table row counts for the integrity check.

        Table names come from sqlite_master (never user input), so the
        f-string interpolation is safe the same way column_exists is.
        """
        snap: dict = {}
        try:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            names = [r[0] for r in rows]
            snap["__table_count__"] = len(names)
            for name in names:
                # FTS shadow tables / internal stuff may not support COUNT
                try:
                    snap[name] = int(
                        self._conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                    )
                except sqlite3.OperationalError:
                    pass
        except sqlite3.OperationalError:
            pass
        return snap

    def _check_integrity(self, before: dict, version: int) -> None:
        """Warn (don't fail) if a migration looks destructive.

        Catches two obvious failure modes: all tables vanished (bad
        DROP/RENAME), or a known table lost more than half its rows (bad
        DELETE/WHERE). Warnings only — the migration already committed.
        """
        after = self._snapshot()
        if before.get("__table_count__", 0) > 0 and after.get("__table_count__", -1) == 0:
            logger.warning(
                "v%d on %s: table count dropped to 0 — migration likely wiped schema",
                version, self.db_path,
            )
        for table, old_count in before.items():
            if table == "__table_count__" or not isinstance(old_count, int):
                continue
            new_count = after.get(table)
            if new_count is None or new_count >= old_count:
                continue
            if old_count > 0 and new_count < old_count * 0.5:
                logger.warning(
                    "v%d on %s: %s rows %d -> %d (>50%% drop, accidental DELETE?)",
                    version, self.db_path, table, old_count, new_count,
                )

    # -- schema_meta: app version + migration history --

    def _record_migration(self, target_version: int) -> None:
        """Stamp app version + migration timestamp into schema_meta.

        Runs inside the migration transaction (no explicit commit) so a
        failed migration doesn't leave a bogus 'migrated_to_vN' row.
        """
        try:
            from huginn import __version__ as app_version
        except Exception:
            app_version = "unknown"
        now = datetime.now().isoformat()
        self._conn.executemany(
            "INSERT OR REPLACE INTO schema_meta (key, value, updated_at) VALUES (?, ?, ?)",
            [
                ("app_version", app_version, now),
                ("schema_version", str(target_version), now),
                (f"migrated_to_v{target_version}", now, now),
            ],
        )

    def close(self) -> None:
        """Close the internal connection."""
        self._conn.close()

    def __enter__(self) -> MigrationManager:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def register_migration(
    db_path: str, version: int, func: Callable[[sqlite3.Connection], None]
) -> None:
    """Convenience: open a manager, run a single migration, close.

    Handy when a module just wants to ensure one specific migration ran
    without building the full migration list itself.
    """
    with MigrationManager(db_path) as mgr:
        mgr.run_migrations([(version, func)])


# (default filename, module path, migration fn name) for each SQLite store.
# Imports are deferred so a missing/circular module can't abort the sweep.
_STORE_MIGRATIONS = [
    ("memory.db", "huginn.memory.longterm", "_migrate_memories_v1"),
    ("anomalies.db", "huginn.anomaly_log", "_migrate_anomaly_log_v1"),
    ("research_log.sqlite", "huginn.research_log", "_migrate_research_log_v1"),
]


def run_all_migrations() -> dict[str, int]:
    """Run pending migrations for every SQLite store, once at startup.

    Call this from the app lifespan before any store is instantiated, so
    backups + integrity checks happen up front rather than lazily when each
    store is first touched. DBs that don't exist yet are skipped — the store
    will create + migrate its own on first use (idempotent by design).

    Returns ``{db_path: new_version}``; ``-1`` means that store failed.

    ponytail: path resolution is best-effort — it uses get_runtime_home(),
    which matches the default for memory + research_log. anomaly_log's path
    can be overridden by checkpointer_path in the agent factory; if so, the
    factory still runs its own migration on init, so we just skip it here.
    """
    import importlib

    from huginn.utils.runtime import get_runtime_home

    home = get_runtime_home()
    results: dict[str, int] = {}
    for filename, mod_name, fn_name in _STORE_MIGRATIONS:
        db_path = str(home / filename)
        if not Path(db_path).exists():
            continue
        try:
            mod = importlib.import_module(mod_name)
            migrate_fn = getattr(mod, fn_name)
        except Exception:
            logger.debug("migration fn for %s unavailable", filename, exc_info=True)
            continue
        try:
            with MigrationManager(db_path) as mgr:
                results[db_path] = mgr.run_migrations([(1, migrate_fn)])
        except Exception:
            logger.warning("migrations failed for %s", db_path, exc_info=True)
            results[db_path] = -1
    return results


def rollback_to(db_path: str, target_version: int) -> bool:
    """Restore ``db_path`` from ``<db_path>.bak_v<target_version>``.

    Manual recovery — call this with the app stopped, not as part of normal
    operation. Returns False if no backup exists for that version.

    ponytail: plain file copy, not the online backup API — rollback is a
    maintenance op done while the app is stopped. If a live connection is
    holding the DB the copy still succeeds at the file level but that
    connection won't see the new content until it reconnects.
    """
    bak_path = Path(f"{db_path}.bak_v{target_version}")
    if not bak_path.exists():
        logger.warning("no backup at %s", bak_path)
        return False
    shutil.copy2(str(bak_path), str(db_path))
    logger.info("rolled back %s to v%d from %s", db_path, target_version, bak_path)
    return True


__all__ = [
    "Migration",
    "MigrationManager",
    "column_exists",
    "table_exists",
    "register_migration",
    "run_all_migrations",
    "rollback_to",
]


if __name__ == "__main__":
    # Smallest check that fails if the core logic breaks: backup gets
    # created, migration applies + stamps schema_meta, rollback restores.
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "t.db")
        c = sqlite3.connect(db)
        c.executescript(
            "CREATE TABLE widgets(id INTEGER PRIMARY KEY, name TEXT);"
            "INSERT INTO widgets(name) VALUES('a'),('b'),('c'),('d'),('e');"
        )
        c.commit()
        c.close()

        def add_sku(conn: sqlite3.Connection) -> None:
            if not column_exists(conn, "widgets", "sku"):
                conn.execute("ALTER TABLE widgets ADD COLUMN sku TEXT")

        mgr = MigrationManager(db)
        assert mgr.run_migrations([(1, add_sku)]) == 1
        mgr.close()

        # pre-migration backup + schema_meta stamp
        assert Path(f"{db}.bak_v0").exists(), "backup missing"
        c = sqlite3.connect(db)
        c.row_factory = sqlite3.Row
        meta = {r["key"] for r in c.execute("SELECT key FROM schema_meta")}
        assert "app_version" in meta and "migrated_to_v1" in meta, meta
        assert column_exists(c, "widgets", "sku")
        assert c.execute("SELECT COUNT(*) FROM widgets").fetchone()[0] == 5
        c.close()

        # rollback restores the v0 snapshot (no sku column, version 0)
        assert rollback_to(db, 0)
        c = sqlite3.connect(db)
        assert _get_user_version(c) == 0
        assert not column_exists(c, "widgets", "sku")
        c.close()

        # a destructive migration must not crash (integrity check logs only)
        def drop_rows(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM widgets WHERE name != 'a'")

        mgr = MigrationManager(db)
        mgr.run_migrations([(2, drop_rows)])
        mgr.close()

    print("migrations self-check OK")
