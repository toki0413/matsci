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
import sqlite3
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

        for target_version, func in pending:
            logger.info(
                "running migration to v%d on %s", target_version, self.db_path
            )
            try:
                # `with conn` is a transaction context: commit on clean exit,
                # rollback on exception. We set user_version INSIDE the
                # transaction so a crash mid-migration doesn't bump the version.
                with self._conn:
                    func(self._conn)
                    _set_user_version(self._conn, target_version)
            except Exception:
                logger.exception(
                    "migration to v%d failed on %s",
                    target_version,
                    self.db_path,
                )
                raise
            self._current_version = target_version
            logger.info(
                "%s migrated to v%d", self.db_path, target_version
            )

        return self._current_version

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


__all__ = [
    "Migration",
    "MigrationManager",
    "column_exists",
    "table_exists",
    "register_migration",
]
