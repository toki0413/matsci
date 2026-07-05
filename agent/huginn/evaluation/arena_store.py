"""Arena 比较历史的 SQLite 存储.

记录每场双盲比较的: 时间, 模型A, 模型B, 胜者, 评判理由, 双方 ELO.
单文件 SQLite, 零外部依赖, 跟项目里其它本地存储一个风格.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ArenaRecord:
    """一条 arena 比较记录."""

    timestamp: float
    model_a: str
    model_b: str
    winner: str  # "A" / "B" / "tie"
    reasoning: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    elo_a: float = 1000.0
    elo_b: float = 1000.0
    meta: dict[str, Any] = field(default_factory=dict)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS arena_battles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   REAL NOT NULL,
    model_a     TEXT NOT NULL,
    model_b     TEXT NOT NULL,
    winner      TEXT NOT NULL,
    reasoning   TEXT DEFAULT '',
    scores      TEXT DEFAULT '{}',
    elo_a       REAL DEFAULT 1000.0,
    elo_b       REAL DEFAULT 1000.0,
    meta        TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_arena_model_a ON arena_battles(model_a);
CREATE INDEX IF NOT EXISTS idx_arena_model_b ON arena_battles(model_b);
CREATE INDEX IF NOT EXISTS idx_arena_ts     ON arena_battles(timestamp);
"""


class ArenaStore:
    """SQLite 后端的 arena 历史存储, 线程安全 (单连接 + 锁)."""

    def __init__(self, db_path: str | Path = "arena.sqlite3") -> None:
        self._path = str(db_path)
        self._lock = threading.Lock()
        # check_same_thread=False: FastAPI 多线程会调到
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, rec: ArenaRecord) -> int:
        """写入一条比较记录, 返回行 id."""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO arena_battles
                   (timestamp, model_a, model_b, winner, reasoning,
                    scores, elo_a, elo_b, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.timestamp,
                    rec.model_a,
                    rec.model_b,
                    rec.winner,
                    rec.reasoning,
                    json.dumps(rec.scores, ensure_ascii=False),
                    rec.elo_a,
                    rec.elo_b,
                    json.dumps(rec.meta, ensure_ascii=False),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_history(
        self, model: str | None = None, limit: int = 100
    ) -> list[ArenaRecord]:
        """查历史. model 不为空时只查该模型参与的对局."""
        with self._lock:
            if model:
                cur = self._conn.execute(
                    """SELECT timestamp, model_a, model_b, winner, reasoning,
                              scores, elo_a, elo_b, meta
                       FROM arena_battles
                       WHERE model_a = ? OR model_b = ?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (model, model, limit),
                )
            else:
                cur = self._conn.execute(
                    """SELECT timestamp, model_a, model_b, winner, reasoning,
                              scores, elo_a, elo_b, meta
                       FROM arena_battles
                       ORDER BY timestamp DESC LIMIT ?""",
                    (limit,),
                )
            rows = cur.fetchall()

        records: list[ArenaRecord] = []
        for r in rows:
            records.append(
                ArenaRecord(
                    timestamp=r[0],
                    model_a=r[1],
                    model_b=r[2],
                    winner=r[3],
                    reasoning=r[4],
                    scores=json.loads(r[5] or "{}"),
                    elo_a=r[6],
                    elo_b=r[7],
                    meta=json.loads(r[8] or "{}"),
                )
            )
        return records

    def latest_elo(self) -> dict[str, float]:
        """每个模型最近一次出现时的 ELO. 取最后一场的 elo_a/elo_b."""
        elo: dict[str, float] = {}
        with self._lock:
            cur = self._conn.execute(
                """SELECT model_a, elo_a, model_b, elo_b, timestamp
                   FROM arena_battles ORDER BY timestamp ASC"""
            )
            for r in cur.fetchall():
                # 顺序遍历, 后出现的覆盖前面的, 得到每个模型最新的 ELO
                elo[r[0]] = r[1]
                elo[r[2]] = r[3]
        return elo

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # 当上下文管理器用, 方便测试
    def __enter__(self) -> ArenaStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def now_ts() -> float:
    """统一的时间戳取法, 方便测试 monkeypatch."""
    return time.time()
