"""v14 Task 14: 跨 task Meta-Trace SQLite 存储.

同 domain 的历史 trace entry 作为 prior 影响当前 task 的 hint 优先级.
跨 domain 隔离 — astronomy 的 entry 不影响 material 的 task.

ponytail: SQLite 单文件 + stdlib sqlite3, 不引入新依赖, 不需要 server.
天花板: 单进程写入 (SQLite 全局锁), 跨 process 并发写会 lock. 升级路径:
Postgres + 连接池.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trace_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    simplex_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    iteration INTEGER,
    ts TEXT,
    role TEXT,
    attempted TEXT,
    found TEXT,
    evidence TEXT,
    darwin_score REAL DEFAULT 0.0,
    supported_ratio REAL DEFAULT 0.0,
    cochain_type TEXT DEFAULT 'legacy',
    raw_json TEXT,
    UNIQUE(simplex_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_domain ON trace_entries(domain);
CREATE INDEX IF NOT EXISTS idx_task ON trace_entries(task_id);
CREATE INDEX IF NOT EXISTS idx_darwin ON trace_entries(darwin_score DESC);
"""


class CrossTaskStore:
    """跨 task Meta-Trace SQLite 存储.

    同 domain 的历史 entry 作为 prior 影响当前 task 的 hint 优先级.
    跨 domain 隔离: domain='astronomy' 的 entry 不影响 domain='material' 的 task.

    ponytail: SQLite 单文件, 不需要 server. 升级路径: Postgres for multi-process.
    """

    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            cache_dir = Path(os.environ.get("HUGINN_CACHE_DIR") or (Path.home() / ".huginn"))
            db_path = cache_dir / "cross_task_complex.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def append(self, entry: dict) -> None:
        """写入一个 trace entry. UNIQUE(simplex_id, task_id) 冲突时 REPLACE.

        dict 中没的字段用默认值. raw_json 存原始 json.dumps(entry).
        evidence 可能是 list/dict, 序列化成 JSON 文本存.
        """
        ev = entry.get("evidence")
        if isinstance(ev, (list, dict)):
            ev_str = json.dumps(ev, ensure_ascii=False, default=str)
        else:
            ev_str = "" if ev is None else str(ev)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trace_entries (
                    simplex_id, task_id, domain, iteration, ts, role,
                    attempted, found, evidence, darwin_score, supported_ratio,
                    cochain_type, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(entry.get("simplex_id") or ""),
                    str(entry.get("task_id") or ""),
                    str(entry.get("domain") or "unknown"),
                    entry.get("iteration"),
                    str(entry.get("ts") or ""),
                    str(entry.get("role") or ""),
                    str(entry.get("attempted") or ""),
                    str(entry.get("found") or ""),
                    ev_str,
                    float(entry.get("darwin_score") or 0.0),
                    float(entry.get("supported_ratio") or 0.0),
                    str(entry.get("cochain_type") or "legacy"),
                    json.dumps(entry, ensure_ascii=False, default=str),
                ),
            )

    def query(
        self,
        domain: str,
        task_id: Optional[str] = None,
        keyword: Optional[str] = None,
        top_k: int = 10,
    ) -> list[dict]:
        """查询历史 entry. 跨 domain 隔离: 只返回指定 domain 的 entry.

        keyword 用 %keyword% 模糊匹配 attempted 或 evidence.
        """
        sql = "SELECT * FROM trace_entries WHERE domain = ?"
        params: list = [domain]
        if task_id is not None:
            sql += " AND task_id = ?"
            params.append(task_id)
        if keyword is not None:
            sql += " AND (attempted LIKE ? OR evidence LIKE ?)"
            kw = f"%{keyword}%"
            params.extend([kw, kw])
        sql += " ORDER BY darwin_score DESC LIMIT ?"
        params.append(top_k)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_high_darwin(self, domain: str, top_k: int = 5) -> list[dict]:
        """查询指定 domain 下 darwin_score > 0.5 的 top_k entry."""
        sql = (
            "SELECT * FROM trace_entries WHERE domain = ? AND darwin_score > 0.5 "
            "ORDER BY darwin_score DESC LIMIT ?"
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, (domain, top_k)).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    # v14 Task 14 self-check: 跨 task 累积 + 跨 domain 隔离.
    # 临时 db, 不依赖 RCB workspace, 全用 assert.
    _tmp_dir = tempfile.mkdtemp()
    _tmp_db = Path(_tmp_dir) / "cross_task_complex.db"
    try:
        store = CrossTaskStore(db_path=_tmp_db)

        # 3 个 task entry: A/B astronomy, C material
        store.append({
            "simplex_id": "trace:Astronomy_000:iter_0:rcb_exec",
            "task_id": "Astronomy_000", "domain": "astronomy",
            "iteration": 0, "ts": "2026-07-20T10:00:00", "role": "rcb_exec",
            "attempted": "superradiance extraction",
            "found": "BdG modes near horizon", "evidence": ["num mode 0.42"],
            "darwin_score": 0.8, "supported_ratio": 0.5,
            "cochain_type": "gradient",
        })
        store.append({
            "simplex_id": "trace:Astronomy_001:iter_0:rcb_exec",
            "task_id": "Astronomy_001", "domain": "astronomy",
            "iteration": 0, "ts": "2026-07-20T11:00:00", "role": "rcb_exec",
            "attempted": "BH shadow formula",
            "found": "shadow diameter formula", "evidence": ["M87 shadow 42 uas"],
            "darwin_score": 0.6, "supported_ratio": 0.3,
            "cochain_type": "gradient",
        })
        store.append({
            "simplex_id": "trace:Material_000:iter_0:rcb_exec",
            "task_id": "Material_000", "domain": "material",
            "iteration": 0, "ts": "2026-07-20T12:00:00", "role": "rcb_exec",
            "attempted": "C-S-H gel hydration",
            "found": "hydration kinetic curve", "evidence": ["Q-residual 0.12"],
            "darwin_score": 0.9, "supported_ratio": 0.6,
            "cochain_type": "gradient",
        })

        # case 1: astronomy domain 应有 2 entry (A + B)
        r1 = store.query(domain="astronomy")
        assert len(r1) == 2, f"case1 expected 2, got {len(r1)}"
        tids = {r["task_id"] for r in r1}
        assert tids == {"Astronomy_000", "Astronomy_001"}, f"case1 tids={tids}"
        print(f"[CHECK v14 Task 14] case1 astronomy query OK ({len(r1)} entries)")

        # case 2: material domain 应有 1 entry (C)
        r2 = store.query(domain="material")
        assert len(r2) == 1, f"case2 expected 1, got {len(r2)}"
        assert r2[0]["task_id"] == "Material_000", f"case2 tid={r2[0]['task_id']}"
        print(f"[CHECK v14 Task 14] case2 material query OK ({len(r2)} entries)")

        # case 3: astronomy + task_id=Astronomy_000 应只返回 A
        r3 = store.query(domain="astronomy", task_id="Astronomy_000")
        assert len(r3) == 1, f"case3 expected 1, got {len(r3)}"
        assert r3[0]["task_id"] == "Astronomy_000", f"case3 tid={r3[0]['task_id']}"
        print("[CHECK v14 Task 14] case3 task_id filter OK")

        # case 4: astronomy + keyword=superradiance 应只返回 A
        r4 = store.query(domain="astronomy", keyword="superradiance")
        assert len(r4) == 1, f"case4 expected 1, got {len(r4)}"
        assert r4[0]["task_id"] == "Astronomy_000", f"case4 tid={r4[0]['task_id']}"
        print("[CHECK v14 Task 14] case4 keyword filter OK")

        # case 5: query_high_darwin(astronomy, top_k=5) — A(0.8) + B(0.6) 都 >0.5
        r5 = store.query_high_darwin(domain="astronomy", top_k=5)
        assert len(r5) == 2, f"case5 expected 2, got {len(r5)}"
        print(f"[CHECK v14 Task 14] case5 high_darwin top_k=5 OK ({len(r5)} entries)")

        # case 6: query_high_darwin(astronomy, top_k=1) — 只 A (0.8 > 0.6)
        r6 = store.query_high_darwin(domain="astronomy", top_k=1)
        assert len(r6) == 1, f"case6 expected 1, got {len(r6)}"
        assert r6[0]["darwin_score"] == 0.8, f"case6 darwin={r6[0]['darwin_score']}"
        assert r6[0]["task_id"] == "Astronomy_000", f"case6 tid={r6[0]['task_id']}"
        print(f"[CHECK v14 Task 14] case6 high_darwin top_k=1 OK (darwin={r6[0]['darwin_score']})")

        # case 7: 跨 domain 隔离 — astronomy 查询不含 Material_000
        r7 = store.query(domain="astronomy")
        assert all(r["task_id"] != "Material_000" for r in r7), "case7 isolation failed"
        print("[CHECK v14 Task 14] case7 cross-domain isolation OK")

        print("v14 Task 14 self-check PASSED")
    finally:
        try:
            _tmp_db.unlink(missing_ok=True)
            _tmp_db.parent.rmdir()
        except OSError:
            pass
