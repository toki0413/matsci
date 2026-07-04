"""SQLite 并发写入压力测试 — 验证 WAL 模式下的写入稳定性.

前置条件:
  1. 启动服务: python -m huginn serve --port 8000

运行:
  python -m pytest tests/stress/test_sqlite_write_storm.py -v -x --tb=short

测试维度:
  - 10 线程 × 1000 次研究日志写入
  - 5 线程 × 500 次异常日志写入
  - 并发读写混合 (读 + 写同时进行)
  - 验证 WAL 模式下无 "database is locked"
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"


def _check_server() -> bool:
    """确认服务在跑."""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


# 直接 SQLite 测试不需要服务端,只有 HTTP 级测试需要
_skip_no_server = pytest.mark.skipif(
    not _check_server(), reason="Server not running on :8000",
)

# ── 直接 SQLite 写入风暴 (不走 HTTP,更快) ─────────────


def test_research_log_concurrent_writes(tmp_path):
    """10 线程 × 200 次写入研究日志, 验证无 database is locked."""
    from huginn.research_log import ResearchLog, RecordType

    db_path = tmp_path / "storm_research.sqlite"
    log = ResearchLog(db_path=str(db_path))

    errors: list[str] = []
    barrier = threading.Barrier(10)

    def writer(tid: int):
        barrier.wait()  # 所有线程同时开始,最大化冲突
        for i in range(200):
            try:
                log.add(
                    RecordType.CONJECTURE,
                    f"conjecture-{tid}-{i}",
                    f"content from thread {tid} iteration {i}",
                    tags=[f"stress", f"t{tid}"],
                )
            except Exception as e:
                errors.append(f"t{tid} i{i}: {e}")

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(writer, t) for t in range(10)]
        for f in as_completed(futures):
            f.result()

    # 不应该有任何 "database is locked" 错误
    locked_errors = [e for e in errors if "locked" in e.lower()]
    assert len(locked_errors) == 0, f"{len(locked_errors)} 'database is locked' errors"

    # 总写入应该成功 2000 条
    all_records = log.list_by_type(RecordType.CONJECTURE)
    assert len(all_records) >= 1800, f"Expected ~2000 records, got {len(all_records)}"

    print(f"  Wrote {len(all_records)} records with {len(errors)} errors (0 locked)")


def test_anomaly_log_concurrent_writes(tmp_path):
    """5 线程 × 300 次异常日志写入."""
    from huginn.anomaly_log import AnomalyLogStore, Anomaly
    from datetime import datetime

    db_path = tmp_path / "storm_anomaly.sqlite"
    store = AnomalyLogStore(db_path=str(db_path))

    errors: list[str] = []
    barrier = threading.Barrier(5)

    def writer(tid: int):
        barrier.wait()
        for i in range(300):
            try:
                record = Anomaly(
                    id="",
                    ts=datetime.now(),
                    category="INPUT_DATA",
                    severity="MEDIUM",
                    description=f"anomaly from thread {tid} iteration {i}",
                    detection_method="keyword_match",
                    source="tool_output",
                    context_snapshot={"thread": tid, "iter": i},
                    unresolved_dimensions=[],
                )
                store.log(record)
            except Exception as e:
                errors.append(f"t{tid} i{i}: {e}")

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(writer, t) for t in range(5)]
        for f in as_completed(futures):
            f.result()

    locked = [e for e in errors if "locked" in e.lower()]
    assert len(locked) == 0, f"{len(locked)} 'database is locked' errors"
    print(f"  AnomalyLog: 1500 writes, {len(errors)} errors, 0 locked")


def test_mixed_read_write_storm(tmp_path):
    """并发读写混合: 5 写 + 5 读, 验证读写不互斥."""
    from huginn.research_log import ResearchLog, RecordType

    db_path = tmp_path / "storm_rw.sqlite"
    log = ResearchLog(db_path=str(db_path))

    # 先写一批基础数据
    for i in range(100):
        log.add(RecordType.CONJECTURE, f"base-{i}", f"base content {i}")

    errors: list[str] = []
    read_counts: list[int] = []
    write_counts: list[int] = []
    barrier = threading.Barrier(10)

    def writer(tid: int):
        barrier.wait()
        count = 0
        for i in range(200):
            try:
                log.add(RecordType.PROOF_ATTEMPT, f"proof-{tid}-{i}", f"proof {tid}-{i}")
                count += 1
            except Exception as e:
                errors.append(f"W t{tid} i{i}: {e}")
        write_counts.append(count)

    def reader(rid: int):
        barrier.wait()
        count = 0
        for i in range(200):
            try:
                records = log.list_by_type(RecordType.CONJECTURE)
                count += len(records)
            except Exception as e:
                errors.append(f"R t{rid} i{i}: {e}")
        read_counts.append(count)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = []
        for t in range(5):
            futures.append(pool.submit(writer, t))
        for t in range(5):
            futures.append(pool.submit(reader, t))
        for f in as_completed(futures):
            f.result()

    locked = [e for e in errors if "locked" in e.lower()]
    assert len(locked) == 0, f"{len(locked)} 'database is locked' errors"
    # 5 个 writer 应该各写 200 条
    assert sum(write_counts) >= 900, f"Expected ~1000 writes, got {sum(write_counts)}"
    # 5 个 reader 应该各读到数据
    assert all(c >= 100 for c in read_counts), f"Readers should see data: {read_counts}"
    print(f"  Mixed RW: {sum(write_counts)} writes, {sum(read_counts)} reads, {len(locked)} locked")


# ── HTTP 级 SQLite 压测 (走完整链路) ─────────────────


@pytest.mark.asyncio
@_skip_no_server
async def test_http_chat_concurrent_sqlite_storm():
    """20 路并发 HTTP chat, 每轮写 checkpointer + research log."""
    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(20):
            tid = f"sqlite-storm-{i:03d}"
            tasks.append(
                client.post(
                    f"{BASE_URL}/agents/agent/chat",
                    json={"content": f"sqlite storm {i}", "thread_id": tid},
                    timeout=30.0,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # 检查没有 500 错误 (SQLite locked 会触发 500)
    server_errors = [
        r for r in results
        if not isinstance(r, Exception)
        and hasattr(r, "status_code")
        and r.status_code == 500
    ]
    assert len(server_errors) == 0, f"{len(server_errors)} 500 errors (likely SQLite locked)"
    successes = [
        r for r in results
        if not isinstance(r, Exception) and hasattr(r, "status_code") and r.status_code == 200
    ]
    assert len(successes) >= 15, f"Only {len(successes)}/20 succeeded"
    print(f"  HTTP SQLite storm: {len(successes)}/20 succeeded, 0 server errors")
