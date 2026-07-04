"""研究日志 (Research Log) 的测试.

覆盖增删改查、父子关系、搜索、统计和容量清理. 每个测试用 tmp_path
建独立 SQLite 文件, 不碰 ~/.huginn.
"""

from __future__ import annotations

import pytest

from huginn.research_log import (
    RecordType,
    ResearchLog,
    ResearchLogConfig,
)


@pytest.fixture()
def log(tmp_path):
    """独立 SQLite 的 ResearchLog, 测试完关连接."""
    rlog = ResearchLog(db_path=str(tmp_path / "research.db"))
    yield rlog
    rlog.close()


# ── 增 / 查 ────────────────────────────────────────────────


def test_add_and_get(log: ResearchLog) -> None:
    rec = log.add(RecordType.CONJECTURE, "钙钛矿带隙线性假设", "容忍因子 t 在 ...")
    fetched = log.get(rec.id)
    assert fetched is not None
    assert fetched.title == "钙钛矿带隙线性假设"
    assert fetched.record_type == RecordType.CONJECTURE
    assert fetched.status == "proposed"


def test_list_by_type(log: ResearchLog) -> None:
    log.add(RecordType.CONJECTURE, "猜想 A", "内容 A")
    log.add(RecordType.PROOF_ATTEMPT, "尝试 B", "内容 B")
    log.add(RecordType.CONJECTURE, "猜想 C", "内容 C")
    conjectures = log.list_by_type(RecordType.CONJECTURE)
    assert len(conjectures) == 2
    assert all(r.record_type == RecordType.CONJECTURE for r in conjectures)


def test_list_by_status(log: ResearchLog) -> None:
    r1 = log.add(RecordType.CONJECTURE, "待验证", "内容")
    r2 = log.add(RecordType.VERIFICATION, "已验证", "内容")
    log.update_status(r2.id, "verified")
    verified = log.list_by_status("verified")
    assert len(verified) == 1
    assert verified[0].id == r2.id


def test_update_status(log: ResearchLog) -> None:
    rec = log.add(RecordType.CONJECTURE, "待更新", "内容")
    assert log.update_status(rec.id, "verified") is True
    assert log.get(rec.id).status == "verified"
    # 不存在的状态不应改库, 返回 False
    assert log.update_status(rec.id, "bogus") is False


# ── 父子关系 ───────────────────────────────────────────────


def test_parent_child(log: ResearchLog) -> None:
    parent = log.add(RecordType.CONJECTURE, "父猜想", "父内容")
    child = log.add(
        RecordType.PROOF_ATTEMPT, "子尝试", "子内容", parent_id=parent.id
    )
    children = log.get_children(parent.id)
    assert len(children) == 1
    assert children[0].id == child.id


# ── 搜索 ───────────────────────────────────────────────────


def test_search(log: ResearchLog) -> None:
    log.add(RecordType.CONJECTURE, "band gap prediction", "用 DFT 算带隙")
    log.add(RecordType.OBSTACLE, "收敛问题", "ENCUT 太低导致不收敛")
    # 标题命中
    results = log.search("band gap")
    assert len(results) >= 1
    assert any("band gap" in r.title for r in results)
    # 内容命中
    results = log.search("ENCUT")
    assert len(results) >= 1
    assert any("ENCUT" in r.content for r in results)


# ── 统计 ───────────────────────────────────────────────────


def test_stats(log: ResearchLog) -> None:
    log.add(RecordType.CONJECTURE, "c1", "x")
    log.add(RecordType.CONJECTURE, "c2", "x")
    log.add(RecordType.PROOF_ATTEMPT, "p1", "x")
    stats = log.get_stats()
    assert stats["total"] == 3
    assert stats["by_type"]["conjecture"] == 2
    assert stats["by_type"]["proof_attempt"] == 1
    assert stats["archived"] == 0


# ── 容量清理 ──────────────────────────────────────────────


def test_cleanup(tmp_path) -> None:
    # max_records=5, auto_archive 开着
    cfg = ResearchLogConfig(max_records=5, auto_archive=True)
    log = ResearchLog(db_path=str(tmp_path / "cleanup.db"), config=cfg)
    try:
        # 先加 3 条, 其中 2 条标记为 refuted (自动归档)
        r1 = log.add(RecordType.CONJECTURE, "c1", "content1")
        r2 = log.add(RecordType.CONJECTURE, "c2", "content2")
        r3 = log.add(RecordType.CONJECTURE, "c3", "content3")
        log.update_status(r1.id, "refuted")
        log.update_status(r2.id, "refuted")
        assert log.get_stats()["archived"] == 2
        # 再加 3 条, 第 6 条触发清理 (total=6 > max=5, 删 1 条归档)
        log.add(RecordType.CONJECTURE, "c4", "content4")
        log.add(RecordType.CONJECTURE, "c5", "content5")
        log.add(RecordType.CONJECTURE, "c6", "content6")
        stats = log.get_stats()
        # 归档记录被删了一条, 总数回到 max_records
        assert stats["total"] <= 5
        assert stats["archived"] == 1
    finally:
        log.close()
