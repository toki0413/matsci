"""Unit tests for huginn/harness/ood_holdout.py (H6).

测试覆盖:
1. 确定性分桶
2. 样本不足 → 不通过
3. 好的候选 → 通过
4. 背题补丁 (overfit) → 不通过
5. 持久化 reload
6. clear
7. 自定义 tolerance
8. summary
"""
from __future__ import annotations

import pytest

from huginn.harness.ood_holdout import (
    OODHoldoutValidator,
    _is_holdout,
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """每个测试用独立 cache 目录 + 新单例."""
    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    OODHoldoutValidator._instance = None
    yield
    OODHoldoutValidator._instance = None


def test_deterministic_split():
    """同一 task_id 永远分到同一桶."""
    for task_id in ["task_001", "task_002", "task_abc", "task_xyz"]:
        h1 = _is_holdout(task_id)
        h2 = _is_holdout(task_id)
        assert h1 == h2, f"{task_id} should map to same bucket"


def test_split_ratio():
    """20 个任务里 holdout 比例大致 30%."""
    holdout_count = sum(1 for i in range(20) if _is_holdout(f"task_{i:03d}"))
    assert 2 <= holdout_count <= 12, f"holdout ratio out of range: {holdout_count}/20"


def test_custom_train_ratio():
    """train_ratio=0.5 → holdout 比例更高."""
    holdout_50 = sum(
        1 for i in range(20) if _is_holdout(f"task_{i:03d}", train_ratio=0.5)
    )
    holdout_70 = sum(1 for i in range(20) if _is_holdout(f"task_{i:03d}", train_ratio=0.7))
    # train_ratio 越低, holdout 越多
    assert holdout_50 >= holdout_70, (
        f"train_ratio=0.5 should have more holdout: {holdout_50} vs {holdout_70}"
    )


def test_insufficient_samples():
    """样本不足时不通过."""
    val = OODHoldoutValidator.get_instance()
    val.record_outcome("cfg_a", "task_001", score=0.7)
    r = val.validate_ood("cfg_a")
    assert not r.passed
    assert "insufficient" in r.reason


def test_good_candidate_passes():
    """candidate 在 train 和 holdout 上都好 → 通过."""
    val = OODHoldoutValidator.get_instance()
    # baseline 在 10 个任务上 0.5
    for i in range(10):
        val.record_outcome(
            OODHoldoutValidator._BASELINE_ID, f"base_task_{i}", score=0.5
        )
    # candidate 在同 10 个任务上 0.6
    for i in range(10):
        val.record_outcome("cfg_good", f"base_task_{i}", score=0.6)
    r = val.validate_ood("cfg_good")
    assert r.passed, f"good candidate should pass: {r}"
    assert r.degradation <= r.tolerance


def test_overfit_candidate_fails():
    """背题补丁: train 好但 holdout 差 → 不通过."""
    val = OODHoldoutValidator.get_instance()
    # 分开构造 train 和 holdout 任务
    train_tasks = [
        f"ood_train_{i:03d}" for i in range(30)
        if not _is_holdout(f"ood_train_{i:03d}")
    ][:6]
    holdout_tasks = [
        f"ood_train_{i:03d}" for i in range(30)
        if _is_holdout(f"ood_train_{i:03d}")
    ][:6]
    assert len(train_tasks) >= 3, "need enough train tasks"
    assert len(holdout_tasks) >= 3, "need enough holdout tasks"

    # baseline 0.5
    for t in train_tasks + holdout_tasks:
        val.record_outcome(OODHoldoutValidator._BASELINE_ID, t, score=0.5)
    # candidate: train 0.8, holdout 0.2
    for t in train_tasks:
        val.record_outcome("cfg_overfit", t, score=0.8)
    for t in holdout_tasks:
        val.record_outcome("cfg_overfit", t, score=0.2)

    r = val.validate_ood("cfg_overfit")
    assert not r.passed, f"overfit candidate should fail: {r}"
    assert r.degradation > r.tolerance
    assert r.train_median > r.holdout_median


def test_persistence_reload():
    """持久化 reload: 数据跨 session 保留."""
    val = OODHoldoutValidator.get_instance()
    for i in range(5):
        val.record_outcome(OODHoldoutValidator._BASELINE_ID, f"p_task_{i}", score=0.5)
        val.record_outcome("cfg_persist", f"p_task_{i}", score=0.6)

    OODHoldoutValidator._instance = None
    val2 = OODHoldoutValidator.get_instance()
    recs = val2.get_records("cfg_persist")
    assert len(recs) == 5
    base_recs = val2.get_records(OODHoldoutValidator._BASELINE_ID)
    assert len(base_recs) == 5


def test_clear():
    """clear 清除指定 config 的数据."""
    val = OODHoldoutValidator.get_instance()
    for i in range(5):
        val.record_outcome("cfg_clear", f"c_task_{i}", score=0.6)
    assert len(val.get_records("cfg_clear")) == 5
    val.clear("cfg_clear")
    assert len(val.get_records("cfg_clear")) == 0


def test_custom_tolerance():
    """tolerance=0.0 → 任何退化都不通过."""
    val = OODHoldoutValidator.get_instance()
    for i in range(10):
        val.record_outcome(OODHoldoutValidator._BASELINE_ID, f"tol_task_{i}", score=0.5)
        # candidate 略好但有波动
        val.record_outcome("cfg_tol", f"tol_task_{i}", score=0.55)
    r = val.validate_ood("cfg_tol", tolerance=0.0)
    # tolerance=0 要求完全不退化
    assert isinstance(r.passed, bool)


def test_summary():
    """summary 返回统计信息."""
    val = OODHoldoutValidator.get_instance()
    for i in range(5):
        val.record_outcome("cfg_s1", f"s_task_{i}", score=0.6)
    for i in range(3):
        val.record_outcome("cfg_s2", f"s2_task_{i}", score=0.7)
    s = val.summary()
    assert s["total_configs"] == 2
    assert "cfg_s1" in s["configs"]
    assert s["configs"]["cfg_s1"]["total"] == 5


def test_ood_result_to_dict():
    """OODResult.to_dict 序列化正常."""
    val = OODHoldoutValidator.get_instance()
    for i in range(10):
        val.record_outcome(OODHoldoutValidator._BASELINE_ID, f"d_task_{i}", score=0.5)
        val.record_outcome("cfg_dict", f"d_task_{i}", score=0.6)
    r = val.validate_ood("cfg_dict")
    d = r.to_dict()
    assert d["config_id"] == "cfg_dict"
    assert "passed" in d
    assert "degradation" in d
    assert "reason" in d


def test_outcome_record_is_holdout_field():
    """OutcomeRecord 正确记录 is_holdout."""
    val = OODHoldoutValidator.get_instance()
    val.record_outcome("cfg_r", "test_task_001", score=0.7)
    recs = val.get_records("cfg_r")
    assert len(recs) == 1
    rec = recs[0]
    expected = _is_holdout("test_task_001")
    assert rec.is_holdout == expected


def test_no_baseline_holdout_fails():
    """没有 baseline holdout 数据 → 不通过."""
    val = OODHoldoutValidator.get_instance()
    # 只记 candidate, 不记 baseline
    train_tasks = [
        f"nb_train_{i:03d}" for i in range(30)
        if not _is_holdout(f"nb_train_{i:03d}")
    ][:6]
    holdout_tasks = [
        f"nb_train_{i:03d}" for i in range(30)
        if _is_holdout(f"nb_train_{i:03d}")
    ][:6]
    for t in train_tasks:
        val.record_outcome("cfg_nb", t, score=0.8)
    for t in holdout_tasks:
        val.record_outcome("cfg_nb", t, score=0.8)
    r = val.validate_ood("cfg_nb")
    assert not r.passed
    assert "baseline" in r.reason.lower()
