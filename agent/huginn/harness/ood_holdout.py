"""H6: OOD Holdout Validator — 分布外留出验证.

防"背题补丁": 一条 patch/variant 在训练任务上有效, 可能只是记住了这些任务,
而非修正了真实失败机制. 本模块把任务分成训练集和留出集, 要求候选配置在
留出集上也表现不退化, 才能被正式采纳.

核心 API:
- assign_split(task_id, train_ratio=0.7): 分配任务到 train/holdout (确定性 hash)
- record_outcome(config_id, task_id, score): 记录某配置在某任务上的得分
- validate_ood(config_id): 返回 OOD 验证结果 (pass/fail/insufficient)

设计决策:
- 确定性分桶: task_id 的 hash 决定 train/holdout, 同一任务永远在同一集.
  避免"重新分桶把之前 holdout 的任务放进 train"导致泄漏.
- 不要求 holdout 上"更好", 只要求"不显著退化": candidate 在 holdout 上
  得分的中位数 >= baseline 的中位数 * (1 - tolerance). 这是保守门控 —
  宁可误杀一个真有效但在 holdout 上碰巧差一点的 patch, 也不放过背题补丁.
- tolerance 默认 0.1: 允许 10% 退化, 因为 holdout 任务可能本身更难.
- train 和 holdout 都需要足够样本 (各 >= min_per_split=3) 才做判断.

toggle: cfg.feature_flags.harness_ood_holdout (默认 off).
跟 significance_gate 配合使用: significance 检验"是否真涨了",
OOD 检验"是否泛化而非背题".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from huginn.utils.runtime import get_runtime_home

logger = logging.getLogger(__name__)

_DEFAULT_TRAIN_RATIO = 0.7
_DEFAULT_TOLERANCE = 0.1
_DEFAULT_MIN_PER_SPLIT = 3


def _harness_enabled(key: str, default: bool = False) -> bool:
    """读 cfg.feature_flags.<key>, fallback 到 FeatureFlags 统一层.

    优先级: 配置文件 > 环境变量 / 运行时 API > default.
    """
    try:
        from huginn.config import get_config

        cfg = get_config()
        ff = getattr(cfg, "feature_flags", None) or {}
        if key in ff:
            return bool(ff[key])
    except Exception:
        pass
    try:
        from huginn.feature_flags import FeatureFlags

        return FeatureFlags.shared().is_enabled(key)
    except Exception:
        return default


def _is_holdout(task_id: str, train_ratio: float = _DEFAULT_TRAIN_RATIO) -> bool:
    """确定性分桶: task_id 的 hash < (1-train_ratio)*65536 → holdout.

    同一 task_id 永远分到同一桶, 避免重新分桶导致泄漏.
    """
    h = int(
        hashlib.md5(task_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:8],
        16,
    )
    threshold = int((1.0 - train_ratio) * 65536)
    return (h % 65536) < threshold


@dataclass
class OutcomeRecord:
    """一条配置-任务-得分记录."""

    config_id: str
    task_id: str
    score: float
    is_holdout: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OODResult:
    """OOD 验证结果."""

    config_id: str
    passed: bool
    reason: str
    train_n: int = 0
    holdout_n: int = 0
    train_median: float = 0.0
    holdout_median: float = 0.0
    baseline_holdout_median: float = 0.0
    degradation: float = 0.0
    tolerance: float = _DEFAULT_TOLERANCE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OODHoldoutValidator:
    """OOD 留出验证器. 单例.

    存 .huginn/ood_holdout/<config_id>.json.
    每个配置维护 (task_id, score, is_holdout) 列表.

    baseline 的得分也存进来, config_id="__baseline__".
    """

    _instance: OODHoldoutValidator | None = None
    _lock = threading.Lock()
    _BASELINE_ID = "__baseline__"

    def __init__(self) -> None:
        cache_dir = get_runtime_home()
        self._store_dir = cache_dir / "ood_holdout"
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("ood_holdout dir create failed", exc_info=True)
        self._records: dict[str, list[OutcomeRecord]] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> OODHoldoutValidator:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_all(self) -> None:
        try:
            for f in self._store_dir.glob("*.json"):
                try:
                    data = json.loads(f.read_text(encoding="utf-8"))
                    config_id = data.get("config_id", f.stem)
                    records_data = data.get("records", [])
                    self._records[config_id] = [
                        OutcomeRecord(
                            config_id=r["config_id"],
                            task_id=r["task_id"],
                            score=r["score"],
                            is_holdout=r["is_holdout"],
                            timestamp=r.get("timestamp", 0.0),
                        )
                        for r in records_data
                    ]
                except Exception:
                    logger.debug("ood load fail: %s", f, exc_info=True)
        except Exception:
            logger.debug("ood dir scan fail", exc_info=True)

    def _save(self, config_id: str) -> None:
        try:
            f = self._store_dir / f"{config_id}.json"
            records = self._records.get(config_id, [])
            data = {
                "config_id": config_id,
                "records": [r.to_dict() for r in records],
                "updated_at": time.time(),
            }
            f.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("ood save fail: %s", config_id, exc_info=True)

    def record_outcome(
        self,
        config_id: str,
        task_id: str,
        score: float,
        train_ratio: float = _DEFAULT_TRAIN_RATIO,
    ) -> bool:
        """记录一条配置-任务-得分. 返回该任务是否在 holdout 集.

        config_id="__baseline__" 记录基线得分.
        """
        is_holdout = _is_holdout(task_id, train_ratio)
        rec = OutcomeRecord(
            config_id=config_id,
            task_id=task_id,
            score=score,
            is_holdout=is_holdout,
        )
        with self._lock:
            self._records.setdefault(config_id, []).append(rec)
        self._save(config_id)
        return is_holdout

    def get_records(self, config_id: str) -> list[OutcomeRecord]:
        with self._lock:
            return list(self._records.get(config_id, []))

    def _median(self, values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        n = len(s)
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def validate_ood(
        self,
        config_id: str,
        tolerance: float = _DEFAULT_TOLERANCE,
        min_per_split: int = _DEFAULT_MIN_PER_SPLIT,
    ) -> OODResult:
        """验证候选配置在 holdout 集上是否不显著退化.

        通过条件:
        1. train 和 holdout 各有 >= min_per_split 条记录
        2. baseline 在 holdout 上有 >= min_per_split 条记录
        3. candidate holdout 中位数 >= baseline holdout 中位数 * (1 - tolerance)

        未通过时返回 passed=False + reason, 但不删数据.
        """
        cand_records = self.get_records(config_id)
        base_records = self.get_records(self._BASELINE_ID)

        cand_train = [r for r in cand_records if not r.is_holdout]
        cand_holdout = [r for r in cand_records if r.is_holdout]
        base_holdout = [r for r in base_records if r.is_holdout]

        if len(cand_train) < min_per_split:
            return OODResult(
                config_id=config_id,
                passed=False,
                reason=f"insufficient train samples: {len(cand_train)}/{min_per_split}",
                train_n=len(cand_train),
                holdout_n=len(cand_holdout),
            )

        if len(cand_holdout) < min_per_split:
            return OODResult(
                config_id=config_id,
                passed=False,
                reason=f"insufficient holdout samples: {len(cand_holdout)}/{min_per_split}",
                train_n=len(cand_train),
                holdout_n=len(cand_holdout),
            )

        if len(base_holdout) < min_per_split:
            return OODResult(
                config_id=config_id,
                passed=False,
                reason=(
                    f"insufficient baseline holdout samples: "
                    f"{len(base_holdout)}/{min_per_split}"
                ),
                train_n=len(cand_train),
                holdout_n=len(cand_holdout),
            )

        cand_train_median = self._median([r.score for r in cand_train])
        cand_holdout_median = self._median([r.score for r in cand_holdout])
        base_holdout_median = self._median([r.score for r in base_holdout])

        # 退化 = (baseline - candidate) / baseline, 正值=退化
        if base_holdout_median > 1e-12:
            degradation = (
                base_holdout_median - cand_holdout_median
            ) / base_holdout_median
        else:
            degradation = 0.0 if cand_holdout_median >= 0 else 1.0

        passed = degradation <= tolerance
        reason = (
            f"holdout degradation {degradation:.1%} "
            f"{'<=' if passed else '>'} tolerance {tolerance:.1%} "
            f"(cand_holdout_med={cand_holdout_median:.3f} "
            f"vs base_holdout_med={base_holdout_median:.3f})"
        )

        return OODResult(
            config_id=config_id,
            passed=passed,
            reason=reason,
            train_n=len(cand_train),
            holdout_n=len(cand_holdout),
            train_median=cand_train_median,
            holdout_median=cand_holdout_median,
            baseline_holdout_median=base_holdout_median,
            degradation=degradation,
            tolerance=tolerance,
        )

    def clear(self, config_id: str | None = None) -> None:
        """清除记录. config_id=None 清除全部."""
        with self._lock:
            if config_id is None:
                self._records.clear()
            else:
                self._records.pop(config_id, None)
        if config_id is not None:
            import contextlib

            f = self._store_dir / f"{config_id}.json"
            with contextlib.suppress(Exception):
                f.unlink(missing_ok=True)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_configs": len(self._records),
                "configs": {
                    cid: {
                        "total": len(recs),
                        "train": sum(1 for r in recs if not r.is_holdout),
                        "holdout": sum(1 for r in recs if r.is_holdout),
                    }
                    for cid, recs in self._records.items()
                },
            }


def _selfcheck() -> None:
    """H6 selfcheck: 分桶 + 记录 + OOD 验证 + 持久化."""
    import shutil
    import tempfile

    import huginn.harness.ood_holdout as ood

    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    ood.OODHoldoutValidator._instance = None

    val = ood.OODHoldoutValidator.get_instance()

    # 1. 确定性分桶: 同一 task_id 永远同一桶
    t1_holdout = ood._is_holdout("task_001")
    t1_holdout_again = ood._is_holdout("task_001")
    assert t1_holdout == t1_holdout_again, "same task should map to same bucket"
    # 分桶比例大致 30% holdout (20 个任务里 4-10 个 holdout)
    holdout_count = sum(1 for i in range(20) if ood._is_holdout(f"task_{i:03d}"))
    assert 2 <= holdout_count <= 12, f"holdout ratio out of range: {holdout_count}/20"
    print(f"1. deterministic split OK ({holdout_count}/20 holdout)")

    # 2. 样本不足 → 不通过
    val.record_outcome("cfg_a", "task_001", score=0.7)
    r = val.validate_ood("cfg_a")
    assert not r.passed, f"should fail with insufficient samples: {r}"
    assert "insufficient" in r.reason, r.reason
    print(f"2. insufficient samples → fail OK ({r.reason})")

    # 3. 构造完整场景: baseline + candidate, candidate 在 holdout 上不退化
    # 先记录 baseline 在 10 个任务上的得分
    for i in range(10):
        val.record_outcome(ood.OODHoldoutValidator._BASELINE_ID, f"base_task_{i}", score=0.5)
    # candidate 在同样 10 个任务上得分 0.6 (train + holdout 都好)
    for i in range(10):
        val.record_outcome("cfg_good", f"base_task_{i}", score=0.6)

    r = val.validate_ood("cfg_good")
    assert r.passed, f"cfg_good should pass: {r}"
    assert r.degradation <= r.tolerance, f"degradation too high: {r}"
    print(
        f"3. good candidate → pass "
        f"(degradation={r.degradation:.1%}, holdout_med={r.holdout_median:.2f}) OK"
    )

    # 4. 背题补丁: candidate 在 train 上好, holdout 上差
    # 用不同 task_id 让 train/holdout 分开
    train_tasks = [f"ood_train_{i:03d}" for i in range(20) if not ood._is_holdout(f"ood_train_{i:03d}")][:6]
    holdout_tasks = [f"ood_train_{i:03d}" for i in range(20) if ood._is_holdout(f"ood_train_{i:03d}")][:6]

    # baseline 在 train 和 holdout 上都是 0.5
    for t in train_tasks + holdout_tasks:
        val.record_outcome(ood.OODHoldoutValidator._BASELINE_ID, t, score=0.5)
    # candidate 在 train 上 0.8 (背题), holdout 上 0.2 (泛化失败)
    for t in train_tasks:
        val.record_outcome("cfg_overfit", t, score=0.8)
    for t in holdout_tasks:
        val.record_outcome("cfg_overfit", t, score=0.2)

    r = val.validate_ood("cfg_overfit")
    assert not r.passed, f"cfg_overfit should fail OOD: {r}"
    assert r.degradation > r.tolerance, f"degradation should be high: {r}"
    assert r.train_median > r.holdout_median, "train should be better than holdout"
    print(
        f"4. overfit candidate → fail OOD "
        f"(train_med={r.train_median:.2f}, holdout_med={r.holdout_median:.2f}, "
        f"degradation={r.degradation:.1%}) OK"
    )

    # 5. 持久化 reload
    ood.OODHoldoutValidator._instance = None
    val2 = ood.OODHoldoutValidator.get_instance()
    recs = val2.get_records("cfg_good")
    assert len(recs) == 10, f"persisted wrong: {len(recs)}"
    base_recs = val2.get_records(ood.OODHoldoutValidator._BASELINE_ID)
    assert len(base_recs) >= 10, f"baseline persisted wrong: {len(base_recs)}"
    print(f"5. persistence reload OK (cfg_good: {len(recs)}, baseline: {len(base_recs)})")

    # 6. clear
    val2.clear("cfg_overfit")
    assert len(val2.get_records("cfg_overfit")) == 0, "clear failed"
    print("6. clear OK")

    shutil.rmtree(tmp, ignore_errors=True)
    del os.environ["HUGINN_CACHE_DIR"]
    ood.OODHoldoutValidator._instance = None
    print("\nH6 ood_holdout selfcheck OK (6/6)")


if __name__ == "__main__":
    _selfcheck()
