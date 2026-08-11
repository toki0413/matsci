"""H5: Significance Gate — 显著性检验门控.

防止"看起来涨了其实是噪声"的回归被接受. 在 patch/variant 被正式采纳前,
要求其增益通过统计显著性检验 (Wilcoxon signed-rank, 非参数, 小样本友好).

核心 API:
- record_pair(baseline_score, candidate_score, config_id): 记录一对配对观测
- passes_gate(config_id, alpha=0.05, min_samples=5): 是否通过显著性门
- gate_decision(config_id): 返回完整决策 (pass/fail/insufficient)

数学:
- Wilcoxon signed-rank: 对配对差值 d_i = c_i - b_i, 检验 H0: median(d) = 0.
  非参数, 不假设正态分布, 适合小样本 (n>=5 即可用, n>=10 更稳).
- 单侧检验 (candidate > baseline), 因为只关心"是否真涨了".
- scipy.stats.wilcoxon 的 alternative="greater" 做单侧检验.

设计决策:
- 用 Wilcoxon 而非 paired t-test: 进化跑次少 (n<30), 分布未知, 非参数更稳.
- 配对设计: 同一批任务跑 baseline 和 candidate, 消除任务间方差.
- min_samples=5: Wilcoxon 在 n<5 时几乎无法拒绝 H0 (功效太低), 5 是实用下限.
- gate 是 advisory: 不阻塞 Beta 更新, 只在 apply 前检查. 未通过时 patch
  仍留在 store 里积累数据, 不删除.

toggle: cfg.feature_flags.harness_significance_gate (默认 off).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from huginn.utils.runtime import get_runtime_home

logger = logging.getLogger(__name__)

_MIN_SAMPLES_DEFAULT = 5
_ALPHA_DEFAULT = 0.05


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
        # best-effort: 配置读不到就回退到 FeatureFlags 层, 但记录以便排查
        logger.debug("feature_flags 读取失败, 回退到 FeatureFlags 统一层", exc_info=True)
    try:
        from huginn.feature_flags import FeatureFlags

        return FeatureFlags.shared().is_enabled(key)
    except Exception:
        return default


@dataclass
class ScorePair:
    """一对配对观测: 同一任务上 baseline 和 candidate 的得分."""

    baseline_score: float
    candidate_score: float
    task_id: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def diff(self) -> float:
        return self.candidate_score - self.baseline_score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateDecision:
    """显著性门控的决策结果."""

    config_id: str
    passed: bool
    reason: str
    n_samples: int = 0
    p_value: float | None = None
    median_diff: float = 0.0
    alpha: float = _ALPHA_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignificanceGate:
    """显著性检验门控. 单例.

    存 .huginn/significance_gate/<config_id>.json, 每个配置的配对观测.
    跨 session 持久化, 进程内单例 + 磁盘文件.
    """

    _instance: SignificanceGate | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        cache_dir = get_runtime_home()
        self._store_dir = cache_dir / "significance_gate"
        try:
            self._store_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.debug("significance_gate dir create failed", exc_info=True)
        self._pairs: dict[str, list[ScorePair]] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> SignificanceGate:
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
                    pairs_data = data.get("pairs", [])
                    self._pairs[config_id] = [
                        ScorePair(
                            baseline_score=p["baseline_score"],
                            candidate_score=p["candidate_score"],
                            task_id=p.get("task_id", ""),
                            timestamp=p.get("timestamp", 0.0),
                        )
                        for p in pairs_data
                    ]
                except Exception:
                    logger.debug("gate load fail: %s", f, exc_info=True)
        except Exception:
            logger.debug("gate dir scan fail", exc_info=True)

    def _save(self, config_id: str) -> None:
        try:
            f = self._store_dir / f"{config_id}.json"
            pairs = self._pairs.get(config_id, [])
            data = {
                "config_id": config_id,
                "pairs": [p.to_dict() for p in pairs],
                "updated_at": time.time(),
            }
            f.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.debug("gate save fail: %s", config_id, exc_info=True)

    def record_pair(
        self,
        config_id: str,
        baseline_score: float,
        candidate_score: float,
        task_id: str = "",
    ) -> None:
        """记录一对配对观测. 同一任务上 baseline 和 candidate 的得分.

        config_id: 标识被检验的配置 (patch_id / variant_id / joint_config_id).
        baseline_score: 基线 harness 在该任务上的得分 (r_phys / success=1.0).
        candidate_score: 候选配置在该任务上的得分.
        task_id: 任务标识, 用于后续 OOD holdout 区分训练/留出集.
        """
        pair = ScorePair(
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            task_id=task_id,
        )
        with self._lock:
            self._pairs.setdefault(config_id, []).append(pair)
        self._save(config_id)

    def get_pairs(self, config_id: str) -> list[ScorePair]:
        with self._lock:
            return list(self._pairs.get(config_id, []))

    def passes_gate(
        self,
        config_id: str,
        alpha: float = _ALPHA_DEFAULT,
        min_samples: int = _MIN_SAMPLES_DEFAULT,
    ) -> bool:
        """检查配置是否通过显著性门.

        通过条件:
        1. 样本量 >= min_samples
        2. Wilcoxon 单侧 p-value < alpha (candidate 显著优于 baseline)
        3. 中位差值 > 0 (方向正确)

        样本不足或检验不显著时返回 False, 但不删除数据 — 继续积累.
        """
        decision = self.gate_decision(config_id, alpha, min_samples)
        return decision.passed

    def gate_decision(
        self,
        config_id: str,
        alpha: float = _ALPHA_DEFAULT,
        min_samples: int = _MIN_SAMPLES_DEFAULT,
    ) -> GateDecision:
        """返回完整决策, 含 p-value 和原因."""
        pairs = self.get_pairs(config_id)
        n = len(pairs)

        if n < min_samples:
            return GateDecision(
                config_id=config_id,
                passed=False,
                reason=f"insufficient samples: {n}/{min_samples}",
                n_samples=n,
                alpha=alpha,
            )

        diffs = [p.diff for p in pairs]
        median_diff = sorted(diffs)[len(diffs) // 2]

        # 所有差值同号 → Wilcoxon 退化, 手动判定
        non_zero_diffs = [d for d in diffs if abs(d) > 1e-12]
        if not non_zero_diffs:
            return GateDecision(
                config_id=config_id,
                passed=False,
                reason="all differences are zero",
                n_samples=n,
                median_diff=0.0,
                alpha=alpha,
            )

        all_positive = all(d > 0 for d in non_zero_diffs)
        all_negative = all(d < 0 for d in non_zero_diffs)

        if all_positive:
            # 全正 → 显著优于, p-value 趋近 0
            return GateDecision(
                config_id=config_id,
                passed=True,
                reason="all differences positive (exact test p≈0)",
                n_samples=n,
                p_value=0.0,
                median_diff=median_diff,
                alpha=alpha,
            )

        if all_negative:
            return GateDecision(
                config_id=config_id,
                passed=False,
                reason="all differences negative (candidate worse)",
                n_samples=n,
                p_value=1.0,
                median_diff=median_diff,
                alpha=alpha,
            )

        # 混合符号 → 用 scipy Wilcoxon signed-rank
        try:
            from scipy.stats import wilcoxon

            # scipy 1.10+ 支持 alternative 参数
            stat, p_value = wilcoxon(
                diffs,
                alternative="greater",
                zero_method="wilcox",
            )
            passed = p_value < alpha and median_diff > 0
            reason = (
                f"Wilcoxon p={p_value:.4f} {'<' if passed else '>='} α={alpha}"
            )
            return GateDecision(
                config_id=config_id,
                passed=passed,
                reason=reason,
                n_samples=n,
                p_value=float(p_value),
                median_diff=median_diff,
                alpha=alpha,
            )
        except ImportError:
            # scipy 不可用 → 降级到符号检验 (sign test)
            n_pos = sum(1 for d in non_zero_diffs if d > 0)
            n_neg = sum(1 for d in non_zero_diffs if d < 0)
            n_total = n_pos + n_neg
            # 二项检验: P(X >= n_pos | p=0.5)
            from math import comb

            p_value = sum(
                comb(n_total, k) for k in range(n_pos, n_total + 1)
            ) / (2**n_total)
            passed = p_value < alpha and median_diff > 0
            reason = (
                f"sign test p={p_value:.4f} (scipy unavailable) "
                f"{'<' if passed else '>='} α={alpha}"
            )
            return GateDecision(
                config_id=config_id,
                passed=passed,
                reason=reason,
                n_samples=n,
                p_value=float(p_value),
                median_diff=median_diff,
                alpha=alpha,
            )
        except Exception:
            logger.warning("gate_decision failed", exc_info=True)
            return GateDecision(
                config_id=config_id,
                passed=False,
                reason="internal error",
                n_samples=n,
                alpha=alpha,
            )

    def clear(self, config_id: str | None = None) -> None:
        """清除配对观测. config_id=None 清除全部."""
        with self._lock:
            if config_id is None:
                self._pairs.clear()
            else:
                self._pairs.pop(config_id, None)
        if config_id is not None:
            f = self._store_dir / f"{config_id}.json"
            with contextlib.suppress(Exception):
                f.unlink(missing_ok=True)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_configs": len(self._pairs),
                "configs": {
                    cid: len(pairs) for cid, pairs in self._pairs.items()
                },
            }


def _selfcheck() -> None:
    """H5 selfcheck: 配对记录 + 显著性检验 + 降级 + 持久化."""
    import shutil
    import tempfile

    import huginn.harness.significance_gate as sg

    tmp = tempfile.mkdtemp()
    os.environ["HUGINN_CACHE_DIR"] = tmp
    sg.SignificanceGate._instance = None

    gate = sg.SignificanceGate.get_instance()

    # 1. 样本不足 → 不通过
    gate.record_pair("cfg_a", baseline_score=0.5, candidate_score=0.6, task_id="t1")
    gate.record_pair("cfg_a", baseline_score=0.5, candidate_score=0.6, task_id="t2")
    d = gate.gate_decision("cfg_a")
    assert not d.passed, f"should fail with insufficient samples: {d}"
    assert "insufficient" in d.reason, d.reason
    assert d.n_samples == 2, d.n_samples
    print(f"1. insufficient samples (n=2/{_MIN_SAMPLES_DEFAULT}) → fail OK")

    # 2. 全正差值 → 通过 (exact test)
    for i in range(5):
        gate.record_pair(
            "cfg_b", baseline_score=0.4, candidate_score=0.7, task_id=f"t{i}"
        )
    d = gate.gate_decision("cfg_b")
    assert d.passed, f"all-positive should pass: {d}"
    assert d.p_value == 0.0, d.p_value
    assert d.median_diff > 0, d.median_diff
    print(f"2. all-positive diffs → pass (p=0.0, median_diff={d.median_diff:.1f}) OK")

    # 3. 全负差值 → 不通过
    for i in range(5):
        gate.record_pair(
            "cfg_c", baseline_score=0.7, candidate_score=0.4, task_id=f"t{i}"
        )
    d = gate.gate_decision("cfg_c")
    assert not d.passed, f"all-negative should fail: {d}"
    assert d.p_value == 1.0, d.p_value
    print("3. all-negative diffs → fail (p=1.0) OK")

    # 4. 混合差值 → Wilcoxon
    # 构造 15 正 1 负, n=16, p-value 远低于 0.05 (scipy 版本无关)
    pairs = [
        (0.3, 0.6), (0.4, 0.7), (0.5, 0.8), (0.2, 0.5),
        (0.3, 0.6), (0.4, 0.7), (0.5, 0.8), (0.2, 0.5),
        (0.3, 0.6), (0.4, 0.7), (0.5, 0.8), (0.2, 0.5),
        (0.3, 0.6), (0.4, 0.7), (0.5, 0.8),
        (0.9, 0.5),  # -0.4 (一个离群)
    ]
    for i, (b, c) in enumerate(pairs):
        gate.record_pair("cfg_d", baseline_score=b, candidate_score=c, task_id=f"t{i}")
    d = gate.gate_decision("cfg_d")
    assert d.p_value is not None, "should have p_value"
    assert d.passed, f"15-positive-1-negative should pass: {d}"
    print(f"4. Wilcoxon mixed (15+, 1-) → pass (p={d.p_value:.4f}) OK")

    # 5. 持久化 reload
    sg.SignificanceGate._instance = None
    gate2 = sg.SignificanceGate.get_instance()
    pairs_b = gate2.get_pairs("cfg_b")
    assert len(pairs_b) == 5, f"persisted wrong: {len(pairs_b)}"
    assert all(p.candidate_score == 0.7 for p in pairs_b), "data corrupted"
    print(f"5. persistence reload OK (cfg_b: {len(pairs_b)} pairs)")

    # 6. passes_gate 快捷方法
    assert gate2.passes_gate("cfg_b"), "cfg_b should pass"
    assert not gate2.passes_gate("cfg_a"), "cfg_a should not pass"
    print("6. passes_gate shortcut OK")

    # 7. clear
    gate2.clear("cfg_b")
    assert len(gate2.get_pairs("cfg_b")) == 0, "clear failed"
    print("7. clear OK")

    shutil.rmtree(tmp, ignore_errors=True)
    del os.environ["HUGINN_CACHE_DIR"]
    sg.SignificanceGate._instance = None
    print("\nH5 significance_gate selfcheck OK (7/7)")


if __name__ == "__main__":
    _selfcheck()
