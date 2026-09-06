"""软门控得分 — 把 H5/H6 从 advisory 记录器升级为采纳优先级, 但永不删除数据.

背景: significance_gate (H5) / ood_holdout (H6) 原本是纯记录器, 零调用方。
直接把它们做成"硬闸"(未过门永久封禁) 会误杀仍在积累样本的候选:
  - min_samples=5 之前 Wilcoxon 功效极低, 必然 fail — 硬闸会杀掉所有新补丁;
  - OOD holdout 的 tolerance=0.1 是保守设计("宁可误杀"), 用于永久删除就过度;
  - 一次不佳跑分不该让一个结构上值得探索的方向消失。

方案: 三级评分 (GREEN/YELLOW/RED), 只决定"是否自动采纳"优先级, 永不删数据:

  GREEN   显著(p<α) 且 OOD 未退化 → 可自动采纳
  YELLOW  样本不足 / 显著但 OOD 未过 / OOD 过但样本不足
          → 不自动采纳但保留探索权, 数据继续积累
  RED     显著负向 / OOD 显著退化 → 不优先采纳, 降为 dormant (可手动复活)

核心原则: 门控只决定"是否自动用", 最坏结果是"少自动用一次", 而非"永久丢方向"。
开关复用 harness_significance_gate / harness_ood_holdout (任一开启即生效)。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ._enabled import _harness_enabled

logger = logging.getLogger(__name__)

# 三级状态
GREEN = "green"
YELLOW = "yellow"
RED = "red"


@dataclass
class AdoptionDecision:
    """软门控对单个候选的采纳决策.

    status: green/yellow/red
    adopt:  True=可自动采纳(GREEN), False=不自动采纳(黄/红)
    reason: 人类可读原因
    significance: H5 决策 dict (gate_decision().to_dict())
    ood:          H6 决策 dict (validate_ood().to_dict())
    """
    config_id: str
    status: str
    adopt: bool
    reason: str
    significance: dict[str, Any] | None = None
    ood: dict[str, Any] | None = None


class AdoptionGate:
    """组合 H5/H6 的软门控. 单例, 复用 SignificanceGate / OODHoldoutValidator.

    只读组合它们的决策, 不做任何数据删除。任何一层异常都降级为 YELLOW
    (不自动采纳但不判死), 绝不因门控崩溃而误杀候选。
    """

    _instance: AdoptionGate | None = None

    @classmethod
    def get_instance(cls) -> AdoptionGate:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def enabled(self) -> bool:
        """任一相关 harness 开关开启即生效 (advisory 评分记录)."""
        return _harness_enabled("harness_significance_gate") or _harness_enabled(
            "harness_ood_holdout"
        )

    def gate_strict(self) -> bool:
        """是否严格 gate (RED 不自动采纳).

        默认 False → 纯 advisory: 门控只评分/记录, 永不排除候选,
        避免"杀错"拖累 agent 能力. 显式开启 harness_adoption_gate 才拦截.
        """
        return _harness_enabled("harness_adoption_gate")

    def decide(
        self,
        config_id: str,
        alpha: float = 0.05,
        min_samples: int = 5,
        tolerance: float = 0.1,
    ) -> AdoptionDecision:
        """对 config_id 返回软门控决策.

        逻辑:
          - 显著通过 且 OOD 通过 → GREEN (adopt=True)
          - 显著通过 但 OOD 未过(样本不足或退化) → YELLOW
          - 未显著(样本不足) → YELLOW (不自动采纳, 继续积累)
          - 显著负向 / OOD 显著退化 → RED
        任一门控不可用 → 退回纯 advisory (YELLOW), 不阻断但不自动采纳。
        """
        if not self.enabled():
            return AdoptionDecision(
                config_id=config_id,
                status=GREEN,
                adopt=True,
                reason="harness gate disabled (advisory)",
            )

        # H5 显著性
        sig: dict[str, Any] | None = None
        try:
            from huginn.harness.significance_gate import SignificanceGate

            gd = SignificanceGate.get_instance().gate_decision(
                config_id, alpha=alpha, min_samples=min_samples
            )
            sig = gd.to_dict()
        except Exception:
            logger.debug("adoption_gate: significance read failed", exc_info=True)

        # H6 OOD
        ood: dict[str, Any] | None = None
        try:
            from huginn.harness.ood_holdout import OODHoldoutValidator

            or_ = OODHoldoutValidator.get_instance().validate_ood(
                config_id, tolerance=tolerance
            )
            ood = or_.to_dict()
        except Exception:
            logger.debug("adoption_gate: ood read failed", exc_info=True)

        # 门控数据完全不可用 → 不自动采纳但保留 (YELLOW), 防止门控本身成为漏洞
        if sig is None and ood is None:
            return AdoptionDecision(
                config_id=config_id,
                status=YELLOW,
                adopt=False,
                reason="no gate data available yet (advisory)",
                significance=sig,
                ood=ood,
            )

        sig_pass = bool(sig.get("passed")) if sig else None
        sig_n = int(sig.get("n_samples", 0)) if sig else 0
        median_diff = float(sig.get("median_diff", 0.0)) if sig else 0.0
        ood_pass = bool(ood.get("passed")) if ood else None
        # OOD"明确退化"by 数据充足时 degradation 超 tolerance; 样本不足不是退化
        ood_degraded = False
        if ood is not None:
            ood_holdout_n = int(ood.get("holdout_n", 0))
            ood_min = int(ood.get("min_per_split", 0)) or 3
            ood_tol = float(ood.get("tolerance", 0.1))
            if ood_pass is False and ood_holdout_n >= ood_min:
                ood_degraded = float(ood.get("degradation", 0.0)) > ood_tol

        # 显著负向 → RED
        if sig is not None and sig_n >= min_samples and median_diff < 0:
            return AdoptionDecision(
                config_id=config_id,
                status=RED,
                adopt=False,
                reason=f"significantly worse (median_diff={median_diff:.3f})",
                significance=sig,
                ood=ood,
            )

        # OOD 数据充足且明确退化 → RED
        if sig is not None and sig_pass is True and ood_degraded:
            return AdoptionDecision(
                config_id=config_id,
                status=RED,
                adopt=False,
                reason="significant but OOD degraded",
                significance=sig,
                ood=ood,
            )

        # 显著 且 OOD 通过(或样本不足未判死) → GREEN
        if sig_pass is True and ood_pass is not False:
            return AdoptionDecision(
                config_id=config_id,
                status=GREEN,
                adopt=True,
                reason="significant and OOD ok",
                significance=sig,
                ood=ood,
            )

        # 其余 (样本不足 / OOD 未过但未显著负向) → YELLOW
        return AdoptionDecision(
            config_id=config_id,
            status=YELLOW,
            adopt=False,
            reason="pending more evidence (insufficient or not decisive)",
            significance=sig,
            ood=ood,
        )

    def should_adopt(self, config_id: str) -> bool:
        """是否应自动采纳该候选.

        纯 advisory (gate_strict=False): 始终 True, 门控只评分不拦 —
        最坏结果只是"少自动采纳一次", 绝不排除任何候选, 保护 agent 能力.
        严格 gate: 仅 GREEN 采纳.
        """
        if not self.gate_strict():
            # advisory: 仍触发一次评分记录 (H5/H6 计数落盘), 但不拦截
            try:
                self.decide(config_id)
            except Exception:
                logger.debug("advisory decide failed", exc_info=True)
            return True
        return self.decide(config_id).adopt


def adoption_gate_enabled() -> bool:
    """供外部模块判断软门控是否生效 (语义入口)."""
    return AdoptionGate.get_instance().enabled()
