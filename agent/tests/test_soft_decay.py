"""I20: 软衰减可逆；硬修剪需两个独立信号 + 二次确认."""
from __future__ import annotations

from huginn.memory.dream import apply_soft_decay, restore_soft_decay
from huginn.memory.hygiene_gate import evaluate_retention


def test_reversible_and_two_signal():
    decayed, handle = apply_soft_decay(0.8, 0.5)
    assert decayed == 0.4
    assert restore_soft_decay(decayed, handle) == 0.8

    expired = {"memory_id": "m1", "expired": True, "importance": 0.9}
    # 只有过期一个信号 → 归档，不硬删
    assert evaluate_retention(expired, allow_prune=True).action == "archive"
    # 过期 + 预测失败，但未显式允许 → 仍归档
    predicted_failure = {**expired, "predicted_failure": True}
    assert evaluate_retention(predicted_failure).action == "archive"
    # 两个独立信号 + 显式允许 → 才允许 prune
    assert (
        evaluate_retention(predicted_failure, allow_prune=True).action == "prune"
    )
