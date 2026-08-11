"""在隔离的测试环境开启 H5/H6 两个 harness 开关并验证效果.

脚本用独立的临时 config + 临时 cache 目录, 不碰生产 huginn.toml.
通过环境变量 HUGINN_CONFIG_FILE 指向一个 feature_flags 里显式开启
两个开关的临时配置, 再验证:

  A. 开关确实生效 (_harness_enabled 返回 True)
  B. H5 SignificanceGate: 显著性检验
       - 真增益 (15正1负) → 通过
       - 纯噪声 (5正3负) → 拒绝
  C. H6 OODHoldout: 分布外留出
       - 泛化好的候选 → 通过
       - 背题补丁 (train 好 holdout 差) → 拦截

跑法:
  cd agent && python harness_gates_verify.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# ── 0. import 前设置测试环境 ─────────────────────────────────────────
_TEST_DIR = tempfile.mkdtemp(prefix="huginn_gates_verify_")
_CACHE_DIR = str(Path(_TEST_DIR) / "cache")
_CONFIG_PATH = str(Path(_TEST_DIR) / "huginn.toml")

os.environ["HUGINN_CACHE_DIR"] = _CACHE_DIR
os.environ["HUGINN_CONFIG_FILE"] = _CONFIG_PATH

# 写临时配置: 显式开启两个 harness 开关
_CONFIG_TOML = """\
provider = "deepseek"
model = "deepseek-reasoner"
api_key = ""
config_version = 1

[feature_flags]
harness_significance_gate = true
harness_ood_holdout = true
"""
Path(_CONFIG_PATH).write_text(_CONFIG_TOML, encoding="utf-8")

# 清 config 缓存, 确保加载上面的临时配置
from huginn.config import clear_config_cache  # noqa: E402

clear_config_cache()

import huginn.harness.ood_holdout as ood  # noqa: E402
import huginn.harness.significance_gate as sg  # noqa: E402

_PASS = 0
_FAIL = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    mark = "PASS" if cond else "FAIL"
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    print("Huginn harness gates — 测试环境开启验证")
    print(f"临时 config: {_CONFIG_PATH}")
    print(f"临时 cache : {_CACHE_DIR}")
    print()

    # ── A. 开关确实生效 ─────────────────────────────────────────────
    print("A. 开关生效检查")
    check(
        "harness_significance_gate 开启",
        sg._harness_enabled("harness_significance_gate") is True,
        f"实际={sg._harness_enabled('harness_significance_gate')}",
    )
    check(
        "harness_ood_holdout 开启",
        ood._harness_enabled("harness_ood_holdout") is True,
        f"实际={ood._harness_enabled('harness_ood_holdout')}",
    )
    print()

    # ── B. H5 SignificanceGate 端到端 ───────────────────────────────
    print("B. H5 显著性检验 (Wilcoxon signed-rank)")
    sg.SignificanceGate._instance = None
    gate = sg.SignificanceGate.get_instance()

    # B1 真增益: 15 正 + 1 负, 应通过
    for i in range(16):
        if i < 15:
            gate.record_pair("candidate_real", 0.4, 0.7, f"t{i}")
        else:
            gate.record_pair("candidate_real", 0.9, 0.5, f"t{i}")
    d = gate.gate_decision("candidate_real")
    check(
        "真增益 → 通过显著门",
        d.passed,
        f"p={d.p_value:.4f}, n={d.n_samples}, median_diff={d.median_diff:.2f}",
    )

    # B2 纯噪声: 5 正 + 3 负, 应拒绝
    for b, c in ((0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6), (0.5, 0.6),
                 (0.5, 0.4), (0.5, 0.4), (0.5, 0.4)):
        gate.record_pair("candidate_noise", b, c, "noise")
    d2 = gate.gate_decision("candidate_noise")
    check(
        "纯噪声 → 拒绝 (不误放)",
        not d2.passed,
        f"p={d2.p_value:.4f}, n={d2.n_samples}",
    )
    print()

    # ── C. H6 OODHoldout 端到端 ─────────────────────────────────────
    print("C. H6 分布外留出验证 (防背题补丁)")
    ood.OODHoldoutValidator._instance = None
    val = ood.OODHoldoutValidator.get_instance()

    # 构造 train/holdout 任务
    tasks = [f"task_{i:03d}" for i in range(30)]
    train = [t for t in tasks if not ood._is_holdout(t)]
    hold = [t for t in tasks if ood._is_holdout(t)]

    # C1 泛化好的候选: baseline 0.5, candidate 0.6 (train+holdout 都好)
    for t in tasks:
        val.record_outcome(ood.OODHoldoutValidator._BASELINE_ID, t, score=0.5)
        val.record_outcome("candidate_good", t, score=0.6)
    r = val.validate_ood("candidate_good")
    check(
        "泛化好的候选 → 通过 OOD",
        r.passed,
        f"degradation={r.degradation:.1%}, train_med={r.train_median:.2f}, "
        f"holdout_med={r.holdout_median:.2f}",
    )

    # C2 背题补丁: baseline 0.5, candidate train 0.8 / holdout 0.2
    for t in tasks:
        val.record_outcome(ood.OODHoldoutValidator._BASELINE_ID + "_of", t, score=0.5)
    for t in train:
        val.record_outcome("candidate_overfit", t, score=0.8)
    for t in hold:
        val.record_outcome("candidate_overfit", t, score=0.2)
    r2 = val.validate_ood("candidate_overfit")
    check(
        "背题补丁 → OOD 拦截",
        not r2.passed,
        f"degradation={r2.degradation:.1%}, train_med={r2.train_median:.2f}, "
        f"holdout_med={r2.holdout_median:.2f}",
    )
    print()

    # ── 收尾 ────────────────────────────────────────────────────────
    print(f"结果: {_PASS} PASS / {_FAIL} FAIL")
    shutil.rmtree(_TEST_DIR, ignore_errors=True)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
