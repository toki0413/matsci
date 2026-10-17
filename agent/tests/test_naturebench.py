"""NatureBench-mini 复现门禁 — 论文复现评测套件的确定性校验.

这不是"跑一遍 agent 看能不能做对"的评测（那需要 LLM/key，属重型集成），
而是把评测套件本身当作被测对象：题集结构、SOTA 判分逻辑、best_score 跟踪
必须可复现地正确。任何改动若让"给定论文 SOTA 值判不准 / 明显偏离仍算过"，
这里立刻变红 —— 这是 user 批评里"评测有壳无闭环"的闭环层。

纯数值 + 纯逻辑，无网络、无 API key、无 LLM，可进 CI 快速门禁。
"""
from __future__ import annotations

import pytest

from huginn.bench import get_suite_tasks
from huginn.bench.naturebench import _make_best_score_eval, build_naturebench_tasks

# SOTA 值 (来自各论文) 与对应"正确答案字符串", 用于验证判分链路真的认对.
# 与 build_naturebench_tasks 里的 evaluator 一一对应 (顺序即题序).
_SOTA_CORRECT = [
    "钙钛矿带隙 Eg = 1.55 eV",
    "单层带隙 Eg = 1.80 eV",
    "扩散系数 D = 1.2e-12 m²/s",
    "费米速度 vF = 1.0e6 m/s",
    "屈服强度 σ = 1.0 GPa",
    "热电优值 ZT = 2.6",
    "Dirac point = -0.35 eV",
    "CO2 吸附量 = 3.0 mmol/g",
    "临界温度 Tc = 203 K",
    "电导率 σ = 1.0e-2 S/cm",
]


# ── 题集结构 ────────────────────────────────────────────────────


def test_builds_ten_naturebench_tasks():
    tasks = build_naturebench_tasks()
    assert len(tasks) == 10


def test_task_ids_unique_and_categorized():
    tasks = build_naturebench_tasks()
    ids = [t.id for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate naturebench task ids"
    assert all(t.category == "naturebench" for t in tasks)
    assert all("naturebench" in t.tags for t in tasks)


def test_tasks_are_llm_repro_tasks():
    """每题都该是需要模型 + key 的真实复现任务, 且有论文出处."""
    tasks = build_naturebench_tasks()
    for t in tasks:
        assert t.prompt, f"{t.id}: empty prompt"
        assert t.requires_api_key, f"{t.id}: should require an API key"
        assert t.reference, f"{t.id}: missing paper reference"


def test_suite_registration_returns_same_tasks():
    """get_suite_tasks('naturebench') 与直接 build 一致 (注册未漂移)."""
    assert [t.id for t in get_suite_tasks("naturebench")] == [
        t.id for t in build_naturebench_tasks()
    ]


# ── 判分链路: 给定论文 SOTA 值必须判过 ──────────────────────────


@pytest.mark.parametrize("idx", range(10))
def test_correct_sota_answer_passes(idx):
    """每题给论文 SOTA 值 -> passed=True, score=1.0."""
    task = build_naturebench_tasks()[idx]
    res = task.evaluate(_SOTA_CORRECT[idx])
    assert res.passed, f"{task.id}: SOTA answer should pass: {res.reason}"
    assert res.score == pytest.approx(1.0)


def test_wrong_answer_fails():
    """明显偏离 -> 不通过 (带隙 3.0 eV vs SOTA 1.55)."""
    task = build_naturebench_tasks()[0]
    res = task.evaluate("带隙 Eg = 3.0 eV")
    assert not res.passed
    assert res.score < 1.0


def test_unparseable_answer_scores_zero():
    """没有可提取数值 -> score=0, 不通过."""
    task = build_naturebench_tasks()[0]
    res = task.evaluate("无法给出数值")
    assert not res.passed
    assert res.score == pytest.approx(0.0)


# ── best_score 跟踪 (NatureBench 核心设计: 多次提交取最优) ───────


def test_best_score_tracks_maximum():
    ev = _make_best_score_eval(
        sota=1.55, tol=0.1, unit="eV", key_hint=r"(?:Eg|带隙)",
    )
    ev("Eg = 1.40 eV")  # 接近但不达 SOTA -> 0.5
    ev("Eg = 1.55 eV")  # 达到 SOTA -> 1.0
    ev("Eg = 1.50 eV")  # 仍达 SOTA
    assert ev.state["n_calls"] == 3
    # best 保留最高分那次
    assert ev.state["best"][1] == 1.0


def test_three_tier_tolerance():
    """三段容差: <=tol -> 1.0, <=2tol -> 0.5, 更远 -> 0.3.

    数值直接跟在 key_hint 前缀后 (``v 2.05``), 走上限前缀提取链路.
    """
    ev = _make_best_score_eval(
        sota=2.0, tol=0.1, unit="x", key_hint=r"v",
    )
    assert ev("v 2.05")[2] == pytest.approx(1.0)   # 距离 0.05 <= 0.1
    assert ev("v 2.15")[2] == pytest.approx(0.5)   # 距离 0.15 <= 0.2
    assert ev("v 5.0")[2] == pytest.approx(0.3)    # 距离 3, 远离


def test_log_space_judging():
    """跨数量级指标 (扩散系数) 在对数空间判分."""
    ev = _make_best_score_eval(
        sota=1e-12, tol=0.5, unit="m²/s", key_hint=r"D", log_space=True,
    )
    # 同一数量级 -> 达 SOTA
    assert ev("D 1e-12 m²/s")[2] == pytest.approx(1.0)
    # 差 2 个数量级 (log10 距离 2 > 2*tol=1) -> 远离
    assert ev("D 1e-14 m²/s")[2] == pytest.approx(0.3)
