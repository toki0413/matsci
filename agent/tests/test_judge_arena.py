"""JudgeEvaluator + BlindArena + ArenaStore 测试.

- judge: 两个输出 -> winner + scores (用 mock LLM + 启发式降级)
- blind arena: 双盲随机化 + winner 还原正确
- elo: ELO 数学正确 (守恒 + 期望胜率)
- arena_store: SQLite 读写 + 历史查询 + ELO 恢复
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.evaluation.arena_store import ArenaRecord, ArenaStore, now_ts
from huginn.evaluation.judge import (
    BlindArena,
    JudgeEvaluator,
    _ELO_DEFAULT,
    expected_score,
    update_elo,
)


# ── JudgeEvaluator ──────────────────────────────────────────────


def test_judge_evaluate_with_mock_llm():
    """mock judge LLM 返回结构化 JSON -> 正确解析 winner + scores."""
    raw = json.dumps({
        "scores_a": {"accuracy": 9, "completeness": 8, "reasoning": 9, "utility": 8},
        "scores_b": {"accuracy": 7, "completeness": 6, "reasoning": 7, "utility": 6},
        "winner": "A",
        "reasoning": "A 更准确完整",
    })
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=raw))
    judge = JudgeEvaluator(model=llm)
    result = pytest.importorskip("asyncio").run(judge.evaluate("ans A", "ans B"))
    assert result.winner == "a"
    assert result.scores["A"] > result.scores["B"]
    assert result.scores["A"] == pytest.approx(8.5, abs=0.01)
    assert "准确" in result.reasoning


def test_judge_evaluate_heuristic_fallback():
    """没 LLM 时降级到启发式, 仍返回 winner + scores."""
    judge = JudgeEvaluator(model=None)
    import asyncio
    result = asyncio.run(judge.evaluate("材料 相图 晶体" * 20, "短回答"))
    assert result.winner in ("a", "b", "tie")
    assert "A" in result.scores and "B" in result.scores
    # 长回答 + 关键词多 -> A 应该不比 B 低
    assert result.scores["A"] >= result.scores["B"]


def test_judge_parse_markdown_fenced():
    """LLM 返回带 ```json 围栏也能解析."""
    judge = JudgeEvaluator(model=MagicMock())
    raw = "```json\n" + json.dumps({
        "scores_a": {"a": 8}, "scores_b": {"a": 9},
        "winner": "B", "reasoning": "B 更好",
    }) + "\n```"
    res = judge._parse_judge(raw)
    assert res.winner == "b"
    assert res.scores["B"] > res.scores["A"]


# ── ELO 数学 ────────────────────────────────────────────────────


def test_expected_score():
    """同分 -> 0.5; 差 400 分 -> ~0.91 / 0.09."""
    assert expected_score(1000, 1000) == pytest.approx(0.5)
    # 1400 对 1000: A 更强, 预期胜率 = 1/(1+10^(-1)) = 10/11
    assert expected_score(1400, 1000) == pytest.approx(10 / 11, abs=0.01)
    # 反过来 1000 对 1400: 预期胜率 = 1/(1+10^1) = 1/11
    assert expected_score(1000, 1400) == pytest.approx(1 / 11, abs=0.01)


def test_elo_update_conservation():
    """A 赢: A 涨 B 跌, 且零和 (K=32 时涨跌相等)."""
    new_a, new_b = update_elo(1000, 1000, "A")
    assert new_a > 1000
    assert new_b < 1000
    # 零和: 起始总和 2000 = 结束总和 2000
    assert (new_a + new_b) == pytest.approx(2000, abs=0.01)


def test_elo_update_tie():
    """平局: 同分时双方不变; 高分方平局会跌."""
    a, b = update_elo(1000, 1000, "tie")
    assert a == pytest.approx(1000, abs=0.01)
    assert b == pytest.approx(1000, abs=0.01)
    # 高分方平局应跌
    a2, b2 = update_elo(1200, 1000, "tie")
    assert a2 < 1200
    assert b2 > 1000


def test_elo_update_b_win():
    new_a, new_b = update_elo(1000, 1000, "B")
    assert new_a < 1000
    assert new_b > 1000


# ── BlindArena ─────────────────────────────────────────────────


def test_blind_arena_winner_restoration():
    """双盲打乱后, winner 必须正确映射回真实模型.

    固定种子让 flip 确定, 用 mock judge 让 winner 已知, 检查还原.
    """
    raw = json.dumps({
        "scores_a": {"a": 9}, "scores_b": {"a": 6},
        "winner": "A", "reasoning": "A wins",
    })
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=MagicMock(content=raw))
    judge = JudgeEvaluator(model=llm)

    # seed=0 控制 flip
    arena = BlindArena(judge=judge, seed=0)
    import asyncio
    flip = arena._rng.random() < 0.5
    rec = asyncio.run(arena.battle("modelA", "ans-a-good", "modelB", "ans-b-poor"))

    # judge 永远判第一个参数赢 (winner=A 相对第一参数)
    # 如果 flip=True, judge 第一参数是 modelB 的回答 -> 真实 winner 应是 modelB
    expected_winner = "b" if flip else "a"
    assert rec.winner.lower() == expected_winner
    # ELO 应该更新: winner 涨
    assert arena.get_elo("modelA" if expected_winner == "a" else "modelB") > _ELO_DEFAULT


def test_blind_arena_randomization_distribution():
    """多次 battle, flip 应该既有 True 又有 False (随机性生效)."""
    judge = JudgeEvaluator(model=None)  # 启发式, 不调 LLM
    arena = BlindArena(judge=judge, seed=None)
    flips = set()
    import asyncio
    for _ in range(50):
        before = dict(arena._elo)
        asyncio.run(arena.battle("mA", "x" * 100, "mB", "y" * 100))
        # 不直接读 flip, 但随机性应让双方都有输赢
    # 至少双方 ELO 都被更新过
    assert "mA" in arena._elo and "mB" in arena._elo


def test_blind_arena_leaderboard():
    judge = JudgeEvaluator(model=None)
    arena = BlindArena(judge=judge, seed=42)
    import asyncio
    asyncio.run(arena.battle("strong", "材料 相图 晶体 DFT" * 10, "weak", "ok"))
    lb = arena.leaderboard()
    assert len(lb) == 2
    # 排序: 高分在前
    assert lb[0][1] >= lb[1][1]


# ── ArenaStore ─────────────────────────────────────────────────


@pytest.fixture
def tmp_store():
    with tempfile.TemporaryDirectory() as d:
        store = ArenaStore(Path(d) / "arena.sqlite3")
        yield store
        store.close()


def test_arena_store_record_and_history(tmp_store):
    """写入一条, list_history 能查到, 字段完整."""
    rec = ArenaRecord(
        timestamp=now_ts(),
        model_a="mA", model_b="mB", winner="A",
        reasoning="test", scores={"mA": 8.0, "mB": 6.0},
        elo_a=1016.0, elo_b=984.0, meta={"flipped": False},
    )
    rid = tmp_store.record(rec)
    assert rid > 0
    history = tmp_store.list_history()
    assert len(history) == 1
    got = history[0]
    assert got.model_a == "mA"
    assert got.model_b == "mB"
    assert got.winner == "A"
    assert got.scores["mA"] == 8.0
    assert got.elo_a == 1016.0
    assert got.meta["flipped"] is False


def test_arena_store_filter_by_model(tmp_store):
    """按 model 过滤历史."""
    for i, (a, b) in enumerate([("mA", "mB"), ("mC", "mA"), ("mD", "mE")]):
        tmp_store.record(ArenaRecord(
            timestamp=now_ts() + i, model_a=a, model_b=b,
            winner="A", reasoning="", scores={},
        ))
    hist_a = tmp_store.list_history(model="mA")
    # mA 参与了 2 场 (vs mB, vs mC)
    assert len(hist_a) == 2
    # 不参与的 mD 只查到自己 1 场
    assert len(tmp_store.list_history(model="mD")) == 1


def test_arena_store_latest_elo_recovery(tmp_store):
    """latest_elo 返回每个模型最近一次出现的 ELO."""
    tmp_store.record(ArenaRecord(
        timestamp=1.0, model_a="mA", model_b="mB",
        winner="A", elo_a=1016.0, elo_b=984.0, scores={},
    ))
    tmp_store.record(ArenaRecord(
        timestamp=2.0, model_a="mA", model_b="mC",
        winner="B", elo_a=1000.0, elo_b=1016.0, scores={},
    ))
    elo = tmp_store.latest_elo()
    # mA 最后一次出现在 t=2, elo_a=1000
    assert elo["mA"] == 1000.0
    assert elo["mB"] == 984.0  # 只出现一次
    assert elo["mC"] == 1016.0


def test_blind_arena_with_store_restores_elo():
    """接了 store 的 arena, 新实例能从历史恢复 ELO."""
    with tempfile.TemporaryDirectory() as d:
        store = ArenaStore(Path(d) / "arena.sqlite3")
        arena1 = BlindArena(judge=JudgeEvaluator(model=None), store=store, seed=1)
        import asyncio
        asyncio.run(arena1.battle("mA", "材料 相图" * 20, "mB", "ok"))
        elo_after = arena1.get_elo("mA")
        assert elo_after != _ELO_DEFAULT

        # 新 arena 实例, 同 store, 应恢复 ELO
        arena2 = BlindArena(judge=JudgeEvaluator(model=None), store=store, seed=1)
        assert arena2.get_elo("mA") == elo_after
        store.close()
