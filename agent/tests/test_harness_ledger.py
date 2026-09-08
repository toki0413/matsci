"""Tests for HarnessLedger (M3: 跨 run 按 task_episode 聚合, 组织层治理账本).

覆盖: append+persist+load 往返 / 同 episode 多 run 聚合均值与计数 / 跨 run 追溯 /
evidence 三态占比暴露 / 格式异常 fail-open skip.
"""
from __future__ import annotations

import json

from huginn.research.harness import HarnessReport
from huginn.research.harness_ledger import HarnessLedger


def _harness(ep: str, overall: float, *, agent: str = "a", machine: str = "m",
             evid: str = "observed") -> dict:
    """构造一条 to_dict() 形态的 harness(dimensions 精简为一个观测维度)."""
    return {
        "task_episode": ep,
        "goal": "g",
        "agent": agent,
        "machine": machine,
        "overall": overall,
        "dimensions": [{"name": "task_understanding", "score": overall, "evidence": evid,
                        "checks": []}],
    }


# ── append + persist + load 往返 ─────────────────────────────────────────
def test_append_persist_load_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ep = "ep-aaaa"
    ledger = HarnessLedger()
    ledger.append(_harness(ep, 0.8)).append(_harness(ep, 0.6))
    ledger.persist(path)

    loaded = HarnessLedger.load(path)
    assert loaded.skipped == 0
    assert len(loaded.entries) == 2
    # 逐行 JSONL: 每行都是一条可反序列化的 harness
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert [e["overall"] for e in lines] == [0.8, 0.6]
    assert all(e["task_episode"] == ep for e in lines)


def test_append_accepts_harnessreport_object():
    ledger = HarnessLedger()
    ledger.append(HarnessReport(goal="g", task_episode="ep-r", overall=0.9))
    assert len(ledger.entries) == 1
    assert ledger.entries[0]["task_episode"] == "ep-r"


# ── 同 episode 多 run 聚合均值/计数 ───────────────────────────────────────
def test_summary_aggregates_mean_min_max_count():
    ep = "ep-bbbb"
    ledger = HarnessLedger()
    for v in (0.8, 0.9, 1.0):
        ledger.append(_harness(ep, v))
    s = ledger.summary(ep)
    # 均值/min/max/count 如实呈现(不补分)
    assert s["runs"] == 3
    assert s["overall"]["count"] == 3
    assert abs(s["overall"]["mean"] - 0.9) < 1e-3
    assert s["overall"]["min"] == 0.8
    assert s["overall"]["max"] == 1.0
    # 每维平均分
    assert abs(s["dimensions"]["task_understanding"]["score_mean"] - 0.9) < 1e-3


def test_summary_none_lists_all_episodes():
    ledger = HarnessLedger()
    ledger.append(_harness("ep-a", 0.7)).append(_harness("ep-b", 0.5))
    alls = ledger.summary()
    assert set(alls) == {"ep-a", "ep-b"}
    assert alls["ep-a"]["overall"]["count"] == 1


# ── 跨 run 追溯 ──────────────────────────────────────────────────────────
def test_cross_run_lists_runs_in_order():
    ep = "ep-cccc"
    ledger = HarnessLedger()
    ledger.append(_harness(ep, 0.5, agent="a"))
    ledger.append(_harness(ep, 0.9, agent="b"))
    ledger.append(_harness("ep-other", 0.1))  # 不应混进来
    runs = ledger.cross_run(ep)
    assert len(runs) == 2
    # 按 append 序编号(序号=时间序), 供 reading 追溯「同一需求的多次实录」
    assert [r["run"] for r in runs] == [0, 1]
    assert [r["overall"] for r in runs] == [0.5, 0.9]
    assert [r["agent"] for r in runs] == ["a", "b"]


# ── evidence 三态占比暴露 ────────────────────────────────────────────────
def test_summary_exposes_evidence_ratio():
    ep = "ep-dddd"
    ledger = HarnessLedger()
    ledger.append(_harness(ep, 1.0, evid="observed"))
    ledger.append(_harness(ep, 0.5, evid="unobserved"))
    ledger.append(_harness(ep, 0.5, evid="missing"))
    s = ledger.summary(ep)
    ev = s["dimensions"]["task_understanding"]["evidence"]
    # 三态占比如实暴露: observed 只占 1/3 → 即「靠没跑的门禁凑分」的维度被点名
    assert ev["observed"] == 1 and ev["unobserved"] == 1 and ev["missing"] == 1
    assert ev["observed_ratio"] == round(1 / 3, 3)
    assert ev["unobserved_ratio"] == round(1 / 3, 3)
    assert ev["missing_ratio"] == round(1 / 3, 3)


# ── 格式异常 fail-open skip ──────────────────────────────────────────────
def test_append_skips_malformed_without_raising():
    ledger = HarnessLedger()
    ledger.append(None)                                                          # 非 dict
    ledger.append({"overall": 1.0, "dimensions": []})                            # 缺 task_episode
    ledger.append({"task_episode": "ep-x", "overall": 1.0})                      # 缺 dimensions
    ledger.append({"task_episode": "", "dimensions": []})                        # 空 episode
    ledger.append({"task_episode": "ep-x", "dimensions": "nope"})                # dimensions 非 list
    assert ledger.skipped == 5
    assert len(ledger.entries) == 0


def test_load_roundtrip_skips_corrupt_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        json.dumps(_harness("ep-e", 0.7), ensure_ascii=False) + "\n"
        + "not-json{{{bad}\n"                                    # 损坏行
        + json.dumps(_harness("ep-e", 0.8), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ledger = HarnessLedger.load(path)
    assert ledger.skipped == 1
    assert len(ledger.entries) == 2


def test_persist_replaces_atomically(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = HarnessLedger()
    ledger.append(_harness("ep-f", 0.7)).persist(path)
    ledger.append(_harness("ep-f", 0.8)).persist(path)  # 再次写入应仍是完整两本
    assert len(HarnessLedger.load(path).entries) == 2
