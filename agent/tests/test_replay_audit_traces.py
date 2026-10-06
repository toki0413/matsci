"""replay_audit 控制面 trace 计数 + 判词可见性回归.

背景: replay_audit 是"离线重放历史轨迹, 核对无进展轮是否**可观测**"的工具
(见 :mod:`huginn.autoloop.replay_audit`). A2 后线上终止出口只留挂钟/目标达成,
故"无进展是否可观测"的实测证据 = run.log 里落盘的 ``control_trace`` 计数.

两个已修/须锁的坑:
1. **计数**: 本轮新登记的 ``progress_invariant`` / ``llm_unavailable`` /
   ``branch_slice_skip`` / ``code_lab_slice_skip`` 必须能被 ``scan_runlog`` 抓进
   ``exits.control_traces`` —— 抓不到就等于观测面失明.
2. **判词可见性**: 原 ``_verdict`` 只在 ``soft>0`` 时才打印 control_trace 行;
   零进展但无软动作的轮 (如 run80 只发 ``branch_slice_skip``) 会被整段吞掉,
   证据只在 ``--json`` 里可见 = 判词失效. 本测试锁"有任何 trace 就要报".
"""
from __future__ import annotations

from pathlib import Path

from huginn.autoloop import replay_audit as ra

# 四类本轮新登记机制的真实 run.log 行形态 (字段顺序与线上 emit 一致).
_TRACE_LINES = [
    "control_trace name=branch_slice_skip iteration=1 "
    "evidence=slice=layer2 (return layer1): budget<212s action=skip",
    "control_trace name=code_lab_slice_skip iteration=3 "
    "evidence=slice=repair#1: budget<max(min,180s) remaining=20s action=skip",
    "control_trace name=progress_invariant iteration=7 "
    "evidence=window=4 tail=hypothesize,plan,hypothesize,plan action=force_route",
    "control_trace name=llm_unavailable iteration=9 "
    "evidence=action=hypothesize reason=transient_empty action=retry_in_place",
    # 第二次 branch_slice_skip: 验证"按名累加"而非"只记存在"
    "control_trace name=branch_slice_skip iteration=1 "
    "evidence=slice=layer1_value: budget<212s action=skip",
]


def _make_run(tmp_path: Path) -> str:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.log").write_text("\n".join(_TRACE_LINES) + "\n", encoding="utf-8")
    return str(run_dir)


def test_scan_runlog_counts_new_mechanism_traces(tmp_path: Path) -> None:
    """scan_runlog 按名累加四类新 trace (bad 名抓不到 = 观测失明)."""
    ev = ra.scan_runlog(_make_run(tmp_path))
    assert ev["traces"]["branch_slice_skip"] == 2, ev["traces"]
    assert ev["traces"]["code_lab_slice_skip"] == 1, ev["traces"]
    assert ev["traces"]["progress_invariant"] == 1, ev["traces"]
    assert ev["traces"]["llm_unavailable"] == 1, ev["traces"]


def test_audit_surfaces_traces_in_exits(tmp_path: Path) -> None:
    """audit 把计数带进 exits.control_traces (JSON 消费面)."""
    a = ra.audit(_make_run(tmp_path))
    ct = a["exits"]["control_traces"]
    assert ct.get("branch_slice_skip") == 2, ct
    assert ct.get("code_lab_slice_skip") == 1, ct
    assert ct.get("progress_invariant") == 1, ct
    assert ct.get("llm_unavailable") == 1, ct


def test_verdict_reports_traces_without_soft_actions(tmp_path: Path) -> None:
    """零进展 / 无软动作的轮也要在判词里露出在盘 trace (原 soft>0 门会吞掉)."""
    a = ra.audit(_make_run(tmp_path))
    # 前提: 该合成轨迹没有任何软动作 —— 正是会触发原门控 bug 的形态.
    assert a["exits"]["soft"] == 0, a["exits"]
    lines = ra._verdict(a)
    joined = "\n".join(lines)
    assert "落盘 control_trace" in joined, joined
    assert "branch_slice_skip×2" in joined, joined
    assert "progress_invariant×1" in joined, joined