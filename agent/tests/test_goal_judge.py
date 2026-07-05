"""GoalJudge — 端到端目标达成判定测试."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

from huginn.evaluation.goal_judge import GoalJudge, _summarize_trajectory


def _mock_llm(response: str):
    """构造一个 fake LLM, invoke 返回固定内容."""
    mock = MagicMock()
    msg = MagicMock()
    msg.content = response
    mock.invoke.return_value = msg
    return mock


def test_rule_based_empty_output():
    """空产出 -> 未达成."""
    judge = GoalJudge(llm=None)
    r = judge.judge("compute Si band gap", final_output="")
    assert not r["achieved"]
    assert r["score"] == 0.0
    assert "no output produced" in r["gaps"]


def test_rule_based_keyword_match():
    """关键词覆盖 -> 部分达成."""
    judge = GoalJudge(llm=None)
    r = judge.judge(
        "compute silicon band gap",
        final_output="The silicon band gap is approximately 1.12 eV.",
    )
    assert r["achieved"]
    assert r["score"] > 0.5


def test_rule_based_low_coverage():
    """关键词不匹配 -> 未达成."""
    judge = GoalJudge(llm=None)
    r = judge.judge(
        "compute silicon band gap",
        final_output="The weather is nice today.",
    )
    assert not r["achieved"]


def test_llm_judge_achieved():
    """LLM 判定达成."""
    llm_resp = json.dumps({
        "achieved": True,
        "score": 0.9,
        "evidence": ["correct band gap value"],
        "gaps": [],
        "reasoning": "all questions answered",
    })
    judge = GoalJudge(llm=_mock_llm(llm_resp))
    r = judge.judge(
        "计算 Si 间接带隙",
        final_output="Si 的间接带隙约为 1.17 eV...",
    )
    assert r["achieved"]
    assert r["score"] == 0.9


def test_llm_judge_not_achieved():
    """LLM 判定未达成."""
    llm_resp = json.dumps({
        "achieved": False,
        "score": 0.3,
        "evidence": [],
        "gaps": ["missing mechanism explanation"],
        "reasoning": "only gave number, no physics",
    })
    judge = GoalJudge(llm=_mock_llm(llm_resp))
    r = judge.judge("compute band gap with mechanism", final_output="1.12 eV")
    assert not r["achieved"]
    assert len(r["gaps"]) == 1


def test_llm_markdown_wrapped():
    """LLM 返回 markdown 包裹的 JSON 也能解析."""
    llm_resp = '```json\n{"achieved": true, "score": 0.85, "evidence": ["ok"], "gaps": [], "reasoning": "good"}\n```'
    judge = GoalJudge(llm=_mock_llm(llm_resp))
    r = judge.judge("test", final_output="output")
    assert r["achieved"]
    assert r["score"] == 0.85


def test_llm_invalid_json_fallback():
    """LLM 返回非 JSON -> 降级到规则判定."""
    judge = GoalJudge(llm=_mock_llm("this is not json at all"))
    r = judge.judge("silicon band gap", final_output="silicon band gap 1.12 eV")
    # 降级到 rule-based, 应该命中关键词
    assert r["achieved"]
    assert "rule-based" in r["reasoning"]


def test_llm_invoke_fails_fallback():
    """LLM 调用抛异常 -> 降级到规则."""
    bad_llm = MagicMock()
    bad_llm.invoke.side_effect = RuntimeError("API down")
    judge = GoalJudge(llm=bad_llm)
    r = judge.judge("silicon band gap", final_output="silicon band gap 1.12 eV")
    assert r["achieved"]  # rule-based fallback
    assert "rule-based" in r["reasoning"]


def test_trajectory_summary():
    """轨迹摘要格式正确."""
    traj = {
        "tool_calls": [
            {"tool": "vasp_tool", "success": True, "result": "energy=-24.0"},
            {"tool": "xrd_tool", "success": False, "result": "error"},
        ]
    }
    s = _summarize_trajectory(traj)
    assert "2 个工具调用" in s
    assert "vasp_tool" in s
    assert "xrd_tool" in s
    assert "✓" in s
    assert "✗" in s


def test_trajectory_summary_empty():
    """空轨迹摘要."""
    assert "(无轨迹数据)" in _summarize_trajectory(None)
    assert "(无工具调用记录)" in _summarize_trajectory({})


def test_trajectory_summary_phases_only():
    """只有 phases 列表也能摘要."""
    traj = {"phases": ["perceive", "hypothesize", "plan"]}
    s = _summarize_trajectory(traj)
    assert "3 个阶段" in s
    assert "perceive" in s
