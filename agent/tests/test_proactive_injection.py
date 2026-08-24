"""方向1/2/3 —— 三个 proactive 改动的自检.

1) 方向2: _inject_failed_direction_lessons 按 objective 关键词回灌失败教训
2) 方向1: agents.speculator.on_turn_start 的返回契约 (预测+预热+hint)
3) 方向3: FeatureFlags 里 curiosity_hint 默认开 + env 别名 HUGINN_CURIOSITY_HINT 生效
"""

from __future__ import annotations

from huginn.autoloop.cognitive_loop import _inject_failed_direction_lessons
from huginn.feature_flags import FeatureFlags


class _FakeMem:
    """伪造 memory, 只暴露 recall_failed_directions."""

    def __init__(self, recs):
        self._recs = recs

    def recall_failed_directions(self, limit=10, persona_id=None):
        return self._recs[:limit]


def test_direction2_lessons_kept_only_relevant_and_capped():
    recs = [
        ("dft relax silicon fails to converge", "scf not converged", "dft"),
        ("dft needs large smearing to converge", "bands overlap at gamma", "dft"),
        ("thermal anneal target missed", "temperature drifts", "anneal"),
        ("band gap underestimates gap", "pbe under gap", "band"),
    ]
    mem = _FakeMem(recs)
    # objective 带 "dft"+"converge", 命中前两条相关 (dft/converge 重叠), limit=2 截断.
    hint = _inject_failed_direction_lessons(
        "make dft converge faster", mem, None, "", limit=2
    )
    assert "[HISTORICAL LESSONS]" in hint
    assert "relax silicon" in hint
    assert "large smearing" in hint
    # 不相关的两条不该混进来
    assert "thermal anneal" not in hint
    assert "band gap" not in hint
    # 有 limit 上限
    assert hint.count("- ") == 2


def test_direction2_no_overlap_or_no_memory_returns_hint_untouched():
    unrelated = _FakeMem([("grain growth model", "bad seed", "grain")])
    # objective 与任何记录零重叠 → 原样返回.
    assert _inject_failed_direction_lessons(
        "just ask prices", unrelated, None, "|existing|"
    ) == "|existing|"
    # memory 缺 recall 接口 → 原样返回.
    assert _inject_failed_direction_lessons("dft", object(), None, "x") == "x"
    # memory 为 None → 原样返回.
    assert _inject_failed_direction_lessons("dft", None, None, "y") == "y"


def test_direction1_on_turn_start_contract():
    from unittest.mock import patch

    from huginn.agents import speculator as spec_mod

    class _FakeSpec:
        def predict(self, msg):
            return [spec_mod.Prediction("dft", 0.9, ["vasp_tool"], 0.9)]

        def prefetch(self, preds, cache=None):
            return {"prefetched": [{"tool": "vasp_tool"}], "skipped": [], "errors": []}

    with patch.object(spec_mod, "IntentSpeculator") as m:
        m.shared.return_value = _FakeSpec()
        res = spec_mod.on_turn_start("relax silicon", cache=None)

    # 闭包返回契约里该有的三件东西.
    assert res["predictions"] and res["predictions"][0]["scenario_name"] == "dft"
    assert res["prefetch_result"]["prefetched"][0]["tool"] == "vasp_tool"
    assert "vasp_tool" in res["hint"]


def test_direction3_curiosity_hint_default_on_and_alias(monkeypatch):
    # 默认打开
    assert FeatureFlags().is_enabled("curiosity_hint") is True
    # 旧 env 变量 HUGINN_CURIOSITY_HINT 别名能关掉它
    monkeypatch.setenv("HUGINN_CURIOSITY_HINT", "0")
    assert FeatureFlags().is_enabled("curiosity_hint") is False
    # 用正确的名也是同一路径
    monkeypatch.setenv("HUGINN_CURIOSITY_HINT", "1")
    assert FeatureFlags().is_enabled("curiosity_hint") is True
