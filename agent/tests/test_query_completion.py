"""缺度追问测试 — 缺失自由度 → 可执行补全查询.

覆盖:
- 只生成"已缺"自由度的查询, 已知的不再生 (不重复检索)
- urgency 分级: temperature(3) > functional(2) > method_family(1)
- should_retrieve: 关键缺度 (urgency>=2) 才值得发起真实检索
- 补全查询明确"补哪个维度 + 为什么"
- 纯函数: 幂等, 无 LLM/网络
- build_followup_input: 一键产出 target_query + fills + 已知条件 tail
"""

from __future__ import annotations

from huginn.tools.literature.query_completion import (
    build_followup_input,
    completion_query,
    summarize_missing,
)


class TestCompletionQuery:
    def test_known_dim_not_regenerated(self):
        # 已知 method_family=experiment → 无 method_family 补全查询, 只有 temperature
        qs = completion_query(
            ["temperature", "method_family"],
            "Li2O",
            "band_gap",
            "eV",
            known={"method_family": "experiment"},
        )
        dims = [q["target_dim"] for q in qs]
        assert "method_family" not in dims
        assert "temperature" in dims

    def test_urgency_ordering(self):
        qs = completion_query(
            ["method_family", "functional", "temperature"], "Li2O", "band_gap", "eV"
        )
        # 按 urgency 降序: temperature(3) 前
        assert qs[0]["target_dim"] == "temperature"
        assert qs[0]["urgency"] == 3
        # functional(2) 需要检索
        func = [q for q in qs if q["target_dim"] == "functional"][0]
        assert func["should_retrieve"] is True
        # method_family(1) 不满足 should_retrieve
        mf = [q for q in qs if q["target_dim"] == "method_family"][0]
        assert mf["should_retrieve"] is False

    def test_query_contains_system_property_unit(self):
        qs = completion_query(["temperature"], "Li2O", "band_gap", "eV")
        q = qs[0]["query"]
        assert "Li2O" in q
        assert "band_gap" in q
        assert "eV" in q

    def test_known_condition_appended(self):
        qs = completion_query(
            ["temperature"],
            "Li2O",
            "band_gap",
            "eV",
            known={"method_family": "experiment"},
        )
        assert "experiment" in qs[0]["query"]

    def test_no_missing_returns_empty(self):
        assert completion_query([], "Li2O", "band_gap", "eV") == []

    def test_unknown_dim_ignored(self):
        # 非模板缺度被忽略, 不空创查询
        qs = completion_query(["bogus_dim"], "Li2O", "band_gap", "eV")
        assert qs == []

    def test_idempotent(self):
        q1 = completion_query(["temperature"], "Li2O", "band_gap", "eV")
        q2 = completion_query(["temperature"], "Li2O", "band_gap", "eV")
        assert q1 == q2

    def test_empty_system_property(self):
        assert completion_query(["temperature"], "", "", "eV") == []


class TestBuildFollowupInput:
    def test_target_query_is_fill(self):
        res = build_followup_input(
            "Li2O",
            "band_gap",
            "eV",
            ["temperature", "method_family"],
            method_family="experiment",
        )
        assert res["target_query"] is not None
        assert "Li2O" in res["target_query"]
        fills = [f["target_dim"] for f in res["fills"]]
        assert fills == ["temperature"]

    def test_known_tail(self):
        res = build_followup_input(
            "Li2O",
            "band_gap",
            "eV",
            ["temperature"],
            method_family="dft",
            functional="HSE06",
            temperature="t_zero",
        )
        tail = res["known_conditions_tail"]
        assert "dft" in tail
        assert "HSE06" in tail


class TestSummarizeMissing:
    def test_readable(self):
        s = summarize_missing(["temperature", "functional"])
        assert "温度" in s["temperature"]
        assert "泛函" in s["functional"]
