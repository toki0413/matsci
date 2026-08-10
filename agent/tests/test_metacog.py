"""metacog/ 模块单元测试 — 覆盖纯逻辑 helper + 数据结构 + 解析器.

只测无外部依赖 (LLM / KB / memory / 网络) 的部分, 保证 fast CI 不被拖慢.
重 IO 的 _llm_evaluate / evaluate_step / HypothesisManifold.mcmc_step 等留给 e2e.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from huginn.metacog import hypothesis_manifold as hm
from huginn.metacog import llm_likelihood as ll
from huginn.metacog import step_evaluator as se
from huginn.metacog import target_chain as tc
from huginn.metacog.reflector import (
    ReflectorAction,
    format_reflector_text,
    reflect,
)
from huginn.metacog.step_evaluator import (
    MeasurementUncertainty,
    StepEvaluation,
    ToolCallHealth,
    check_uncertainty_propagation,
    compute_tool_call_health,
)

# ───────────────────────── reflector ─────────────────────────


class TestReflector:
    def _health(self, **kw) -> ToolCallHealth:
        return ToolCallHealth(**kw)

    def test_none_health_returns_empty(self) -> None:
        assert reflect(None) == []

    def test_healthy_returns_empty(self) -> None:
        assert reflect(self._health(success_rate=1.0, total_calls=10)) == []

    def test_all_fail_returns_switch_tool_and_model(self) -> None:
        actions = reflect(self._health(success_rate=0.0, total_calls=5))
        types = {a.action_type for a in actions}
        assert "switch_tool" in types
        assert "switch_model" in types

    def test_zero_calls_low_sr_falls_to_low_sr_rule(self) -> None:
        # total_calls=0 → 第一规则 (total_calls>0) 不匹配;
        # 但 success_rate<0.5 仍命中第二规则 → check_config + check_params
        actions = reflect(self._health(success_rate=0.0, total_calls=0))
        assert {a.action_type for a in actions} == {"check_config", "check_params"}

    def test_zero_calls_high_sr_returns_empty(self) -> None:
        # total_calls=0 且 success_rate>=0.5 → 全部规则不命中 → 空 list
        assert reflect(self._health(success_rate=1.0, total_calls=0)) == []

    def test_low_success_rate_returns_check_config_and_params(self) -> None:
        actions = reflect(self._health(success_rate=0.3, total_calls=10))
        assert len(actions) == 2
        assert {a.action_type for a in actions} == {"check_config", "check_params"}

    def test_param_error_dominates_returns_check_params(self) -> None:
        actions = reflect(
            self._health(
                success_rate=0.6, total_calls=10,
                param_error_count=3, retry_count=1,
            )
        )
        assert len(actions) == 1
        assert actions[0].action_type == "check_params"

    def test_timeout_dominates_returns_check_config(self) -> None:
        actions = reflect(
            self._health(success_rate=0.8, total_calls=10, timeout_count=3)
        )
        assert len(actions) == 1
        assert actions[0].action_type == "check_config"

    def test_severity_ordering_all_fail_beats_low_sr(self) -> None:
        # success_rate==0 优先于 <0.5 规则
        actions = reflect(self._health(success_rate=0.0, total_calls=5))
        assert all(a.action_type.startswith("switch") for a in actions)

    def test_dict_health_duck_typing(self) -> None:
        actions = reflect({"success_rate": 0.2, "total_calls": 5})
        assert len(actions) == 2
        assert {a.action_type for a in actions} == {"check_config", "check_params"}

    def test_last_step_evaluations_and_audit_log_ignored(self) -> None:
        # 升级路径参数, 当前版本不影响规则
        assert reflect(
            self._health(success_rate=1.0, total_calls=10),
            last_step_evaluations=[1, 2, 3],
            audit_log_path=Path("/tmp/audit.jsonl"),
        ) == []

    def test_action_details_populated(self) -> None:
        actions = reflect(self._health(success_rate=0.0, total_calls=7))
        for a in actions:
            assert isinstance(a.details, dict)
            assert a.details  # 非空

    def test_format_empty_returns_empty_string(self) -> None:
        assert format_reflector_text([]) == ""

    def test_format_has_header_and_actions(self) -> None:
        text = format_reflector_text(reflect(self._health(success_rate=0.0, total_calls=5)))
        assert "[Reflector]" in text
        assert "switch_tool" in text

    def test_format_includes_details(self) -> None:
        actions = [ReflectorAction(action_type="check_config", description="d", details={"k": "v"})]
        text = format_reflector_text(actions)
        assert "k=v" in text

    def test_reflector_action_default_details_is_dict(self) -> None:
        a = ReflectorAction(action_type="check_config", description="d")
        assert a.details == {}


# ───────────────────────── step_evaluator dataclasses ─────────────────────────


class TestToolCallHealth:
    def test_default_is_not_anomalous(self) -> None:
        assert not ToolCallHealth().is_anomalous()

    def test_low_success_rate_is_anomalous(self) -> None:
        assert ToolCallHealth(success_rate=0.2).is_anomalous()

    def test_boundary_success_rate_0_3_not_anomalous(self) -> None:
        # < 0.3 才异常, 0.3 本身不算
        assert not ToolCallHealth(success_rate=0.3).is_anomalous()

    def test_timeout_makes_anomalous(self) -> None:
        assert ToolCallHealth(success_rate=1.0, timeout_count=1).is_anomalous()

    def test_param_error_makes_anomalous(self) -> None:
        assert ToolCallHealth(success_rate=1.0, param_error_count=1).is_anomalous()


class TestMeasurementUncertainty:
    def test_empty_has_no_uncertainty(self) -> None:
        mu = MeasurementUncertainty()
        assert not mu.has_uncertainty()
        assert not mu.is_point_only()

    def test_point_only(self) -> None:
        mu = MeasurementUncertainty(point_estimate=2.6)
        assert mu.is_point_only()
        assert not mu.has_uncertainty()

    def test_has_uncertainty(self) -> None:
        mu = MeasurementUncertainty(point_estimate=2.6, uncertainty=0.3)
        assert mu.has_uncertainty()
        assert not mu.is_point_only()

    def test_relative_uncertainty(self) -> None:
        mu = MeasurementUncertainty(point_estimate=2.0, uncertainty=0.5)
        assert mu.relative_uncertainty() == 0.25

    def test_relative_uncertainty_zero_point(self) -> None:
        mu = MeasurementUncertainty(point_estimate=0.0, uncertainty=0.5)
        assert mu.relative_uncertainty() is None

    def test_relative_uncertainty_without_uncertainty(self) -> None:
        mu = MeasurementUncertainty(point_estimate=2.0)
        assert mu.relative_uncertainty() is None


class TestCheckUncertaintyPropagation:
    def _ev(self, mu: MeasurementUncertainty | None = None, step_id: int = 1) -> StepEvaluation:
        return StepEvaluation(
            step_id=step_id,
            attempted="a",
            found="f",
            target_chain_ref=None,
            on_track="true",
            structure_check="passed",
            evidence_quality="high",
            deviation="",
            measurement_uncertainty=mu or MeasurementUncertainty(),
        )

    def test_empty_list_returns_empty(self) -> None:
        assert check_uncertainty_propagation([]) == []

    def test_no_point_estimates_returns_empty(self) -> None:
        assert check_uncertainty_propagation([self._ev()]) == []

    def test_point_only_flagged(self) -> None:
        ev = self._ev(MeasurementUncertainty(point_estimate=1.0, unit="eV"), step_id=3)
        issues = check_uncertainty_propagation([ev])
        assert any(i["issue"] == "point_only" and i["step_id"] == 3 for i in issues)

    def test_high_relative_flagged(self) -> None:
        ev = self._ev(MeasurementUncertainty(point_estimate=1.0, uncertainty=0.5, unit="eV"))
        issues = check_uncertainty_propagation([ev])
        assert any(i["issue"] == "high_relative" for i in issues)

    def test_unpropagated_chain_flagged(self) -> None:
        evs = [
            self._ev(MeasurementUncertainty(point_estimate=1.0, uncertainty=0.1, unit="eV"), sid)
            for sid in (1, 2, 3)
        ]
        issues = check_uncertainty_propagation(evs)
        assert any(i["issue"] == "unpropagated" for i in issues)

    def test_propagated_does_not_trigger_unpropagated(self) -> None:
        evs = [
            self._ev(MeasurementUncertainty(point_estimate=1.0, uncertainty=0.1, unit="eV", propagated=True), sid)
            for sid in (1, 2)
        ]
        issues = check_uncertainty_propagation(evs)
        assert not any(i["issue"] == "unpropagated" for i in issues)


# ───────────────────────── step_evaluator pure helpers ─────────────────────────


class TestStepEvaluatorHelpers:
    def test_clamp01_below_zero(self) -> None:
        assert se._clamp01(-1.5) == 0.0

    def test_clamp01_above_one(self) -> None:
        assert se._clamp01(2.5) == 1.0

    def test_clamp01_in_range(self) -> None:
        assert se._clamp01(0.5) == 0.5

    def test_compute_darwin_score_none(self) -> None:
        assert se._compute_darwin_score(None) == 0.5

    def test_compute_darwin_score_dict_with_gap_severity(self) -> None:
        # darwin = 1 - gap_severity
        assert se._compute_darwin_score({"gap_severity": 0.2}) == pytest.approx(0.8)

    def test_compute_darwin_score_dict_bad_gap_severity_falls_back(self) -> None:
        assert se._compute_darwin_score({"gap_severity": "bad", "on_track": "true"}) == pytest.approx(0.9)

    def test_compute_darwin_score_on_track_true(self) -> None:
        assert se._compute_darwin_score({"on_track": "true"}) == pytest.approx(0.9)

    def test_compute_darwin_score_on_track_false(self) -> None:
        assert se._compute_darwin_score({"on_track": "false"}) == pytest.approx(0.1)

    def test_compute_darwin_score_on_track_unsure(self) -> None:
        assert se._compute_darwin_score({"on_track": "unsure"}) == pytest.approx(0.5)

    def test_compute_darwin_score_dataclass(self) -> None:
        ev = StepEvaluation(
            step_id=1, attempted="a", found="f", target_chain_ref=None,
            on_track="true", structure_check="passed",
            evidence_quality="high", deviation="",
        )
        assert se._compute_darwin_score(ev) == pytest.approx(0.9)

    def test_extract_keywords_empty(self) -> None:
        assert se._extract_keywords("") == []

    def test_extract_keywords_strips_stopwords_and_single_chars(self) -> None:
        kws = se._extract_keywords("the a band structure of silicon")
        assert "band" in kws and "structure" in kws and "silicon" in kws
        assert "the" not in kws and "a" not in kws

    def test_extract_keywords_chinese_blocks(self) -> None:
        kws = se._extract_keywords("能带 结构 硅")
        assert "能带" in kws and "结构" in kws and "硅" not in kws  # 单字符被过滤

    def test_check_target_chain_match_empty(self) -> None:
        assert se._check_target_chain_match("", []) == (None, "")

    def test_check_target_chain_match_hit(self) -> None:
        chains = [{"target_id": "t1", "required_results": ["band gap value"]}]
        tid, matched = se._check_target_chain_match("found the band gap value is 1.1", chains)
        assert tid == "t1"
        assert "band gap value" in matched

    def test_check_target_chain_match_no_hit(self) -> None:
        chains = [{"target_id": "t1", "required_results": ["completely unrelated phrase"]}]
        assert se._check_target_chain_match("found the band gap", chains) == (None, "")

    def test_check_structure_no_signals(self) -> None:
        assert se._check_structure(None, "found") == "not_applicable"

    def test_check_structure_explicit_verified_pass(self) -> None:
        sigs = [{"verifiable_via": "dimensional", "verified": True}]
        assert se._check_structure(sigs, "anything") == "passed"

    def test_check_structure_hard_fail(self) -> None:
        sigs = [{"verifiable_via": "dimensional", "verified": False}]
        assert se._check_structure(sigs, "anything") == "failed"

    def test_check_structure_soft_fail(self) -> None:
        sigs = [{"verifiable_via": "symmetry_argument", "verified": False}]
        assert se._check_structure(sigs, "anything") == "soft_warning"

    def test_check_structure_heuristic_dimensional_match(self) -> None:
        sigs = [{"verifiable_via": "dimensional"}]
        assert se._check_structure(sigs, "energy is 2.5 eV") == "passed"

    def test_check_structure_unknown_via_not_applicable(self) -> None:
        sigs = [{"verifiable_via": "weird_unknown"}]
        assert se._check_structure(sigs, "anything") == "not_applicable"


class TestParseLlmJson:
    def test_empty(self) -> None:
        on_track, eq, dev = se._parse_llm_json("")
        assert on_track == "unsure"
        assert eq == "unknown"

    def test_no_json(self) -> None:
        on_track, _, _ = se._parse_llm_json("no json here")
        assert on_track == "unsure"

    def test_valid_json(self) -> None:
        text = '{"on_track": "true", "evidence_quality": "high", "deviation": "off by 1"}'
        on_track, eq, dev = se._parse_llm_json(text)
        assert on_track == "true"
        assert eq == "high"
        assert dev == "off by 1"

    def test_invalid_on_track_falls_back(self) -> None:
        text = '{"on_track": "maybe", "evidence_quality": "high"}'
        on_track, _, _ = se._parse_llm_json(text)
        assert on_track == "unsure"

    def test_invalid_evidence_quality_falls_back(self) -> None:
        text = '{"on_track": "true", "evidence_quality": "ultra"}'
        _, eq, _ = se._parse_llm_json(text)
        assert eq == "unknown"

    def test_json_with_surrounding_text(self) -> None:
        text = 'Here is my eval: {"on_track": "false", "evidence_quality": "low", "deviation": "x"} done'
        on_track, eq, dev = se._parse_llm_json(text)
        assert on_track == "false"
        assert eq == "low"
        assert dev == "x"


class TestParseDarwinJson:
    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            se._parse_darwin_json("")

    def test_no_array_raises(self) -> None:
        with pytest.raises(ValueError):
            se._parse_darwin_json("not json at all")

    def test_valid_array(self) -> None:
        text = json.dumps([
            {"simplex_id": "s1", "darwin_score": 0.8, "reason": "good"},
            {"simplex_id": "s2", "darwin_score": 0.2, "reason": "bad"},
        ])
        result = se._parse_darwin_json(text)
        assert len(result) == 2
        assert result[0]["darwin_score"] == 0.8

    def test_skips_invalid_entries(self) -> None:
        text = json.dumps([
            {"simplex_id": "s1", "darwin_score": 0.8, "reason": "good"},
            {"simplex_id": 123, "darwin_score": 0.5, "reason": "bad sid type"},
            {"simplex_id": "s3", "darwin_score": 1.5, "reason": "out of range"},
            {"simplex_id": "s4", "darwin_score": True, "reason": "bool not int"},
        ])
        result = se._parse_darwin_json(text)
        assert len(result) == 1
        assert result[0]["simplex_id"] == "s1"

    def test_not_a_list_raises(self) -> None:
        with pytest.raises(ValueError):
            se._parse_darwin_json('{"not": "a list"}')

    def test_array_in_text(self) -> None:
        text = 'prefix [{"simplex_id": "s1", "darwin_score": 0.5, "reason": "ok"}] suffix'
        result = se._parse_darwin_json(text)
        assert len(result) == 1


class TestBuildDarwinEvalPrompt:
    def test_returns_tuple_of_two(self) -> None:
        sys_text, user_text = se._build_darwin_eval_prompt([], "task")
        assert isinstance(sys_text, str) and isinstance(user_text, str)
        assert "task" in user_text

    def test_includes_entries(self) -> None:
        entries = [
            {"simplex_id": "s1", "attempted": "do x", "found": "got y", "evidence": "z"},
        ]
        _, user_text = se._build_darwin_eval_prompt(entries, "task")
        assert "s1" in user_text
        assert "do x" in user_text


class TestComputeToolCallHealth:
    def test_none_path(self) -> None:
        assert compute_tool_call_health(None, step_id=1) is None

    def test_nonexistent_path(self, tmp_path: Path) -> None:
        assert compute_tool_call_health(tmp_path / "nope.jsonl", step_id=1) is None

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        p.write_text("")
        assert compute_tool_call_health(p, step_id=1) is None

    def test_no_tool_events(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        p.write_text(json.dumps({"type": "other", "data": {}}) + "\n")
        assert compute_tool_call_health(p, step_id=1) is None

    def test_counts_calls_and_results(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        lines = [
            {"type": "tool.call", "data": {"tool": "x", "input": "a"}},
            {"type": "tool.result", "data": {"tool": "x"}},
            {"type": "tool.call", "data": {"tool": "y", "input": "b"}},
            {"type": "tool.error", "data": {"tool": "y", "error": "timeout"}},
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        h = compute_tool_call_health(p, step_id=1)
        assert h is not None
        assert h.total_calls == 2
        assert h.success_rate == 0.5
        assert h.timeout_count == 1

    def test_retry_detection(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        lines = [
            {"type": "tool.call", "data": {"tool": "x", "input": "a"}},
            {"type": "tool.call", "data": {"tool": "x", "input": "a"}},  # retry
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        h = compute_tool_call_health(p, step_id=1)
        assert h is not None
        assert h.retry_count == 1

    def test_param_error_detection(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        lines = [
            {"type": "tool.call", "data": {"tool": "x", "input": "a"}},
            {"type": "tool.error", "data": {"tool": "x", "error": "TypeError: bad param"}},
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        h = compute_tool_call_health(p, step_id=1)
        assert h is not None
        assert h.param_error_count == 1

    def test_step_id_filter(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        lines = [
            {"type": "tool.call", "data": {"tool": "x"}, "step_id": 1},
            {"type": "tool.call", "data": {"tool": "y"}, "step_id": 2},
        ]
        p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
        h = compute_tool_call_health(p, step_id=1)
        assert h is not None
        assert h.total_calls == 1

    def test_malformed_lines_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "audit.jsonl"
        p.write_text(
            "not json\n"
            + json.dumps({"type": "tool.call", "data": {"tool": "x"}}) + "\n"
        )
        h = compute_tool_call_health(p, step_id=1)
        assert h is not None
        assert h.total_calls == 1


# ───────────────────────── llm_likelihood ─────────────────────────


class TestLlmLikelihoodEnv:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HUGINN_LLM_LIKELIHOOD", raising=False)
        monkeypatch.delenv("HUGINN_DARWIN_LLM_EVAL", raising=False)
        assert not ll.is_llm_likelihood_enabled()

    @pytest.mark.parametrize("val", ["1", "true", "yes", "TRUE", "Yes"])
    def test_enabled_via_llm_likelihood(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        monkeypatch.delenv("HUGINN_DARWIN_LLM_EVAL", raising=False)
        monkeypatch.setenv("HUGINN_LLM_LIKELIHOOD", val)
        assert ll.is_llm_likelihood_enabled()

    @pytest.mark.parametrize("val", ["0", "false", "no", ""])
    def test_disabled_via_llm_likelihood(
        self, monkeypatch: pytest.MonkeyPatch, val: str
    ) -> None:
        monkeypatch.delenv("HUGINN_DARWIN_LLM_EVAL", raising=False)
        monkeypatch.setenv("HUGINN_LLM_LIKELIHOOD", val)
        assert not ll.is_llm_likelihood_enabled()

    def test_legacy_darwin_env_enables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HUGINN_LLM_LIKELIHOOD", raising=False)
        monkeypatch.setenv("HUGINN_DARWIN_LLM_EVAL", "1")
        assert ll.is_llm_likelihood_enabled()

    def test_interval_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HUGINN_LLM_LIKELIHOOD_INTERVAL", raising=False)
        assert ll.get_llm_likelihood_interval() == 5

    def test_interval_custom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HUGINN_LLM_LIKELIHOOD_INTERVAL", "10")
        assert ll.get_llm_likelihood_interval() == 10

    def test_interval_clamped_to_min_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HUGINN_LLM_LIKELIHOOD_INTERVAL", "0")
        assert ll.get_llm_likelihood_interval() == 1

    def test_interval_bad_value_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HUGINN_LLM_LIKELIHOOD_INTERVAL", "not-a-number")
        assert ll.get_llm_likelihood_interval() == 5


class TestParseLogLikJson:
    def test_empty(self) -> None:
        val, reason = ll._parse_log_lik_json("")
        assert val is None
        assert "empty" in reason

    def test_no_json(self) -> None:
        val, reason = ll._parse_log_lik_json("no json here")
        assert val is None
        assert "no JSON" in reason

    def test_valid(self) -> None:
        val, reason = ll._parse_log_lik_json('{"log_lik": -2.5, "reason": "close fit"}')
        assert val == -2.5
        assert reason == "close fit"

    def test_json_with_surrounding_text(self) -> None:
        text = 'prefix {"log_lik": -1.0, "reason": "ok"} suffix'
        val, _ = ll._parse_log_lik_json(text)
        assert val == -1.0

    def test_bool_rejected(self) -> None:
        # bool 是 int 子类, 必须排除
        val, reason = ll._parse_log_lik_json('{"log_lik": true, "reason": "x"}')
        assert val is None
        assert "numeric" in reason

    def test_string_rejected(self) -> None:
        val, reason = ll._parse_log_lik_json('{"log_lik": "-2.5", "reason": "x"}')
        assert val is None
        assert "numeric" in reason

    def test_out_of_range_low(self) -> None:
        val, reason = ll._parse_log_lik_json('{"log_lik": -100.0, "reason": "x"}')
        assert val is None
        assert "range" in reason

    def test_out_of_range_high(self) -> None:
        val, reason = ll._parse_log_lik_json('{"log_lik": 5.0, "reason": "x"}')
        assert val is None
        assert "range" in reason

    def test_reason_truncated(self) -> None:
        long_reason = "x" * 500
        _, reason = ll._parse_log_lik_json(
            f'{{"log_lik": -1.0, "reason": "{long_reason}"}}'
        )
        assert len(reason) <= 200

    def test_missing_reason_defaults_empty(self) -> None:
        val, reason = ll._parse_log_lik_json('{"log_lik": -1.0}')
        assert val == -1.0
        assert reason == ""

    def test_zero_is_valid(self) -> None:
        val, _ = ll._parse_log_lik_json('{"log_lik": 0, "reason": "perfect"}')
        assert val == 0.0


class TestBuildLlmLikPrompt:
    def test_prompt_contains_hypothesis_and_observation(self) -> None:
        h = hm.Hypothesis(h_id="h1", description="band gap model", predictions={"x": 1.0})
        o = hm.Observation(name="x", value=1.1, sigma=0.1)
        sys_text, usr = ll._build_llm_lik_prompt(h, o)
        assert "band gap model" in usr
        assert "1.1" in usr
        assert "log_lik" in sys_text

    def test_prompt_includes_task_context(self) -> None:
        h = hm.Hypothesis(h_id="h1", description="d", predictions={"x": 1.0})
        o = hm.Observation(name="x", value=1.1)
        _, usr = ll._build_llm_lik_prompt(h, o, task_ctx="extra context here")
        assert "extra context here" in usr

    def test_prompt_truncates_long_context(self) -> None:
        h = hm.Hypothesis(h_id="h1", description="d", predictions={"x": 1.0})
        o = hm.Observation(name="x", value=1.1)
        long_ctx = "Z" * 2000
        _, usr = ll._build_llm_lik_prompt(h, o, task_ctx=long_ctx)
        # task_ctx 截到 800 字符
        assert "Z" * 800 in usr
        assert "Z" * 801 not in usr


# ───────────────────────── hypothesis_manifold ─────────────────────────


class TestLogsumexp:
    def test_empty_returns_neg_inf(self) -> None:
        assert hm._logsumexp([]) == float("-inf")

    def test_all_neg_inf(self) -> None:
        assert hm._logsumexp([float("-inf"), float("-inf")]) == float("-inf")

    def test_single_value(self) -> None:
        assert hm._logsumexp([0.0]) == 0.0

    def test_numerical_stability(self) -> None:
        # 大数值不会 overflow
        result = hm._logsumexp([1000.0, 1001.0])
        assert math.isfinite(result)
        assert result == pytest.approx(1001.0 + math.log(math.exp(0.0) + math.exp(-1.0)))


class TestMdlLogprior:
    def test_empty_description_no_params_neg_inf(self) -> None:
        assert hm._mdl_logprior("", 0) == float("-inf")

    def test_description_only_penalty(self) -> None:
        # 越长描述, prior 越小 (更复杂)
        short = hm._mdl_logprior("ab", 0)
        long = hm._mdl_logprior("abcdefghij", 0)
        assert long < short

    def test_params_add_penalty(self) -> None:
        no_params = hm._mdl_logprior("desc", 0)
        with_params = hm._mdl_logprior("desc", 5)
        assert with_params < no_params

    def test_returns_negative(self) -> None:
        assert hm._mdl_logprior("some description", 2) < 0


class TestGramSchmidtQr:
    def test_empty(self) -> None:
        assert hm._gram_schmidt_qr([]) == ([], [])
        assert hm._gram_schmidt_qr([[]]) == ([], [])

    def test_identity_matrix(self) -> None:
        # 2x2 单位矩阵, Q 应保持正交, R 应是单位矩阵
        vecs = [[1.0, 0.0], [0.0, 1.0]]
        Q, R = hm._gram_schmidt_qr(vecs)
        assert R[0][0] == pytest.approx(1.0)
        assert R[1][1] == pytest.approx(1.0)

    def test_orthogonalization(self) -> None:
        # 非正交输入, R 上三角应能正交化
        vecs = [[1.0, 1.0], [1.0, -1.0]]
        Q, R = hm._gram_schmidt_qr(vecs)
        # R 应是上三角
        assert R[0][1] != 0.0 or R[1][0] == 0.0

    def test_rank_deficient_column_skipped(self) -> None:
        # 第二列是第一列的倍数 → 秩亏, R[1][1] 应为 0
        vecs = [[1.0, 2.0], [1.0, 2.0]]
        _, R = hm._gram_schmidt_qr(vecs)
        assert R[1][1] == 0.0


class TestMatvec:
    def test_empty(self) -> None:
        assert hm._matvec([], []) == []
        assert hm._matvec([[1.0]], []) == []

    def test_upper_triangular_multiply(self) -> None:
        R = [[2.0, 3.0], [0.0, 4.0]]
        v = [1.0, 1.0]
        out = hm._matvec(R, v)
        # out[0] = R[0][0]*v[0] = 2, out[1] = R[0][1]*v[0] + R[1][1]*v[1] = 3 + 4 = 7
        assert out == [2.0, 7.0]


class TestHypothesis:
    def test_log_prior_uses_mdl_by_default(self) -> None:
        h = hm.Hypothesis(h_id="h1", description="desc", n_params=2)
        assert h.log_prior() == hm._mdl_logprior("desc", 2)

    def test_log_prior_override(self) -> None:
        h = hm.Hypothesis(h_id="h1", description="desc", prior_override=0.5)
        assert h.log_prior() == math.log(0.5)

    def test_default_predictions_empty(self) -> None:
        h = hm.Hypothesis(h_id="h1", description="d")
        assert h.predictions == {}
        assert h.n_params == 0


class TestObservation:
    def test_default_sigma(self) -> None:
        o = hm.Observation(name="x", value=1.0)
        assert o.sigma == 1.0


class TestHypothesisManifold:
    def _manifold(self) -> hm.HypothesisManifold:
        m = hm.HypothesisManifold()
        m.add(hm.Hypothesis(h_id="h1", description="model A", predictions={"x": 1.0}))
        m.add(hm.Hypothesis(h_id="h2", description="model B", predictions={"x": 2.0}))
        return m

    def test_add_duplicate_raises(self) -> None:
        m = hm.HypothesisManifold()
        m.add(hm.Hypothesis(h_id="h1", description="d"))
        with pytest.raises(ValueError):
            m.add(hm.Hypothesis(h_id="h1", description="d2"))

    def test_register_structure_unknown_hypothesis_raises(self) -> None:
        m = hm.HypothesisManifold()
        with pytest.raises(KeyError):
            m.register_structure("nope", "s1", object())

    def test_register_haptic_unknown_hypothesis_raises(self) -> None:
        m = hm.HypothesisManifold()
        with pytest.raises(KeyError):
            m.register_haptic("nope", object())

    def test_log_posterior_returns_all_hypotheses(self) -> None:
        m = self._manifold()
        post = m.log_posterior([hm.Observation("x", 1.1)])
        assert set(post.keys()) == {"h1", "h2"}

    def test_posterior_normalizes_to_one(self) -> None:
        m = self._manifold()
        post = m.posterior([hm.Observation("x", 1.1)])
        assert sum(post.values()) == pytest.approx(1.0)

    def test_posterior_favors_closer_prediction(self) -> None:
        m = self._manifold()
        # obs=1.1 离 h1(pred=1.0) 更近, h1 posterior 应更高
        post = m.posterior([hm.Observation("x", 1.1, sigma=0.1)])
        assert post["h1"] > post["h2"]

    def test_abductive_inference_picks_best(self) -> None:
        m = self._manifold()
        best = m.abductive_inference([hm.Observation("x", 1.1)])
        assert best is not None
        assert best.h_id == "h1"

    def test_abductive_inference_empty_returns_none(self) -> None:
        m = hm.HypothesisManifold()
        assert m.abductive_inference([]) is None

    def test_gaussian_log_likelihood_missing_prediction(self) -> None:
        h = hm.Hypothesis(h_id="h", description="d", predictions={})
        o = hm.Observation("x", 1.0)
        # 缺预测走 uniform fallback: -0.5 * log(2)
        assert hm.HypothesisManifold._gaussian_log_likelihood(h, o) == pytest.approx(-0.5 * math.log(2.0))

    def test_gaussian_log_likelihood_exact_match(self) -> None:
        h = hm.Hypothesis(h_id="h", description="d", predictions={"x": 1.0})
        o = hm.Observation("x", 1.0, sigma=1.0)
        assert hm.HypothesisManifold._gaussian_log_likelihood(h, o) == pytest.approx(0.0)

    def test_custom_likelihood_injection(self) -> None:
        calls: list[tuple[str, str]] = []

        def custom_log_lik(h: hm.Hypothesis, o: hm.Observation) -> float:
            calls.append((h.h_id, o.name))
            return 0.0

        m = hm.HypothesisManifold(likelihood_log=custom_log_lik)
        m.add(hm.Hypothesis(h_id="h1", description="d", predictions={"x": 1.0}))
        m.log_posterior([hm.Observation("x", 1.0)])
        assert calls == [("h1", "x")]


# ───────────────────────── target_chain ─────────────────────────


class TestTargetChain:
    def test_is_complete_empty_results(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=[],
            required_methods=[], required_data=[], verification="v",
        )
        assert chain.is_complete()

    def test_is_complete_all_done(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["a", "b"],
            required_methods=[], required_data=[], verification="v",
            completed_results={"a", "b"},
        )
        assert chain.is_complete()

    def test_is_complete_partial(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["a", "b"],
            required_methods=[], required_data=[], verification="v",
            completed_results={"a"},
        )
        assert not chain.is_complete()

    def test_default_progress_zero(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["a"],
            required_methods=[], required_data=[], verification="v",
        )
        assert chain.progress == 0.0
        assert chain.completed_results == set()


class TestBuildTargetChains:
    def test_skips_non_mode_a(self) -> None:
        checklist = [
            {"item": "reproduce X", "mode": "A"},
            {"item": "understand Y", "mode": "B"},
            {"item": "reproduce Z", "mode": "a"},  # 小写也接受
        ]
        chains = tc.build_target_chains(checklist, kb=None, model=None)
        assert len(chains) == 2
        assert chains[0].target_id == "target_1"
        assert chains[1].target_id == "target_2"

    def test_skips_empty_target(self) -> None:
        checklist = [
            {"item": "", "mode": "A"},
            {"item": "valid", "mode": "A"},
        ]
        chains = tc.build_target_chains(checklist, kb=None, model=None)
        assert len(chains) == 1

    def test_fallback_when_no_model(self) -> None:
        # model=None → 降级路径: required_results=[target], verification="目视检查"
        checklist = [{"item": "reproduce the band gap", "mode": "A"}]
        chains = tc.build_target_chains(checklist, kb=None, model=None)
        assert len(chains) == 1
        chain = chains[0]
        assert chain.target == "reproduce the band gap"
        # 降级: required_results 就是 target 本身
        assert chain.required_results == ["reproduce the band gap"]

    def test_empty_checklist(self) -> None:
        assert tc.build_target_chains([], kb=None, model=None) == []


class TestUpdateProgress:
    def test_empty_required_results(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=[],
            required_methods=[], required_data=[], verification="v",
        )
        assert tc.update_progress(chain, "anything") == 1.0

    def test_empty_found_no_change(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["a"],
            required_methods=[], required_data=[], verification="v",
        )
        assert tc.update_progress(chain, "") == 0.0

    def test_match_adds_completed(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["band gap value"],
            required_methods=[], required_data=[], verification="v",
        )
        prog = tc.update_progress(chain, "found the band gap value is 1.1")
        assert prog == 1.0
        assert "band gap value" in chain.completed_results

    def test_no_match_no_change(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["unrelated"],
            required_methods=[], required_data=[], verification="v",
        )
        assert tc.update_progress(chain, "found the band gap") == 0.0

    def test_partial_progress(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["a", "b"],
            required_methods=[], required_data=[], verification="v",
        )
        prog = tc.update_progress(chain, "completed a")
        assert prog == 0.5


class TestDetectDrift:
    def test_empty_evaluations(self) -> None:
        assert tc.detect_drift([]) == (False, "")

    def test_insufficient_evaluations(self) -> None:
        assert tc.detect_drift([{"on_track": False}], window=3) == (False, "")

    def test_drift_detected(self) -> None:
        evals = [{"on_track": False}, {"on_track": False}, {"on_track": False}]
        is_drift, msg = tc.detect_drift(evals, window=3)
        assert is_drift is True
        assert "连续" in msg

    def test_no_drift_when_on_track(self) -> None:
        evals = [{"on_track": True}, {"on_track": True}, {"on_track": True}]
        assert tc.detect_drift(evals, window=3) == (False, "")

    def test_mixed_no_drift(self) -> None:
        evals = [{"on_track": False}, {"on_track": True}, {"on_track": False}]
        assert tc.detect_drift(evals, window=3) == (False, "")

    def test_dataclass_form(self) -> None:
        from huginn.metacog.step_evaluator import StepEvaluation
        evals = [
            StepEvaluation(step_id=1, attempted="a", found="f", target_chain_ref=None,
                           on_track="false", structure_check="failed",
                           evidence_quality="low", deviation=""),
            StepEvaluation(step_id=2, attempted="a", found="f", target_chain_ref=None,
                           on_track="false", structure_check="failed",
                           evidence_quality="low", deviation=""),
            StepEvaluation(step_id=3, attempted="a", found="f", target_chain_ref=None,
                           on_track="false", structure_check="failed",
                           evidence_quality="low", deviation=""),
        ]
        is_drift, _ = tc.detect_drift(evals, window=3)
        assert is_drift is True

    def test_unsure_not_counted_as_off_track(self) -> None:
        from huginn.metacog.step_evaluator import StepEvaluation
        evals = [
            StepEvaluation(step_id=1, attempted="a", found="f", target_chain_ref=None,
                           on_track="unsure", structure_check="not_applicable",
                           evidence_quality="unknown", deviation=""),
        ] * 3
        # unsure 不算 off-track, 不触发漂移
        assert tc.detect_drift(evals, window=3) == (False, "")


class TestFormatTargetChainText:
    def test_empty_chains(self) -> None:
        assert tc.format_target_chain_text([], current_step=1) == ""

    def test_has_header(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="reproduce X", required_results=["a"],
            required_methods=[], required_data=[], verification="v",
        )
        text = tc.format_target_chain_text([chain], current_step=5)
        assert "[TargetChain @ step 5]" in text
        assert "reproduce X" in text
        assert "t1" in text

    def test_shows_done_and_missing(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["a", "b"],
            required_methods=[], required_data=[], verification="v",
            completed_results={"a"},
        )
        text = tc.format_target_chain_text([chain], current_step=1)
        assert "done" in text
        assert "missing" in text

    def test_done_chain_shows_done_marker(self) -> None:
        chain = tc.TargetChain(
            target_id="t1", target="t", required_results=["a"],
            required_methods=[], required_data=[], verification="v",
            completed_results={"a"},
        )
        text = tc.format_target_chain_text([chain], current_step=1)
        assert "DONE" in text
