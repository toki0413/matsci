"""Tests for huginn.autoloop.conjecture — the Moonshine cross-domain conjecture pipeline.

Covers the 3-step flow (extract_pattern -> transfer_domain -> generate_conjecture)
plus the run() wrapper, template fallbacks, log-id chaining, and the singleton
accessor. Everything is synchronous and hermetic — no real LLM calls.

Run with: pytest --override-ini="addopts=" tests/test_conjecture_engine.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from huginn.autoloop.conjecture import (
    ConjectureGenerator,
    get_conjecture_generator,
)


# ── fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def gen() -> ConjectureGenerator:
    """Fresh generator per test — never touches the module-level singleton."""
    return ConjectureGenerator()


@pytest.fixture
def semiconductor_pattern(gen: ConjectureGenerator) -> dict:
    """A pattern extracted from the canonical doping/conductivity example."""
    return gen.extract_pattern(
        "doping increases conductivity in semiconductors",
        "semiconductors",
    )


# ── helper: a model that looks real but always blows up ─────────────


class _ExplodingModel:
    """Looks like a real LLM to _is_real_model (no _mock_name attr) but
    raises on every invoke/ainvoke. Used to exercise the template fallback."""

    def invoke(self, messages):  # noqa: ANN001
        raise RuntimeError("simulated LLM failure")

    async def ainvoke(self, messages):  # noqa: ANN001
        raise RuntimeError("simulated LLM failure")


class _StubModel:
    """Minimal real-looking model whose ainvoke returns a canned JSON blob.
    Used when we want the LLM path to succeed without hitting any network."""

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def invoke(self, messages):  # noqa: ANN001
        return _Content(self._payload)

    async def ainvoke(self, messages):  # noqa: ANN001
        return _Content(self._payload)


class _Content:
    """Mimics a langchain response — _invoke_model just reads .content."""

    def __init__(self, text: str) -> None:
        self.content = text


# ── 1. extract_pattern (English) ────────────────────────────────────


class TestExtractPattern:
    def test_english_doping_conductivity(self, gen: ConjectureGenerator):
        result = gen.extract_pattern(
            "doping increases conductivity in semiconductors",
            "semiconductors",
        )

        assert result["action"] == "杂质引入"
        assert result["property"] == "电子输运性质"
        assert result["direction"] == "increases"
        # the assembled pattern should glue action + verb + property together
        assert "杂质引入" in result["abstract_pattern"]
        assert "电子输运性质" in result["abstract_pattern"]
        assert result["method"] == "template"
        # log_id is always present even if research_log is unavailable
        assert "log_id" in result

    def test_chinese_doping_conductivity(self, gen: ConjectureGenerator):
        # same semantics, Chinese source text — keyword tables are bilingual
        result = gen.extract_pattern("掺杂提高导电率", "semiconductors")

        assert result["action"] == "杂质引入"
        assert result["property"] == "电子输运性质"
        assert result["direction"] == "increases"

    def test_decreases_direction(self, gen: ConjectureGenerator):
        result = gen.extract_pattern("annealing reduces band gap", "semiconductors")
        assert result["direction"] == "decreases"
        assert result["action"] == "热处理"
        assert result["property"] == "电子结构"

    def test_modifies_direction(self, gen: ConjectureGenerator):
        result = gen.extract_pattern("strain alters hardness", "catalysts")
        assert result["direction"] == "modifies"

    def test_unrecognized_keywords_use_defaults(self, gen: ConjectureGenerator):
        # nothing in the keyword tables matches -> generic fallbacks kick in
        result = gen.extract_pattern("something totally unrelated", "semiconductors")
        assert result["action"] == "结构调制"
        assert result["property"] == "材料性质"
        assert result["direction"] == "modifies"

    def test_source_fields_echoed_back(self, gen: ConjectureGenerator):
        result = gen.extract_pattern("doping increases conductivity", "semiconductors")
        assert result["source_problem"] == "doping increases conductivity"
        assert result["source_domain"] == "semiconductors"

    def test_caches_last_pattern(self, gen: ConjectureGenerator):
        result = gen.extract_pattern("doping increases conductivity", "semiconductors")
        assert gen._last_pattern is result


# ── 2. transfer_domain ──────────────────────────────────────────────


class TestTransferDomain:
    def test_battery_cathodes_mapping(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        # R14: 硬编码 _DOMAIN_KNOWLEDGE 表已删, 改走 RAG recall.
        # 测试环境无 RAG 数据, target_action 降级到抽象概念本身 (跟 unknown domain 一致).
        result = gen.transfer_domain(semiconductor_pattern, "battery cathodes")

        mapping = result["domain_mapping"]
        assert mapping["source_action"] == "杂质引入"
        # RAG 无数据 -> 抽象概念本身
        assert mapping["target_action"] == "杂质引入"
        assert mapping["source_property"] == "电子输运性质"
        assert mapping["target_property"] == "电子输运性质"
        assert mapping["direction"] == "increases"
        assert result["method"] == "template"
        assert result["transferred_pattern"]

    def test_analogy_notes_mention_both_domains(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        result = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        notes = result["analogy_notes"]
        assert "semiconductors" in notes
        assert "battery cathodes" in notes

    def test_perovskites_target(self, gen: ConjectureGenerator, semiconductor_pattern: dict):
        # R14: RAG 模式下, perovskites 也降级到抽象概念
        result = gen.transfer_domain(semiconductor_pattern, "perovskites")
        assert result["domain_mapping"]["target_action"] == "杂质引入"

    def test_caches_last_transfer(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        result = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        assert gen._last_transfer is result


# ── 3. generate_conjecture ───────────────────────────────────────────


class TestGenerateConjecture:
    def test_returns_required_fields(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        transfer = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        result = gen.generate_conjecture(transfer)

        for key in ("statement", "prediction", "rationale", "confidence"):
            assert key in result, f"missing field: {key}"
            assert isinstance(result[key], str) and result[key]
        assert result["confidence"] in ("low", "medium", "high")
        assert result["method"] == "template"

    def test_statement_mentions_target_domain(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        transfer = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        result = gen.generate_conjecture(transfer)
        assert "battery cathodes" in result["statement"]

    def test_prediction_is_falsifiable(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        transfer = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        result = gen.generate_conjecture(transfer)
        # prediction should reference the target action so it's actually testable
        assert result["prediction"]
        assert "battery cathodes" in result["prediction"]

    def test_caches_last_conjecture(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        transfer = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        result = gen.generate_conjecture(transfer)
        assert gen._last_conjecture is result


# ── 4. full run() pipeline ──────────────────────────────────────────


class TestRunPipeline:
    def test_returns_all_top_level_fields(self, gen: ConjectureGenerator):
        result = gen.run(
            source_problem="doping increases conductivity in semiconductors",
            source_domain="semiconductors",
            target_domain="battery cathodes",
        )

        for key in (
            "pattern",
            "transfer",
            "conjecture",
            "log_chain",
            "method",
            "timestamp",
            "source_problem",
            "source_domain",
            "target_domain",
        ):
            assert key in result, f"run() missing key: {key}"

        assert result["method"] == "template"
        # timestamp should be ISO-ish with a Z suffix
        assert result["timestamp"].endswith("Z")

    def test_log_chain_structure(self, gen: ConjectureGenerator):
        result = gen.run(
            source_problem="doping increases conductivity in semiconductors",
            source_domain="semiconductors",
            target_domain="battery cathodes",
        )
        chain = result["log_chain"]
        assert {"pattern_id", "transfer_id", "conjecture_id"} == set(chain.keys())

    def test_run_threads_pipeline_correctly(self, gen: ConjectureGenerator):
        result = gen.run(
            source_problem="doping increases conductivity in semiconductors",
            source_domain="semiconductors",
            target_domain="battery cathodes",
        )
        # the transfer step should have received the pattern's action/property
        assert result["pattern"]["action"] == "杂质引入"
        assert result["transfer"]["domain_mapping"]["source_action"] == "杂质引入"
        # conjecture should carry the target domain through
        assert "battery cathodes" in result["conjecture"]["statement"]

    def test_run_with_chinese_input(self, gen: ConjectureGenerator):
        result = gen.run(
            source_problem="掺杂提高导电率",
            source_domain="semiconductors",
            target_domain="perovskites",
        )
        assert result["pattern"]["action"] == "杂质引入"
        assert result["conjecture"]["confidence"] in ("low", "medium", "high")


# ── 5. _is_real_model ───────────────────────────────────────────────


class TestIsRealModel:
    def test_magicmock_is_not_real(self):
        # MagicMock carries _mock_name, so it's treated as a test double
        assert ConjectureGenerator._is_real_model(MagicMock()) is False

    def test_real_object_is_real(self):
        # plain objects have no _mock_name -> treated as a real LLM
        assert ConjectureGenerator._is_real_model(object()) is True

    def test_none_is_treated_as_real_by_the_method(self):
        # NOTE: _is_real_model itself returns True for None because None has no
        # _mock_name attr. The None -> template fallback is enforced by the
        # caller's `model is not None and _is_real_model(model)` guard, not
        # inside _is_real_model. We assert the actual behaviour here and cover
        # the None -> template path separately in TestTemplateMode.
        assert ConjectureGenerator._is_real_model(None) is True

    def test_none_input_uses_template_mode(self, gen: ConjectureGenerator):
        # passing model=None must always land on template mode
        result = gen.extract_pattern("doping increases conductivity", "semiconductors")
        assert result["method"] == "template"

    def test_exploding_model_is_treated_as_real(self):
        # _ExplodingModel has no _mock_name, so _is_real_model says it's real
        assert ConjectureGenerator._is_real_model(_ExplodingModel()) is True


# ── 6. template fallback when LLM raises ────────────────────────────


class TestTemplateFallback:
    def test_extract_falls_back_on_llm_exception(self, gen: ConjectureGenerator):
        result = gen.extract_pattern(
            "doping increases conductivity in semiconductors",
            "semiconductors",
            model=_ExplodingModel(),
        )
        # LLM blew up -> template result, method downgraded
        assert result["method"] == "template"
        assert result["action"] == "杂质引入"

    def test_transfer_falls_back_on_llm_exception(self, gen: ConjectureGenerator):
        pattern = gen.extract_pattern(
            "doping increases conductivity in semiconductors", "semiconductors"
        )
        result = gen.transfer_domain(pattern, "battery cathodes", model=_ExplodingModel())
        assert result["method"] == "template"
        # R14: RAG 无数据时降级到抽象概念
        assert result["domain_mapping"]["target_action"] == "杂质引入"

    def test_generate_falls_back_on_llm_exception(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        transfer = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        result = gen.generate_conjecture(transfer, model=_ExplodingModel())
        assert result["method"] == "template"
        assert result["statement"]

    def test_run_falls_back_end_to_end(self, gen: ConjectureGenerator):
        # if the LLM fails at every step, run() should still return a complete
        # result built entirely from templates
        result = gen.run(
            source_problem="doping increases conductivity in semiconductors",
            source_domain="semiconductors",
            target_domain="battery cathodes",
            model=_ExplodingModel(),
        )
        assert result["method"] == "template"
        assert result["pattern"]["method"] == "template"
        assert result["transfer"]["method"] == "template"
        assert result["conjecture"]["method"] == "template"


# ── 7. unknown domain — graceful degradation ───────────────────────


class TestUnknownDomain:
    def test_unknown_domain_still_transfers(self, gen: ConjectureGenerator):
        pattern = gen.extract_pattern(
            "doping increases conductivity in semiconductors", "semiconductors"
        )
        # domain not in the knowledge table -> abstract terms pass through
        result = gen.transfer_domain(pattern, "totally unknown domain xyz")
        assert result["transferred_pattern"]
        # no mapping found, so target_action falls back to the abstract action
        assert result["domain_mapping"]["target_action"] == "杂质引入"
        assert result["domain_mapping"]["target_property"] == "电子输运性质"
        assert result["method"] == "template"

    def test_unknown_domain_full_run(self, gen: ConjectureGenerator):
        result = gen.run(
            source_problem="doping increases conductivity in semiconductors",
            source_domain="semiconductors",
            target_domain="quantum dot lasers maybe",
        )
        # the pipeline should never crash on an unknown domain
        assert result["conjecture"]["statement"]
        assert result["conjecture"]["prediction"]

    def test_case_insensitive_domain_lookup(self, gen: ConjectureGenerator):
        # R14: _lookup_domain 走 RAG recall, RAG 自己负责大小写/模糊匹配.
        # 测试环境无 RAG 数据, 任何 domain 都降级到抽象概念. 此测试验证不抛异常.
        pattern = gen.extract_pattern(
            "doping increases conductivity in semiconductors", "semiconductors"
        )
        result = gen.transfer_domain(pattern, "Battery Cathodes")
        assert result["domain_mapping"]["target_action"] == "杂质引入"

    def test_fuzzy_domain_match(self, gen: ConjectureGenerator):
        # R14: RAG 模式下模糊匹配由 RAG recall 负责 (query="layered battery cathodes").
        # 测试环境无 RAG 数据, 验证不抛异常 + 返回结构完整.
        pattern = gen.extract_pattern(
            "doping increases conductivity in semiconductors", "semiconductors"
        )
        result = gen.transfer_domain(pattern, "layered battery cathodes")
        assert result["domain_mapping"]["target_action"] == "杂质引入"


# ── 8. log_id chaining ──────────────────────────────────────────────


class TestLogIdChaining:
    def test_pattern_to_transfer_chain(self, gen: ConjectureGenerator):
        pattern = gen.extract_pattern(
            "doping increases conductivity in semiconductors", "semiconductors"
        )
        transfer = gen.transfer_domain(pattern, "battery cathodes")
        # transfer.parent_log_id must point back at the pattern's log_id
        assert transfer["parent_log_id"] == pattern["log_id"]

    def test_transfer_to_conjecture_chain(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        transfer = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        conjecture = gen.generate_conjecture(transfer)
        assert conjecture["parent_log_id"] == transfer["log_id"]

    def test_full_chain_via_run(self, gen: ConjectureGenerator):
        result = gen.run(
            source_problem="doping increases conductivity in semiconductors",
            source_domain="semiconductors",
            target_domain="battery cathodes",
        )
        pattern = result["pattern"]
        transfer = result["transfer"]
        conjecture = result["conjecture"]

        # OPEN_QUESTION -> BRIDGE -> CONJECTURE, parent links all the way up
        assert transfer["parent_log_id"] == pattern["log_id"]
        assert conjecture["parent_log_id"] == transfer["log_id"]

        # log_chain should mirror the same ids
        chain = result["log_chain"]
        assert chain["pattern_id"] == pattern["log_id"]
        assert chain["transfer_id"] == transfer["log_id"]
        assert chain["conjecture_id"] == conjecture["log_id"]


# ── 9. singleton accessor ───────────────────────────────────────────


class TestSingletonAccessor:
    def test_get_conjecture_generator_returns_instance(self):
        instance = get_conjecture_generator()
        assert isinstance(instance, ConjectureGenerator)

    def test_singleton_returns_same_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # reset the singleton so we observe the lazy init in isolation
        import huginn.autoloop.conjecture as mod

        monkeypatch.setattr(mod, "_conjecture_generator_singleton", None)
        first = get_conjecture_generator()
        second = get_conjecture_generator()
        assert first is second

    def test_singleton_is_thread_safe_lazy_init(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # after first call the singleton should be cached, second call must
        # not construct a fresh instance
        import huginn.autoloop.conjecture as mod

        monkeypatch.setattr(mod, "_conjecture_generator_singleton", None)
        first = get_conjecture_generator()
        # the cached singleton is now set; calling again returns the same obj
        assert mod._conjecture_generator_singleton is first
        assert get_conjecture_generator() is first

    def test_module_exports(self):
        # __all__ should expose both the class and the accessor
        import huginn.autoloop.conjecture as mod

        assert set(mod.__all__) == {"ConjectureGenerator", "get_conjecture_generator"}


# ── 10. LLM path with a working stub model ───────────────────────────


class TestLlmPath:
    def test_llm_extract_used_when_model_returns_json(self, gen: ConjectureGenerator):
        # a stub model that returns valid JSON should switch method to "llm"
        payload = (
            '{"abstract_pattern": "X 提升 Y", "action": "杂质引入", '
            '"property": "电子输运性质", "direction": "increases", '
            '"mechanism": "stubbed"}'
        )
        result = gen.extract_pattern(
            "doping increases conductivity", "semiconductors", model=_StubModel(payload)
        )
        assert result["method"] == "llm"
        assert result["action"] == "杂质引入"

    def test_llm_generate_with_partial_json_falls_back_fields(
        self, gen: ConjectureGenerator, semiconductor_pattern: dict
    ):
        # LLM returns an empty object -> every field falls back to template
        transfer = gen.transfer_domain(semiconductor_pattern, "battery cathodes")
        result = gen.generate_conjecture(transfer, model=_StubModel("{}"))
        # method is still llm (the call succeeded), but the content is templated
        assert result["method"] == "llm"
        assert result["statement"]
        assert result["confidence"] == "medium"
