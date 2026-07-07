"""Tests for conjecture <-> knowledge graph integration.

Covers:
1. generate_conjecture writes the conjecture to KG and returns kg_node_id.
2. KG write failures do not break conjecture generation.
3. run() pulls domain context from KG before extract_pattern.

Run with: pytest --override-ini="addopts=" tests/test_conjecture_kg_integration.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from huginn.autoloop.conjecture import ConjectureGenerator


@pytest.fixture
def gen() -> ConjectureGenerator:
    # fresh generator per test, never touches the module singleton
    return ConjectureGenerator()


def _make_transfer(gen: ConjectureGenerator) -> dict:
    pattern = gen.extract_pattern(
        "doping increases conductivity in semiconductors", "semiconductors"
    )
    return gen.transfer_domain(pattern, "battery cathodes")


# ── 1. kg_node_id present on success ────────────────────────────────


class TestKgWriteBack:
    def test_generate_returns_kg_node_id(self, gen: ConjectureGenerator):
        # mock KG: add_entity returns a fake node id, add_relation/save are no-ops
        fake_kg = MagicMock()
        fake_kg.add_entity.return_value = "Fact:some-conjecture"

        transfer = _make_transfer(gen)
        with patch("huginn.autoloop.conjecture.get_kg", return_value=fake_kg):
            result = gen.generate_conjecture(transfer)

        # kg_node_id should be the id add_entity returned
        assert result["kg_node_id"] == "Fact:some-conjecture"
        # the conjecture itself is still intact
        assert result["statement"]
        assert result["confidence"] in ("low", "medium", "high")

        # first add_entity call creates the FACT node (the conjecture)
        first_call = fake_kg.add_entity.call_args_list[0]
        assert first_call.kwargs["entity_type"] == "Fact"
        # 3 entities total: conjecture + source material + target material
        assert fake_kg.add_entity.call_count == 3

        # two relations: DERIVED_FROM + APPLIES
        assert fake_kg.add_relation.call_count == 2
        relations = [c.args[1] for c in fake_kg.add_relation.call_args_list]
        assert "derived_from" in relations
        assert "applies" in relations

        # save was called to persist
        fake_kg.save.assert_called_once()

    def test_relation_direction_is_correct(self, gen: ConjectureGenerator):
        # source MATERIAL --DERIVED_FROM--> conjecture --APPLIES--> target MATERIAL
        fake_kg = MagicMock()
        fake_kg.add_entity.side_effect = [
            "Fact:conj",
            "Material:semiconductors",
            "Material:battery cathodes",
        ]

        transfer = _make_transfer(gen)
        with patch("huginn.autoloop.conjecture.get_kg", return_value=fake_kg):
            gen.generate_conjecture(transfer)

        der = fake_kg.add_relation.call_args_list[0]
        app = fake_kg.add_relation.call_args_list[1]
        # DERIVED_FROM: source_material -> conjecture
        assert der.args[0] == "Material:semiconductors"
        assert der.args[1] == "derived_from"
        assert der.args[2] == "Fact:conj"
        # APPLIES: conjecture -> target_material
        assert app.args[0] == "Fact:conj"
        assert app.args[1] == "applies"
        assert app.args[2] == "Material:battery cathodes"


# ── 2. KG failures don't break conjecture generation ─────────────────


class TestKgFailureResilience:
    def test_kg_none_does_not_break_generation(self, gen: ConjectureGenerator):
        # get_kg returns None (KG unavailable) -> kg_node_id is None, rest works
        transfer = _make_transfer(gen)
        with patch("huginn.autoloop.conjecture.get_kg", return_value=None):
            result = gen.generate_conjecture(transfer)

        assert result["kg_node_id"] is None
        assert result["statement"]
        assert result["method"] == "template"

    def test_kg_write_exception_does_not_break_generation(
        self, gen: ConjectureGenerator
    ):
        # add_entity blows up -> caught, kg_node_id is None, conjecture survives
        fake_kg = MagicMock()
        fake_kg.add_entity.side_effect = RuntimeError("disk full")

        transfer = _make_transfer(gen)
        with patch("huginn.autoloop.conjecture.get_kg", return_value=fake_kg):
            result = gen.generate_conjecture(transfer)

        assert result["kg_node_id"] is None
        assert result["statement"]
        assert result["prediction"]

    def test_kg_context_fetch_failure_does_not_break_run(
        self, gen: ConjectureGenerator
    ):
        # query raises -> domain_context falls back to None, pipeline still runs
        fake_kg = MagicMock()
        fake_kg.query.side_effect = RuntimeError("KG corrupt")
        fake_kg.add_entity.return_value = "Fact:conjecture"

        with patch("huginn.autoloop.conjecture.get_kg", return_value=fake_kg):
            result = gen.run(
                source_problem="doping increases conductivity in semiconductors",
                source_domain="semiconductors",
                target_domain="battery cathodes",
            )

        # extract_pattern ran without KG context, pipeline completed
        assert result["pattern"]["action"] == "杂质引入"
        assert result["conjecture"]["statement"]


# ── 3. run() pulls domain context from KG ─────────────────────────────


class TestRunKgContext:
    def test_run_queries_kg_for_source_domain(self, gen: ConjectureGenerator):
        fake_kg = MagicMock()
        fake_kg.query.return_value = {
            "nodes": [{"id": "Material:semiconductors", "label": "semiconductors"}],
            "edges": [],
        }
        fake_kg.to_text.return_value = "- Material:semiconductors"
        fake_kg.add_entity.return_value = "Fact:conjecture"

        with patch("huginn.autoloop.conjecture.get_kg", return_value=fake_kg):
            result = gen.run(
                source_problem="doping increases conductivity in semiconductors",
                source_domain="semiconductors",
                target_domain="battery cathodes",
            )

        # KG query was called with the source domain as seed
        fake_kg.query.assert_called_once()
        assert fake_kg.query.call_args.args[0] == "semiconductors"
        # to_text turned the subgraph into prompt-friendly text
        fake_kg.to_text.assert_called_once()
        # pipeline still produced a full result
        assert result["conjecture"]["statement"]
        assert result["conjecture"]["kg_node_id"] == "Fact:conjecture"

    def test_run_skips_kg_when_empty(self, gen: ConjectureGenerator):
        # KG returns no nodes -> domain_context is None, extract_pattern still runs
        fake_kg = MagicMock()
        fake_kg.query.return_value = {"nodes": [], "edges": []}
        fake_kg.add_entity.return_value = "Fact:conjecture"

        with patch("huginn.autoloop.conjecture.get_kg", return_value=fake_kg):
            result = gen.run(
                source_problem="doping increases conductivity in semiconductors",
                source_domain="semiconductors",
                target_domain="battery cathodes",
            )

        # to_text not called because there were no nodes
        fake_kg.to_text.assert_not_called()
        assert result["pattern"]["action"] == "杂质引入"
