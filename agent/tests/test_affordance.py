"""Tests for affordance-based planning."""
import pytest


class TestAffordance:
    def test_method_exists(self):
        from huginn.provenance.pipeline import SimulationPipeline
        assert hasattr(SimulationPipeline, "affordance")

    def test_returns_list_of_suggestions(self):
        from huginn.provenance.pipeline import SimulationPipeline, PipelineSuggestion
        pipeline = SimulationPipeline()
        result = pipeline.affordance()
        assert isinstance(result, list)
        for s in result:
            assert isinstance(s, PipelineSuggestion)

    def test_empty_provenance_returns_starting_points(self):
        from huginn.provenance.pipeline import SimulationPipeline
        pipeline = SimulationPipeline()
        # With no provenance entries, should return structure/cheminfo entry points
        result = pipeline.affordance()
        # Should have at least 2 starting options
        assert len(result) >= 2

    def test_no_duplicates(self):
        from huginn.provenance.pipeline import SimulationPipeline
        pipeline = SimulationPipeline()
        result = pipeline.affordance()
        # No exact duplicate suggestions (same stage + same hint + same reason)
        keys = [(s.stage, s.tool_hint, s.reason) for s in result]
        assert len(keys) == len(set(keys))

    def test_affordance_superset_of_suggest_next(self):
        """affordance() should be a superset of suggest_next() when
        there are provenance entries, because it considers ALL completed
        stages, not just the last tool."""
        from huginn.provenance.pipeline import SimulationPipeline
        from huginn.provenance.registry import ProvenanceRegistry
        # Register a structure_tool entry to simulate progress
        reg = ProvenanceRegistry.shared()
        reg.register(
            file_path="/tmp/test_affordance_POSCAR",
            produced_by="structure_tool",
            file_format="poscar",
        )
        pipeline = SimulationPipeline()
        suggestions = pipeline.suggest_next("structure_tool", {}, {})
        affordances = pipeline.affordance()
        # affordance should have at least as many options
        assert len(affordances) >= len(suggestions)
