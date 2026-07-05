"""Tests for the project-level knowledge graph."""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.agent import HuginnAgent
from huginn.kg import (
    ProjectKnowledgeGraph,
    build_from_seeds,
    build_from_session_text,
)
from huginn.kg.entities import EntityType, Relation, node_id
from huginn.kg.extractor import (
    extract_entities,
    extract_error_pattern,
    extract_materials,
)
from huginn.kg.query import GraphQuery
from huginn.utils.prompt_cache import PromptCacheBuilder


class TestProjectKnowledgeGraph:
    def test_adds_entity_and_returns_stable_id(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        eid = kg.add_entity("VASP", EntityType.TOOL, confidence=0.9)
        assert eid == node_id("VASP", EntityType.TOOL)
        assert kg.get_entity("VASP", EntityType.TOOL)["confidence"] == pytest.approx(
            0.9
        )

    def test_entity_confidence_increases_on_re_add(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        kg.add_entity("DFT", EntityType.METHOD, confidence=0.5)
        kg.add_entity("DFT", EntityType.METHOD, confidence=0.5)
        data = kg.get_entity("DFT", EntityType.METHOD)
        assert data["mentions"] == 2
        assert data["confidence"] == pytest.approx(0.55)

    def test_add_relation_requires_existing_nodes(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        a = kg.add_entity("A", EntityType.METHOD)
        b = kg.add_entity("B", EntityType.MATERIAL)
        kg.add_relation(a, Relation.APPLIES, b, confidence=0.8)
        assert kg._graph.has_edge(a, b)
        # Relation to missing node is a no-op.
        kg.add_relation(a, Relation.APPLIES, "Missing:Node", confidence=0.8)
        assert kg._graph.number_of_edges() == 1

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        a = kg.add_entity("Si", EntityType.MATERIAL)
        b = kg.add_entity("VASP", EntityType.TOOL)
        kg.add_relation(a, Relation.SOLVED_BY, b)
        kg.save()

        kg2 = ProjectKnowledgeGraph(tmp_path)
        assert kg2._graph.number_of_nodes() == 2
        assert kg2._graph.number_of_edges() == 1
        assert kg2.has_entity("Si", EntityType.MATERIAL)

    def test_stats_and_export(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        kg.add_entity("Si", EntityType.MATERIAL)
        kg.add_entity("VASP", EntityType.TOOL)
        kg.add_entity("DFT", EntityType.METHOD)
        stats = kg.stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 0
        assert set(stats["node_types"].keys()) == {"Material", "Tool", "Method"}

        exported = kg.export(fmt="json")
        assert "nodes" in exported and ("links" in exported or "edges" in exported)
        gml = kg.export(fmt="gml")
        assert "graph" in gml or "node" in gml

    def test_to_text_includes_outgoing_edges(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        a = kg.add_entity("VASP", EntityType.TOOL)
        b = kg.add_entity("Si", EntityType.MATERIAL)
        kg.add_relation(a, Relation.APPLIES, b)
        text = kg.to_text({a, b})
        assert "Tool:VASP" in text
        assert "Material:Si" in text
        assert "applies" in text


class TestGraphQuery:
    def test_find_nodes_prioritizes_exact_match(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        kg.add_entity("VASP", EntityType.TOOL, confidence=0.5)
        kg.add_entity("VASPkit", EntityType.TOOL, confidence=0.9)
        q = GraphQuery(kg._graph)
        hits = q.find_nodes("VASP")
        assert hits[0] == node_id("VASP", EntityType.TOOL)

    def test_neighborhood_honors_depth(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        a = kg.add_entity("A", EntityType.METHOD)
        b = kg.add_entity("B", EntityType.MATERIAL)
        c = kg.add_entity("C", EntityType.TOOL)
        kg.add_relation(a, Relation.APPLIES, b)
        kg.add_relation(b, Relation.SOLVED_BY, c)
        q = GraphQuery(kg._graph)

        result = q.neighborhood([a], depth=1, top_k=10)
        assert len(result["nodes"]) == 2

        result = q.neighborhood([a], depth=2, top_k=10)
        assert len(result["nodes"]) == 3

    def test_query_returns_empty_for_unknown_seed(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path)
        q = GraphQuery(kg._graph)
        result = q.query("nonexistent", depth=1, top_k=5)
        assert result["nodes"] == []
        assert result["edges"] == []


class TestExtractors:
    def test_extract_tools_and_methods(self) -> None:
        text = "Run a DFT calculation using VASP on Si."
        entities = extract_entities(text)
        assert "VASP" in entities["tools"]
        assert "DFT" in entities["methods"]
        assert "Si" in entities["materials"]

    def test_extract_materials_filters_false_positives(self) -> None:
        text = "Use VASP for DFT and MD on TiO2."
        mats = extract_materials(text)
        assert "TiO2" in mats
        assert "VASP" not in mats
        assert "DFT" not in mats
        assert "MD" not in mats

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("SCF did not converge", "SCF convergence failure"),
            ("ionic relaxation failed", "Geometry relaxation failure"),
            ("Out of memory", "Out of memory"),
            ("Connection timed out", "Timeout"),
            ("lost atoms on processor", "Lost atoms / broken topology"),
        ],
    )
    def test_extract_error_pattern(self, message: str, expected: str) -> None:
        assert extract_error_pattern(message) == expected


class TestBuilders:
    def test_build_from_seeds_imports_markdown_topics(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir()
        (seed_dir / "band_theory.md").write_text(
            "Band theory explains electronic structure."
        )
        (seed_dir / "md_basics.md").write_text(
            "Molecular dynamics uses Newton's equations."
        )

        kg = ProjectKnowledgeGraph(tmp_path / "kg")
        stats = build_from_seeds(kg, seed_dir=seed_dir)
        assert stats["topics"] == 2
        assert stats["links"] > 0
        assert kg.has_entity("band_theory", EntityType.TOPIC)

    def test_build_from_session_text_creates_links(self, tmp_path: Path) -> None:
        kg = ProjectKnowledgeGraph(tmp_path / "kg")
        stats = build_from_session_text(kg, "sess-1", "Run VASP on SiO2")
        assert stats["links"] >= 2
        assert kg.has_entity("VASP", EntityType.TOOL)
        assert kg.has_entity("SiO2", EntityType.MATERIAL)


class TestAgentKgIntegration:
    def test_kg_context_included_when_enabled(self, tmp_path: Path) -> None:
        kg_root = tmp_path / ".huginn"
        kg_root.mkdir()
        kg = ProjectKnowledgeGraph(kg_root)
        kg.add_entity("VASP", EntityType.TOOL)
        kg.add_entity("Si", EntityType.MATERIAL)
        kg.save()

        agent = HuginnAgent(
            model=None,
            tools=[],
            workspace=str(tmp_path),
            kg_enabled=True,
            kg_depth=1,
            kg_top_k=10,
        )
        msgs = agent._build_input_messages("How do I run VASP on Si?")
        contents = [m.content for m in msgs]
        assert any("Project Knowledge Context" in c for c in contents)
        assert any("VASP" in c for c in contents)

    def test_kg_context_absent_when_disabled(self, tmp_path: Path) -> None:
        kg_root = tmp_path / ".huginn"
        kg_root.mkdir()
        kg = ProjectKnowledgeGraph(kg_root)
        kg.add_entity("VASP", EntityType.TOOL)
        kg.save()

        agent = HuginnAgent(
            model=None,
            tools=[],
            workspace=str(tmp_path),
            kg_enabled=False,
        )
        msgs = agent._build_input_messages("How do I run VASP?")
        contents = [m.content for m in msgs]
        assert not any("Project Knowledge Context" in c for c in contents)


class TestPromptCacheBuilderKg:
    def test_kg_text_appended_after_memory(self) -> None:
        builder = PromptCacheBuilder(
            system_prompt="static",
            begin_dialogs=[("assistant", "hi")],
            cache_control=False,
        )
        msgs = builder.build_input_messages("memory", "question", kg_text="kg context")
        assert isinstance(msgs[0], type(builder.build_input_messages("", "")[0]))
        # Memory and KG are both SystemMessages; KG comes after memory.
        system_msgs = [m for m in msgs if m.__class__.__name__ == "SystemMessage"]
        assert len(system_msgs) == 2
        assert system_msgs[0].content == "memory"
        assert system_msgs[1].content == "kg context"
