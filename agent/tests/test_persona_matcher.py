"""Tests for persona query matching and runtime switching."""

from __future__ import annotations

import pytest

from huginn.agent import HuginnAgent
from huginn.persona_emotion import EmotionTracker
from huginn.persona_matcher import PersonaMatcher, match_persona_for_query
from huginn.personas import Persona, PersonaManager


@pytest.fixture
def manager(tmp_path):
    return PersonaManager(workspace=tmp_path)


class TestPersonaMatcher:
    def test_keyword_match_prefers_relevant_description(self, manager):
        manager.create(
            name="dft_expert",
            system_prompt="DFT expert",
            description="Density functional theory calculations and band structures",
            when_to_use=["DFT", "VASP", "band structure"],
        )
        manager.create(
            name="md_expert",
            system_prompt="MD expert",
            description="Molecular dynamics and LAMMPS simulations",
            when_to_use=["MD", "LAMMPS", "trajectory"],
        )
        matcher = PersonaMatcher(manager=manager)
        results = matcher.match("Run a VASP band structure calculation", top_k=1)
        assert results[0][0].name == "dft_expert"

    def test_keyword_match_returns_empty_when_no_overlap(self, manager):
        manager.create(
            name="reviewer",
            system_prompt="Reviewer",
            description="Peer review of manuscripts",
        )
        matcher = PersonaMatcher(manager=manager)
        # Force the keyword-only path so the test is deterministic and fast.
        matcher._embedding_available = False
        results = matcher.match("xyz abc qrs", top_k=1)
        assert results == []

    def test_match_persona_for_query_none_below_threshold(self, manager):
        manager.create(
            name="tutor",
            system_prompt="Tutor",
            description="Explain materials science concepts",
        )
        assert match_persona_for_query(
            "completely unrelated query", manager=manager, score_threshold=1.0
        ) is None

    def test_name_match_boosts_score(self, manager):
        manager.create(
            name="vasp_helper",
            system_prompt="VASP helper",
            description="General computational helper",
        )
        assert (
            match_persona_for_query("vasp_helper please help", manager=manager)
            == "vasp_helper"
        )


class TestRuntimePersonaSwitch:
    def test_set_persona_updates_prompt_and_graph(self):
        agent = HuginnAgent(model=None, tools=[])
        original_prompt = agent.system_prompt

        agent.set_persona(
            Persona(name="reviewer", system_prompt="You are a reviewer."),
            emotion_tracker=EmotionTracker("reviewer"),
        )

        assert agent.persona_name == "reviewer"
        assert agent.system_prompt == "You are a reviewer."
        assert agent._agent_graph is None
        assert agent.emotion_tracker is not None
        # Switching back should restore the previous prompt
        agent.set_persona(
            Persona(name="default", system_prompt=original_prompt),
        )
        assert agent.system_prompt == original_prompt

    def test_set_persona_preserves_memory(self):
        agent = HuginnAgent(model=None, tools=[])
        agent.remember("silicon band gap", category="fact")
        agent.set_persona(Persona(name="tutor", system_prompt="You are a tutor."))
        assert any("silicon" in r["content"] for r in agent.recall("silicon"))
