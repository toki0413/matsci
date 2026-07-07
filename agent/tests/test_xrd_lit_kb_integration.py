"""Integration tests for the XRD sim/hint chain and literature KB write-back.

Covers two complementary links that were previously broken:
1. characterization_tool emits auto_indexing_hint so the agent can chain into
   xrd_sim_tool.index_peaks without hard-wiring the two tools together.
2. literature_tool.benchmark_lookup writes extracted reported values back into
   the knowledge base so later queries on the same system can hit locally.
"""

from __future__ import annotations

import asyncio
import csv
import json
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from huginn.tools.characterization_tool import CharacterizationTool
from huginn.tools.literature.tool import LiteratureTool, LiteratureInput
from huginn.types import ToolContext


# ── Test 1: XRD peak detect returns auto_indexing_hint ───────────────


def _make_xrd_csv(path, n_points=500):
    x = np.linspace(10, 80, n_points)
    y = np.zeros_like(x)
    for center, amp in [(25.0, 100.0), (45.0, 80.0), (65.0, 60.0)]:
        y += amp / (1 + ((x - center) / 1.5) ** 2)
    y += np.random.default_rng(0).normal(0, 1, size=x.shape)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["2theta", "intensity"])
        for xi, yi in zip(x, y):
            writer.writerow([xi, yi])


class TestXrdAutoIndexingHint:
    def test_hint_present_with_peak_list(self, tmp_path):
        csv_path = tmp_path / "xrd.csv"
        _make_xrd_csv(csv_path)

        tool = CharacterizationTool()
        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="xrd_peak_detect",
                    data_path=str(csv_path),
                    parameters={"threshold": 10.0, "distance": 10},
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert "auto_indexing_hint" in result.data

        hint = result.data["auto_indexing_hint"]
        # hint carries the full peak list (2theta + intensity) and a flat
        # positions array ready to feed straight into xrd_sim_tool.index_peaks
        assert "peaks" in hint
        assert "peak_positions" in hint
        assert "next_action" in hint
        assert hint["peaks"] == result.data["peaks"] or len(hint["peaks"]) == result.data["n_peaks"]
        assert len(hint["peak_positions"]) == result.data["n_peaks"]
        # every peak entry has the keys xrd_sim_tool expects
        for p in hint["peaks"]:
            assert "two_theta" in p
            assert "intensity" in p
        # the hint message mentions how many peaks and which tool to call next
        assert "index_peaks" in hint["next_action"]
        assert str(result.data["n_peaks"]) in hint["next_action"]


# ── Test 2: benchmark_lookup writes reported values back to KB ───────


def _fake_papers():
    return [
        {
            "title": "Band gap of GaN from first principles",
            "doi": "10.1234/fake.001",
            "year": 2021,
            "venue": "J. Fake Mater.",
            "abstract": "We report a band gap of 3.4 eV for wurtzite GaN using HSE06.",
        },
        {
            "title": "GaN optoelectronic properties review",
            "doi": "10.1234/fake.002",
            "year": 2020,
            "venue": "Rev. Fake Phys.",
            "abstract": "The experimental band gap of GaN is 3.39 eV at room temperature.",
        },
    ]


# LLM is supposed to return this JSON shape (see _BENCHMARK_SYSTEM_PROMPT)
_FAKE_LLM_JSON = json.dumps({
    "values": [
        {"value": 3.4, "unit": "eV", "method": "HSE06", "paper_idx": 1, "note": ""},
        {"value": 3.39, "unit": "eV", "method": "experiment", "paper_idx": 2, "note": ""},
    ]
})


class FakeKB:
    """Minimal stand-in for KnowledgeBase — just records add_text calls."""

    def __init__(self):
        self.added: list[dict] = []

    def add_text(self, text, filename="auto", metadata=None):
        self.added.append({"text": text, "filename": filename, "metadata": metadata})
        return {"doc_id": f"fake_{len(self.added)}", "chunks": 1}


class TestBenchmarkLookupKBWriteback:
    def test_kb_written_flag_and_calls(self, tmp_path, monkeypatch):
        tool = LiteratureTool()

        # skip real LLM init + invocation
        monkeypatch.setattr(tool, "_get_model", lambda ctx: MagicMock())
        monkeypatch.setattr(
            tool, "_llm_invoke", AsyncMock(return_value=_FAKE_LLM_JSON)
        )

        # swap in our fake KB so we can inspect the calls
        fake_kb = FakeKB()
        monkeypatch.setattr(
            "huginn.knowledge.store.get_knowledge_base", lambda *a, **k: fake_kb
        )

        args = LiteratureInput(
            action="benchmark_lookup",
            system="GaN wurtzite",
            property="band gap",
            papers=_fake_papers(),
        )
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace=str(tmp_path)))
        )

        assert result.success is True
        assert "kb_written" in result.data
        n_values = len(result.data["reported_values"])
        assert n_values == 2
        assert result.data["kb_written"] == n_values
        # KB got one add_text per reported value, each carrying DOI + title
        assert len(fake_kb.added) == n_values
        for entry in fake_kb.added:
            meta = entry["metadata"]
            assert meta["source"] == "benchmark_lookup"
            assert meta["doi"]
            assert meta["title"]

    def test_kb_failure_does_not_break_lookup(self, tmp_path, monkeypatch):
        tool = LiteratureTool()

        monkeypatch.setattr(tool, "_get_model", lambda ctx: MagicMock())
        monkeypatch.setattr(
            tool, "_llm_invoke", AsyncMock(return_value=_FAKE_LLM_JSON)
        )

        # KB blows up — main flow should still return the extracted values
        def _boom(*a, **k):
            raise RuntimeError("kb unavailable")

        monkeypatch.setattr("huginn.knowledge.store.get_knowledge_base", _boom)

        args = LiteratureInput(
            action="benchmark_lookup",
            system="GaN wurtzite",
            property="band gap",
            papers=_fake_papers(),
        )
        result = asyncio.run(
            tool.call(args, ToolContext(session_id="test", workspace=str(tmp_path)))
        )

        assert result.success is True
        assert result.data["kb_written"] == 0
        assert len(result.data["reported_values"]) == 2
