"""P0 integration tests for the coder loop (huginn/coder/loop.py).

CoderRunner.run() is a synchronous ReAct loop: it calls model.invoke(),
processes tool calls, and stops when it sees the done marker ("[DONE]")
or hits max_iterations. These tests drive the real run() method with a
FakeLLM so we verify the loop control flow, not the LLM itself.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from huginn.coder.loop import CoderRunner
from tests.fixtures.fake_llm import FakeLLM, make_scripted_llm


# ── helpers ────────────────────────────────────────────────────────


def _make_coder(monkeypatch, fake_llm: FakeLLM, **kw) -> CoderRunner:
    """Build a CoderRunner whose model is the FakeLLM.

    Monkeypatches get_model in the coder.loop namespace so __init__ picks
    up the fake. Passes tools=[] to skip the default toolset build (we
    don't need real file/bash tools for loop-control-flow tests).
    """
    monkeypatch.setattr("huginn.coder.loop.get_model", lambda settings: fake_llm)
    return CoderRunner(tools=[], **kw)


# ── 1. basic coding: LLM returns "[DONE]" → final_answer ──────────


class TestCoderBasicDone:
    def test_done_marker_stops_loop(self, monkeypatch, tmp_path):
        """When the LLM's response contains [DONE], the loop stops immediately."""
        llm = make_scripted_llm([
            AIMessage(content="def hello():\n    return 'world'\n[DONE]"),
        ])
        coder = _make_coder(monkeypatch, llm)
        result = coder.run("write a hello function")

        assert result["final_answer"] == "def hello():\n    return 'world'"
        assert "[DONE]" not in result["final_answer"]
        # exactly one LLM call before stopping
        assert llm.call_count == 1

    def test_plain_text_without_done(self, monkeypatch, tmp_path):
        """No tool calls and no done marker → treated as final answer."""
        llm = make_scripted_llm([
            AIMessage(content="The answer is 42"),
        ])
        coder = _make_coder(monkeypatch, llm)
        result = coder.run("what is the answer")

        assert result["final_answer"] == "The answer is 42"


# ── 2. iteration limit: keeps calling tools → forced stop ─────────


class TestCoderIterationLimit:
    def test_forced_stop_at_max(self, monkeypatch, tmp_path):
        """LLM keeps calling tools; loop stops at max_iterations=3."""
        # Every response has a tool call — the loop will never self-terminate
        llm = make_scripted_llm([
            AIMessage(
                content="",
                tool_calls=[{"name": "bash_tool", "args": {"cmd": "ls"}, "id": "c1"}],
            ),
        ])
        coder = _make_coder(monkeypatch, llm)
        result = coder.run("keep running commands", max_iterations=3)

        assert "maximum iteration limit" in result["final_answer"].lower()
        assert llm.call_count == 3

    def test_done_on_second_iteration(self, monkeypatch, tmp_path):
        """First call has a tool call, second has [DONE] → stops at 2."""
        llm = make_scripted_llm([
            AIMessage(
                content="",
                tool_calls=[{"name": "file_read_tool", "args": {"path": "x"}, "id": "c1"}],
            ),
            AIMessage(content="Done reading.\n[DONE]"),
        ])
        coder = _make_coder(monkeypatch, llm)
        result = coder.run("read the file and summarize")

        assert result["final_answer"] == "Done reading."
        assert llm.call_count == 2


# ── 3. done marker stripping ──────────────────────────────────────


class TestCoderDoneMarkerStripping:
    def test_marker_removed_from_answer(self, monkeypatch, tmp_path):
        """[DONE] is stripped from the final answer, including trailing text."""
        llm = make_scripted_llm([
            AIMessage(content="Result: success\n[DONE]\n(garbage after marker)"),
        ])
        coder = _make_coder(monkeypatch, llm)
        result = coder.run("do something")

        # split(done_marker, 1)[0] keeps everything before [DONE]
        assert result["final_answer"] == "Result: success"
        assert "[DONE]" not in result["final_answer"]

    def test_marker_mid_content(self, monkeypatch, tmp_path):
        """Marker in the middle: only the prefix is kept."""
        llm = make_scripted_llm([
            AIMessage(content="part one [DONE] part two"),
        ])
        coder = _make_coder(monkeypatch, llm)
        result = coder.run("task")

        assert result["final_answer"] == "part one"
        assert "part two" not in result["final_answer"]
