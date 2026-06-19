"""Tests for the autonomous coder loop."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from huginn.coder.loop import CoderRunner
from huginn.config import CoderSettings, HuginnConfig, Settings
from huginn.permissions import PermissionConfig


class FakeBoundModel:
    """Fake bound model returned by FakeModel.bind_tools."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = iter(responses)

    def invoke(self, messages: list[Any]) -> AIMessage:
        return next(self.responses)


class FakeModel:
    """Fake chat model for unit tests."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses

    def bind_tools(self, tools: list[Any]) -> FakeBoundModel:
        return FakeBoundModel(self.responses)


def _make_settings() -> Settings:
    return Settings(
        config=HuginnConfig(provider="openai", model="gpt-4o", api_key="test-key"),
        coder=CoderSettings(max_iterations=5, done_marker="[DONE]"),
    )


def _fake_get_model_factory(responses: list[AIMessage]):
    def fake_get_model(config: Any = None) -> FakeModel:
        return FakeModel(responses)

    return fake_get_model


def test_default_tools_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "huginn.coder.loop.get_model",
        _fake_get_model_factory([]),
    )
    runner = CoderRunner(settings=_make_settings())
    names = {t.name for t in runner.tools}
    assert names == {
        "file_read_tool",
        "file_write_tool",
        "file_edit_tool",
        "bash_tool",
        "git_tool",
        "code_tool",
    }


def test_run_returns_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    final = AIMessage(content="All done. [DONE]")
    monkeypatch.setattr(
        "huginn.coder.loop.get_model",
        _fake_get_model_factory([final]),
    )
    runner = CoderRunner(settings=_make_settings())
    result = runner.run("Say hello")
    assert "All done." in result["final_answer"]
    assert "[DONE]" not in result["final_answer"]


def test_run_executes_tool_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    call_id = "call_1"
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "file_write_tool",
                "args": {
                    "file_path": "test.txt",
                    "content": "hello coder",
                    "working_dir": str(tmp_path),
                },
                "id": call_id,
            }
        ],
    )
    final = AIMessage(content="File written. [DONE]")

    monkeypatch.setattr(
        "huginn.coder.loop.get_model",
        _fake_get_model_factory([tool_call, final]),
    )
    # Auto-approve so the test can focus on tool invocation, not permission prompts.
    runner = CoderRunner(
        settings=_make_settings(),
        permission_config=PermissionConfig(auto_approve_all=True),
    )
    result = runner.run("Write a file")
    assert "File written." in result["final_answer"]
    assert (tmp_path / "test.txt").read_text() == "hello coder"


def test_read_only_tool_auto_approved(monkeypatch: pytest.MonkeyPatch) -> None:
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "file_read_tool",
                "args": {"file_path": "nonexistent.txt"},
                "id": "call_ro",
            }
        ],
    )
    final = AIMessage(content="Done. [DONE]")

    monkeypatch.setattr(
        "huginn.coder.loop.get_model",
        _fake_get_model_factory([tool_call, final]),
    )
    runner = CoderRunner(settings=_make_settings())
    result = runner.run("Read a file")
    assert "Done." in result["final_answer"]


def test_destructive_tool_denied_without_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "file_write_tool",
                "args": {
                    "file_path": "test.txt",
                    "content": "should not write",
                    "working_dir": str(tmp_path),
                },
                "id": "call_denied",
            }
        ],
    )
    final = AIMessage(content="Finished. [DONE]")

    monkeypatch.setattr(
        "huginn.coder.loop.get_model",
        _fake_get_model_factory([tool_call, final]),
    )
    runner = CoderRunner(settings=_make_settings())  # no callback
    result = runner.run("Write a file")
    assert "Finished." in result["final_answer"]
    assert not (tmp_path / "test.txt").exists()


def test_destructive_tool_approved_by_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "file_write_tool",
                "args": {
                    "file_path": "test.txt",
                    "content": "approved",
                    "working_dir": str(tmp_path),
                },
                "id": "call_approved",
            }
        ],
    )
    final = AIMessage(content="Finished. [DONE]")

    monkeypatch.setattr(
        "huginn.coder.loop.get_model",
        _fake_get_model_factory([tool_call, final]),
    )
    runner = CoderRunner(
        settings=_make_settings(),
        approval_callback=lambda _name, _reason: True,
    )
    result = runner.run("Write a file")
    assert "Finished." in result["final_answer"]
    assert (tmp_path / "test.txt").read_text() == "approved"


def test_run_respects_max_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    # Model keeps emitting tool calls and never finishes, so the loop should
    # hit the iteration limit.
    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "file_read_tool",
                "args": {"file_path": "nonexistent.txt"},
                "id": "call_n",
            }
        ],
    )

    monkeypatch.setattr(
        "huginn.coder.loop.get_model",
        _fake_get_model_factory([tool_call] * 5),
    )
    settings = _make_settings()
    settings.coder.max_iterations = 3
    runner = CoderRunner(settings=settings)
    result = runner.run("Loop forever")
    assert "maximum iteration limit" in result["final_answer"]
    # 1 system + 1 human + 3 (assistant + tool) pairs = 8 messages total
    assert len(result["messages"]) == 8
