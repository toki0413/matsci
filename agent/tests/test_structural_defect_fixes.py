"""Regression tests for the P0/P1/P2 structural-defect fixes.

Every test pins one sink that previously swallowed or faked a failure:

- P0-1: ``adapter._validate_tool_args`` repairs shape-only arg errors once and
  otherwise returns a self-repair receipt instead of a bare ValidationError.
- P0-2: ``EngineAct._derive_coder_outcome`` derives success from real receipts
  (it used to be hardcoded ``True``).
- P1-1: pre-execution paths emit a ``tool_call`` telemetry span.
- P1-2: a failed coder falls back to code_lab / probe_exec.
- P2-1: reasoning phases can call the read-only tool surface, and ``bind_tools``
  failure degrades to the old no-tool call instead of breaking the phase.
- P2-2: tools whose backing deps are missing are demoted out of the LLM surface.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from huginn.autoloop.engine_act import EngineAct
from huginn.telemetry import TelemetryCollector, set_telemetry_collector
from huginn.tools.adapter import _emit_span, _validate_tool_args
from huginn.tools.bash_tool import BashTool


# ── P0-1: argument boundary validation ─────────────────────────────────────


def test_json_string_array_is_repaired_to_list():
    tool = BashTool()
    data, error, repairs = _validate_tool_args(tool, {"command": '["echo","hi"]'})

    assert error is None
    assert data is not None
    assert list(data.command) == ["echo", "hi"]
    assert repairs


def test_unparseable_command_returns_self_repair_receipt():
    tool = BashTool()
    data, error, _ = _validate_tool_args(tool, {"command": "echo hi there"})

    assert data is None
    assert error is not None
    assert error["_invalid_args"] is True
    assert error["tool"] == tool.name
    assert error["detail"][0]["field"] == "command"
    assert error["expected_shape"]["command"] == ["string"]


def test_valid_list_arguments_pass_through_unchanged():
    tool = BashTool()
    data, error, repairs = _validate_tool_args(tool, {"command": ["echo", "ok"]})

    assert error is None
    assert data.command == ["echo", "ok"]
    assert repairs == []


# ── P1-1: failure paths are visible in telemetry ───────────────────────────


def test_pre_execution_failure_emits_tool_call_span():
    from huginn.telemetry import get_telemetry_collector

    previous = get_telemetry_collector()
    collector = TelemetryCollector()
    set_telemetry_collector(collector)
    try:
        _emit_span(
            "bash_tool",
            {"command": "echo hi"},
            {"error": "invalid_tool_args"},
            stage="schema_validation",
        )
    finally:
        set_telemetry_collector(previous)

    spans = [s for s in collector._roots if s.name == "tool_call"]
    assert len(spans) == 1
    assert spans[0].metadata["success"] is False
    assert spans[0].metadata["stage"] == "schema_validation"
    assert spans[0].metadata["error"] == "invalid_tool_args"


def test_internal_fast_path_counts_as_tool_call_in_trajectory():
    """code_lab / probe_exec 不经 adapter, 但必须出现在 trajectory.tool_calls.

    否则"真跑了实验"在遥测里仍是 0 次工具调用, 利用率被系统性低估,
    G31 审计还会把它误判成空转.
    """
    from huginn.telemetry import _extract_tool_calls, get_telemetry_collector

    previous = get_telemetry_collector()
    collector = TelemetryCollector()
    set_telemetry_collector(collector)
    try:
        with collector.span("phase:execute"):
            EngineAct._record_internal_tool_call(
                object(),
                "code_lab",
                {"mode": "code_lab"},
                {"status": "completed", "success": True, "result": {"score": 1.0}},
            )
            EngineAct._record_internal_tool_call(
                object(),
                "probe_exec",
                {"mode": "probe_exec"},
                {"status": "failed", "success": False, "error": "syntax error"},
            )
    finally:
        set_telemetry_collector(previous)

    calls = _extract_tool_calls(collector.to_dict())
    assert [c["tool"] for c in calls] == ["code_lab", "probe_exec"]
    assert calls[0]["success"] is True
    assert calls[1]["success"] is False
    assert calls[1]["error"] == "syntax error"


# ── P0-2: coder success is derived, not hardcoded ──────────────────────────


def _coder_host() -> object:
    class _Host:
        _CODER_LIMIT_PREFIX = EngineAct._CODER_LIMIT_PREFIX

    host = _Host()
    host._derive_coder_outcome = EngineAct._derive_coder_outcome.__get__(host)
    return host


def test_all_tool_calls_failed_means_coder_failed():
    msgs = [
        AIMessage(content="", tool_calls=[{"name": "bash_tool", "args": {}, "id": "1"}]),
        ToolMessage(
            content='{"error": "list_type"}', tool_call_id="1", name="bash_tool"
        ),
    ]
    success, diag = _coder_host()._derive_coder_outcome("looks done", msgs)

    assert success is False
    assert diag["failure_reason"] == "all_tool_calls_failed"
    assert diag["tool_error_count"] == 1


def test_usable_tool_result_means_coder_succeeded():
    msgs = [
        AIMessage(content="", tool_calls=[{"name": "bash_tool", "args": {}, "id": "1"}]),
        ToolMessage(content='{"stdout": "hi"}', tool_call_id="1", name="bash_tool"),
    ]
    success, diag = _coder_host()._derive_coder_outcome("done", msgs)

    assert success is True
    assert diag["usable_tool_results"] == 1


def test_iteration_limit_and_no_tools_are_failures():
    host = _coder_host()
    msgs = [
        AIMessage(content="", tool_calls=[{"name": "bash_tool", "args": {}, "id": "1"}]),
        ToolMessage(content='{"stdout": "hi"}', tool_call_id="1", name="bash_tool"),
    ]

    limited, diag = host._derive_coder_outcome(
        EngineAct._CODER_LIMIT_PREFIX + "\nstill working", msgs
    )
    assert limited is False
    assert diag["failure_reason"] == "iteration_limit"

    idle, diag = host._derive_coder_outcome("done", [AIMessage(content="done")])
    assert idle is False
    assert diag["failure_reason"] == "no_tool_calls"


# ── P1-2: failed coder falls back to a real execution path ─────────────────


class _FallbackHost:
    _CODER_LIMIT_PREFIX = EngineAct._CODER_LIMIT_PREFIX
    _objective = "compute a deterministic numeric value"

    def __init__(self, lab_success: bool):
        self._lab_success = lab_success
        self.calls: list[str] = []

    async def _execute_code_lab(self, description, context):
        self.calls.append("code_lab")
        if self._lab_success:
            return {
                "mode": "code_lab",
                "status": "completed",
                "success": True,
                "objectives": {"ok": 1},
            }
        return {"mode": "code_lab", "status": "failed", "success": False,
                "error": "no code"}

    def _is_deterministic_numeric(self, text):
        return True

    async def _request_numeric_probe(self, description):
        self.calls.append("probe")
        return "print(1 + 1)"

    def _run_snippet_to_output(self, snippet):
        self.calls.append("run_probe")
        return "2"


async def test_coder_failure_falls_back_to_code_lab():
    host = _FallbackHost(lab_success=True)
    # 走未绑定方法: EngineAct 自带 _execute_code_lab, 直接构造会遮住替身.
    result = await EngineAct._fallback_from_coder(host, "do it", {})

    assert result is not None
    assert result["mode"] == "code_lab"
    assert result["fallback_from"] == "coder"
    assert host.calls == ["code_lab"]


async def test_coder_fallback_escalates_to_probe_exec():
    host = _FallbackHost(lab_success=False)
    result = await EngineAct._fallback_from_coder(host, "do it", {})

    assert result is not None
    assert result["mode"] == "probe_exec"
    assert result["result"] == "2"
    assert host.calls == ["code_lab", "probe", "run_probe"]


async def test_coder_fallback_exhausted_returns_none():
    class _NoProbe(_FallbackHost):
        def _is_deterministic_numeric(self, text):
            return False

    host = _NoProbe(lab_success=False)
    assert await EngineAct._fallback_from_coder(host, "do it", {}) is None


# ── P2-1: read-only tools in reasoning phases ──────────────────────────────


class _FakeTool:
    name = "recall"

    def __init__(self):
        self.seen: list[dict] = []

    async def ainvoke(self, args):
        self.seen.append(args)
        return {"hit": True}


class _FakeBoundModel:
    def __init__(self):
        self.rounds = 0

    async def ainvoke(self, messages):
        self.rounds += 1
        if self.rounds == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "recall", "args": {"query": "x"}, "id": "c1"},
                    {"name": "ghost", "args": {}, "id": "c2"},
                ],
            )
        return AIMessage(content="final plan text")


class _FakeLLM:
    def __init__(self):
        self.bound = _FakeBoundModel()

    def bind_tools(self, tools):
        return self.bound


class _NoBindLLM:
    def bind_tools(self, tools):
        raise NotImplementedError("no function calling")

    async def ainvoke(self, messages):
        return AIMessage(content="plain answer")


class _ChatHost:
    _current_phase = "planning"

    async def _track_llm_usage(self, usage):
        return None


async def test_phase_tool_loop_executes_tools_and_stops():
    tool = _FakeTool()
    result = await EngineAct(_ChatHost())._llm_chat_with_tools(
        _FakeLLM(), [], [tool], max_tool_rounds=2
    )

    assert result == "final plan text"
    assert tool.seen == [{"query": "x"}]


async def test_bind_tools_failure_degrades_to_plain_call():
    result = await EngineAct(_ChatHost())._llm_chat_with_tools(
        _NoBindLLM(), [], [_FakeTool()]
    )

    assert result == "plain answer"


# ── P2-2: degraded tools leave the LLM surface ─────────────────────────────


async def test_degraded_tool_is_demoted_out_of_schemas():
    from huginn.tools import _demote_unavailable_tool
    from huginn.tools.base import HuginnTool
    from huginn.tools.registry import ToolRegistry

    class _DegradedTool(HuginnTool):
        name = "degraded_probe_tool"
        description = "needs a missing heavy dependency"
        input_schema = None

    snapshot = ToolRegistry.snapshot()
    try:
        tool = _DegradedTool()
        ToolRegistry.register(tool)
        assert any(
            s["function"]["name"] == "degraded_probe_tool"
            for s in ToolRegistry.get_all_schemas()
        )

        assert _demote_unavailable_tool(
            "huginn.tools.missing_module", "_DegradedTool", ["torch"]
        )
        assert not any(
            s["function"]["name"] == "degraded_probe_tool"
            for s in ToolRegistry.get_all_schemas()
        )
    finally:
        ToolRegistry.restore(snapshot)
