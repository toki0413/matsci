"""Regression tests for the six residual risks left after the ws9/ws10 runs.

Each test pins one way the control plane still failed to notice a real defect:

- R1: a sign/exponent typo (``1.93e-14`` reported as ``1.93e14``) is a single
  untraced value, below the "3 untraced" threshold, so it never surfaced.
- R2: bare small integers ("10 steps") are counted as citation candidates.
- R3: a rounded restatement (``6061`` for ``6061.85``) is counted as untraced.
- R4: ``provenance.jsonl`` only records internal fast-path executions; adapter
  tool calls leave no snapshot.
- R5: sandbox/tool default ``work_dir`` falls back to ``cwd`` (the source tree)
  instead of the run workspace.
- R6: a ``code_lab`` timeout is retried at the same timeout, so one iteration
  can burn N x timeout.
"""

from __future__ import annotations

# ── R1: single-value magnitude / sign typo must surface ────────────────────


def test_sign_flip_single_value_is_flagged():
    from huginn.autoloop.engine_reflect import _magnitude_typos

    report = "## Results\n\nenergy drift was 1.93e14 J after 10^4 steps."
    pairs = _magnitude_typos(report, "energy_drift = 1.93e-14")

    assert pairs == [("1.93e+14", "1.93e-14")]


def test_matching_value_is_not_flagged():
    from huginn.autoloop.engine_reflect import _magnitude_typos

    report = "## Results\n\nenergy drift was 1.93e-14 J."
    assert _magnitude_typos(report, "energy_drift = 1.93e-14") == []


# ── R2: bare small integers are structural noise, not candidates ───────────


def test_bare_small_integers_are_not_candidates():
    from huginn.autoloop.engine_reflect import _citation_gap

    report = "## Results\n\nRun completed in 10 steps with L2 = 63.16."
    assert _citation_gap(report, '{"l2": 63.16}') == (0, 1)


def test_large_integer_still_counts_as_candidate():
    from huginn.autoloop.engine_reflect import _citation_gap

    report = "## Results\n\nSamples = 12000 and L2 = 63.16."
    assert _citation_gap(report, '{"l2": 63.16}') == (1, 2)


# ── R3: rounded restatement traces back to the full-precision value ────────


def test_rounded_restatement_is_traceable():
    from huginn.autoloop.engine_reflect import _citation_gap

    report = "## Results\n\nmean score = 6061 across seeds."
    assert _citation_gap(report, '{"mean_score": 6061.85}') == (0, 1)


def test_off_by_more_than_one_ulp_is_still_untraced():
    from huginn.autoloop.engine_reflect import _citation_gap

    report = "## Results\n\nmean score = 6069 across seeds."
    assert _citation_gap(report, '{"mean_score": 6061.85}') == (1, 1)


# ── R4: provenance must cover adapter tool calls, not just fast paths ──────


def test_provenance_snapshots_cover_adapter_tool_calls():
    from huginn.autoloop.cognitive_loop import _provenance_from_tool_calls

    calls = [
        {
            "step": 1,
            "tool": "bash",
            "args": {"command": ["ls"]},
            "result": {"ok": 1},
            "success": True,
            "error": None,
            "duration_ms": 3.0,
            "span_id": "s1",
        }
    ]
    chain = _provenance_from_tool_calls(calls, "/workspace/ws")

    assert len(chain) == 1
    assert chain[0]["tool_name"] == "bash"
    assert chain[0]["input_params"] == {"command": ["ls"]}
    assert chain[0]["workspace"] == "/workspace/ws"


def test_provenance_keeps_internal_fast_path_entries():
    from huginn.autoloop.cognitive_loop import _provenance_from_tool_calls

    calls = [
        {"tool": "code_lab", "args": {"cfg": {}}, "result": {"success": True}},
        {"tool": "file_read", "args": {"file_path": "a.txt"}, "result": "x"},
    ]
    chain = _provenance_from_tool_calls(calls, "/ws")

    assert [c["tool_name"] for c in chain] == ["code_lab", "file_read"]
    assert chain[0]["output_hash"] is None


# ── R5: default work_dir is the run workspace, not the source tree ─────────


def test_default_workspace_dir_prefers_env(tmp_path, monkeypatch):
    from huginn.utils.common import default_workspace_dir

    monkeypatch.setenv("HUGINN_WORKSPACE", str(tmp_path))
    assert default_workspace_dir() == tmp_path


def test_get_executor_scopes_to_workspace(tmp_path, monkeypatch):
    from huginn.security.execution import get_executor
    from huginn.security.sandbox import SandboxExecutor

    monkeypatch.setenv("HUGINN_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
    executor = get_executor()

    assert isinstance(executor, SandboxExecutor)
    assert tmp_path.resolve() in executor.config.allowed_work_dirs


def test_file_write_tool_defaults_to_workspace(tmp_path, monkeypatch):
    import asyncio

    from huginn.tools.file_write_tool import FileWriteTool

    monkeypatch.setenv("HUGINN_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("WORKSPACE_DIR", raising=False)
    tool = FileWriteTool()

    result = asyncio.run(tool.call({"file_path": "out.txt", "content": "x"}))

    assert result.success is True
    assert (tmp_path / "out.txt").exists()


# ── R6: repeated timeouts tighten the cap and stop ─────────────────────────


def test_timeout_cap_halves_after_first_timeout():
    from huginn.autoloop.engine_act import _next_timeout_cap

    cap, allow = _next_timeout_cap(None, 600.0, timed_out=True)

    assert cap == 300.0
    assert allow is True


def test_consecutive_timeouts_stop_retrying():
    from huginn.autoloop.engine_act import _next_timeout_cap

    _, allow = _next_timeout_cap(300.0, 300.0, timed_out=True)

    assert allow is False


def test_no_timeout_keeps_the_cap():
    from huginn.autoloop.engine_act import _next_timeout_cap

    assert _next_timeout_cap(None, 600.0, timed_out=False) == (None, True)


def test_sandbox_executor_scopes_empty_config(tmp_path, monkeypatch):
    """code_tool 等直接 `SandboxExecutor()` 的调用点也必须落在 run workspace.

    `get_executor()` 修好后仍有一条旁路: code_tool 的实例化沙箱用默认 config,
    allowed_work_dirs 为空 → strict 静默失效, 代码执行可写源码树 (启动日志里的
    "path scoping inactive")。
    """
    from huginn.security.sandbox import SandboxConfig, SandboxExecutor

    monkeypatch.setenv("HUGINN_WORKSPACE", str(tmp_path))
    executor = SandboxExecutor(SandboxConfig())

    assert tmp_path.resolve() in executor.config.allowed_work_dirs


def test_sandbox_explicit_scope_is_kept(tmp_path):
    """显式给了 allowed_work_dirs 就尊重调用方, 不用 workspace 覆盖."""
    from huginn.security.sandbox import SandboxConfig, SandboxExecutor

    explicit = tmp_path / "explicit"
    forced = tmp_path / "forced"
    explicit.mkdir()
    forced.mkdir()
    executor = SandboxExecutor(SandboxConfig(allowed_work_dirs={explicit.resolve()}))

    assert executor.config.allowed_work_dirs == {explicit.resolve()}
