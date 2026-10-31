"""执行覆盖采样守护 (结构完整性修复 D 轨 T11–T12).

- ``HUGINN_TRACE=1`` 才采样, 默认关;
- 采样按顶层模块计数, 只算 ``huginn/`` 内的帧;
- 落 ``trace.json``, 并能对照 census 列出「KEEP 但零执行」;
- driver 结束时写 trace, 且不虚报未执行模块.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from huginn import run_cognitive_driver as driver
from huginn.env_access import is_declared
from huginn.trace import (
    HUGINN_ROOT,
    TraceSampler,
    keep_but_zero,
    keep_modules,
    load_census,
    module_of,
    trace_enabled,
)


def test_trace_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HUGINN_TRACE", raising=False)
    assert trace_enabled() is False
    monkeypatch.setenv("HUGINN_TRACE", "1")
    assert trace_enabled() is True


def test_module_of_maps_files_to_top_level_module():
    assert module_of(str(HUGINN_ROOT / "autoloop/cognitive_loop.py")) == "autoloop"
    assert module_of(str(HUGINN_ROOT / "env_schema.py")) == "env_schema"
    assert module_of("/usr/lib/python3/json/__init__.py") is None


def test_sampler_counts_huginn_calls_twice():
    sampler = TraceSampler()
    with sampler:
        is_declared("HUGINN_API_KEY")
        is_declared("HUGINN_API_KEY")
    assert sampler.counts.get("env_access", 0) >= 2


def test_sampler_ignores_non_huginn_frames():
    sampler = TraceSampler()
    sampler.start()
    json.dumps({"x": 1})
    sampler.stop()
    # 只允许采样器自身的 stop 帧 (调用时采样尚未关闭), stdlib 帧一律不计.
    assert set(sampler.counts) <= {"trace"}


def test_write_trace_json(tmp_path):
    sampler = TraceSampler()
    with sampler:
        is_declared("HUGINN_API_KEY")
    out = sampler.write(tmp_path)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["sampled_by"] == "sys.setprofile"
    assert payload["counts"]["env_access"] >= 1
    assert payload["total_calls"] >= 1


def test_keep_but_zero_lists_unexecuted_keep_modules(tmp_path):
    census = tmp_path / "census.csv"
    census.write_text(
        "module,verdict\nagent,KEEP(可达)\nautoloop,KEEP(在工作流内)\npet,MOVE(范围外)\n",
        encoding="utf-8",
    )
    rows = load_census(census)
    assert keep_modules(rows) == {"agent", "autoloop"}
    assert keep_but_zero({"agent": 3}, rows) == ["autoloop"]


class _FakeEngine:
    def __init__(self, **_kwargs):
        pass

    async def run_cognitive(self, objective, max_iterations=1):
        # 模拟真实运行里被调用到的 huginn 模块.
        is_declared("HUGINN_API_KEY")
        return SimpleNamespace(
            success=True,
            objective=objective,
            run_id="fake-run",
            goal_achieved=True,
            phases=[],
            tests_passed=True,
            constraints_satisfied=True,
        )


@pytest.fixture
def driver_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("HUGINN_ALLOW_UNCONTRACTED", "1")
    monkeypatch.setenv("HUGINN_TRACE", "1")
    monkeypatch.setattr(driver, "AutoloopEngine", _FakeEngine)
    monkeypatch.setattr(driver, "MemoryManager", lambda: object())
    return tmp_path


def test_driver_writes_trace_on_exit(driver_workspace):
    asyncio.run(driver._main("two offset rounds", 2))
    trace_path = driver_workspace / "trace.json"
    assert trace_path.is_file()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["counts"].get("env_access", 0) >= 1
    # 未执行的模块不得被虚报成已执行.
    assert "pet" not in payload["counts"]
    summary = json.loads(
        (driver_workspace / "run_result.json").read_text(encoding="utf-8")
    )
    assert summary["trace_file"] == str(trace_path)
