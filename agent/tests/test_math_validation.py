"""run_math_validation 回归测试.

锁住:
  * execution_result 带 equations → BourbakiTool 被调, conservation 填充
  * 带 lagrangian + coordinates → LeanTool 被调, variational 填充
  * 带 autodiff function spec → AutoDiffTool 被调, autodiff 填充
  * 缺字段时对应子项跳过, 不报错
  * 非 dict execution_result → 空结果
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.math_validation import run_math_validation
from huginn.types import ToolResult


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    # ponytail: KB 冷启动跑 ONNX embedding > 120s, KG 写 ~/.huginn 污染 home
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    return AutoloopEngine(workspace=tmp_path)


class _FakeBourbaki:
    last_args: dict = {}

    async def call(self, args, ctx):
        _FakeBourbaki.last_args = args
        return {"verified": True, "message": "SymPy ok", "fallback": False}


class _FakeLean:
    last_args = None

    async def call(self, args, ctx):
        _FakeLean.last_args = args
        return ToolResult(
            success=True,
            data={"euler_lagrange": "m*x'' + k*x = 0"},
            error=None,
        )


class _FakeAutoDiff:
    last_args = None

    async def call(self, args, ctx):
        _FakeAutoDiff.last_args = args
        return ToolResult(
            success=True,
            data={"grad_norm": 0.42},
            error=None,
        )


class TestRunMathValidation:
    def test_all_three_subitems_populated(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "huginn.tools.bourbaki_tool.BourbakiTool", lambda: _FakeBourbaki()
        )
        monkeypatch.setattr(
            "huginn.tools.lean_tool.LeanTool", lambda: _FakeLean()
        )
        monkeypatch.setattr(
            "huginn.tools.sci.autodiff_tool.AutoDiffTool", lambda: _FakeAutoDiff()
        )

        exec_result = {
            "equations": "x + y = x + y",
            "conservation_domain": "continuum_mechanics",
            "lagrangian": "1/2*m*v**2 - 1/2*k*x**2",
            "coordinates": ["x"],
            "velocities": ["v"],
            "autodiff": {
                "function_type": "birch_murnaghan",
                "function_params": {"E0": 0.0, "B0": 100.0},
                "variables": {"V": [100.0]},
                "target_variable": "V",
            },
        }
        out = asyncio.run(run_math_validation(engine, exec_result))
        assert out["conservation"]["verified"] is True
        assert out["conservation"]["method"] == "bourbaki"
        assert out["variational"]["ok"] is True
        assert out["variational"]["method"] == "lean"
        assert out["autodiff"]["ok"] is True
        assert out["autodiff"]["data"]["grad_norm"] == 0.42

    def test_skips_missing_subitems(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 只有 equations, 没 lagrangian / autodiff
        monkeypatch.setattr(
            "huginn.tools.bourbaki_tool.BourbakiTool", lambda: _FakeBourbaki()
        )
        out = asyncio.run(run_math_validation(engine, {"equations": "a = a"}))
        assert "conservation" in out
        assert "variational" not in out
        assert "autodiff" not in out

    def test_lagrangian_without_coordinates_skipped(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("huginn.tools.lean_tool.LeanTool", lambda: _FakeLean())
        out = asyncio.run(
            run_math_validation(engine, {"lagrangian": "L = T - V", "coordinates": []})
        )
        assert "variational" not in out

    def test_non_dict_execution_result_returns_empty(
        self, engine: AutoloopEngine
    ) -> None:
        out = asyncio.run(run_math_validation(engine, "not a dict"))
        assert out == {}

    def test_empty_execution_result_returns_empty(
        self, engine: AutoloopEngine
    ) -> None:
        out = asyncio.run(run_math_validation(engine, {}))
        assert out == {}

    def test_tool_error_recorded_not_raised(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # BourbakiTool 构造抛错 → 记 conservation_error, 不炸
        def _boom():
            raise RuntimeError("init failed")

        monkeypatch.setattr("huginn.tools.bourbaki_tool.BourbakiTool", _boom)
        out = asyncio.run(run_math_validation(engine, {"equations": "a = a"}))
        assert "conservation" not in out
        assert "conservation_error" in out
