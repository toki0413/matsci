"""Tests for ExtractTool — math-anything 提取器包装.

覆盖 5 个行为路径: ImportError 回退 / 透传 file_paths / 结构化返回 /
异常吞掉 / constraints 透传.
用 monkeypatch.setitem(sys.modules, "math_anything", ...) 注入 fake 模块.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from huginn.tools.extract_tool import ExtractTool, ExtractToolInput


class _FakeExtractResult:
    """模拟 math_anything 的 extract() 返回值."""

    def __init__(self, schema=None, warnings=None, errors=None, success=True):
        self.schema = schema if schema is not None else {}
        self.warnings = warnings or []
        self.errors = errors or []
        self.success = success


def _make_fake_module(extract_result=None, extract_raises=None):
    """造一个 fake math_anything 模块, MathAnything.extract 按预设返回或抛异常.

    返回 (module, calls) — calls 记录 extract 的调用参数, 供断言.
    """
    calls: list[dict] = []

    class _FakeMathAnything:
        def __init__(self):
            pass

        def extract(self, engine, file_paths):
            calls.append({"engine": engine, "file_paths": file_paths})
            if extract_raises is not None:
                raise extract_raises
            return extract_result

    module = SimpleNamespace(MathAnything=_FakeMathAnything)
    module._calls = calls  # 暴露给测试断言
    return module


# ════════════════════════════════════════════════════════════════════
# ExtractTool
# ════════════════════════════════════════════════════════════════════


async def test_extract_missing_math_anything_falls_back(monkeypatch):
    """math_anything 不可用 (sys.modules=None) → ImportError 回退, 返回 note."""
    # sys.modules[name]=None 会让 import name 抛 ImportError
    monkeypatch.setitem(sys.modules, "math_anything", None)
    tool = ExtractTool()
    args = ExtractToolInput(engine="vasp", file_paths={"incar": "INCAR"})
    result = await tool.call(args, context=None)
    assert result.success is True
    assert result.data["engine"] == "vasp"
    assert result.data["files"] == {"incar": "INCAR"}
    assert "math-anything not installed" in result.data["note"]


async def test_extract_passes_file_paths(monkeypatch):
    """fake MathAnything.extract 被调用时, engine + file_paths 透传."""
    fake_mod = _make_fake_module(extract_result=_FakeExtractResult())
    monkeypatch.setitem(sys.modules, "math_anything", fake_mod)
    tool = ExtractTool()
    args = ExtractToolInput(
        engine="lammps", file_paths={"data": "in.data", "input": "in.lammps"}
    )
    result = await tool.call(args, context=None)
    assert result.success is True
    assert fake_mod._calls[0]["engine"] == "lammps"
    assert fake_mod._calls[0]["file_paths"] == {"data": "in.data", "input": "in.lammps"}


async def test_extract_returns_structure(monkeypatch):
    """fake extract 返回带 mathematical_structure 的 schema → 工具原样返回."""
    schema = {
        "mathematical_structure": {"equations": ["E_kin = 0.5 m v^2"]},
        "symbolic_constraints": [],
        "approximations": ["Born-Oppenheimer"],
    }
    fake_mod = _make_fake_module(
        extract_result=_FakeExtractResult(schema=schema, warnings=["minor"])
    )
    monkeypatch.setitem(sys.modules, "math_anything", fake_mod)
    tool = ExtractTool()
    args = ExtractToolInput(engine="vasp", file_paths={"incar": "INCAR"})
    result = await tool.call(args, context=None)
    assert result.success is True
    assert result.data["mathematical_structure"] == {
        "equations": ["E_kin = 0.5 m v^2"]
    }
    assert result.data["approximations"] == ["Born-Oppenheimer"]
    assert result.data["warnings"] == ["minor"]


async def test_extract_exception_swallowed(monkeypatch):
    """fake extract 抛非 ImportError 异常 → 外层 except 吞掉, success=False."""
    fake_mod = _make_fake_module(extract_raises=ValueError("bad input"))
    monkeypatch.setitem(sys.modules, "math_anything", fake_mod)
    tool = ExtractTool()
    args = ExtractToolInput(engine="vasp", file_paths={"incar": "INCAR"})
    result = await tool.call(args, context=None)
    assert result.success is False
    assert "Extraction failed" in result.error
    assert "bad input" in result.error


async def test_extract_with_constraints(monkeypatch):
    """fake schema 含 symbolic_constraints → 工具透传到 constraints 字段."""
    schema = {
        "mathematical_structure": {},
        "symbolic_constraints": ["ENCUT >= 400", "EDIFF < 1e-4"],
        "approximations": [],
    }
    fake_mod = _make_fake_module(extract_result=_FakeExtractResult(schema=schema))
    monkeypatch.setitem(sys.modules, "math_anything", fake_mod)
    tool = ExtractTool()
    args = ExtractToolInput(engine="vasp", file_paths={"incar": "INCAR"})
    result = await tool.call(args, context=None)
    assert result.success is True
    assert result.data["constraints"] == ["ENCUT >= 400", "EDIFF < 1e-4"]
