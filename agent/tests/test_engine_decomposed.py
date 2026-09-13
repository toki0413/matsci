"""Engine 方法族去 mixin化 — 阶段1(MathValidator)/阶段2(EnginePerceive) 的 TDD 测试."""

from __future__ import annotations

import asyncio

import pytest


# ===== 阶段1: MathValidator =====

def test_no_math_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.math_validation import MathValidator

    assert MathValidator not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 MathValidator 当作基类 —— 去 mixin 阶段1 未完成"
    )


def test_engine_new_holds_math_validator() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.math_validation import MathValidator

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.workspace = "/tmp"
    eng.settings = object()
    eng._query_kb_reference = lambda eq, lag: None
    eng._math_validator = MathValidator(eng)
    assert isinstance(eng._math_validator, MathValidator)
    # 委托路径真实可调用 (不经完整 __init__, __new__ + 手动挂 validator)
    out = asyncio.run(eng._run_math_validation({"equations": "", "lagrangian": ""}))
    assert isinstance(out, dict)


def test_delegation_method_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    # 委托方法保持, 让 engine_reflect.py:133 调用点零改动
    assert hasattr(AutoloopEngine, "_run_math_validation")


async def test_math_validator_run_uses_engine_deps() -> None:
    from huginn.autoloop.math_validation import MathValidator

    class _StubEngine:
        workspace = "/tmp"
        settings = object()

        def _query_kb_reference(self, equations, lagrangian):
            if lagrangian:
                return {"src": "kb"}
            return None

    v = MathValidator(_StubEngine())
    # 空 equations/lagrangian → 无守恒/变分工具分支, 返回空 dict 不抛
    out = await v.run({"equations": "", "lagrangian": "", "coordinates": []})
    assert isinstance(out, dict)
    # 有 lagrangian → 查 KB 并写 reference_principles; 缺工具只记 *_error 不抛
    out2 = await v.run({"equations": "", "lagrangian": "L=T-V", "coordinates": []})
    assert isinstance(out2, dict)
    assert out2.get("reference_principles") == {"src": "kb"}


# ===== 阶段2: EnginePerceive =====

def test_no_perceive_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_perceive import EnginePerceive

    assert EnginePerceive not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 EnginePerceive 当作基类 —— 去 mixin 阶段2 未完成"
    )


def test_perceive_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    # 委托方法保持 → plan_check/engine_observe/cognitive_loop 等调用点零改动
    for name in (
        "_build_kb_text",
        "_build_kg_text",
        "_build_memory_text",
        "_build_pm_text",
        "_build_metacog_block",
        "_get_kb",
        "_get_persona_manager",
        "_extract_search_query",
        "_maybe_expire_inbox",
        "_perceive",
    ):
        assert hasattr(AutoloopEngine, name), f"EnginePerceive 委托方法 {name} 缺失"


def test_engine_new_holds_perceiver() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_perceive import EnginePerceive

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.workspace = "/tmp"
    eng._kb = None
    eng._perception = None
    eng._persona_manager = None
    eng._engine_perceiver = EnginePerceive(eng)
    # 委托到 perceiver: _get_kb 空 → None (不抛)
    assert eng._get_kb() is None


def test_engine_perceiver_forwards_state() -> None:
    """属性转发: perceiver 读写 self.workspace/_kb 落到 engine, 缓存共享."""
    from huginn.autoloop.engine_perceive import EnginePerceive

    class _StubEngine:
        workspace = "/tmp"
        _kb = None
        _perception = None
        _persona_manager = None

    eng = _StubEngine()
    p = EnginePerceive(eng)
    # _extract_search_query 读 self._objective → 空 objective → 兜底 JSON dump
    q = p._extract_search_query({"objective": ""})
    assert isinstance(q, str)
    # 读转发: perceiver 上访问 engine 字段
    assert p.workspace == eng.workspace