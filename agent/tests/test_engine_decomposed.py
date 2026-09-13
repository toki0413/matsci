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


# ===== 阶段3: VisualInspect =====

def test_no_visual_inspect_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.visual_inspect import VisualInspect

    assert VisualInspect not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 VisualInspect 当作基类 —— 去 mixin 阶段3 未完成"
    )


def test_visual_inspect_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "_execute_visual_inspect",
        "_measure_nearest_primitive",
        "_annotate_visual_features",
        "_extract_text_visual_features",
        "_compare_visual_data",
        "_call_image_analysis_tool",
        "_pick_image_action",
    ):
        assert hasattr(AutoloopEngine, name), f"VisualInspect 委托方法 {name} 缺失"


def test_engine_new_holds_visual_inspector() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.signals import EngineSignals
    from huginn.autoloop.visual_inspect import VisualInspect

    eng = AutoloopEngine.__new__(AutoloopEngine)
    # 类级 SignalBridge property(_set) 会把被写字段转发到 self.signals, __new__ 前需挂
    eng.signals = EngineSignals()
    eng._last_visual_context = ""
    eng._visual_base64 = ""
    eng._last_visual_base64 = None
    eng._visual_inspector = VisualInspect(eng)
    assert isinstance(eng._visual_inspector, VisualInspect)


async def test_engine_visual_inspector_reads_engine_field() -> None:
    """只读转发: inspector 从引擎读到 _last_visual_context(空)", 委托方法等价可调."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.signals import EngineSignals
    from huginn.autoloop.visual_inspect import VisualInspect

    eng = AutoloopEngine.__new__(AutoloopEngine)
    # 类级 SignalBridge property(_set) 会把被写字段转发到 self.signals, __new__ 前需挂
    eng.signals = EngineSignals()
    eng._last_visual_context = ""
    eng._visual_base64 = ""
    eng._last_visual_base64 = None
    eng._visual_inspector = VisualInspect(eng)
    # 无视觉数据 → 走 no-visual-data 分支, 委托方法真执行
    res = await eng._execute_visual_inspect("zoom into region [0,0]-[10,10]", {})
    assert res["success"] is False
    assert "No visual data" in res["error"]


# ===== 阶段4: EngineAct =====

def test_no_act_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_act import EngineAct

    assert EngineAct not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 EngineAct 当作基类 —— 去 mixin 阶段4 未完成"
    )


def test_act_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "_plan",
        "_execute",
        "_execute_coder",
        "_execute_workflow",
        "_execute_dynamic_workflow",
        "_execute_skill",
        "_llm_chat",
    ):
        assert hasattr(AutoloopEngine, name), f"EngineAct 委托方法 {name} 缺失"


def test_engine_new_holds_actor() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_act import EngineAct
    from huginn.autoloop.signals import EngineSignals

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.signals = EngineSignals()
    eng._engine_actor = EngineAct(eng)
    assert isinstance(eng._engine_actor, EngineAct)


async def test_engine_actor_delegates_llm_chat() -> None:
    """全属性转发: actor 转发读引擎字段 + 委托方法真可调(经 stub 包一层)."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_act import EngineAct
    from huginn.autoloop.signals import EngineSignals

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.signals = EngineSignals()
    eng.verification_model = None
    eng._engine_actor = EngineAct(eng)
    # 委托方法确实绑定到引擎 (可调用, 内部经转发访问引擎字段)
    assert hasattr(eng, "_is_deterministic_numeric")
    assert eng._is_deterministic_numeric("compute peak of 3*x+1") is True
    assert eng._is_deterministic_numeric("just observe") is False


# ===== 阶段5: EngineControl =====

def test_no_control_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_control import EngineControl

    assert EngineControl not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 EngineControl 当作基类 —— 去 mixin 阶段5 未完成"
    )


def test_control_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "_check_gate",
        "_check_budget",
        "_maybe_clarify",
        "_maybe_save_engine_state",
        "_dispatch_stage_event",
        "_get_plan_store",
        "_plan_missing_executable",
        "stop",
    ):
        assert hasattr(AutoloopEngine, name), f"EngineControl 委托方法 {name} 缺失"


def test_engine_new_holds_controller() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_control import EngineControl
    from huginn.autoloop.signals import EngineSignals

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.signals = EngineSignals()
    eng._engine_controller = EngineControl(eng)
    assert isinstance(eng._engine_controller, EngineControl)


def test_engine_controller_plan_missing_executable() -> None:
    """全属性转发: controller 方法真可调, 读引擎字段走转发."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_control import EngineControl
    from huginn.autoloop.signals import EngineSignals

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.signals = EngineSignals()
    eng._engine_controller = EngineControl(eng)
    assert eng._plan_missing_executable({"mode": "", "description": ""}) is True
    # description 带 import 计算标记 → 判定有可执行片段
    assert eng._plan_missing_executable(
        {"mode": "coder", "description": "import numpy as np; print(x)"}
    ) is False


# ===== 阶段6: PlanCheck =====

def test_no_plan_check_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.plan_check import PlanCheck

    assert PlanCheck not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 PlanCheck 当作基类 —— 去 mixin 阶段6 未完成"
    )


def test_plan_check_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "_build_plan_prompt",
        "_parse_plan",
        "_override_plan_mode",
        "_plan_check_and_refine",
        "_plan_check_tier",
        "_plan_check_scene_tag",
        "_refine_plan",
        "_load_plan_check_patterns",
        "_save_plan_check_patterns",
        "_build_subgoal_block",
    ):
        assert hasattr(AutoloopEngine, name), f"PlanCheck 委托方法 {name} 缺失"


def test_engine_new_holds_plan_checker() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.plan_check import PlanCheck

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._plan_checker = PlanCheck(eng)
    assert isinstance(eng._plan_checker, PlanCheck)


def test_plan_checker_parse_plan_check() -> None:
    """全属性转发: checker 方法真可调, no-json 跳过不抛."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.plan_check import PlanCheck

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._plan_checker = PlanCheck(eng)
    # 无 JSON → is_valid=True (跳过, 不阻塞)
    out = eng._parse_plan_check("no json here")
    assert out.get("is_valid") is True


def test_plan_checker_override_plan_mode_forwards_state() -> None:
    """读引擎字段走转发: 连败 5 次 → coder 被硬路由成 explore."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.plan_check import PlanCheck
    from huginn.autoloop.signals import EngineSignals

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._plan_checker = PlanCheck(eng)
    eng.signals = EngineSignals()  # _consecutive_failures/_last_surprise 是信号桥字段
    eng._consecutive_failures = 0
    eng._last_surprise = 0.0
    eng._current_hyp_id_for_plan = None
    plan = {"mode": "coder", "description": "fix bug"}
    out = eng._override_plan_mode(dict(plan))
    assert out["mode"] == "coder"  # 无信号 → 不覆盖
    eng._consecutive_failures = 5
    out = eng._override_plan_mode(dict(plan))
    assert out["mode"] == "explore"  # 连败 → 强制 explore


# ===== 阶段7: EngineObserve =====

def test_no_observe_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_observe import EngineObserve

    assert EngineObserve not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 EngineObserve 当作基类 —— 去 mixin 阶段7 未完成"
    )


def test_observe_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "_build_hypothesis_prompt",
        "_build_curiosity_block",
        "_apply_block_patches",
        "_trim_to_budget",
        "_get_metacog_auditor",
        "_metacog_check_completion",
        "trigger_isomorphic_anomaly_hypothesis",
        "_extract_lucid_prereqs",
        "_persona_system_prompt",
    ):
        assert hasattr(AutoloopEngine, name), f"EngineObserve 委托方法 {name} 缺失"


def test_engine_new_holds_observer() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_observe import EngineObserve

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._engine_observer = EngineObserve(eng)
    assert isinstance(eng._engine_observer, EngineObserve)


def test_engine_observe_class_constants_bridged() -> None:
    """类常量桥: AutoloopEngine 保留 _MATH_DEPTH_PROMPT_BLOCK 等类级访问."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_observe import EngineObserve

    for name in (
        "_PROMPT_BUDGET",
        "_PROMPT_BUDGET_BY_PHASE",
        "_MATH_DEPTH_PROMPT_BLOCK",
        "_IMAGINATION_PROMPT_BLOCK",
    ):
        assert getattr(AutoloopEngine, name) == getattr(EngineObserve, name), (
            f"常量桥 {name} 不一致"
        )


def test_observe_static_and_instance_delegation() -> None:
    """委托零参-静态方法真可调: _files_jaccard 纯函数经静态委托."""

    def _engine() -> "AutoloopEngine":
        from huginn.autoloop.engine import AutoloopEngine
        from huginn.autoloop.engine_observe import EngineObserve

        eng = AutoloopEngine.__new__(AutoloopEngine)
        eng._engine_observer = EngineObserve(eng)
        return eng

    eng = _engine()
    assert eng._files_jaccard(["a.py", "b.py"], ["a.py"]) == 1 / 2


# ===== 阶段8: EngineReflect =====

def test_no_reflect_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_reflect import EngineReflect

    assert EngineReflect not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 EngineReflect 当作基类 —— 去 mixin 阶段8 未完成"
    )


def test_reflect_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "_validate",
        "_learn",
        "_report",
        "_literature_comparison",
        "_generative_verify",
        "_compute_surprise",
        "_query_kb_reference",
        "_blind_spot_pass",
        "_feynman_learn",
        "_extract_text",
    ):
        assert hasattr(AutoloopEngine, name), f"EngineReflect 委托方法 {name} 缺失"


def test_engine_new_holds_reflector() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_reflect import EngineReflect

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._engine_reflector = EngineReflect(eng)
    assert isinstance(eng._engine_reflector, EngineReflect)


def test_engine_reflect_class_constants_bridged() -> None:
    """类常量桥: AutoloopEngine 保留 _FEYNMAN_PROMPT 等类级访问."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_reflect import EngineReflect

    for name in ("_FEYNMAN_PROMPT", "_BLIND_SPOT_PROMPT", "_NEXT_STEP_ADVISOR_PROMPT"):
        assert getattr(AutoloopEngine, name) == getattr(EngineReflect, name), (
            f"常量桥 {name} 不一致"
        )


def test_reflect_static_and_instance_delegation() -> None:
    """委托-纯函数真可调: _extract_text static 经静态委托."""

    def _engine() -> "AutoloopEngine":
        from huginn.autoloop.engine import AutoloopEngine
        from huginn.autoloop.engine_reflect import EngineReflect

        eng = AutoloopEngine.__new__(AutoloopEngine)
        eng._engine_reflector = EngineReflect(eng)
        return eng

    eng = _engine()
    assert eng._extract_text({"result": "hello world"}) == "hello world"
    # 全属性转发: reflector 方法可读引擎字段 (经 stub engine)
    assert isinstance(eng._engine_reflector._extract_text({"result": "x"}), str)