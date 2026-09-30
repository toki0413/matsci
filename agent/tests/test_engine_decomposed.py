"""Engine 方法族去 mixin化 — 阶段1(MathValidator)/阶段2(EnginePerceive) 的 TDD 测试."""

from __future__ import annotations

import asyncio

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


def test_engine_new_holds_perceiver(monkeypatch) -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.engine_perceive import EnginePerceive

    # 感知器委托下 KB 不可用 → None: EnginePerceive._get_kb 会惰性建真实 KB
    # (CI 装了 chromadb 会成功建库, 甚至因初始化阻塞). 这里隔离该依赖, 让
    # get_knowledge_base 返回 None, 以验证"KB 不可用时感知器不抛、返回 None"契约.
    monkeypatch.setattr(
        "huginn.knowledge.store.get_knowledge_base", lambda *a, **k: None
    )
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
    # 中文数值目标同样要认 (autoloop 的真实 objective 多为中文), 否则内建
    # probe 执行路径永不触发 -> execute 落到 explore 空转
    assert eng._is_deterministic_numeric(
        "训练小 MLP 拟合 N 个点值约束, 在 128 点留出集统计均方违规, 求 N_c"
    ) is True
    assert eng._is_deterministic_numeric("复核这个结论是否正确") is False


def test_build_codelab_focus_unfreezes_author_input() -> None:
    """v11 反冻结: 作者提示的 focus 随'当前假设 + 本轮步骤'变化, 不再逐轮恒同.

    run47 实测作者提示 prompt_len 恒 5379 → 同一问题反复问、执行输出恒同、零新证据.
    focus 取随迭代演进的可变文本, 让提示解冻; 无可用文本时返空串(退回旧路径).
    """
    from huginn.autoloop.engine_act import EngineAct

    class _StubEngine:
        _objective = "恒定研究目标"
        _last_hypothesis = ""

    eng = _StubEngine()
    act = EngineAct(eng)
    # 无假设无步骤 → 空 focus (向后兼容, 行为退回旧路径)
    assert act._build_codelab_focus("") == ""
    # 有假设/步骤 → focus 非空, 且随内容演进 (解冻)
    eng._last_hypothesis = "H1: 约束密度升高会降低容量 N_c"
    f1 = act._build_codelab_focus("扫描宽度 w=10")
    eng._last_hypothesis = "H2: 改用 fat 约束族后 N_c 不可达"
    f2 = act._build_codelab_focus("扫描宽度 w=20")
    assert f1 and f2 and f1 != f2
    assert "H1" in f1 and "w=10" in f1


def test_build_author_prompt_focus_changes_prompt() -> None:
    """focus 非空 → 作者提示随之变化, 恒定 goal 不再产出逐字节相同的 prompt."""
    from huginn.research.code_lab import build_author_prompt

    goal = "恒定研究目标"
    p_plain = build_author_prompt(goal)
    p_a = build_author_prompt(goal, focus="假设 A")
    p_b = build_author_prompt(goal, focus="假设 B")
    assert "假设 A" in p_a
    assert p_a != p_plain and p_a != p_b


def test_codelab_focus_injects_hard_variation_directive() -> None:
    """v12: 重复执行越阈值后, 强制变异令必须进**实验作者**提示(而非只进假设提示).

    run50 实测: 软 pivot hint 只进 _speculator_hint(假设生成提示), 写实验的
    build_author_prompt 不读它 → streak 1-4 指纹恒同, 循环撞收敛提前离场.
    这里验证 _force_exec_variation 置位后, focus 带上硬指令与上一轮真实结果.
    """
    from huginn.autoloop.engine_act import EngineAct

    class _StubEngine:
        _objective = "恒定研究目标"
        _last_hypothesis = "H1: N_c 随 w 饱和"

    eng = _StubEngine()
    act = EngineAct(eng)
    # 未置位 → 无硬指令 (不误伤正常探索)
    assert "强制变异" not in act._build_codelab_focus("扫描 w=10")
    # 置位 → 硬指令 + 上一轮真实结果一并注入
    eng._force_exec_variation = True
    eng._repeat_exec_last_result = '{"o": {"Nc": [2, 36, 65]}}'
    focus = act._build_codelab_focus("扫描 w=10")
    assert "强制变异" in focus
    assert "family" in focus
    assert "Nc" in focus  # 上一轮真实结果回灌, 逼产出不同数值


# ===== 重复执行指纹口径 (v13) =====


def test_exec_fingerprint_same_code_is_repeat_despite_value_jitter() -> None:
    """同代码重跑 = 重复, 即使两轮 objectives 数值不同 (浮点抖动) 也要认出.

    旧口径对 objectives/summary 做内容哈希: 沙箱虽固定 seed=0, 但浮点归约序/线程
    调度可抖动 → 同一段代码两次跑出末位不同的数 → 漏判 (run53 实测 55 次执行零命中).
    新口径对**代码结构**做指纹, 与数值无关.
    """
    from huginn.autoloop.engine_reflect import _exec_fingerprint

    code = "def run(cfg):\n    return {'Nc': 3}\n"
    # 注释/格式/空行差异不进结构
    code_reformatted = "# 换个注释\ndef run(cfg):\n\n    return {'Nc': 3}  # 尾注\n"
    a = {"mode": "code_lab", "script": code, "objectives": {"Nc_rigid": [3, 3]}}
    b = {"mode": "code_lab", "script": code_reformatted,
         "objectives": {"Nc_rigid": [3.0000001, 2.9999998]}}  # 数值抖动
    fa, _ = _exec_fingerprint(a)
    fb, _ = _exec_fingerprint(b)
    assert fa and fa == fb, "同代码重跑(仅注释/格式/末位数值不同)应判为重复"


def test_exec_fingerprint_scan_param_change_is_not_repeat() -> None:
    """正常参数扫描必须被认作**新实验**, 否则每轮扫描都被强制变异, 破坏探索."""
    from huginn.autoloop.engine_reflect import _exec_fingerprint

    a = {"mode": "code_lab", "script": "W = [2, 4, 6]\n", "objectives": {"n": 1}}
    b = {"mode": "code_lab", "script": "W = [2, 4, 8]\n", "objectives": {"n": 2}}
    fa, _ = _exec_fingerprint(a)
    fb, _ = _exec_fingerprint(b)
    assert fa and fb and fa != fb, "扫描参数改变 → 指纹必须变 (数字常量保留)"


def test_exec_fingerprint_falls_back_to_content_without_script() -> None:
    """无 script (explore/probe 等) 时退回旧口径: 内容哈希, 行为不变."""
    from huginn.autoloop.engine_reflect import _exec_fingerprint

    same1 = {"mode": "explore", "objectives": {"x": 1}, "summary": {"y": 2}}
    same2 = {"mode": "explore", "objectives": {"x": 1}, "summary": {"y": 2}}
    diff = {"mode": "explore", "objectives": {"x": 9}, "summary": {"y": 2}}
    f1, src1 = _exec_fingerprint(same1)
    f2, _ = _exec_fingerprint(same2)
    f3, _ = _exec_fingerprint(diff)
    assert f1 == f2 and f1 != f3
    assert '"x": 1' in src1  # 人读摘要仍来自 objectives/summary


def test_normalize_script_for_fp_survives_syntax_error() -> None:
    """残缺代码不能抛异常 (指纹失败会中止重复检测), 退化为去空行原文."""
    from huginn.autoloop.engine_reflect import _normalize_script_for_fp

    broken = "def run(cfg:\n    return 1\n"
    out = _normalize_script_for_fp(broken)
    assert out and "return 1" in out


def test_exec_fingerprint_empty_when_no_evidence() -> None:
    """既无代码也无 objectives/summary → 空指纹 (不参与重复/收敛记账)."""
    from huginn.autoloop.engine_reflect import _exec_fingerprint

    assert _exec_fingerprint({"mode": "explore"})[0] == ""
    assert _exec_fingerprint(None)[0] == ""


# ===== 非有限数值 (inf/nan) 不计为证据 (v13) =====


def test_non_finite_objective_keys_flags_inf_nan() -> None:
    """inf/nan 键被点名; 全有限/无 objectives 返回空."""
    from huginn.autoloop.engine_reflect import _non_finite_objective_keys

    res = {"mode": "code_lab", "objectives": {"a": 1.0, "b": float("inf"),
                                              "c": float("nan")}}
    assert _non_finite_objective_keys(res) == ["b", "c"]
    assert _non_finite_objective_keys(
        {"mode": "code_lab", "objectives": {"a": 1.0}}
    ) == []
    assert _non_finite_objective_keys({"mode": "code_lab"}) == []
    assert _non_finite_objective_keys(None) == []


def test_is_code_lab_solved_rejects_non_finite_objectives() -> None:
    """run53 根因: objectives 非空但含 ∞ 时曾判 solved → ∞ 进报告.

    命题口径要求"所有报告数值必须是有限数"; inf/nan 不是测到的数, 不能当证据.
    """
    from huginn.autoloop.engine_reflect import EngineReflect

    class _StubEngine:
        pass

    ref = EngineReflect(_StubEngine())
    ok = {"mode": "code_lab", "success": True, "objectives": {"Nc": 3.0}}
    bad = {"mode": "code_lab", "success": True, "objectives": {"Nc": float("inf")}}
    assert ref._is_code_lab_solved(ok) is True
    assert ref._is_code_lab_solved(bad) is False
    # 失败/空 objectives 仍按旧口径拒绝
    assert ref._is_code_lab_solved(
        {"mode": "code_lab", "success": False, "objectives": {"Nc": 3.0}}
    ) is False
    assert ref._is_code_lab_solved(
        {"mode": "code_lab", "success": True, "objectives": {}}
    ) is False


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

    def _engine():  # -> AutoloopEngine (函数内导入, 见下)
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

    def _engine():  # -> AutoloopEngine (函数内导入, 见下)
        from huginn.autoloop.engine import AutoloopEngine
        from huginn.autoloop.engine_reflect import EngineReflect

        eng = AutoloopEngine.__new__(AutoloopEngine)
        eng._engine_reflector = EngineReflect(eng)
        return eng

    eng = _engine()
    assert eng._extract_text({"result": "hello world"}) == "hello world"
    # 全属性转发: reflector 方法可读引擎字段 (经 stub engine)
    assert isinstance(eng._engine_reflector._extract_text({"result": "x"}), str)


# ===== 阶段9: HypothesisLoop =====

def test_no_hypothesis_mixin_in_bases() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.hypothesis_loop import HypothesisLoop

    assert HypothesisLoop not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 HypothesisLoop 当作基类 —— 去 mixin 阶段9 未完成"
    )


def test_hypothesis_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "_hypothesize",
        "_hypothesize_via_branch_incubator",
        "_classify_failure",
        "_should_imaginate",
        "_conjecture_hint",
        "_symreg_hint",
        "_evaluate_informativeness",
        "_sync_simplicials_to_kg",
        "_pick_hypothesis_persona",
    ):
        assert hasattr(AutoloopEngine, name), f"HypothesisLoop 委托方法 {name} 缺失"


def test_engine_new_holds_hypothesis_loop() -> None:
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.hypothesis_loop import HypothesisLoop

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._hypothesis_loop = HypothesisLoop(eng)
    assert isinstance(eng._hypothesis_loop, HypothesisLoop)


def test_hypothesis_loop_sync_simplicials_forwards() -> None:
    """全属性转发: 方法真可调, 读引擎字段走转发 (无 kg → 静默降级不抛)."""
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.hypothesis_loop import HypothesisGraph, HypothesisLoop

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.hypothesis_graph = HypothesisGraph()
    eng.kg = None
    eng._hypothesis_loop = HypothesisLoop(eng)
    eng._sync_simplicials_to_kg()  # 无 kg → 不抛
    assert isinstance(eng.hypothesis_graph, HypothesisGraph)


# ===== 阶段10: CognitiveRunner =====

def test_no_cognitive_mixin_in_bases() -> None:
    from huginn.autoloop.cognitive_loop import CognitiveRunner
    from huginn.autoloop.engine import AutoloopEngine

    assert CognitiveRunner not in AutoloopEngine.__bases__, (
        "AutoloopEngine 仍把 CognitiveRunner 当作基类 —— 去 mixin 阶段10 未完成"
    )


def test_cognitive_delegation_methods_still_present() -> None:
    from huginn.autoloop.engine import AutoloopEngine

    for name in (
        "run_cognitive",
        "_await_human_decision_via_inbox",
        "_run_phase",
        "_run_phase_async",
        "_darwin_ratchet_check",
        "_classify_stall",
        "_emit_campaign",
        "_prepare_run",
        "_decide_next_action_llm",
        "_build_decider_prompt",
        "_is_action_legal",
        "_finalize_run",
        "_rollback_on_execute_failure",
    ):
        assert hasattr(AutoloopEngine, name), f"CognitiveRunner 委托方法 {name} 缺失"


def test_cognitive_staticmethods_bridged() -> None:
    """两个 @staticmethod 走类常量桥, 保持 AutoloopEngine._X 类级 unbound 访问."""
    from huginn.autoloop.engine import AutoloopEngine

    # 类级调用不传 self → 必须是 staticmethod 而非实例委托.
    assert AutoloopEngine._extract_timeseries({"status": "ok"}) == []
    assert isinstance(AutoloopEngine._snapshot_provenance_version(), int)


def test_engine_new_holds_cognitive_runner() -> None:
    from huginn.autoloop.cognitive_loop import CognitiveRunner
    from huginn.autoloop.engine import AutoloopEngine

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._cognitive_runner = CognitiveRunner(eng)
    assert isinstance(eng._cognitive_runner, CognitiveRunner)


def test_cognitive_runner_run_cognitive_signature_defaults() -> None:
    """run_cognitive 委托保留 max_refines 参数与默认 8 (test_loop_inspired 依赖)."""
    import inspect

    from huginn.autoloop.engine import AutoloopEngine

    sig = inspect.signature(AutoloopEngine.run_cognitive)
    assert "max_refines" in sig.parameters
    assert sig.parameters["max_refines"].default == 8


def test_cognitive_runner_state_forwards_via_getattr() -> None:
    """全属性转发: 读/写引擎状态字段走转发; 协作方法可调 (空图降级不抛)."""
    from huginn.autoloop.cognitive_loop import CognitiveRunner
    from huginn.autoloop.engine import AutoloopEngine
    from huginn.autoloop.hypothesis_loop import HypothesisGraph
    from huginn.autoloop.signals import EngineSignals

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.signals = EngineSignals()
    eng._iteration = 3
    eng._should_stop = False
    eng.hypothesis_graph = HypothesisGraph()
    eng._cognitive_runner = CognitiveRunner(eng)

    # 字段读转发到引擎
    assert eng._cognitive_runner._iteration == 3
    # 协作方法真可调 (空图 → 提前 return, 不抛)
    eng._cognitive_runner._darwin_ratchet_check()
    assert eng._cognitive_runner._should_stop is False
