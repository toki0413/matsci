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


def test_build_author_prompt_timeout_hint_reduces_compute() -> None:
    """超时失败要回灌**降算力**指令, 不能套用 NameError/形状那套对症提示.

    run59/60 实测: 三组都被 900s 超时饿死, 报错里没有语法线索, 但提示仍讲
    NameError → 书生继续产出同样重的扫描. 这里锁定"超时 → 削减计算量".
    """
    from huginn.research.code_lab import build_author_prompt

    p = build_author_prompt("恒定目标", repair_hint="code_lab 执行超时 (> 900.0s)")
    assert "削减计算量" in p
    assert "NameError" not in p  # 不再误导书生去查语法


def test_build_author_prompt_bug_hint_keeps_syntax_advice() -> None:
    """非超时失败仍走原"对症改 bug"分支 (回归保护)."""
    from huginn.research.code_lab import build_author_prompt

    p = build_author_prompt(
        "恒定目标", repair_hint="NameError: name 'foo' is not defined"
    )
    assert "NameError" in p
    assert "削减计算量" not in p


def test_codelab_focus_injects_hard_variation_directive() -> None:
    """v12 / B1: 重复执行越阈值后, 强制变异令必须进**实验作者**提示(而非只进假设提示).

    run50 实测: 软 pivot hint 只进 _speculator_hint(假设生成提示), 写实验的
    build_author_prompt 不读它 → streak 1-4 指纹恒同, 循环撞收敛提前离场.
    B1 后无独立标志状态机 —— focus 直接按 _repeat_exec_streak 现算硬指令.
    """
    from huginn.autoloop.engine_act import EngineAct
    from huginn.autoloop.engine_reflect import _REPEAT_HARD_STREAK

    class _StubEngine:
        _objective = "恒定研究目标"
        _last_hypothesis = "H1: N_c 随 w 饱和"

    eng = _StubEngine()
    act = EngineAct(eng)
    # 未越阈值 → 无硬指令 (不误伤正常探索)
    assert "强制变异" not in act._build_codelab_focus("扫描 w=10")
    # 越阈值 → 硬指令 + 上一轮真实结果一并注入
    eng._repeat_exec_streak = _REPEAT_HARD_STREAK
    eng._prev_exec_fp_src = '{"o": {"Nc": [2, 36, 65]}}'
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


# ===== 定点验证: 检测 → 软提示 → 硬指令 全链 (v13) =====


def _repeat_stub() -> object:
    """带真实字段的引擎替身: EngineReflect/EngineAct 都按转发取属性."""
    class _StubEngine:
        _objective = "恒定研究目标"
        _last_hypothesis = "H1: N_c 随 w 饱和"
        _speculator_hint = ""

    return _StubEngine()


def _lab_result(script: str, nc: float = 4.0) -> dict:
    return {
        "mode": "code_lab", "success": True, "script": script,
        "objectives": {"Nc": nc}, "summary": {"note": "trace"},
    }


def test_repeat_chain_lights_hard_directive_after_threshold() -> None:
    """定点验证: 同代码重跑 → streak 累积 → 越阈值 → 作者提示带强制变异令.

    B1 后无独立标志状态机: `_build_codelab_focus` 直接按 `_repeat_exec_streak`
    现算硬指令. 这是 pivot 硬约束的端到端(harness 内)证据: 不经 LLM, 直接驱动
    真实代码路径, 补上 run53/run54 都没能自然触发的那一环.
    """
    from huginn.autoloop.engine_act import EngineAct
    from huginn.autoloop.engine_reflect import EngineReflect, _REPEAT_HARD_STREAK

    eng = _repeat_stub()
    ref = EngineReflect(eng)
    act = EngineAct(eng)
    script = "W = [2, 4, 6]\nres = {}\n"

    # 第 1 轮: 新实验 → 不记重复
    r1: dict = {}
    ref._detect_repeat_execution(_lab_result(script), r1)
    assert not r1.get("repeat_execution")
    assert getattr(eng, "_repeat_exec_streak", 0) == 0

    # 第 2..N 轮: 同代码 → streak 递增
    for _ in range(_REPEAT_HARD_STREAK):
        rn: dict = {}
        ref._detect_repeat_execution(_lab_result(script), rn)
        assert rn["repeat_execution"] is True

    assert eng._repeat_exec_streak == _REPEAT_HARD_STREAK
    # 硬指令真的进得了**实验作者**提示 (run50 的病根是软提示进不去)
    focus = act._build_codelab_focus("扫描 w=10")
    assert "强制变异" in focus
    assert "Nc" in focus  # 上一轮真实结果回灌


def test_control_trace_emits_uniform_schema() -> None:
    """控制面观测契约: `_control_trace` 必须发 `campaign.control_trace` 且 schema 统一.

    触发率统计依赖该 schema (name/iteration/evidence/action/advisory). 这里锁定它,
    防止后续机制各写各的字段导致"跑若干轮后按数据删减"无法聚合.
    """
    from huginn.autoloop.cognitive_loop import CognitiveRunner

    captured: list[tuple[str, dict]] = []

    class _Stub:
        _iteration = 7

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            captured.append((event_type, data))

    CognitiveRunner._control_trace(
        _Stub(), "failure_budget", "type=tool_error count=20 limit=20",
        action="stop",
    )
    assert captured[0][0] == "campaign.control_trace"
    _d = captured[0][1]
    assert _d["name"] == "failure_budget"
    assert _d["iteration"] == 7
    assert _d["action"] == "stop"
    assert _d["evidence"] == "type=tool_error count=20 limit=20"
    assert "advisory" in _d


def test_reflect_control_trace_forwards_uniform_schema() -> None:
    """engine_reflect 侧 trace 与 cognitive_loop 同 schema (B4 effort_floor 用它)."""
    from huginn.autoloop.engine_reflect import EngineReflect

    captured: list[tuple[str, dict]] = []

    class _StubEngine:
        _iteration = 3

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            captured.append((event_type, data))

    EngineReflect(_StubEngine())._emit_control_trace(
        "effort_floor", "deficits=x", action="advisory_hint"
    )
    assert captured[0][0] == "campaign.control_trace"
    assert captured[0][1]["name"] == "effort_floor"
    assert captured[0][1]["action"] == "advisory_hint"


def test_control_trace_emits_telemetry_event() -> None:
    """观测面: control_trace 除 WARNING/campaign 外还落 OTel event → Langfuse 可查."""
    from huginn.autoloop.cognitive_loop import CognitiveRunner
    from huginn.telemetry import TelemetryCollector, set_telemetry_collector

    collector = TelemetryCollector()
    set_telemetry_collector(collector)

    class _Stub:
        _iteration = 5

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            pass

    try:
        CognitiveRunner._control_trace(
            _Stub(), "goal_judge", "achieved=False", action="advisory_hint"
        )
        roots = collector.to_dict()
        ev = next(r for r in roots if r["name"] == "control_trace")
        assert ev["metadata"]["name"] == "goal_judge"
        assert ev["metadata"]["action"] == "advisory_hint"
    finally:
        set_telemetry_collector(None)


def test_blind_reconstruct_skip_emits_control_trace() -> None:
    """无当前假设时盲重建此前**静默 return** → 现在必须留 trace (A/B 才测得出来)."""
    from huginn.autoloop.engine_reflect import EngineReflect

    captured: list[tuple[str, dict]] = []

    class _StubEngine:
        _iteration = 2
        _agent_factory = object()

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            captured.append((event_type, data))

    # 无 _current_hyp_id_for_plan → 命中第一个静默 return 分支
    asyncio.run(EngineReflect(_StubEngine())._blind_reconstruct_verify(None, {}))
    trace = next(d for _, d in captured if d.get("name") == "collab_blind_reconstruct")
    assert trace["action"] == "skip"
    assert "no current_hyp_id_for_plan" in trace["evidence"]


def test_blind_reconstruct_refute_emits_control_trace(monkeypatch) -> None:
    """成功派发的反证此前**只落 logger.info** → 控制面只见 skip, 会误判空转.

    run65 实测 3 次真反证全靠 FAILED.md 才看出, run.log 里只有 "skip: node
    status=refuted". 现在成功路径也必须落 trace.
    """
    import json as _json

    import huginn.agents.subagent as sub_mod
    from huginn.autoloop.engine_reflect import EngineReflect

    captured: list[tuple[str, dict]] = []

    class _Node:
        status = "untested"
        statement = "N_c(w) stays bounded as constraints grow"
        evidence: dict = {}
        dimension = ""

    class _Graph:
        _nodes = {"h1": _Node()}

        def refute(self, hid, ev):  # 断言反证路径被走到
            self.refuted = (hid, ev)

    class _Res:
        success = True
        summary = _json.dumps({"holds": False, "derivation": "bounded counterexample"})
        full_output = "bounded counterexample"
        tool_calls = [{"name": "code_lab"}]
        spec_name = "blind_reconstructor"

    class _FakeDispatch:
        async def dispatch(self, name, task, context=None):
            return _Res()

    monkeypatch.setattr(sub_mod, "SubagentDispatch", _FakeDispatch)
    monkeypatch.delenv("HUGINN_PER_HYP_BUDGET", raising=False)

    class _StubEngine:
        _iteration = 7
        _agent_factory = object()
        hypothesis_graph = _Graph()

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            captured.append((event_type, data))

    eng = EngineReflect(_StubEngine())
    eng._current_hyp_id_for_plan = "h1"
    # results.tests_passed=True → orig_holds=True, 盲重建 holds=False → mismatch → refute
    asyncio.run(eng._blind_reconstruct_verify(None, {"tests_passed": True}))

    trace = next(d for _, d in captured if d.get("name") == "collab_blind_reconstruct")
    assert trace["action"] == "refute"
    assert "blind_holds=False" in trace["evidence"]
    # 受控独立观察者: mismatch → 差分读数 True; 置信度落盘
    assert eng._last_reconstruct_disagree is True
    assert eng._last_blind_confidence == 0.5
    # 观察者是传感器不是 reward: 不得写 _last_task_perf (darwin 第 5 维)
    assert "_last_task_perf" not in vars(eng.engine)
    assert "summary_len=" in trace["evidence"]
    assert eng.hypothesis_graph.refuted[0] == "h1"


def test_blind_reconstruct_gate_force_and_default(monkeypatch) -> None:
    """观察者门控: =1 强制开; 未设/0 关 (默认零行为变化)."""
    from huginn.autoloop.engine_reflect import EngineReflect

    class _StubEngine:
        _iteration = 1

    eng = EngineReflect(_StubEngine())
    monkeypatch.delenv("HUGINN_BLIND_RECONSTRUCTION", raising=False)
    assert eng._blind_reconstruct_enabled() is False
    monkeypatch.setenv("HUGINN_BLIND_RECONSTRUCTION", "0")
    assert eng._blind_reconstruct_enabled() is False
    monkeypatch.setenv("HUGINN_BLIND_RECONSTRUCTION", "1")
    assert eng._blind_reconstruct_enabled() is True


def test_blind_reconstruct_gate_auto_uses_budget_tier(monkeypatch) -> None:
    """auto: 只 medium/open 档开; light 关 (让位真实实验); 解析失败保守关."""
    from huginn.autoloop.budget import IterationBudget
    from huginn.autoloop.engine_reflect import EngineReflect

    class _Ctl:
        def __init__(self, label: str) -> None:
            self._label = label

        def _resolve_budget_tier(self, iteration: int) -> IterationBudget:
            return IterationBudget(
                allowed_modes=None, max_calls=None, label=self._label
            )

    class _Boom:
        def _resolve_budget_tier(self, iteration: int):  # noqa: ANN201
            raise RuntimeError("no budget")

    class _StubEngine:
        _iteration = 5
        _engine_controller = None

    monkeypatch.setenv("HUGINN_BLIND_RECONSTRUCTION", "auto")
    eng = EngineReflect(_StubEngine())
    for label, expected in (("open", True), ("medium", True), ("light", False)):
        eng._engine_controller = _Ctl(label)
        assert eng._blind_reconstruct_enabled() is expected, label
    eng._engine_controller = _Boom()
    assert eng._blind_reconstruct_enabled() is False  # fail-closed


def test_writeback_hypothesis_status_supports_on_tests_passed() -> None:
    """B: 盲重建未跑时, tests_passed=True → support 当前假设 + 记任务性能."""
    from huginn.autoloop.engine_reflect import EngineReflect
    from huginn.autoloop.hypothesis_loop import HypothesisGraph

    captured: list[tuple[str, dict]] = []

    class _StubEngine:
        _iteration = 3
        hypothesis_graph = HypothesisGraph()

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            captured.append((event_type, data))

    eng = EngineReflect(_StubEngine())
    hid = eng.hypothesis_graph.add_hypothesis("掺杂增加则带隙减小")
    eng._current_hyp_id_for_plan = hid

    eng._writeback_hypothesis_status({"tests_passed": True}, r_phys=None)

    assert eng.hypothesis_graph.get(hid).status == "supported"
    # D: r_phys 缺省 → 回落 tests_passed 1.0, 写到引擎字段 (供 darwin 第 5 维)
    assert eng._last_task_perf == 1.0
    trace = next(
        d for _, d in captured if d.get("name") == "hypothesis_status_writeback"
    )
    assert trace["action"] == "support"


def test_writeback_hypothesis_status_refutes_and_records_r_phys() -> None:
    """B+D: tests_passed=False → refute; r_phys 优先作为任务性能."""
    from huginn.autoloop.engine_reflect import EngineReflect
    from huginn.autoloop.hypothesis_loop import HypothesisGraph

    class _StubEngine:
        _iteration = 3
        hypothesis_graph = HypothesisGraph()

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            pass

    eng = EngineReflect(_StubEngine())
    hid = eng.hypothesis_graph.add_hypothesis("原假设")
    eng._current_hyp_id_for_plan = hid

    eng._writeback_hypothesis_status({"tests_passed": False}, r_phys=0.2)

    assert eng.hypothesis_graph.get(hid).status == "refuted"
    assert eng._last_task_perf == 0.2


def test_writeback_hypothesis_status_skips_already_decided_node() -> None:
    """B: 盲重建已表态 (status!=untested) 时不重复写, 但仍记性能 (幂等)."""
    from huginn.autoloop.engine_reflect import EngineReflect
    from huginn.autoloop.hypothesis_loop import HypothesisGraph

    class _StubEngine:
        _iteration = 3
        hypothesis_graph = HypothesisGraph()

        def _emit_campaign(self, event_type: str, data: dict) -> None:
            pass

    eng = EngineReflect(_StubEngine())
    hid = eng.hypothesis_graph.add_hypothesis("已支持的假设")
    eng.hypothesis_graph.support(hid, {"modality": "blind_reconstruction"})
    eng._current_hyp_id_for_plan = hid

    # tests_passed=False 也不该把已支持的节点翻成 refuted (不撞状态机)
    eng._writeback_hypothesis_status({"tests_passed": False}, r_phys=None)

    assert eng.hypothesis_graph.get(hid).status == "supported"
    assert eng._last_task_perf == 0.0


def test_record_backup_candidates_extracts_predict_fields() -> None:
    """A: [DIM:...] 候选的 predict 字段必须落盘 (旧正则让 testable_prediction 恒空).

    选中候选的 predict → 引擎 _last_selected_prediction (主路径写 testable_prediction);
    backup 候选的 predict → 进图节点的 testable_prediction. 且 predict 正文不得
    并进 statement (run78 实测 support/testable 比例恒 0).
    """
    from huginn.autoloop.hypothesis_loop import HypothesisGraph, HypothesisLoop

    class _StubEngine:
        _iteration = 1

        def __init__(self) -> None:
            self.hypothesis_graph = HypothesisGraph()

        def _emit_control_trace(self, *args, **kwargs) -> None:
            pass

    eng = _StubEngine()
    loop = HypothesisLoop(eng)

    raw = (
        "[DIM: mechanism] 掺杂增加则带隙减小 | predict: dE/dx < 0 | pro: 有效质量\n"
        "[DIM: kinetics] 反应速率随温度指数增加 | predict: k = A exp(-Ea/RT)\n"
    )
    loop._record_backup_candidates(raw, "掺杂增加则带隙减小")

    # 选中候选的预测落盘给主路径
    assert eng._last_selected_prediction == "dE/dx < 0"
    stmts = {n.statement: n for n in eng.hypothesis_graph.all_nodes()}
    # 选中候选不重复进图 (主路径已加)
    assert "掺杂增加则带隙减小" not in stmts
    # backup 候选进图 + 其 predict 写入 testable_prediction, 正文不并进 statement
    backup = stmts["反应速率随温度指数增加"]
    assert backup.testable_prediction == "k = A exp(-Ea/RT)"
    assert "predict" not in backup.statement


def test_branch_incubator_empty_candidates_emits_trace() -> None:
    """孵化跑了但 N 路全空手 → 留 trace, 区分"没跑"与"跑了没结果"."""
    from huginn.autoloop.hypothesis_loop import HypothesisLoop

    captured: list[dict] = []

    class _EmptyIncubator:
        async def run_round(self, **kwargs):
            class _R:
                success = False
                hypothesis = ""
                tokens_used = 1

            return [_R(), _R(), _R()]

    class _StubEngine:
        _iteration = 4
        _max_pivots = 2
        _agent_factory = object()
        _branch_incubator = None

        def _emit_control_trace(
            self, name: str, evidence: str, action: str = "advisory_hint"
        ) -> None:
            captured.append({"name": name, "evidence": evidence, "action": action})

        async def _symreg_hint(self, context):
            return ""

        def _conjecture_hint(self, context):
            return ""

        def _build_hypothesis_prompt(self, context):
            return "prompt"

    eng = _StubEngine()
    eng._branch_incubator = _EmptyIncubator()
    loop = HypothesisLoop(eng)
    out = asyncio.run(loop._hypothesize_via_branch_incubator({}))
    assert out is None
    trace = next(d for d in captured if d["name"] == "collab_branch_incubator")
    assert trace["action"] == "skip"
    assert "empty" in trace["evidence"]


def test_engine_exposes_emit_control_trace_delegation() -> None:
    """协作对象经 __getattr__ 走 engine._emit_control_trace → 必须有该委托方法."""
    from huginn.autoloop.engine import AutoloopEngine

    assert hasattr(AutoloopEngine, "_emit_control_trace")


def test_repeat_chain_varied_script_resets_and_frees_directive() -> None:
    """改参数 = 新实验: streak 归零, 硬标志撤销 (不误伤正常扫描)."""
    from huginn.autoloop.engine_act import EngineAct
    from huginn.autoloop.engine_reflect import EngineReflect, _REPEAT_HARD_STREAK

    eng = _repeat_stub()
    ref = EngineReflect(eng)
    act = EngineAct(eng)
    for _ in range(_REPEAT_HARD_STREAK + 1):
        ref._detect_repeat_execution(_lab_result("W = [2, 4, 6]\n"), {})
    assert eng._repeat_exec_streak >= _REPEAT_HARD_STREAK
    assert "强制变异" in act._build_codelab_focus("扫描 w=10")

    changed: dict = {}
    ref._detect_repeat_execution(_lab_result("W = [2, 4, 8]\n"), changed)
    assert not changed.get("repeat_execution")
    assert eng._repeat_exec_streak == 0
    assert "强制变异" not in act._build_codelab_focus("扫描 w=10")


def test_repeat_chain_ignores_reused_result_object() -> None:
    """run52 回归: 同一个 result 对象再喂一遍 (= 本轮没跑实验) 不得记重复/收敛.

    否则假收敛会把长程 run 提前结题.
    """
    from huginn.autoloop.engine_reflect import EngineReflect

    eng = _repeat_stub()
    ref = EngineReflect(eng)
    res = _lab_result("W = [2, 4, 6]\n")
    for _ in range(8):  # 同一对象反复投喂
        ref._detect_repeat_execution(res, {})
    assert getattr(eng, "_repeat_exec_streak", 0) == 0
    assert getattr(eng, "_exec_converged", False) is False


def test_repeat_chain_converges_only_on_two_distinct_results() -> None:
    """收敛判定: 5+ 轮里去重后 ≤2 种指纹 → 结题; 每轮都不同 → 不结题."""
    from huginn.autoloop.engine_reflect import EngineReflect

    eng = _repeat_stub()
    ref = EngineReflect(eng)
    for i in range(6):  # A,B,A,B... 两种等价 family 来回换
        ref._detect_repeat_execution(_lab_result(f"W = [2, 4, {6 if i % 2 else 7}]\n"), {})
    assert eng._exec_converged is True

    eng2 = _repeat_stub()
    ref2 = EngineReflect(eng2)
    for i in range(6):  # 每轮都不同 → 正常探索, 不结题
        ref2._detect_repeat_execution(_lab_result(f"W = [2, 4, {i + 10}]\n"), {})
    assert eng2._exec_converged is False


class _FakeReply:
    def __init__(self, content: str) -> None:
        self.content = content
        self.usage_metadata = None


class _FakeAuthorLLM:
    """假模型: 记录每次作者提示(build_author_prompt 的真实产物), 回放固定脚本."""

    def __init__(self, script: str) -> None:
        self._script = script
        self.prompts: list[str] = []

    async def ainvoke(self, messages):  # noqa: ANN001
        self.prompts.append(str(messages[-1].content))
        return _FakeReply("```python\n" + self._script + "\n```")


def _author_stub(script: str) -> object:
    """引擎替身: 作者提示走真实 ``build_author_prompt`` 组装后才被假模型接住.

    注入点在 ``model``: ``EngineAct._llm_chat`` 用 ``model or self.model`` 真调模型,
    ``_request_code_lab_experiment`` 传的正是 ``self.verification_model``.
    """

    class _StubEngine:
        _objective = "恒定研究目标"
        _last_hypothesis = "H1: N_c 随 w 饱和"
        _speculator_hint = ""
        _current_phase = None

        def __init__(self) -> None:
            self.verification_model = _FakeAuthorLLM(script)
            self.prompts = self.verification_model.prompts

        async def _track_llm_usage(self, usage):  # noqa: ANN001
            return None

    return _StubEngine()


async def test_hard_directive_reaches_real_author_prompt() -> None:
    """集成: 同代码重跑越阈值后, 真实 build_author_prompt 组装出的提示里带强制变异令.

    与上面定点验证的区别: 这里**不经手搓 focus**, 而是走 ``EngineAct._execute_code_lab``
    的真实链路 —— 真实作者提示组装(build_author_prompt) + 真实沙箱执行(_run_code_lab)
    产出的真实 execution_result. 补上 run53/54/55 都没能自然触发的那一环.
    """
    from huginn.autoloop.engine_act import EngineAct
    from huginn.autoloop.engine_reflect import EngineReflect, _REPEAT_HARD_STREAK

    script = (
        "import numpy as np\n\n\n"
        "def run(cfg):\n"
        "    ws = [2, 4, 6]\n"
        "    out = [float(w) ** 2 for w in ws]\n"
        "    return {'success': True, 'summary': {'out': out},\n"
        "            'objectives': {'score': float(np.mean(out))}}\n"
    )
    eng = _author_stub(script)
    act = EngineAct(eng)
    ref = EngineReflect(eng)

    # 1) 真实作者路径: 提示由 build_author_prompt 组装; 未越阈值 → 不带硬指令
    code = await act._request_code_lab_experiment("恒定研究目标")
    assert "def run(" in code
    assert "强制变异" not in eng.prompts[-1]

    # 2) 真实沙箱跑两次同代码 → 指纹相同 → streak 累积越阈值
    first: dict = {}
    for i in range(_REPEAT_HARD_STREAK + 1):
        res, reason = act._run_code_lab(code)
        assert res is not None, reason
        out: dict = {}
        ref._detect_repeat_execution(res, out)
        first = out
    assert first.get("repeat_execution") is True
    assert getattr(eng, "_repeat_exec_streak", 0) >= _REPEAT_HARD_STREAK

    # 3) 下一轮作者提示: 真实 build_author_prompt 里必须带硬指令 + 上轮真实结果
    focus = act._build_codelab_focus("扫描 w")
    await act._request_code_lab_experiment("恒定研究目标", focus=focus)
    assert "强制变异" in eng.prompts[-1]
    assert "上一轮真实结果" in eng.prompts[-1]


async def test_codelab_repair_loop_stops_on_wall_clock_expiry(monkeypatch) -> None:
    """挂钟预算是合法硬出口: 修复循环不得越过它继续起新沙箱尝试.

    run56 实测: 3600s 预算下跑到 ~65min 仍卡在修复循环 (每次尝试可烧满
    HUGINN_CODELAB_TIMEOUT_S, 最多 max_repairs+1 次), 单轮越限近 1h. 迭代内不查
    预算 → "挂钟耗尽即停" 失效. D1: 时间口径下沉为统一 deadline 原语
    `_budget_exhausted` (旧 `_wall_clock_expired` 已删除), 每次尝试前查一次.
    """
    from huginn.autoloop.engine_act import EngineAct

    monkeypatch.setenv("HUGINN_CODELAB_REPAIR_ATTEMPTS", "3")

    attempts: list[int] = []

    class _StubEngine:
        # D1: 统一 deadline 原语 — 已耗尽, 不应再起新沙箱尝试
        def _budget_exhausted(self) -> bool:
            return True

    monkeypatch.setattr(EngineAct, "_build_codelab_focus", lambda self, d: "")

    async def _fake_author(self, goal, **kw):  # noqa: ANN001
        return "def run(cfg):\n    return {'success': True, 'objectives': {}}\n"

    monkeypatch.setattr(EngineAct, "_request_code_lab_experiment", _fake_author)

    def _fake_run(self, code):  # noqa: ANN001
        attempts.append(1)
        return None, "执行超时"

    monkeypatch.setattr(EngineAct, "_run_code_lab", _fake_run)

    act = EngineAct(_StubEngine())
    out = await act._execute_code_lab("扫描 w", {})

    assert attempts == [], "挂钟已耗尽, 修复循环不应再起新的沙箱尝试"
    assert out["success"] is False


async def test_codelab_repair_loop_runs_when_no_long_horizon(monkeypatch) -> None:
    """预算未耗尽 (短程): 修复循环照旧跑满 max_repairs+1 次 (防误伤)."""
    from huginn.autoloop.engine_act import EngineAct

    monkeypatch.setenv("HUGINN_CODELAB_REPAIR_ATTEMPTS", "2")

    attempts: list[int] = []

    class _StubEngine:
        # D1: 统一 deadline 原语 — 未耗尽, 修复循环照旧
        def _budget_exhausted(self) -> bool:
            return False

    monkeypatch.setattr(EngineAct, "_build_codelab_focus", lambda self, d: "")

    async def _fake_author(self, goal, **kw):  # noqa: ANN001
        return "code"

    monkeypatch.setattr(EngineAct, "_request_code_lab_experiment", _fake_author)

    def _fake_run(self, code):  # noqa: ANN001
        attempts.append(1)
        return None, "执行异常"

    monkeypatch.setattr(EngineAct, "_run_code_lab", _fake_run)

    act = EngineAct(_StubEngine())
    out = await act._execute_code_lab("扫描 w", {})

    assert len(attempts) == 3, f"短程模式应跑满 max_repairs+1=3 次, 实际 {len(attempts)}"
    assert out["success"] is False


# ===== 报告 citation 门 (C2: 数值必须溯源到本轮真实 execution_result) =====


def test_execution_ledger_appends_drops_script_and_caps() -> None:
    """台账: 每次 execute 留一条数值面; 丢掉脚本体; 容量封顶."""
    from huginn.autoloop.engine_act import _EXEC_LEDGER_MAX, EngineAct

    class _Stub:
        _execution_ledger: list = []

    eng = _Stub()
    act = EngineAct(eng)
    act._append_execution_ledger(
        "code_lab",
        {"success": True, "objectives": {"Nc": 45}, "script": "x = 1\n" * 500},
    )
    assert len(eng._execution_ledger) == 1
    entry = eng._execution_ledger[0]
    assert entry["tool"] == "code_lab"
    assert "45" in entry["result"]
    assert "script" not in entry["result"]  # 脚本体不该进台账

    for _ in range(_EXEC_LEDGER_MAX + 5):
        act._append_execution_ledger("code_lab", {"objectives": {"Nc": 1}})
    assert len(eng._execution_ledger) == _EXEC_LEDGER_MAX


def test_citation_gap_flags_untraceable_numbers() -> None:
    """报告 Results 里台账查无出处的数值计入 gap; 只审 Results, 不碰 Methods/Discussion."""
    from huginn.autoloop.engine_reflect import _citation_gap

    evidence = '[ev1] code_lab: {"objectives": {"Nc": 45}}'
    fabricated = (
        "## Methods\n训练 1000 轮, 宽度 999。\n\n"
        "## Results\n\n| w | Nc |\n|---|---|\n| 10 | 12 |\n| 20 | 15 |\n"
        "| 50 | 20 |\n\n## Discussion\n富集 1000 次。\n"
    )
    gap, total = _citation_gap(fabricated, evidence)
    # Results 里 |值|>=10 的候选去重 {10,12,15,20,50}; 台账只有 45 → 全部未溯源
    assert total == 5
    assert gap == 5  # Methods/Discussion 的 1000/999 未进审


def test_citation_gap_zero_when_numbers_traceable() -> None:
    from huginn.autoloop.engine_reflect import _citation_gap

    evidence = '[ev1] code_lab: {"objectives": {"Nc": 12, "w": 50}}'
    gap, total = _citation_gap("## Results\n\nNc(w=50) = 12.\n", evidence)
    assert total == 2  # {12, 50}
    assert gap == 0


def test_science_report_prompt_includes_citation_rule_with_ledger() -> None:
    from huginn.autoloop.engine_reflect import EngineReflect

    base = {"objective": "x", "total_time_seconds": 1.0, "phases": []}
    with_ledger = EngineReflect._build_science_report_prompt(
        dict(base), evidence_ledger="[ev1] code_lab: {Nc: 4}"
    )
    assert "Execution Evidence Ledger" in with_ledger
    assert "CITATION RULE" in with_ledger
    without = EngineReflect._build_science_report_prompt(dict(base))
    assert "CITATION RULE" not in without
    assert "Execution Evidence Ledger" not in without


async def test_report_flags_untraceable_numbers_and_annotates(tmp_path) -> None:
    """端到端(harness 内): 报告编造 Results 数值 → 落 trace + 附 Citation Audit 告示."""
    from huginn.autoloop.engine_reflect import EngineReflect

    captured: list[tuple[str, dict]] = []
    fabricated = (
        "## Results\n\n| w | Nc |\n|---|---|\n| 10 | 12 |\n| 20 | 15 |\n"
        "| 50 | 20 |\n\n## Discussion\nok\n"
    )

    class _StubEngine:
        workspace = tmp_path
        _iteration = 4
        _last_execution_result = {
            "_tool_name": "code_lab",
            "result": {"objectives": {"Nc": 45}},
        }
        _last_visual_context = ""
        _last_validation = ""
        _last_surprise = 0.0
        _last_hypothesis = ""
        _execution_ledger = [
            {"idx": 1, "tool": "code_lab", "result": '{"objectives": {"Nc": 45}}'}
        ]

        def _build_kb_text(self, query):  # noqa: ANN001
            return ""

        def _render_report(self, data):  # noqa: ANN001
            return "# Huginn Autoloop Report\n"

        def _emit_campaign(self, event_type, data):  # noqa: ANN001
            captured.append((event_type, data))

        async def _llm_chat(self, prompt, **kw):  # noqa: ANN001
            return fabricated

    path = await EngineReflect(_StubEngine())._report("就业余命题", [], 1.0)
    text = (tmp_path / path.split("/")[-1]).read_text(encoding="utf-8")
    assert "Citation Audit" in text
    assert "[ev1]" in text  # 证据台账随报告自包含, 供读者核对
    traces = [d for _, d in captured if d.get("name") == "report_citation"]
    assert traces and traces[0]["action"] == "annotate"


# ===== 单一完成出口 + 验收门 (完成 ≠ 验收) =====

def test_ledger_has_finite_evidence_accepts_finite_objectives() -> None:
    from huginn.autoloop.engine_reflect import _ledger_has_finite_evidence

    assert _ledger_has_finite_evidence(
        [{"idx": 1, "tool": "code_lab", "result": '{"objectives": {"gap": 1.17}}'}]
    )


def test_ledger_has_finite_evidence_rejects_nonfinite_and_text() -> None:
    from huginn.autoloop.engine_reflect import _ledger_has_finite_evidence

    # inf/nan 不是"测量到的数值", 纯文本错误与 exec 结构字段也不算证据.
    assert not _ledger_has_finite_evidence(
        [{"idx": 1, "tool": "code_lab", "result": '{"objectives": {"gap": Infinity}}'}]
    )
    assert not _ledger_has_finite_evidence(
        [{"idx": 1, "tool": "code_lab", "result": "Traceback: timeout"}]
    )
    assert not _ledger_has_finite_evidence(
        [{"idx": 1, "tool": "code_lab", "result": '{"exit_code": 0}'}]
    )
    assert not _ledger_has_finite_evidence([])


def test_ledger_evidence_text_renders_indexed_lines() -> None:
    from huginn.autoloop.engine_reflect import _ledger_evidence_text

    text = _ledger_evidence_text(
        [{"idx": 1, "tool": "code_lab", "result": "a"}, {"tool": "x", "result": "b"}]
    )
    assert "[ev1] code_lab: a" in text
    assert "[ev2] x: b" in text


def _goal():
    from huginn.autoloop.goal_store import Goal

    return Goal(
        id="g-test",
        text="计算 Si 间接带隙并给出机制",
        objective="计算 Si 间接带隙并给出机制",
        status="active",
    )


class _LoopState:
    def __init__(self, iteration: int = 2) -> None:
        self.iteration = iteration
        self.should_stop = False


def _fake_judge(achieved: bool):
    class _J:
        def __init__(self, llm=None):  # noqa: ANN001
            self._llm = llm

        def judge(self, objective, trajectory=None, final_output=""):  # noqa: ANN001
            return {
                "achieved": achieved,
                "score": 0.9 if achieved else 0.1,
                "evidence": [],
                "gaps": [] if achieved else ["缺机制解释"],
            }

    return _J


async def test_evaluate_completion_stops_only_with_evidence(monkeypatch) -> None:
    """有证据 + judge achieved + 无需 LLM skeptic → 单一出口收结."""
    import huginn.evaluation.goal_judge as gj_mod

    monkeypatch.setattr(gj_mod, "GoalJudge", _fake_judge(True))
    from huginn.autoloop.cognitive_loop import CognitiveRunner

    captured: list[tuple[str, dict]] = []
    # _emit_campaign 是 CognitiveRunner 的 own-method, 实例化后不走 engine 转发,
    # 故直接 patch 类方法以捕获 trace (与 test_control_trace_emits_uniform_schema 同源).
    monkeypatch.setattr(
        CognitiveRunner,
        "_emit_campaign",
        lambda self, et, data: captured.append((et, data)),  # noqa: ANN001
    )

    class _Stub:
        _execution_ledger = [
            {"idx": 1, "tool": "code_lab", "result": '{"objectives": {"gap": 1.17}}'}
        ]
        _last_completion_iter = -1
        _speculator_hint = ""

        def _metacog_check_completion(self):
            return (False, "")

    res = await CognitiveRunner(_Stub())._evaluate_completion(
        _goal(), {}, _LoopState(2), 5
    )
    assert res["ran"] and res["stop"] is True
    assert any(d.get("name") == "goal_judge" for _, d in captured)


async def test_evaluate_completion_blocks_without_execution_evidence(monkeypatch) -> None:
    """judge 说达成但台账无有限数值 → 证据门拦截 (完成 ≠ 验收)."""
    import huginn.evaluation.goal_judge as gj_mod

    monkeypatch.setattr(gj_mod, "GoalJudge", _fake_judge(True))
    from huginn.autoloop.cognitive_loop import CognitiveRunner

    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        CognitiveRunner,
        "_emit_campaign",
        lambda self, et, data: captured.append((et, data)),  # noqa: ANN001
    )

    class _Stub:
        _execution_ledger: list = []
        _last_completion_iter = -1
        _speculator_hint = ""

        def _metacog_check_completion(self):
            return (False, "")

    res = await CognitiveRunner(_Stub())._evaluate_completion(
        _goal(), {}, _LoopState(2), 5
    )
    assert res["ran"] and res["stop"] is False
    assert "验收门" in res["hint"]
    assert any(d.get("name") == "goal_acceptance" for _, d in captured)


async def test_evaluate_completion_skeptic_can_block(monkeypatch) -> None:
    """证据门过了, 但独立对抗审查未通过 → 不终止, 反例回灌为 hint."""
    import huginn.evaluation.goal_judge as gj_mod

    monkeypatch.setattr(gj_mod, "GoalJudge", _fake_judge(True))
    from huginn.autoloop.cognitive_loop import CognitiveRunner

    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        CognitiveRunner,
        "_emit_campaign",
        lambda self, et, data: captured.append((et, data)),  # noqa: ANN001
    )

    class _Resp:
        content = (
            '{"overall_verdict": "fail", "implausible_metrics": '
            '[{"metric": "gap", "paper": 1.1, "yours": 0.4, "red_flag": "beats baseline"}]}'
        )

    class _Model:
        async def ainvoke(self, messages):  # noqa: ANN001
            return _Resp()

    class _Stub:
        _execution_ledger = [
            {"idx": 1, "tool": "code_lab", "result": '{"objectives": {"gap": 0.4}}'}
        ]
        _last_completion_iter = -1
        _speculator_hint = ""
        verification_model = _Model()
        model = _Model()

        def _metacog_check_completion(self):
            return (False, "")

    res = await CognitiveRunner(_Stub())._evaluate_completion(
        _goal(), {}, _LoopState(2), 5
    )
    assert res["ran"] and res["stop"] is False
    assert "SKEPTIC" in res["hint"]
    assert any(d.get("name") == "goal_skeptic" for _, d in captured)


async def test_evaluate_completion_dedups_same_iteration(monkeypatch) -> None:
    import huginn.evaluation.goal_judge as gj_mod

    monkeypatch.setattr(gj_mod, "GoalJudge", _fake_judge(False))
    from huginn.autoloop.cognitive_loop import CognitiveRunner

    monkeypatch.setattr(
        CognitiveRunner,
        "_emit_campaign",
        lambda self, et, data: None,  # noqa: ANN001
    )

    class _Stub:
        _execution_ledger: list = []
        _last_completion_iter = -1
        _speculator_hint = ""

        def _metacog_check_completion(self):
            return (False, "")

    stub = _Stub()
    first = await CognitiveRunner(stub)._evaluate_completion(_goal(), {}, _LoopState(2), 5)
    second = await CognitiveRunner(stub)._evaluate_completion(_goal(), {}, _LoopState(2), 5)
    assert first["ran"] is True
    assert second["ran"] is False


def test_cognitive_loop_has_single_completion_exit() -> None:
    """回归守卫: 三条判停路径已合并, 旧散装 F2/F17 不应复活."""
    import inspect

    from huginn.autoloop import cognitive_loop as cl

    src = inspect.getsource(cl)
    assert "_evaluate_completion(" in src
    assert "_use_unified_decision" not in src
    assert "_use_gate" not in src
    assert "v10 F17 GoalJudge" not in src


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
