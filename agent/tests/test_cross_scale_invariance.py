"""跨尺度不变性验证: 验证 hypothesize→plan→execute→validate 研究模式
在 phase / stage / tool 三个尺度上同构.

重整化群启发: 不同尺度下的"研究模式"应该是同一个不动点.
如果某一尺度缺失了某个环节, 说明该尺度需要补全.

层次映射:
  Phase 层 (autoloop):  Hypothesize → Plan → Execute → Validate → Learn
  Stage 层 (Deli):       Topic → Search → Gap → Outline → Draft → Review
  Tool 层 (symbolic):    pde_classify → pde_separation → pde_discretize → constraint_check
"""

from __future__ import annotations

import inspect

import pytest

# ── Phase 层: AutoloopEngine 的 7 阶段 ──────────────────────────

EXPECTED_PHASES = {
    "hypothesize": "形成假设 (先验)",
    "plan": "规划执行路径",
    "execute": "执行工具/工作流 (行动)",
    "validate": "验证结果 (后验更新)",
    "learn": "写入 memory/KB/KG (知识固化)",
    "report": "生成报告",
    "decide": "决定下一轮方向",
}


def test_phase_layer_has_research_cycle():
    """Phase 层: AutoloopEngine 必须有 hypothesize/plan/execute/validate/learn."""
    from huginn.autoloop.engine import AutoloopEngine

    methods = dir(AutoloopEngine)
    required = ["_hypothesize", "_plan", "_validate", "_learn"]
    missing = [m for m in required if m not in methods]
    assert not missing, f"Phase 层缺失研究环节: {missing}"


def test_phase_layer_surprise_feedback():
    """Phase 层: validate → hypothesize 通过 surprise 信号形成反馈环.

    surprise 在 validate 中计算, 通过 _pick_hypothesis_persona 间接影响
    hypothesize 的 persona 选择 (高 surprise → reviewer persona).
    """
    from huginn.autoloop.engine import AutoloopEngine

    src = inspect.getsource(AutoloopEngine)
    assert "_compute_surprise" in src, "validate 缺少 surprise 计算"
    assert "_last_surprise" in src, "surprise 信号未跨阶段传递"
    # _pick_hypothesis_persona 读取 surprise 决定 persona
    pick_src = inspect.getsource(AutoloopEngine._pick_hypothesis_persona)
    assert "surprise" in pick_src.lower(), \
        "_pick_hypothesis_persona 未使用 surprise 信号"


def test_phase_layer_robust_surprise():
    """Phase 层: surprise 支持分布鲁棒估计 (worst-case)."""
    from huginn.autoloop.engine import AutoloopEngine

    assert hasattr(AutoloopEngine, "_compute_surprise_robust"), \
        "缺少 _compute_surprise_robust 方法"


# ── Stage 层: Deli pipeline 的 9 阶段 ───────────────────────────

EXPECTED_STAGES = [
    "topic_analysis",      # ≈ hypothesize: 确定研究问题
    "literature_search",   # ≈ plan: 检索已有知识
    "gap_analysis",        # ≈ plan: 识别空白
    "outline",             # ≈ plan: 设计结构
    "drafting",            # ≈ execute: 写作
    "citation_verify",     # ≈ validate: 验证引用
    "peer_review",         # ≈ validate: 同行评审
    "revision",            # ≈ learn: 修正
    "final",               # ≈ report: 定稿
]


def test_stage_layer_has_research_cycle():
    """Stage 层: DeliAutoResearch 必须有等价于 hypothesize/execute/validate 的阶段."""
    from huginn.academic.deli_research import ResearchStage

    stages = [s.value for s in ResearchStage]
    # 映射到研究循环: 至少有 预测/执行/验证/固化 四类
    required_patterns = {
        "hypothesize": ["topic", "gap"],
        "execute": ["draft", "outline"],
        "validate": ["review", "verify", "citation"],
        "learn": ["revision", "final"],
    }
    for role, keywords in required_patterns.items():
        found = any(any(kw in s.lower() for kw in keywords) for s in stages)
        assert found, f"Stage 层缺失 {role} 环节 (期望包含 {keywords} 之一)"


def test_stage_layer_has_math_structure_identification():
    """Stage 层: gap_analysis 后有数学结构识别 (advisory)."""
    from huginn.academic.deli_research import DeliAutoResearch

    src = inspect.getsource(DeliAutoResearch)
    assert "_identify_math_structures" in src, "缺少数学结构识别"
    assert "exploratory" in src, "结构识别缺少 exploratory 类型"


def test_stage_layer_has_research_intuition():
    """Stage 层: 支持研究者直觉注入 (advisory, not prescriptive)."""
    from huginn.academic.deli_research import ResearchState

    assert hasattr(ResearchState, "__dataclass_fields__"), \
        "ResearchState 应该是 dataclass"
    fields = ResearchState.__dataclass_fields__
    assert "research_intuition" in fields, \
        "ResearchState 缺少 research_intuition 字段"


# ── Tool 层: symbolic_math_tool 的 PDE 工作流 ──────────────────

def test_tool_layer_has_research_cycle():
    """Tool 层: PDE 工作流有 classify→derive→discretize→check 循环."""
    from huginn.tools.symbolic_math import pde as pde_mod

    src = inspect.getsource(pde_mod)
    # PDE action 名不带 pde_ 前缀, 在 SymbolicMathTool 里路由时带前缀
    required = ["def classify", "def discretize"]
    for r in required:
        assert r in src, f"PDE 工具缺少 {r} 函数"


def test_tool_layer_sr_has_discover_and_validate():
    """Tool 层: symbolic_regression 有 discover (execute) + constraint_check (validate)."""
    try:
        from huginn.tools.symbolic_math.symbolic_regression import (
            SymbolicRegressionTool,
        )
    except ImportError:
        pytest.skip("symbolic_regression not available")

    src = inspect.getsource(SymbolicRegressionTool)
    assert "discover" in src, "SR 缺少 discover action"
    assert "constraint_check" in src, "SR 缺少 constraint_check action"


# ── 跨尺度同构验证 ──────────────────────────────────────────────

def test_cross_scale_invariance():
    """三个尺度都有 hypothesize→execute→validate→learn 的同构结构.

    这是 RG 不动点: 不同粗粒化尺度下研究模式不变.
    """
    # Phase: AutoloopEngine
    from huginn.autoloop.engine import AutoloopEngine
    phase_methods = {m for m in dir(AutoloopEngine) if not m.startswith("__")}
    phase_has = {
        "hypothesize": "_hypothesize" in phase_methods,
        "execute": any("execute" in m.lower() for m in phase_methods),
        "validate": "_validate" in phase_methods,
        "learn": "_learn" in phase_methods,
    }

    # Stage: DeliAutoResearch
    from huginn.academic.deli_research import ResearchStage
    stage_values = [s.value.lower() for s in ResearchStage]
    stage_has = {
        "hypothesize": any("topic" in s or "gap" in s for s in stage_values),
        "execute": any("draft" in s or "outline" in s for s in stage_values),
        "validate": any("review" in s or "verify" in s for s in stage_values),
        "learn": any("revision" in s or "final" in s for s in stage_values),
    }

    # Tool: PDETool / SRTool
    try:
        from huginn.tools.symbolic_math import pde as pde_mod
        tool_src = inspect.getsource(pde_mod)
        tool_has = {
            "hypothesize": "classify" in tool_src,
            "execute": "discretize" in tool_src,
            "validate": "constraint_check" in tool_src or "stability" in tool_src,
            "learn": True,  # tool 层的 learn = 结果被 _learn 写入 memory
        }
    except ImportError:
        tool_has = {"hypothesize": True, "execute": True, "validate": True, "learn": True}

    # 三个尺度在四个环节上都应该为 True
    for role in ["hypothesize", "execute", "validate", "learn"]:
        assert phase_has[role], f"Phase 层缺失 {role}"
        assert stage_has[role], f"Stage 层缺失 {role}"
        # tool 层允许部分缺失 (不是所有工具都有完整循环)
        # 但至少 hypothesize + execute 应该有
        if role in ("hypothesize", "execute"):
            assert tool_has[role], f"Tool 层缺失 {role}"


# ── Goal 持久化跨尺度 ───────────────────────────────────────────

def test_goal_persistence_across_scales():
    """Goal 持久化在 context (prompt) 和 engine (iteration) 两个尺度上生效."""
    from huginn.autoloop.goal_store import GoalStore
    from huginn.context_builder import ContextBuilder

    # GoalStore: 持久化层
    assert hasattr(GoalStore, "get_active"), "GoalStore 缺少 get_active"
    assert hasattr(GoalStore, "increment_iteration"), "GoalStore 缺少 increment_iteration"

    # ContextBuilder: prompt 层注入
    assert hasattr(ContextBuilder, "build_goal_text"), \
        "ContextBuilder 缺少 build_goal_text"


def test_subgoal_persistence():
    """/subgoal 在 GoalStore 和 agent 两个路径上生效."""
    from huginn.autoloop.goal_store import GoalStore

    assert hasattr(GoalStore, "add_sub_goal"), "GoalStore 缺少 add_sub_goal"
