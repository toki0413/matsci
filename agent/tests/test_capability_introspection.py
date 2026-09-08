"""能力自省闭环 §1–§3 测试 —— 自审计能力面的机械保证.

对齐契约(tool_surface + capabilities-and-registration-spec), 纯标准库/轻依赖, 零网络/零 LLM:
  1. §1 盘点: 孤岛(dangling) 检出 —— 已注册但 LLM 不可见的工具被如实报出.
  2. §2 诊断: 目标含领域关键词但可见面无覆盖 → 缺口检出 (确定性规则).
  3. §2 覆盖: 关键词已被可见工具覆盖 → 不再误报缺口.
  4. §3 提案: 缺口 → staged 提案, 强转 canonical_tool_shape, handler_ref=None 不伪装实现.
  5. §3 入轨迹: stage_to_trace 产物可直接并入研究 trace, 报告引用即可被 grounding 门禁核实.
  6. 端到端: run_research_program(self_audit=...) 把审计工件并入 trace, 门禁 pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SYS = Path(__file__).resolve().parents
sys.path.insert(0, str(_SYS[1]))  # agent
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))  # examples

# huginn.capabilities.__init__ 会 eager 拉 pydantic(重栈); 本模块自身轻 —— 按文件路径加载,
# 绕过重 __init__ (与既有 claim_grounding fallback 同一模式), 保证沙箱/轻量环境可测。
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_introspection", str(_SYS[1] / "huginn/capabilities/introspection.py"))
_intro = _ilu.module_from_spec(_spec)
sys.modules["_introspection"] = _intro  # 须在 exec_module **前** 注册, dataclass 装饰在模块体里执行
_spec.loader.exec_module(_intro)
from _introspection import (  # noqa: E402
    diagnose_gaps, diagnose_surface, propose_capabilities, run_capability_self_audit,
    stage_to_trace,
)
from huginn.research.tool_surface import canonical_tool_shape  # noqa: E402


def test_s1_detect_island_dangling():
    """注册的 3 个工具里 1 个不可见 → 被识别为 dangling(孤岛/悬挂)."""
    s = diagnose_surface(registered=["a", "b", "hidden_x"],
                         discoverable=["a", "b"])
    assert s.dangling == ["hidden_x"]
    assert s.registered == ["a", "b", "hidden_x"]
    assert s.discoverable == ["a", "b"]


def test_s2_gap_detected_when_uncovered():
    """目标涉拓扑诊断, 但可见面无拓扑能力 → 命中缺口."""
    gaps = diagnose_gaps("研究热木星超自转环的拓扑环流结构", discoverable=["insolation_tool"])
    assert any("拓扑" in g["capability"] for g in gaps)


def test_s2_no_false_gap_when_covered():
    """可见工具已覆盖该关键词 → 不误报缺口."""
    gaps = diagnose_gaps("研究热木星超自转环的拓扑环流结构",
                         discoverable=["hodge_circulation"])
    assert not any("拓扑" in g["capability"] for g in gaps)


def test_s3_proposal_is_staged_spec_not_fake_implementation():
    """缺口的提案是 staged spec: schema 走 canonical 形状, handler_ref=None(未实现, 不伪装)."""
    props = propose_capabilities("研究系外行星的日晒与平衡温度", discoverable=["hodge_circulation"])
    assert props, "应产出至少一个提案"
    p = props[0]
    assert p["status"] == "staged"
    assert p["origin"] == "runtime-proposed"
    assert p["handler_ref"] is None            # 诚实: 未实现, 不把缺失当拥有
    # schema 形状与工具面单一形状一致
    shape = canonical_tool_shape(p["name"], p["description"], p["parameters"])
    assert shape["type"] == "function"
    assert shape["function"]["name"] == p["name"]


def test_s3_trace_artifacts_grounding():
    """stage_to_trace 工件可作为 grounding 证据.

    门禁契约是**数值真实性** (claim_grounding 只锁数值、不动方法):
      - 正例: 报告只**引用提案名**(不编造任何数字) → 无未落地主张 → pass;
      - 反例: 给**未实现**能力编造数值(如平衡温度 0.0000245K) → 该值不在 trace
              (stage_to_trace 的工件只有 name/描述, 无数值) → 被门禁拦截。
    这正是不把"缺失当拥有"的可证伪性: staged 提案 handler_implemented=False,
    报告可以引用它, 但**不能**据此编造数字。
    """
    from huginn.research import grounding_verifier
    verify = grounding_verifier()
    traces = stage_to_trace(propose_capabilities("系外行星日晒", discoverable=[]))
    assert traces, "无部署时 (discoverable=[], 全缺口) 应产出 trace 工件"
    name = json.loads(traces[0])["name"]
    report = f"研究结论: 我们识别到能力缺口并提案 '{name}' 以闭合该缺口。"
    g = verify(report, traces)
    assert g["verdict"] == "pass"
    # 反例: 给未实现能力编造数值 → 数值不在 trace → 门禁拦截
    g2 = verify(f"我们提案『{name}』并由该能力测得平衡温度 2435.7 K。", traces)
    assert g2["verdict"] != "pass"
    assert 2435.7 in g2["unsubstantiated"]


def test_end_to_end_self_audit_joins_trace():
    """run_research_program(self_audit=...) 把审计工件并入 trace, 门禁 pass. """
    import ai4s_backends as ab
    from huginn.research import run_research_program

    goal = "研究系外行星日晒与热木星平衡温度的深研"
    backend = ab.hotjupiter_backend() + ab.exoplanet_backend()
    obj = dict(ab.OBJECTIVES["hotjupiter"])
    o = run_research_program(
        goal=goal, experiments=backend, objectives_config=obj,
        max_iterations=8, min_iterations=3,
        self_audit=lambda: run_capability_self_audit(goal, discoverable=[])["trace"],
    )
    # 审计工件已并入 trace → 门禁 pass (报告由确定性组装, 数值全落地)
    assert o.verdict == "pass", (o.verdict, o.ungrounded[:3])