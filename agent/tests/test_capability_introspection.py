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
    CAPABILITY_KINDS,
    confirm_proposal,
    create_proposal,
    diagnose_gaps,
    diagnose_islands,
    diagnose_surface,
    evolution_run,
    island_totals,
    propose_capabilities,
    reactivate_proposal,
    realize_proposal,
    reject_proposal,
    rollback_realization,
    run_capability_self_audit,
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


# ── §4 确认门禁: staged → approved → implemented / rejected 状态机 ──────────


def test_s4_proposal_starts_staged_and_needs_confirmation():
    """新建提案初始恒为 staged; 未确认前不可实现(不伪装)."""
    rec = create_proposal("topology_diag", "高阶拓扑诊断", {"F": 1})
    assert rec.status == "staged"
    assert rec.handler_implemented is False
    assert rec.auth == "runtime-proposed"
    # 未确认就尝试落地 → 拒绝
    try:
        realize_proposal(rec, lambda a: a)
        raised = False
    except ValueError:
        raised = True
    assert raised, "未确认(approved)前 realize 必须被拒绝"


def test_s4_confirm_then_reject_and_reactivate():
    """确认门禁: staged→approved; 可驳回→rejected; 可 reactivate 回 staged."""
    rec = create_proposal("insolation_diag", "日晒诊断")
    confirm_proposal(rec, authorized_by="human")
    assert rec.status == "approved"
    assert rec.auth == "human"
    # rejected 分支
    reject_proposal(rec, reason="优先级不足")
    assert rec.status == "rejected"
    # 已驳回不能再 confirm (需先 reactivate)
    try:
        confirm_proposal(rec)
        raised = False
    except ValueError:
        raised = True
    assert raised
    reactivate_proposal(rec)
    assert rec.status == "staged"
    confirm_proposal(rec, authorized_by="gate")
    assert rec.status == "approved"


def test_s4_implemented_cannot_reject_directly():
    """已实现的能力不能直接驳回 —— 必须先回滚(防止没清理就改状态的脏路径)."""
    rec = create_proposal("flux_diag", "守恒/通量诊断")
    confirm_proposal(rec)
    j = realize_proposal(rec, lambda a: {"flux": 1.0})
    assert j.registered is False     # 无注册表 → 仅 journal, 不伪装落表
    assert rec.status == "implemented"
    try:
        reject_proposal(rec)
        raised = False
    except ValueError:
        raised = True
    assert raised, "已实现须先 rollback 再驳"


# ── §5 落地实现: 仅 approved + 真实 handler；无 handler 拒绝(诚实边界) ───────


def test_s5_realize_refuses_none_handler():
    """§5 核心诚实边界: handler=None 一律拒绝, 绝不把 spec 伪装成已实现."""
    rec = create_proposal("topology_diag", "高阶拓扑诊断")
    confirm_proposal(rec)
    try:
        realize_proposal(rec, None)
        raised = False
    except ValueError:
        raised = True
    assert raised
    assert rec.status == "approved", "拒绝实现时不篡改状态(保持 未实现 但已确认)"
    assert rec.handler_implemented is False


def test_s5_realize_with_real_handler_then_run():
    """有真实 handler 才实现: 状态置 implemented; handler 本体可真跑出结果.

    真实 Capability 装配需 pydantic(重栈, 沙箱可缺); 这里只验证 §5 的门楣:
    - handler 非 None → 实现被接受、记录 Journal(registered=False 仅注册表缺位);
    - handler 本身是可调用数值逻辑, 直接调用验证其得出真实结果(非编造)。
    """
    rec = create_proposal("insolation_diag", "日晒/平衡温度诊断", {"S0": 1361})
    confirm_proposal(rec)
    handler = lambda a: {"T_eq": a["S0"] / 4}   # noqa: E731
    j = realize_proposal(rec, handler)
    assert rec.status == "implemented"
    assert rec.handler_implemented is True
    assert j.name == "insolation_diag"
    assert j.prev_registered == []              # 无注册表, registered=False(仅 journal)
    # handler 可调用性: 真实返回来自计算, 非伪造
    assert handler({"S0": 1361}) == {"T_eq": 340.25}


# ── §6 回滚机制: 依据 Journal 移除新落地能力, 恢复实现前快照 ────────────────


def test_s6_rollback_removes_realized_capability():
    """journal 为 registered 的实现, 回滚时精确移除新增能力, 保留 prev 快照."""
    # 用假注册表(纯内存)验证登记/回滚对账
    class FakeRegistry:
        def __init__(self, existing):
            self._caps = dict.fromkeys(existing)
        def list_capabilities(self):
            return list(self._caps)
        def register(self, cap):
            self._caps[cap.name] = cap
        def unregister(self, name):
            self._caps.pop(name, None)
        def __contains__(self, name):
            return name in self._caps

    reg = FakeRegistry(["a", "b"])
    rec = create_proposal("c_diag", "新增诊断", {"x": 1})
    confirm_proposal(rec)
    j = realize_proposal(rec, lambda a: a, registry=reg)
    assert j.registered is True
    assert "c_diag" in reg and "a" in reg and "b" in reg
    assert j.prev_registered == ["a", "b"]

    rb = rollback_realization(j, registry=reg)
    assert rb.removed is True
    assert "c_diag" not in reg               # 只移除自己新增的
    assert set(rb.restored) == {"a", "b"}    # 快照恢复不动的老能力


def test_s6_evolution_run_with_handler_store():
    """evolution_run 全闭环: 有 handler 的落地, 无 handler 的保持 approved 挂起."""
    out = evolution_run(
        "研究系外行星日晒与环流",
        discoverable=["hodge_circulation"],
        handler_store={"proposed_日晒_平衡温度第一性原理诊断": lambda a: {"T_eq": 300}},
        authorized_by="human",
    )
    assert [r["name"] for r in out["realized"]], "有 handler 的提案应被实现(Journal)"
    assert any(p["status"] == "implemented" for p in out["proposals"])
    assert out["proposals"], "应至少产出提案并通过确认门禁"
    # 拓扑已被 hodge_circulation 覆盖 → 不再当缺口; 日晒缺口若缺 handler 记 pending
    assert "pending_no_handler" in out


# ── P0 孤岛盘点: 四类注册表分别登记, 分别可见, 分别悬挂 ────────────────────


def _patch_island_registries(monkeypatch, *, caps, skills, workflows):
    """把四类真实注册表替换成确定性快照, 隔离重栈与全局状态."""
    import _introspection as intro

    monkeypatch.setattr(intro, "_capability_names", lambda: list(caps))
    monkeypatch.setattr(intro, "_skill_names", lambda: list(skills))
    monkeypatch.setattr(intro, "_workflow_names", lambda: list(workflows))
    monkeypatch.setattr(intro, "_capability_unresolved", lambda *a: {})
    monkeypatch.setattr(intro, "_skill_unresolved", lambda *a: {})
    monkeypatch.setattr(intro, "_workflow_unresolved", lambda *a: {})


def test_p0_all_kinds_reported_with_own_dangling(monkeypatch):
    """蓝队: 四类都进盘点; 每类的悬挂 = 本类注册 - 本类可见."""
    _patch_island_registries(monkeypatch, caps=["cap_a"], skills=["sk_a"], workflows=["wf_a"])
    surfaces = diagnose_islands(
        tool_registered=["tool_a", "hidden_tool"],
        tool_discoverable=["tool_a", "capability_tool", "skill"],
    )
    assert set(surfaces) == set(CAPABILITY_KINDS)
    assert surfaces["tool"].dangling == ["hidden_tool"]
    assert surfaces["capability"].discoverable == ["cap_a"]
    assert surfaces["capability"].dangling == []
    assert surfaces["skill"].discoverable == ["sk_a"]
    # workflow 的两个入口(catalog_search / workflow_tool)都不可见 → 如实全量悬挂
    assert surfaces["workflow"].discoverable == []
    assert surfaces["workflow"].dangling == ["wf_a"]


def test_p0_entrypoint_hidden_makes_whole_kind_dangling(monkeypatch):
    """红队: 类目已注册但发现入口工具不可见 → 整类悬挂(不能算作可达)."""
    _patch_island_registries(monkeypatch, caps=["cap_a"], skills=["sk_a"], workflows=[])
    surfaces = diagnose_islands(
        tool_registered=["tool_a", "capability_tool", "skill"],
        tool_discoverable=["tool_a"],            # 两个入口都被裁掉
    )
    assert surfaces["capability"].discoverable == []
    assert surfaces["capability"].dangling == ["cap_a"]
    assert surfaces["skill"].dangling == ["sk_a"]


def test_p0_explicit_surface_does_not_touch_real_registries(monkeypatch):
    """确定性: 显式传入 tool 面时只盘点 tool 类, 不暗中读其他注册表."""
    import _introspection as intro

    def _boom():
        raise AssertionError("显式盘点不应读取真实注册表")

    monkeypatch.setattr(intro, "_capability_names", _boom)
    s = diagnose_surface(registered=["a", "b"], discoverable=["a"])
    assert s.dangling == ["b"]
    assert set(s.by_kind) == {"tool"}
    assert s.unresolved_refs == {}


def test_p0_island_totals_sum_all_kinds(monkeypatch):
    """汇总计数覆盖全部 kind, 不隐藏任何一类."""
    _patch_island_registries(monkeypatch, caps=["c1", "c2"], skills=[], workflows=[])
    surfaces = diagnose_islands(
        tool_registered=["t1", "t2"], tool_discoverable=["t1"])
    totals = island_totals(surfaces)
    assert totals["registered"] == 4          # 2 tool + 2 cap (+ 0 skill + 0 workflow)
    assert totals["dangling"] == 3            # t2 + c1 + c2


def test_p0_catalog_entrypoint_makes_all_kinds_discoverable(monkeypatch):
    """蓝队: catalog_search 可见时, capability/skill/workflow 三类都经它可达."""
    _patch_island_registries(monkeypatch, caps=["cap_a"], skills=["sk_a"], workflows=["wf_a"])
    surfaces = diagnose_islands(
        tool_registered=["tool_a", "catalog_search"],
        tool_discoverable=["tool_a", "catalog_search"],
    )
    for kind in ("capability", "skill", "workflow"):
        assert surfaces[kind].discoverable == surfaces[kind].registered
        assert surfaces[kind].dangling == []


def test_p0_workflow_unresolved_ref_is_reported(monkeypatch):
    """跨类失效引用: workflow 指向不存在的工具 → 记为孤岛边."""
    import _introspection as intro

    from huginn.workflows.registry import WorkflowPackage, WorkflowRegistry

    pkg = WorkflowPackage(name="wf_ghost", kind="stages",
                          stages=[{"tool": "ghost_tool"}])
    monkeypatch.setattr(WorkflowRegistry, "list_names",
                        classmethod(lambda cls: ["wf_ghost"]))
    monkeypatch.setattr(WorkflowRegistry, "get",
                        classmethod(lambda cls, n: pkg if n == "wf_ghost" else None))
    assert intro._workflow_unresolved({"real_tool"}) == {"wf_ghost": ["ghost_tool"]}


def test_p0_skill_unresolved_ref_is_reported(monkeypatch):
    """跨类失效引用: skill 步骤指向不存在的工具 → 记为孤岛边."""
    from types import SimpleNamespace

    import _introspection as intro

    from huginn.skills.registry import SkillRegistry

    skill = SimpleNamespace(name="sk_ghost",
                            steps=[SimpleNamespace(tool="ghost_tool")],
                            required_tools=["also_ghost"])
    monkeypatch.setattr(SkillRegistry, "get_all_definitions",
                        classmethod(lambda cls: [skill]))
    assert intro._skill_unresolved({"real_tool"}) == {
        "sk_ghost": ["also_ghost", "ghost_tool"]}
