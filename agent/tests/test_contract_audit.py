"""MECE 契约审计 (contract_audit) 自检.

两类断言:
  - **真实仓上的不变量**: 奖励面/授权面的接线状态与 MECE 违例是可复现的事实
    (如 `reconcile_r_phys` 零调用者 + 跨模块同名), 编码在这里防回归.
  - **合成树上的归属性**: 用临时仓验证"模块限定扫描"能隔离同名 —— 裸同名调用
    归属给别的模块, 不污染本模块的公开面 (这是本工具相对裸名计数的关键修正).
"""

from __future__ import annotations

import pathlib

from huginn.cli import contract_audit as ca

_REPO = pathlib.Path(__file__).resolve().parents[1]


# ──────────────────── 真实仓: 奖励面 ────────────────────


def test_reward_surface_lists_all_declared_terms():
    c = ca.build_reward_contract()
    names = {t["name"] for t in c["terms"]}
    assert {
        "numeric_accuracy_reward",
        "grounding_source_reward",
        "grounded_accuracy_reward",
        "strict_scope_reward",
        "efficiency_discount",
        "idle_turn_penalty",
        "anti_hacking_reward",
        "reconcile_r_phys",
    } <= names


def test_reward_status_classification():
    by = {t["name"]: t for t in ca.build_reward_contract()["terms"]}
    # 生产消费 (engine_reflect 直接 import).
    assert by["anti_hacking_reward"]["status"] == "wired"
    # 仅模块内经 anti_hacking_reward 组合调用.
    assert by["strict_scope_reward"]["status"] == "internal-only"
    assert by["efficiency_discount"]["status"] == "internal-only"
    assert by["idle_turn_penalty"]["status"] == "internal-only"
    assert by["numeric_accuracy_reward"]["status"] == "internal-only"


def test_reconcile_r_phys_not_borrowed_from_sibling_module():
    """关键正确性: claim_reward.reconcile_r_phys 不得"借"走 world_state 同名的调用点.

    裸名扫描会把 security/world_state.py 的生产调用算到 claim_reward 头上, 误报
    wired. 模块限定归属后, claim_reward 侧的 prod 必须为 0.
    """
    by = {t["name"]: t for t in ca.build_reward_contract()["terms"]}
    assert by["reconcile_r_phys"]["prod"] == 0
    assert by["reconcile_r_phys"]["status"] == "dead"


def test_penalty_axis_overlap_detected():
    axes = {a["axis"]: a for a in ca.build_reward_contract()["penalty_axes"]}
    assert axes["轮次"]["overlap"] is True
    assert set(axes["轮次"]["terms"]) == {"efficiency_discount", "idle_turn_penalty"}
    assert axes["授权越界"]["overlap"] is False


def test_cross_module_duplicate_flagged():
    dupes = {d["name"]: d for d in ca.build_reward_contract()["cross_module_dupes"]}
    assert "reconcile_r_phys" in dupes
    assert "huginn/security/world_state.py" in dupes["reconcile_r_phys"]["modules"]


# ──────────────────── 真实仓: 授权面 ────────────────────


def test_scope_sources_and_flags():
    c = ca.build_scope_contract()
    by = {s["name"]: s for s in c["sources"]}
    assert by["compute_authorized_ratio"]["status"] == "wired"  # S1 合规口径
    assert by["compute_intent_ratio"]["status"] == "wired"  # S2 意图口径
    # 两口径独立开关注册齐备 (MECE: 口径互斥且可独立启用).
    assert c["flags"]["anti_hacking_reward"]["registered"] is True
    assert c["flags"]["intent_scope_reward"]["registered"] is True


def test_find_issues_reports_expected_categories():
    issues = ca.find_issues(ca.build_mece_snapshot())
    joined = "\n".join(issues)
    assert "reconcile_r_phys" in joined
    assert "同轴惩罚候选" in joined
    assert "跨模块同名" in joined


def test_render_contains_sections():
    md = ca.render_mece_markdown(ca.build_mece_snapshot())
    assert "奖励面" in md
    assert "授权面" in md
    assert "发现汇总" in md


# ──────────────────── 真实仓: 工作流面 ────────────────────


def test_workflow_dispatch_matches_hardcoded_branches():
    """登记面 (dispatch_table) 与执行面 (engine_act 硬编码分支) 必须穷尽一致.

    防回归: 审计曾把 phase_spec `_selfcheck` 里的 `custom_mode` override fixture
    误当契约, 假报"登记面有而执行面没有"。剥掉自测块后两者都应是 6 个真实 mode。
    """
    wf = ca.build_workflow_contract()
    assert wf["branch_matches_dispatch"] is True
    assert "custom_mode" not in wf["dispatch"]


def test_workflow_planner_prompts_consistent():
    """planner 多处 MODE 提示必须一致 (visual_inspect 曾在主 prompt 格式行漏列)。"""
    wf = ca.build_workflow_contract()
    assert wf["prompt_inconsistent"] is False
    assert "visual_inspect" in wf["reachable_from_prompt"]


def test_selftest_block_excluded_from_dispatch(tmp_path):
    """自测块里的 mode fixture 不得计入登记面 (合成树验证剥块逻辑)。"""
    _write(
        tmp_path,
        ca._EXEC_SPEC_REL,
        'dispatch_table={\n    "coder": ["_execute_coder", "description"],\n}\n'
        "def _selfcheck() -> None:\n"
        '    reg.register_phase_override("_execute", {"dispatch_table": '
        '{"ghost": ["_execute_explore", "description"]}})\n',
    )
    _write(tmp_path, ca._ENGINE_ACT_REL, 'if mode == "coder":\n    pass\n')
    _write(tmp_path, ca._PLAN_CHECK_REL, "MODE: <coder>\n")

    wf = ca.build_workflow_contract(tmp_path)
    assert wf["dispatch"] == ["coder"]
    assert "ghost" not in wf["dispatch"]


# ──────────────────── 词汇面 ────────────────────


def test_vocab_private_all_caps_names_are_declared():
    """回归: 带下划线前缀的全大写词表名 (`_KINDS`/`_NEGATIVE_WORDS`) 也算声明枚举.

    旧正则 `^[A-Z][A-Z0-9_]*$` 拒首字符 `_`, 把私有词表误判为函数内局部临时集合,
    整面漏报 (闭集簇 108 → 159)。小写局部元组 (`required`) 仍不算声明。
    """
    assert ca._VOCAB_ALL_CAPS_RE.match("_KINDS")
    assert ca._VOCAB_ALL_CAPS_RE.match("_NEGATIVE_WORDS")
    assert not ca._VOCAB_ALL_CAPS_RE.match("required")


def test_vocab_mapping_roundtrip_loss_detected():
    """非单射 + 声明了反向表 ⇒ 往返丢信息: `AUTOLOOP_TO_PHASE` 把 learn·validate
    共像到 `ResearchPhase.VALIDATION`, `PHASE_TO_AUTOLOOP` 反向只能还原一个。"""
    maps = {m["name"]: m for m in ca.build_vocabulary_contract()["mappings"]}
    m = maps["AUTOLOOP_TO_PHASE"]
    assert m["injective"] is False
    assert m["reverse_present"] is True
    assert len(m["collisions"]) == 1


def test_vocab_memory_type_drift_detected():
    """同名跨模块定义且值域不等: `memory/types.py::MemoryType` 是 `typing.py` 的子集
    (前者缺 cross_domain_transfer 等 5 词) —— mutually exclusive 的"词表漂移"。"""
    c = ca.build_vocabulary_contract()
    dups = {d["name"]: d for d in c["duplicate_defs"]}
    assert dups["class MemoryType"]["same_values"] is False
    assert any(
        {m["name"] for m in d["members"]} == {"class MemoryType"} for d in c["divergence"]
    )


def test_vocab_render_contains_mapping_and_collision_tables():
    md = ca.render_vocabulary_markdown(ca.build_vocabulary_contract())
    assert "词汇面" in md
    assert "未登记撞名" in md
    assert "AUTOLOOP_TO_PHASE" in md


def test_vocab_local_tuple_not_treated_as_vocabulary(tmp_path):
    """合成树: 函数内同名局部元组不算词表, 模块级全大写元组才算 (旧版把
    `required = ("location", ...)` 误报成跨模块同名定义)。"""
    _write(tmp_path, "huginn/a.py", 'def f():\n    required = ("a", "b", "c")\n')
    _write(tmp_path, "huginn/b.py", 'def g():\n    required = ("x", "y", "z")\n')
    _write(tmp_path, "huginn/c.py", 'KINDS = ("alpha", "beta", "gamma")\n')
    _write(tmp_path, "huginn/d.py", 'KINDS = ("delta", "epsilon", "zeta")\n')

    names = {d["name"] for d in ca.build_vocabulary_contract(tmp_path)["duplicate_defs"]}
    assert "KINDS" in names
    assert "required" not in names


# ──────────────────── 真实仓: 工具面 ────────────────────


def test_tool_registry_specs_all_resolve():
    """注册清单 `_CORE_MODULES`/`_OPTIONAL_MODULES` 引用的类必须静态可解析.

    解析不到 ⇒ 该工具根本没注册成功 (import 名或类名写错).
    """
    c = ca.build_tool_contract()
    assert c["unresolved_specs"] == []
    assert c["spec_count"] > 100


def test_tool_registry_names_unique():
    """同名工具名由多类声明 ⇒ 注册表里后者覆盖前者 (静默丢工具)."""
    assert ca.build_tool_contract()["duplicate_names"] == []


def test_tool_allowlists_have_no_dead_items():
    """真实允许面已无死项 — `READ_ONLY_TOOLS` 的短名 (`read_file`/`list_dir`) 等已改回
    注册名 (`file_read_tool`/`grep`/`glob`).

    死项 ⇒ `set_mode` 永不命中, sidecar 的 auto_approve 对该读工具失效 —— 曾是真实缺陷.
    """
    c = ca.build_tool_contract()
    by = {a["name"]: a for a in c["allowlists"]}
    ro = by["READ_ONLY_TOOLS"]
    assert ro["namespace"] == "registry"
    assert ro["phantoms"] == []
    assert ro["matched"] == ro["size"]
    offenders = {
        f"{a['rel']}::{a['name']}": a["phantoms"]
        for a in c["allowlists"]
        if a["namespace"] == "registry" and a["phantoms"]
    }
    assert offenders == {}


def test_tool_allowlist_bare_name_alias_not_dead():
    """裸名↔`_tool` 别名算别名不算死项 (`_DFT_MD_TOOLS` 的 `vasp`↔`vasp_tool`)."""
    by = {a["name"]: a for a in ca.build_tool_contract()["allowlists"]}
    dft = by["_DFT_MD_TOOLS"]
    assert "vasp" in dft["aliases"]
    assert "vasp" not in dft["phantoms"]


def test_tool_external_namespace_not_flagged_dead():
    """与注册名零重叠的白名单 (MCP 外部工具名) 整表判外部命名空间, 不报死项."""
    by = {a["name"]: a for a in ca.build_tool_contract()["allowlists"]}
    mcp = by["_HIGH_VALUE_MCP_TOOLS"]
    assert mcp["namespace"] == "external"
    assert mcp["phantoms"] == []
    assert mcp["aliases"] == []


def test_tool_declared_classes_all_registered():
    """声明了 `name` 的 HuginnTool 子类都要在注册清单里 (宣称即注册).

    回归: `PyBulletTool` 曾声明 `name` 却不在 `_OPTIONAL_MODULES`, 工具永不进注册表.
    """
    assert ca.build_tool_contract()["unregistered_classes"] == []


def test_tool_render_sections_present():
    md = ca.render_tool_markdown(ca.build_tool_contract())
    assert "工具面" in md
    assert "死项" in md
    assert "注册声明缺口" in md
    # 允许面已收敛: find_issues 不再报工具面死项.
    assert "允许表死项" not in "\n".join(ca.find_issues(ca.build_mece_snapshot()))


def test_tool_synthetic_alias_phantom_and_external(tmp_path):
    """合成树: 别名/真死项/外部命名空间三类互不混淆."""
    _write(
        tmp_path,
        "huginn/tools/vasp_tool.py",
        'class VaspTool(HuginnTool):\n    name = "vasp_tool"\n',
    )
    _write(
        tmp_path,
        "huginn/allow.py",
        'A_TOOLS = {"vasp", "ghost_tool"}\nB_TOOLS = {"mcp_a", "mcp_b"}\n',
    )
    by = {a["name"]: a for a in ca.build_tool_contract(tmp_path)["allowlists"]}
    assert "vasp" in by["A_TOOLS"]["aliases"]  # vasp ↔ vasp_tool
    assert "ghost_tool" in by["A_TOOLS"]["phantoms"]  # 真死项
    assert by["B_TOOLS"]["namespace"] == "external"  # 零重叠 → 外部命名空间
    assert by["B_TOOLS"]["phantoms"] == []


def test_tool_synthetic_cross_module_consumer_marks_wired(tmp_path):
    """合成树: 别的模块 import 白名单 ⇒ wired; 仅定义文件内引用 ⇒ internal-only."""
    _write(
        tmp_path,
        "huginn/tools/vasp_tool.py",
        'class VaspTool(HuginnTool):\n    name = "vasp_tool"\n',
    )
    _write(tmp_path, "huginn/allow.py", 'A_TOOLS = {"vasp_tool"}\nB_TOOLS = {"vasp_tool"}\n')
    _write(tmp_path, "huginn/consumer.py", "from huginn.allow import A_TOOLS\nX = A_TOOLS\n")
    by = {a["name"]: a for a in ca.build_tool_contract(tmp_path)["allowlists"]}
    assert by["A_TOOLS"]["status"] == "wired"
    assert by["B_TOOLS"]["status"] == "dead"


# ──────────────────── 真实仓: 钩子面 ────────────────────


_REQUIRED_HOOK_EVENTS = {
    "PRE_TOOL_USE",
    "POST_TOOL_USE",
    "SESSION_START",
    "SESSION_END",
    "STOP",
    "SUBAGENT_STOP",
    "PRE_COMPACT",
    "POST_COMPACT",
    "USER_PROMPT_SUBMIT",
    "POST_TOOL_USE_FAILURE",
}
_HOOK_WIRED = {"PRE_TOOL_USE", "POST_TOOL_USE", "STOP", "USER_PROMPT_SUBMIT"}


def test_hook_declaration_face_exhaustive_and_mutex():
    """声明面 `ALL_EVENTS` 与事件常量定义面双向一致, 且值两两不同 (无撞值)."""
    c = ca.build_hook_contract()
    by = {e["const"]: e for e in c["events"]}
    assert set(by) >= _REQUIRED_HOOK_EVENTS
    assert c["collisions"] == []
    assert c["not_in_all_events"] == []
    assert c["unresolved_members"] == []


def test_hook_wired_events_have_trigger_and_consumer():
    by = {e["const"]: e for e in ca.build_hook_contract()["events"]}
    for k in _HOOK_WIRED:
        assert by[k]["status"] == "wired"
        assert by[k]["prod_trigger"] > 0 and by[k]["prod_register"] > 0


def test_hook_trigger_only_events_flagged():
    """6 个事件有触发点却零注册 —— 扩展点候选 (对偶于奖励面「宣称项零调用者」).

    回归锚点: `run_post` 补发的 `POST_TOOL_USE_FAILURE` 触发点在定义文件自身, 归属
    须把 `hooks/__init__.py` 的本地常量算进去, 否则会被漏成 `dead`.
    """
    by = {e["const"]: e for e in ca.build_hook_contract()["events"]}
    for k in _REQUIRED_HOOK_EVENTS - _HOOK_WIRED:
        assert by[k]["status"] == "trigger-only"
        assert by[k]["prod_trigger"] > 0 and by[k]["prod_register"] == 0
    # 该触发点自带 `if self._callbacks[...]` 守卫: 零注册 ⇒ 分支恒不执行.
    assert "守卫" in by["POST_TOOL_USE_FAILURE"]["note"]


def test_hook_find_issues_reports_trigger_only():
    joined = "\n".join(ca.find_issues(ca.build_mece_snapshot()))
    assert "钩子: 事件有生产触发点但零生产注册" in joined
    assert "POST_TOOL_USE_FAILURE" in joined


def test_hook_render_sections_present():
    md = ca.render_hook_markdown(ca.build_hook_contract())
    assert "钩子面" in md
    assert "互斥违例" in md
    assert "声明缺口" in md
    # 无绕过常量的字面量.
    assert "无绕过常量的字面量" in md


def test_hook_synthetic_same_name_event_not_borrowed(tmp_path):
    """合成树: 别模块同名的点分事件不得算作钩子事件引用 (归属须模块限定).

    `events/event_types.SESSION_START = "session.start"` 与钩子的 `"session_start"`
    同名不同值 —— 裸同名扫描会把它的注册点借给钩子面, 误报 wired.
    """
    _write(
        tmp_path,
        ca._HOOKS_MODULE,
        'PRE_TOOL_USE = "pre_tool_use"\n'
        'SESSION_START = "session_start"\n'
        'ALL_EVENTS = (PRE_TOOL_USE, SESSION_START,)\n',
    )
    _write(
        tmp_path,
        "huginn/elsewhere.py",
        'SESSION_START = "session.start"\n',
    )
    _write(
        tmp_path,
        "huginn/user.py",
        "from huginn.elsewhere import SESSION_START\n"
        "def f(hm):\n    hm.register(SESSION_START, cb)\n",
    )
    by = {e["const"]: e for e in ca.build_hook_contract(tmp_path)["events"]}
    assert by["SESSION_START"]["prod_register"] == 0
    assert by["SESSION_START"]["status"] == "dead"
    assert by["PRE_TOOL_USE"]["status"] == "dead"


def test_hook_synthetic_trigger_and_literal(tmp_path):
    """合成树: 模块限定解析触发点 (`from huginn.hooks import …`) + 字面量接线告警."""
    _write(
        tmp_path,
        ca._HOOKS_MODULE,
        'PRE_TOOL_USE = "pre_tool_use"\n'
        'SESSION_START = "session_start"\n'
        'ALL_EVENTS = (PRE_TOOL_USE, SESSION_START,)\n',
    )
    _write(
        tmp_path,
        "huginn/runner.py",
        "from huginn.hooks import SESSION_START\n"
        "async def go(hm):\n    await hm.trigger(SESSION_START, ctx)\n"
        'def bad(hm):\n    hm.register("pre_tool_use", cb)\n',
    )
    c = ca.build_hook_contract(tmp_path)
    by = {e["const"]: e for e in c["events"]}
    assert by["SESSION_START"]["prod_trigger"] == 1
    assert by["SESSION_START"]["status"] == "trigger-only"
    assert any(w["value"] == "pre_tool_use" for w in c["literal_wiring"])


# ──────────────────── 真实仓: 事件面 ────────────────────


_EVENT_REQUIRED = {
    "TOOL_CALL",
    "TOOL_RESULT",
    "TOOL_ERROR",
    "TOOL_BLOCKED",
    "COMPACT_START",
    "COMPACT_END",
    "CONTEXT_OVERFLOW",
    "PIPELINE_SUGGEST",
    "PIPELINE_STAGE_CHANGE",
    "CAMPAIGN_ITERATION",
    "CAMPAIGN_REFINE",
    "CAMPAIGN_HYPOTHESIS",
    "SNAPSHOT_TAKE",
    "SNAPSHOT_REVERT",
    "QUALITY_CHECK",
    "HEAT_ENGINE_HEALTH",
    "SESSION_START",
    "SESSION_END",
    "DECISION_POINT",
    "COST_NARRATIVE",
    "STEP_RETRY",
}


def test_event_declaration_face_exhaustive_and_mutex():
    """声明面 (点分常量定义) 与 `ALL_TYPES` 辅助清单双向一致, 且值两两不同."""
    c = ca.build_event_contract()
    by = {e["const"]: e for e in c["events"]}
    assert set(by) >= _EVENT_REQUIRED
    assert c["collisions"] == []
    assert c["not_in_all_types"] == []
    assert c["unresolved_members"] == []


def test_event_declared_types_all_have_producers():
    """collectively exhaustive: 21 个声明类型全部有生产发布点 — 无孤儿类型.

    与钩子面 (6 个 trigger-only) 相反: 事件面的声明清单是"已投产"清单, 不藏扩展点.
    """
    events = ca.build_event_contract()["events"]
    assert events  # 非空, 否则下面的全量断言空转
    for e in events:
        assert e["prod_publish"] > 0, f"{e['const']} ({e['value']}) 零生产发布"
    assert {e["status"] for e in events} == {"published"}


def test_event_declared_subscribers_from_loop_reverse_resolved():
    """`for evt_type in ("campaign.iteration", …)` 里的字面量元组要被反解成订阅点.

    audit_log.install_campaign_subscriber 用循环订阅, 静态解析若不做循环变量
    反解, 这四个已声明类型会被误判成"仅发布无消费者".
    """
    by = {e["const"]: e for e in ca.build_event_contract()["events"]}
    for k in ("CAMPAIGN_ITERATION", "CAMPAIGN_HYPOTHESIS", "CAMPAIGN_REFINE", "QUALITY_CHECK"):
        assert by[k]["prod_subscribe"] == 1, k
        assert any(s.startswith("huginn/events/audit_log.py") for s in by[k]["subscribe_sites"])


def test_event_undeclared_publishers_surfaced():
    """发布/订阅了却无常量的类型作为候选缺口列出 (设计允许 `ALL_TYPES` 非穷尽)."""
    c = ca.build_event_contract()
    by = {u["value"]: u for u in c["undeclared"]}
    # 团队协作事件族整族未登记常量 (只发不收).
    for v in ("team.run.start", "team.run.done", "team.member.start", "embedding.download.start"):
        assert by[v]["prod_publish"] > 0, v
        assert by[v]["prod_subscribe"] == 0, v
    # campaign.retry: 未登记常量, 但被 audit_log 的 `for` 循环订阅反解命中 (发+收两侧).
    assert by["campaign.retry"]["prod_publish"] > 0
    assert by["campaign.retry"]["prod_subscribe"] > 0
    assert any(
        s.startswith("huginn/events/audit_log.py") for s in by["campaign.retry"]["subscribe_sites"]
    )


def test_event_wildcard_subscriber_detected():
    """`bus.subscribe(ALL, cb)` 是收全量的通配订阅, 单列不摊进每行计数."""
    c = ca.build_event_contract()
    assert any(s.startswith("huginn/events/audit_log.py") for s in c["wildcard_subscribers"])
    assert c["wildcard_subscribers"] == sorted(c["wildcard_subscribers"])


def test_event_find_issues_reports_undeclared():
    joined = "\n".join(ca.find_issues(ca.build_mece_snapshot()))
    assert "事件: 发布了未声明类型" in joined
    assert "team.run.start" in joined
    # 声明面已收敛: 不报死项/订阅孤儿.
    assert "事件: 类型声明零生产发布零订阅" not in joined
    assert "事件: 类型有 .subscribe 却零生产发布" not in joined


def test_event_render_sections_present():
    md = ca.render_event_markdown(ca.build_event_contract())
    assert "事件面" in md
    assert "未声明类型" in md
    assert "互斥违例 + 声明缺口" in md
    assert "诚实边界" in md


# ──────────────────── 合成树: 事件面 ────────────────────


def test_event_synthetic_status_classification(tmp_path):
    """合成树: published / subscribed-only / dead 三态互斥分类."""
    _write(
        tmp_path,
        ca._EVENTS_MODULE,
        'TOOL_CALL = "tool.call"\n'
        'SESSION_START = "session.start"\n'
        'GHOST = "ghost.void"\n'
        'ALL = "*"\n'
        "ALL_TYPES = frozenset({TOOL_CALL, SESSION_START, GHOST})\n",
    )
    _write(
        tmp_path,
        "huginn/consumer.py",
        "from huginn.events.event_types import TOOL_CALL, SESSION_START\n"
        "def go(bus, cb):\n"
        "    bus.publish_event(TOOL_CALL, {})\n"
        "    bus.subscribe(TOOL_CALL, cb)\n"
        "    bus.subscribe(SESSION_START, cb)\n",
    )
    by = {e["const"]: e for e in ca.build_event_contract(tmp_path)["events"]}
    assert by["TOOL_CALL"]["status"] == "published"
    assert by["SESSION_START"]["status"] == "subscribed-only"
    assert by["GHOST"]["status"] == "dead"


def test_event_synthetic_loop_var_and_literal_undeclared(tmp_path):
    """合成树: `for t in <常量集合>` 反解订阅 + 字面量发布归入未声明候选."""
    _write(
        tmp_path,
        ca._EVENTS_MODULE,
        'TOOL_CALL = "tool.call"\nALL = "*"\nALL_TYPES = frozenset({TOOL_CALL})\n',
    )
    _write(
        tmp_path,
        "huginn/sub.py",
        "from huginn.events.event_types import TOOL_CALL\n"
        "_WATCHED = (TOOL_CALL,)\n"
        "def wire(bus, cb):\n"
        "    for t in _WATCHED:\n"
        "        bus.subscribe(t, cb)\n"
        '    bus.publish_generic_sync("zeta.new", {})\n',
    )
    c = ca.build_event_contract(tmp_path)
    by = {e["const"]: e for e in c["events"]}
    assert by["TOOL_CALL"]["prod_subscribe"] == 1
    assert by["TOOL_CALL"]["status"] == "subscribed-only"
    und = {u["value"]: u for u in c["undeclared"]}
    assert und["zeta.new"]["prod_publish"] == 1


def test_event_synthetic_module_alias_and_wildcard(tmp_path):
    """合成树: `import huginn.events.event_types as et` 的 `et.X` 归属, 及 `ALL` 通配."""
    _write(
        tmp_path,
        ca._EVENTS_MODULE,
        'TOOL_CALL = "tool.call"\nALL = "*"\nALL_TYPES = frozenset({TOOL_CALL})\n',
    )
    _write(
        tmp_path,
        "huginn/wire.py",
        "import huginn.events.event_types as et\n"
        "def go(bus, cb):\n"
        "    bus.subscribe(et.TOOL_CALL, cb)\n"
        "    bus.subscribe(et.ALL, cb)\n",
    )
    c = ca.build_event_contract(tmp_path)
    by = {e["const"]: e for e in c["events"]}
    assert by["TOOL_CALL"]["prod_subscribe"] == 1
    assert any(s.startswith("huginn/wire.py") for s in c["wildcard_subscribers"])


def test_event_synthetic_collision_and_all_types_gaps(tmp_path):
    """合成树: 撞值 (mutually exclusive 违例) 与 `ALL_TYPES` 两侧缺口的三个方向."""
    _write(
        tmp_path,
        ca._EVENTS_MODULE,
        'A = "dup.value"\n'
        'B = "dup.value"\n'
        'C = "only.here"\n'
        'ALL = "*"\n'
        "ALL_TYPES = frozenset({A, B, GHOST})\n",
    )
    c = ca.build_event_contract(tmp_path)
    assert c["collisions"] == [{"value": "dup.value", "consts": ["A", "B"]}]
    assert c["not_in_all_types"] == ["C"]
    assert c["unresolved_members"] == ["GHOST"]


# ──────────────────── 文档漂移 ────────────────────


def test_mece_audit_doc_not_drifted():
    """提交的 docs/mece-audit.md == 当前代码重渲染, 防契约文档漂移.

    代码改了 (奖励项增删 / 接线状态变化 / 惩罚轴变更) 而文档没重生成即失败,
    提醒跑 `python -m huginn.cli.contract_audit --out docs/mece-audit.md`。
    """
    committed = (_REPO / "docs" / "mece-audit.md").read_text(encoding="utf-8")
    fresh = ca.render_mece_markdown(ca.build_mece_snapshot())
    assert committed == fresh, (
        "docs/mece-audit.md 与代码漂移, 请重新生成: "
        "`python -m huginn.cli.contract_audit --out docs/mece-audit.md`"
    )


def test_mece_audit_doc_referenced_in_index():
    """docs/INDEX.md 应登记 mece-audit.md, 防止文档不被发现."""
    index = (_REPO / "docs" / "INDEX.md").read_text(encoding="utf-8")
    assert "mece-audit.md" in index


# ──────────────────── 合成树: 模块限定归属 ────────────────────


def _write(root: pathlib.Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_module_qualified_attribution_isolates_same_name(tmp_path):
    _write(
        tmp_path,
        ca._REWARD_MODULE,
        '__all__ = ["shared_name", "only_here"]\n'
        "def shared_name():\n    return 1\n"
        "def only_here():\n    return 2\n",
    )
    _write(tmp_path, "huginn/elsewhere.py", "def shared_name():\n    return 99\n")
    _write(
        tmp_path,
        "huginn/consumer.py",
        "from huginn.validation.claim_reward import only_here\nX = only_here()\n",
    )
    # 裸同名调用指向 elsewhither, 不应污染 claim_reward 的公开面.
    _write(
        tmp_path,
        "huginn/other_user.py",
        "from huginn.elsewhere import shared_name\nY = shared_name()\n",
    )

    c = ca.build_reward_contract(tmp_path)
    by = {t["name"]: t for t in c["terms"]}
    assert by["only_here"]["status"] == "wired"
    assert by["shared_name"]["status"] == "dead"
    assert any(d["name"] == "shared_name" for d in c["cross_module_dupes"])
