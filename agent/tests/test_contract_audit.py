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


# ──────────────────── 真实仓: SSE 消费面 ────────────────────


def test_sse_real_repo_listeners_all_resolved():
    """真实仓每个 `addEventListener` 帧监听都归位: 要么 wired, 要么 DOM (external).

    防回归: 帧名只在**发它的那条 EventSource** 上才命中, 故监听点必须按通道核;
    出现 `channel-mismatch` / `no-source` 即前端挂在永不触发的通道上.
    """
    c = ca.build_sse_contract()
    assert c["listeners"]  # 非空, 否则下面的全量断言空转
    for s in c["listeners"]:
        assert s["status"] in {"wired", "external"}, (s["frame"], s["status"])
        if s["status"] == "external":
            assert s["channel"] is None


def test_sse_real_repo_channels_and_bus_producers():
    """event_bus 通道帧名面 = 总线生产发布的事件值面 (穷尽); progress 取字面帧名."""
    c = ca.build_sse_contract()
    frames = c["channels"]["event_bus"]["frames"]
    assert frames == sorted(set(frames))  # 无重复
    published = {
        e["value"] for e in ca.build_event_contract()["events"] if e["prod_publish"]
    }
    # 已声明且有生产发布的事件, 必须都在 event_bus 通道帧名面里.
    assert published <= set(frames)
    # 未登记常量但已生产发布的团队事件族, 也应出现在帧名面 (事件面报的候选).
    assert {"team.run.start", "team.run.done", "campaign.retry"} <= set(frames)
    # progress 通道取 `interaction/progress.py` 的字面 `event:` 行.
    assert c["channels"]["progress"]["frames"] == ["campaign", "heartbeat", "snapshot", "update"]
    # pet 通道只发无名帧, 无命名帧.
    assert c["channels"]["pet"]["frames"] == []


def test_sse_real_repo_progress_listeners_wired():
    """autoloop 三条命名帧 (snapshot/update/campaign) 必须挂在 progress 通道上.

    历史缺陷: 前端曾用 `es.onmessage` 收 autoloop, 后端却发命名帧 ⇒ 永不触发.
    """
    by = {(s["frame"], s["channel"]): s["status"] for s in ca.build_sse_contract()["listeners"]}
    for frame in ("snapshot", "update", "campaign"):
        assert by[(frame, "progress")] == "wired", frame


def test_sse_real_repo_payload_sources_classified():
    """前端 payload 字段匹配的事件名要能溯源到生产面, 且无静态不可见 (unknown)."""
    c = ca.build_sse_contract()
    by = {(p["value"], p["channel"]): p["source"] for p in c["payloads"]}
    # campaign 帧族的 payload 经 `campaign` 帧隧道 (progress 通道) 进来, 但值本身由总线发布.
    assert by[("heat_engine.health", "progress")] == "bus"
    # plan.* 只在 `emit_campaign_event(event_type="…")` 处静态可见.
    assert by[("plan.exec_start", "progress")] == "campaign"
    assert by[("team.run.start", "event_bus")] == "bus"
    assert not any(p["source"] == "unknown" for p in c["payloads"])


def test_sse_real_repo_no_hard_violations_in_find_issues():
    """真实仓无监听挂错通道 / 监听无源帧 —— 发现汇总里不应出现 SSE 硬违例."""
    joined = "\n".join(ca.find_issues(ca.build_mece_snapshot()))
    assert "SSE: 监听挂错通道" not in joined
    assert "SSE: 监听的后端无此帧名" not in joined


def test_sse_render_sections_present():
    md = ca.render_sse_markdown(ca.build_sse_contract())
    assert "SSE 消费面" in md
    assert "前端帧监听" in md
    assert "生产帧名零前端监听" in md
    assert "前端 payload 字段匹配的事件名" in md
    assert "诚实边界" in md


# ──────────────────── 合成树: SSE 消费面 ────────────────────


def _sse_backend(tmp_path, frames: str, extra: str = "") -> None:
    """合成后端: progress 字面帧名 + 事件类型声明 + 一个总线发布点."""
    _write(tmp_path, ca._SSE_FRAME_MODULE, frames)
    _write(
        tmp_path,
        ca._EVENTS_MODULE,
        'TOOL_CALL = "tool.call"\nALL = "*"\nALL_TYPES = frozenset({TOOL_CALL})\n',
    )
    _write(
        tmp_path,
        "huginn/pub.py",
        "from huginn.events.event_types import TOOL_CALL\n"
        "def go(bus):\n"
        "    bus.publish_event(TOOL_CALL, {})\n"
        + extra,
    )


def test_sse_synthetic_channel_attribution_and_status(tmp_path):
    """合成树: 帧监听按通道归属, 四态 (wired/mismatch/no-source/external) 互斥."""
    _sse_backend(tmp_path, 'A = "event: update"\nB = "event: heartbeat"\n')
    _write(
        tmp_path,
        "fe/app.ts",
        "const es = new EventSource(`${API_BASE}/tasks/stream`);\n"
        "const onUpdate = (e: MessageEvent) => {\n"
        "  const t = JSON.parse(e.data);\n"
        "};\n"
        'es.addEventListener("update", onUpdate);\n'
        'es.addEventListener("tool.call", onUpdate);\n'
        'es.addEventListener("ghost.void", onUpdate);\n'
        'window.addEventListener("keydown", onUpdate);\n',
    )
    c = ca.build_sse_contract(tmp_path, tmp_path / "fe")
    assert c["channels"]["progress"]["frames"] == ["heartbeat", "update"]
    assert c["channels"]["event_bus"]["frames"] == ["tool.call"]
    by = {s["frame"]: s for s in c["listeners"]}
    assert by["update"]["status"] == "wired"
    assert by["update"]["channel"] == "progress"
    # tool.call 是 event_bus 通道的帧名, 挂在 progress 通道上 ⇒ 永不触发.
    assert by["tool.call"]["status"] == "channel-mismatch"
    assert "event_bus" in by["tool.call"]["note"]
    assert by["ghost.void"]["status"] == "no-source"
    assert by["keydown"]["status"] == "external"
    assert by["keydown"]["channel"] is None


def test_sse_synthetic_array_literal_frames(tmp_path):
    """合成树: `[\"a.b\", \"c.d\"].forEach((ev) => es.addEventListener(ev, h))` 两帧都归属该通道."""
    _sse_backend(tmp_path, "")
    _write(
        tmp_path,
        "fe/bus.ts",
        "const es = new EventSource(`${API_BASE}/events/stream`);\n"
        "const handleX = (e: MessageEvent) => {\n"
        "  const t = JSON.parse(e.data);\n"
        "};\n"
        '["tool.call", "ghost.void"]\n'
        "  .forEach((ev) => es.addEventListener(ev, handleX));\n",
    )
    c = ca.build_sse_contract(tmp_path, tmp_path / "fe")
    by = {s["frame"]: s for s in c["listeners"]}
    assert by["tool.call"]["status"] == "wired"
    assert by["tool.call"]["channel"] == "event_bus"
    assert by["ghost.void"]["status"] == "no-source"


def test_sse_synthetic_payload_sources_and_zero_consumer(tmp_path):
    """合成树: payload 事件名四源 (bus/campaign/declared/unknown) + 零监听帧按 payload 消费标记."""
    _sse_backend(
        tmp_path,
        'A = "event: heartbeat"\n',
        extra=(
            "def emit(bus):\n"
            "    bus.emit_campaign_event(event_type=\"plan.exec_start\")\n"
        ),
    )
    _write(
        tmp_path,
        ca._EVENTS_MODULE,
        'TOOL_CALL = "tool.call"\n'
        'DECISION_POINT = "decision.point"\n'
        'ALL = "*"\n'
        "ALL_TYPES = frozenset({TOOL_CALL, DECISION_POINT})\n",
    )
    _write(
        tmp_path,
        "fe/bus.ts",
        "const es = new EventSource(`${API_BASE}/events/stream`);\n"
        "const handleBus = (e: MessageEvent) => {\n"
        "  const t = JSON.parse(e.data);\n"
        '  if (t.type === "tool.call") { }\n'
        '  else if (t.type === "decision.point") { }\n'
        '  else if (t.type === "plan.exec_start") { }\n'
        '  else if (t.type === "mystery.void") { }\n'
        "};\n"
        'es.addEventListener("ghost.void", handleBus);\n',
    )
    c = ca.build_sse_contract(tmp_path, tmp_path / "fe")
    by = {(p["value"], p["channel"]): p["source"] for p in c["payloads"]}
    assert by[("tool.call", "event_bus")] == "bus"
    assert by[("plan.exec_start", "event_bus")] == "campaign"
    assert by[("decision.point", "event_bus")] == "declared"
    assert by[("mystery.void", "event_bus")] == "unknown"
    assert "plan.exec_start" in c["campaign_literals"]
    zero = {(z["channel"], z["frame"]): z["via_payload"] for z in c["zero_consumer"]}
    # tool.call 有生产发布但无帧监听 —— 只在 payload 字段里被匹配.
    assert zero[("event_bus", "tool.call")] is True
    # heartbeat 既无帧监听也无 payload 匹配.
    assert zero[("progress", "heartbeat")] is False
    # 未生产发布的常量不进零监听候选 (它压根没帧).
    assert ("event_bus", "decision.point") not in zero


# ──────────────────── 真实仓: WS 消费面 ────────────────────


def test_ws_registry_parsed_from_annotated_assignment():
    """回归: `_MESSAGE_HANDLERS: dict[str, Any] = {…}` 是带注解赋值.

    只按 `ast.Assign` 解析会读空注册表, 把全部前端入站类型误报 unhandled.
    """
    keys = set(ca._ws_registry_keys(_REPO))
    assert {"user_input", "plan_confirm", "approval_response", "guide"} <= keys
    inbound = ca._ws_inbound_types(_REPO)["agent"]
    for k in ("user_input", "plan_confirm", "approval_response"):
        assert k in inbound


def test_ws_server_frames_include_variable_assigned_frame():
    """回归: 帧名先赋给变量再发送 (`tool_result_msg = {…}` 后 `_ws_send(tool_result_msg)`).

    只扫 send 实参字面量会漏掉它, 误判前端 case "tool_result" 无源.
    """
    frames = ca._ws_server_frames(_REPO)["agent"]["frames"]
    assert "tool_result" in frames
    assert "text_delta" in frames


def test_ws_real_repo_consumers_all_resolved():
    """真实仓: 每个前端 WS type 判别都归属到通道, 且无一挂到不发的帧名上."""
    c = ca.build_ws_contract()
    assert c["consumers"], "应扫到 agent 通道的 type 判别点"
    assert not [x for x in c["consumers"] if x["status"] == "no-source"]
    assert not [x for x in c["consumers"] if x["channel"] is None]
    assert {x["status"] for x in c["consumers"]} <= {"wired", "dynamic"}


def test_ws_real_repo_sends_all_handled():
    """真实仓: 前端发送的入站类型均被后端分发表认."""
    c = ca.build_ws_contract()
    assert c["sends"]
    assert not [x for x in c["sends"] if x["status"] == "unhandled"]


def test_ws_non_type_switch_not_mistaken_for_frames():
    """回归: Pet.tsx 的 `switch (mood)` 用同样 case 标签, 但不是 WS 帧名.

    真实仓判别点不得出现 `thinking` / `happy` / `levelup` 等宠物心情.
    """
    frames = {x["frame"] for x in ca.build_ws_contract()["consumers"]}
    assert frames.isdisjoint({"thinking", "happy", "levelup", "hungry", "eating"})


def test_ws_sse_and_search_type_checks_not_mistaken():
    """回归: SSE 的 `data.type === "heartbeat"` 与搜索结果的 `result.type === "thread"`.

    这些 `.type` 判别不以 WSMessage 标注变量为对象, 不得进 WS 判别面.
    """
    frames = {x["frame"] for x in ca.build_ws_contract()["consumers"]}
    assert frames.isdisjoint({"heartbeat", "state", "event", "thread", "memory", "knowledge"})


def test_ws_real_repo_no_hard_violations_in_find_issues():
    """真实仓无挂错帧名 / 未认入站类型 —— 发现汇总里不应出现 WS 硬违例."""
    joined = "\n".join(ca.find_issues(ca.build_mece_snapshot()))
    assert "WS: 前端 type 判别的后端不发此帧名" not in joined
    assert "WS: 前端发送的入站类型后端分发面不认" not in joined


def test_ws_render_sections_present():
    md = ca.render_ws_markdown(ca.build_ws_contract())
    assert "WS 消费面" in md
    assert "生产帧名 × 前端判别" in md
    assert "前端 server→client 判别点" in md
    assert "前端 client→server 发送点" in md
    assert "已确认有意" in md
    assert "诚实边界" in md


def test_ws_candidates_all_triaged():
    """每个 WS 候选都须在 `_WS_CONFIRMED_INTENTIONAL` 分诊为"已确认有意".

    候选是三分类信号, 不是硬违例 —— 但必须逐条确认它是缺陷还是有意的公开
    API/声明面产物. 新出现的未分诊候选即失败, 逼人工落地判定.
    """
    c = ca.build_ws_contract()
    untriaged = []
    for f in c["zero_consumer"]:
        if ca._ws_triage("zero-consumer", "agent", f) is None:
            untriaged.append(("zero-consumer", "agent", f))
    for f in c["undeclared"]:
        if ca._ws_triage("undeclared", "agent", f) is None:
            untriaged.append(("undeclared", "agent", f))
    for f in c["declared_only"]:
        if ca._ws_triage("declared-only", "agent", f) is None:
            untriaged.append(("declared-only", "agent", f))
    for z in c["phantom_inbound"]:
        if ca._ws_triage("phantom-inbound", z["channel"], z["frame"]) is None:
            untriaged.append(("phantom-inbound", z["channel"], z["frame"]))
    assert not untriaged, f"未分诊的 WS 候选: {untriaged}"


# ──────────────────── 合成树: WS 消费面 ────────────────────


def test_ws_synthetic_channel_and_status(tmp_path):
    """合成树: 双向四态 (wired/no-source · handled/unhandled) 与通道归属."""
    _write(
        tmp_path,
        ca._WS_REGISTRY_REL,
        "from typing import Any\n"
        "_MESSAGE_HANDLERS: dict[str, Any] = {\n"
        '    "user_input": h,\n'
        '    "plan_confirm": h,\n'
        "}\n\n"
        "async def go(websocket):\n"
        '    await websocket.send_json({"type": "welcome"})\n',
    )
    _write(
        tmp_path,
        "fe/chat.ts",
        'const url = "/ws/agent";\n'
        "const handle = (data: WSMessage) => {\n"
        "  switch (data.type) {\n"
        '    case "welcome": break;\n'
        '    case "ghost": break;\n'
        "  }\n"
        '  ws.send(JSON.stringify({ type: "user_input" }));\n'
        '  ws.send(JSON.stringify({ type: "nope" }));\n'
        "};\n",
    )
    c = ca.build_ws_contract(tmp_path, tmp_path / "fe")
    assert c["channels"]["agent"]["frames"] == ["welcome"]
    cons = {x["frame"]: x for x in c["consumers"]}
    assert cons["welcome"]["status"] == "wired"
    assert cons["welcome"]["channel"] == "agent"
    # ghost 不属于该通道生产面, 也不在 WSMessage 声明里 ⇒ 永不命中.
    assert cons["ghost"]["status"] == "no-source"
    sends = {x["frame"]: x for x in c["sends"]}
    assert sends["user_input"]["status"] == "handled"
    assert sends["nope"]["status"] == "unhandled"


def test_ws_synthetic_declared_frame_is_dynamic(tmp_path):
    """合成树: 已声明但只在后端变量透传转发的帧记为 dynamic (候选), 不判死."""
    _write(
        tmp_path,
        ca._WS_REGISTRY_REL,
        "async def go(websocket, state):\n"
        "    await websocket.send_json(dict(state))\n",
    )
    _write(
        tmp_path,
        "fe/types/ws.ts",
        'export type WSMessage = { type: "mode_banner" } | { type: "text_delta" };\n',
    )
    _write(
        tmp_path,
        "fe/chat.ts",
        'const url = "/ws/agent";\n'
        "const handle = (data: WSMessage) => {\n"
        "  switch (data.type) {\n"
        '    case "mode_banner": break;\n'
        "  }\n"
        "};\n",
    )
    c = ca.build_ws_contract(tmp_path, tmp_path / "fe")
    cons = {x["frame"]: x for x in c["consumers"]}
    assert cons["mode_banner"]["status"] == "dynamic"


# ──────────────────── 真实仓: HTTP API 消费面 ────────────────────


def test_http_backend_registry_fully_mounted():
    """真实仓: `ALL_ROUTERS` 挂全每个定义 APIRouter 的模块, 且无悬空别名."""
    reg = ca.build_http_contract()["registration"]
    assert reg["mounted"]
    assert reg["unmounted"] == [], reg["unmounted"]
    assert reg["dangling"] == []


def test_http_ws_endpoints_excluded_from_http_face():
    """WS 端点归 WS 消费面: 本面端点不得含 WEBSOCKET, 且如实计数."""
    c = ca.build_http_contract()
    assert c["ws_endpoint_count"] > 0
    for ep in c["endpoints"]:
        assert "WEBSOCKET" not in ep["methods"]


def test_http_real_repo_every_call_is_wired_or_triaged():
    """除已分诊硬违例外, 每个前端调用都命中后端注册的方法+路径."""
    c = ca.build_http_contract()
    bad = [x for x in c["calls"] if x["status"] not in {"wired", "external"}]
    assert bad == c["hard_violations"]
    assert c["hard_untriaged"] == [], c["hard_untriaged"]
    for v in c["hard_violations"]:
        assert v["triage"] in {"defect", "intentional"}
        assert v["triage_reason"]
    # 路由遮蔽是互斥违例, 真实仓目前无.
    assert c["duplicates"] == []


def test_http_confirmed_violations_registry_not_stale():
    """分诊表登记的每条都必须仍是真实硬违例 —— 修好后要同步删登记."""
    observed = {
        (v["method"], v["path"].split("?")[0])
        for v in ca.build_http_contract()["hard_violations"]
    }
    for key in ca._HTTP_CONFIRMED_VIOLATIONS:
        assert key in observed, f"分诊表登记 {key} 已不再是硬违例, 请删除登记"


def test_http_hard_violation_mark_labels_triage():
    """分诊标注: 未登记 → 待分诊; 已登记 → 对应标签 + 理由."""
    assert ca._http_triage("GET", "/nope/none") is None
    assert "待分诊" in ca._http_violation_mark("GET", "/nope/none")
    for (method, path), (label, _reason) in ca._HTTP_CONFIRMED_VIOLATIONS.items():
        mark = ca._http_violation_mark(method, path)
        assert "待分诊" not in mark
        assert ca._HTTP_TRIAGE_DOC[label] in mark


def test_http_render_sections_present():
    md = ca.render_http_markdown(ca.build_http_contract())
    assert "HTTP API 消费面" in md
    assert "前端调用点 (按状态)" in md
    assert "方法不符 (405)" in md
    assert "路由挂载面" in md
    assert "桌面零调用的路由模块" in md
    assert "诚实边界" in md


# ──────────────────── 合成树: HTTP API 消费面 ────────────────────


def test_http_synthetic_404_and_405(tmp_path):
    """合成树: wired / 404 死链 / 405 方法不符 / 绝对 URL 四态互斥."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        '@router.get("/list")\n'
        "async def thing_list():\n"
        "    return {}\n"
        '@router.post("/save")\n'
        "async def thing_save():\n"
        "    return {}\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "await api.get('/thing/list');\n"
        "await api.post('/thing/list');\n"
        "await api.get('/thing/ghost');\n"
        "await api.get('https://cdn.example/x.json');\n",
    )
    c = ca.build_http_contract(tmp_path, tmp_path / "fe")
    by = {(x["method"], x["path"]): x for x in c["calls"]}
    assert by[("GET", "/thing/list")]["status"] == "wired"
    assert by[("POST", "/thing/list")]["status"] == "method-mismatch"
    assert by[("GET", "/thing/ghost")]["status"] == "no-source"
    assert by[("GET", "https://cdn.example/x.json")]["status"] == "external"
    eps = {ep["path"]: ep for ep in c["endpoints"]}
    assert eps["/thing/list"]["called"] is True
    assert eps["/thing/save"]["called"] is False
    # 该模块有一个端点被调用 ⇒ 不进"桌面零调用模块".
    assert c["zero_modules"] == []
    # 两条硬违例都未登记 ⇒ 如实标"待分诊".
    assert {v["triage"] for v in c["hard_violations"]} == {"untriaged"}
    assert len(c["hard_untriaged"]) == 2


def test_http_synthetic_duplicate_and_unmounted(tmp_path):
    """合成树: 同 method+path 多模块注册 → 路由遮蔽; 未进 ALL_ROUTERS → 永不生效."""
    for name in ("a", "b", "c"):
        _write(
            tmp_path,
            f"huginn/routes/{name}.py",
            "from fastapi import APIRouter\n"
            "router = APIRouter()\n"
            '@router.get("/dup")\n'
            "async def dup():\n"
            "    return {}\n",
        )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.a import router as a_router\n"
        "from huginn.routes.b import router as b_router\n"
        "ALL_ROUTERS = [a_router, b_router]\n",
    )
    _write(tmp_path, "fe/a.ts", "await api.get('/dup');\n")
    c = ca.build_http_contract(tmp_path, tmp_path / "fe")
    assert c["duplicates"] == [
        {
            "method": "GET",
            "path": "/dup",
            "modules": ["huginn/routes/a.py", "huginn/routes/b.py"],
        }
    ]
    assert c["registration"]["mounted"] == ["a.router", "b.router"]
    assert c["registration"]["unmounted"] == ["c.router"]
    assert c["registration"]["dangling"] == []


def test_http_synthetic_dynamic_segments_and_method_override(tmp_path):
    """合成树: `/v1` 前缀剥离、`${id}` ↔ `{uid}` 互配、query 截断、getBlob 方法覆盖."""
    _write(
        tmp_path,
        "huginn/routes/user.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/user")\n'
        '@router.get("/{uid}/files")\n'
        "async def files(uid: str):\n"
        "    return {}\n"
        '@router.post("/{uid}/save")\n'
        "async def save(uid: str):\n"
        "    return {}\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.user import router as user_router\n"
        "ALL_ROUTERS = [user_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "await api.get(`/v1/user/${id}/files?x=1`);\n"
        "await api.getBlob(`/user/${id}/save`, { method: 'POST' });\n",
    )
    c = ca.build_http_contract(tmp_path, tmp_path / "fe")
    assert {x["status"] for x in c["calls"]} == {"wired"}
    assert all(ep["called"] for ep in c["endpoints"])
    assert c["hard_violations"] == []
    assert c["zero_modules"] == []


# ──────────────────── 请求负载面 ────────────────────


def test_payload_real_repo_no_untriaged_violations():
    """真实仓: 每个命中端点的调用, 静态可辨的负载都满足后端必填 (无待分诊)."""
    c = ca.build_payload_contract()
    assert c["wired_call_count"] > 0
    assert c["model_count"] > 0
    assert c["untriaged"] == [], c["untriaged"]
    for v in c["violations"]:
        assert v["triage"] in {"defect", "intentional"}
        assert v["triage_reason"]
    # 三个维度至少各自有可静态核对的样本, 否则本面等于空转.
    assert c["coverage"]["body_checked"] > 0
    assert c["coverage"]["query_checked"] > 0
    assert c["coverage"]["form_checked"] > 0


def test_payload_confirmed_violations_registry_not_stale():
    """分诊表登记的每条都必须仍是真实硬违例 —— 修好后要同步删登记."""
    observed = {
        (v["kind"], v["method"], v["path"].split("?")[0])
        for v in ca.build_payload_contract()["violations"]
    }
    for key in ca._PAYLOAD_CONFIRMED_VIOLATIONS:
        assert key in observed, f"分诊表登记 {key} 已不再是硬违例, 请删除登记"


def test_payload_violation_mark_labels_triage():
    """分诊标注: 未登记 → 待分诊; 已登记 → 对应标签 + 理由."""
    assert ca._payload_triage("missing-query", "GET", "/nope/none") is None
    assert "待分诊" in ca._payload_violation_mark("missing-query", "GET", "/nope/none")
    for (kind, method, path), (label, _reason) in ca._PAYLOAD_CONFIRMED_VIOLATIONS.items():
        mark = ca._payload_violation_mark(kind, method, path)
        assert "待分诊" not in mark
        assert ca._PAYLOAD_TRIAGE_DOC[label] in mark


def test_payload_render_sections_present():
    md = ca.render_payload_markdown(ca.build_payload_contract())
    assert "请求负载面" in md
    assert "违例类型" in md
    assert "静态核对覆盖面" in md
    assert "诚实边界" in md


def test_payload_synthetic_missing_query_and_body_field(tmp_path):
    """合成树: 必填 query 未传 / 模型必填字段未含 → 硬违例, 未登记即待分诊."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel\n"
        'router = APIRouter(prefix="/thing")\n'
        "class SaveBody(BaseModel):\n"
        "    name: str\n"
        '    note: str = ""\n'
        '@router.get("/list")\n'
        "async def thing_list(tag: str):\n"
        "    return {}\n"
        '@router.post("/save")\n'
        "async def thing_save(body: SaveBody):\n"
        "    return {}\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "await api.get('/thing/list');\n"
        "await api.get('/thing/list', { params: new URLSearchParams({ tag: 'x' }) });\n"
        "await api.post('/thing/save', { name: 'a' });\n"
        "await api.post('/thing/save', { note: 'b' });\n",
    )
    c = ca.build_payload_contract(tmp_path, tmp_path / "fe")
    got = {(v["kind"], v["detail"]) for v in c["violations"]}
    assert ("missing-query", "后端必填 query 未传: tag") in got
    assert any(k == "missing-body-field" and "name" in d for k, d in got)
    # 必填齐全的那两条调用不得误报.
    assert len(c["violations"]) == 2
    assert len(c["untriaged"]) == 2
    # URLSearchParams 字面量必须被解析出来 (否则 query 维度空转).
    assert c["coverage"]["query_checked"] == 2
    assert c["coverage"]["body_checked"] == 2


def test_payload_synthetic_shape_and_form(tmp_path):
    """合成树: JSON↔multipart 形状不符 / multipart 必填 Form 字段未含."""
    _write(
        tmp_path,
        "huginn/routes/up.py",
        "from fastapi import APIRouter, File, Form, UploadFile\n"
        "from pydantic import BaseModel\n"
        'router = APIRouter(prefix="/up")\n'
        "class Body(BaseModel):\n"
        "    name: str\n"
        '@router.post("/json")\n'
        "async def up_json(body: Body):\n"
        "    return {}\n"
        '@router.post("/form")\n'
        "async def up_form(credential_id: str = Form(...), file: UploadFile = File(...)):\n"
        "    return {}\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.up import router as up_router\n"
        "ALL_ROUTERS = [up_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "await api.post('/up/form', { credential_id: 'x' });\n"
        "await api.uploadWithProgress('/up/form', file, cb, {});\n"
        "await api.uploadStream('/up/json', file);\n",
    )
    c = ca.build_payload_contract(tmp_path, tmp_path / "fe")
    kinds = sorted(v["kind"] for v in c["violations"])
    assert kinds == ["missing-form-field", "shape-mismatch", "shape-mismatch"]
    assert len(c["untriaged"]) == 3


# ──────────────────── 响应结构面 ────────────────────


def test_response_real_repo_no_untriaged_violations():
    """真实仓: 前端 `api.*<T>` 声明要读的响应字段都命中端点后端 return 里有 (无待分诊)."""
    c = ca.build_response_contract()
    assert c["call_count"] > 0
    assert c["wired_call_count"] > 0
    assert c["type_count"] > 0
    assert c["untriaged"] == [], c["untriaged"]
    for v in c["violations"]:
        assert v["triage"] in {"defect", "intentional"}
        assert v["triage_reason"]
    # 至少有一批声明可静态核对, 否则本面等于空转.
    assert c["coverage"]["checked"] > 0


def test_response_confirmed_violations_registry_not_stale():
    """分诊表登记的每条都必须仍是真实硬违例 —— 修好后要同步删登记."""
    observed = {
        (v["kind"], v["method"], v["path"].split("?")[0])
        for v in ca.build_response_contract()["violations"]
    }
    for key in ca._RESP_CONFIRMED_VIOLATIONS:
        assert key in observed, f"分诊表登记 {key} 已不再是硬违例, 请删除登记"


def test_response_violation_mark_labels_triage():
    """分诊标注: 未登记 → 待分诊; 已登记 → 对应标签 + 理由."""
    assert ca._resp_triage("missing-field", "GET", "/nope/none") is None
    assert "待分诊" in ca._resp_violation_mark("missing-field", "GET", "/nope/none")
    for (kind, method, path), (label, _reason) in ca._RESP_CONFIRMED_VIOLATIONS.items():
        mark = ca._resp_violation_mark(kind, method, path)
        assert "待分诊" not in mark
        assert ca._RESP_TRIAGE_DOC[label] in mark


def test_response_type_keys_parses_members_and_generics():
    """TS 类型成员按 `;`/`,` 切; `Record<…>` 泛型里的逗号不得误切; `=>` 不得误当泛型."""
    assert ca._resp_type_keys("{ a?: string; b?: number }") == ({"a", "b"}, False)
    assert ca._resp_type_keys("{ x?: Record<string, number>; y?: string }") == (
        {"x", "y"},
        False,
    )
    assert ca._resp_type_keys("{ cb?: () => void; z: string }") == ({"cb", "z"}, False)
    assert ca._resp_type_keys("{ data: { a: number; b: number }; ok: boolean }") == (
        {"data", "ok"},
        False,
    )
    # 数组 / 标量 / 交叉 / index signature 一律开放 (读不出确切键).
    assert ca._resp_type_keys("Foo[]") == (set(), True)
    assert ca._resp_type_keys("any") == (set(), True)
    assert ca._resp_type_keys("{ a: string } & Bar") == (set(), True)
    assert ca._resp_type_keys("{ [k: string]: number }") == (set(), True)


def test_response_render_sections_present():
    md = ca.render_response_markdown(ca.build_response_contract())
    assert "响应结构面" in md
    assert "违例类型" in md
    assert "静态核对覆盖面" in md
    assert "诚实边界" in md


def test_response_synthetic_missing_field_detected(tmp_path):
    """合成树: 前端声明 `derived` 但后端 return 无此键 → 硬违例, 未登记即待分诊."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        '@router.post("/derive")\n'
        "async def thing_derive():\n"
        '    return {"success": True, "equations": {"a": "b"}}\n',
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(tmp_path, "fe/a.ts", "await api.post<{ derived?: string }>('/thing/derive', {});\n")
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    assert len(c["violations"]) == 1
    v = c["violations"][0]
    assert v["kind"] == "missing-field"
    assert v["endpoint"] == "/thing/derive"
    assert v["missing"] == ["derived"]
    assert sorted(v["produced"]) == ["equations", "success"]
    assert v["triage"] == "untriaged"
    assert len(c["untriaged"]) == 1


def test_response_synthetic_named_type_and_open_shape(tmp_path):
    """合成树: 具名 interface 解析出字段 (命中即无违例); 后端 return 变量 → 开放跳过."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        '@router.get("/ok")\n'
        "async def thing_ok():\n"
        '    return {"a": 1, "b": "x"}\n'
        '@router.get("/open")\n'
        "async def thing_open():\n"
        "    return _payload()\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "interface Thing { a?: number; b?: string }\n"
        "const one = await api.get<Thing>('/thing/ok');\n"
        "await api.get<{ c?: number }>('/thing/open');\n",
    )
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["type_count"] == 1
    assert c["coverage"]["checked"] == 1
    assert c["coverage"]["skip_shape"] == 1


def test_response_synthetic_open_declaration_and_ambiguity(tmp_path):
    """合成树: `<any>` 声明跳过; 动态段并列命中多端点 → 歧义跳过 (不猜端点)."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        '@router.get("/any")\n'
        "async def thing_any():\n"
        '    return {"a": 1}\n'
        '@router.get("/{x}/one")\n'
        "async def thing_one():\n"
        '    return {"p": 1}\n'
        '@router.get("/{y}/one")\n'
        "async def thing_two():\n"
        '    return {"q": 2}\n',
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "await api.get<any>('/thing/any');\n"
        "await api.get<{ p?: number }>('/thing/z/one');\n",
    )
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    # `<any>` 声明开放 → 跳过 (不误报); 动态段并列 → 歧义跳过.
    assert c["violations"] == []
    assert c["coverage"]["skip_decl"] == 1
    assert c["coverage"]["skip_ambiguous"] == 1


def test_response_synthetic_same_module_helper_closes(tmp_path):
    """合成树: `return _shape()` 中 helper 是同模块函数 → 递归取形状, 可核对出缺字段."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        "def _shape():\n"
        '    return {"a": 1, "b": 2}\n'
        '@router.get("/derived")\n'
        "async def thing_derived():\n"
        "    return _shape()\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(tmp_path, "fe/a.ts", "await api.get<{ a?: number; c?: number }>('/thing/derived');\n")
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    assert c["coverage"]["skip_shape"] == 0
    assert len(c["violations"]) == 1
    v = c["violations"][0]
    assert v["missing"] == ["c"]
    assert v["produced"] == ["a", "b"]


def test_response_synthetic_guarded_return_excludes_null(tmp_path):
    """合成树: `if err: return err` 排除 helper 的 null 分支 → 封闭; 无守卫版本仍开放."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        "def _check(x):\n"
        "    if x:\n"
        "        return None\n"
        '    return {"error": "e"}\n'
        '@router.get("/guard")\n'
        "async def thing_guard(x: int = 0):\n"
        "    err = _check(x)\n"
        "    if err:\n"
        "        return err\n"
        '    return {"ok": True}\n'
        '@router.get("/noguard")\n'
        "async def thing_noguard(x: int = 0):\n"
        "    err = _check(x)\n"
        "    return err\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "await api.get<{ ok?: boolean }>('/thing/guard');\n"
        "await api.get<{ error?: string }>('/thing/noguard');\n",
    )
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    # /guard 已核对 (ok/error 齐备); /noguard 的 `err` 可能为 null → 开放跳过.
    assert c["violations"] == []
    assert c["coverage"]["checked"] == 1
    assert c["coverage"]["skip_shape"] == 1


def test_response_synthetic_nested_def_return_ignored(tmp_path):
    """合成树: 嵌套 def 的 `return (1, 2)` 不属于端点响应, 不得使其开放."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        '@router.get("/nested")\n'
        "async def thing_nested():\n"
        "    def _inner():\n"
        "        return (1, 2)\n"
        "    _inner()\n"
        '    return {"a": 1}\n',
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(tmp_path, "fe/a.ts", "await api.get<{ a?: number }>('/thing/nested');\n")
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["coverage"]["skip_shape"] == 0
    assert c["coverage"]["checked"] == 1


def test_response_synthetic_helper_cycle_stays_open(tmp_path):
    """合成树: helper 互递归 (环) → 保守开放, 且不得死循环."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        "def _a():\n"
        "    return _b()\n"
        "def _b():\n"
        "    return _a()\n"
        '@router.get("/cyc")\n'
        "async def thing_cyc():\n"
        "    return _a()\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(tmp_path, "fe/a.ts", "await api.get<{ a?: number }>('/thing/cyc');\n")
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["coverage"]["skip_shape"] == 1


def test_response_synthetic_subscript_writes(tmp_path):
    """合成树: `result={}; result["k"]=…` (常量键) 记入键集; `.update()` 等读不出 ⇒ 开放."""
    _write(
        tmp_path,
        "huginn/routes/thing.py",
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        '@router.get("/sub")\n'
        "async def thing_sub():\n"
        "    result = {}\n"
        '    result["a"] = 1\n'
        "    return result\n"
        '@router.get("/upd")\n'
        "async def thing_upd():\n"
        "    result = {}\n"
        '    result.update({"b": 2})\n'
        "    return result\n",
    )
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(
        tmp_path,
        "fe/a.ts",
        "await api.get<{ a?: number }>('/thing/sub');\n"
        "await api.get<{ b?: number }>('/thing/upd');\n",
    )
    c = ca.build_response_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["coverage"]["checked"] == 1  # /sub 已核对
    assert c["coverage"]["skip_shape"] == 1  # /upd 读不出改写 ⇒ 开放


def test_response_real_repo_refinements_close_known_endpoints():
    """真实仓不变量: 三处精化后这些端点由开放转封闭 (防解析能力回退)."""
    shapes = ca._resp_backend_shapes(_REPO)
    assert shapes[("GET", "/health")]["closed"] is True
    assert {"model_pool", "mcp_servers"} <= shapes[("GET", "/health")]["keys"]
    assert shapes[("POST", "/transfer/web/upload")]["closed"] is True
    assert shapes[("GET", "/memory/layers")]["closed"] is True
    assert shapes[("DELETE", "/threads/{thread_id}")]["closed"] is True
    load = shapes[("POST", "/viewer3d/load")]
    assert load["closed"] is True
    assert "success" in load["keys"]


# ──────────────────── WS 请求负载面 ────────────────────


def test_ws_payload_real_repo_no_untriaged_violations():
    """真实仓: handler 读的字段都在 WSMessage 里, 前端发的字段也都认得 (无待分诊)."""
    c = ca.build_ws_payload_contract()
    assert c["handler_count"] > 0
    assert c["handlers_resolved"] == c["handler_count"]
    assert c["untriaged"] == [], c["untriaged"]
    for v in c["violations"]:
        assert v["triage"] in {"defect", "intentional"}
        assert v["triage_reason"]
    # 两个方向至少各自有可静态核对的样本, 否则本面等于空转.
    assert c["handlers_resolved"] > 0
    assert c["sends_static"] > 0


def test_ws_payload_confirmed_registry_not_stale():
    """分诊表登记的每条都必须仍是真实硬违例 —— 修好后要同步删登记."""
    observed = {
        (v["kind"], v["type"], v["field"])
        for v in ca.build_ws_payload_contract()["violations"]
    }
    for key in ca._WS_PAYLOAD_CONFIRMED:
        assert key in observed, f"分诊表登记 {key} 已不再是硬违例, 请删除登记"


def test_ws_payload_violation_mark_labels_triage():
    """分诊标注: 未登记 → 待分诊; 已登记 → 对应标签 + 理由."""
    assert ca._ws_payload_triage("handler-undeclared", "user_input", "nope") is None
    assert "待分诊" in ca._ws_payload_violation_mark(
        "handler-undeclared", "user_input", "nope"
    )
    for (kind, mtype, field), (label, _reason) in ca._WS_PAYLOAD_CONFIRMED.items():
        mark = ca._ws_payload_violation_mark(kind, mtype, field)
        assert "待分诊" not in mark
        assert ca._WS_PAYLOAD_TRIAGE_DOC[label] in mark


def test_ws_payload_render_sections_present():
    md = ca.render_ws_payload_markdown(ca.build_ws_payload_contract())
    assert "WS 请求负载面" in md
    assert "违例类型" in md
    assert "静态核对覆盖面" in md
    assert "诚实边界" in md


_WS_PAYLOAD_SCHEMA_SRC = (
    "from pydantic import BaseModel\n"
    "\n"
    "\n"
    "class WSMessage(BaseModel):\n"
    '    type: str = "user_input"\n'
    '    content: str = ""\n'
)


def _ws_payload_tree(tmp_path, handlers_src: str, fe_src: str) -> None:
    """合成 WS 负载树: schemas.py 声明 WSMessage / ws.py 注册表+handler / 前端发送."""
    _write(tmp_path, ca._WS_PAYLOAD_SCHEMA_REL, _WS_PAYLOAD_SCHEMA_SRC)
    _write(tmp_path, ca._WS_REGISTRY_REL, handlers_src)
    _write(tmp_path, "fe/chat.ts", fe_src)


def test_ws_payload_synthetic_handler_undeclared(tmp_path):
    """合成树: handler 读 `msg.secret` 而 WSMessage 未声明 → 硬违例 (AttributeError)."""
    _ws_payload_tree(
        tmp_path,
        "_MESSAGE_HANDLERS: dict = {\n"
        '    "user_input": _handle_user_input,\n'
        "}\n"
        "async def _handle_user_input(websocket, msg, ctx):\n"
        "    return await do(msg.secret)\n",
        'const url = "/ws/agent";\n',
    )
    c = ca.build_ws_payload_contract(tmp_path, tmp_path / "fe")
    got = [(v["kind"], v["type"], v["field"]) for v in c["violations"]]
    assert got == [("handler-undeclared", "user_input", "secret")]
    assert len(c["untriaged"]) == 1
    assert c["handlers_resolved"] == 1


def test_ws_payload_synthetic_fe_undeclared(tmp_path):
    """合成树: 前端发 `user_input` 带了 WSMessage 未声明的 `bogus` → 硬违例 (静默丢弃)."""
    _ws_payload_tree(
        tmp_path,
        "_MESSAGE_HANDLERS: dict = {\n"
        '    "user_input": _handle_user_input,\n'
        "}\n"
        "async def _handle_user_input(websocket, msg, ctx):\n"
        "    return await do(msg.content)\n",
        'const url = "/ws/agent";\n'
        'ws.send(JSON.stringify({ type: "user_input", content: "hi", bogus: 1 }));\n',
    )
    c = ca.build_ws_payload_contract(tmp_path, tmp_path / "fe")
    got = [(v["kind"], v["type"], v["field"]) for v in c["violations"]]
    assert got == [("fe-undeclared", "user_input", "bogus")]
    assert c["agent_send_count"] == 1
    assert c["sends_static"] == 1
    assert c["sends_unknown"] == 0


def test_ws_payload_synthetic_dead_field_and_unattributed(tmp_path):
    """合成树: 声明却无人接的字段入 dead_fields; 前端发未知 type 记 unattributed (跳过)."""
    _write(
        tmp_path,
        ca._WS_PAYLOAD_SCHEMA_REL,
        "from pydantic import BaseModel\n"
        "\n"
        "\n"
        "class WSMessage(BaseModel):\n"
        '    type: str = "user_input"\n'
        '    content: str = ""\n'
        "    orphan: str | None = None\n",
    )
    _write(
        tmp_path,
        ca._WS_REGISTRY_REL,
        "_MESSAGE_HANDLERS: dict = {\n"
        '    "user_input": _handle_user_input,\n'
        "}\n"
        "async def _handle_user_input(websocket, msg, ctx):\n"
        "    return await do(msg.content)\n",
    )
    _write(
        tmp_path,
        "fe/chat.ts",
        'const url = "/ws/agent";\n'
        'ws.send(JSON.stringify({ type: "user_input", content: "hi" }));\n'
        'ws.send(JSON.stringify({ type: "ghost", orphan: 1 }));\n',
    )
    c = ca.build_ws_payload_contract(tmp_path, tmp_path / "fe")
    # orphan 零 handler 读取且其唯一前端出现处 type 不在分发面 → 仍记"无人接".
    assert c["dead_fields"] == ["orphan"]
    assert c["agent_send_count"] == 2
    assert c["sends_unattributed"] == 1
    assert c["sends_static"] == 1
    assert c["violations"] == []


def test_ws_payload_synthetic_spread_send_is_unknown(tmp_path):
    """合成树: 前端发送对象含 `...` 展开 → 该处记 unknown (键集为下界, 不据此判违例)."""
    _ws_payload_tree(
        tmp_path,
        "_MESSAGE_HANDLERS: dict = {\n"
        '    "user_input": _handle_user_input,\n'
        "}\n"
        "async def _handle_user_input(websocket, msg, ctx):\n"
        "    return await do(msg.content)\n",
        'const url = "/ws/agent";\n'
        'ws.send(JSON.stringify({ type: "user_input", ...extra, content: "x" }));\n',
    )
    c = ca.build_ws_payload_contract(tmp_path, tmp_path / "fe")
    assert c["sends_unknown"] == 1
    assert c["sends_static"] == 0
    # 展开只使键集成下界, 已读到的键都在声明面 → 不误报.
    assert c["violations"] == []


# ──────────────────── SSE 事件负载面 ────────────────────


def test_sse_payload_real_repo_no_untriaged_violations():
    """真实仓: 前端读的帧 payload 顶层字段都在后端发帧形状里 (无待分诊)."""
    c = ca.build_sse_payload_contract()
    assert c["read_count"] > 0
    assert c["coverage"]["checked"] > 0
    assert c["untriaged"] == [], c["untriaged"]
    for v in c["violations"]:
        assert v["triage"] in {"defect", "intentional"}
        assert v["triage_reason"]


def test_sse_payload_confirmed_registry_not_stale():
    """分诊表登记的每条都必须仍是真实硬违例 —— 修好后要同步删登记."""
    observed = {
        (v["channel"], v["frame"], v["field"])
        for v in ca.build_sse_payload_contract()["violations"]
    }
    for key in ca._SSE_PAYLOAD_CONFIRMED:
        assert key in observed, f"分诊表登记 {key} 已不再是硬违例, 请删除登记"


def test_sse_payload_violation_mark_labels_triage():
    """分诊标注: 未登记 → 待分诊; 已登记 → 对应标签 + 理由."""
    assert ca._sse_payload_triage("event_bus", "tool.call", "nope") is None
    assert "待分诊" in ca._sse_payload_violation_mark("event_bus", "tool.call", "nope")
    for (ch, frame, field), (label, _reason) in ca._SSE_PAYLOAD_CONFIRMED.items():
        mark = ca._sse_payload_violation_mark(ch, frame, field)
        assert "待分诊" not in mark
        assert ca._SSE_PAYLOAD_TRIAGE_DOC[label] in mark


def test_sse_payload_render_sections_present():
    md = ca.render_sse_payload_markdown(ca.build_sse_payload_contract())
    assert "SSE 事件负载面" in md
    assert "违例类型" in md
    assert "静态核对覆盖面" in md
    assert "诚实边界" in md


def _sse_payload_backend(tmp_path, payload_body: str) -> None:
    """合成后端: 总线 to_sse 信封 (`payload = {…}`) + 事件声明 + 一个发布点."""
    _sse_backend(tmp_path, "")
    _write(
        tmp_path,
        ca._SSE_PAYLOAD_EVENT_BUS_MODULE,
        "class AgentEvent:\n"
        "    def to_sse(self):\n"
        "        payload = {\n"
        + payload_body
        + "        }\n"
        "        return payload\n",
    )


def test_sse_payload_synthetic_read_undeclared(tmp_path):
    """合成树: 前端读 `t.bogus` 而 event_bus 信封无此顶层键 → 硬违例 (恒 undefined)."""
    _sse_payload_backend(
        tmp_path,
        '            "type": self.type,\n            "data": self.data,\n',
    )
    _write(
        tmp_path,
        "fe/app.ts",
        "const es = new EventSource(`${API_BASE}/events/stream`);\n"
        "const handle = (e: MessageEvent) => {\n"
        "  const t = JSON.parse(e.data);\n"
        "  if (t.bogus) { }\n"
        "};\n"
        'es.addEventListener("tool.call", handle);\n',
    )
    c = ca.build_sse_payload_contract(tmp_path, tmp_path / "fe")
    assert c["channels"]["event_bus"]["tool.call"] == {
        "keys": ["data", "type"],
        "closed": True,
    }
    assert c["frame_reads"]["event_bus"]["tool.call"] == ["bogus"]
    got = [(v["kind"], v["channel"], v["frame"], v["field"]) for v in c["violations"]]
    assert got == [("read-undeclared", "event_bus", "tool.call", "bogus")]
    assert len(c["untriaged"]) == 1


def test_sse_payload_synthetic_declared_field_ok(tmp_path):
    """合成树: 前端读的顶层字段都在信封里 → 无违例, 且该帧记前端读取字段."""
    _sse_payload_backend(
        tmp_path,
        '            "type": self.type,\n            "data": self.data,\n',
    )
    _write(
        tmp_path,
        "fe/app.ts",
        "const es = new EventSource(`${API_BASE}/events/stream`);\n"
        "const handle = (e: MessageEvent) => {\n"
        "  const t = JSON.parse(e.data);\n"
        "  if (t.type === \"tool.call\") { use(t.data); }\n"
        "};\n"
        'es.addEventListener("tool.call", handle);\n',
    )
    c = ca.build_sse_payload_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["frame_reads"]["event_bus"]["tool.call"] == ["data", "type"]


def test_sse_payload_synthetic_shape_open_skipped(tmp_path):
    """合成树: 信封经变量间接构造 (无字面 `payload = {…}`) → 形状开放, 跳过不猜."""
    _sse_backend(tmp_path, "")
    _write(
        tmp_path,
        ca._SSE_PAYLOAD_EVENT_BUS_MODULE,
        "class AgentEvent:\n"
        "    def to_sse(self):\n"
        "        payload = build_envelope(self)\n"
        "        return payload\n",
    )
    _write(
        tmp_path,
        "fe/app.ts",
        "const es = new EventSource(`${API_BASE}/events/stream`);\n"
        "const handle = (e: MessageEvent) => {\n"
        "  const t = JSON.parse(e.data);\n"
        "  if (t.mystery) { }\n"
        "};\n"
        'es.addEventListener("tool.call", handle);\n',
    )
    c = ca.build_sse_payload_contract(tmp_path, tmp_path / "fe")
    assert c["channels"]["event_bus"]["tool.call"]["closed"] is False
    assert c["violations"] == []
    assert c["coverage"]["skip_shape"] == 1
    assert c["coverage"]["checked"] == 0


def test_sse_payload_synthetic_frame_not_in_channel_skipped(tmp_path):
    """合成树: 帧名不属该通道 (幽灵帧) → payload 面无权威, 跳过错开 (消费面另报)."""
    _sse_payload_backend(
        tmp_path,
        '            "type": self.type,\n            "data": self.data,\n',
    )
    _write(
        tmp_path,
        "fe/app.ts",
        "const es = new EventSource(`${API_BASE}/events/stream`);\n"
        "const handle = (e: MessageEvent) => {\n"
        "  const t = JSON.parse(e.data);\n"
        "  if (t.ghost) { }\n"
        "};\n"
        'es.addEventListener("ghost.void", handle);\n',
    )
    c = ca.build_sse_payload_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["coverage"]["skip_frame"] == 1
    assert c["coverage"]["checked"] == 0


# ──────────────────── WS 事件负载面 ────────────────────


def test_ws_ev_payload_real_repo_no_untriaged_violations():
    """真实仓: 前端各 type 分支读的顶层字段后端该帧都发 (无待分诊)."""
    c = ca.build_ws_ev_payload_contract()
    assert c["read_count"] > 0
    assert c["coverage"]["checked"] > 0
    assert c["untriaged"] == [], c["untriaged"]
    for v in c["violations"]:
        assert v["triage"] in {"defect", "intentional"}
        assert v["triage_reason"]


def test_ws_ev_payload_confirmed_registry_not_stale():
    """分诊表登记的每条都必须仍是真实硬违例 —— 修好后要同步删登记."""
    observed = {
        (v["channel"], v["frame"], v["field"])
        for v in ca.build_ws_ev_payload_contract()["violations"]
    }
    for key in ca._WS_EV_PAYLOAD_CONFIRMED:
        assert key in observed, f"分诊表登记 {key} 已不再是硬违例, 请删除登记"


def test_ws_ev_payload_violation_mark_labels_triage():
    """分诊标注: 未登记 → 待分诊; 已登记 → 对应标签 + 理由."""
    assert ca._ws_ev_payload_triage("agent", "text_delta", "nope") is None
    assert "待分诊" in ca._ws_ev_payload_violation_mark("agent", "text_delta", "nope")
    for (ch, frame, field), (label, _reason) in ca._WS_EV_PAYLOAD_CONFIRMED.items():
        mark = ca._ws_ev_payload_violation_mark(ch, frame, field)
        assert "待分诊" not in mark
        assert ca._WS_EV_PAYLOAD_TRIAGE_DOC[label] in mark


def test_ws_ev_payload_render_sections_present():
    md = ca.render_ws_ev_payload_markdown(ca.build_ws_ev_payload_contract())
    assert "WS 事件负载面" in md
    assert "违例类型" in md
    assert "静态核对覆盖面" in md
    assert "诚实边界" in md


def _ws_ev_payload_backend(tmp_path, ws_src: str) -> None:
    """合成后端: agent 通道 WS 路由模块 (server→client 帧字面量来源)."""
    _write(tmp_path, ca._WS_BACKEND_MODULES["agent"][0], ws_src)


def _ws_ev_frontend(reads: str, frame: str = "text_delta") -> str:
    """合成前端: WSMessage 标注的变量 + `switch (data.type)` 分支内读取."""
    return (
        'const url = "/ws/agent";\n'
        "const onMsg = (data: WSMessage) => {\n"
        "  switch (data.type) {\n"
        f'    case "{frame}":\n'
        f"      {reads}\n"
        "      break;\n"
        "  }\n"
        "};\n"
    )


def test_ws_ev_payload_synthetic_read_undeclared(tmp_path):
    """合成树: 前端读 `data.bogus` 而后端该帧不发此顶层键 → 硬违例 (恒 undefined)."""
    _ws_ev_payload_backend(
        tmp_path,
        "async def _send(ws):\n"
        '    await _ws_send({"type": "text_delta", "text": "hi"})\n',
    )
    _write(tmp_path, "fe/chat.ts", _ws_ev_frontend("use(data.bogus);"))
    c = ca.build_ws_ev_payload_contract(tmp_path, tmp_path / "fe")
    assert c["channels"]["agent"]["text_delta"] == {"keys": ["text"], "closed": True}
    assert c["frame_reads"]["agent"]["text_delta"] == ["bogus"]
    got = [(v["kind"], v["channel"], v["frame"], v["field"]) for v in c["violations"]]
    assert got == [("read-undeclared", "agent", "text_delta", "bogus")]
    assert len(c["untriaged"]) == 1
    # 后端发了 `text` 前端不读 → 只列候选, 不判违例.
    assert c["zero_read"] == [{"channel": "agent", "field": "text"}]


def test_ws_ev_payload_synthetic_declared_field_ok(tmp_path):
    """合成树: 前端读的顶层字段都在该帧 payload 里 → 无违例."""
    _ws_ev_payload_backend(
        tmp_path,
        "async def _send(ws):\n"
        '    await _ws_send({"type": "text_delta", "text": "hi"})\n',
    )
    _write(tmp_path, "fe/chat.ts", _ws_ev_frontend("use(data.text);"))
    c = ca.build_ws_ev_payload_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["frame_reads"]["agent"]["text_delta"] == ["text"]
    assert c["zero_read"] == []


def test_ws_ev_payload_synthetic_envelope_keys_not_read(tmp_path):
    """信封键 (`type` 判别键 / `thread_id` 注入) 不属各帧 payload, 不计入读取."""
    _ws_ev_payload_backend(
        tmp_path,
        "async def _send(ws):\n"
        '    await _ws_send({"type": "text_delta", "text": "hi"})\n',
    )
    _write(
        tmp_path,
        "fe/chat.ts",
        _ws_ev_frontend("use(data.type, data.thread_id, data.text);"),
    )
    c = ca.build_ws_ev_payload_contract(tmp_path, tmp_path / "fe")
    assert c["frame_reads"]["agent"]["text_delta"] == ["text"]
    assert c["violations"] == []


def test_ws_ev_payload_synthetic_shape_open_skipped(tmp_path):
    """合成树: 帧字面量含 `**` 展开 → 键集不可穷尽, 记开放并跳过不猜."""
    _ws_ev_payload_backend(
        tmp_path,
        "async def _send(ws, extra):\n"
        '    await _ws_send({"type": "text_delta", **extra})\n',
    )
    _write(tmp_path, "fe/chat.ts", _ws_ev_frontend("use(data.mystery);"))
    c = ca.build_ws_ev_payload_contract(tmp_path, tmp_path / "fe")
    assert c["channels"]["agent"]["text_delta"]["closed"] is False
    assert c["violations"] == []
    assert c["coverage"]["skip_shape"] == 1
    assert c["coverage"]["checked"] == 0


def test_ws_ev_payload_synthetic_frame_not_in_channel_skipped(tmp_path):
    """合成树: 帧名不属该通道 (幽灵帧) → payload 面无权威, 跳过错开 (消费面另报)."""
    _ws_ev_payload_backend(
        tmp_path,
        "async def _send(ws):\n"
        '    await _ws_send({"type": "text_delta", "text": "hi"})\n',
    )
    _write(tmp_path, "fe/chat.ts", _ws_ev_frontend("use(data.ghostfield);", frame="ghost"))
    c = ca.build_ws_ev_payload_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["coverage"]["skip_frame"] == 1
    assert c["coverage"]["checked"] == 0


# ──────────────────── HTTP 请求字段面 ────────────────────


def test_http_field_real_repo_no_untriaged_violations():
    """真实仓: 请求体字段的声明/读取/发送三面一致 (无待分诊违例)."""
    c = ca.build_http_field_contract()
    assert c["endpoint_count"] > 0
    assert c["untriaged"] == [], c["untriaged"]
    for v in c["violations"]:
        assert v["triage"] in {"defect", "intentional"}
        assert v["triage_reason"]
    # 至少有一批字段可静态核对, 否则本面等于空转.
    assert c["coverage"]["handler_checked"] > 0
    assert c["coverage"]["dict_checked"] > 0


def test_http_field_confirmed_registry_not_stale():
    """分诊表登记的每条都必须仍是真实硬违例 —— 修好后要同步删登记."""
    observed = {
        (v["kind"], v["method"], v["endpoint"], v["field"])
        for v in ca.build_http_field_contract()["violations"]
    }
    for key in ca._HTTP_FIELD_CONFIRMED:
        assert key in observed, f"分诊表登记 {key} 已不再是硬违例, 请删除登记"


def test_http_field_violation_mark_labels_triage():
    """分诊标注: 未登记 → 待分诊; 已登记 → 对应标签 + 理由."""
    assert ca._http_field_triage("handler-undeclared", "POST", "/nope/none", "f") is None
    assert "待分诊" in ca._http_field_violation_mark(
        "handler-undeclared", "POST", "/nope/none", "f"
    )
    for (kind, method, endpoint, field), (label, _reason) in ca._HTTP_FIELD_CONFIRMED.items():
        mark = ca._http_field_violation_mark(kind, method, endpoint, field)
        assert "待分诊" not in mark
        assert ca._HTTP_FIELD_TRIAGE_DOC[label] in mark


def test_http_field_render_sections_present():
    md = ca.render_http_field_markdown(ca.build_http_field_contract())
    assert "HTTP 请求字段面" in md
    assert "违例类型" in md
    assert "静态核对覆盖面" in md
    assert "诚实边界" in md


def _http_field_tree(tmp_path, backend: str, fe: str) -> None:
    """合成树: routes/thing.py 声明模型+端点 / routes/__init__.py 挂载 / 前端调用."""
    _write(tmp_path, "huginn/routes/thing.py", backend)
    _write(
        tmp_path,
        "huginn/routes/__init__.py",
        "from huginn.routes.thing import router as thing_router\n"
        "ALL_ROUTERS = [thing_router]\n",
    )
    _write(tmp_path, "fe/a.ts", fe)


def test_http_field_synthetic_handler_undeclared(tmp_path):
    """合成树: handler 读 `body.secret` 而请求体模型未声明 → 硬违例 (AttributeError)."""
    _http_field_tree(
        tmp_path,
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel\n"
        'router = APIRouter(prefix="/thing")\n'
        "class SaveBody(BaseModel):\n"
        "    name: str\n"
        '    note: str = ""\n'
        '@router.post("/save")\n'
        "async def thing_save(body: SaveBody):\n"
        "    return {'n': body.name, 's': body.secret}\n",
        "await api.post('/thing/save', { name: 'a' });\n",
    )
    c = ca.build_http_field_contract(tmp_path, tmp_path / "fe")
    got = [(v["kind"], v["method"], v["endpoint"], v["field"]) for v in c["violations"]]
    assert got == [("handler-undeclared", "POST", "/thing/save", "secret")]
    assert len(c["untriaged"]) == 1
    assert c["coverage"]["handler_checked"] == 1


def test_http_field_synthetic_fe_undeclared(tmp_path):
    """合成树: 前端发模型未声明的 `bogus` 键 → 硬违例 (Pydantic 静默丢弃)."""
    _http_field_tree(
        tmp_path,
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel\n"
        'router = APIRouter(prefix="/thing")\n'
        "class SaveBody(BaseModel):\n"
        "    name: str\n"
        '    note: str = ""\n'
        '@router.post("/save")\n'
        "async def thing_save(body: SaveBody):\n"
        "    return {'n': body.name}\n",
        "await api.post('/thing/save', { name: 'a', bogus: 1 });\n",
    )
    c = ca.build_http_field_contract(tmp_path, tmp_path / "fe")
    got = [(v["kind"], v["method"], v["endpoint"], v["field"]) for v in c["violations"]]
    assert got == [("fe-undeclared", "POST", "/thing/save", "bogus")]
    assert c["coverage"]["fe_checked"] == 1


def test_http_field_synthetic_dict_key_unsent(tmp_path):
    """合成树: body-dict 端点 handler 下标读键而前端调用从不发 → 硬违例 (KeyError)."""
    _http_field_tree(
        tmp_path,
        "from fastapi import APIRouter\n"
        'router = APIRouter(prefix="/thing")\n'
        '@router.post("/raw")\n'
        "async def thing_raw(body: dict):\n"
        "    return {'x': body['must']}\n",
        "await api.post('/thing/raw', { other: 1 });\n",
    )
    c = ca.build_http_field_contract(tmp_path, tmp_path / "fe")
    got = [(v["kind"], v["method"], v["endpoint"], v["field"]) for v in c["violations"]]
    assert got == [("dict-key-unsent", "POST", "/thing/raw", "must")]
    assert c["coverage"]["dict_checked"] == 1


def test_http_field_synthetic_extra_allow_skips_fe(tmp_path):
    """合成树: 模型 `extra=allow` → 前端发送面开放 (不判 fe-undeclared), handler 读取照核."""
    _http_field_tree(
        tmp_path,
        "from fastapi import APIRouter\n"
        "from pydantic import BaseModel, ConfigDict\n"
        'router = APIRouter(prefix="/thing")\n'
        "class SaveBody(BaseModel):\n"
        "    model_config = ConfigDict(extra='allow')\n"
        "    name: str\n"
        '@router.post("/save")\n'
        "async def thing_save(body: SaveBody):\n"
        "    return {'n': body.name}\n",
        "await api.post('/thing/save', { name: 'a', bogus: 1 });\n",
    )
    c = ca.build_http_field_contract(tmp_path, tmp_path / "fe")
    assert c["violations"] == []
    assert c["rows"][0]["shape"] == "开放(extra=allow)"
    assert c["coverage"]["fe_skipped"] == 1


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
