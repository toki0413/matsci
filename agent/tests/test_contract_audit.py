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


def test_tool_allowlist_true_dead_items_detected():
    """`READ_ONLY_TOOLS` 用短名 (`read_file`/`list_dir`), 注册名却是 `file_read_tool`.

    死项 ⇒ `set_mode` 永不命中, sidecar 的 auto_approve 对该读工具失效 —— 真实缺陷.
    """
    by = {a["name"]: a for a in ca.build_tool_contract()["allowlists"]}
    ro = by["READ_ONLY_TOOLS"]
    assert ro["namespace"] == "registry"
    assert "read_file" in ro["phantoms"]
    assert "list_dir" in ro["phantoms"]


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


def test_tool_unregistered_class_reported():
    """声明了 `name` 却不在注册清单的类要被报出来 (宣称未注册)."""
    assert "PyBulletTool" in ca.build_tool_contract()["unregistered_classes"]


def test_tool_render_and_issues_contain_sections():
    md = ca.render_tool_markdown(ca.build_tool_contract())
    assert "工具面" in md
    assert "死项" in md
    assert "注册声明缺口" in md
    assert "允许表死项" in "\n".join(ca.find_issues(ca.build_mece_snapshot()))


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
