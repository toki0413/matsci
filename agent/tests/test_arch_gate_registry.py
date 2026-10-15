"""低熵治理台账 (governance_audit) 的门禁自检.

治理层自身的熵: **门存在却未登记 / CI step 指向不存在的测试**. 本测试把
`tests/test_arch_*.py` 门族与 `ci.yml` 的 python guard step 对齐成硬不变量:

- 每个 `test_arch_*.py` 必须登记 (显式快速失败门, 或显式声明 full-suite-only);
  未登记者 = 孤儿门 → 红.
- 每个 CI guard step 指向的测试文件必须存在; 指向幽灵文件的悬空门 → 红.
- 门数只减不增 (棘轮); 删掉一个已登记门 → 红.
- 生成文档 `docs/governance-state.md` 必须与当前扫描逐字一致 (确定性再生成).

重生成: `python -m huginn.cli.governance_audit --out docs/governance-state.md`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

_AGENT = pathlib.Path(__file__).resolve().parents[1]

# weightless: 直接按文件路径加载治理台账模块, 不走 huginn.cli 包 __init__
# (后者拉 typer/rich/pydantic 等重依赖). 门禁本身要轻, 才能先于重型套件快速 fail.
_MOD = _AGENT / "huginn" / "cli" / "governance_audit.py"
_spec = importlib.util.spec_from_file_location("governance_audit", _MOD)
ga = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ga)

# 冻结门数棘轮: 值由治理台账 (governance_audit.BUDGET_SPECS) **权威声明**, 本门只读.
# 编辑入口: huginn/cli/governance_audit.py 的 BUDGET_SPECS["arch_gate_floor"]["value"].
ARCH_GATE_FLOOR = ga.budget_value("arch_gate_floor")


def test_all_arch_tests_are_registered():
    """不变量: 每个 `tests/test_arch_*.py` 都在登记表里, 无孤儿门."""
    reg = ga.build_gate_registry()
    assert reg["orphans"] == [], (
        f"未登记治理门: {reg['orphans']} —— 要么在 ci.yml 加显式 guard step, "
        f"要么显式登记进 governance_audit.FULL_SUITE_ONLY (可见决策)."
    )


def test_ci_guard_steps_point_at_existing_files():
    """不变量: 每个 CI python guard step 指向的测试文件必须存在."""
    reg = ga.build_gate_registry()
    assert reg["dangling"] == [], (
        f"悬空门 (CI step 指向不存在的测试): {reg['dangling']} —— "
        f"重命名/删除测试后忘了同步 ci.yml."
    )


def test_gate_entries_are_unique():
    """每个门只登记一次, 避免重复行掩盖真实缺口."""
    reg = ga.build_gate_registry()
    tests = [g["test"] for g in reg["gates"]]
    assert len(tests) == len(set(tests)), "登记表存在重复门条目"


def test_arch_gate_count_ratchet_only_shrinks():
    """棘轮: arch 门族条目数只减不增."""
    reg = ga.build_gate_registry()
    arch = [g for g in reg["gates"] if g["test"].startswith("tests/test_arch_")]
    assert len(arch) >= ARCH_GATE_FLOOR, (
        f"arch 门数从 {ARCH_GATE_FLOOR} 降到 {len(arch)} —— 门被删了? "
        f"若确为有意合并, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"arch_gate_floor 的 value (可见的发布决策)."
    )


def test_markdown_render_is_deterministic():
    """同一登记快照渲染两次逐字一致 (跨机确定性)."""
    reg = ga.build_gate_registry()
    assert ga.render_markdown(reg) == ga.render_markdown(reg)


def test_committed_state_doc_matches_registry():
    """AC-1: 生成文档与扫描结果逐字一致 (防漂移)."""
    reg = ga.build_gate_registry()
    want = ga.render_markdown(reg)
    doc = _AGENT / "docs" / "governance-state.md"
    assert doc.exists(), "缺 docs/governance-state.md, 跑 governance_audit --out 生成"
    got = doc.read_text(encoding="utf-8")
    assert got == want, (
        "docs/governance-state.md 与当前扫描漂移; "
        "跑 `python -m huginn.cli.governance_audit --out docs/governance-state.md`."
    )


# ──────────────────── 合成树: 防门禁自己失效 ────────────────────


def test_orphan_detection_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证孤儿门能被抓到 (否则门禁自身失效)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_arch_demo_gate.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    reg = ga.audit(tmp_path, ci_text="")
    assert "tests/test_arch_demo_gate.py" in reg["orphans"]


def test_dangling_guard_detection_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证 CI step 指向幽灵文件能被抓到."""
    (tmp_path / "tests").mkdir()
    ci = "steps:\n  - name: Ghost guard\n    run: python -m pytest tests/test_arch_ghost.py -q\n"
    reg = ga.audit(tmp_path, ci_text=ci)
    assert any(d["test"] == "tests/test_arch_ghost.py" for d in reg["dangling"])


def test_registered_guard_on_synthetic_tree_is_not_orphan(tmp_path: pathlib.Path):
    """合成树验证: 已接入 ci.yml 的门不算孤儿 (无假阳)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_arch_ok.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    ci = "steps:\n  - name: OK guard\n    run: python -m pytest tests/test_arch_ok.py -q\n"
    reg = ga.audit(tmp_path, ci_text=ci)
    assert reg["orphans"] == []
    assert reg["dangling"] == []


# ──────────────────── 预算台账 (BUDGET_SPECS) ────────────────────

# 冻结覆盖集: 应被登记的治理棘轮预算. 新增/删除棘轮必须同步此集 (可见决策).
_EXPECTED_BUDGET_IDS = frozenset(
    {
        "mece_findings",
        "mirror_drift",
        "env_raw_read",
        "gateway_imports",
        "gateway_subcommands",
        "sanctioned_plugins",
        "hardcoded_allow",
        "mcp_wiring_allow",
        "arch_gate_floor",
        "field_audit_golden",
    }
)


def test_budget_ids_unique_and_complete():
    """预算登记: id 唯一, 且与冻结覆盖集一致 (只许显式增删)."""
    ids = [s["id"] for s in ga.BUDGET_SPECS]
    assert len(ids) == len(set(ids)), f"BUDGET_SPECS 存在重复 id: {ids}"
    have = set(ids)
    assert have == _EXPECTED_BUDGET_IDS, (
        f"预算覆盖变化 —— 未登记: {sorted(_EXPECTED_BUDGET_IDS - have)}; "
        f"多出: {sorted(have - _EXPECTED_BUDGET_IDS)}. 新增/删除棘轮须同步此冻结集."
    )


def test_budget_frozen_values_extract_cleanly():
    """每个预算项的冻结值都能抽取 (无 '?'), 防位置/写法漂移让台账静默腐烂."""
    for b in ga.collect_budgets(_AGENT):
        assert "?" not in b["frozen"], (
            f"预算 `{b['id']}` 冻结值抽取失败 ({b['frozen']}) —— "
            f"源文件位置或写法变了? 同步 governance_audit.BUDGET_SPECS."
        )


def test_budget_probe_headroom_is_nonneg():
    """有实测探针的预算项余量非负 (实测未越过冻结界)."""
    for b in ga.collect_budgets(_AGENT):
        if b["headroom"] is not None:
            assert b["headroom"] >= 0, (
                f"预算 `{b['id']}` 实测 {b['current']} 越过冻结界 {b['frozen']} —— "
                f"棘轮破了? 检查对应消费者门."
            )


def test_authoritative_budgets_not_redeclared():
    """权威(inline)预算的值只在台账声明; 消费者门不得再硬编码同名字面量."""
    for spec in ga.BUDGET_SPECS:
        if spec.get("source") != "inline":
            continue
        gate = (_AGENT / spec["path"]).read_text(encoding="utf-8")
        keys = (spec["key"],) if "key" in spec else tuple(spec.get("keys", ()))
        for key in keys:
            assert not re.search(rf"^{re.escape(key)}\s*=\s*\d", gate, re.M), (
                f"`{spec['id']}` 是权威预算, 值应只在 BUDGET_SPECS 声明; 但 "
                f"`{spec['path']}` 仍硬编码 `{key} = <数字>`. "
                f"改为 ga.budget_value({spec['id']!r})."
            )
