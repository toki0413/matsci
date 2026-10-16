"""低熵治理台账 (governance_audit) 的门禁自检.

治理层自身的熵: **门存在却未登记 / CI step 指向不存在的测试 / CLI 门指向已删模块 /
本地门 (.pre-commit) 指向已删测试**. 本测试把 `tests/test_arch_*.py` 门族、`ci.yml`
的 pytest guard step、`python -m huginn.<module>` CLI 门与 `.pre-commit-config.yaml`
本地门四者对齐成硬不变量:

- 每个 `test_arch_*.py` 必须登记 (显式快速失败门, 或显式声明 full-suite-only);
  未登记者 = 孤儿门 → 红.
- 每个 CI pytest guard step 指向的测试文件必须存在; 指向幽灵文件的悬空门 → 红.
- 每个 CLI 门 (`python -m huginn.<module>`) 指向的模块必须存在; 指向已删模块 → 红.
- 每个本地门 (`.pre-commit-config.yaml`) 引用的测试文件必须存在; 指向已删文件 → 红.
- 每个本地门 `entry` 引用的测试必须被其 `files:` 过滤器覆盖; 否则改它不触发 hook → 红.
- 门数只减不增 (棘轮); 删掉一个已登记门 → 红.
- 每个循环闸位 (自主轮数/步数上限) 实测 ≤ 台账上限; 提高上限须显式改台账 → 红.
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

# CLI 门 (`python -m huginn.<module>`) 数下限棘轮: 同上, 值由台账权威声明.
CLI_GUARD_FLOOR = ga.budget_value("cli_guard_floor")

# 本地门 (.pre-commit-config.yaml) 数下限棘轮: 同上, 值由台账权威声明.
PRECOMMIT_GATE_FLOOR = ga.budget_value("precommit_gate_floor")

# 循环闸位数下限棘轮: 同上, 值由台账权威声明.
LOOP_CAP_FLOOR = ga.budget_value("loop_cap_floor")


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


# ──────────────────── CLI 门 (非 pytest 形态) ────────────────────


def test_cli_guards_point_at_existing_modules():
    """不变量: 每个 CLI 门 (`python -m huginn.<module>`) 指向的模块都存在."""
    reg = ga.build_gate_registry()
    assert reg["dangling_cli"] == [], (
        f"CLI 悬空门 (python -m 指向不存在的模块): {reg['dangling_cli']} —— "
        f"模块改名/删除后忘了同步 ci.yml."
    )


def test_cli_guard_entries_are_unique():
    """每个 CLI 门模块只登记一次 (去重; 一处多处调用不算多门)."""
    reg = ga.build_gate_registry()
    mods = [c["module"] for c in reg["cli_guards"]]
    assert len(mods) == len(set(mods)), "CLI 门登记表存在重复模块条目"


def test_cli_guard_count_ratchet_only_shrinks():
    """棘轮: CLI 门数只减不增."""
    reg = ga.build_gate_registry()
    assert len(reg["cli_guards"]) >= CLI_GUARD_FLOOR, (
        f"CLI 门数从 {CLI_GUARD_FLOOR} 降到 {len(reg['cli_guards'])} —— 门被删了? "
        f"若确为有意合并, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"cli_guard_floor 的 value (可见的发布决策)."
    )


def test_cli_guard_dangling_detection_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证 CLI 门指向幽灵模块能被抓到."""
    (tmp_path / "tests").mkdir()
    ci = "steps:\n  - name: Ghost CLI\n    run: python -m huginn.cli.ghost --check\n"
    reg = ga.audit(tmp_path, ci_text=ci)
    assert any(d["module"] == "huginn.cli.ghost" for d in reg["dangling_cli"])


def test_cli_guard_ignores_comment_lines():
    """注释行里的 `python -m huginn.<module>` (如重生成提示) 不算门.

    这是 yaml 主路径与逐行 fallback **共用判据**的确定性锚点: 否则注释里的
    命令会被误登记成幻影门, 且结果随运行环境是否装了 PyYAML 而变. 用合法
    `jobs` 结构, 装了 PyYAML 时走主路径、没装时走 fallback, 两条结果须一致.
    """
    ci = (
        "jobs:\n"
        "  g:\n"
        "    steps:\n"
        "      - name: Real CLI gate\n"
        "        run: |\n"
        "          # 重生成: python -m huginn.cli.fake --update\n"
        "          python -m huginn.cli.real --check\n"
    )
    mods = [m for _, m in ga._parse_cli_guards(ci)]
    assert "huginn.cli.real" in mods
    assert "huginn.cli.fake" not in mods


# ──────────────────── 本地门 (.pre-commit-config.yaml) ────────────────────


def test_precommit_gates_reference_existing_tests():
    """不变量: 每个本地门 (.pre-commit) 引用的测试文件必须存在.

    本地门引用的测试改名/删除后, hook 的 `files:` 过滤器会静默不再匹配 —— 门悄悄死掉,
    无任何信号 (与悬空 CI 门同源).
    """
    reg = ga.build_gate_registry()
    assert reg["dangling_precommit"] == [], (
        f"悬空本地门 (pre-commit hook 引用不存在的测试): {reg['dangling_precommit']} —— "
        f"测试改名/删除后忘了同步 .pre-commit-config.yaml 的 entry 与 files:."
    )


def test_precommit_gate_count_ratchet_only_shrinks():
    """棘轮: 本地门数只减不增."""
    reg = ga.build_gate_registry()
    assert len(reg["precommit_gates"]) >= PRECOMMIT_GATE_FLOOR, (
        f"本地门数从 {PRECOMMIT_GATE_FLOOR} 降到 {len(reg['precommit_gates'])} —— 门被删了? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"precommit_gate_floor 的 value (可见的发布决策)."
    )


def test_precommit_dangling_detection_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证本地门指向幽灵测试能被抓到 (前向非空转)."""
    (tmp_path / "tests").mkdir()
    pc = (
        "repos:\n  - repo: local\n    hooks:\n"
        "      - id: demo\n        entry: bash -c 'pytest tests/test_ghost.py'\n"
    )
    reg = ga.audit(tmp_path, ci_text="", precommit_text=pc)
    assert any(
        d["hook"] == "demo" and d["test"] == "tests/test_ghost.py"
        for d in reg["dangling_precommit"]
    )


def test_precommit_entry_tests_covered_by_files_filter():
    """不变量: 每个本地门 `entry` 引用的测试都被其 `files:` 过滤器覆盖.

    否则改这些测试不会触发 hook —— 门静默失效 (比悬空 CI 门更隐蔽).
    """
    reg = ga.build_gate_registry()
    assert reg["uncovered_precommit"] == [], (
        f"本地门 files:/entry 不一致: {reg['uncovered_precommit']} —— entry 引用的测试"
        f"不在 files: 过滤范围内, 改它不会触发 hook. 同步 .pre-commit-config.yaml 的 files:."
    )


def test_precommit_files_mismatch_detection_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证 files:/entry 不一致能被抓到 (前向非空转)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    pc = (
        "repos:\n  - repo: local\n    hooks:\n"
        "      - id: demo\n"
        "        entry: bash -c 'pytest tests/test_a.py'\n"
        "        files: ^agent/tests/test_b\\.py$\n"
    )
    reg = ga.audit(tmp_path, ci_text="", precommit_text=pc)
    assert any(
        d["hook"] == "demo" and d["test"] == "tests/test_a.py"
        for d in reg["uncovered_precommit"]
    )


def test_precommit_files_covers_entry_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证: entry 测试被 files: 覆盖时不误报 (无假阳)."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    pc = (
        "repos:\n  - repo: local\n    hooks:\n"
        "      - id: demo\n"
        "        entry: bash -c 'pytest tests/test_a.py'\n"
        "        files: ^agent/tests/test_a\\.py$\n"
    )
    reg = ga.audit(tmp_path, ci_text="", precommit_text=pc)
    assert reg["uncovered_precommit"] == []


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
        "cli_guard_floor",
        "precommit_gate_floor",
        "field_audit_golden",
        "loop_autoloop_cli",
        "loop_autoloop_engine",
        "loop_recursion_limit",
        "loop_tool_calls",
        "loop_tool_calls_per_tool",
        "loop_codeact_turns",
        "loop_coder_iter",
        "loop_exploration",
        "loop_deep_research",
        "loop_cap_floor",
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


def test_heavy_probes_disabled_by_default():
    """门禁路径不跑重型探针 (保持 weightless): 凡标 heavy 的项实测应为 None."""
    by = {b["id"]: b for b in ga.collect_budgets(_AGENT)}
    heavy = [s["id"] for s in ga.BUDGET_SPECS if s.get("heavy")]
    assert heavy, "应至少有一个重型探针项"
    for bid in heavy:
        assert by[bid]["current"] is None, (
            f"默认路径不应跑重型探针 `{bid}` (会拖慢快速失败门); 该探针只在 CLI --probe 下跑."
        )


def test_ci_step_owner_gates_reference_real_steps():
    """预算 `owner_kind` 口径统一: 标 `ci_step` 的消费者门必须是 ci.yml 真实 step 名.

    `owner_gate` 混用两种语义: `ci_step` (ci.yml 真实 step, 可校验存在性) 与 `full_suite`
    (仅全量套件跑, 不在 ci.yml). 若不区分, step 改名后台账会指向**幻影门**而无声. 本门固化:
    ①`owner_kind` 合法枚举; ②`ci_step` 型 `owner_gate` 命中 ci.yml 真实 step 名 (取
    `gates ∪ cli_guards` 的 step 名集).
    """
    reg = ga.build_gate_registry()
    real_steps = {g["ci_step"] for g in reg["gates"] if g["ci_step"]}
    real_steps |= {c["ci_step"] for c in reg["cli_guards"] if c["ci_step"]}
    for spec in ga.BUDGET_SPECS:
        kind = spec["owner_kind"]
        assert kind in {"ci_step", "full_suite"}, (
            f"预算 `{spec['id']}` 的 owner_kind `{kind}` 非法 (须 ci_step / full_suite)."
        )
        if kind == "ci_step":
            assert spec["owner_gate"] in real_steps, (
                f"预算 `{spec['id']}` 标 `ci_step`, 但 owner_gate `{spec['owner_gate']}` 不是 "
                f"ci.yml 真实 step 名 —— step 改名后台账指向幻影门? 同步 BUDGET_SPECS / ci.yml."
            )


# ──────────────────── 循环闸位 (loop caps) ────────────────────


def test_loop_caps_live_within_ceiling():
    """不变量: 每个循环闸位仍能读到活值, 且活值 ≤ 台账上限 (只增不减的反向棘轮).

    循环闸位是自主轮数/步数上限; 提高上限 = 扩大自主面, 必须走台账 (可见决策).
    """
    caps = [s for s in ga.BUDGET_SPECS if s.get("source") == "ceiling"]
    assert len(caps) >= LOOP_CAP_FLOOR, (
        f"循环闸位数从 {LOOP_CAP_FLOOR} 降到 {len(caps)} —— 闸位被删了? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 loop_cap_floor 的 value."
    )
    for spec in caps:
        live = ga._probe_current(spec, _AGENT, heavy=False)
        assert live is not None, (
            f"循环闸位 `{spec['id']}` 活值读不到 —— pattern 或源文件位置失效, 闸位静默失效."
        )
        assert live <= spec["value"], (
            f"循环闸位 `{spec['id']}` 活值 {live} > 台账上限 {spec['value']} —— "
            f"扩大自主面须显式更新 BUDGET_SPECS 的 value (可见决策)."
        )
