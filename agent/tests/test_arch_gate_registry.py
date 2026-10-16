"""低熵治理台账 (governance_audit) 的门禁自检.

治理层自身的熵: **门存在却未登记 / CI step 指向不存在的测试 / CLI 门指向已删模块 /
本地门 (.pre-commit) 指向已删测试 / 工程门 step 脱节 / 工作流整体消失**. 本测试把
`tests/test_arch_*.py` 门族 + 显式治理元门 (`GOVERNANCE_META_TESTS`)、**全部**工作流的
pytest guard step、`python -m huginn.<module>` CLI 门、`ENGINEER_GATE_SPECS` 工程门、
仓根与 agent/ 两份 `.pre-commit-config.yaml` 本地门对齐成硬不变量:

- 每个治理门 (arch 门族 + 治理元门) 必须登记 (显式快速失败门, 或显式声明 full-suite-only);
  未登记者 = 孤儿门 → 红.
- 每个 CI pytest guard step 指向的测试文件必须存在; 指向幽灵文件的悬空门 → 红.
- 每个 CLI 门 (`python -m huginn.<module>`) 指向的模块必须存在; 指向已删模块 → 红.
- 每个本地门 (两份 `.pre-commit-config.yaml`) 引用的测试文件必须存在; 指向已删文件 → 红.
- 每个本地门 `entry` 引用的测试必须被其 `files:` 过滤器覆盖; 否则改它不触发 hook → 红.
- 每个工程门 (`ENGINEER_GATE_SPECS`) 的 step 名仍在对应工作流; step 脱节 → 红.
- 每个工作流的**触发器** (`on:`) 与 `WORKFLOW_TRIGGER_SPECS` 期望集一致; 触发器被删/收窄
  (门永不运行) → 红; 未登记触发规格的工作流 / 规格未落地的工作流 → 红.
- 全部工作流**命名 step 总数** ≥ 台账下限 (阻断步静默增删的反向兜底) → 红.
- 每个命名子系统 (`CAPABILITY_SPECS`) 的锚点 (开关默认值未翻转 + 承载模块存在 + 不变量测试
  存在) 都不脱节; 能力锚点数只减不增 → 红 (能力面静默萎缩).
- 每个 feature flag (`FeatureFlags._DEFAULTS`) 都被「锚定 (CAPABILITY_SPECS) or 豁免
  (SUBSYSTEM_EXEMPT_FLAGS)」**恰好**分诊一次 (并集 = 全集且不相交); 新增开关未分诊 → 红
  (能力面闭合, 消除"新加开关又出现静默缺口").
- 每个 prompt 段 (`prompt_segments._APPLIES_TO`) 的**生效面**与声明一致 (「声明 == 渲染」
  由 `tests/test_prompt_contract.py` 硬判); 段清单 (`PROMPT_SPECS`) 与源对齐且只减不增 → 红
  (prompt 段面静默萎缩).
- 门数/工程门数/本地门数/CLI 门数/工作流数只减不增 (棘轮); 删掉已登记项 → 红.
- 每个循环闸位 (自主轮数/步数上限) 实测 ≤ 台账上限; 提高上限须显式改台账 → 红.
- 每个豁免/白名单 (gitleaks 密钥豁免 / pip-audit 漏洞豁免 / dependabot 忽略项) 条数 ≤ 台账
  上限; 覆盖率下限 `fail_under` ≥ 台账下限 —— 放宽豁免或下调覆盖率 = 静默削弱对应门 → 红.
- 生成文档 `docs/governance-state.md` 必须与当前扫描逐字一致 (确定性再生成).

重生成: `python -m huginn.cli.governance_audit --out docs/governance-state.md`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

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

# 工程门 (非 pytest 阻断门) 数下限棘轮: 同上, 值由台账权威声明.
ENGINEER_GATE_FLOOR = ga.budget_value("engineer_gate_floor")

# 工作流数下限棘轮: 同上, 值由台账权威声明.
WORKFLOW_FLOOR = ga.budget_value("workflow_floor")

# 快速失败门 (全部工作流 pytest guard step) 数下限棘轮: 同上.
GUARD_STEP_FLOOR = ga.budget_value("guard_step_floor")

# 工作流命名 step 总数下限棘轮 (阻断步静默增删的反向兜底): 同上.
WORKFLOW_STEP_FLOOR = ga.budget_value("workflow_step_floor")

# 能力锚点数下限棘轮 (命名子系统脱离治理面): 同上, 值由台账权威声明.
CAPABILITY_FLOOR = ga.budget_value("capability_floor")

# 循环闸位数下限棘轮: 同上, 值由台账权威声明.
LOOP_CAP_FLOOR = ga.budget_value("loop_cap_floor")

# prompt 段数下限棘轮 (prompt 段声明清单只减不增): 同上, 值由台账权威声明.
PROMPT_SEGMENT_FLOOR = ga.budget_value("prompt_segment_floor")


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


# ──────────────────── 工程门 (非 pytest 阻断门) ────────────────────


def test_engineer_gates_steps_exist():
    """不变量: 每个工程门 (`ENGINEER_GATE_SPECS`) 的 step 名仍在对应工作流里.

    工程门由工作流 step 承载 (非 pytest 测试文件), 旧台账完全看不见 —— step 删/改名即
    静默缩小治理面, 无任何信号.
    """
    reg = ga.build_gate_registry()
    assert reg["dangling_engineer"] == [], (
        f"悬空工程门 (step 名与工作流脱节): {reg['dangling_engineer']} —— "
        f"step 改名/删除后忘了同步 governance_audit.ENGINEER_GATE_SPECS."
    )


def test_engineer_gate_count_ratchet_only_shrinks():
    """棘轮: 工程门数只减不增."""
    reg = ga.build_gate_registry()
    assert len(reg["engineer_gates"]) >= ENGINEER_GATE_FLOOR, (
        f"工程门数从 {ENGINEER_GATE_FLOOR} 降到 {len(reg['engineer_gates'])} —— 门被删了? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"engineer_gate_floor 的 value (可见的发布决策)."
    )


def test_engineer_dangling_detection_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证工程门 step 名缺失能被抓到 (前向非空转)."""
    (tmp_path / "tests").mkdir()
    ci = (
        "jobs:\n  g:\n    steps:\n      - name: Some other step\n        run: echo hi\n"
    )
    reg = ga.audit(tmp_path, ci_text=ci)
    # 合成树里工作流只含一个无关 step, 故登记为 ci.yml 的工程门应全部判为不存在.
    assert any(d["workflow"] == "ci.yml" for d in reg["dangling_engineer"])


# ──────────────────── 工作流覆盖面 ────────────────────


def test_workflow_count_ratchet_only_shrinks():
    """棘轮: 被扫描工作流数只减不增 (整个工作流被删/改名 → 门集体消失, 无信号)."""
    reg = ga.build_gate_registry()
    assert len(reg["workflows"]) >= WORKFLOW_FLOOR, (
        f"工作流数从 {WORKFLOW_FLOOR} 降到 {len(reg['workflows'])} —— 工作流被删了? "
        f"若确为有意合并, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"workflow_floor 的 value (可见的发布决策)."
    )


def test_guard_step_count_ratchet_only_shrinks():
    """棘轮: 全部工作流的 pytest guard step 数只减不增.

    覆盖**非 arch** 的快速失败门 (如 testclient_hygiene / naturebench / security / stress):
    它们既不在 arch_gate_floor, 也不在工程门口径内 —— 删掉其 guard step 会静默缩小快速失败面.
    """
    reg = ga.build_gate_registry()
    guards = [g for g in reg["gates"] if g["status"] == "fast-fail-guard"]
    assert len(guards) >= GUARD_STEP_FLOOR, (
        f"快速失败门数从 {GUARD_STEP_FLOOR} 降到 {len(guards)} —— guard step 被删了? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"guard_step_floor 的 value (可见的发布决策)."
    )


def test_workflow_step_count_ratchet_only_shrinks():
    """棘轮: 全部工作流命名 step 总数只减不增 (阻断步静默增删的反向兜底).

    `ENGINEER_GATE_SPECS` 只挡「已登记工程门 → 被删」的**单向**; 未登记的阻断步被删/新增
    无任何信号. 命名 step 总数是一张反向网: 任何 step 增删都会动它, 须显式改台账 (可见决策).
    """
    reg = ga.build_gate_registry()
    total = sum(len(ga._parse_step_names(t)) for t in ga._read_workflow_texts(_AGENT).values())
    assert total >= WORKFLOW_STEP_FLOOR, (
        f"工作流命名 step 总数从 {WORKFLOW_STEP_FLOOR} 降到 {total} —— 有阻断步被删? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"workflow_step_floor 的 value (可见的发布决策)."
    )


# ──────────────────── 能力锚点 (capability anchors) ────────────────────


def test_capability_anchors_intact():
    """不变量: 每个命名子系统的锚点 (开关默认值 + 承载模块 + 不变量测试) 都不脱节.

    门存在性判据看不见能力面: 开关还在但**默认翻转** (从开变关 = 能力悄悄关掉)、承载模块
    被删 (= 能力失去实现)、不变量测试被删 (= 能力失去回归网) 都会让能力静默萎缩而门全绿.
    """
    reg = ga.build_gate_registry()
    assert reg["capability_gaps"] == [], (
        f"能力锚点脱节: {reg['capability_gaps']} —— "
        f"开关默认翻转/承载模块或回归测试被删/改名后忘了同步 CAPABILITY_SPECS 或补齐锚点."
    )


def test_capability_count_ratchet_only_shrinks():
    """棘轮: 能力锚点数只减不增 (删掉一个条目 = 该能力脱离治理面)."""
    reg = ga.build_gate_registry()
    assert len(reg["capabilities"]) >= CAPABILITY_FLOOR, (
        f"能力锚点数从 {CAPABILITY_FLOOR} 降到 {len(reg['capabilities'])} —— 能力被移出登记? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 capability_floor 的 value."
    )


def test_capability_flags_all_triaged():
    """不变量: 每个 feature flag 都已被「锚定 or 豁免」分诊 (能力面闭合, 无静默缺口).

    `CAPABILITY_SPECS` 逐条加锚点仍是**点修** —— 新加一枚功能开关时若无人登记, 能力面又
    出现静默缺口. 本门把 `FeatureFlags._DEFAULTS` 全集收成**闭合面**: 每个 flag 键必须
    **恰好**落在「`CAPABILITY_SPECS` 锚定 (feature_flag 型)」或 `SUBSYSTEM_EXEMPT_FLAGS`
    之一, 二者并集 = 全集且不相交. 新增开关未分诊 → 红 (须显式决定它是锚点还是豁免).
    """
    anchored = {
        s["flag"] for s in ga.CAPABILITY_SPECS if s["flag_kind"] == "feature_flag"
    }
    exempt = set(ga.SUBSYSTEM_EXEMPT_FLAGS)
    all_flags = set(ga._feature_flag_defaults(_AGENT))

    overlap = sorted(anchored & exempt)
    assert not overlap, (
        f"以下 flag 同时被锚定与豁免 (分诊表必须不相交): {overlap} —— "
        f"从 SUBSYSTEM_EXEMPT_FLAGS 移除 (既已锚定)."
    )
    untriaged = sorted(all_flags - anchored - exempt)
    assert not untriaged, (
        f"以下 feature flag 未分诊 (新增开关未登记为锚点或豁免): {untriaged} —— "
        f"要么在 governance_audit.CAPABILITY_SPECS 补锚点 (开关+承载模块+不变量测试三者"
        f"齐备), 要么在 SUBSYSTEM_EXEMPT_FLAGS 诚实登记豁免 (可见决策)."
    )
    stale = sorted((anchored | exempt) - all_flags)
    assert not stale, (
        f"以下 flag 已登记但不在 FeatureFlags._DEFAULTS (开关被删?): {stale} —— "
        f"同步 CAPABILITY_SPECS / SUBSYSTEM_EXEMPT_FLAGS."
    )


def test_capability_flag_default_parsing_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证: 开关默认值能从 FeatureFlags._DEFAULTS 读出 (前向非空转)."""
    (tmp_path / "huginn").mkdir()
    (tmp_path / "huginn" / "feature_flags.py").write_text(
        "class F:\n"
        "    _DEFAULTS: dict[str, bool] = {\"world_model\": True, \"x\": False}\n",
        encoding="utf-8",
    )
    assert ga._feature_flag_defaults(tmp_path) == {"world_model": True, "x": False}


def test_capability_env_default_parsing_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证: env 开关默认值能从 env_schema.SCHEMA 的 EnvSpec 读出 (前向非空转)."""
    (tmp_path / "huginn").mkdir()
    (tmp_path / "huginn" / "env_schema.py").write_text(
        "SCHEMA: dict[str, object] = {\n"
        "    \"HUGINN_PMK_INJECT\": EnvSpec(\"int\", \"0\", \"tuning\"),\n"
        "}\n",
        encoding="utf-8",
    )
    assert ga._env_schema_defaults(tmp_path) == {"HUGINN_PMK_INJECT": "0"}


def test_capability_default_flip_detected_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证: 开关默认翻转能被判为锚点脱节 (前向非空转).

    `world_model` 期望默认 True, 合成树里写成 False → flag_ok=False (翻转被抓);
    `rsi` 期望 False 且实际 False, `pmk` env 默认一致 → 均 flag_ok=True (无假阳).
    承载模块/测试在合成树里天然缺失, 故只断言 flag_ok 维度 (与缺模块/缺测试正交).
    """
    (tmp_path / "huginn").mkdir()
    (tmp_path / "huginn" / "feature_flags.py").write_text(
        "class F:\n"
        "    _DEFAULTS: dict[str, bool] = {\"world_model\": False, \"harness_meta_improver\": False}\n",
        encoding="utf-8",
    )
    (tmp_path / "huginn" / "env_schema.py").write_text(
        "SCHEMA: dict[str, object] = {\n"
        "    \"HUGINN_PMK_INJECT\": EnvSpec(\"int\", \"0\", \"tuning\"),\n"
        "}\n",
        encoding="utf-8",
    )
    caps = {c["id"]: c for c in ga._collect_capabilities(tmp_path)}
    assert caps["world_model"]["flag_ok"] is False
    assert caps["rsi"]["flag_ok"] is True
    assert caps["pmk"]["flag_ok"] is True


# ──────────────────── prompt 段生效面 (prompt segments) ────────────────────


def test_prompt_segment_count_ratchet_only_shrinks():
    """棘轮: prompt 段声明清单只减不增 (段静默消失 = 输出规范萎缩).

    prompt 面此前无门禁: 段被删/改名后主路径(or s7)静默少渲染一段, 无任何信号. 段数下限
    棘轮把"段清单萎缩"变成红灯 —— 删段必须显式改台账 `prompt_segment_floor` (可见决策).
    """
    spec = next(s for s in ga.BUDGET_SPECS if s["id"] == "prompt_segment_floor")
    live = ga._probe_current(spec, _AGENT, heavy=False)
    assert live is not None, "prompt 段声明活值读不到 —— 源文件/变量名失效, 棘轮静默失效."
    assert live >= PROMPT_SEGMENT_FLOOR, (
        f"prompt 段数从 {PROMPT_SEGMENT_FLOOR} 降到 {live} —— 段被删了? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"prompt_segment_floor 的 value (可见决策)."
    )


def test_prompt_specs_match_source():
    """不变量: 台账 `PROMPT_SPECS` 清单与源文件 `_APPLIES_TO` 段名一致 (声明面不分叉).

    清单有、源里无 = 段被删; 源里有、清单无 = 新增段未登记. 二者任一即段面漂移 → 红.
    """
    reg = ga.build_gate_registry()
    ps = reg["prompt_segments"]
    assert ps["missing_in_source"] == [], (
        f"台账 PROMPT_SPECS 有而源里无: {ps['missing_in_source']} —— 段被删? 同步清单."
    )
    assert ps["unregistered"] == [], (
        f"源里有而台账 PROMPT_SPECS 无: {ps['unregistered']} —— 新增段未登记? 同步清单."
    )


# ──────────────────── 工作流触发器 (workflow triggers) ────────────────────


def test_workflows_have_registered_trigger_specs():
    """不变量: 每个工作流都有触发规格, 且每个规格都落在真实工作流上.

    `WORKFLOW_TRIGGER_SPECS` 是可见决策表: 新增工作流不登记触发规格 (无法判其是否被收窄)
    → 红; 规格登记但工作流文件被删/改名 → 红.
    """
    reg = ga.build_gate_registry()
    assert reg["unregistered_workflows"] == [], (
        f"未登记触发规格的工作流: {reg['unregistered_workflows']} —— "
        f"在 governance_audit.WORKFLOW_TRIGGER_SPECS 显式登记其期望触发集 (可见决策)."
    )
    assert reg["unchecked_workflows"] == [], (
        f"触发规格未落地的工作流: {reg['unchecked_workflows']} —— "
        f"规格登记但工作流文件缺失 (被删/改名)? 同步 WORKFLOW_TRIGGER_SPECS / 工作流."
    )


def test_workflow_triggers_match_spec():
    """不变量: 每个工作流的实际触发器与期望集一致 (门存在却永不运行的假绿).

    门 step 存在 ≠ 门会运行: 删 `on: schedule` 或收窄 `push`/`pull_request` 的 `paths` 会让门
    永不执行, 而 step 仍在文件里. 触发规格把「运行面」也纳入登记, 使收窄成为可见决策.
    """
    reg = ga.build_gate_registry()
    assert reg["trigger_drift"] == [], (
        f"触发器漂移 (实际触发集 ≠ 期望集): {reg['trigger_drift']} —— "
        f"工作流触发器被删/收窄(门永不运行)或新增(未登记). 同步 WORKFLOW_TRIGGER_SPECS / 工作流."
    )


def test_trigger_parsing_on_synthetic_tree(tmp_path: pathlib.Path):
    """合成树验证触发器抽取与漂移判定 (前向非空转).

    单值 `on: push`、行内列表 `on: [push, pull_request]`、块状 `on:` 三种写法都要被正确识别;
    期望集里缺 `schedule` 时须判为漂移.
    """
    # 块状写法: 缺 schedule (期望含 schedule) → 漂移
    block = "on:\n  push:\n    branches: [main]\n  workflow_dispatch:\n"
    assert ga._parse_triggers(block) == ["push", "workflow_dispatch"]
    # 行内列表写法
    assert ga._parse_triggers("on: [push, pull_request]\n") == ["pull_request", "push"]
    # 单值写法
    assert ga._parse_triggers("on: push\n") == ["push"]
    # 漂移判定: 一个期望 schedule 的工作流只声明 push
    wf = {"audit-regression.yml": "on:\n  push:\n"}
    got = ga._collect_workflow_triggers(wf)
    assert got["trigger_drift"] and got["trigger_drift"][0]["missing"] == [
        "schedule",
        "workflow_dispatch",
    ]


# ──────────────────── 跨环境解析确定性 ────────────────────


def test_yaml_and_fallback_parsing_agree(tmp_path: pathlib.Path, monkeypatch):
    """不变量: 装了 / 没装 PyYAML 两条解析路径给出**逐字一致**的登记快照.

    台账对工作流/pre-commit 的解析都走「有 PyYAML 用主路径, 无则退化行扫描」。若两路径
    判据不一致, 同一份文件在 CI (装了 PyYAML) 与本地 (没装) 会给出不同门集 —— 幻影门或漏门,
    且再生成文档随环境漂移。合成树里在步骤之间放一条 job 级注释 (同时含 `pytest` 与
    `tests/*.py` 字样): 只有**按缩进收边**的退化解析才不会把它并进上一个 step 块。
    """
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )
    ci = (
        "jobs:\n"
        "  lint:\n"
        "    steps:\n"
        "      - name: Ruff only\n"
        "        run: |\n"
        "          ruff check .\n"
        "  test:\n"
        "    # 单一测试 job: pytest 自动发现整个 tests/ (含 tests/test_a.py 护栏)\n"
        "    steps:\n"
        "      - name: Real guard\n"
        "        run: python -m pytest tests/test_a.py -q\n"
    )
    with_yaml = ga.audit(tmp_path, ci_text=ci, precommit_text="")
    monkeypatch.setitem(sys.modules, "yaml", None)
    without_yaml = ga.audit(tmp_path, ci_text=ci, precommit_text="")
    assert without_yaml == with_yaml, (
        "yaml 主路径与无-PyYAML 退化路径解析结果不一致 —— 环境相关的幻影门/漏门。"
    )


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
        "engineer_gate_floor",
        "workflow_floor",
        "guard_step_floor",
        "workflow_step_floor",
        "capability_floor",
        "prompt_segment_floor",
        "field_audit_golden",
        "skill_audit_baseline",
        "gitleaks_allowlist",
        "pip_audit_ignore",
        "dependabot_ignore",
        "coverage_fail_under",
        "loop_autoloop_cli",
        "loop_autoloop_engine",
        "loop_recursion_limit",
        "loop_tool_calls",
        "loop_tool_calls_per_tool",
        "loop_codeact_turns",
        "loop_coder_iter",
        "loop_exploration",
        "loop_deep_research",
        "loop_rcb_step2_iters",
        "loop_harness_phase_iter",
        "loop_skill_step",
        "loop_skill_preset",
        "loop_codeact_sandbox_turns",
        "loop_ws_auto_continue",
        "loop_agent_profile_steps",
        "loop_causal_runner_tool_calls",
        "loop_deli_research",
        "loop_cap_floor",
        "runtime_probe_floor",
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
    """预算 `owner_kind` 口径统一: `ci_step` / `workflow_step` 的消费者门必须是真实 step 名.

    `owner_gate` 混用三种语义: `ci_step` (工作流 pytest/CLI guard step)、`workflow_step`
    (非 pytest 工程门 step)、`full_suite` (仅全量套件跑, 不在工作流). 若不区分, step 改名后
    台账会指向**幻影门**而无声. 本门固化: ①`owner_kind` 合法枚举; ②`ci_step` 型 `owner_gate`
    命中工作流真实 pytest/CLI guard step 名; ③`workflow_step` 型命中 ENGINEER_GATE_SPECS 的
    step 名 (取 `engineer_gates` 的 step 集).
    """
    reg = ga.build_gate_registry()
    ci_steps = {g["ci_step"] for g in reg["gates"] if g["ci_step"]}
    ci_steps |= {c["ci_step"] for c in reg["cli_guards"] if c["ci_step"]}
    workflow_steps = {e["step"] for e in reg["engineer_gates"] if e["exists"]}
    for spec in ga.BUDGET_SPECS:
        kind = spec["owner_kind"]
        assert kind in {"ci_step", "workflow_step", "full_suite"}, (
            f"预算 `{spec['id']}` 的 owner_kind `{kind}` 非法 "
            f"(须 ci_step / workflow_step / full_suite)."
        )
        if kind == "ci_step":
            assert spec["owner_gate"] in ci_steps, (
                f"预算 `{spec['id']}` 标 `ci_step`, 但 owner_gate `{spec['owner_gate']}` 不是 "
                f"工作流真实 pytest/CLI guard step 名 —— step 改名后台账指向幻影门? "
                f"同步 BUDGET_SPECS / 工作流."
            )
        if kind == "workflow_step":
            assert spec["owner_gate"] in workflow_steps, (
                f"预算 `{spec['id']}` 标 `workflow_step`, 但 owner_gate `{spec['owner_gate']}` "
                f"不是 ENGINEER_GATE_SPECS 的真实工程门 step 名 —— step 改名后台账指向幻影门? "
                f"同步 BUDGET_SPECS / ENGINEER_GATE_SPECS."
            )


# ──────────────────── 循环闸位 (loop caps) ────────────────────


def test_loop_caps_live_within_ceiling():
    """不变量: 每个循环闸位仍能读到活值, 且活值 ≤ 台账上限 (只增不减的反向棘轮).

    循环闸位是自主轮数/步数上限; 提高上限 = 扩大自主面, 必须走台账 (可见决策).
    """
    caps = [
        s
        for s in ga.BUDGET_SPECS
        if s.get("source") == "ceiling" and s.get("group") == "loop"
    ]
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


# ──────────────────── 能力可运行性探针 (runtime probe) ────────────────────


def test_runtime_probe_floor_ratchet_only_shrinks():
    """可运行性探针清单只增不减: SMOKE_SPECS 条目数 ≥ 台账下限.

    L0/L1 能力可运行性面的探针清单 (SMOKE_SPECS) 若被悄悄删除, 「哪些能力因缺依赖
    跑不了」的观察面就无声萎缩 —— 与门全绿同时发生. 本棘轮兜底: 条目数下降即红.
    """
    floor = ga.budget_value("runtime_probe_floor")
    spec = next(s for s in ga.BUDGET_SPECS if s["id"] == "runtime_probe_floor")
    live = ga._probe_current(spec, _AGENT, heavy=False)
    assert live is not None, "探针清单活值读不到 —— path/变量名失效, 棘轮静默失效."
    assert live >= floor, (
        f"可运行性探针清单从 {floor} 降到 {live} —— 探针被删了? "
        f"若确为有意删除, 显式下调 governance_audit.BUDGET_SPECS 里 "
        f"runtime_probe_floor 的 value."
    )


# ──────────────────── 豁免/白名单面 + 配置级门 ────────────────────

# 豁免/白名单 (放宽即静默削弱对应门) 与配置级门 (覆盖率下限) 的冻结 id. 增删须同步此集.
_EXEMPTION_CEILING_IDS = ("gitleaks_allowlist", "pip_audit_ignore", "dependabot_ignore")
_CONFIG_FLOOR_IDS = ("coverage_fail_under",)


def test_exemption_whitelists_not_relaxed():
    """不变量: 每个豁免/白名单条数 ≤ 台账上限 (放宽豁免 = 静默削弱对应门).

    gitleaks 密钥豁免 / pip-audit 漏洞豁免 / dependabot 忽略项都是「把某个门对某条免除」——
    往里加一条即扩大免检面. 台账持有当前条数作 ceiling; 加一条即越界 → 红 (须显式抬高 value).
    """
    by = {b["id"]: b for b in ga.collect_budgets(_AGENT)}
    for bid in _EXEMPTION_CEILING_IDS:
        b = by[bid]
        assert b["current"] is not None, (
            f"豁免项 `{bid}` 实测读不到 —— 源文件位置/写法变了? 同步 BUDGET_SPECS."
        )
        assert b["current"] <= b["headroom"] + b["current"], f"`{bid}` 余量为负"
        assert int(b["frozen"]) >= b["current"], (
            f"豁免项 `{bid}` 实测 {b['current']} > 台账上限 {b['frozen']} —— "
            f"往豁免名单加条 = 静默放宽免检面. 显式抬高 BUDGET_SPECS 的 value (可见决策)."
        )


def test_config_coverage_floor_not_lowered():
    """不变量: 覆盖率下限 `fail_under` ≥ 台账下限 (下调 = 静默削弱覆盖率门)."""
    by = {b["id"]: b for b in ga.collect_budgets(_AGENT)}
    for bid in _CONFIG_FLOOR_IDS:
        b = by[bid]
        assert b["current"] is not None, (
            f"配置级门 `{bid}` 实测读不到 —— 源文件位置/写法变了? 同步 BUDGET_SPECS."
        )
        assert b["current"] >= int(b["frozen"]), (
            f"配置级门 `{bid}` 实测 {b['current']} < 台账下限 {b['frozen']} —— "
            f"下调覆盖率下限 = 静默削弱覆盖率门. 显式下调 BUDGET_SPECS 的 value (可见决策)."
        )
