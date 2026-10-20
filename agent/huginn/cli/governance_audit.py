"""低熵治理台账 (governance_audit): 治理门登记 + 孤儿门检测 + 确定性再生成.

治理层**自身**的熵有十种典型形态, 本模块把它们变成可观察、可门禁的事实:

1. **孤儿门** —— `tests/test_arch_*.py` 门族与显式声明的治理元门
   (`GOVERNANCE_META_TESTS`) 存在, 却既未接入**任工作流**的 guard step, 也未显式声明
   「只靠全量套件兜底」(状态未知 = 熵).
2. **悬空门** —— 任工作流的 guard step 指向一个已被改名/删除的测试文件
   (门永远绿灯, 因为它根本没跑).
3. **CLI 门漏登记** —— 工作流里 `python -m huginn.<module>` 形态的治理门
   (CLI 自查 / 棘轮, 如 contract_audit 棘轮), 旧解析只认 pytest 会漏掉; 现单列.
4. **工程门漏登记** —— 非 pytest 的阻断门 (ruff lint / bandit SAST / pip-audit /
   npm audit / cargo audit / gitleaks / lock 漂移 / 前端构建·单测·E2E / 技能库审计 /
   ultra-long benchmark / quickstart smoke), 删掉即静默缩小治理面; 现由
   `ENGINEER_GATE_SPECS` 登记并判存在 (存在性门).
5. **基线漂移** —— 棘轮/预算冻结值散落在 `tests/golden/*`、`huginn/`、`scripts/` 与各
   测试常量里, 没有单一台账可读; 现由 `BUDGET_SPECS` 收成一张表 (含自主循环的轮数/步数
   **上限** —— 散在 `env_schema` 与各引擎/路由默认里, 作 ceiling 棘轮单列).
6. **本地门漏登记/悬空/漏匹配** —— `.pre-commit-config.yaml`(仓根与 `agent/` 两份) 的
   本地门 (如 G37 audit-regression) 既不在工作流也不在台账; 其 `entry` 引用的测试改名/
   删除即 **悬空**, 而 `entry` 引用的测试未被 `files:` 过滤器覆盖时, 改该测试不会触发
   hook —— 门静默失效, 无任何信号.
7. **工作流覆盖面萎缩** —— 门可能因整个工作流文件被删/改名而集体消失, 无任何信号; 现由
   `workflow_floor` (工作流数下限) 兜底.
8. **触发器漂移 (门存在却永不运行)** —— 门 step 仍在文件里, 但承载它的工作流**触发器**
   (`on: schedule` / `push` / `pull_request`) 被删或收窄 (`paths` 过滤掉改动) —— 门永不
   执行, 而存在性判据仍判「在」(假绿). 现由 `WORKFLOW_TRIGGER_SPECS` 登记每工作流的
   期望触发集并判在否.
9. **阻断步漏登记 / 静默增删** —— 非 pytest 阻断门可能整步被删或新增却无人登记;
   `ENGINEER_GATE_SPECS` 只做「已登记→判存在」单向, 故补反向 `workflow_step_floor`
   (全部工作流命名 step 总数下限) —— 任何未登记的步骤增删都抬不动/降不下即红 (可见决策).
10. **能力面静默萎缩** —— 前面九形态管的都是治理的**执行面** (门在不在跑、棘轮有没有被放宽),
    却看不见**能力面**: 一个命名子系统 (RSI / PMK / 世界模型 …) 靠「一枚开关 + 一组承载模块
    + 一组不变量测试」立起来, 三者任一被删/改名/**默认翻转** (开关还在但默认从开变关 =
    能力悄悄关掉) 即静默萎缩, 而门存在性判据全绿. 现由 `CAPABILITY_SPECS` 登记每个能力的
    锚点并逐条判在否.
11. **prompt 段面静默萎缩** —— 输出规范的执行单元是 prompt 段; 段注册了却不在任何生产路径
    生效 (孤儿段), 或段清单被删/改名而无人登记, 都让输出面无声萎缩, 而门存在性判据全绿.
    现由 `PROMPT_SPECS` 登记期望段名清单 + 按文件路径 ast 读源文件 `_APPLIES_TO` (段名 ->
    必须生效的生产路径) 逐段判在否; 「声明 == 渲染」由 `tests/test_prompt_contract.py` 硬判,
    段数下限棘轮 `prompt_segment_floor` 兜底.

同时把**豁免/白名单面** (gitleaks 密钥豁免条数、`ci.yml` 里 pip-audit `--ignore-vuln` 串、
`dependabot.yml` 的 `ignore:` 项数) 与**配置级门** (pyproject 覆盖率下限 `fail_under`) 一并
纳入 `BUDGET_SPECS` 作棘轮 —— 放宽豁免或下调覆盖率下限 = 静默削弱对应门, 须显式改台账.

扫描 `tests/test_arch_*.py` 门族 + 声明式治理元门 + `.github/workflows/*.yml` **全部**工作流的
两类 python guard step (pytest 与 `python -m huginn.<module>`) + 非 pytest 工程门 +
`.pre-commit-config.yaml` 本地门 + 各工作流触发器 + 命名子系统的能力锚点
(`CAPABILITY_SPECS`: 开关默认值 / 承载模块 / 不变量测试) + prompt 段的生效面声明
(`PROMPT_SPECS` 清单 + 源文件 `_APPLIES_TO`: 段名 -> 必须生效的生产路径), 并把散落的
棘轮预算/基线收成单一台账, 渲染成 `docs/governance-state.md`.

设计 (对齐 config_audit / contract_audit):
- **纯 stdlib**: 不 import huginn.*, 不硬依赖 yaml (有则用, 无则退化行扫描),
  保证门禁与生成器都 weightless.
- **确定性输出**: 固定列序 + 字典序, 跨机逐字一致.
- **只提示不判死**: 硬判定交给 `tests/test_arch_gate_registry.py`.

重生成: ``python -m huginn.cli.governance_audit --out docs/governance-state.md``.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# huginn/cli/governance_audit.py -> parents[2] == agent/ (tests/ 与 docs/ 的宿主).
AGENT_ROOT = Path(__file__).resolve().parents[2]

# 显式声明「只靠全量套件兜底、不做快速失败门」的门测试 (可见决策).
# 新增 arch / 治理元门若不接入任工作流 guard 且不在此列, 会被判为孤儿门 (强制一次明确选择).
FULL_SUITE_ONLY = frozenset(
    {
        "tests/test_arch_cleanliness.py",
        "tests/test_arch_no_import_cycles.py",
        "tests/test_arch_rag_memory_seam.py",
        "tests/test_arch_seam_integrity.py",
        "tests/test_arch_tools_direction.py",
        # 治理元门 (契约漂移 / MECE 审计 / 环境面棘轮 / ponytail 债清单 / 镜像漂移):
        # 均为重型治理自检, 跑全量套件兜底; 显式声明后不再判为孤儿门.
        "tests/test_config_contracts.py",
        "tests/test_contract_audit.py",
        "tests/test_env_surface_governance.py",
        "tests/test_ponytail_debt_registry.py",
        "tests/test_publish_mirror_drift.py",
        # prompt「声明↔生效」一致性门 (本 spec 落地): weightless, 全量套件兜底.
        # 注: 钩子事件面「声明↔触发↔注册」已有 tests/test_contract_audit.py 全量覆盖
        # (build_hook_contract 逐事件判 wired/trigger-only), 不再新建重复门 —— 单一来源.
        "tests/test_prompt_contract.py",
    }
)

# 治理元门族: 除 `tests/test_arch_*.py` 外, 显式声明的治理自检测试. 它们同样受"孤儿门"
# 逻辑约束 —— 存在却既未接入工作流 guard, 也未声明 full-suite-only → 孤儿门. 这是一张
# **可见决策表**: 新增治理元门须显式登记, 否则红.
GOVERNANCE_META_TESTS = frozenset(
    {
        "tests/test_config_contracts.py",
        "tests/test_contract_audit.py",
        "tests/test_env_surface_governance.py",
        "tests/test_ponytail_debt_registry.py",
        "tests/test_publish_mirror_drift.py",
        "tests/test_prompt_contract.py",
    }
)

# 工作流触发器登记: 每工作流的**期望触发集** (on:). 门 step 存在 ≠ 门会运行 —— 删
# `on: schedule` 或收窄 push/pull_request 的 `paths` 会让门永不执行, 而 step 仍在文件里
# (存在性判据判「在」= 假绿). 新增工作流若不登记触发规格, 亦会被判缺口 (强制一次明确选择).
WORKFLOW_TRIGGER_SPECS: dict[str, tuple[str, ...]] = {
    "audit-regression.yml": ("schedule", "workflow_dispatch"),
    "ci.yml": ("push", "pull_request"),
    "coverage-matrix.yml": ("pull_request", "push", "schedule", "workflow_dispatch"),
    "desktop.yml": ("push", "pull_request", "workflow_dispatch"),
    "release.yml": ("push",),
    "skill-audit.yml": ("schedule", "workflow_dispatch", "push", "pull_request"),
    "stress-test.yml": ("pull_request", "schedule", "workflow_dispatch"),
}

# ── prompt 段生效面登记 (weightless: 只登记段名, 不 import prompt_segments 重栈) ──
# prompt 段的**声明面** (`prompt_segments._APPLIES_TO`: 段名 → 必须生效的生产路径
# {main,s7,autoloop}) 与**生效面** (各路径实际渲染点) 的一致性, 由
# `tests/test_prompt_contract.py` 硬判定. 这里登记**期望段名清单**作可见决策表: 段被删
# (清单有、源里无) 或新增未登记 (源里有、清单无) → 台账缺口可见, 段名下限棘轮兜底.
# 逐段的生效路径**不在此复制** —— 台账按文件路径 ast 读源文件 `_APPLIES_TO`, 无镜像漂移.
PROMPT_SPECS: tuple[str, ...] = (
    "persona",
    # §3/§4/§7 从 persona 串剥离成的独立段 (单一事实来源 huginn.prompts.PROMPT_SECTIONS).
    "core_principles",
    "behavioral_discipline",
    "response_format",
    "mode",
    "phase",
    "metacog",
    "tools",
    "multimodal",
    "writing",
    "multi_agent_protocol",
    "thinking",
    "safety",
)

# prompt 段声明面源文件 (相对 agent/): ast 解析其 `_APPLIES_TO` 字面量 (不 import 重栈).
_PROMPT_SEGMENTS_MODULE = "huginn/plugins/prompt_segments.py"

_ARCH_GLOB = "test_arch_*.py"
_TEST_PATH_RE = re.compile(r"tests/[A-Za-z0-9_./-]+\.py")
# 非 pytest 形态的治理门: `python -m huginn.<module>` (CLI 自查 / 棘轮). 旧解析只认
# pytest, 会漏掉 contract_audit 棘轮、policy_engine 自查这类门.
_CLI_GUARD_RE = re.compile(r"python\s+-m\s+(huginn\.[A-Za-z0-9_.]+)")


def discover_arch_tests(agent_root: Path) -> list[str]:
    """仓内 `tests/test_arch_*.py` (相对 agent_root 的 posix 路径), 字典序."""
    tests_dir = agent_root / "tests"
    if not tests_dir.is_dir():
        return []
    return sorted(f"tests/{p.name}" for p in tests_dir.glob(_ARCH_GLOB))


def discover_governance_tests(agent_root: Path) -> list[str]:
    """治理门族 = `tests/test_arch_*.py` + 存在着的显式治理元门 (`GOVERNANCE_META_TESTS`).

    旧孤儿检测只看 `test_arch_*.py`, 使 `test_config_contracts.py` / `test_contract_audit.py`
    这类治理元门游离在登记面之外 —— 它们若不再被任何工作流跑也无信号. 现并入同一判据.
    """
    found = set(discover_arch_tests(agent_root))
    found |= {t for t in GOVERNANCE_META_TESTS if (agent_root / t).exists()}
    return sorted(found)


def _parse_guard_steps(text: str) -> list[tuple[str, str]]:
    """从 ci.yml 文本抽出 (step_name, test_path). 优先 yaml, 退化行扫描."""
    try:
        import yaml  # 可选依赖: 有则精确, 无则退化

        data = yaml.safe_load(text) or {}
        out: list[tuple[str, str]] = []
        for job in (data.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                run = step.get("run") or ""
                if "pytest" in run:
                    m = _TEST_PATH_RE.search(run)
                    if m:
                        out.append((step.get("name") or "", m.group(0)))
        if out:
            return out
    except Exception:  # noqa: BLE001 — yaml 缺失/解析失败则退化为行扫描
        logger.debug("yaml 解析 ci.yml 失败, 退化为行扫描", exc_info=True)
    # 行扫描退化: 每个 `- ` 列表项起一个 step 块, 块 = 该行 + 缩进**更深**的续行;
    # 遇到缩进 ≤ 破折号缩进的非空行即收尾 (即下一个 `- ` 步骤或上一层 job 键)。
    # 关键是**按缩进收边**, 否则 job 级注释 (可能同时出现 pytest 与 tests/*.py 字样)
    # 会被并进上一个 step 块, 造出 yaml 主路径没有的幻影门 (跨环境非确定性)。
    # 收边后再对整块施加与主路径**相同**的判据: 整块含 pytest → 取块内首个 tests/*.py。
    out = []
    lines = text.splitlines()
    i = 0
    total = len(lines)
    while i < total:
        m = re.match(r"^(\s*)- (.*)$", lines[i])
        if not m:
            i += 1
            continue
        dash_indent = len(m.group(1))
        block = [lines[i]]
        i += 1
        while i < total:
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip(" "))) <= dash_indent:
                break
            block.append(line)
            i += 1
        body = "\n".join(block)
        name = ""
        mn = re.match(r"^\s*-\s*name:\s*(.*)$", block[0])
        if mn:
            name = mn.group(1).strip()
        else:
            mn = re.search(r"^\s+name:\s*(.*)$", body, re.M)
            if mn:
                name = mn.group(1).strip()
        if "pytest" in body:
            mp = _TEST_PATH_RE.search(body)
            if mp:
                out.append((name, mp.group(0)))
    return out


def _parse_step_names(text: str) -> list[str]:
    """抽出工作流**全部** step 的 name (工程门存在性判据). 优先 yaml, 退化行扫描.

    与 `_parse_guard_steps` 共用 yaml/fallback 两路径哲学: 结果只取决于文件内容, 而非
    运行环境是否装了 PyYAML (确定性的前提).
    """
    try:
        import yaml  # 可选依赖: 有则精确, 无则退化

        data = yaml.safe_load(text) or {}
        out: list[str] = []
        for job in (data.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                name = step.get("name")
                if name:
                    out.append(name)
        if out:
            return out
    except Exception:  # noqa: BLE001 — yaml 缺失/解析失败则退化为行扫描
        logger.debug("yaml 解析工作流失败, 退化为行扫描", exc_info=True)
    out = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- name:"):
            out.append(s[len("- name:") :].strip())
    return out


def _read_workflow_texts(agent_root: Path) -> dict[str, str]:
    """读 `<root>/../.github/workflows/*.yml(|.yaml)` 全部工作流 (文件名 -> 文本), 字典序.

    治理门不止活在 ci.yml: audit-regression.yml / skill-audit.yml / stress-test.yml 也各承载
    门 (G37 回归 / 技能库审计 / ultra-long benchmark). 旧实现只读 ci.yml, 使这些门对台账不可见
    —— 删掉它们无声. 现全量扫描.
    """
    wf_dir = agent_root.parent / ".github" / "workflows"
    if not wf_dir.is_dir():
        return {}
    files = [*wf_dir.glob("*.yml"), *wf_dir.glob("*.yaml")]
    return {
        p.name: p.read_text(encoding="utf-8") for p in sorted(files, key=lambda q: q.name)
    }


def _parse_triggers(text: str) -> list[str]:
    """抽出工作流顶层 `on:` 的触发器键 (如 `push`/`pull_request`/`schedule`/`workflow_dispatch`).

    只做**行扫描** (不用 PyYAML): PyYAML 会把裸键 `on` 解析成布尔 `True`, 处理反而更绕; 且
    单一解析路径保证跨环境 (装不装 PyYAML) 输出逐字一致 —— 确定性再生成的前提. 只取 `on:`
    块里与首个子键同缩进的一级键, 故 `schedule: - cron:` 的列表项与 `branches/paths` 二级键
    不会混入.
    """
    lines = text.splitlines()
    found: list[str] = []
    base: int | None = None
    key_indent: int | None = None
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        s = line.strip()
        if base is None:
            if indent == 0 and re.match(r"^on\s*:", s):
                base = indent
                rest = s.split(":", 1)[1].strip()
                if rest and not rest.startswith("#"):
                    token = rest.split("#", 1)[0].strip()
                    if token.startswith("["):  # 行内列表 on: [push, pull_request]
                        found.extend(re.findall(r"[A-Za-z_]+", token))
                    elif token:  # 单值 on: push
                        found.append(token)
            continue
        if indent <= base:  # 下一个顶层键 → on: 块结束
            break
        if s.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z_]+)\s*:", s)
        if not m:
            continue
        if key_indent is None:
            key_indent = indent
        if indent == key_indent:
            found.append(m.group(1))
    return sorted(set(found))


def _collect_workflow_triggers(workflows: dict[str, str]) -> dict:
    """比对每个工作流的**实际触发器**与 `WORKFLOW_TRIGGER_SPECS` 的**期望集**.

    门 step 存在 ≠ 门会运行: 删 `on: schedule` 或收窄 push/pull_request 的 `paths` 会让门
    永不执行, 而 step 仍在文件里 (存在性判据判「在」= 假绿). 同时暴露两个反向缺口:
    ①**未登记工作流** (文件在但无触发规格 —— 新增工作流须显式登记, 强制一次明确选择);
    ②**未落地规格** (规格登记但文件不存在 —— 工作流被删/改名).
    """
    registered: list[dict] = []
    unregistered: list[str] = []
    for wf, text in workflows.items():
        spec = WORKFLOW_TRIGGER_SPECS.get(wf)
        if spec is None:
            unregistered.append(wf)
            continue
        actual = _parse_triggers(text)
        expected = sorted(spec)
        registered.append(
            {
                "workflow": wf,
                "expected": expected,
                "actual": actual,
                "missing": sorted(set(expected) - set(actual)),
                "extra": sorted(set(actual) - set(expected)),
            }
        )
    registered.sort(key=lambda r: r["workflow"])
    return {
        "workflow_triggers": registered,
        "unregistered_workflows": sorted(unregistered),
        "unchecked_workflows": sorted(set(WORKFLOW_TRIGGER_SPECS) - set(workflows)),
        "trigger_drift": [r for r in registered if r["missing"] or r["extra"]],
    }


def _cli_guard_in_line(line: str) -> str | None:
    """一行里若有 CLI 门则返回模块名, 否则 ``None``.

    注释行 (含 `# 重生成: python -m huginn...`) 与 pytest 行都不算门 —— yaml 主路径
    与逐行 fallback **共用此判据**, 使结果只取决于 ci.yml 内容, 而非运行环境是否装了
    PyYAML (否则同一份文件会给出不同门集, 破坏确定性再生成).
    """
    s = line.strip()
    if s.startswith("#") or "pytest" in s:
        return None
    m = _CLI_GUARD_RE.search(s)
    return m.group(1) if m else None


def _parse_cli_guards(text: str) -> list[tuple[str, str]]:
    """从 ci.yml 抽出 `python -m huginn.<module>` 形态的门 `(step_name, module)`.

    这是**非 pytest** 的治理门 (CLI 自查 / 棘轮) —— 旧解析只认 pytest, 会漏掉
    contract_audit 棘轮、policy_engine 自查这类门. 只收 ``huginn.*``, 排除
    pip / pytest / uv / bandit 等.
    """
    try:
        import yaml  # 可选依赖: 有则精确, 无则退化

        data = yaml.safe_load(text) or {}
        out: list[tuple[str, str]] = []
        for job in (data.get("jobs") or {}).values():
            for step in (job or {}).get("steps") or []:
                name = step.get("name") or ""
                for line in (step.get("run") or "").splitlines():
                    module = _cli_guard_in_line(line)
                    if module:
                        out.append((name, module))
        if out:
            return out
    except Exception:  # noqa: BLE001 — yaml 缺失/解析失败则退化为行扫描
        logger.debug("yaml 解析 ci.yml 失败, 退化为行扫描", exc_info=True)
    out = []
    name = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            name = stripped[len("- name:") :].strip()
            continue
        module = _cli_guard_in_line(line)
        if module:
            out.append((name, module))
    return out


def _read_precommit_texts(agent_root: Path) -> dict[str, str]:
    """读仓根与 `agent/` 两份 `.pre-commit-config.yaml` (相对标签 -> 文本).

    `agent/` 下的那份用于 IDE 内 pre-commit, 也是本地门载体; 旧实现只读仓根那份, 第二份
    不可见.
    """
    out: dict[str, str] = {}
    for label, path in (
        (".pre-commit-config.yaml", agent_root.parent / ".pre-commit-config.yaml"),
        ("agent/.pre-commit-config.yaml", agent_root / ".pre-commit-config.yaml"),
    ):
        if path.exists():
            out[label] = path.read_text(encoding="utf-8")
    return out


def _parse_precommit_hooks(text: str) -> list[tuple[str, str]]:
    """抽出 `.pre-commit-config.yaml` 本地门引用的 `(hook_id, test_path)`.

    本地门 (如 G37 audit-regression) 在 `entry` 里 pytest 若干测试. 其引用的测试文件
    改名/删除后 hook 的 `files:` 过滤器静默不再匹配, 门悄悄死掉 —— 故须登记并判存在.
    优先 yaml, 退化行扫描; 两路径共用同一判据 (只认 `entry` 里的 `tests/*.py`).
    """
    try:
        import yaml  # 可选依赖: 有则精确, 无则退化

        data = yaml.safe_load(text) or {}
        out: list[tuple[str, str]] = []
        for repo in data.get("repos") or []:
            for hook in (repo or {}).get("hooks") or []:
                hid = hook.get("id") or ""
                for m in _TEST_PATH_RE.finditer(hook.get("entry") or ""):
                    out.append((hid, m.group(0)))
        if out:
            return out
    except Exception:  # noqa: BLE001 — yaml 缺失/解析失败则退化为行扫描
        logger.debug("yaml 解析 .pre-commit-config.yaml 失败, 退化为行扫描", exc_info=True)
    # 行扫描退化: 记住最近的 `- id:`, 只从 `entry:` 字段 (含其多行续行) 抽 tests/*.py.
    # **字段感知**: 显式跳过 `files:` 等其它字段行 —— 否则当 `files:` 写成裸路径
    # (`^agent/tests/x\.py$`) 时会被误当 entry 引用, 与 yaml 主路径分叉 (幻影门).
    out = []
    hid = ""
    in_entry = False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("- id:"):
            hid = s[len("- id:") :].strip()
            in_entry = False
            continue
        field = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*):", s)
        if field:  # 字段起始行: entry 起算入口, 其它字段 (files/name/...) 关入口
            in_entry = field.group(1) == "entry"
            if in_entry:
                for mp in _TEST_PATH_RE.finditer(s):
                    out.append((hid, mp.group(0)))
            continue
        if in_entry:  # `entry:` 的多行续行 (如 bash -c '... \ 换行)
            for mp in _TEST_PATH_RE.finditer(s):
                out.append((hid, mp.group(0)))
    return out


def _parse_precommit_files(text: str) -> dict[str, str]:
    """抽出 `.pre-commit-config.yaml` 每个本地 hook 的 `files:` 过滤器 (hook_id -> 正则).

    本地门靠 `files:` 决定何时触发; 若 `entry` 引用的测试不在过滤范围内, 改该测试不会
    触发 hook —— 门静默失效. 优先 yaml, 退化行扫描; 两路径共用判据 (字段起始 `files:`).
    """
    out: dict[str, str] = {}
    try:
        import yaml  # 可选依赖: 有则精确, 无则退化

        data = yaml.safe_load(text) or {}
        for repo in data.get("repos") or []:
            for hook in (repo or {}).get("hooks") or []:
                hid = hook.get("id") or ""
                files = hook.get("files")
                if hid and isinstance(files, str):
                    out[hid] = files
        if out:
            return out
    except Exception:  # noqa: BLE001 — yaml 缺失/解析失败则退化为行扫描
        logger.debug("yaml 解析 .pre-commit-config.yaml 失败, 退化为行扫描", exc_info=True)
    hid = ""
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("- id:"):
            hid = s[len("- id:") :].strip()
            continue
        m = re.match(r"^files:(.*)$", s)
        if m and hid:
            out[hid] = m.group(1).strip().strip("'\"")
    return out


# ── 工程门登记 (非 pytest 的阻断门) ──
# ruff lint / bandit SAST / pip-audit / npm audit / cargo audit / gitleaks / lock 漂移 /
# 前端构建·单测·E2E / 技能库审计 / ultra-long benchmark / quickstart smoke 等: 这些门
# 由工作流 step 承载 (非 pytest 测试文件), 删/改名即静默缩小治理面 —— 旧台账完全看不见.
# 这里登记每门的 (工作流文件, step 名, 类别); 台账只读 + 判存在, 硬判定在元门禁里.
ENGINEER_GATE_SPECS: tuple[dict, ...] = (
    {"id": "lint_ruff", "workflow": "ci.yml", "kind": "lint", "step": "Run ruff check (gating)"},
    {"id": "pip_audit", "workflow": "ci.yml", "kind": "supply-chain", "step": "Dependency vulnerability scan"},
    {"id": "npm_audit", "workflow": "ci.yml", "kind": "supply-chain", "step": "npm dependency vulnerability scan"},
    {"id": "bandit_sast", "workflow": "ci.yml", "kind": "sast", "step": "SAST scan with bandit"},
    {"id": "lock_drift", "workflow": "ci.yml", "kind": "supply-chain", "step": "Dependency lock drift guard"},
    {"id": "gitleaks", "workflow": "ci.yml", "kind": "secret", "step": "Secret scan with gitleaks"},
    {"id": "cargo_check", "workflow": "ci.yml", "kind": "build", "step": "Cargo check"},
    {"id": "cargo_test", "workflow": "ci.yml", "kind": "test", "step": "Cargo test"},
    {"id": "cargo_audit", "workflow": "ci.yml", "kind": "supply-chain", "step": "Cargo dependency vulnerability scan"},
    {"id": "frontend_build", "workflow": "ci.yml", "kind": "build", "step": "Install and build"},
    {"id": "frontend_unit", "workflow": "ci.yml", "kind": "test", "step": "Run vitest unit tests"},
    {"id": "frontend_e2e", "workflow": "ci.yml", "kind": "test", "step": "Run frontend E2E (chromium)"},
    {"id": "quickstart_smoke", "workflow": "audit-regression.yml", "kind": "smoke", "step": "Quickstart smoke test (G38)"},
    {"id": "skill_audit", "workflow": "skill-audit.yml", "kind": "ratchet", "step": "Run skill library audit (gate)"},
    {"id": "ultra_long_bench", "workflow": "stress-test.yml", "kind": "benchmark", "step": "Run ultra-long benchmark (gating)"},
    # 交付/构建阻断门 (desktop.yml / release.yml): 整步删/改名会让交付产物失去可见产出,
    # 且它们是 release.yml 与 desktop.yml 里**未曾在旧登记表出现**的阻断步.
    {"id": "desktop_rust_build", "workflow": "desktop.yml", "kind": "build", "step": "Build Rust CLI + sidecar"},
    {"id": "desktop_frontend_typecheck", "workflow": "desktop.yml", "kind": "build", "step": "Type-check frontend"},
    {"id": "desktop_tauri_build", "workflow": "desktop.yml", "kind": "build", "step": "Build Tauri desktop app"},
    {"id": "release_rust_build", "workflow": "release.yml", "kind": "build", "step": "Build Rust CLI + sidecar"},
    {"id": "release_frontend_typecheck", "workflow": "release.yml", "kind": "build", "step": "Type-check frontend"},
    {"id": "release_tauri_build", "workflow": "release.yml", "kind": "build", "step": "Build Tauri desktop app"},
)


# ── 能力锚点登记 (子系统级: 开关 + 承载模块 + 不变量测试) ──
# 工作流/门/预算登记管的是**治理的执行面** (门是否在跑、棘轮是否被放宽); 它们看不见
# **能力面** —— 一个命名子系统 (如 RSI / PMK / 世界模型) 是靠「一枚开关 + 一组承载模块
# + 一组不变量测试」立起来的. 三者任一被删/改名/默认翻转, 该能力就静默萎缩: 开关还在
# 但默认从开变关 = 能力悄悄关掉; 承载模块被删 = 能力失去实现; 不变量测试被删 = 能力失去
# 回归网. 旧台账对这三点全盲. 这里把每个能力登记为一张**可见决策表**: 任一锚点脱节即红.
#
# `flag_kind`: "feature_flag" (开关在 FeatureFlags._DEFAULTS, 值 bool) 或 "env"
# (开关在 env_schema.SCHEMA 的 HUGINN_*, 值 str). `default` = 期望默认值 (翻转即红).
CAPABILITY_SPECS: tuple[dict, ...] = (
    {
        "id": "world_model",
        "name": "世界模型",
        "flag_kind": "feature_flag",
        "flag": "world_model",
        "default": True,
        "modules": (
            "huginn/capabilities/world_model.py",
            "huginn/security/world_model.py",
            "huginn/research/law_model.py",
        ),
        "tests": ("tests/test_world_model.py", "tests/test_research_planning_team.py"),
    },
    {
        "id": "rsi",
        "name": "递归自我改进 (RSI)",
        "flag_kind": "feature_flag",
        "flag": "harness_meta_improver",
        "default": False,
        "modules": (
            "huginn/harness/meta_improver.py",
            "huginn/self_improvement/core.py",
            "huginn/evolution/engine.py",
        ),
        "tests": ("tests/test_meta_improver.py", "tests/test_evolution_engine.py"),
    },
    {
        "id": "pmk",
        "name": "PMK 一致性 (persona/memory/knowledge)",
        "flag_kind": "env",
        "flag": "HUGINN_PMK_INJECT",
        "default": "0",
        "modules": (
            "huginn/autoloop/cognitive_guard.py",
            "huginn/autoloop/engine_observe.py",
            "huginn/runtime/task_lifecycle.py",
        ),
        "tests": ("tests/test_temporal_pmk.py",),
    },
    # ── 其余命名子系统 (第二轮扩展: 补齐"诸如此类"的能力面) ──
    # 判据: 一枚开关 (feature_flag/env) + 一组承载模块 + 一张不变量回归测试.
    # 只登记**三者齐备**的命名子系统; 无独立回归测试的开关归 SUBSYSTEM_EXEMPT_FLAGS
    # (见下), 不硬凑锚点 —— 诚实边界: 锚点的 test 必须真实存在.
    {
        "id": "asd_ste100",
        "name": "ASD-STE100 受控语言纪律",
        "flag_kind": "feature_flag",
        "flag": "asd_ste100",
        "default": True,
        "modules": ("huginn/plugins/asd_ste100/main.py",),
        "tests": ("tests/test_asd_ste100_plugin.py",),
    },
    {
        "id": "clarification",
        "name": "主动提问 (clarification)",
        "flag_kind": "feature_flag",
        "flag": "clarification",
        "default": True,
        "modules": (
            "huginn/tools/clarification_tool.py",
            "huginn/interaction/clarification.py",
        ),
        "tests": ("tests/test_clarification_tool.py",),
    },
    {
        "id": "compute_policy",
        "name": "计算路由策略 + 预算",
        "flag_kind": "feature_flag",
        "flag": "compute_policy",
        "default": False,
        "modules": (
            "huginn/execution/compute_policy.py",
            "huginn/execution/compute_router.py",
        ),
        "tests": ("tests/test_compute_router_audit.py",),
    },
    {
        "id": "frontier",
        "name": "ARTEX 假设前沿 (frontier)",
        "flag_kind": "feature_flag",
        "flag": "frontier_claim",
        "default": False,
        "modules": (
            "huginn/autoloop/control_mechanisms.py",
            "huginn/autoloop/cognitive_loop.py",
        ),
        "tests": ("tests/test_control_mechanism_governance.py",),
    },
    {
        "id": "jev",
        "name": "JEV 外部判断接入",
        "flag_kind": "feature_flag",
        "flag": "jev_enabled",
        "default": False,
        "modules": (
            "huginn/runtime/jev/_enabled.py",
            "huginn/runtime/jev/client.py",
            "huginn/runtime/jev/tool_router.py",
        ),
        "tests": ("tests/test_jev_wiring.py",),
    },
    {
        "id": "personalization",
        "name": "个性化 (风格/口味)",
        "flag_kind": "feature_flag",
        "flag": "personalization",
        "default": True,
        "modules": (
            "huginn/personalization/user_style.py",
            "huginn/personalization/taste_profile.py",
            "huginn/tools/personalization_tool.py",
        ),
        "tests": ("tests/test_taste_profile.py",),
    },
    {
        "id": "pi_mode",
        "name": "Pi 极简内核模式",
        "flag_kind": "feature_flag",
        "flag": "pi_mode",
        "default": False,
        "modules": (
            "huginn/modes/pi.py",
            "huginn/tools/pi_mode_tool.py",
        ),
        "tests": ("tests/test_make_tool_pi_mode.py",),
    },
    {
        "id": "provenance",
        "name": "计算 provenance 快照",
        "flag_kind": "feature_flag",
        "flag": "provenance",
        "default": True,
        "modules": (
            "huginn/provenance/pipeline.py",
            "huginn/provenance/registry.py",
            "huginn/provenance/correction.py",
        ),
        "tests": ("tests/test_provenance.py", "tests/test_argus_provenance.py"),
    },
    {
        "id": "speculator",
        "name": "投机执行 (speculator)",
        "flag_kind": "feature_flag",
        "flag": "speculator",
        "default": True,
        "modules": ("huginn/agents/speculator.py",),
        "tests": ("tests/test_proactive_injection.py",),
    },
    {
        "id": "system_health",
        "name": "系统资源监控",
        "flag_kind": "feature_flag",
        "flag": "system_health_monitor",
        "default": True,
        "modules": ("huginn/diagnostics/system_health.py",),
        "tests": ("tests/test_system_health.py",),
    },
    {
        "id": "three_cabin",
        "name": "三舱反射模型",
        "flag_kind": "feature_flag",
        "flag": "three_cabin",
        "default": False,
        "modules": ("huginn/metacog/three_cabin_reflector.py",),
        "tests": ("tests/test_verify_4flags.py", "tests/test_metacog_smoke.py"),
    },
    {
        "id": "trace_inspect",
        "name": "VISTA 过程级 trace 检索",
        "flag_kind": "feature_flag",
        "flag": "trace_inspect",
        "default": False,
        "modules": (
            "huginn/autoloop/recall_modes.py",
            "huginn/autoloop/plan_check.py",
        ),
        "tests": ("tests/test_vista_recall_modes.py",),
    },
    {
        "id": "visual_frame_memory",
        "name": "VISTA 无损帧记忆",
        "flag_kind": "feature_flag",
        "flag": "visual_frame_memory",
        "default": False,
        "modules": (
            "huginn/autoloop/frame_store.py",
            "huginn/autoloop/recall_modes.py",
        ),
        "tests": ("tests/test_vista_recall_modes.py",),
    },
    # ── 科学发现能力缺口补齐 (G-A/G-B/G-C): always_on 型锚点 ──
    # 这三项是**工具 action 面 / 机制面**能力, 没有独立开关 —— 硬凑开关 = 噪声.
    # 锚点退化为双判据: 承载模块存在 + 不变量测试存在 (无可翻转的开关, flag_ok 恒 True,
    # 渲染为「常开」); L0 探针 (runtime_probe) 另守依赖面.
    {
        "id": "mol_structure_input",
        "name": "分子结构文件输入 (MOL/SDF/PDB)",
        "flag_kind": "always_on",
        "flag": "—",
        "default": None,
        "modules": ("huginn/tools/sci/rdkit_tool.py",),
        "tests": ("tests/test_rdkit_mol_file.py",),
    },
    {
        "id": "spectra_parse",
        "name": "NMR/质谱谱图解析 (JCAMP-DX/mzML)",
        "flag_kind": "always_on",
        "flag": "—",
        "default": None,
        "modules": (
            "huginn/tools/sci/spectra_parsers.py",
            "huginn/tools/characterization_tool.py",
        ),
        "tests": ("tests/test_spectra_parsers.py",),
    },
    {
        "id": "reproduce_closed_loop",
        "name": "复现重跑闭环 (reproduce)",
        "flag_kind": "always_on",
        "flag": "—",
        "default": None,
        "modules": (
            "huginn/autoloop/reproduce.py",
            "huginn/cli/reproduce.py",
        ),
        "tests": ("tests/test_reproduce.py",),
    },
    # ── 多协议适配层 (缺口①): always_on 型锚点 ──
    # 协议面靠共享 IR 契约而非 feature flag 治理 (无开关可翻转). 锚点双判据:
    # 承载模块 (protocols 包存在) + 不变量回归网 (红蓝双轨契约门存在).
    {
        "id": "protocol_adapter",
        "name": "多协议适配层 (共享 IR + 每协议成对)",
        "flag_kind": "always_on",
        "flag": "—",
        "default": None,
        "modules": ("huginn/protocols/__init__.py",),
        "tests": ("tests/test_protocol_contract.py",),
    },
)

# ── 能力面闭合分诊表 (feature flag 全部登记: 锚定 or 豁免) ──
# 光靠 `CAPABILITY_SPECS` 逐条加锚点仍是**点修**: 新加一枚功能开关时, 若无人把它
# 登记为锚点或豁免, 能力面又会出现静默缺口. 本表把「feature flag 全集」收成**闭合面**:
# `FeatureFlags._DEFAULTS` 的每个键必须**恰好**落在「CAPABILITY_SPECS 锚定」或本豁免集
# 之一 —— 二者并集 = 全集且不相交 (`tests/test_arch_gate_registry.py` 的
# `test_capability_flags_all_triaged` 硬断言). 新增开关未分诊 → 红 (可见决策).
#
# 豁免语义: 该开关是**调参/实验栅栏/格式开关/隐私档位/微特性**, 不是一个"靠开关+承载模块+
# 回归测试"立起来的命名子系统 (无独立不变量回归测试); 或虽是子系统但当前**尚无**专属回归
# 测试 (诚实登记, 待补测试后升级为锚点, 从本表移出). 分组仅注释, 集合本身平铺.
SUBSYSTEM_EXEMPT_FLAGS: frozenset[str] = frozenset(
    {
        # 微路由 / 单点 toggle (有模块, 无专属回归测试 —— 待补测试可升级为锚点)
        "tool_call_router",       # agents/tool_call_router.py 重型工具 sanity check
        "loop_detector",          # agents/loop_detector.py 对话循环检测
        "context_router",         # runtime/context_router.py 信息路径稀疏化
        "task_tool_router",       # runtime/task_tool_router.py task→工具子集路由
        "bandit_mdp",             # autoloop/bandit.py + agent/bandit_controller.py
        # 实验栅栏 (harness 家族, 默认关; 由 harness_evolution_spec 承载)
        "harness_workflow_evolution",
        "harness_ood_holdout",
        "harness_significance_gate",
        "harness_adoption_gate",
        "harness_joint_optimizer",
        "harness_phase_evolve",
        "harness_prompt_patch",
        # 奖励口径开关 (validation/scope_authority.py, 默认关)
        "anti_hacking_reward",
        "intent_scope_reward",
        # 沙箱硬化 (security/sandbox.py) —— 安全基线 toggle, 非能力子系统
        "sandbox_net_isolation",
        # 格式 / 日志 / 遥测
        "json_logs",
        "prompt_cache_control",
        "telemetry",
        # 认知微特性 (单点 toggle)
        "belief_update",
        "belief_darwin",
        "belief_mode_switch",
        "memory_typing",
        "crdt_merge",
        "crdt_branch_merge",
        "ising_rerank",
        "ising_frontier",
        "hils_attention",
        "fts_auto_rebuild",
        "curiosity_hint",
        "hypothesis_llm_semantic",
        "comms_contract",
        "extreme_dispatch",
        "use_cognitive_map",
        "use_atomworld",
        "persistent_terminal",
        "system_health_auto_fix",   # system_health 的熔断子档 (子系统见锚点 system_health)
        # 隐私档位 (PrivacyGuard 互斥三档 + 细粒度开关)
        "privacy_off",
        "privacy_redact",
        "privacy_local_only",
        "privacy_redact_secrets",
        "privacy_block_on_secrets",
        # 命名子系统的**子开关** (总闸已锚定; 子开关无独立回归测试)
        "jev_tool_router",
        "jev_guardrail",
        "frontier_single_writer_strict",
        "frontier_empty_stop",
        # 外部草稿纸 (deep_think; 有 doc + flag, 无专属回归测试)
        "external_thinking",
    }
)


def _feature_flag_defaults(agent_root: Path) -> dict[str, object]:
    """从 `FeatureFlags._DEFAULTS` 读 ``flag -> 默认值`` (ast, 不 import 模块).

    weightless: 只 parse 源文件, 不拉 huginn 包 (后者拖重依赖). 找不到/解析失败返回空表.
    """
    path = agent_root / "huginn" / "feature_flags.py"
    if not path.is_file():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        target: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        if target != "_DEFAULTS" or not isinstance(value, ast.Dict):
            continue
        out: dict[str, object] = {}
        for k, v in zip(value.keys, value.values):
            try:
                out[ast.literal_eval(k)] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                continue
        return out
    return {}


def _env_schema_defaults(agent_root: Path) -> dict[str, str]:
    """从 `env_schema.SCHEMA` 读 ``HUGINN_* -> 默认值`` (ast, 不 import 模块)."""
    path = agent_root / "huginn" / "env_schema.py"
    if not path.is_file():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        target: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        if target != "SCHEMA" or not isinstance(value, ast.Dict):
            continue
        out: dict[str, str] = {}
        for k, v in zip(value.keys, value.values):
            try:
                name = ast.literal_eval(k)
            except (ValueError, SyntaxError):
                continue
            if isinstance(v, ast.Call) and len(v.args) >= 2:
                try:
                    out[name] = str(ast.literal_eval(v.args[1]))
                except (ValueError, SyntaxError):
                    continue
        return out
    return {}


def _collect_capabilities(agent_root: Path) -> list[dict]:
    """收集能力锚点, 判 ①开关默认值未翻转; ②每个承载模块存在; ③每个不变量测试存在.

    三者任一脱节 = 该能力静默萎缩. 开关从 `FeatureFlags._DEFAULTS` / `env_schema.SCHEMA`
    读实际默认值 (ast, weightless); `flag_ok=False` 即默认翻转或开关被删.
    `flag_kind="always_on"` 型 (工具 action 面/机制面能力, 无开关) 跳过 ①:
    硬凑开关 = 噪声, 锚点退化为双判据 (模块/测试存在), flag_ok 恒 True.
    """
    flags = _feature_flag_defaults(agent_root)
    envs = _env_schema_defaults(agent_root)
    out: list[dict] = []
    for spec in CAPABILITY_SPECS:
        if spec["flag_kind"] == "always_on":
            actual = None
            expected = None
            flag_ok = True
        elif spec["flag_kind"] == "feature_flag":
            actual = flags.get(spec["flag"], "__MISSING__")
            expected = spec["default"]
            flag_ok = actual == expected
        else:
            actual = envs.get(spec["flag"], "__MISSING__")
            expected = str(spec["default"])
            flag_ok = actual == expected
        missing_modules = [m for m in spec["modules"] if not (agent_root / m).exists()]
        missing_tests = [t for t in spec["tests"] if not (agent_root / t).exists()]
        out.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "flag": spec["flag"],
                "flag_kind": spec["flag_kind"],
                "expected_default": _render_default(expected),
                "actual_default": _render_default(actual),
                "flag_ok": flag_ok,
                "modules": list(spec["modules"]),
                "tests": list(spec["tests"]),
                "missing_modules": missing_modules,
                "missing_tests": missing_tests,
            }
        )
    out.sort(key=lambda c: c["id"])
    return out


def _render_default(value: object) -> str:
    """能力开关默认值的可读渲染 (bool 小写, 缺失显形)."""
    if value == "__MISSING__":
        return "缺失"
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "—"
    return str(value)


def _collect_cli_guards(agent_root: Path, workflows: dict[str, str]) -> list[dict]:
    """跨全部工作流收集 CLI 门 (`python -m huginn.<module>`), 按模块去重, 判模块存在.

    去重 = 同一模块在多处/多工作流调用只登记一次 (与旧实现一致); module 映射到仓内
    `.py` 或包目录判存在.
    """
    out: list[dict] = []
    seen: set[str] = set()
    for wf, text in workflows.items():
        for name, module in _parse_cli_guards(text):
            if module in seen:
                continue
            seen.add(module)
            rel = module.replace(".", "/")
            exists = (agent_root / f"{rel}.py").exists() or (agent_root / rel).is_dir()
            out.append({"module": module, "ci_step": name, "workflow": wf, "exists": exists})
    out.sort(key=lambda g: g["module"])
    return out


def _collect_precommit_gates(agent_root: Path, precommit_texts: dict[str, str]) -> list[dict]:
    """收集两份 `.pre-commit-config.yaml` 的本地门, 判引用测试存在 + `files:` 是否覆盖.

    按 (配置来源, hook_id, test) 去重 (同名文件的多个 selector 算一门). `files:` 过滤器
    未覆盖 entry 引用的测试时, 改该测试不触发 hook —— 门静默失效 (比悬空更隐蔽).
    """
    out: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for src, text in precommit_texts.items():
        files_by_hook = _parse_precommit_files(text)
        for hook, test in _parse_precommit_hooks(text):
            if (src, hook, test) in seen:
                continue
            seen.add((src, hook, test))
            gate = {
                "hook": hook,
                "test": test,
                "source": src,
                "exists": (agent_root / test).exists(),
            }
            regex = files_by_hook.get(hook)
            if not regex:
                gate["covered"] = True  # 无 files: 则对全部路径生效
            else:
                try:
                    pat = re.compile(regex)
                    # files: 匹配 **仓根相对** 路径; entry 的测试路径可能相对仓根或 agent/
                    # (取决于 entry 是否 `cd agent`), 两种都认.
                    gate["covered"] = bool(
                        pat.search(test) or pat.search(f"agent/{test}")
                    )
                except re.error:
                    gate["covered"] = False
            out.append(gate)
    out.sort(key=lambda g: (g["source"], g["hook"], g["test"]))
    return out


def _collect_engineer_gates(workflows: dict[str, str]) -> list[dict]:
    """收集工程门 (非 pytest 阻断门), 判其 step 名是否仍在对应工作流里."""
    step_names = {wf: set(_parse_step_names(text)) for wf, text in workflows.items()}
    out: list[dict] = []
    for spec in ENGINEER_GATE_SPECS:
        out.append(
            {
                "id": spec["id"],
                "workflow": spec["workflow"],
                "kind": spec["kind"],
                "step": spec["step"],
                "exists": spec["step"] in step_names.get(spec["workflow"], set()),
            }
        )
    out.sort(key=lambda g: g["id"])
    return out


# ── prompt 段生效面 (声明面清单 vs 源文件 `_APPLIES_TO`) ──
def _read_prompt_applies_to(agent_root: Path) -> dict[str, tuple[str, ...]]:
    """ast 读 prompt_segments 的 `_APPLIES_TO` 字面量 (weightless, 不 import 重栈).

    这是 prompt 段**声明面**的单一事实来源 (段名 -> 必须生效的生产路径). 读源文件而非
    import, 以免门禁拉 langchain/pydantic 等重栈, 也无镜像漂移.
    """
    path = agent_root / _PROMPT_SEGMENTS_MODULE
    if not path.is_file():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        logger.debug("prompt_segments 解析失败: %s", path, exc_info=True)
        return {}
    for node in ast.walk(tree):
        target: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        if target != "_APPLIES_TO" or value is None:
            continue
        try:
            data = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): tuple(v) for k, v in data.items()}
    return {}


def _collect_prompt_segments(agent_root: Path) -> dict:
    """prompt 段生效面: 台账声明清单 (PROMPT_SPECS) vs 源文件 `_APPLIES_TO` 对齐.

    **清单漂移**单列: 清单有、源里无 (段被删) / 源里有、清单无 (新增未登记); 逐段的
    生效路径**不在此复制**, 直接 ast 读源文件, 无镜像漂移. 「声明 == 渲染」由
    `tests/test_prompt_contract.py` 硬判定.
    """
    source = _read_prompt_applies_to(agent_root)
    declared = list(PROMPT_SPECS)
    declared_set = set(declared)
    return {
        "declared": declared,
        "source": sorted(source),
        "segments": [{"name": n, "paths": list(source.get(n, ()))} for n in declared],
        "missing_in_source": [n for n in declared if n not in source],
        "unregistered": sorted(n for n in source if n not in declared_set),
    }


# ── 预算登记 (单一机器可读来源) ──
# 治理棘轮的冻结预算/基线此前散落在 tests/、huginn/、tests/golden/ 各处. 这里收成
# **一张声明表**: 每项指明位置、抽取方式与消费者门, 供 governance-state.md 单点可读.
# 抽取**只读、weightless**(regex + ast, 不 import 被测模块), 以免拖慢快速失败门.
#
# phase 2 (M1′, 未落): 把测试侧冻结清单逐项提升为**权威读取**(门 from 本表 import),
# 使预算编辑收敛到单一入口; env_schema.RAW_READ_BASELINE 因层级原因保留原处、此处引用.
BUDGET_SPECS: tuple[dict, ...] = (
    {
        "id": "mece_findings",
        "name": "MECE 发现基线",
        "path": "tests/golden/mece_findings_baseline.txt",
        "kind": "file-lines",
        "owner_gate": "MECE wiring-contract guard (ratchet)",
        "owner_kind": "ci_step",
    },
    {
        "id": "mirror_drift",
        "name": "镜像漂移预算",
        "path": "tests/test_publish_mirror_drift.py",
        "kind": "const-triple",
        "keys": ("MAX_MISSING", "MAX_EXTRA", "MAX_DIFFERING"),
        "source": "inline",
        "value": (0, 0, 0),
        "owner_gate": "publish mirror drift (only-shrink)",
        "owner_kind": "full_suite",
        "probe": "mirror_drift_live",
        "heavy": True,
    },
    {
        "id": "env_raw_read",
        "name": "环境裸读基线",
        "path": "huginn/env_schema.py",
        "kind": "const",
        "key": "RAW_READ_BASELINE",
        "owner_gate": "env surface governance",
        "owner_kind": "full_suite",
        "probe": "env_raw_read_live",
        "heavy": True,
        "direction": "cap",
    },
    {
        "id": "gateway_imports",
        "name": "外部直连白名单",
        "path": "tests/test_arch_single_gateway.py",
        "kind": "collection-len",
        "var": "ALLOWED_EXTERNAL_IMPORTS",
        "owner_gate": "Architecture single-gateway guard",
        "owner_kind": "ci_step",
    },
    {
        "id": "gateway_subcommands",
        "name": "冻结委托子命令",
        "path": "tests/test_arch_single_gateway.py",
        "kind": "collection-len",
        "var": "CLI_DELEGATED_SUBCOMMANDS",
        "owner_gate": "Architecture single-gateway guard",
        "owner_kind": "ci_step",
    },
    {
        "id": "sanctioned_plugins",
        "name": "插件准入清单",
        "path": "tests/test_arch_plugin_governance.py",
        "kind": "collection-len",
        "var": "SANCTIONED_PLUGINS",
        "owner_gate": "Plugin admission guard",
        "owner_kind": "ci_step",
    },
    {
        "id": "hardcoded_allow",
        "name": "硬编码路径白名单",
        "path": "tests/test_arch_no_hardcoded_paths.py",
        "kind": "collection-len",
        "var": "ALLOWED_HARDCODED_MATCHES",
        "owner_gate": "No-hardcoded-machine-paths guard",
        "owner_kind": "ci_step",
    },
    {
        "id": "mcp_wiring_allow",
        "name": "MCP 接线白名单",
        "path": "tests/test_arch_mcp_governance.py",
        "kind": "collection-len",
        "var": "_ALLOWED_WIRING_FILES",
        "owner_gate": "MCP single-gateway guard",
        "owner_kind": "ci_step",
    },
    {
        "id": "arch_gate_floor",
        "name": "arch 门数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "ARCH_GATE_FLOOR",
        "source": "inline",
        "value": 17,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "arch_gate_count",
        "direction": "floor",
    },
    {
        "id": "cli_guard_floor",
        "name": "CLI 门数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "CLI_GUARD_FLOOR",
        "source": "inline",
        "value": 2,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "cli_guard_count",
        "direction": "floor",
    },
    {
        "id": "precommit_gate_floor",
        "name": "本地门数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "PRECOMMIT_GATE_FLOOR",
        "source": "inline",
        "value": 5,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "precommit_gate_count",
        "direction": "floor",
    },
    {
        "id": "engineer_gate_floor",
        "name": "工程门数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "ENGINEER_GATE_FLOOR",
        "source": "inline",
        "value": 21,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "engineer_gate_count",
        "direction": "floor",
    },
    {
        "id": "workflow_floor",
        "name": "工作流数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "WORKFLOW_FLOOR",
        "source": "inline",
        "value": 6,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "workflow_count",
        "direction": "floor",
    },
    {
        "id": "guard_step_floor",
        "name": "快速失败门数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "GUARD_STEP_FLOOR",
        "source": "inline",
        "value": 19,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "guard_step_count",
        "direction": "floor",
    },
    {
        "id": "field_audit_golden",
        "name": "字段审计 golden",
        "path": "tests/golden/field_audit",
        "kind": "dir-file-count",
        "owner_gate": "MECE 字段级面",
        "owner_kind": "full_suite",
    },
    {
        "id": "skill_audit_baseline",
        "name": "技能库审计基线",
        "path": "scripts/skill_audit_baseline.json",
        "root": "repo",
        "kind": "json-scalars",
        "keys": ("skills", "tools"),
        "owner_gate": "Run skill library audit (gate)",
        "owner_kind": "workflow_step",
    },
    # ── 阻断步静默增删 (反向兜底) ──
    # 全部工作流**命名 step 总数**下限: ENGINEER_GATE_SPECS 只挡「已登记→被删」, 挡不住
    # 未登记的步骤被删/新增. 总数只减不增 —— 任何步骤增删都会动它, 须显式改台账 (可见决策).
    {
        "id": "workflow_step_floor",
        "name": "工作流命名 step 总数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "WORKFLOW_STEP_FLOOR",
        "source": "inline",
        "value": 122,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "workflow_step_total",
        "direction": "floor",
    },
    # ── 能力面 (子系统锚点数下限) ──
    # 命名子系统 (RSI / PMK / 世界模型 …) 的锚点数下限: 删掉一个 `CAPABILITY_SPECS` 条目
    # = 该能力脱离治理面, 须显式下调 (可见决策). 与门数棘轮同构.
    {
        "id": "capability_floor",
        "name": "能力锚点数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "source": "inline",
        "value": 19,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "capability_count",
        "direction": "floor",
    },
    # ── prompt 段生效面 (声明清单只减不增) ──
    # prompt 段声明清单 (PROMPT_SPECS: 段名 -> 必须生效的生产路径) 的下限: 删掉一个段 =
    # 输出规范的一段静默消失, 须显式下调 (可见决策). 与能力锚点棘轮同构.
    {
        "id": "prompt_segment_floor",
        "name": "prompt 段数下限",
        "path": "tests/test_arch_gate_registry.py",
        "kind": "const",
        "key": "PROMPT_SEGMENT_FLOOR",
        "source": "inline",
        "value": 13,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "prompt_segment_count",
        "direction": "floor",
    },
    # ── 豁免/白名单面 (放宽即静默削弱对应门) ──
    # 值取**当前条数** (上限): 往豁免名单里加一条 = 扩大免检面, 必须显式抬高台账 value
    # (可见决策). 探针 weightless (tomllib / regex), 不 import 被测模块.
    {
        "id": "gitleaks_allowlist",
        "name": "gitleaks 密钥豁免数",
        "path": ".gitleaks.toml",
        "root": "repo",
        "source": "ceiling",
        "value": 4,
        "probe": "toml_list_len",
        "key_path": ("allowlist", "regexes"),
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "direction": "cap",
    },
    {
        "id": "pip_audit_ignore",
        "name": "pip-audit 漏洞豁免数",
        "path": ".github/workflows/ci.yml",
        "root": "repo",
        "source": "ceiling",
        "value": 24,
        "probe": "regex_match_count",
        # 只数真正的 `--ignore-vuln <ID>` 标志 (ID 首字符为大写/数字), 排除注释里的同名字样.
        "pattern": r"--ignore-vuln\s+[A-Z]",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "direction": "cap",
    },
    {
        "id": "dependabot_ignore",
        "name": "dependabot 忽略项数",
        "path": ".github/dependabot.yml",
        "root": "repo",
        "source": "ceiling",
        "value": 5,
        "probe": "regex_match_count",
        "pattern": r"- dependency-name:",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "direction": "cap",
    },
    # ── 配置级门 ──
    # 覆盖率下限 = **下限 (floor)**: 下调即静默削弱本地覆盖率门, 须显式改台账 value.
    {
        "id": "coverage_fail_under",
        "name": "覆盖率下限",
        "path": "pyproject.toml",
        "source": "floor",
        "value": 60,
        "probe": "regex_int_live",
        "pattern": r"^fail_under\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "direction": "floor",
    },
    # ── 循环闸位 (loop caps): 自主轮数/步数上限 ──
    # 这些值此前散在 env_schema 与各引擎/路由的硬编码默认里, 无单一上限可读. 记在此处作
    # **上限(ceiling)**: 代码持有实测值, 台账持有允许的最大值; 棘轮 = 实测 ≤ 上限, 提高上限
    # (扩大自主面) 必须显式改台账 value (可见决策). 实测由 weightless regex 探针从源码读出.
    {
        "id": "loop_autoloop_cli",
        "name": "autoloop CLI 迭代上限",
        "path": "huginn/cli/commands/autoloop.py",
        "group": "loop",
        "source": "ceiling",
        "value": 20,
        "pattern": r'"--iterations",\s*"-i",\s*default=(\d+)',
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_autoloop_engine",
        "name": "autoloop 引擎迭代上限",
        "path": "huginn/autoloop/cognitive_loop.py",
        "group": "loop",
        "source": "ceiling",
        "value": 50,
        "pattern": r"def run_cognitive\([\s\S]*?max_iterations:\s*int\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_recursion_limit",
        "name": "长程 recursion_limit 上限",
        "path": "huginn/agent/core.py",
        "group": "loop",
        "source": "ceiling",
        "value": 500,
        "pattern": r"_LONG_HORIZON_MODES:\s*return (\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_tool_calls",
        "name": "单轮 tool-call 上限",
        "path": "huginn/env_schema.py",
        "group": "loop",
        "source": "ceiling",
        "value": 15,
        "pattern": r'"HUGINN_MAX_TOOL_CALLS":\s*EnvSpec\("int",\s*"(\d+)"',
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_tool_calls_per_tool",
        "name": "单轮单工具调用上限",
        "path": "huginn/env_schema.py",
        "group": "loop",
        "source": "ceiling",
        "value": 5,
        "pattern": r'"HUGINN_MAX_TOOL_CALLS_PER_TOOL":\s*EnvSpec\("int",\s*"(\d+)"',
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_codeact_turns",
        "name": "CodeAct 轮数上限",
        "path": "huginn/agent/code_act_loop.py",
        "group": "loop",
        "source": "ceiling",
        "value": 15,
        "pattern": r"^_MAX_TURNS\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_coder_iter",
        "name": "coder 路由迭代上限",
        "path": "huginn/env_schema.py",
        "group": "loop",
        "source": "ceiling",
        "value": 50,
        "pattern": r'"HUGINN_CODER_MAX_ITER":\s*EnvSpec\("int",\s*"(\d+)"',
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_exploration",
        "name": "探索编排迭代上限",
        "path": "huginn/exploration/orchestrator.py",
        "group": "loop",
        "source": "ceiling",
        "value": 40,
        "pattern": r"async def explore\([\s\S]*?max_iterations:\s*int\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_deep_research",
        "name": "deep_research 迭代上限",
        "path": "huginn/routes/deep_research.py",
        "group": "loop",
        "source": "ceiling",
        "value": 12,
        "pattern": r'opts\.setdefault\("max_iterations",\s*(\d+)\)',
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_rcb_step2_iters",
        "name": "RCB Step-2 执行迭代上限",
        "path": "huginn/cli/rcb_step2.py",
        "group": "loop",
        "source": "ceiling",
        "value": 20,
        "pattern": r'"HUGINN_RCB_EXEC_ITERS",\s*"(\d+)"',
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_harness_phase_iter",
        "name": "harness phase 迭代上限",
        "path": "huginn/harness/phase_spec.py",
        "group": "loop",
        "source": "ceiling",
        "value": 5,
        "pattern": r"max_iterations:\s*int\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_skill_step",
        "name": "技能 step 循环上限",
        "path": "huginn/skills/base.py",
        "group": "loop",
        "source": "ceiling",
        "value": 20,
        "pattern": r"loop_max_iterations:\s*int\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_skill_preset",
        "name": "技能优化迭代上限",
        "path": "huginn/skills/presets.py",
        "group": "loop",
        "source": "ceiling",
        "value": 10,
        "pattern": r'"max_iterations",\s*"int",\s*"[^"]*",\s*default=(\d+)',
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_codeact_sandbox_turns",
        "name": "CodeAct sandbox 轮数上限",
        "path": "huginn/security/code_act_sandbox.py",
        "group": "loop",
        "source": "ceiling",
        "value": 15,
        "pattern": r"^_MAX_TURNS\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_ws_auto_continue",
        "name": "WS 每 turn 自动续跑上限",
        "path": "huginn/routes/ws_helpers.py",
        "group": "loop",
        "source": "ceiling",
        "value": 3,
        "pattern": r"_MAX_AUTO_CONTINUES\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_agent_profile_steps",
        "name": "per-agent 步数上限",
        "path": "huginn/config.py",
        "group": "loop",
        "source": "ceiling",
        "value": 10,
        "pattern": r"max_steps:\s*int\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_causal_runner_tool_calls",
        "name": "causal_runner tool-call 上限",
        "path": "huginn/cli/causal_runner.py",
        "group": "loop",
        "source": "ceiling",
        "value": 40,
        "pattern": r"max_tool_calls:\s*int\s*=\s*(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_deli_research",
        "name": "deli_research 迭代上限",
        "path": "huginn/academic/deli_research.py",
        "group": "loop",
        "source": "ceiling",
        "value": 5,
        "pattern": r"max_iterations=(\d+)",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "regex_int_live",
        "direction": "cap",
    },
    {
        "id": "loop_cap_floor",
        "name": "循环闸位数下限",
        "path": "tests/test_arch_gate_registry.py",
        "group": "loop",
        "kind": "const",
        "source": "inline",
        "value": 18,
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "loop_cap_live_count",
        "direction": "floor",
    },
    {
        "id": "runtime_probe_floor",
        "name": "可运行性探针清单下限",
        "path": "huginn/capabilities/smoke_specs.py",
        "group": "runtime",
        "kind": "collection-len",
        "source": "inline",
        "value": 7,
        "var": "SMOKE_SPEC_NAMES",
        "owner_gate": "Governance gate registry guard",
        "owner_kind": "ci_step",
        "probe": "runtime_probe_live_count",
        "direction": "floor",
    },
)

BUDGET_BY_ID: dict[str, dict] = {s["id"]: s for s in BUDGET_SPECS}


def budget_value(budget_id: str) -> int | tuple[int, ...]:
    """取**权威**(inline)预算的值, 供消费者门 import, 使编辑收敛到单一入口.

    非 inline 项 (值在原位 / 列表内容即契约) 不提供读数并抛 KeyError —— 避免把值
    搬进台账而丢失其自然归属 (如 env_schema 的层级、插件清单的内容).
    """
    spec = BUDGET_BY_ID[budget_id]
    if spec.get("source") != "inline":
        raise KeyError(
            f"预算 `{budget_id}` 非 inline 权威项; 其值在原位 (BUDGET_SPECS.path={spec.get('path')})"
        )
    return spec["value"]


def _extract_int_const(text: str, key: str) -> int | None:
    """抽取形如 `KEY = 123` 的整数常量 (行首锚定)."""
    m = re.search(rf"^{re.escape(key)}\s*=\s*(\d+)", text, re.M)
    return int(m.group(1)) if m else None


def _extract_collection_len(module_text: str, var: str) -> int | None:
    """用 ast 读 `var = <dict/set/frozenset 字面量>` 的元素数 (不 import 模块)."""
    try:
        tree = ast.parse(module_text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        target_name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target_name, value = node.targets[0].id, node.value
        if target_name != var or value is None:
            continue
        container = value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"frozenset", "set", "list", "tuple", "dict"}
            and value.args
        ):
            container = value.args[0]
        try:
            return len(ast.literal_eval(container))
        except (ValueError, SyntaxError):
            return None
    return None


def _frozen_value(spec: dict, agent_root: Path) -> str:
    """取某预算项的冻结值 (weightless): 台账声明项 (inline/ceiling) 读声明, 其余从原位抽取."""
    if spec.get("source") in ("inline", "ceiling", "floor"):
        value = spec["value"]
        if isinstance(value, tuple):
            return "/".join(str(v) for v in value)
        return str(value)
    # root=repo 的项 (如技能库审计基线) 相对**仓根**而非 agent/; 其余相对 agent/.
    base = agent_root.parent if spec.get("root") == "repo" else agent_root
    path = base / spec["path"]
    kind = spec["kind"]
    if kind == "dir-file-count":
        if not path.is_dir():
            return "?"
        return f"{sum(1 for p in path.iterdir() if p.is_file())} 文件"
    if not path.is_file():
        return "?"
    text = path.read_text(encoding="utf-8")
    if kind == "json-scalars":
        import json

        try:
            data = json.loads(text)
        except ValueError:
            return "?"
        vals = [data.get(k) for k in spec["keys"]]
        return "/".join("?" if v is None else str(v) for v in vals)
    if kind == "file-lines":
        n = sum(1 for ln in text.splitlines() if ln.strip())
        return f"{n} 项"
    if kind == "const":
        v = _extract_int_const(text, spec["key"])
        return str(v) if v is not None else "?"
    if kind == "const-triple":
        vals = [_extract_int_const(text, k) for k in spec["keys"]]
        return "/".join("?" if v is None else str(v) for v in vals)
    if kind == "collection-len":
        n = _extract_collection_len(text, spec["var"])
        return f"{n} 项" if n is not None else "?"
    return "?"


def _probe_current(
    spec: dict, agent_root: Path, heavy: bool = False
) -> int | tuple[int, ...] | None:
    """实测当前值.

    默认 (``heavy=False``) 只跑 **weightless** 探针, 保持快速失败门的轻量; ``heavy=True``
    才跑需遍历全仓的重型探针 (如镜像漂移). 其余项由消费者门自行校验, 此处不重复扫描.
    """
    probe = spec.get("probe")
    if probe == "arch_gate_count":
        return len(discover_arch_tests(agent_root))
    if probe in ("cli_guard_count", "engineer_gate_count", "workflow_count"):
        workflows = _read_workflow_texts(agent_root)
        if probe == "workflow_count":
            return len(workflows)
        if probe == "cli_guard_count":
            return len(_collect_cli_guards(agent_root, workflows))
        return len(_collect_engineer_gates(workflows))
    if probe == "guard_step_count":
        # 全部工作流里的**去重 pytest guard step 数** (arch 与非 arch 一视同仁). 与 audit()
        # 的 fast-fail gate 集一致; 不调 audit() 以免 collect_budgets 递归.
        tests: set[str] = set()
        for text in _read_workflow_texts(agent_root).values():
            tests |= {t for _, t in _parse_guard_steps(text)}
        return len(tests)
    if probe == "precommit_gate_count":
        return len(_collect_precommit_gates(agent_root, _read_precommit_texts(agent_root)))
    if probe == "capability_count":
        return len(_collect_capabilities(agent_root))
    if probe == "prompt_segment_count":  # prompt 段声明清单条目数 (floor 棘轮)
        return len(_read_prompt_applies_to(agent_root))
    if probe == "regex_int_live":  # 循环闸位/覆盖率下限: 从源码按 spec["pattern"] 抽活值 (第 1 捕获组)
        path = agent_root / spec["path"]
        if not path.is_file():
            return None
        m = re.search(spec["pattern"], path.read_text(encoding="utf-8"), re.M)
        return int(m.group(1)) if m else None
    if probe == "workflow_step_total":  # 全部工作流命名 step 总数 (任意步骤增删都动它)
        return sum(
            len(_parse_step_names(t)) for t in _read_workflow_texts(agent_root).values()
        )
    if probe in ("toml_list_len", "regex_match_count"):
        base = agent_root.parent if spec.get("root") == "repo" else agent_root
        path = base / spec["path"]
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        if probe == "regex_match_count":
            return len(re.findall(spec["pattern"], text))
        try:  # toml_list_len: 按 spec["key_path"] 逐层下钻取列表长度 (如 .gitleaks.toml allowlist.regexes)
            import tomllib

            node: object = tomllib.loads(text)
        except Exception:  # noqa: BLE001 — 解析失败不拖垮台账
            logger.debug("toml 解析 %s 失败", path, exc_info=True)
            return None
        for key in spec["key_path"]:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return len(node) if isinstance(node, list) else None
    if probe == "loop_cap_live_count":  # 循环闸位数下限: 可读到的 loop ceiling 项数 (棘轮)
        return sum(
            1
            for s in BUDGET_SPECS
            if s.get("group") == "loop"
            and s.get("source") == "ceiling"
            and isinstance(_probe_current(s, agent_root, heavy=heavy), int)
        )
    if probe == "runtime_probe_live_count":  # 可运行性探针清单条目数 (floor 棘轮)
        path = agent_root / spec["path"]
        if not path.is_file():
            return None
        return _extract_collection_len(path.read_text(encoding="utf-8"), spec["var"])
    if probe == "mirror_drift_live" and heavy:
        return _probe_mirror_drift(agent_root)
    if probe == "env_raw_read_live" and heavy:
        return _probe_env_raw_read(agent_root)
    return None


def _load_module_by_path(name: str, path: Path):
    """按文件路径加载模块并登记进 ``sys.modules`` (兼容 @dataclass). 失败返回 None.

    ``sys.modules`` 登记是必需的: ``@dataclass`` 处理时会 ``sys.modules.get(cls.__module__)``,
    未登记则 ``None.__dict__`` → 崩 (曾让镜像探针静默失效).
    """
    try:
        import importlib.util
        import sys

        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:  # noqa: BLE001 — 加载失败不拖垮台账
        logger.debug("按路径加载 %s 失败", path, exc_info=True)
        return None


def _probe_mirror_drift(agent_root: Path) -> tuple[int, int, int] | None:
    """实测镜像漂移 (missing/extra/differing 合计).

    复用 ``scripts/mcpb_vendor`` 的既有口径 (非重复实现), 失败返回 None.
    重型: 遍历两套镜像 + 源树, 故只在 ``--probe`` 下跑.
    """
    mod = _load_module_by_path(
        "mcpb_vendor_probe", agent_root.parent / "scripts" / "mcpb_vendor.py"
    )
    if mod is None:
        return None
    try:
        missing = extra = differing = 0
        for bundle in mod.BUNDLES:
            rep = mod.drift_report(bundle)
            missing += len(rep.missing)
            extra += len(rep.extra)
            differing += len(rep.differing)
        return (missing, extra, differing)
    except Exception:  # noqa: BLE001 — 探针失败不影响台账
        logger.debug("mirror drift 探针失败", exc_info=True)
        return None


def _probe_env_raw_read(agent_root: Path) -> int | None:
    """实测裸 ``HUGINN_*`` 读点数 (不含经 env_access 的规范读点).

    复用 ``config_audit.build_inventory`` 的既有口径 (其模块级仅 stdlib, 可按文件路径轻量
    加载), 与 ``test_env_surface_governance`` 的棘轮**同源** —— 不重复实现, 无口径分叉.
    重型: 全仓 regex 扫 (约 1s), 故只在 ``--probe`` 下跑.
    """
    mod = _load_module_by_path(
        "config_audit_probe", agent_root / "huginn" / "cli" / "config_audit.py"
    )
    if mod is None:
        return None
    try:
        inv = mod.build_inventory(agent_root / "huginn")
        return sum(
            1 for item in inv.values() for r in item["reads"] if not r.get("canonical")
        )
    except Exception:  # noqa: BLE001 — 探针失败不影响台账
        logger.debug("env 裸读探针失败", exc_info=True)
        return None


def collect_budgets(agent_root: Path, heavy: bool = False) -> list[dict]:
    """治理棘轮预算的单一台账 (只读): 冻结值 + 可测项的实测值/余量.

    ``heavy=True`` 才跑重型探针 (遍历全仓); 默认只跑 weightless 探针, 保证门禁轻量.
    """
    out: list[dict] = []
    for spec in BUDGET_SPECS:
        frozen = _frozen_value(spec, agent_root)
        current = _probe_current(spec, agent_root, heavy=heavy)
        headroom: int | None = None
        if isinstance(current, int):
            m = re.match(r"^(\d+)", frozen)
            if m:
                frozen_n = int(m.group(1))
                # floor 型: 实测应 ≥ 冻结下限, 余量 = 实测 - 下限; cap 型: 余量 = 上限 - 实测.
                headroom = (
                    current - frozen_n
                    if spec.get("direction") == "floor"
                    else frozen_n - current
                )
        out.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "path": spec["path"],
                "mode": {"inline": "权威", "ceiling": "上限", "floor": "下限"}.get(
                    spec.get("source"), "引用"
                ),
                "group": spec.get("group", "budget"),
                "frozen": frozen,
                "current": current,
                "headroom": headroom,
                "owner_gate": spec["owner_gate"],
                "owner_kind": spec["owner_kind"],
            }
        )
    return out


def audit(
    agent_root: Path,
    ci_text: str | None = None,
    precommit_text: str | None = None,
    heavy: bool = False,
) -> dict:
    """扫描一棵仓树, 返回治理门登记快照.

    ``ci_text`` / ``precommit_text`` 显式给定时作为**单文件**合成输入, 不读盘 (供合成树
    测试); 为 None 时遍历 ``<root>/../.github/workflows/*.yml`` **全部**工作流与仓根 +
    ``<root>/`` 两份 ``.pre-commit-config.yaml`` (真实仓: agent_root=/repo/agent).
    ``heavy=True`` 才在预算台账里跑重型探针 (见 ``collect_budgets``).
    """
    if ci_text is None:
        workflows = _read_workflow_texts(agent_root)
    else:
        workflows = {"ci.yml": ci_text}
    if precommit_text is None:
        precommit_texts = _read_precommit_texts(agent_root)
    else:
        precommit_texts = {".pre-commit-config.yaml": precommit_text}

    guard_by_test: dict[str, str] = {}
    for text in workflows.values():
        for name, test in _parse_guard_steps(text):
            guard_by_test.setdefault(test, name)

    gates: list[dict] = []
    for test, name in guard_by_test.items():
        gates.append(
            {
                "test": test,
                "arch": test.startswith("tests/test_arch_"),
                "status": "fast-fail-guard",
                "ci_step": name,
                "exists": (agent_root / test).exists(),
            }
        )

    orphans: list[str] = []
    for test in discover_governance_tests(agent_root):
        if test in guard_by_test:
            continue
        if test in FULL_SUITE_ONLY:
            gates.append(
                {
                    "test": test,
                    "arch": test.startswith("tests/test_arch_"),
                    "status": "full-suite-only",
                    "ci_step": None,
                    "exists": True,
                }
            )
        else:
            orphans.append(test)

    gates.sort(key=lambda g: g["test"])
    dangling = [
        {"ci_step": g["ci_step"], "test": g["test"]}
        for g in gates
        if g["status"] == "fast-fail-guard" and not g["exists"]
    ]

    # 非 pytest 形态的治理门 (CLI 自查 / 棘轮): `python -m huginn.<module>`, 跨**全部**
    # 工作流按模块去重 (一个门多处调用只登记一次), module 映射到仓内 .py / 包目录判存在.
    cli_guards = _collect_cli_guards(agent_root, workflows)
    dangling_cli = [
        {"ci_step": g["ci_step"], "module": g["module"]}
        for g in cli_guards
        if not g["exists"]
    ]

    # 本地门 (仓根与 agent/ 两份 .pre-commit-config.yaml): 既不在工作流也不在台账, 故单列.
    precommit_gates = _collect_precommit_gates(agent_root, precommit_texts)
    dangling_precommit = [
        {"hook": g["hook"], "test": g["test"]}
        for g in precommit_gates
        if not g["exists"]
    ]
    uncovered_precommit = [
        {"hook": g["hook"], "test": g["test"]}
        for g in precommit_gates
        if not g.get("covered")
    ]

    # 工程门 (非 pytest 阻断门): 由工作流 step 承载, 判 step 名是否仍在对应工作流 ——
    # step 删/改名即静默缩小治理面 (旧台账完全看不见).
    engineer_gates = _collect_engineer_gates(workflows)
    dangling_engineer = [
        {"id": g["id"], "workflow": g["workflow"], "step": g["step"]}
        for g in engineer_gates
        if not g["exists"]
    ]

    # 工作流触发器 (运行面): 门 step 存在 ≠ 门会运行. 触发规格 ①判在否/是否收窄,
    # ②暴露未登记工作流 (须显式登记触发规格) 与未落地规格 (工作流文件被删/改名).
    triggers = _collect_workflow_triggers(workflows)

    # 能力锚点 (子系统级): 开关默认值未翻转 + 承载模块存在 + 不变量测试存在. 三者任一脱节
    # = 该能力静默萎缩 (前面九形态的门存在性判据全绿也看不出).
    capabilities = _collect_capabilities(agent_root)
    capability_gaps = [
        {
            "id": c["id"],
            "name": c["name"],
            "flag_ok": c["flag_ok"],
            "missing_modules": c["missing_modules"],
            "missing_tests": c["missing_tests"],
        }
        for c in capabilities
        if not c["flag_ok"] or c["missing_modules"] or c["missing_tests"]
    ]

    # prompt 段生效面 (声明清单 vs 源文件 `_APPLIES_TO`): 段是输出规范的执行单元, 一个段
    # 静默消失即规范萎缩. 门存在性/能力锚点判据都看不见 prompt 面.
    prompt_segments = _collect_prompt_segments(agent_root)

    return {
        "gates": gates,
        "orphans": sorted(orphans),
        "dangling": dangling,
        "cli_guards": cli_guards,
        "dangling_cli": dangling_cli,
        "precommit_gates": precommit_gates,
        "dangling_precommit": dangling_precommit,
        "uncovered_precommit": uncovered_precommit,
        "engineer_gates": engineer_gates,
        "dangling_engineer": dangling_engineer,
        "workflows": sorted(workflows),
        "workflow_triggers": triggers["workflow_triggers"],
        "unregistered_workflows": triggers["unregistered_workflows"],
        "unchecked_workflows": triggers["unchecked_workflows"],
        "trigger_drift": triggers["trigger_drift"],
        "capabilities": capabilities,
        "capability_gaps": capability_gaps,
        "prompt_segments": prompt_segments,
        "budgets": collect_budgets(agent_root, heavy=heavy),
    }


def build_gate_registry() -> dict:
    """真实仓快照 (agent_root = 本文件所在包根); 只跑 weightless 探针, 门禁轻量."""
    return audit(AGENT_ROOT)


def _render_current(current: int | tuple[int, ...] | list[int] | None) -> str:
    if current is None:
        return "—"
    if isinstance(current, tuple | list):
        return "/".join(str(v) for v in current)
    return str(current)


def render_markdown(reg: dict) -> str:
    """确定性渲染 (固定列序 + 门按测试路径字典序)."""
    lines: list[str] = []
    lines.append("# 低熵治理台账 (governance state)")
    lines.append("")
    lines.append(
        "自动生成: `python -m huginn.cli.governance_audit --out docs/governance-state.md`."
    )
    lines.append(
        "登记 `tests/test_arch_*.py` 门族 + 显式治理元门 (`GOVERNANCE_META_TESTS`)、"
        "`.github/workflows/*.yml` **全部工作流**的 python guard step "
        "(pytest 与 `python -m huginn.<module>`)、非 pytest 工程门 (`ENGINEER_GATE_SPECS`)、"
        "仓根 + `agent/` 两份 `.pre-commit-config.yaml` 本地门, 以及命名子系统的**能力锚点** "
        "(`CAPABILITY_SPECS`: 开关默认值/承载模块/不变量测试) 的对齐面: "
        "**孤儿门** (存在却未登记)、**悬空门** (指向不存在的测试/模块/step)、**能力锚点脱节** "
        "(开关默认翻转/承载模块或回归测试缺失) 在此显形。"
        "只提示不判死; 硬判定在 `tests/test_arch_gate_registry.py`."
    )
    lines.append("")
    lines.append(f"## 工作流 (workflows) — {len(reg['workflows'])}")
    lines.append("")
    lines.append(
        "被扫描的**全部**工作流文件 (门可能因整个工作流被删/改名而集体消失; 数下限棘轮 "
        "`workflow_floor` 兜底)."
    )
    lines.append("")
    lines.append(", ".join(f"`{w}`" for w in reg["workflows"]))
    lines.append("")
    lines.append("## 工作流触发器 (workflow triggers)")
    lines.append("")
    lines.append(
        "每个工作流的**期望触发集**与**实际触发集**比对 (源: "
        "`governance_audit.WORKFLOW_TRIGGER_SPECS`). 门 step 存在 ≠ 门会运行 —— 删 "
        "`on: schedule` 或收窄 `push`/`pull_request` 的 `paths` 会让门永不执行, 而 step 仍在"
        "文件里 (存在性判据判「在」= 假绿). **状态=漂移** 即触发器缺失/多出; 未登记触发规格的"
        "工作流 (文件在、规格缺) 与规格未落地的工作流 (规格在、文件缺) 单列于缺口区."
    )
    lines.append("")
    lines.append("| 工作流 | 期望触发 | 实际触发 | 状态 |")
    lines.append("|---|---|---|---|")
    for t in reg["workflow_triggers"]:
        drift = "漂移" if (t["missing"] or t["extra"]) else "一致"
        exp = ", ".join(f"`{e}`" for e in t["expected"]) or "—"
        act = ", ".join(f"`{a}`" for a in t["actual"]) or "—"
        lines.append(f"| `{t['workflow']}` | {exp} | {act} | {drift} |")
    lines.append("")
    lines.append("## 门 (gates)")
    lines.append("")
    lines.append("| 测试 | arch | 状态 | CI step |")
    lines.append("|---|---|---|---|")
    for g in reg["gates"]:
        step = g["ci_step"] if g["ci_step"] else "—"
        if g["status"] == "fast-fail-guard" and not g["exists"]:
            step = f"{step} (悬空)"
        lines.append(f"| `{g['test']}` | {'是' if g['arch'] else '否'} | `{g['status']}` | {step} |")
    lines.append("")
    lines.append("## CLI 门 (cli gates)")
    lines.append("")
    lines.append(
        "**全部工作流**里 `python -m huginn.<module>` 形态的治理门 (CLI 自查 / 棘轮, 非 pytest). "
        "旧解析只认 pytest 会漏掉; 现单列。**存在=否** 即悬空 CLI 门 (指向已删/改名的模块)."
    )
    lines.append("")
    lines.append("| 模块 | 工作流 | CI step | 存在 |")
    lines.append("|---|---|---|---|")
    for c in reg["cli_guards"]:
        step = c["ci_step"] if c["ci_step"] else "—"
        mark = "是" if c["exists"] else "否 (悬空)"
        lines.append(f"| `{c['module']}` | `{c['workflow']}` | {step} | {mark} |")
    lines.append("")
    lines.append("## 本地门 (pre-commit local gates)")
    lines.append("")
    lines.append(
        "仓根与 `agent/` 两份 `.pre-commit-config.yaml` 的本地门 (如 G37 audit-regression). "
        "既不在工作流也不在别处登记, 故单列。**存在=否** 即悬空本地门 (引用的测试已改名/删除 —— "
        "此时 hook 的 `files:` 过滤器会静默不再匹配, 门悄悄死掉). **files 覆盖=否** 即 "
        "`entry` 引用的测试不在 `files:` 过滤范围内 —— 改它不会触发 hook, 门静默失效."
    )
    lines.append("")
    lines.append("| 来源 | 本地门 (hook id) | 引用测试 | 存在 | files 覆盖 |")
    lines.append("|---|---|---|---|---|")
    for g in reg["precommit_gates"]:
        mark = "是" if g["exists"] else "否 (悬空)"
        cov = "是" if g.get("covered") else "否 (漏匹配)"
        lines.append(f"| `{g['source']}` | `{g['hook']}` | `{g['test']}` | {mark} | {cov} |")
    lines.append("")
    lines.append("## 工程门 (engineer gates)")
    lines.append("")
    lines.append(
        "非 pytest 的**阻断门** (ruff lint / pip-audit / npm audit / bandit SAST / cargo audit / "
        "gitleaks / lock 漂移 / 前端构建·单测·E2E / 技能库审计 / ultra-long benchmark / quickstart "
        "smoke). 由工作流 step 承载, 旧台账完全看不见 —— step 删/改名即静默缩小治理面. 现由 "
        "`ENGINEER_GATE_SPECS` 登记并判存在。**存在=否** 即悬空工程门 (step 名与工作流脱节)."
    )
    lines.append("")
    lines.append("| 门 (id) | 工作流 | 类别 | CI step | 存在 |")
    lines.append("|---|---|---|---|---|")
    for e in reg["engineer_gates"]:
        mark = "是" if e["exists"] else "否 (悬空)"
        lines.append(f"| `{e['id']}` | `{e['workflow']}` | {e['kind']} | {e['step']} | {mark} |")
    lines.append("")
    lines.append("## 能力锚点 (capability anchors)")
    lines.append("")
    lines.append(
        "命名子系统 (RSI / PMK / 世界模型 …) 的生存锚点 (源: `CAPABILITY_SPECS`): "
        "**一枚开关 + 一组承载模块 + 一组不变量测试**. 门存在性判据看不见能力面 —— 开关还在"
        "但**默认翻转** (从开变关 = 能力悄悄关掉)、承载模块被删 (= 能力失去实现)、不变量测试被删"
        "(= 能力失去回归网) 都会让能力静默萎缩而门全绿. **默认一致=否** 即开关被删/默认翻转; "
        "承载模块/不变量测试列**存在=否** 即锚点脱节. 数下限棘轮 `capability_floor` 兜底. "
        "`always_on` 型 = 工具 action 面/机制面能力 (无开关可翻转, 硬凑开关 = 噪声), "
        "锚点退化为双判据: 只判承载模块/不变量测试存在, 依赖面另由 L0 探针 (runtime_probe) 守."
    )
    lines.append("")
    lines.append("| 能力 | 开关 | 类型 | 期望默认 | 实际默认 | 默认一致 | 承载模块 | 不变量测试 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in reg["capabilities"]:
        kind = {
            "feature_flag": "feature_flag",
            "always_on": "常开 (无开关)",
        }.get(c["flag_kind"], "env")
        flag = "是" if c["flag_ok"] else "否 (翻转/缺失)"
        mods = ", ".join(
            f"`{m}`" + ("" if (m not in c["missing_modules"]) else " (缺)")
            for m in c["modules"]
        )
        tsts = ", ".join(
            f"`{t}`" + ("" if (t not in c["missing_tests"]) else " (缺)")
            for t in c["tests"]
        )
        lines.append(
            f"| {c['name']} (`{c['id']}`) | `{c['flag']}` | {kind} | "
            f"{c['expected_default']} | {c['actual_default']} | {flag} | {mods} | {tsts} |"
        )
    lines.append("")
    lines.append("## prompt 段生效面 (prompt segments)")
    lines.append("")
    lines.append(
        "prompt 段的**声明面** (`prompt_segments._APPLIES_TO`: 段名 -> 必须生效的生产路径 "
        "{main,s7,autoloop}) 与**生效面** (各生产路径实际渲染点) 的登记 (源: "
        "`governance_audit.PROMPT_SPECS` 期望段名 + 按文件路径 ast 读源文件 `_APPLIES_TO`). "
        "逐段的生效路径**不在此复制** (无镜像漂移); 「声明 == 渲染」由 "
        "`tests/test_prompt_contract.py` 硬判定. 此处只显形**清单漂移** "
        "(清单有、源里无 = 段被删; 源里有、清单无 = 新增未登记) 与逐段声明路径. "
        "段数下限棘轮 `prompt_segment_floor` 兜底."
    )
    lines.append("")
    lines.append("| 段 | 声明生效路径 |")
    lines.append("|---|---|")
    for seg in reg["prompt_segments"]["segments"]:
        paths = ", ".join(f"`{p}`" for p in seg["paths"]) or "—"
        lines.append(f"| `{seg['name']}` | {paths} |")
    lines.append("")
    def _budget_rows(rows_budgets: list[dict]) -> list[str]:
        out_rows = [
            "| 预算 | 位置 | 模式 | 冻结值 | 实测 | 余量 | 消费者门 |",
            "|---|---|---|---|---|---|---|",
        ]
        for b in rows_budgets:
            cur = _render_current(b["current"])
            head = "—" if b["headroom"] is None else str(b["headroom"])
            owner = b["owner_gate"] + (" · 全量" if b["owner_kind"] == "full_suite" else "")
            out_rows.append(
                f"| {b['name']} | `{b['path']}` | {b['mode']} | {b['frozen']} | {cur} | {head} | {owner} |"
            )
        return out_rows

    lines.append("## 预算 (budgets)")
    lines.append("")
    lines.append(
        "治理棘轮的冻结预算/基线单一台账 (源: `governance_audit.BUDGET_SPECS`). "
        "**权威** = 值在台账声明、消费者门 `budget_value()` 只读; **引用** = 值仍在其自然归属, "
        "台账只登记。实测/余量仅对 weightless 可测项给出; 其余由各自消费者门校验 (current ≤ frozen); "
        "重型探针 (如镜像漂移实测) 需 `--probe`。**消费者门**列: `ci_step` 型是工作流里 pytest/CLI "
        "guard step (有存在性门); `workflow_step` 型是非 pytest 工程门 step; 带 `· 全量` 标记的是 "
        "`full_suite` 型 (仅全量套件跑, 不在工作流 step 里)。"
    )
    lines.append("")
    lines.extend(_budget_rows([b for b in reg["budgets"] if b.get("group") != "loop"]))
    lines.append("")
    lines.append("## 循环闸位 (loop caps)")
    lines.append("")
    lines.append(
        "自主循环的轮数/步数**上限** (源: `governance_audit.BUDGET_SPECS` 中 `group=loop` 项). "
        "这些值此前散在 `env_schema` 与各引擎/路由的硬编码默认里, 无单一上限可读。**模式=上限** "
        "(ceiling): 代码持有实测值 (weightless regex 探针从源码读出), 台账持有允许的最大值; "
        "棘轮 = 实测 ≤ 上限 —— 提高上限 (扩大自主面) 必须显式改台账 `value` (可见决策)。"
        "**模式=权威** 一行为闸位**数下限** (只减不增)。"
    )
    lines.append("")
    lines.extend(_budget_rows([b for b in reg["budgets"] if b.get("group") == "loop"]))
    lines.append("")
    lines.append("## 缺口 (gaps)")
    lines.append("")
    if reg["orphans"]:
        lines.append(f"- 孤儿门 ({len(reg['orphans'])}): " + ", ".join(f"`{t}`" for t in reg["orphans"]))
    else:
        lines.append("- 孤儿门 (0): 无")
    if reg["dangling"]:
        items = ", ".join(f"`{d['ci_step']}` → `{d['test']}`" for d in reg["dangling"])
        lines.append(f"- 悬空门 ({len(reg['dangling'])}): {items}")
    else:
        lines.append("- 悬空门 (0): 无")
    if reg["dangling_cli"]:
        items = ", ".join(f"`{d['ci_step']}` → `{d['module']}`" for d in reg["dangling_cli"])
        lines.append(f"- CLI 悬空门 ({len(reg['dangling_cli'])}): {items}")
    else:
        lines.append("- CLI 悬空门 (0): 无")
    if reg["dangling_precommit"]:
        items = ", ".join(
            f"`{d['hook']}` → `{d['test']}`" for d in reg["dangling_precommit"]
        )
        lines.append(f"- 本地悬空门 ({len(reg['dangling_precommit'])}): {items}")
    else:
        lines.append("- 本地悬空门 (0): 无")
    if reg["uncovered_precommit"]:
        items = ", ".join(
            f"`{d['hook']}` → `{d['test']}`" for d in reg["uncovered_precommit"]
        )
        lines.append(
            f"- 本地门 files/entry 不一致 ({len(reg['uncovered_precommit'])}): {items}"
        )
    else:
        lines.append("- 本地门 files/entry 不一致 (0): 无")
    if reg["dangling_engineer"]:
        items = ", ".join(
            f"`{d['id']}` (`{d['workflow']}`) → `{d['step']}`"
            for d in reg["dangling_engineer"]
        )
        lines.append(f"- 悬空工程门 ({len(reg['dangling_engineer'])}): {items}")
    else:
        lines.append("- 悬空工程门 (0): 无")
    if reg["trigger_drift"]:
        items = ", ".join(
            f"`{d['workflow']}` (缺 {d['missing'] or '—'} / 多 {d['extra'] or '—'})"
            for d in reg["trigger_drift"]
        )
        lines.append(f"- 触发器漂移 ({len(reg['trigger_drift'])}): {items}")
    else:
        lines.append("- 触发器漂移 (0): 无")
    if reg["unregistered_workflows"]:
        items = ", ".join(f"`{w}`" for w in reg["unregistered_workflows"])
        lines.append(
            f"- 未登记触发规格的工作流 ({len(reg['unregistered_workflows'])}): {items}"
        )
    else:
        lines.append("- 未登记触发规格的工作流 (0): 无")
    if reg["unchecked_workflows"]:
        items = ", ".join(f"`{w}`" for w in reg["unchecked_workflows"])
        lines.append(
            f"- 触发规格未落地的工作流 ({len(reg['unchecked_workflows'])}): {items}"
        )
    else:
        lines.append("- 触发规格未落地的工作流 (0): 无")
    if reg["capability_gaps"]:
        items = ", ".join(_capability_gap_desc(g) for g in reg["capability_gaps"])
        lines.append(f"- 能力锚点脱节 ({len(reg['capability_gaps'])}): {items}")
    else:
        lines.append("- 能力锚点脱节 (0): 无")
    ps = reg["prompt_segments"]
    if ps["missing_in_source"]:
        items = ", ".join(f"`{n}`" for n in ps["missing_in_source"])
        lines.append(
            f"- prompt 段清单有、源里无 ({len(ps['missing_in_source'])}): {items}"
        )
    else:
        lines.append("- prompt 段清单有、源里无 (0): 无")
    if ps["unregistered"]:
        items = ", ".join(f"`{n}`" for n in ps["unregistered"])
        lines.append(
            f"- prompt 段源里有、清单无 ({len(ps['unregistered'])}): {items}"
        )
    else:
        lines.append("- prompt 段源里有、清单无 (0): 无")
    lines.append("")
    return "\n".join(lines)


def _capability_gap_desc(gap: dict) -> str:
    """单项能力锚点脱节的缺口描述 (开关默认翻转 / 承载模块缺 / 不变量测试缺)."""
    parts: list[str] = []
    if not gap["flag_ok"]:
        parts.append("开关默认翻转/缺失")
    if gap["missing_modules"]:
        parts.append("缺模块 " + "、".join(f"`{m}`" for m in gap["missing_modules"]))
    if gap["missing_tests"]:
        parts.append("缺测试 " + "、".join(f"`{t}`" for t in gap["missing_tests"]))
    return f"`{gap['name']}` ({'; '.join(parts)})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="低熵治理台账 (governance_audit)")
    parser.add_argument("--out", default="", help="写 markdown 到文件 (缺省打印到 stdout)")
    parser.add_argument("--json", action="store_true", help="输出 JSON 快照")
    parser.add_argument(
        "--probe", action="store_true", help="额外跑重型探针 (遍历全仓, 较慢; 不入快速失败门)"
    )
    args = parser.parse_args(argv)

    reg = audit(AGENT_ROOT, heavy=args.probe)
    if args.json:
        import json

        print(json.dumps(reg, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    md = render_markdown(reg)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
