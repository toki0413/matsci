# Agent 架构 / 工作流 / 低熵治理 完善 Implementation Plan

> Status: APPROVED
> Source: `.claude/artifacts/designs/agent-architecture-workflow-governance.md`
> Mode: (default)
> Iterations: 1 / 3
> Author: user
> Last updated: 2026-10-15

## Requirements summary

落实 spec 的**低熵治理面**：把散落的治理门登记为单一机器可读来源，并补上治理层
自身的熵缺口 —— 「门存在却永不运行 / CI step 指向不存在的测试」。范围限定为
**只读扫描 + 新门禁**，不动任何生产行为。

## Acceptance criteria

- AC-1 `python -m huginn.cli.governance_audit --out docs/governance-state.md`
  产出与已提交文件**逐字符相等**（确定性再生成）。
- AC-2 `tests/test_arch_gate_registry.py` 断言：每个 `tests/test_arch_*.py` 与每个
  `ci.yml` 里的 python guard step 都在登记表中出现；未登记 → 红。
- AC-3 CI guard step 指向不存在的测试文件（dangling guard）→ 红。
- AC-4 门数棘轮只减不增：删掉一个已登记门 → 红。
- AC-5 未登记治理门时给出可执行的修复提示（含需改的文件路径）。

## RALPLAN-DR

### Principles

- 最小代码：不引入新依赖（复用 stdlib + 已有 PyYAML 若已可用）。
- 外科手术式改动：只新增 1 个 CLI + 1 个测试 + 1 个生成文档 + 1 处 CI step + INDEX 登记。
- 与既有约定同构：对齐 `config_audit` / `contract_audit` 的 `--out` / `--check` / 确定性再生成范式。
- 失败可定位：门禁报错必须点名文件路径与修复动作，不报「有漂移」这种空话。
- 不新增不可逆面：纯静态扫描，无网络、无写副作用（`--out` 除外）。

### Decision drivers

1. **一致性**：仓库已有 11 篇自动再生成契约文档 + `test_config_contracts` 漂移门；
   治理状态文档跟随同一范式，读者不必学两套。
2. **可观察性**：单一 `docs/governance-state.md` 让「有几个门 / 预算多少 / 当前值」
   一眼可见，替代散读 `ci.yml` + 各 golden。
3. **维护成本**：门散落且无登记时，新增门易漏接线；登记 + 孤儿检测把这类漏接
   从「上线后才发现」提前到 PR。

### Viable options

**Option A：CI-only bash 门（无 CLI、无文档）**
- 实现思路：在 `ci.yml` 加一个 inline `python -c` 扫描，发现即 exit 1。
- 改动文件：`.github/workflows/ci.yml`。
- Pros：改动最小，零新文件。
- Cons：**违背 spec G-1/G-3**（无登记来源、无可观察工件）；inline bash 难测试、
  难复用；逻辑藏在 YAML 里可维护性差。

**Option B：CLI 模块 + 元测试 + 生成文档（spec 主张）**
- 实现思路：新增 `huginn/cli/governance_audit.py`（argparse，仿 `config_audit`），
  扫描 `tests/test_arch_*.py` + `ci.yml` guard steps + `tests/golden/*` 基线 +
  `test_publish_mirror_drift.py` 预算，产出 `docs/governance-state.md`；新增
  `tests/test_arch_gate_registry.py` 承载 AC-2/3/4/5；`ci.yml` 里以 weightless
  快速失败门接入。
- 改动文件：`huginn/cli/governance_audit.py`(新)、`tests/test_arch_gate_registry.py`(新)、
  `docs/governance-state.md`(新)、`.github/workflows/ci.yml`、`agent/docs/INDEX.md`。
- Pros：直接满足 G-1/G-2/G-3；CLI 可本地跑、可测；与既有契约文档范式同构。
- Cons：新增一个需同步的生成文档（一处 golden 熵）；需解析 YAML。

**Option C：纯 pytest 元测试（登记表硬编码在测试里）**
- 实现思路：不写 CLI，登记表作为常量放在 `test_arch_gate_registry.py`。
- 改动文件：`tests/test_arch_gate_registry.py`(新)、`.github/workflows/ci.yml`。
- Pros：最少文件；无 YAML 解析。
- Cons：**无 G-1 单一来源**（登记表在测试里不可被人读/被其他工具消费）；
  **无 G-3 可观察工件**；门数增长时要改测试常量，反而不如生成。

**被砍选项的 invalidation rationale**：A 与 C 都只满足 spec 三个子目标中的 1–2 个，
不满足「单一机器可读登记 + 可观察工件」的核心诉求，故不选。

**Favored：Option B**（与 spec 一致），并采纳 Architect/Critic 的收敛（见下）。

### Implementation steps（基于 Option B）

1. 新增 `huginn/cli/governance_audit.py`：
   - `main(argv)` argparse，含 `--out`（默认 `""`）、`--json`、`--check`
     （配 `--baseline` 棘轮）、`--update-baseline`，对齐
     [contract_audit.py](file:///workspace/agent/huginn/cli/contract_audit.py#L7591-L7623)。
   - `build_gate_registry()` 扫描：
     a) `git ls-files tests/test_arch_*.py`（或 `Path.glob`）→ 全部 arch 测试族；
     b) 解析 [.github/workflows/ci.yml](file:///workspace/.github/workflows/ci.yml#L89-L165)
        的 python guard step（`name` + `run` 里的 `tests/...py` 路径）；
     c) `tests/golden/mece_findings_baseline.txt` 行数
        （[mece_findings_baseline.txt](file:///workspace/agent/tests/golden/mece_findings_baseline.txt)）；
     d) `tests/test_publish_mirror_drift.py` 的 `MAX_MISSING/MAX_EXTRA/MAX_DIFFERING`
        常量（[test_publish_mirror_drift.py](file:///workspace/agent/tests/test_publish_mirror_drift.py)）。
   - `render_markdown(registry)` 确定性输出（固定列序、字典序），风格对齐
     [config_audit.py](file:///workspace/agent/huginn/cli/config_audit.py)。
2. 生成并提交 `agent/docs/governance-state.md`。
3. 在 [INDEX.md](file:///workspace/agent/docs/INDEX.md#L29-L53) 第 2 节登记该文档
   （`test_config_contracts` 的 `test_contract_docs_referenced_in_index` 要求 INDEX 覆盖）。
4. 新增 `tests/test_arch_gate_registry.py`（不依赖 CLI 的纯静态断言，weightless）：
   - 覆盖不变量：`{test_arch_*.py} ⊆ registry`；`{ci.yml python guard 的 tests/*.py} ⊆ 存在文件`；
     门数 ≥ 冻结基线（棘轮）。
   - 合成树自检：临时造 `test_arch_demo.py` 不入登记 → 断言红（防门禁自己失效）。
5. 在 [ci.yml](file:///workspace/.github/workflows/ci.yml#L154-L156) 的 Credential gateway
   guard 之后、MECE ratchet（L163）之前，插入 weightless guard step：
   `python -m pytest tests/test_arch_gate_registry.py -q -o addopts=""`。
6. 重跑 `python -m huginn.cli.governance_audit --out docs/governance-state.md`
   固化文档；若与提交不一致则失败（AC-1）。

### Workspace setup

- 实施前运行 `git status --short` 与 `git branch --show-current`。
- 实测当前：分支 `fix/mece-hook-audit`（非 main/master），working tree 仅
  `.claude/artifacts/**` 两文件待提交（属本轮产物）。
- 结论：**无需新 worktree** —— 已在特性分支、tree 干净；本 plan 的改动与
  待提交 spec/plan 同源，不混入他人改动。

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| YAML 解析依赖不确定（PyYAML 未必是运行依赖） | 优先 `import yaml`；不可用时退化为行扫描（匹配 `run:` 行里 `tests/.*\.py`），并在文档标注所用模式。 |
| 行扫描漏抓新写法（`run: \|` 多行 / matrix） | 合成树自检 + 断言「ci.yml 里每个 `python -m pytest tests/...` 都进登记」双向覆盖。 |
| 新增 golden 文档 = 新增一处同步点 | 与既有 11 篇契约文档同范式；漂移由 CI 门自动抓，非人肉；不引入第二个 manual 同步点。 |
| 「未登记 arch 测试」强制决策可能引起 churn | 登记表允许显式状态 `fast-fail-guard` / `full-suite-only`；后者是合法登记（当前 5 个未显式 guard 的 arch 测试归此类），不强迫加 CI step。 |
| `docs/governance-state.md` 内容随门数量变化而漂 | 由 `--check --baseline` 棘轮 + 逐字漂移测试共同锁；稳定后可作为「治理台账」长期读。 |

## Verification steps

- AC-1: `cd agent && python -m huginn.cli.governance_audit --out docs/governance-state.md && git -C /workspace diff --exit-code agent/docs/governance-state.md`
- AC-2/AC-5: `cd agent && python -m pytest tests/test_arch_gate_registry.py -q -o addopts=""`
- AC-3: 在测试内合成一棵临时树，写一个指向不存在文件的 CI step，断言红。
- AC-4: 测试内把登记表删一项、断言棘轮红。
- 回归: `cd agent && python -m pytest tests/test_config_contracts.py tests/test_contract_audit.py -q -o addopts=""`（确保 INDEX/契约文档未漂）。
- 门禁本身: CI 对应 step 必须与本地同名同命令（weightless，先于重型套件）。

## ADR

- **Decision**：新增 `huginn/cli/governance_audit.py` 管治门登记与 `docs/governance-state.md`
  确定性再生成，并以 `tests/test_arch_gate_registry.py` + CI weightless guard 锁死
  「登记覆盖 / 孤儿门 / 门数棘轮」三条不变量。
- **Drivers**：一致性（对齐既有契约文档范式）> 可观察性（单一台账）> 维护成本（漏接提前到 PR）。
- **Alternatives considered**：
  - Option A（CI-only bash）→ rejected：无单一来源、无可观察工件、不可测。
  - Option C（纯 pytest 常量登记）→ rejected：登记表不可被人读/被工具消费，无 G-3。
  - Option B → **chosen**。
- **Why chosen**：唯一同时满足 spec G-1/G-2/G-3 且与仓库既有治理范式同构。
- **Consequences**：
  - 正：治理门可枚举、可观察，新增门漏接在 PR 阶段即红。
  - 负：多一个生成文档与一处 CI step；`docs/governance-state.md` 随门数变化需重生成。
  - 约束：后续新增任何治理门必须同步登记表，否则 CI 红（这正是期望的熵约束）。
- **Follow-ups**（应做未做，交回用户/backlog）：
  - 是否把 `docs/governance-state.md` 的确定性门纳入 CI 门（本轮已按 OQ-3 改判为「是」，
    因与 11 篇契约文档同侪——**spec drift，已报**）。
  - 治理门的时间序列（趋势）本 plan 不做。

## Review trail

- Planner draft v1：Option A/B/C，favored B（CLI + 元测试 + 生成文档）。
- Architect challenge v1：最强反驳 —— 「生成文档本身就是新增 golden，反而增加熵；
  纯 pytest 元测试（Option C）已足够，文档可由 `--json` 输出按需生成，不必提交」。
  发现 tradeoff tension：**一致性 vs 零新增同步点**；以及真实细节矛盾 —— ci.yml 只显式
  guard 了 16 个 arch 测试中的 11 个，其余 5 个靠全量套件兜底，因此「未入 ci.yml 即孤儿」
  的原始 AC-2 是**假不变量**，必须改为「未登记（含 full-suite-only 状态）才算孤儿」。
  Synthesis：保留 Option B，但把登记表扩展为带 `status` 字段；文档提交但漂移由门自动抓。
- Critic verdict v1：**APPROVED**，采纳三处改进：①AC-2 加 `status` 语义；②补 INDEX
  登记步骤；③补合成树自检防门禁自失效。
- Final iterations: 1 / 3

---

## 增量 (2026-10-15, M1′)：统一预算台账

> 状态：已落（M1 部分）；权威读取（phase 2）未落。

原 plan 的 `_collect_baselines` 只登记 2 项（MECE + 镜像）。按用户「先落预算台账」，
把 `governance_audit` 的预算面补成单一 `BUDGET_SPECS` 台账。

### 改动

- `huginn/cli/governance_audit.py`：新增 `BUDGET_SPECS`（10 项棘轮）+ `collect_budgets()`；
  weightless 抽取（`file-lines` / `const` / `const-triple` / `collection-len`(ast) /
  `dir-file-count`）；可测项探针（当前仅 `arch_gate_floor`，floor 型余量）；
  `render_markdown` 的「棘轮基线」节改为「预算 (budgets)」表。
- `tests/test_arch_gate_registry.py`：新增 3 条元门禁 —— 预算 id 唯一且覆盖集与
  `_EXPECTED_BUDGET_IDS` 一致（只许显式增删）、冻结值抽取无 `?`、探针余量非负。
- `docs/governance-state.md`：重生成（含预算表）。
- `publish/mcpb/*`：`scripts/mcpb_vendor.py --write` 同步镜像（改动的 `governance_audit.py`）。

### AC

- AC-1 `python -m huginn.cli.governance_audit --out docs/governance-state.md` 与提交文件逐字相等（沿用）。
- AC-2 每个预算项冻结值抽取成功（无 `?`）；新增/删除棘轮必须同步 `_EXPECTED_BUDGET_IDS`。
- AC-3 有探针的预算项余量非负。
- AC-4 镜像漂移门仍绿（`test_publish_mirror_drift.py`）。

### 验证（本地）

- `pytest tests/test_arch_gate_registry.py -o addopts=""` → 14 passed（0.5s，weightless）。
- `ruff check`（三文件）→ All checks passed。
- `pytest tests/test_publish_mirror_drift.py` → 2 passed（同步后 0 漂移）。
- 契约门（`test_config_contracts` / `test_contract_audit`）本地因缺 dev 依赖（rich/pydantic/
  langchain_core…）未能跑；基线无 `governance_audit` 引用，预期不漂 → 由 CI 兜底。

### phase 2（权威读取）— 已落（数值型）

把**数值型**棘轮的值收敛到 `BUDGET_SPECS` 单一入口，消费者门只读：

- `BUDGET_SPECS` 新增 `source: "inline"` + `value`（`mirror_drift` = (0,0,0)、
  `arch_gate_floor` = 17）；新增 `budget_value(id)` 供各门 import。
- `tests/test_publish_mirror_drift.py`：`MAX_MISSING/MAX_EXTRA/MAX_DIFFERING` ← `budget_value("mirror_drift")`。
- `tests/test_arch_gate_registry.py`：`ARCH_GATE_FLOOR` ← `budget_value("arch_gate_floor")`；
  新增元门禁 `test_authoritative_budgets_not_redeclared`（inline 值不得在消费者门重新硬编码）。
- `governance-state.md` 预算表新增「模式」列（权威 / 引用）。

**按设计保留原位（引用）**：`env_schema.RAW_READ_BASELINE`（层级原因）、以及各名单/清单型
棘轮（`ALLOWED_EXTERNAL_IMPORTS` 等 —— 其**内容即契约**，大小只是登记面）。

### 探针（weightless 常驻 + 重型 `--probe`）

- **常驻**（入台账文档，保证门禁轻量）：`arch_gate_count`（arch 门数下限，实测 17 / 余量 0）。
- **重型**（仅 `--probe`，遍历全仓，不入快速失败门，fail-soft 返回 None）：
  - `mirror_drift_live` —— 复用 `scripts/mcpb_vendor.drift_report` 口径，实测合计 0/0/0。
  - `env_raw_read_live` —— 复用 `config_audit.build_inventory` 口径（与 env 棘轮**同源**，
    不重复实现、无口径分叉），实测裸读点 467（正好顶基线，余量 0）。
- 新增元门禁 `test_heavy_probes_disabled_by_default`：凡标 `heavy` 的项默认实测必须为 None
  （门禁 0.5s 仍 weightless）。
- 复用前提：`config_audit` / `mcpb_vendor` 的**模块级仅 stdlib**（重依赖都在函数内），
  故按文件路径加载不拉 rich/pydantic；`config_audit` 若 `import huginn.cli` 会被包
  `__init__` 拉 rich —— 所以必须按文件路径加载。
- 实现踩坑：按文件路径加载模块须先登记进 `sys.modules` —— `@dataclass` 会
  `sys.modules.get(cls.__module__)`，未登记取到 None → 崩（曾让镜像探针静默失效）。
  已抽成 `_load_module_by_path` 供两个重型探针共用。

### 未落（后续）

- 若要把名单型棘轮也纳入权威读取，需先把「内容」与「大小」职责拆开（当前不建议）。

---

## 增量 (2026-10-16, M1′ 续)：CLI 门（非 pytest 形态）纳入台账

> 状态：已落。补齐治理层第 3 种熵形态 —— **CLI 门漏登记/悬空**。

原台账只认 `ci.yml` 里 `python -m pytest tests/...` 形态的 guard step，
`python -m huginn.<module>` 形态的治理门（`contract_audit` 棘轮、`policy_engine`
自查）被漏掉；其指向的模块改名/删除会形成「永远绿灯的悬空门」。

### 改动

- `huginn/cli/governance_audit.py`：
  - 新增 `_CLI_GUARD_RE` + `_cli_guard_in_line()`（**单一判据**，yaml 主路径与逐行
    fallback 共用）+ `_read_ci_text()` + `_parse_cli_guards()`。
  - `audit()` 增 `cli_guards`（按模块去重、映射到仓内 `.py`/包目录判存在）与
    `dangling_cli`。
  - `BUDGET_SPECS` 增 `cli_guard_floor`（inline 权威，值 2）；`_probe_current` 增
    `cli_guard_count`（weightless，读 `ci.yml` 文本）。
  - `render_markdown` 增「CLI 门 (cli gates)」表 + 缺口区「CLI 悬空门」。
- `tests/test_arch_gate_registry.py`：新增 5 条元门禁 —— CLI 悬空检测、去重、
  数量棘轮（`CLI_GUARD_FLOOR` ← `budget_value`）、合成树幽灵模块自检、
  **注释行不算门**（yaml/fallback 共用判据的确定性锚点）；`_EXPECTED_BUDGET_IDS`
  同步增 `cli_guard_floor`。
- `docs/governance-state.md`：重生成（含 CLI 门表）。
- `publish/mcpb/*`：`scripts/mcpb_vendor.py --write` 同步镜像。

### AC

- AC-1 逐字再生成（沿用）。
- AC-2 每个 CLI 门指向的模块存在（`dangling_cli == []`）。
- AC-3 CLI 门数只减不增（棘轮 ≥ `cli_guard_floor`）。
- AC-4 注释行里的 `python -m huginn.<module>` 不计为门（消除环境相关幻影门）。
- AC-5 镜像漂移门仍绿。

### 验证（本地）

- `pytest tests/test_arch_gate_registry.py tests/test_publish_mirror_drift.py -o addopts=""`
  → **21 passed**（with/without PyYAML 各一次）。
- `ruff check`（两文件）→ All checks passed。
- 实测 CLI 门 = 2（`huginn.cli.contract_audit` / `huginn.security.policy_engine`），
  yaml 主路径与 fallback 逐字一致（确定性回归确认）。
- `scripts/mcpb_vendor.py --check` → 0 漂移（同步后）。
