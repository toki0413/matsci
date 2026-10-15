# Agent 架构 / 工作流 / 低熵治理 完善 Spec

> Status: STUCK
> Author: user
> Last updated: 2026-10-15

## Background

用户要求「完善 agent 架构、工作流和低熵治理，提升可维护性、可观察性和可解释性」。
这是**系统级工程改进**诉求，横跨三个面（运行时架构 / 工作流编排 / 防漂移治理），
无新功能边界。仓库（`matsci` / `huginn-agent`）**已有**成熟地基：MECE 契约审计、
11 篇自动再生成契约文档、~14 个 CI「arch 门」+ MECE 棘轮基线、`publish/mcpb`
镜像 only-shrink 棘轮、`ponytail` 技术债登记、OTel / explainability / governance
facade。当前**目标粒度不足以写出可执行验收**，且「低熵治理」是仓库既有词表外的
新造词（grep 仅在 env-contract 命中同名英文变量，非项目语义）。

## In scope (candidate — 待用户确认优先面)

推荐面 = **低熵治理面**（增量收益最大、风险最低，沿用既有地基）：

- **G-1 治理门登记**：把散落的治理门（`tests/test_arch_*.py` + CI step + 预算/基线）
  收敛为单一机器可读登记（gate → 测试入口 → 预算/基线 → 当前值）。
- **G-2 孤儿门禁检测**：新增 `tests/test_arch_*.py` 但未接入 `ci.yml` 即变红
  （治理层自身的熵：门存在却永不运行）。
- **G-3 治理状态工件**：仿 `config_audit` 的确定性再生成，产出 `docs/governance-state.md`。

## Out of scope (this delivery)

- Agent 运行时架构重构：`agent/core.py`、`agents/*`、`hooks/*`、`events/*` 接线与生命周期。
- 工作流编排重构：`workflows/`、`skills/`、`autoloop/`、`runtime/` 生命周期。
- 可观察 / 可解释统一视图：`otel.py` ↔ `explainability.py` ↔ audit ↔ provenance 端到端打通。
- 任何生产行为变更 / 公开 API 变更。

## Assumptions

- A-1 目标为系统级工程改进，非新功能。
- A-2 三个面本轮只落 **一面**（优先面待用户确认）。
- A-3 「低熵治理」≈ 既有防漂移体系（MECE / 契约 / 棘轮 / 债务登记）的延续，非另起体系。
- A-4 现状事实：治理门众多但登记与预算分散在 `ci.yml`、`tests/golden/*`、
  `test_publish_mirror_drift.py`、`TECHNICAL_DEBT_PONYTAIL.md` 各处，缺统一可观察视图与孤儿门检测。

## Solution

最小可行草图（低熵治理面）：

- 新增 `huginn/cli/governance_audit.py`：静态扫描 `tests/test_arch_*.py` +
  `.github/workflows/ci.yml` 的 guard steps + `tests/golden/*` 基线 +
  `test_publish_mirror_drift.py` 预算常量，产出 `docs/governance-state.md`。
  确定性再生成，风格对齐既有 `config_audit`。
- 新增 `tests/test_arch_gate_registry.py`：断言每个治理门都在 `ci.yml` 被引用、
  且在治理状态工件中登记（孤儿门即红）。
- 棘轮：冻结当前门数，only-shrink 漂移预算（对齐 `test_publish_mirror_drift.py` 范式）。

## Edge cases & risks

| Category | Notes |
|---|---|
| Boundary conditions | `ci.yml` 中非 arch 的 guard（如 dependency lock drift）是否纳入登记范围需定。 |
| Failure modes | 扫描器正则漏抓新写法 → 治理门静默漏登记；需合成树自检防「门禁自己失效」。 |
| Risks | 若把「治理状态工件」设为逐字符门，会新增一处需同步的 golden，反而增加熵。 |
| Mitigation | 先只做**登记 + 孤儿检测**（可观测），工件确定性门列为 OQ-3 待定。 |

## Acceptance criteria

- AC-1 运行 `python -m huginn.cli.governance_audit --out docs/governance-state.md`，
  产出与已提交文件**逐字符相等**（确定性再生成）。
- AC-2 在 `tests/` 放入 `test_arch_demo_gate.py` 但不加入 `ci.yml` →
  `tests/test_arch_gate_registry.py` 断言变红。
- AC-3 把该门加入 `ci.yml` 后同一测试转绿（无假阳/假阴）。
- AC-4 治理门计数基线只减不增：人为删除一个已登记门 → 棘轮测试变红。

## Open questions

- **OQ-1 (BLOCKER, 需用户决策)**：本轮优先面 = 低熵治理面（建议）/ Agent 运行时架构 /
  工作流编排 / 可观察-可解释统一？—— 决定 `In scope`，未定则本 spec 无法进入 `dev-plan`。
- **OQ-2 (需用户决策)**：「低熵治理」是否等同既有防漂移体系？若另有含义，需给出边界。
- **OQ-3 (可选)**：治理状态工件是否设为 CI 确定性门（防漂移），还是仅作文档？

## Core entities (ontology)

| Entity | Type | Key fields | Relationship |
|---|---|---|---|
| GovernanceGate | 概念 | name, test_entry, ci_step, budget | 1 gate → 1 测试入口 + 1 ci step |
| RatchetBaseline | 工件 | path, frozen_count | gate 消费 baseline |
| GovernanceState | 工件 | gates[], budgets[], current_values[] | 由 governance_audit 生成 |
| DriftGuard | 已有测试族 | `test_arch_*.py` | 实现 gate 断言 |

## Interview metadata

- Mode: default
- Waves: 1
- Final ambiguity: ~72%
- Status: STUCK (CAP pending user；最高杠杆问题未获答)

### Clarity breakdown

| Dimension | Score | Weight | Weighted | gap |
|---|---|---|---|---|
| Goal | 0.30 | 0.40 | 0.12 | 三面并列，未定优先面 |
| Scope | 0.30 | 0.25 | 0.075 | in/out 仅为候选 |
| AC | 0.40 | 0.25 | 0.10 | 仅对候选面可写 AC |
| Context | 0.80 | 0.10 | 0.08 | 已实测既有治理地基 |

### Ontology convergence

Wave 1 首现实体：GovernanceGate / RatchetBaseline / GovernanceState / DriftGuard
（无历史轮次可比；「低熵治理」为新造词，未与既有词表对齐）。
