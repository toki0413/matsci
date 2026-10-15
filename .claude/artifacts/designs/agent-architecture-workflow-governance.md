# Agent 架构 / 工作流 / 低熵治理 完善 Spec

> Status: ALIGNED
> Author: user
> Last updated: 2026-10-15

## Background

用户要求「完善 agent 架构、工作流和低熵治理，提升可维护性、可观察性和可解释性」。
这是**系统级工程改进**诉求，横跨三个面（运行时架构 / 工作流编排 / 防漂移治理），
无新功能边界。仓库（`matsci` / `huginn-agent`）**已有**成熟地基：MECE 契约审计、
11 篇自动再生成契约文档、~14 个 CI「arch 门」+ MECE 棘轮基线、`publish/mcpb`
镜像 only-shrink 棘轮、`ponytail` 技术债登记、OTel / explainability / governance
facade。经 Wave 2 确认，本轮优先面锁定**低熵治理面**，且「低熵治理」= 既有防漂移
体系（MECE / 契约 / 棘轮 / 债务登记）的延续。

## In scope

面 = **低熵治理面**（用户 2026-10-15 确认；增量收益最大、风险最低，沿用既有地基）：

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
- A-2 三个面本轮只落 **低熵治理面**（已确认）。
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

### 增量 (2026-10-15, M1′)：统一预算台账

G-1 的「预算/基线 → 当前值」最初只落 MECE + 镜像两项；本轮补齐为单一
`governance_audit.BUDGET_SPECS` 台账，覆盖 10 项棘轮：MECE 发现基线 / 镜像漂移预算 /
环境裸读基线 / 外部直连白名单 / 冻结委托子命令 / 插件准入清单 / 硬编码路径白名单 /
MCP 接线白名单 / arch 门数下限 / 字段审计 golden。weightless（regex + ast，不 import
被测模块）抽取冻结值，对可测项给实测/余量；新增元门禁（id 唯一且覆盖集冻结、
冻结值抽取无 `?`、探针余量非负、inline 值不得在消费者门重声明）。**权威读取**已落实：
数值型棘轮（`mirror_drift`、`arch_gate_floor`）的值在台账声明、消费者门 `budget_value()`
只读；名单型棘轮与 `env_schema.RAW_READ_BASELINE` 按设计保留原位（内容即契约 / 层级原因），
台账以「引用」登记。**实测探针**分两档：weightless 常驻（arch 门数，入文档，门禁仍 0.5s）与
重型 `--probe`（镜像漂移实测，复用 `scripts/mcpb_vendor` 口径，不入快速失败门）。

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

- ~~OQ-1 优先面~~ → 已决（2026-10-15）：低熵治理面。
- ~~OQ-2「低熵治理」含义~~ → 已决（2026-10-15）：= 既有防漂移体系的延续。
- **OQ-3 (可选，不阻塞)**：治理状态工件是否设为 CI 确定性门（防漂移），还是仅作文档？
  默认取「仅作文档 + 孤儿门/门数棘轮做门」，实施中如需改判再报告 spec drift。

## Core entities (ontology)

| Entity | Type | Key fields | Relationship |
|---|---|---|---|
| GovernanceGate | 概念 | name, test_entry, ci_step, budget | 1 gate → 1 测试入口 + 1 ci step |
| RatchetBaseline | 工件 | path, frozen_count | gate 消费 baseline |
| GovernanceState | 工件 | gates[], budgets[], current_values[] | 由 governance_audit 生成 |
| DriftGuard | 已有测试族 | `test_arch_*.py` | 实现 gate 断言 |

## Interview metadata

- Mode: default
- Waves: 2
- Final ambiguity: ~25%
- Status: ALIGNED (Wave 2 由用户确认优先面后收敛)

### Clarity breakdown

| Dimension | Score | Weight | Weighted | gap |
|---|---|---|---|---|
| Goal | 0.80 | 0.40 | 0.32 | 单面 + 具体缺口已定 |
| Scope | 0.75 | 0.25 | 0.1875 | in/out 已定 |
| AC | 0.75 | 0.25 | 0.1875 | 候选面 AC 可执行 |
| Context | 0.80 | 0.10 | 0.08 | 已实测既有治理地基 |

### Ontology convergence

Wave 1 首现实体：GovernanceGate / RatchetBaseline / GovernanceState / DriftGuard
（无历史轮次可比；「低熵治理」为新造词，未与既有词表对齐）。
