# 文档索引（单一事实来源导航）

本文件是 Huginn/Matsci 代码库文档的**导航入口**。所有文档按类别在此登记，
标注状态，避免"文档多但找不到/不知道哪个有效"。新增文档请先在此登记。

- **active**：现行有效，改动时应同步更新
- **staging**：草稿/待评审，尚未定稿
- **report**：历史报告/审计快照，非活文档

---

## 1. 项目总览与路径

| 文档 | 状态 | 用途 |
|---|---|---|
| [README.md](../README.md) | active | 项目总览、快速上手 |
| [.huginn.md](../.huginn.md) | active | Huginn 项目级引导（agent 读用） |
| [ROADMAP.md](../ROADMAP.md) | active | 路线图与里程碑 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | active | 贡献指南 |
| [SECURITY.md](../SECURITY.md) | active | 安全策略与报告渠道 |

## 2. 架构与设计与主要规范

| 文档 | 状态 | 用途 |
|---|---|---|
| [architecture.md](architecture.md) | active | 系统架构总览（技术架构） |
| [tech-spec.md](tech-spec.md) | active | 技术规格 |
| [enhanced_modules.md](enhanced_modules.md) | active | 增强模块说明 |
| [reward_design.md](reward_design.md) | active | 奖励设计 |
| [harness_evolution_spec.md](harness_evolution_spec.md) | active | harness 演进规范 |
| [hils_active_inference_p2_spec.md](hils_active_inference_p2_spec.md) | active | HILS 主动推断 P2 规范 |
| [async_dispatch_spec.md](async_dispatch_spec.md) | active | 异步分发规范 |
| [memory_dispatch_integration_spec.md](memory_dispatch_integration_spec.md) | active | 记忆分发集成规范 |
| [layered_memory_spec.md](layered_memory_spec.md) | active | 分层记忆规范 |
| [lsp_hashline_spec.md](lsp_hashline_spec.md) | active | LSP/hashline 规范 |
| [reinforcement_event_sourcing_sandbox_incremental_ui.md](reinforcement_event_sourcing_sandbox_incremental_ui.md) | active | 强化事件溯源 + 增量 UI 规范 |
| [ising_crdt_p1_spec.md](ising_crdt_p1_spec.md) | active | Ising CRDT P1 规范 |
| [SPEC_openworker_adoption](../huginn/SPEC_openworker_adoption.md) | active | OpenWorker 采纳规范 |
| [SPEC_visual_kb_loop](../huginn/metacog/SPEC_visual_kb_loop.md) | active | 视觉知识库闭环规范 |

## 3. 草稿 / 待评审（staging）

| 文档 | 状态 | 说明 |
|---|---|---|
| [staging/plans/2026-08-12-p0-state-store-and-rust.md](staging/plans/2026-08-12-p0-state-store-and-rust.md) | staging | P0 状态存储 + Rust 计划 |
| [staging/specs/2026-08-13-external-thinking.md](staging/specs/2026-08-13-external-thinking.md) | staging | 外部思考规范 |

> staging 下文档定稿通过评审后，移入"架构与设计"并标记 active。

## 4. 审计 / 报告 / 快照（report）

| 文档 | 状态 | 说明 |
|---|---|---|
| [dependency-audit-2026-08-13.md](dependency-audit-2026-08-13.md) | report | 依赖审计快照（2026-08-13） |
| [RELEASE_REPORT_v0.2.0.md](../RELEASE_REPORT_v0.2.0.md) | report | v0.2.0 发布报告 |
| [polish-reports/loop-polish-report.md](../polish-reports/loop-polish-report.md) | report | 循环打磨报告 |
| [polish-reports/industrialization-gap-analysis.md](../polish-reports/industrialization-gap-analysis.md) | report | 工业化缺口分析 |
| [polish-reports/benchmark-vs-claude-codex-evoscientist.md](../polish-reports/benchmark-vs-claude-codex-evoscientist.md) | report | 对标报告（Claude/Codex/Evoscientist） |

## 5. 部署与运维

| 文档 | 状态 | 用途 |
|---|---|---|
| [DEPLOYMENT.md](../DEPLOYMENT.md) | active | 部署指南 |
| [MONITORING.md](../MONITORING.md) | active | 监控指南 |
| [e2e_deployment_checklist.md](../tests/e2e_deployment_checklist.md) | active | 端到端部署检查清单 |

---

## 维护约定

- 新增/废弃文档请更新本索引，并标注状态。
- 废弃文档不直接删除，改标 `report` 并注明"已废弃，由 <新文档> 替代"。
- 本索引本身是导航，不替代各文档正文。