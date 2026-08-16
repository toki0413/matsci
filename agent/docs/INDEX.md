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
| [env-contract.md](env-contract.md) | active | 环境变量契约（265 个 HUGINN_*，可自动再生成） |
| [feature-flags-contract.md](feature-flags-contract.md) | active | 功能开关契约（44 个 FeatureFlags，可自动再生成） |
| [plugins-contract.md](plugins-contract.md) | active | 插件契约（Everything is a Plugin 注册面，可自动再生成） |
| [tools-contract.md](tools-contract.md) | active | 工具契约（ToolRegistry 核心工具面，可自动再生成） |
| [events-contract.md](events-contract.md) | active | 事件契约（EventType + UnifiedBus 发射面，可自动再生成） |
| [routes-contract.md](routes-contract.md) | active | 路由契约（ModelRouter task→tag，可自动再生成） |
| [errors-contract.md](errors-contract.md) | active | 错误语义契约（ErrorKind 分类面，可自动再生成） |
| [modes-contract.md](modes-contract.md) | active | Mode/Phase 契约（prompt 面，可自动再生成） |
| [model-tier-contract.md](model-tier-contract.md) | active | 模型档位契约（ModelTier 极简模式聚合面，可自动再生成） |
| [permission-contract.md](permission-contract.md) | active | 权限契约（PermissionMode + RiskLevel 五档 + PermissionConfig 细粒度面，可自动再生成） |
| [tech-spec.md](tech-spec.md) | active | 技术规格（已合并下述 8 个已废弃 spec） |
| [harness_evolution_spec.md](harness_evolution_spec.md) | active | harness 演进规范（H0-H4 落地，H5-a/H5-b 已落地） |
| [reward_design.md](reward_design.md) | staging | 奖励设计（未实现理论稿） |
| [SPEC_openworker_adoption](../huginn/SPEC_openworker_adoption.md) | active | OpenWorker 采纳规范 |
| [SPEC_visual_kb_loop](../huginn/metacog/SPEC_visual_kb_loop.md) | active | 视觉知识库闭环规范 |

已合并入 [tech-spec.md](tech-spec.md)，不再单独列出的 spec：`async_dispatch_spec`、`layered_memory_spec`、`lsp_hashline_spec`、`memory_dispatch_integration_spec`、`enhanced_modules`、`reinforcement_event_sourcing_sandbox_incremental_ui`、`ising_crdt_p1_spec`、`hils_active_inference_p2_spec`。

## 3. 草稿 / 待评审（staging）

| 文档 | 状态 | 说明 |
|---|---|---|
| [staging/plans/2026-08-12-p0-state-store-and-rust.md](staging/plans/2026-08-12-p0-state-store-and-rust.md) | staging | P0 状态存储 + Rust 计划 |
| [staging/specs/2026-08-13-external-thinking.md](staging/specs/2026-08-13-external-thinking.md) | staging | 外部思考规范 |

> staging 下文档定稿通过评审后，移入"架构与设计"并标记 active。

## 4. 审计 / 报告 / 快照（report）

| 文档 | 状态 | 说明 |
|---|---|---|
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