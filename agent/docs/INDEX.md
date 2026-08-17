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
| [DESIGN_RATIONALE.md](DESIGN_RATIONALE.md) | active | 设计论据：全局 how+why、设计原则、诚实边界、数学层与朗兰兹精神的定调 |
| [HOW_TO_READ.md](HOW_TO_READ.md) | active | 新人上手导览：目录地图、主线走通、改哪找哪、文档真相层级 |
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
| [pluginized-segments-design.md](pluginized-segments-design.md) | active | 插件化分段设计（Everything is a Plugin 落地） |
| [external-thinking.md](external-thinking.md) | active | deep_think 外部草稿纸工具（已实现，含 `external_thinking` 开关） |
| [p0-state-store-and-rust.md](p0-state-store-and-rust.md) | active | P0 里程碑记录（共享状态后端化 + Rust 沙箱确定性，已完成） |
| [reward_design.md](reward_design.md) | staging | 奖励设计（未实现理论稿） |
| [cost-participation-contract.md](cost-participation-contract.md) | active | 成本-剪枝参与感契约（决策点对话 + 成本叙事，已实现） |
| [SPEC_openworker_adoption](../huginn/SPEC_openworker_adoption.md) | active | OpenWorker 采纳规范 |
| [SPEC_visual_kb_loop](../huginn/metacog/SPEC_visual_kb_loop.md) | active | 视觉知识库闭环规范 |

已合并入 [tech-spec.md](tech-spec.md)，不再单独列出的 spec：`async_dispatch_spec`、`layered_memory_spec`、`lsp_hashline_spec`、`memory_dispatch_integration_spec`、`enhanced_modules`、`reinforcement_event_sourcing_sandbox_incremental_ui`、`ising_crdt_p1_spec`、`hils_active_inference_p2_spec`。

## 3. 审计 / 报告 / 快照（report）

| 文档 | 状态 | 说明 |
|---|---|---|
| [RELEASE_REPORT_v0.2.0.md](../RELEASE_REPORT_v0.2.0.md) | report | v0.2.0 发布报告 |
| [research-notes/physical-rsi-and-world-model-interpretability.md](research-notes/physical-rsi-and-world-model-interpretability.md) | report | 研究备忘：Physical RSI 与视频世界模型可解释性对 agent 的启发 |
| [research-notes/attractor-identifiability-limits-system-discovery.md](research-notes/attractor-identifiability-limits-system-discovery.md) | report | 研究备忘：吸引子几何决定系统发现辨识上限（λ_min(M)）的启发与落地（identification/validation.identifiability） |
| [research-notes/metacog-de-islanding-audit.md](research-notes/metacog-de-islanding-audit.md) | report | 独立审计：metacog 头接线核查 + blind_spot_mapper 去孤岛（per-skill SelfModel 升级路径/触发条件） |
| [research-notes/third-party-audit-final.md](research-notes/third-party-audit-final.md) | report | 第三方独立综合审计：security-auditor × loop-polish preflight × praxis review（OWASP/BLOCK-FIX-NIT） |
| [polish-reports/loop-polish-report.md](../polish-reports/loop-polish-report.md) | report | 循环打磨报告 |
| [polish-reports/industrialization-gap-analysis.md](../polish-reports/industrialization-gap-analysis.md) | report | 工业化缺口分析 |
| [polish-reports/benchmark-vs-claude-codex-evoscientist.md](../polish-reports/benchmark-vs-claude-codex-evoscientist.md) | report | 对标报告（Claude/Codex/Evoscientist） |
| [analysis_20260717/](../../analysis_20260717/) | report | 历史归因/对标分析快照（17 篇，入口见 `00_综合归因报告.md`） |

## 4. 部署与运维

| 文档 | 状态 | 用途 |
|---|---|---|
| [DEPLOYMENT.md](../DEPLOYMENT.md) | active | 部署指南 |
| [MONITORING.md](../MONITORING.md) | active | 监控指南 |
| [e2e_deployment_checklist.md](../tests/e2e_deployment_checklist.md) | active | 端到端部署检查清单 |

## 5. 根级文档与其他组件

| 文档 | 状态 | 用途 |
|---|---|---|
| [快速上手分步指南](../../docs/quickstart.md) | active | 根级安装/配置/验证/聊天分步指南 |
| [威胁模型](../../docs/threat_model.md) | active | 攻击面 / STRIDE / 信任边界 / 事件响应 |
| [ToolUniverse 集成](../../docs/tooluniverse-integration.md) | active | 工具生态集成方案 |
| [ADR-0001 单网关](../../docs/architecture/decisions/0001-single-gateway.md) | active | 架构决策记录：唯一业务网关原则 |
| [agent 包 README](../README.md) | active | agent 子包特性与开发 |
| [根 README](../../README.md) | active | 全项目总览、两条控制轴、文档导航 |
| [desktop README](../../desktop/README.md) | active | 桌面应用（Tauri v2 + React）开发/构建 |
| [Rust CLI README](../../cli/README.md) | active | `huginn` 二进制前端：子命令 / 单网关 / 构建 |
| [Sidecar README](../../sidecar/README.md) | active | `huginn-sidecar` 进程管理与事件总线 |
| [MCP 服务器 README](../../servers/README.md) | active | 3 个 MCP 服务器总览（mat-db / math-anything / vision-pixel） |
| [plan.md](../plan.md) | active | 根级开发计划 |
| [CODE_WIKI.md](../../CODE_WIKI.md) | active | 代码库百科 |
| [CHANGELOG.md](../../CHANGELOG.md) | active | 变更日志 |
| [RELEASE_REPORT_v0.2.0.md](../RELEASE_REPORT_v0.2.0.md) | report | 历史发布报告（v0.2.0，当前版本已更新） |

> 组件目录（`cli/`、`sidecar/`、`servers/`、`desktop/`）各自带 README；
> 本索引统一登记，避免散落不可发现。

> 根 README 同时是"文档导航"的另一个入口，与本索引互为补充（本索引更细粒度）。

---

## 维护约定

- 新增/废弃文档请更新本索引，并标注状态。
- 废弃文档不直接删除，改标 `report` 并注明"已废弃，由 <新文档> 替代"。
- 本索引本身是导航，不替代各文档正文。