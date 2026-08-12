# 产业化对标：Claude Code / Codex / EvoScientist 查漏补缺

> 目标：把 `huginn-agent` 从"可跑的单租户科研工具"推到"可真实产业化的产品"。
> 方法：逐一对照三家对标产品的能力画像，标注 Huginn `已有` / `缺口` / `对标动作`。
> Huginn 现状依据 `docs/tech-spec.md` 与源码；对标产品为公开能力画像（一般知识，非虚构）。

## 对标对象能力画像

| 产品 | 定位 | 定义性能力 |
|---|---|---|
| **Claude Code** | 终端原生编码 agent | 终端 CLI、MCP、子代理、hooks、权限/审批、checkpoint+resume、会话分支、plan mode、Git 集成、skills/插件市场、云同步、IDE 集成 |
| **Codex** | 编码 agent（CLI+云） | 代码评审、沙箱执行、MCP、GitHub Actions 集成、并行任务、云端长任务、本地+云端双模式 |
| **EvoScientist** | 自主材料/科学 agent | 假设生成→实验→分析闭环、实验室/仿真编排、知识库累积、自我改进、多假设并行探索 |

## 逐维度对比

### 1. 执行形态与入口
- **Claude Code**：终端 CLI + IDE 内嵌 + 云。
- **Codex**：CLI + 云 web + API。
- **EvoScientist**：headless 自主循环 + 日志/报告输出。
- **Huginn 现状**：CLI（`huginn-agent` 30+ 子命令）、FastAPI server、`/ws/agent` 实时、Tauri v2+React 桌面（WIP）。
- **缺口**：无可用生产 UI 工作台；无 IDE 插件；云端/托管形态未定。
- **对标动作**：P2 先做浏览器端工作台（登录→任务→知识→结果），Serve 为对外 API 服务。

### 2. 自主研究闭环（EvoScientist 核心）
- **EvoScientist**：假设→实验→分析→再假设，全自动闭环，多假设并行。
- **Huginn 现状**：`autoloop`（plan_store/assumption）、`autoresearch`、`/autoresearch`、`hypothesis_generator_tool`、`active_learning`、`high_throughput`、`multi_fidelity`、`/team` 多 agent fusion。
- **缺口**：**autoloop 假设事件/库在内存、markdown append，未上 SQLite**（ROADMAP P1）→ 长时自主探索无持久任务状态，无法长时间无人值守。
- **对标动作**：P1 把 autoloop 任务状态持久化；这是对标 EvoScientist"连续自主"的关键。

### 3. 工具 / MCP / 沙箱
- **Claude Code/Codex**：MCP 生态、沙箱执行、权限分级。
- **Huginn 现状**：约 178 工具、MCP 服务端（`/mcp/*`）、容器沙箱（docker/podman/apptainer）、Rust sandbox（**默认关闭、静默崩溃，P0**）、本地 bash/code fallback。
- **缺口**：**Rust 沙箱崩溃（P0）**；本地 fallback 在未配置容器时可达，需强制默认容器。
- **对标动作**：P0 移除/禁用 Rust 快路径；P1 强制容器默认、去 fallback。

### 4. 多 agent 编排
- **EvoScientist**：多假设并行、评审委员会。
- **Huginn 现状**：`Orchestrator`、`/team/v2/members|plan|run|fusion`、`/team/profiles`、`review_committee_tool`、`subagent_tool`。
- **缺口**：多 agent 任务的分散状态亦依赖共享状态后端化（P0-2）。
- **对标动作**：随 P0-2 一并获得一致性。

### 5. 记忆 / 知识 / 进化
- **Claude Code**：会话记忆、`.claude/` 配置、skills 复用。
- **EvoScientist**：知识库累积、自我改进。
- **Huginn 现状**：长期记忆 SQLite+FTS5、ChromaDB 向量、知识库入库管道、知识验证闭环（unverified/confirmed）、`evolution/`、`evolve` 命令、`seed-knowledge`。
- **缺口**：知识库/记忆**未按租户分域**（多租户 P1）；skills/插件市场形态未产品化。
- **对标动作**：P1 多租户数据分域；P2 知识/技能可导出分享、形成可复用资产库。

### 6. 会话生命周期与可恢复性（Claude Code checkpoint/resume）
- **Claude Code**：checkpoint + resume、会话分支/切换。
- **Huginn 现状**：`/threads` CRUD + fork/branches/switch-branch、内存 `_checkpoints` 快照、langgraph-checkpoint-sqlite。
- **缺口**：**线程/检查点为进程内内存态（P0-2）**，重启/多 worker 即丢；会话快照只读最新一条、无版本化（P2）。
- **对标动作**：P0-2 后端化即为"可恢复会话"的产业化基础。

### 7. 权限 / 审批 / 安全
- **Claude Code**：细粒度权限、审批流、deny/allow 规则。
- **Codex**：沙箱隔离、评审门禁。
- **Huginn 现状**：全端点 `require_api_key` + `require_admin_key`、JWT+RBAC、写操作 capability 强制、fail-closed 工具元数据、AES-128-CBC 密钥内存态、审计 hash 链、限流。
- **缺口**：RBAC 角色粒度（operator/researcher/admin）与租户配额未落地；`/metrics` 默认公开需收紧。
- **对标动作**：P1 强制 `/metrics` 鉴权 + 租户配额；细粒度角色。

### 8. 可观测性 / 追踪
- **Claude Code/Codex**：结构化追踪、日志聚合、云侧观测。
- **Huginn 现状**：`/metrics` Prometheus、错误信封 `request_id`、审计日志、`telemetry.py`、`runtime/trace_context.py`。
- **缺口**：遥测默认内存态不跨重启；无 OpenTelemetry/集中日志；trace 链路贯通度未验证。
- **对标动作**：P1 切 OTel + 集中日志、`/metrics` 鉴权。

### 9. 打包 / 分发 / IDE 集成
- **Claude Code**：安装即用、IDE 插件、云同步。
- **Codex**：CLI + GitHub Actions 内嵌。
- **Huginn 现状**：wheel/sdist、cyclonedx SBOM、GitHub Release、Dockerfile、Desktop Build（Tauri WIP）。
- **缺口**：无镜像仓库推送/签名；无 IDE 插件；无安装引导/升级通道。
- **对标动作**：P1 镜像推送+签名；P2 一键安装 + 升级通道。

### 10. 多租户 / SaaS
- **Codex**：云账号体系、用量、配额。
- **Huginn 现状**：`/users`、JWT+RBAC、人均会话隔离雏形。
- **缺口**：租户级数据隔离、配额、计费数据模型未落地。
- **对标动作**：P1 隔离+配额；P2 计费/用量。

### 11. 云 / 长任务
- **Codex**：云端长任务、异步执行。
- **Huginn 现状**：`/autoloop/start|status`、异步任务、`scheduler`、`bg` 命令。
- **缺口**：依赖 P0-2 状态后端化 + P1 autoloop 持久化才可支撑云侧长任务。
- **对标动作**：随 P0/P1 达成。

### 12. 插件 / 技能复用
- **Claude Code**：skills/插件市场。
- **Huginn 现状**：`skill_tool`、`skill-import`、`plugins/science_skills_bridge`、`/mcp/*`。
- **缺口**：无对外技能市场/分享体系。
- **对标动作**：P2 技能资产复用与导出。

## 结论：差距优先级映射

| 等级 | 对标动作 | 关联竞品 |
|---|---|---|
| **P0** | Rust 沙箱确定性；共享状态后端化 | Codex 沙箱 / Claude Code 可恢复会话 |
| **P1** | autoloop 持久化；OTel+集中日志；/metrics 鉴权；多租户隔离+配额；镜像推送+签名；覆盖率+性能门禁 | EvoScientist / Codex / Claude Code |
| **P2** | 生产 UI 工作台；IDE 插件；技能/知识资产库；SLO/错误预算；计费/用量；合规审查 | Claude Code / Codex |

**一句话定位**：比拼 EvoScientist = 先把**自主研究闭环持久化**（P1 autoloop + P0 状态后端）做扎实；比拼 Claude Code/Codex = 补齐**生产 UI、可观测、多租户、分发通道**（P1/P2）。P0 两项是"敢不敢上生产"的第一道闸门，先过闸。