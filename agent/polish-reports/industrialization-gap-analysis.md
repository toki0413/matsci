# Huginn 产业化落地差距评估

> 评估对象：`huginn-agent` v0.2.0（材料科学 LLM agent）。
> 方法：以 `docs/tech-spec.md`（现状事实）为基线，对照"可稳定运行"→"生产 MVP（单租户）"→"商业 SaaS（多租户 + HA）"三个成熟度档位，逐维度打分（1-5）。所有判断基于代码/文档已存在的证据，不臆测。

## Executive Summary

**总体成熟度：约 3.0 / 5** —— 已具备生产级代码质量底盘（安全、测试、SBOM、错误信封、审计），核心短板集中在**可扩展性（单进程内存态）、可观测性（默认内存遥测）、多租户**与**产品化前端**。当前可用于内部单租户/科研部署，但**尚不能直接支撑对外商业 SaaS 或高并发生产**。

最大的三个产业化阻断项：
1. **共享状态为进程内内存态**（`server_core._context/_threads/_checkpoints`）→ 多 worker/多实例水平扩展会撕裂状态，无法 HA。
2. **Rust sandbox 静默崩溃（P0）**→ 已在 ROADMAP 标注，当前默认关闭，是生产正确性隐患。
3. **遥测默认内存态、/metrics 默认公开**→ 生产可观测性与安全收紧均未完成。

---

## 逐维度评估

### 1. 部署与打包（3.5/5）
**已有**：`DEPLOYMENT.md`（Docker/compose/nginx/Caddy/健康检查/Prometheus 规则/Vault 密钥）；`release.yml` 打 wheel/sdist + **cyclonedx SBOM** + GitHub Release；`build-wheels` 覆盖 3.10–3.13；`Dockerfile`。
**缺口**：
- 无镜像仓库推送（GHCR/DockerHub）与镜像签名（cosign）。
- 无环境差异化配置治理（dev/staging/prod 的 config 模板化、密钥注入编排）。
- 无滚动发布/回滚策略文档。

### 2. 可扩展性与高可用（1.5/5）★ 关键短板
**现状**：`server.py` 明确"所有共享状态在 `huginn.server_core`"；`_context/_checkpoints/_threads` 为进程内模块态。`_threads` 用内存 `defaultdict`，`_checkpoints` 为内存存储。
**缺口**：
- **多 worker（`uvicorn --workers 4`）会各持一份内存态，请求漂移即丢会话/线程上下文** → 无法水平扩展。
- 无分布式状态后端（Redis/Postgres/DB-backed checkpointer、thread store）。
- 无会话亲和 / 有状态会话路由策略。
- 无多实例部署拓扑（单个 uvicorn bump 到多副本会破坏一致）。

### 3. 可观测性（2.5/5）
**已有**：`/metrics` Prometheus 指标（请求/工具耗时/沙箱/内存/审计计数）；`monitoring/alerts.yml`；统一错误信封含 `request_id`；`events/audit_log.py`（append-only + hash-chained）；`telemetry.py`。
**缺口**：
- **遥测默认内存态，不跨重启持久**（SECURITY.md 自述）；OpenTelemetry exporter 仅文档提及，未确认接线。
- 无结构化分布式追踪（trace context 仅 `runtime/trace_context.py` 存在，链路贯通度未验证）。
- 无集中式日志聚合（无 JSON 日志到 Loki/ELK 的配置）。
- `/metrics` **默认公开**（DEPLOYMENT 自述靠反代收紧），生产需强制鉴权。
- 无 SLO/SLI 定义与错误预算。

### 4. 安全与隔离（3.5/5）
**已有**：全端点 `require_api_key` + 管理 `require_admin_key`；限流（120/min，认证 10/min）；请求体大小/超时限制；fail-closed 工具元数据；AES-128-CBC + 内存仅存密钥；容器沙箱（docker/podman/apptainer）；审计 hash 链；`pip-audit` + SBOM；`HUGINN_HIDE_DOCS`。
**缺口**：
- **Rust sandbox 静默崩溃（P0，ROADMAP 自述）**：RDKit+sklearn GPR 场景返回空 stderr → "Unknown error"，默认关闭。生产启用前必须定位并验证。
- 本地 `bash_tool/code_tool` fallback 在未配置容器时可达，需强制默认容器且去 fallback。
- 无第三方依赖漏洞的持续扫描（CI 未接线 automated dependency CVE 门禁，仅 release 时 SBOM）。
- 密钥管理：Vault backend 在 DEPLOYMENT 提及，需确认实现与 fallback 风险。

### 5. 数据持久化与一致性（2.5/5）
**已有**：长期记忆 SQLite+FTS5；`utils/migrations.py`；checkpoint（`langgraph-checkpoint-sqlite`）；`persistence/`（checkpointer/state_registry）；知识库入库管道。
**缺口**：
- **autoloop 假设事件 log 在内存、假设库用 markdown append（未上 SQLite）**（ROADMAP P1）→ 长时自主探索无持久任务状态。
- 会话快照只读最新一条、无版本化（ROADMAP P2）。
- 多进程写 SQLite 的锁竞争与迁移治理未验证（conftest 用 per-worker 目录规避而非根治）。

### 6. 多租户 / 多用户（2/5）
**已有**：`/users` 端点、`/auth/login+token+refresh`、凭据管理（`/credentials`）、persona 系统。
**缺口**：
- 未确认租户级数据隔离（threads/memory/KB 是否按 tenant 分域）。
- 细粒度 RBAC/角色（operator/researcher/admin）未确认。
- 无租户配额（工具调用、token、并发、存储上限）。
- 无审计按租户分账/计费数据模型。

### 7. 测试与质量门禁（4/5）★ 强项
**已有**：CI 覆盖 3.11/3.12/3.13 + lint/rust-check/security-check/frontend-build + Desktop Build；覆盖率门禁 60；反回归护栏（`ToolRegistry` 逐位一致 + `TestClient` 上下文管理器强制）；`test_testclient_hygiene.py`；磁盘 I/O flaky 自动 skip。
**缺口**：
- 覆盖率 60 偏低（生产建议 ≥80，尤其安全/支付/核心编排路径）。
- 无端到端生产镜像冒烟（部署后真实 LLM + 仿真工具链 E2E）。
- 无性能/负载基准门禁（`test_performance_load.py` 存在但未作为 CI 硬门禁）。

### 8. 文档（4/5）★ 本次已对齐
**已有**：`docs/tech-spec.md`（现状事实）、`docs/architecture.md`（已重写）、`README.md`（已对齐）、`DEPLOYMENT.md`、`SECURITY.md`。
**缺口**：无 API 消费者指南/快速上手样例；无运维 Runbook（故障排查手册完整版）。

### 9. 产品化前端 / UI（1.5/5）★ 短板
**已有**：Tauri v2 + React 18（WIP）；`/ws/agent` 实时通道；`side_conversation.py`。
**缺口**：无可用生产 UI；无登录/租户工作台；无任务可视化/知识浏览界面。

### 10. 仿真工具链与 HPC 集成（3/5）
**已有**：VASP/QE/CP2K/Gaussian/ORCA/LAMMPS/GROMACS/OpenMM/OpenFOAM/COMSOL/ABAQUS/FEniCS 等工具；未找到可执行时降级为"导出输入文件"；`hpc/` 远程执行（client/connection_pool/resource_selector）。
**缺口**：核心工具 VASP/COMSOL/ABAQUS 为专有软件 → 生产需授权 + 许可管理；HPC 调度器适配广度与作业配额未确认；仿真可复现性（版本锁、容器化仿真环境）未覆盖。

### 11. 合规 / 供应链 / 许可（3/5）
**已有**：MIT 许可；cyclonedx SBOM；`SECURITY.md` 响应策略（48h 确认 / 7 天修复）。
**缺口**：三方依赖许可证清单与合规审查（算法/法规）；专有仿真工具再分发许可；数据合规（科研数据隐私/加密存储已部分覆盖，缺 DPA/数据出境）。

---

## 优先级路线图

### P0（阻断生产正确性/可用性）
1. **修复 Rust sandbox 静默崩溃**（RDKit/sklearn GPR 空 stderr）并默认启用；或彻底移除 Rust 路径、稳走 Python。
2. **共享状态后端化**：`_threads/_checkpoints/_context` 从进程内存迁移到 Redis/DB，达成多 worker 一致与 HA。

### P1（生产 MVP 必需）
3. 遥测切 OpenTelemetry + 集中日志；`/metrics` 强制鉴权。
4. autoloop 假设事件/库持久化到 SQLite（ROADMAP P1）。
5. 多租户数据隔离 + 基础 RBAC + 租户配额。
6. Docker 镜像推送 + 签名 + 生产镜像 E2E 冒烟测试。
7. 覆盖率门禁提升至 70+，性能基准作为 CI 门禁。

### P2（商业 SaaS / 产品化）
8. 生产 UI 工作台（登录 → 任务 → 知识 → 结果可视化）。
9. 依赖许可证合规审查 + 专有仿真工具许可编排。
10. SLO/错误预算 + 告警到值班链路；运维 Runbook。
11. 租户计费/用量数据模型。

---

## 结论

- **当前定位**：可稳定运行的单租户科研/内部部署工具，代码工程底盘（安全/测试/文档）已接近生产标准。
- **到"生产 MVP"**：完成 P0（Rust 崩溃 + 状态后端化）+ P1 的遥测/持久化/镜像，即可支撑单租户对外试用。
- **到"商业 SaaS"**：还需多租户隔离、生产 UI、SLO/计费、合规审查（P1 后半 + P2）。
- 建议第一步聚焦 **P0 两项**——它们是"代码能跑但不敢上生产"的直接原因，与之前"一劳永逸"的稳定性目标同源。