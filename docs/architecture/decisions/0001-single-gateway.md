# ADR-0001: 单网关架构原则（Single Gateway）

- 状态：Accepted
- 日期：2026-08-12
- 决策者：huginn 架构维护
- 相关：docs/architecture.md、.github/workflows/ci.yml、cli/、desktop/、sidecar/

## 背景

huginn 的业务能力（agent 循环、工具、技能、记忆、知识蒸馏、工作流、探索、HPC 等）
此前可被多个入口直接触达：

- `huginn.server`（FastAPI + WS/SSE，端口 8000）
- `huginn.cli`（Rust CLI 通过 `python -m huginn.cli` 子进程直接运行 Python CLI）
- `scripts/`、`examples/` 等直接 `import huginn.*` 业务模块
- desktop Tauri 壳直接做本地文件 I/O，绕过业务后端
- sidecar（端口 8001）与后端（8000）双 HTTP/WS 拓扑并存

这导致：同一套业务逻辑被多入口、多鉴权姿态、多配置/工作目录、多端口发现机制
暴露，行为各自为政，鉴权/审计易被绕过，测试与运维难以断言"系统怎样才一致"。

## 决策

**`huginn.server` 是唯一的业务网关。** 所有业务逻辑只有通过其后端 HTTP/WS API
才能被消费。具体约束：

1. **服务器层是业务模块的唯一 importer**。只有服务器引导代码
   （`huginn/server*.py`、`huginn/routes/**`、`huginn/api/**`）与显式引导入口
   可以 `import huginn.*` 业务模块并调用其内部 API。
2. **外部消费者都是 API 客户端**。`scripts/`、`examples/`、`servers/`、
   `desktop/src-tauri/`、`sidecar/`、`cli/` 不得直接 `import huginn.*` 业务模块，
   必须通过 `huginn.server` 的 HTTP/WS 端点访问能力。
3. **CLI 不再是第二前门**。`huginn serve` 负责启动后端；其余子命令
   （chat/explore/coder/execute/workflow/bench/evolve/hpc/...）作为 HTTP/WS
   客户端连接运行中的后端，不再 spawn `python -m huginn.cli` 子进程。
4. **文件 I/O 归口后端**。需要 provenance/审计/安全策略的文件读写由后端端点
   提供；Tauri 壳只保留进程管理与窗口职责。
5. **端口与鉴权单源**。后端监听端口由单一来源（`backend_port` 文件）决定；
   `huginn.server` 的鉴权姿态对 UI 与 CLI 一致（均为 API key / JWT），
   不设桌面免鉴权旁路。

## 理由

- 减少前门数量，使"系统一致性"可被断言（鉴权、配置、审计集中一处）。
- 让 `tests/test_arch_single_gateway.py` 这类自动门禁能机械地拦截新的旁路导入。
- 收敛 process 拓扑：后端进程 → 由 sidecar / Tauri 统一管理，避免多进程各自起。

## 后果

- **正面**：单一入口、统一鉴权/审计、CI 可强制分层、运维心智简单。
- **代价**：CLI 交互命令需依赖一个运行中的后端（离线单机场景需降级：自动拉起
  后端或明确报错）；一次性迁移成本（CLI 改 HTTP 客户端、scripts 改 API 调用、
  Tauri 文件 IO 归口）。

## 迁移路径

1. 先用 `tests/test_arch_single_gateway.py` 冻结并枚举现有旁路直连，阻断新增。
2. 逐步把冻结清单中的脚本迁移为 API 调用，清单只减不增。
3. CLI 从"spawn 子进程"改为"HTTP/WS 客户端"。
4. Tauri 文件 I/O / 端口 / 鉴权归口后端。

## 强制

由 `tests/test_arch_single_gateway.py` 在 CI（ci.yml test job 的 fast-fail 阶段）
强制执行。任何新增的"外部直连业务模块"或"CLI 子进程委托"都会让 CI 变红。

## 落地进展（2026-08-12）

制度层已到位；实现层按此推进，每完成一项更新本清单：

- [x] **导入门禁**：`tests/test_arch_single_gateway.py` 冻结外部直连清单并阻断新增，
      已接入 CI fast-fail。
- [x] **CLI 委托冻结**：同一门禁新增 `CLI_DELEGATED_SUBCOMMANDS` 冻结清单，扫描
      `cli/src/main.rs` 的 `delegate_to_python(...)`，只许缩、不许涨 —— 新子命令
      必须改成 HTTP/WS 客户端，禁止新增 `python -m huginn.cli` 子进程旁路。
- [x] **端口单源**：`get_backend_port` 优先读后端写入的 `backend_port` 文件
      （`desktop/src-tauri/src/main.rs`），不再默认 8000 导致端口文件读取成为死代码；
      `start_backend` 同样先认端口文件。前端 `syncBackendUrl` 优先走该命令。
- [x] **鉴权加固（dev-mode 仅本机豁免）**：`huginn.security.auth` 的 dev-mode 旁路
      拆成 `_dev_mode_exempt()`，只豁免 loopback（127.0.0.1/::1）请求；服务若绑定到
      非 loopback 接口，即使 `HUGINN_DEV_MODE=1` 也须鉴权。测试
      `tests/test_security_auth.py` 新增非本机 401 用例。
- [x] **文件 I/O 归口后端**：新增 `huginn/routes/fs.py`（`/v1/fs/cwd|list|read|write`），
      继承 Tauri 原有路径安全语义（敏感目录/其他用户 profile 拦截）；桌面
      `useWorkspace` 改走后端 `/v1/fs/*`，并移除 Tauri 的 `get_cwd/read_dir/read_file/write_file`
      命令行（Tauri 只保留进程管理与终端职责）。测试 `tests/test_fs_gateway.py`。
- [ ] **CLI 瘦身为 HTTP 客户端**：`huginn serve` 起后端，其余子命令连 HTTP/WS。
      - 已新增 `cli/src/http.rs`（`ureq` 阻塞客户端），复用 `backend_port` 文件发现端口；
        `huginn tools` 优先走后端 `/v1/tools`，后端起不来才回退 Python spawn。
      - 待迁移：chat / explore / coder / bench / evolve / execute / workflow /
        diagnose / hpc / encrypt-config（委托清单已冻结防新增）。
- [ ] **桌面免鉴权旁路移除**：Tauri 不再强制 `HUGINN_DEV_MODE=1`，桌面走 API key。