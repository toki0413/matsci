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

机器无关性由 `tests/test_arch_no_hardcoded_paths.py` 在同一 fast-fail 阶段强制：
git 跟踪的代码文件里出现 Windows 用户绝对路径（`C:\Users\...`）或黑名单机器
token（如 `wanzh`）都会让 CI 变红。机器相关路径必须改为 env 变量展开
（如 `workspace = "env:HUGINN_WORKSPACE"`）。

## 落地进展（2026-08-12）

制度层已到位；实现层按此推进，每完成一项更新本清单：

- [x] **导入门禁**：`tests/test_arch_single_gateway.py` 冻结外部直连清单并阻断新增，
      已接入 CI fast-fail。
- [x] **机器无关门禁**：`tests/test_arch_no_hardcoded_paths.py` 扫描 git 跟踪的代码
      文件，拦截 Windows 用户绝对路径（`C:\Users\...`）与黑名单机器 token（`wanzh`），
      已接入 CI fast-fail。配套把 `huginn.toml` 的 workspace 改为 `env:HUGINN_WORKSPACE`
      展开（`config.from_dict` 支持 `env:` 前缀），并 gitignore 掉
      `pyext/.cargo/config.toml`（含机器专用 Python 库绝对路径）。
- [x] **端点契约门禁**：`tests/test_arch_endpoint_contract.py` 从 huginn.server 的
      OpenAPI schema 提取 `/v1` 端点模板，与 CLI（`cli/src/http.rs`）引用的 `/v1` 路径
      双向校验 —— 后端路由改名/删除后，CLI 若仍引用旧路径会静默 404，本门禁在 CI
      fast-fail 阶段立即拦截，强制 CLI 与后端端点契约保持一致。
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
      - 已新增 `cli/src/http.rs`（`ureq` 阻塞客户端，`json` feature），复用 `backend_port`
        文件发现端口；采用「后端可达 → HTTP；后端未起 → Python spawn 兜底」策略。
      - **已迁移（HTTP 优先）**：`tools` → GET `/v1/tools`；`diagnose` → POST `/v1/diagnose`；
        `bench` → POST `/v1/bench/run`；`evolve` → POST `/v1/evolve/run`；
        `workflow` → POST `/v1/workflows/execute`（KEY=VALUE 参数解析）；
        `hpc` → POST `/v1/hpc/test|submit|status`；`execute` → POST `/v1/execute`
        （stages 支持文件路径或内联 JSON，解析后交给后端）；`explore` → POST `/v1/explore`；
        `coder` → POST `/v1/coder`（有 task 时走 HTTP，无 task 需交互输入则走本地）；
        `encrypt-config` → POST `/v1/config/encrypt`（加密后端活跃配置）。
        端到端验证：`huginn tools` 列出 130 工具、`huginn diagnose` 返回诊断 JSON、
        `huginn workflow` 正确执行模板、`huginn explore/execute/coder` 返回结构化 JSON、
        `huginn encrypt-config` 加密后端配置。
      - **交互式 SSE/WS**：`chat` → POST `/v1/agents/lead/chat/stream`（SSE 流式），
        CLI 内实现 REPL：读 stdin → 逐事件打印 token / 工具调用 / 思考 → 命中
        `done`/`error` 结束；后端不可达时回退 Python spawn。
        端到端验证：SSE 端点返回标准 `event:`/`data:` 事件，CLI 正确解析并打印。
      - **待迁移**：无（chat / explore / coder / execute / hpc / encrypt-config 均已迁移；
        仅剩交互式 chat 的 WS 双向增强与桌面鉴权旁路未做）。
      - 注：已迁移命令仍保留 Python spawn 兜底，故仍留在 `CLI_DELEGATED_SUBCOMMANDS`
        冻结清单；待彻底取消兜底后再从清单移除。
- [ ] **桌面免鉴权旁路移除**：Tauri 不再强制 `HUGINN_DEV_MODE=1`，桌面走 API key。

### 制度化闭环拓展（P2）

除"单网关"这条架构主线外，给 agent 补了供应链与密钥两道制度化闭环，均由
`ci.yml` 的 `deps-and-secrets` job 强制执行：

- [x] **依赖 lock 漂移门禁**：`agent/requirements.lock` 由 `uv pip compile` 生成
      （`--python-version 3.12 --no-strip-markers`，仅锁核心运行依赖；ML 互斥组
      ml-mace/ml-fairchem 因 e3nn 版本冲突刻意不纳入全量 lock）。CI 把 pyproject
      重编译到临时文件，按钉版行（`grep -v '^#'`，忽略 header/注释）与已提交 lock
      比对，不一致即红 —— pyproject 依赖变更后必须同步重生成 lock，保证生产依赖
      可复现、供应链可审计。交互式 `uv pip compile` 无 `--check`，故用「重编译→diff」。
- [x] **密钥扫描门禁**：gitleaks（`gitleaks detect`，`--redact`）扫全 git 历史，
      真实凭据直接 fail CI。`.gitleaks.toml` 用 `[extend] useDefault` 继承内置规则，
      并按"具体占位假值"豁免 `agent/tests/` 测试夹具里的假 key（AWS 官方示例
      `AKIAIOSFODNN7EXAMPLE`、模板化 OpenSSH 私钥、`wrong-key-12345`）——按值豁免
      意味着往那些文件塞真实 key 仍会被拦下。注：`secrets` allowlist 在本版 gitleaks
      有解析问题，改用 `regexes`。
- [x] **MCP `.mcp.json` 消费门禁**：lifespan 启动时消费仓库根 `.mcp.json` 的
      `mcpServers`（IDE/桌面共用那份标准配置），把外部 server（如 `flint-chart`，
      `npx -y flint-chart-mcp`）拉起并注册进 agent 工具池，而不只是 IDE 能连。
      best-effort：文件缺失/非法 JSON/单条配置非法都跳过，不阻塞启动。
      `tests/test_arch_mcp_json.py` 在 CI 固话该行为。