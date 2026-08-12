# Huginn 技术规格（现状事实记录）

> 本文件由 `praxis/es onboard` 生成，只记录代码中真实存在的事实，不含解释、计划或差距分析。目标：为已存在的代码库提供一份可维护的"living documentation"。以 `docs/architecture.md` 为历史参照，本文件反映当前目录结构的真实状态。

## stack

- 语言/运行时：Python `>=3.11,<4.0`（CI 覆盖 3.11/3.12/3.13）。
- Web 服务：FastAPI + uvicorn、sse-starlette、websockets、python-multipart、httpx、requests。
- Agent 框架：langchain `>=1.3`、langchain-core、langchain-openai、langgraph `>=1.2`、langgraph-checkpoint-sqlite（生产必需，见 `pyproject.toml` 注释）、deepagents。
- 前端/桌面：Tauri v2 + React 18（WIP），包名 `huginn-agent` v0.2.0，MIT。
- 数值/符号/数据：numpy、scipy、sympy、z3-solver、networkx、Pillow。
- 存储：SQLite + FTS5、ChromaDB（可选 extra `rag`/`all`）、toml（读写 TOML 配置）。
- 加密：cryptography（AES-128-CBC + HMAC-SHA256，PBKDF2）。
- 可选科学栈（extra）：pymatgen、ase、MACE/fairchem/NEP（互斥分组的 `ml-mace`/`ml-fairchem`/`ml-nep`，因 e3nn 版本冲突拆分）、spglib、matplotlib 等。
- 工具链：black（line-length 88）、ruff、mypy（渐进式）、pytest + xdist + cov + hypothesis、pre-commit。

## entry

- 命令行入口：console script `huginn-agent = huginn.cli:main`（`huginn/cli/main.py` 的 click group）。子命令注册于 `huginn/cli/commands/__init__.py` 的 `register_commands`，包括：`chat`、`coder`、`refactor`、`explore`、`autoloop`、`serve`、`tools`、`version`、`configure`、`bench`、`evolve`、`execute`、`workflow`、`diagnose`、`model-list`、`memory-maintenance`、`telemetry`、`seed-knowledge`、`encrypt-config`、`export`、`build-kg`、`api-keys`、`hpc`、`remote`、`scheduler`、`autoresearch`、`plot`、`unified`、`persona`、`swarm`/`team`、`visualize`、`sessions`、`skill-import`、`bg` 等。
- API 服务：`python -m huginn.server` → uvicorn 监听 `127.0.0.1:8000`，`ws_ping_interval=300`；端口写入运行时目录 `backend_port` 文件供前端发现。
- 包入口：`huginn/__init__.py` 导出 `__version__`。

## contract

- 工具注册表契约（`huginn/tools/registry.py`）：`ToolRegistry.register/get/list_tools/unregister/clear/snapshot/restore`；装饰器 `register_tool`。全局注册表维护在 `ToolRegistry._tools`。
- 工具基类契约（`huginn/tools/base.py`）：`HuginnTool` 暴露 `name/description/input_schema/output_schema`；标准入口 `call() -> ToolResult`（内含 provenance 捕获），子类实现 `_execute()`；可选覆写 `check_permissions/validate_input/estimate_cost`。
- 工具安全元数据（`huginn/tools/defaults.py`）：`ToolMetadata` 含 `is_read_only/is_destructive/requires_confirmation`，默认 fail-closed（不显式声明即为不安全）。
- 工具注册分层（`huginn/tools/__init__.py`）：`register_core_tools()`（约 35 个轻量工具，同步）与 `register_optional_tools()`（重依赖仿真/科学工具，建议后台）→ `register_all_tools()`。配置 `_CORE_MODULES` / `_OPTIONAL_MODULES` 元组列表。注册后重建 phase/router/constraint 调度表（`_rebuild_dispatch_tables`）。
- HTTP API 面（`huginn/server.py` + `huginn/routes/*`）：`include_v1_routes(app, keep_root_compat=True)` 将所有路由挂到 `/v1` 前缀，并保留根路径兼容（带弃用提醒）。全应用统一 `Depends(require_api_key)`、统一错误信封 `huginn_error_response`（含 `request_id`）。端点族（真实路径）：
  - 健康：`/health`、`/health/live`、`/health/ready`、`/ready`、`/health/guidance`、`/health/rust`
  - 会话线程：`/threads` CRUD、`/threads/{id}/messages|state|archive|unarchive|fork|branches|switch-branch`
  - 执行：`/execute`、`/explore`、`/diagnose`、`/plan`、`/workflows`、`/workflows/execute`
  - 多 agent：`/team/v2/members|plan|run|plans/*|fusion`、`/team/profiles|plan|run`
  - 认证：`/auth/login|token|me|refresh|logout`
  - 凭据：`/credentials` CRUD、`/credentials/{cid}/test|link-model/{alias}`、`/credentials/import-from-config`、`/credentials/{service}`
  - MCP：`/mcp/servers`、`/mcp/servers/discover|connect|{name}/disconnect|reconnect`、`/mcp/tools/{tool}/call`、`/mcp/connect/batch`、`/mcp/prompts`、`/mcp/resources/subscribe`、`/mcp/status`
  - 传输：`/transfer/upload|download|browse|sync`（require_admin_key）
  - 研究项目：`/projects` CRUD、`/projects/{pid}/threads|knowledge|run-research|research-status`
  - 导出导入：`/export/status|all|memory|knowledge`、`/import/all`
  - 事件/WS：`/events/stream`（SSE）、`/ws/agent`（WebSocket 实时对话）
  - 其他：`/data/dictionary*`、`/data/validate`、`/bench/run`、`/evolve/run`、`/autoloop/start|status`、`/live/execute`、`/live/capabilities`、`/system/components`、`/advisor/models|recommend|compare`、`/users/*`、`/projects/*`、`/maintenance`、`/firewall/status`、`/sandbox/execute`、`/analyze/symmetry|spectral|dynamics|tda|sindy`、`/viz/dos|phase|persistence|sindy`、`/project-context`
- COT 捕获契约（`huginn/agent/streaming.py`）：AIMessage 上的 `reasoning_content`（如 DeepSeek）必须经 `memory.add_reasoning(...)` 写入 `session.reasoning_trace`，供下游蒸馏消费。
- 共享状态后端契约（`huginn/server_core.py` + `huginn/persistence/state_store.py`）：`_threads`/`_checkpoints` 为 `MutableMapping`，默认进程内内存 dict（`HUGINN_STATE_BACKEND` 未设置）；设置 `HUGINN_STATE_BACKEND=sqlite` 时切换为 `SqliteStore`（SQLite WAL + busy_timeout，持久化到 `<runtime_home>/state.sqlite`）。`SqliteStore` 实现 MutableMapping 并缓存解码值，保持 `_threads[tid]["label"]=x` 等原地修改语义与 dict 等价；检查点值 `(Path, dict)` 经 `encode_checkpoint`/`decode_checkpoint` 编解码。
- 知识验证闭环（`huginn/evolution/knowledge_distiller.py` + `huginn/memory/manager.py`）：蒸馏知识 `verification_status ∈ {unverified, confirmed, rejected}`；`verify_knowledge(knowledge_id, status)` 将成功使用过的知识升级为 confirmed；`auto_ingest_to_kb` 仅吸收 `confirmed` 条目。`MemoryManager._verify_distilled_for_tool` 解析 `source="distiller:{id}"` 并调用 `verify_knowledge`。

## convention

- 测试目录：`tests/`，pytest `asyncio_mode=auto`，`--cov=huginn`，覆盖率门禁 60（CI 显式 `--cov-fail-under=60`）。`tests/conftest.py` 顶部 `collect_ignore_glob` 忽略 7 个脚本式测试文件。
- 测试夹具约定（`tests/conftest.py`）：
  - session 级 `shared_huginn_app`：单例 FastAPI app，构建约 2–3GB，每个 xdist worker 只建一次。
  - module 级 `app_client`：`with TestClient(...)` 上下文管理器，模块结束时关闭 transport 并跑 lifespan 关闭，杜绝 anyio portal 线程泄漏与内存累积。
  - autouse `_clear_config_cache_between_tests`：清 config 缓存与路径覆盖，隔离认证状态。
  - session autouse `_canonical_tool_registry`：会话启动时填充一次规范注册表，作恢复护栏基线。
  - autouse `_restore_tool_registry`：每测试前后对比 `ToolRegistry.snapshot()`，不一致即红名报错并回滚，拦截全局注册表状态泄漏。
  - hook：CI runner 偶发 `sqlite3 disk I/O / database is locked` 判定为环境问题并 skip。
- 工具编码约定：每个工具一个类文件置于 `huginn/tools/`，继承 `HuginnTool`，在 `huginn/tools/__init__.py` 的模块列表登记；`ToolResult` 作为所有工具统一返回结构。
- 风格：ruff select `E,F,W,I,N,UP,B,C4,SIM`（ignore `E501,B008,SIM108,N803,N806,B905`）；black 88 列；mypy 渐进式（`disallow_untyped_defs=false`，`warn_redundant_casts`/`warn_unused_ignores` 开启）。
- 新增工具流程（README）：继承 `HuginnTool` → 定义 `name/description/input_schema`（Pydantic）→ 实现 `_execute`/`call` → 登记 → 测试。

## invariant

- 全局 `ToolRegistry` 在任意测试之后必须与运行前逐位一致（autouse 护栏强制）。
- `TestClient` 必须通过上下文管理器使用（`test_testclient_hygiene.py` 强制扫描所有测试文件）。
- 每个 API 端点都受 `require_api_key` 保护；错误响应统一走 `huginn_error_response` 信封（含 `request_id`）。
- 加密密钥仅存内存、绝不落盘；静态数据 AES-128-CBC + per-item salt。
- 模型暴露的 `reasoning_content` 必须落入 `session.reasoning_trace`，蒸馏/进化才能读到。
- 成功使用的蒸馏知识必须可升级为 `confirmed` 才能进入知识库（`auto_ingest_to_kb` 门槛）。

## constraint

- Python 版本硬约束 `>=3.11,<4.0`。
- `mcp` 锁定 `>=1.28.1,<2.0`（2.0 为重大 API 重构，暂不升级）。
- `chromadb` 存在 PYSEC-2026-311（预认证代码注入，仅影响 HTTP API + `trust_remote_code=true`）；huginn 用本地嵌入式 `PersistentClient`，不暴露 HTTP，不受影响，无 fix 版本。
- ML 势能依赖互斥：`ml-mace`（e3nn==0.4.4）与 `ml-fairchem`（e3nn>=0.5）必须分装，避免 uv 解析失败。
- 默认限流 120 req/min/IP，认证端点 10 req/min/IP；`HUGINN_RATE_LIMIT_PER_MINUTE=0` 关闭。
- 请求体上限默认 10MB（`HUGINN_MAX_BODY_SIZE_MB`），请求超时默认 180s（`HUGINN_REQUEST_TIMEOUT_SEC`）。
- 生产环境隐藏交互式文档（`HUGINN_HIDE_DOCS` 或 `HUGINN_ENV=production`）。
- WebSocket ping 超时 300s（适配 DeepSeek reasoner 长推理）。
- 进程内 `~/.huginn` 写入在测试中重定向到 per-xdist-worker 的 `.test_cache/<worker>` 目录，规避 SQLite 锁冲突。