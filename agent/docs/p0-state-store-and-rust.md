# P0 里程碑：共享状态后端化 + Rust 沙箱确定性

> 状态：**已完成**（T1–T5 全部落地，由 `staging/plans/2026-08-12-p0-state-store-and-rust.md` 提升，
> 保留为里程碑记录）。现状事实见 [tech-spec.md](tech-spec.md)；差距依据见
> [industrialization-gap-analysis.md](../polish-reports/industrialization-gap-analysis.md)（P0 两项）。
> 验收以命令/测试为准，不写步骤与精确代码。

## 背景事实（不重复 spec）

- `huginn_ext` Rust 扩展源码在 `/workspace/pyext`（cargo workspace `/workspace/Cargo.toml` 成员，maturin 构建为 `huginn_ext` 模块）。但 `agent/pyproject.toml` 不声明其依赖，CI `release.yml` 从不构建/发布它 → 发布链路与 Rust 扩展**脱钩**，运行期默认不可达（`bash_tool.py` 用 try/except 动态导入，默认无编译产物）。因此"静默崩溃"难以在默认环境复现。P0-1 方向：复现→修根因→接入构建/发布链路→验证后默认启用。
- `langgraph-checkpoint-sqlite>=2.0.0` 已是生产依赖；`huginn/persistence/checkpointer.py` 已有 `SQLiteCheckpointerBackend` 抽象，可复用为共享状态底座。
- `_threads`/`_checkpoints` 被约 10 个模块直接以 dict 方式访问（`routes/threads.py`、`routes/checkpoints.py`、`routes/search.py`、`routes/ws_helpers.py`、`export_share.py`、`cli/slash_commands.py`、`server.py` `_DELEGATED_SC`、`ws.py`）。迁移须保持存取接口不变，故引入 `MutableMapping` 风格 store。

## 任务

### P0-1 Rust 沙箱确定性：复现修复 + 接入发布链路

- [x] T1: 复现 Rust sandbox 静默崩溃，修根因，接入构建/发布链路
```
goal:       消除 RDKit+sklearn GPR 下 Rust sandbox 静默崩溃（空 stderr），使
             huginn_ext 可复现、可修复、随发布产物交付
files:      pyext/src/sandbox.rs, huginn/tools/bash_tool.py, huginn/tools/code_tool.py,
            agent/pyproject.toml, .github/workflows/release.yml, ROADMAP.md, 相关测试
acceptance: 本地 cargo build/maturin 构建 huginn_ext 成功后，在 RDKit+sklearn GPR
             场景复现出崩溃并给出根因；修复后该场景返回可读错误或正常结果；
             CI 增加 pyext 构建与单元测试 job；发布产物（wheel/sdist）包含
             huginn_ext 或作为独立 wheel 发布；验证通过前保持 HUGINN_USE_RUST_SANDBOX
             默认关闭
status:     已通过。pyext/src/sandbox.rs 新增 decode_exit_status 捕获子进程 signal
             （崩溃命令返回 rc=-11 + "killed by signal 11" 而非空 stderr + -1），
             子模块注册进 sys.modules 消除 ModuleNotFoundError；release.yml 新增
             build-rust-ext job（cargo test + maturin wheel 多 python 版本上传
             Release）；pyproject 新增 [project.optional-dependencies.rust]。
spec:       ../polish-reports/industrialization-gap-analysis.md  #P0-1
```

### P0-2 共享状态后端化

- [x] T2: 引入 `ThreadStore`（SQLite 持久化的 MutableMapping）替换内存 `_threads`
```
goal:       会话元数据可跨进程/重启持久，多 worker 一致，消除"请求漂移丢会话"
files:      huginn/persistence/state_store.py, huginn/server_core.py, tests/test_state_store.py
acceptance: SqliteStore("huginn_threads") 实现 MutableMapping 且基于 SQLite（WAL）；
             get_or_create_thread/touch_thread/列列举经 store；重启后既有会话仍在；
             threads 路由测试全绿（MutableMapping 保持原地修改语义）
spec:       ../polish-reports/industrialization-gap-analysis.md  #P0-2
```

- [x] T3: 引入 `CheckpointStore`（SQLite 持久化的 MutableMapping）替换内存 `_checkpoints`
```
goal:       检查点快照可跨进程/重启持久，多 worker 一致
files:      huginn/persistence/state_store.py, huginn/server_core.py,
            huginn/routes/checkpoints.py, huginn/routes/ws_helpers.py,
            huginn/cli/slash_commands.py, tests/test_state_store.py
acceptance: SqliteStore("huginn_checkpoints", encode=encode_checkpoint) 实现
             MutableMapping 且基于 SQLite（WAL）；cp 增删改查/回滚经 store；
             重启后检查点可恢复（encode/decode 往返）；checkpoints/undo 相关测试全绿
spec:       ../polish-reports/industrialization-gap-analysis.md  #P0-2
```

- [x] T4: 状态后端开关 `HUGINN_STATE_BACKEND=memory|sqlite` + 多进程一致性验证
```
goal:       默认 memory 保持单进程行为不变；显式 sqlite 时多 worker 共享同一库一致
files:      huginn/server_core.py, DEPLOYMENT.md, tests/test_state_store.py,
            tests/test_multiprocess_state.py
acceptance: 开关切换后线程/检查点行为等价；多 worker 并发写同一 sqlite 无
             "database is locked"（WAL + busy_timeout）；tests/test_multiprocess_
             state.py 自动化覆盖两种后端（memory 默认 dict + sqlite SqliteStore）
spec:       ../polish-reports/industrialization-gap-analysis.md  #P0-2
```

### 收尾

- [x] T5: 持久化冒烟 + 文档更新
```
goal:       P0 两项落地后，单机多 worker 状态一致可复现，文档如实
files:      DEPLOYMENT.md, tech-spec.md, tests/test_state_persistence_smoke.py
acceptance: 一处线程/检查点跨"重启 + 多 worker"的端到端冒烟通过；tech-spec
             contract/convention 与代码一致
status:     已通过。端到端冒烟 tests/test_state_persistence_smoke.py：writer 进程
             get_or_create_thread + _checkpoints 写线程/检查点，独立 reader 子进程
             （重启）读回，断言 label/snapshot 一致。路由冒烟 test_route_smoke.py +
             test_server_endpoints.py 37 过（唯一失败为缺 matplotlib 的 all extra，
             与本次改动无关）。文档：DEPLOYMENT.md 多 worker 状态共享 + tech-spec.md
             状态后端契约。仓库无 CHANGELOG.md，未新建。
spec:       ../polish-reports/industrialization-gap-analysis.md  #结论
```

## 并行性

- `[parallel] T2, T3`：两个 store 相互独立，共享契约（MutableMapping + SQLite WAL + accessor 接口）已闭合。
- T1 与 T2/T3 无依赖，可并行；T4 依赖 T2+T3；T5 依赖全部。

## 边界（不在本里程碑）

- `_context`（ServerContext 持有 agent/factory/memory 等活对象）为进程内单例，不纳入本里程碑；无状态化改造、会话亲和路由、Redis 后端、跨实例 HA 列为 P1。
- 不改动 `runtime/checkpoint.py`（磁盘 `.huginn_checkpoints` 的 LangGraph checkpointer 体系），本里程碑只迁移 `server_core._checkpoints` 内存快照。