# P0 里程碑：共享状态后端化 + Rust 沙箱确定性

> 面向"真实产业化发布"的首个里程碑。现状事实见 `docs/tech-spec.md`；差距依据见 `polish-reports/industrialization-gap-analysis.md`（P0 两项）。
> 验收以命令/测试为准，不写步骤与精确代码。

## 背景事实（不重复 spec）

- `huginn_ext` Rust 扩展源码**不在本仓库**（无 `.rs`/`maturin` 配置），仅 `from huginn_ext.sandbox import run_sandboxed` 动态导入。P0-1 务实方案：移除/禁用快路径，统一走 Python，彻底消除静默崩溃隐患。
- `langgraph-checkpoint-sqlite>=2.0.0` 已是生产依赖；`huginn/persistence/checkpointer.py` 已有 `SQLiteCheckpointerBackend` 抽象，可复用为共享状态底座。
- `_threads`/`_checkpoints` 被约 10 个模块直接以 dict 方式访问（`routes/threads.py`、`routes/checkpoints.py`、`routes/search.py`、`routes/ws_helpers.py`、`export_share.py`、`cli/slash_commands.py`、`server.py` `_DELEGATED_SC`、`ws.py`）。迁移须保持存取接口不变，故引入 `MutableMapping` 风格 store。

## 任务

### P0-1 移除 Rust 沙箱快路径

- [ ] T1: 移除 Rust 沙箱快路径，命令执行确定性走 Python
```
goal:       消除 Rust sandbox 静默崩溃（RDKit+sklearn GPR 空 stderr）这条生产正确性隐患
files:      huginn/tools/bash_tool.py, huginn/tools/code_tool.py, ROADMAP.md, 相关测试
acceptance: bash_tool/code_tool 不再动态 import huginn_ext.sandbox；设置与未设置
             HUGINN_USE_RUST_SANDBOX 时行为一致；RDKit+sklearn GPR 回归路径通过；
             grep 确认无运行期 huginn_ext 引用
spec:       polish-reports/industrialization-gap-analysis.md  #P0-1
```

### P0-2 共享状态后端化

- [ ] T2: 引入 `ThreadStore`（SQLite 持久化的 MutableMapping）替换内存 `_threads`
```
goal:       会话元数据可跨进程/重启持久，多 worker 一致，消除"请求漂移丢会话"
files:      huginn/persistence/thread_store.py, huginn/server_core.py, 相关测试
acceptance: ThreadStore 实现 MutableMapping 且基于 SQLite（WAL）；get_or_create_thread/
             touch_thread/列列举经 store；进程重启后既有会话仍在；threads 路由测试全绿
spec:       polish-reports/industrialization-gap-analysis.md  #P0-2
```

- [ ] T3: 引入 `CheckpointStore`（SQLite 持久化的 MutableMapping）替换内存 `_checkpoints`
```
goal:       检查点快照可跨进程/重启持久，多 worker 一致
files:      huginn/persistence/checkpoint_store.py, huginn/server_core.py,
            huginn/routes/checkpoints.py, huginn/routes/ws_helpers.py,
            huginn/cli/slash_commands.py, 相关测试
acceptance: CheckpointStore 实现 MutableMapping 且基于 SQLite（WAL）；cp 增删改查/回滚
            经 store；重启后检查点可恢复；checkpoints/undo 相关测试全绿
spec:       polish-reports/industrialization-gap-analysis.md  #P0-2
```

- [ ] T4: 状态后端开关 `HUGINN_STATE_BACKEND=memory|sqlite` + 多进程一致性验证
```
goal:       默认 memory 保持单进程行为不变；显式 sqlite 时多 worker 共享同一库一致
files:      huginn/config.py, huginn/server.py, DEPLOYMENT.md,
            tests/test_multiprocess_state.py
acceptance: 开关切换后线程/检查点行为等价；多 worker 并发写同一 sqlite 无
             "database is locked"（WAL + busy_timeout）；自动化测试覆盖两种后端
spec:       polish-reports/industrialization-gap-analysis.md  #P0-2
```

### 收尾

- [ ] T5: 持久化冒烟 + 文档更新
```
goal:       P0 两项落地后，单机多 worker 状态一致可复现，文档如实
files:      CHANGELOG.md, DEPLOYMENT.md, docs/tech-spec.md（若事实变化）
acceptance: 一处线程/检查点跨"重启 + 多 worker"的端到端冒烟通过；tech-spec
              contract/convention 与代码一致
spec:       polish-reports/industrialization-gap-analysis.md  #结论
```

## 并行性

- `[parallel] T2, T3`：两个 store 相互独立，共享契约（MutableMapping + SQLite WAL + accessor 接口）已闭合。
- T1 与 T2/T3 无依赖，可并行；T4 依赖 T2+T3；T5 依赖全部。

## 边界（不在本里程碑）

- `_context`（ServerContext 持有 agent/factory/memory 等活对象）为进程内单例，不纳入本里程碑；无状态化改造、会话亲和路由、Redis 后端、跨实例 HA 列为 P1。
- 不改动 `runtime/checkpoint.py`（磁盘 `.huginn_checkpoints` 的 LangGraph checkpointer 体系），本里程碑只迁移 `server_core._checkpoints` 内存快照。