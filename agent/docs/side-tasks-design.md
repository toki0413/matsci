# 侧边任务设计文档

> 面向开发者：说明侧边任务为什么这样实现、数据如何流动、哪些不变量不能破。
> 使用说明见 [side-tasks.md](side-tasks.md)；接口细节见 [side-tasks-api.md](side-tasks-api.md)；
> 前端组件见 [side-tasks-ui.md](side-tasks-ui.md)。

## 1. 目标与非目标

### 目标

1. **不打断**：主对话运行中，用户可以从任意一条消息 fork 出独立分支处理次要需求。
2. **能 fork**：分支继承 fork 点之前的 user/assistant 上下文，不要求用户重述背景。
3. **能转正**：侧边对话可以原地升级为正式任务，拥有任务卡片、历史与产物。
4. **用完即弃**：未转正的侧边任务按 TTL 清理，thread 注册与 checkpointer 行一起删除。

### 非目标

- 不做第二套对话/消息存储系统；
- 不做分布式任务队列或跨副本调度；
- 不替代 `Orchestrator` / `SubagentDispatch` / `Team` 等现有并行机制；
- 不自动把侧边对话内容合并回主对话（转正也不合并，二者是独立 thread）。

## 2. 总体架构

```
 Desktop (React)                         Backend (FastAPI)
┌─────────────────────────┐             ┌──────────────────────────────────┐
│ ChatPanel               │  GitFork    │ POST /side/tasks                 │
│  └─ onForkToSideTask ───┼────────────▶│  ├─ _load_parent_messages        │
│                         │             │  ├─ SideTaskStore.create (seed)  │
│ SidePanel (dock/page)   │             │  └─ get_or_create_thread(side_id)│
│  ├─ GET /side/tasks ────┼────────────▶│                                  │
│  ├─ POST .../messages ──┼────────────▶│  └─ _run_side_agent              │
│  ├─ POST .../promote ───┼────────────▶│       └─ agent.chat(thread_id)   │
│  └─ DELETE ...          │             │            ├─ seed 注入 (一次)     │
└─────────────────────────┘             │            └─ checkpointer 接管   │
                                        │                                  │
                                        │ POST .../promote                 │
                                        │  ├─ store.promote → task_card    │
                                        │  └─ thread meta kind=task        │
                                        │                                  │
                                        │ DELETE / TTL                     │
                                        │  ├─ store.discard                │
                                        │  ├─ _threads.pop                 │
                                        │  └─ purge_thread_checkpoints     │
                                        └──────────────────────────────────┘
```

关键选择：**侧边任务就是一个普通 thread**（`id = side_<8hex>`），
额外元数据存进 `SideTaskStore`。因此：

- 消息读写、checkpointer、WS/REST 全部复用现有链路；
- 转正时 thread_id 不变，历史与产物不需要迁移；
- 删除时按 thread_id 精确清理。

## 3. 数据模型

### 3.1 `SideTask` 字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | 任务 id，同时是 thread_id（`side_<8hex>`） |
| `title` | str | 标题；默认取种子里第一条 user 消息（截断 80） |
| `parent_thread_id` | str | fork 来源 thread |
| `fork_message_index` | int \| None | fork 点下标（user/assistant 过滤后，含该条）；None = 全量继承 |
| `seed_messages` | list[dict] | fork 时快照的继承消息 `{role, content}` |
| `seed_consumed` | bool | 种子是否已注入；保证只注入一次 |
| `messages` | list[dict] | 侧边对话轮次 `{role, content, at}` |
| `artifacts` | list[dict] | 产物 `{path, tool}`，按 path 去重 |
| `status` | str | `side` 或 `promoted` |
| `created_at` / `updated_at` | ISO str | 创建/最近活动时间 |
| `promoted_at` | ISO str \| None | 转正时间 |
| `task_card` | dict \| None | 转正后生成的任务卡片 |

### 3.2 持久化

- 路径：`<runtime_home>/side_tasks.json`，可用 `HUGINN_SIDE_TASKS_PATH` 覆盖；
- 原子写：`atomic_write_json`（tmp + rename），进程崩溃不会留下半截 JSON；
- 进程内 `RLock`，所有读写串行化；
- 单进程语义：多进程同时写同一文件会互相覆盖（与 `todo_tool` 同一已知边界）。

保存内容为 `SideTask.to_dict()` + 原始 `seed_messages` / `messages`
（列表接口只投影计数，不返回正文；详情接口才返回正文）。

### 3.3 任务卡片

转正时生成，字段：

```json
{
  "task_id": "side_ab12cd34",
  "title": "GaN k 点核查",
  "goal": "主任务: 计算 GaN 带隙",
  "status": "active",
  "source_thread_id": "main_thread_id",
  "fork_message_index": 3,
  "history_count": 6,
  "artifacts": [{"path": "out/kpoints.png", "tool": "file_write_tool"}],
  "created_at": "2026-10-21T08:00:00+00:00",
  "promoted_at": "2026-10-21T08:30:00+00:00"
}
```

任务卡片同时写入 thread 元数据（`task_card`），供任务列表/前端读取。

## 4. 状态机

```
                  create()
                     │
                     ▼
                ┌─────────┐   promote()   ┌───────────┐
                │  side   │──────────────▶│ promoted  │
                └─────────┘               └───────────┘
                     │                          │
          TTL 过期 / │                          │ 不参与 TTL
          discard()  │                          │ discard()（API 允许，
                     ▼                          ▼  UI 隐藏）
                  deleted                    deleted
```

- `side`：默认状态；可对话、可转正、可丢弃、受 TTL 约束。
- `promoted`：正式任务；可继续对话与追加产物；不再受 TTL 约束；
  重复 `promote()` 返回失败。
- `deleted`：任务表条目、thread 注册、SQLite checkpointer 行均被清理。

## 5. 关键流程

### 5.1 fork

1. 前端从消息操作栏拿到 fork 下标（user/assistant 过滤后），调用
   `POST /side/tasks`。
2. 后端校验父 thread 存在，读取父 thread checkpointer 消息。
3. `SideTaskStore.create()`：
   - 过滤出 user/assistant 且非空内容；
   - 按下标截断（含该条），越界抛 `ValueError`；
   - 生成 `side_<hex>`，写入任务表。
4. 立即 `get_or_create_thread(side_id)`，thread 元数据标记
   `kind=side`、`parent_thread_id`、`fork_message_index`。
5. 前端打开侧边面板，聚焦新任务。

### 5.2 首次对话与种子注入

```
agent.chat(content, thread_id=side_id)
  │
  ├─ SideTaskStore.peek_seed(side_id)          # 未消费才返回
  ├─ for m in seed: conversation_tree.add_message(m)
  ├─ SideTaskStore.consume_seed(side_id)
  ├─ add_message(user, content)
  └─ graph.invoke → checkpointer 写入 seed + 本轮消息
```

- 种子只在首次 `chat()` 注入；
- 后续轮次由 checkpointer 提供历史，`conversation_tree_to_messages()` 不会重复放大；
- tool / system 消息不进入种子，避免重放 dangling tool calls 导致 provider 400。

### 5.3 对话与产物

`POST /side/tasks/{id}/messages` 为侧边任务提供非流式调用：

1. `factory.create_lead(thread_id=side_id)` 创建独立 agent；
2. 遍历 `agent.chat()` 的 state，取最后一条非空 AI 文本；
3. 从最近 50 条消息的 `AIMessage.tool_calls` 抽取写类工具目标路径；
4. 只有**非空回答**才追加 user/assistant 轮次与产物；空回答返回失败且不落历史。

产物识别源是 `huginn/permissions.py::WRITE_EXEC_TOOLS`，参数键：
`path` / `file_path` / `output` / `output_path`，按路径去重。

### 5.4 转正

1. `store.promote()`：状态翻到 `promoted`，生成 `task_card`；
2. `get_or_create_thread(side_id, label=title)` 确保注册；
3. thread 元数据写回 `kind=task`、`promoted_from`、`task_card`；
4. 前端关闭 dock、切换到该正式 thread 并刷新列表。

原地转正意味着：历史消息、checkpointer、产物、thread_id 全部不变。

### 5.5 清理

两个入口：

- `DELETE /side/tasks/{id}`：显式丢弃；
- `GET /side/tasks`：顺带 `store.sweep()` 清理空闲超过 TTL 的 `side` 任务。

清理步骤：

1. 从任务表移除并落盘；
2. `_threads.pop(side_id)` 删除 thread 注册；
3. `purge_thread_checkpoints(side_id)` 删除 SQLite `checkpoints` / `writes` 行
   （先读 `config.checkpointer_path`，否则用 `HUGINN_CHECKPOINTER_PATH` /
   `<runtime_home>/checkpoints.sqlite`）。

checkpointer 不是 SQLite、库不存在或表不存在时，清理 best-effort，不阻断请求。

## 6. 不变量（红队检查点）

1. **不得伪装成功**：父 thread 不存在、下标越界、无模型、空回答一律 `success=false`。
2. **种子只注入一次**：`seed_consumed` 一旦置位，后续 `peek_seed` 返回空。
3. **只继承 user/assistant**：tool/system 不进入种子，避免 dangling tool call。
4. **转正不搬家**：`promoted.thread_id == side.thread_id`。
5. **转正不受 TTL**：`sweep()` 跳过 `promoted`。
6. **产物只认写类工具**：读类工具不产生产物，避免把"看过的文件"当交付物。
7. **空回答不落历史**：失败请求不得留下半截 user/assistant 轮次。
8. **删除幂等**：已删除任务再次删除返回 `success=false`，不抛异常。

这些不变量由 [test_side_tasks.py](../tests/test_side_tasks.py) 红蓝双轨覆盖。

## 7. 与现有机制的边界

| 机制 | 关系 |
|---|---|
| `ConversationTree` | 侧边 fork 种子在首次 chat 注入它；转正后仍由 checkpointer 主导历史 |
| checkpointer | 侧边 thread 与主 thread 共用同一 saver，但 thread_id 隔离 |
| `SideChannel`（旧 `/side`） | 并存的轻量问答队列，不参与侧边任务；桌面端 UI 已改用 SidePanel |
| `Orchestrator` / `SubagentDispatch` | 任务内并行；侧边任务是会话级并行，二者可叠加 |
| `TaskLifecycle`（autoloop） | 研究循环状态机，与侧边任务状态无关，不共用 |
| thread 注册表 | 侧边任务与正式任务都是其中的普通条目，用 `kind` 区分 |

## 8. 扩展点

- **新增任务元数据**：给 `SideTask` 加字段，同步 `to_dict` / `from_dict` / 详情投影，
  并补契约门。
- **更强的产物识别**：替换 `extract_artifacts()`，或在 `_run_side_agent` 中注入
  自定义解析器；不要改 `permissions.WRITE_EXEC_TOOLS` 的事实源。
- **不同 TTL 策略**：`HUGINN_SIDE_TASK_TTL_S` 或调用 `sweep(ttl_s=...)`；
  转正任务保留策略若变化，需先改不变量 5 和测试。
- **多进程/多副本**：当前 JSON 存储是单进程语义；升级时替换 `SideTaskStore`
  后端（SQLite/Redis），保持 `create/get/list_tasks/promote/sweep` 接口。
- **鉴权隔离**：thread 元数据带 `user_id`；如需侧边任务级鉴权，在路由层加 owner 检查，
  并在 `create` 时记录 `user_id`。

## 9. 治理登记

- 能力锚点：`governance_audit.CAPABILITY_SPECS["side_tasks"]`
  （模块：`huginn/side_tasks.py`、`huginn/routes/side.py`、`huginn/agent/streaming.py`；
  测试：`tests/test_side_tasks.py`）；
- CI：`.github/workflows/ci.yml` → `Side tasks contract guard`；
- 文档索引：`docs/INDEX.md`。
